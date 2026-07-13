#!/usr/bin/env python3
"""Evaluate exact answer-family coverage/correctness Pareto curves."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from eval_exact_answer_mode_coverage import (
        _answer_mode_count,
        _discover_final_checkpoint,
        _discover_run_dir,
        _mean_metric_dicts,
        _parse_checkpoint_specs,
        _parse_csv,
        _repo_root,
        _select_template,
        _step_number,
        compute_coverage_metrics,
    )
except ModuleNotFoundError:
    from ops.eval_exact_answer_mode_coverage import (
        _answer_mode_count,
        _discover_final_checkpoint,
        _discover_run_dir,
        _mean_metric_dicts,
        _parse_checkpoint_specs,
        _parse_csv,
        _repo_root,
        _select_template,
        _step_number,
        compute_coverage_metrics,
    )


METRIC_KEYS = [
    "sampled_pass_at_1",
    "mean_at_k",
    "any_correct_at_k",
    "distinct_correct_modes_at_k",
    "mode_coverage_at_k",
    "all_modes_covered_at_k",
    "correct_mode_entropy_total_norm",
    "correct_mode_entropy_observed_norm",
    "answer_key_extracted_frac",
    "correct_answer_key_extracted_frac",
    "answer_mode_count",
]


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(part) for part in _parse_csv(raw)]
    if not values:
        raise ValueError("Expected at least one integer value.")
    if any(value <= 0 for value in values):
        raise ValueError("All K values must be positive.")
    return sorted(set(values))


def _parse_float_csv(raw: str) -> list[float]:
    values = [float(part) for part in _parse_csv(raw)]
    if not values:
        raise ValueError("Expected at least one float value.")
    if any(value < 0 for value in values):
        raise ValueError("Temperatures must be non-negative.")
    return values


def _temp_label(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _parse_alias(alias: str) -> tuple[int | None, str]:
    match = re.search(r"_s(\d+)(?:_|$)", alias)
    seed = int(match.group(1)) if match else None
    variant = alias[: match.start()] if match else alias
    return seed, variant


def _checkpoint_label(path: Path) -> str:
    step = _step_number(path)
    return f"step_{step:05d}" if step >= 0 else path.name


def _discover_checkpoint_by_step(run_dir: Path, step: int) -> Path:
    candidates: list[Path] = []
    for root in [run_dir, *sorted(run_dir.glob("debug_*"))]:
        saved = root / "saved_models"
        if not saved.is_dir():
            continue
        candidates.extend(
            path
            for path in saved.glob("step_*")
            if path.is_dir()
            and (path / "config.json").is_file()
            and _step_number(path) == int(step)
        )
    if not candidates:
        raise FileNotFoundError(f"No step_{int(step):05d} checkpoint found under {run_dir}")
    return sorted(candidates, key=lambda path: str(path))[-1]


def _parse_checkpoint_steps(raw: str) -> list[str]:
    values = _parse_csv(raw)
    if not values:
        return ["final"]
    normalized: list[str] = []
    for value in values:
        lower = value.lower()
        if lower in {"final", "last"}:
            normalized.append("final")
            continue
        step = int(value)
        if step < 0:
            raise ValueError("Checkpoint steps must be non-negative or 'final'.")
        normalized.append(str(step))
    return normalized


def compute_prefix_coverage_metrics(
    *,
    rewards: Sequence[float],
    answer_keys: Sequence[str | None],
    answer_mode_count: int,
    sample_counts: Sequence[int],
) -> dict[int, dict[str, float]]:
    if len(rewards) != len(answer_keys):
        raise ValueError("rewards and answer_keys must have matching lengths.")
    if not rewards:
        raise ValueError("Need at least one sampled reward.")
    max_available = len(rewards)
    metrics_by_k: dict[int, dict[str, float]] = {}
    for sample_count in sorted(set(int(value) for value in sample_counts)):
        if sample_count <= 0:
            raise ValueError("K values must be positive.")
        if sample_count > max_available:
            raise ValueError(
                f"K={sample_count} exceeds available sample count {max_available}."
            )
        metrics_by_k[sample_count] = compute_coverage_metrics(
            rewards=rewards[:sample_count],
            answer_keys=answer_keys[:sample_count],
            answer_mode_count=answer_mode_count,
        )
    return metrics_by_k


def _numeric_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in row.items()
        if isinstance(value, (int, float)) and key in METRIC_KEYS
    }


def _evaluate_checkpoint_grid(
    *,
    alias: str,
    checkpoint_path: Path,
    datasets_by_split: Any,
    splits: Sequence[str],
    output_root: Path,
    template: str,
    sample_counts: Sequence[int],
    temperatures: Sequence[float],
    top_p: float,
    max_tokens: int,
    max_model_len: int,
    dtype: str,
    batch_size: int,
    seed: int,
    gpu_memory_utilization: float,
    swap_space: float,
    limit_per_split: int,
    include_text: bool,
) -> list[dict[str, Any]]:
    import torch
    import vllm
    from oat_drgrpo.math_grader import (
        boxed_reward_fn,
        extract_normalized_final_answer_for_clustering,
    )

    apply_template = _select_template(template)
    max_sample_count = max(int(value) for value in sample_counts)
    llm = vllm.LLM(
        model=str(checkpoint_path),
        dtype=str(dtype),
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
        swap_space=float(swap_space),
        enable_prefix_caching=True,
    )

    checkpoint_root = output_root / "checkpoints" / alias
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    seed_value = int(seed)
    for temperature in temperatures:
        params = vllm.SamplingParams(
            n=int(max_sample_count),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            seed=seed_value,
        )
        for split in splits:
            dataset = datasets_by_split[split]
            row_count = (
                min(len(dataset), int(limit_per_split))
                if limit_per_split
                else len(dataset)
            )
            rows = [dataset[index] for index in range(row_count)]
            prompts = [apply_template(row["problem"]) for row in rows]
            attempts: list[dict[str, Any]] = []
            prompt_metrics_by_k: dict[int, list[dict[str, Any]]] = {
                int(sample_count): [] for sample_count in sample_counts
            }
            for batch_start in range(0, len(prompts), max(int(batch_size), 1)):
                batch_end = min(batch_start + max(int(batch_size), 1), len(prompts))
                outputs = llm.generate(prompts[batch_start:batch_end], params)
                for offset, output in enumerate(outputs):
                    row_index = batch_start + offset
                    row = rows[row_index]
                    answer = row["answer"]
                    rewards: list[float] = []
                    answer_keys: list[str | None] = []
                    for sample_index, sample_output in enumerate(
                        output.outputs, start=1
                    ):
                        text = sample_output.text
                        _info, reward = boxed_reward_fn(text, answer, fast=False)
                        answer_key = extract_normalized_final_answer_for_clustering(
                            text,
                            template=template,
                            gt_answer=answer,
                        )
                        rewards.append(float(reward))
                        answer_keys.append(answer_key)
                        attempt = {
                            "alias": alias,
                            "split": split,
                            "temperature": float(temperature),
                            "dataset_index": row_index,
                            "sample_index": sample_index,
                            "reward": float(reward),
                            "correct": bool(float(reward) > 0.0),
                            "answer_key": answer_key,
                            "token_length": len(sample_output.token_ids),
                        }
                        if include_text:
                            attempt["text"] = text
                        attempts.append(attempt)

                    metrics_by_k = compute_prefix_coverage_metrics(
                        rewards=rewards,
                        answer_keys=answer_keys,
                        answer_mode_count=_answer_mode_count(answer, row),
                        sample_counts=sample_counts,
                    )
                    for sample_count, metrics in metrics_by_k.items():
                        prompt_metrics_by_k[int(sample_count)].append(
                            {
                                **metrics,
                                "dataset_index": float(row_index),
                                "split": split,
                                "temperature": float(temperature),
                                "sample_count": float(sample_count),
                            }
                        )

            temp_root = checkpoint_root / f"temp_{_temp_label(float(temperature))}" / split
            temp_root.mkdir(parents=True, exist_ok=True)
            attempts_path = temp_root / "attempts.json"
            attempts_path.write_text(
                json.dumps(attempts, indent=2, sort_keys=True) + "\n"
            )
            for sample_count in sample_counts:
                prompt_metrics = prompt_metrics_by_k[int(sample_count)]
                prompt_metrics_path = temp_root / f"prompt_metrics_k{sample_count}.json"
                prompt_metrics_path.write_text(
                    json.dumps(prompt_metrics, indent=2, sort_keys=True) + "\n"
                )
                summary_metrics = _mean_metric_dicts(
                    [_numeric_metrics(row) for row in prompt_metrics]
                )
                seed_id, variant = _parse_alias(alias)
                summary_row = {
                    "alias": alias,
                    "seed": seed_id,
                    "variant": variant,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_label": _checkpoint_label(checkpoint_path),
                    "checkpoint_step": _step_number(checkpoint_path),
                    "split": split,
                    "temperature": float(temperature),
                    "sample_count": int(sample_count),
                    "row_count": int(row_count),
                    "attempts_path": str(attempts_path),
                    "prompt_metrics_path": str(prompt_metrics_path),
                    "metrics": summary_metrics,
                }
                summary_rows.append(summary_row)

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary_rows


def _aggregate_grid(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int, str, float, int], list[Mapping[str, float]]
    ] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        key = (
            str(row["variant"]),
            str(row.get("checkpoint_label", "")),
            int(row.get("checkpoint_step", -1)),
            str(row["split"]),
            float(row["temperature"]),
            int(row["sample_count"]),
        )
        grouped[key].append({k: float(v) for k, v in metrics.items()})
    aggregate_rows: list[dict[str, Any]] = []
    for (
        variant,
        checkpoint_label,
        checkpoint_step,
        split,
        temperature,
        sample_count,
    ), metric_rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][3],
            item[0][0],
            item[0][2],
            item[0][4],
            item[0][5],
        ),
    ):
        aggregate_rows.append(
            {
                "variant": variant,
                "checkpoint_label": checkpoint_label,
                "checkpoint_step": checkpoint_step,
                "split": split,
                "temperature": temperature,
                "sample_count": sample_count,
                "seed_count": len(metric_rows),
                "metrics": _mean_metric_dicts(metric_rows),
            }
        )
    return aggregate_rows


def _pairwise_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str, str, int, str, float, int], Mapping[str, Any]] = {}
    for row in rows:
        seed = row.get("seed")
        if seed is None:
            continue
        key = (
            int(seed),
            str(row["variant"]),
            str(row.get("checkpoint_label", "")),
            int(row.get("checkpoint_step", -1)),
            str(row["split"]),
            float(row["temperature"]),
            int(row["sample_count"]),
        )
        by_key[key] = row
    deltas: list[dict[str, Any]] = []
    seeds = sorted({key[0] for key in by_key})
    checkpoints = sorted({(key[2], key[3]) for key in by_key}, key=lambda item: item[1])
    splits = sorted({key[4] for key in by_key})
    temperatures = sorted({key[5] for key in by_key})
    sample_counts = sorted({key[6] for key in by_key})
    comparisons = [("grpo", "answer_maxent"), ("dpo", "maxent_dpo")]
    for baseline_variant, maxent_variant in comparisons:
        for seed in seeds:
            for checkpoint_label, checkpoint_step in checkpoints:
                for split in splits:
                    for temperature in temperatures:
                        for sample_count in sample_counts:
                            baseline = by_key.get(
                                (
                                    seed,
                                    baseline_variant,
                                    checkpoint_label,
                                    checkpoint_step,
                                    split,
                                    temperature,
                                    sample_count,
                                )
                            )
                            maxent = by_key.get(
                                (
                                    seed,
                                    maxent_variant,
                                    checkpoint_label,
                                    checkpoint_step,
                                    split,
                                    temperature,
                                    sample_count,
                                )
                            )
                            if baseline is None or maxent is None:
                                continue
                            base_metrics = baseline["metrics"]
                            maxent_metrics = maxent["metrics"]
                            delta = {
                                "seed": seed,
                                "baseline_variant": baseline_variant,
                                "maxent_variant": maxent_variant,
                                "checkpoint_label": checkpoint_label,
                                "checkpoint_step": checkpoint_step,
                                "split": split,
                                "temperature": temperature,
                                "sample_count": sample_count,
                            }
                            for metric in METRIC_KEYS:
                                delta[f"{metric}_delta"] = float(
                                    maxent_metrics.get(metric, 0.0)
                                ) - float(base_metrics.get(metric, 0.0))
                            deltas.append(delta)
    return deltas


def _matched_correctness_rows(
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    x_metric: str = "mean_at_k",
    y_metric: str = "mode_coverage_at_k",
) -> list[dict[str, Any]]:
    by_split: dict[tuple[str, str, int], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in aggregate_rows:
        key = (
            str(row["split"]),
            str(row.get("checkpoint_label", "")),
            int(row.get("checkpoint_step", -1)),
        )
        by_split[key][str(row["variant"])].append(row)

    matched_rows: list[dict[str, Any]] = []
    for (split, checkpoint_label, checkpoint_step), by_variant in sorted(
        by_split.items(), key=lambda item: (item[0][0], item[0][2])
    ):
        baseline_rows = by_variant.get("grpo", [])
        candidate_rows = by_variant.get("answer_maxent", [])
        for candidate in candidate_rows:
            candidate_metrics = candidate["metrics"]
            candidate_x = float(candidate_metrics.get(x_metric, 0.0))
            eligible = [
                row
                for row in baseline_rows
                if float(row["metrics"].get(x_metric, 0.0)) >= candidate_x - 1e-12
            ]
            if not eligible:
                eligible = list(baseline_rows)
            if not eligible:
                continue
            baseline_best = max(
                eligible, key=lambda row: float(row["metrics"].get(y_metric, 0.0))
            )
            baseline_metrics = baseline_best["metrics"]
            matched_rows.append(
                {
                    "split": split,
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_step": checkpoint_step,
                    "candidate_temperature": float(candidate["temperature"]),
                    "candidate_sample_count": int(candidate["sample_count"]),
                    "candidate_x": candidate_x,
                    "candidate_y": float(candidate_metrics.get(y_metric, 0.0)),
                    "baseline_temperature": float(baseline_best["temperature"]),
                    "baseline_sample_count": int(baseline_best["sample_count"]),
                    "baseline_x": float(baseline_metrics.get(x_metric, 0.0)),
                    "baseline_y": float(baseline_metrics.get(y_metric, 0.0)),
                    "coverage_advantage_at_matched_or_better_correctness": float(
                        candidate_metrics.get(y_metric, 0.0)
                    )
                    - float(baseline_metrics.get(y_metric, 0.0)),
                }
            )
    return matched_rows


def _flatten_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        item = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "metrics",
                "attempts_path",
                "prompt_metrics_path",
                "checkpoint_path",
            }
        }
        metrics = row.get("metrics", {})
        if isinstance(metrics, Mapping):
            item.update({key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS})
        flat.append(item)
    return flat


def _flatten_checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        item = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "metrics",
                "checkpoint_path",
            }
        }
        item["checkpoint_path"] = str(row.get("checkpoint_path", ""))
        metrics = row.get("metrics", {})
        if isinstance(metrics, Mapping):
            item.update({key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS})
        flat.append(item)
    return flat


def _budget_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str, str, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        seed = row.get("seed")
        if seed is None:
            continue
        key = (
            int(seed),
            str(row["variant"]),
            str(row["split"]),
            float(row["temperature"]),
            int(row["sample_count"]),
        )
        by_key[key].append(row)

    out: list[dict[str, Any]] = []
    for (seed, variant, split, temperature, sample_count), group_rows in sorted(
        by_key.items(), key=lambda item: item[0]
    ):
        sorted_rows = sorted(
            group_rows, key=lambda row: int(row.get("checkpoint_step", -1))
        )
        base_row = next(
            (row for row in sorted_rows if int(row.get("checkpoint_step", -1)) == 0),
            None,
        )
        if base_row is None and sorted_rows:
            base_row = sorted_rows[0]
        base_metrics = base_row.get("metrics", {}) if base_row is not None else {}
        base_cov = float(base_metrics.get("mode_coverage_at_k", 0.0))
        base_mean = float(base_metrics.get("mean_at_k", 0.0))
        base_any = float(base_metrics.get("any_correct_at_k", 0.0))
        base_distinct = float(base_metrics.get("distinct_correct_modes_at_k", 0.0))
        base_extract = float(base_metrics.get("answer_key_extracted_frac", 0.0))
        for row in sorted_rows:
            metrics = row.get("metrics", {})
            if not isinstance(metrics, Mapping):
                continue
            checkpoint_step = int(row.get("checkpoint_step", -1))
            mode_cov = float(metrics.get("mode_coverage_at_k", 0.0))
            out.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "split": split,
                    "temperature": temperature,
                    "sample_count": sample_count,
                    "checkpoint_label": str(row.get("checkpoint_label", "")),
                    "checkpoint_step": checkpoint_step,
                    "prompt_epoch_approx": (
                        float(checkpoint_step) / 192.0 if checkpoint_step >= 0 else ""
                    ),
                    "any_correct_at_k": float(metrics.get("any_correct_at_k", 0.0)),
                    "mean_at_k": float(metrics.get("mean_at_k", 0.0)),
                    "mode_coverage_at_k": mode_cov,
                    "distinct_correct_modes_at_k": float(
                        metrics.get("distinct_correct_modes_at_k", 0.0)
                    ),
                    "answer_key_extracted_frac": float(
                        metrics.get("answer_key_extracted_frac", 0.0)
                    ),
                    "any_correct_delta_from_step0": float(
                        metrics.get("any_correct_at_k", 0.0)
                    )
                    - base_any,
                    "mean_delta_from_step0": float(metrics.get("mean_at_k", 0.0))
                    - base_mean,
                    "mode_coverage_delta_from_step0": mode_cov - base_cov,
                    "distinct_correct_delta_from_step0": float(
                        metrics.get("distinct_correct_modes_at_k", 0.0)
                    )
                    - base_distinct,
                    "extract_delta_from_step0": float(
                        metrics.get("answer_key_extracted_frac", 0.0)
                    )
                    - base_extract,
                    "mode_coverage_collapse_ratio": (
                        mode_cov / base_cov if abs(base_cov) > 1e-12 else ""
                    ),
                }
            )
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.4f}"


def _write_plot(
    *,
    path: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    split: str,
    x_metric: str,
    y_metric: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    rows = [row for row in aggregate_rows if str(row["split"]) == split]
    if not rows:
        return False
    colors = {
        "grpo": "#2563eb",
        "grpo_entropy": "#059669",
        "answer_maxent": "#dc2626",
        "dpo": "#7c3aed",
        "maxent_dpo": "#ea580c",
    }
    labels = {
        "grpo": "GRPO",
        "grpo_entropy": "GRPO+Entropy",
        "answer_maxent": "Dr.X-GRPO",
        "dpo": "DPO",
        "maxent_dpo": "MaxEnt-DPO",
    }
    markers = {1: "o", 4: "s", 8: "^", 16: "D", 32: "P"}

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for variant in sorted({str(row["variant"]) for row in rows}):
        variant_rows = [row for row in rows if str(row["variant"]) == variant]
        for sample_count in sorted({int(row["sample_count"]) for row in variant_rows}):
            sample_rows = sorted(
                [
                    row
                    for row in variant_rows
                    if int(row["sample_count"]) == sample_count
                ],
                key=lambda row: float(row["temperature"]),
            )
            xs = [float(row["metrics"].get(x_metric, 0.0)) for row in sample_rows]
            ys = [float(row["metrics"].get(y_metric, 0.0)) for row in sample_rows]
            ax.plot(
                xs,
                ys,
                color=colors.get(variant, "#111827"),
                alpha=0.30,
                linewidth=1.0,
            )
            ax.scatter(
                xs,
                ys,
                color=colors.get(variant, "#111827"),
                marker=markers.get(sample_count, "o"),
                s=48,
                label=f"{labels.get(variant, variant)} K={sample_count}",
                alpha=0.88,
            )
            for row, x_value, y_value in zip(sample_rows, xs, ys):
                ax.annotate(
                    f"T={float(row['temperature']):g}",
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color="#374151",
                )
    ax.set_title(f"{split}: coverage vs correctness")
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.grid(True, alpha=0.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, legend_labels):
        unique.setdefault(label, handle)
    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=7,
        ncols=2,
        loc="best",
        frameon=False,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _write_trajectory_plot(
    *,
    path: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    split: str,
    temperature: float,
    sample_count: int,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    rows = [
        row
        for row in aggregate_rows
        if str(row["split"]) == split
        and int(row["sample_count"]) == int(sample_count)
        and abs(float(row["temperature"]) - float(temperature)) < 1e-9
        and int(row.get("checkpoint_step", -1)) >= 0
    ]
    if len({int(row.get("checkpoint_step", -1)) for row in rows}) < 2:
        return False

    colors = {
        "grpo": "#2563eb",
        "grpo_entropy": "#059669",
        "answer_maxent": "#dc2626",
        "dpo": "#7c3aed",
        "maxent_dpo": "#ea580c",
    }
    labels = {
        "grpo": "GRPO",
        "grpo_entropy": "GRPO+Entropy",
        "answer_maxent": "Dr.X-GRPO",
        "dpo": "DPO",
        "maxent_dpo": "MaxEnt-DPO",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    for variant in sorted({str(row["variant"]) for row in rows}):
        variant_rows = sorted(
            [row for row in rows if str(row["variant"]) == variant],
            key=lambda row: int(row.get("checkpoint_step", -1)),
        )
        xs = [int(row.get("checkpoint_step", -1)) for row in variant_rows]
        any_correct = [
            float(row["metrics"].get("any_correct_at_k", 0.0))
            for row in variant_rows
        ]
        mode_cov = [
            float(row["metrics"].get("mode_coverage_at_k", 0.0))
            for row in variant_rows
        ]
        label = labels.get(variant, variant)
        color = colors.get(variant, "#111827")
        axes[0].plot(xs, any_correct, marker="o", label=label, color=color)
        axes[1].plot(xs, mode_cov, marker="o", label=label, color=color)

    axes[0].set_title(f"{split}: AnyCorrect@{sample_count}")
    axes[1].set_title(f"{split}: ModeCoverage@{sample_count}")
    for ax in axes:
        ax.set_xlabel("checkpoint step")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("AnyCorrect")
    axes[1].set_ylabel("ModeCoverage")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Sampled checkpoint trajectory, T={float(temperature):g}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _write_budget_trajectory_plot(
    *,
    path: Path,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    split: str,
    temperature: float,
    sample_count: int,
    batches_per_epoch: int,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    rows = [
        row
        for row in checkpoint_rows
        if str(row["split"]) == split
        and int(row["sample_count"]) == int(sample_count)
        and abs(float(row["temperature"]) - float(temperature)) < 1e-9
        and int(row.get("checkpoint_step", -1)) >= 0
        and str(row.get("variant")) in {"grpo", "answer_maxent"}
    ]
    if len({int(row.get("checkpoint_step", -1)) for row in rows}) < 2:
        return False

    colors = {"grpo": "#2563eb", "answer_maxent": "#dc2626"}
    labels = {"grpo": "GRPO", "answer_maxent": "Answer-MaxEnt"}
    metrics = [
        ("mean_at_k", f"MeanCorrect@{sample_count}"),
        ("mode_coverage_at_k", f"ModeCoverage@{sample_count}"),
        ("distinct_correct_modes_at_k", f"DistinctCorrect@{sample_count}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    for variant in ["grpo", "answer_maxent"]:
        variant_rows = [row for row in rows if str(row.get("variant")) == variant]
        if not variant_rows:
            continue
        seeds = sorted({int(row["seed"]) for row in variant_rows if row.get("seed") is not None})
        for seed in seeds:
            seed_rows = sorted(
                [row for row in variant_rows if int(row.get("seed", -1)) == seed],
                key=lambda row: int(row.get("checkpoint_step", -1)),
            )
            xs = [
                int(row.get("checkpoint_step", -1)) / float(batches_per_epoch)
                for row in seed_rows
            ]
            for ax, (metric_key, _) in zip(axes, metrics):
                ax.plot(
                    xs,
                    [float(row["metrics"].get(metric_key, 0.0)) for row in seed_rows],
                    color=colors[variant],
                    linewidth=0.9,
                    alpha=0.22,
                )
        steps = sorted({int(row.get("checkpoint_step", -1)) for row in variant_rows})
        mean_xs: list[float] = []
        mean_values: dict[str, list[float]] = {metric_key: [] for metric_key, _ in metrics}
        for step in steps:
            step_rows = [
                row
                for row in variant_rows
                if int(row.get("checkpoint_step", -1)) == step
            ]
            if not step_rows:
                continue
            mean_xs.append(step / float(batches_per_epoch))
            for metric_key, _ in metrics:
                mean_values[metric_key].append(
                    sum(float(row["metrics"].get(metric_key, 0.0)) for row in step_rows)
                    / float(len(step_rows))
                )
        for ax, (metric_key, _) in zip(axes, metrics):
            ax.plot(
                mean_xs,
                mean_values[metric_key],
                color=colors[variant],
                marker="o",
                linewidth=2.4,
                label=labels[variant],
            )

    for ax, (_, title) in zip(axes, metrics):
        ax.set_title(title)
        ax.set_xlabel("Prompt epochs")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Graph-coloring budget curve, K={sample_count}, T={float(temperature):g}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_markdown(
    *,
    path: Path,
    stamp_prefix: str,
    aggregate_rows: Sequence[Mapping[str, Any]],
    matched_rows: Sequence[Mapping[str, Any]],
    plot_paths: Sequence[Path],
    budget_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    lines = [
        "# Exact Answer-Mode Pareto Sweep",
        "",
        f"Stamp prefix: `{stamp_prefix}`.",
        "",
        "## Aggregate Grid",
        "",
    ]
    has_checkpoints = any(str(row.get("checkpoint_label", "")) for row in aggregate_rows)
    if has_checkpoints:
        lines.extend(
            [
                "| Checkpoint | Split | Variant | Temp | K | Mean@K | Any@K | ModeCov@K | Distinct | Acc@1 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Split | Variant | Temp | K | Mean@K | Any@K | ModeCov@K | Distinct | Acc@1 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for row in aggregate_rows:
        metrics = row["metrics"]
        if has_checkpoints:
            lines.append(
                "| {ckpt} | {split} | `{variant}` | {temp:.2f} | {k} | {mean} | {any} | {cov} | {distinct} | {acc1} |".format(
                    ckpt=row.get("checkpoint_label", ""),
                    split=row["split"],
                    variant=row["variant"],
                    temp=float(row["temperature"]),
                    k=int(row["sample_count"]),
                    mean=_fmt(metrics.get("mean_at_k", 0.0)),
                    any=_fmt(metrics.get("any_correct_at_k", 0.0)),
                    cov=_fmt(metrics.get("mode_coverage_at_k", 0.0)),
                    distinct=_fmt(metrics.get("distinct_correct_modes_at_k", 0.0)),
                    acc1=_fmt(metrics.get("sampled_pass_at_1", 0.0)),
                )
            )
        else:
            lines.append(
                "| {split} | `{variant}` | {temp:.2f} | {k} | {mean} | {any} | {cov} | {distinct} | {acc1} |".format(
                    split=row["split"],
                    variant=row["variant"],
                    temp=float(row["temperature"]),
                    k=int(row["sample_count"]),
                    mean=_fmt(metrics.get("mean_at_k", 0.0)),
                    any=_fmt(metrics.get("any_correct_at_k", 0.0)),
                    cov=_fmt(metrics.get("mode_coverage_at_k", 0.0)),
                    distinct=_fmt(metrics.get("distinct_correct_modes_at_k", 0.0)),
                    acc1=_fmt(metrics.get("sampled_pass_at_1", 0.0)),
                )
            )

    if matched_rows:
        lines.extend(
            [
                "",
                "## Matched-Correctness Check",
                "",
                "For each MaxEnt aggregate point, this compares against the best GRPO coverage point with equal-or-higher `Mean@K`; if none exists, it uses the best available GRPO point.",
                "",
                "| Split | MaxEnt T | MaxEnt K | MaxEnt Mean | MaxEnt Cov | GRPO T | GRPO K | GRPO Mean | GRPO Cov | Cov Adv |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in matched_rows:
            lines.append(
                "| {split} | {ct:.2f} | {ck} | {cx} | {cy} | {bt:.2f} | {bk} | {bx} | {by} | {adv} |".format(
                    split=row["split"],
                    ct=float(row["candidate_temperature"]),
                    ck=int(row["candidate_sample_count"]),
                    cx=_fmt(row["candidate_x"]),
                    cy=_fmt(row["candidate_y"]),
                    bt=float(row["baseline_temperature"]),
                    bk=int(row["baseline_sample_count"]),
                    bx=_fmt(row["baseline_x"]),
                    by=_fmt(row["baseline_y"]),
                    adv=_fmt(
                        row["coverage_advantage_at_matched_or_better_correctness"]
                    ),
                )
            )

    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for plot_path in plot_paths:
            lines.append(f"- `{plot_path}`")
    if budget_rows:
        lines.extend(
            [
                "",
                "## Budget-Curve Collapse Ratios",
                "",
                "| Seed | Variant | Checkpoint | Epoch | Mean@K | Any@K | ModeCov@K | Distinct | Extract | Cov/Step0 |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in budget_rows:
            lines.append(
                "| {seed} | `{variant}` | {ckpt} | {epoch} | {mean} | {any} | {cov} | {distinct} | {extract} | {ratio} |".format(
                    seed=row["seed"],
                    variant=row["variant"],
                    ckpt=row["checkpoint_label"],
                    epoch=_fmt(row["prompt_epoch_approx"]),
                    mean=_fmt(row["mean_at_k"]),
                    any=_fmt(row["any_correct_at_k"]),
                    cov=_fmt(row["mode_coverage_at_k"]),
                    distinct=_fmt(row["distinct_correct_modes_at_k"]),
                    extract=_fmt(row["answer_key_extracted_frac"]),
                    ratio=_fmt(row["mode_coverage_collapse_ratio"]),
                )
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-prefix", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("var/artifacts"))
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument(
        "--checkpoint-steps",
        default="final",
        help="Comma-separated checkpoint steps to auto-discover, e.g. 0,64,128,final.",
    )
    parser.add_argument(
        "--strict-checkpoint-steps",
        action="store_true",
        help="Fail if any requested auto-discovered checkpoint step is missing.",
    )
    parser.add_argument("--seeds", default="43,44,45")
    parser.add_argument("--variants", default="grpo,answer_maxent")
    parser.add_argument("--model-tag", default="qwen25_0p5b_instruct")
    parser.add_argument("--splits", default="multi_answer,unique_answer")
    parser.add_argument("--template", default="qwen_boxed")
    parser.add_argument("--sample-counts", default="1,4,8,16,32")
    parser.add_argument("--temperatures", default="0.6,0.8,1.0,1.2")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--swap-space", type=float, default=16.0)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--plot-splits", default="multi_answer,unique_answer")
    parser.add_argument("--trajectory-temperature", type=float, default=1.0)
    parser.add_argument("--trajectory-sample-count", type=int, default=32)
    parser.add_argument(
        "--budget-batches-per-epoch",
        type=int,
        default=192,
        help="Prompt batches per epoch for budget-curve x-axis and collapse summaries.",
    )
    args = parser.parse_args()

    import sys

    repo_src = _repo_root() / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from datasets import load_from_disk

    sample_counts = _parse_int_csv(args.sample_counts)
    temperatures = _parse_float_csv(args.temperatures)
    eval_root = args.data_root / "eval"
    datasets_by_split = load_from_disk(str(eval_root))
    splits = (
        list(datasets_by_split.keys()) if args.splits == "all" else _parse_csv(args.splits)
    )
    checkpoints = _parse_checkpoint_specs(args.checkpoint)
    if not checkpoints:
        checkpoint_steps = _parse_checkpoint_steps(args.checkpoint_steps)
        for seed_text in _parse_csv(args.seeds):
            seed = int(seed_text)
            for variant in _parse_csv(args.variants):
                run_dir = _discover_run_dir(
                    stamp_prefix=args.stamp_prefix,
                    seed=seed,
                    variant=variant,
                    model_tag=args.model_tag,
                    run_data_root=args.run_data_root,
                )
                for checkpoint_step in checkpoint_steps:
                    try:
                        if checkpoint_step == "final":
                            checkpoint_path = _discover_final_checkpoint(run_dir)
                        else:
                            checkpoint_path = _discover_checkpoint_by_step(
                                run_dir, int(checkpoint_step)
                            )
                    except FileNotFoundError as exc:
                        if bool(args.strict_checkpoint_steps):
                            raise
                        print(
                            f"[pareto-eval] skip missing checkpoint: {exc}",
                            flush=True,
                        )
                        continue
                    checkpoints.append(
                        (
                            f"{variant}_s{seed}_{_checkpoint_label(checkpoint_path)}",
                            checkpoint_path,
                        )
                    )

    run_output_root = args.output_root / args.stamp_prefix
    run_output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for alias, checkpoint_path in checkpoints:
        print(
            f"[pareto-eval] start alias={alias} checkpoint={checkpoint_path}",
            flush=True,
        )
        summary_rows.extend(
            _evaluate_checkpoint_grid(
                alias=alias,
                checkpoint_path=checkpoint_path,
                datasets_by_split=datasets_by_split,
                splits=splits,
                output_root=run_output_root,
                template=args.template,
                sample_counts=sample_counts,
                temperatures=temperatures,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_model_len=args.max_model_len,
                dtype=args.dtype,
                batch_size=args.batch_size,
                seed=args.seed,
                gpu_memory_utilization=args.gpu_memory_utilization,
                swap_space=args.swap_space,
                limit_per_split=args.limit_per_split,
                include_text=args.include_text,
            )
        )
        print(f"[pareto-eval] done alias={alias}", flush=True)

    aggregate_rows = _aggregate_grid(summary_rows)
    delta_rows = _pairwise_deltas(summary_rows)
    matched_rows = _matched_correctness_rows(aggregate_rows)
    plot_paths: list[Path] = []
    for split in _parse_csv(args.plot_splits):
        plot_path = args.output_root / f"{args.stamp_prefix}_{split}_pareto.png"
        if _write_plot(
            path=plot_path,
            aggregate_rows=aggregate_rows,
            split=split,
            x_metric="mean_at_k",
            y_metric="mode_coverage_at_k",
        ):
            plot_paths.append(plot_path)
        trajectory_path = (
            args.output_root / f"{args.stamp_prefix}_{split}_checkpoint_trajectory.png"
        )
        if _write_trajectory_plot(
            path=trajectory_path,
            aggregate_rows=aggregate_rows,
            split=split,
            temperature=float(args.trajectory_temperature),
            sample_count=int(args.trajectory_sample_count),
        ):
            plot_paths.append(trajectory_path)
        budget_path = (
            args.output_root / f"{args.stamp_prefix}_{split}_budget_curve.png"
        )
        if _write_budget_trajectory_plot(
            path=budget_path,
            checkpoint_rows=summary_rows,
            split=split,
            temperature=float(args.trajectory_temperature),
            sample_count=int(args.trajectory_sample_count),
            batches_per_epoch=int(args.budget_batches_per_epoch),
        ):
            plot_paths.append(budget_path)

    budget_rows = _budget_summary_rows(summary_rows)
    payload = {
        "stamp_prefix": args.stamp_prefix,
        "data_root": str(args.data_root.resolve()),
        "sample_counts": sample_counts,
        "temperatures": temperatures,
        "top_p": float(args.top_p),
        "splits": splits,
        "checkpoint_rows": summary_rows,
        "aggregate_rows": aggregate_rows,
        "pairwise_deltas": delta_rows,
        "matched_correctness": matched_rows,
        "budget_curve": budget_rows,
        "plot_paths": [str(path) for path in plot_paths],
    }
    json_path = args.output_root / f"{args.stamp_prefix}_pareto_summary.json"
    checkpoint_csv_path = (
        args.output_root / f"{args.stamp_prefix}_pareto_checkpoint_rows.csv"
    )
    csv_path = args.output_root / f"{args.stamp_prefix}_pareto_aggregate.csv"
    deltas_csv_path = args.output_root / f"{args.stamp_prefix}_pareto_pairwise_deltas.csv"
    matched_csv_path = args.output_root / f"{args.stamp_prefix}_pareto_matched.csv"
    budget_csv_path = args.output_root / f"{args.stamp_prefix}_budget_curve.csv"
    md_path = args.output_root / f"{args.stamp_prefix}_pareto_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_csv(checkpoint_csv_path, _flatten_checkpoint_rows(summary_rows))
    _write_csv(csv_path, _flatten_summary_rows(aggregate_rows))
    _write_csv(deltas_csv_path, delta_rows)
    _write_csv(matched_csv_path, matched_rows)
    _write_csv(budget_csv_path, budget_rows)
    _write_markdown(
        path=md_path,
        stamp_prefix=args.stamp_prefix,
        aggregate_rows=aggregate_rows,
        matched_rows=matched_rows,
        plot_paths=plot_paths,
        budget_rows=budget_rows,
    )
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {checkpoint_csv_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {deltas_csv_path}", flush=True)
    print(f"wrote {matched_csv_path}", flush=True)
    print(f"wrote {budget_csv_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    for plot_path in plot_paths:
        print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
