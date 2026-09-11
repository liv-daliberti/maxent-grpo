#!/usr/bin/env python3
"""Render E64's full held-out MATH-500 live monitor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e64_math500_realism_matched_identity.json"
MANIFEST = (
    ROOT / "var/artifacts/mte64_math500_realism_05b_12ep_comparative_jobs.tsv"
)
OUT = ROOT / "paper/figures/e64_math500_realism_05b_12ep_live"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARM_LABELS = {
    CONTROL: "matched Dr.GRPO",
    TREATMENT: "E58 verified replay",
}
COLORS = {
    CONTROL: "#202124",
    TREATMENT: "#6d28d9",
}
PASSES_PER_STEP = 1.0 / 384.0
GREEDY_EVAL_LOG_RE = re.compile(
    r"Finished eval benchmark \d+/\d+: math "
    r"accuracy=(?P<accuracy>[0-9.]+) "
    r"score=[^ ]+ avg_len=(?P<avg_len>[0-9.]+) "
    r"at step (?P<step>\d+)"
)


def _finite(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _run_dir(run_stamp: str, job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for raw in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _coverage_sidecar(
    path: Path,
) -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    """Recover landed eval metrics before train_metrics.jsonl is emitted."""

    series_values: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    parseability_values: dict[int, list[float]] = defaultdict(list)
    if not path.is_file():
        return {}, {}
    for raw in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        step = payload.get("step")
        if _finite(step) is None:
            continue
        step = int(step)
        kind = str(payload.get("evaluation_kind", ""))
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        if kind == "deterministic_greedy_trace_neutral":
            value = _finite(metrics.get("mean_at_k"))
            if value is not None:
                series_values["eval/math/accuracy"][step].append(value)
            continue
        if kind != "fixed_seed_sampled_k_neutral":
            continue
        if int(payload.get("sample_count", 0) or 0) != 8:
            continue
        sampled_metric_map = {
            "eval/math/sampled_mean_at_8": "mean_at_k",
            "eval/math/sampled_any_correct_at_8": "any_correct_at_k",
        }
        for output_metric, sidecar_metric in sampled_metric_map.items():
            value = _finite(metrics.get(sidecar_metric))
            if value is not None:
                series_values[output_metric][step].append(value)
        extracted = 0
        total = 0
        for prompt in payload.get("prompts", []):
            keys = prompt.get("answer_keys", [])
            if not isinstance(keys, list):
                continue
            extracted += sum(key is not None for key in keys)
            total += len(keys)
        if total:
            parseability_values[step].append(extracted / total)
    series = {
        metric: {
            step: float(np.mean(rows))
            for step, rows in by_step.items()
            if rows
        }
        for metric, by_step in series_values.items()
    }
    parseability = {
        step: float(np.mean(rows))
        for step, rows in parseability_values.items()
        if rows
    }
    return series, parseability


def _greedy_log_sidecar(job_id: int) -> dict[str, dict[int, float]]:
    """Recover exact greedy accuracy/length from completed eval log rows."""

    path = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.out"
    values: dict[str, dict[int, float]] = defaultdict(dict)
    if not path.is_file():
        return {}
    for match in GREEDY_EVAL_LOG_RE.finditer(
        path.read_text(encoding="utf-8", errors="replace")
    ):
        step = int(match.group("step"))
        values["eval/math/accuracy"][step] = float(match.group("accuracy"))
        values["eval/math/response_tok_len"][step] = float(
            match.group("avg_len")
        )
    return dict(values)


def _sidecar_records(
    coverage: dict[str, dict[int, float]],
    greedy_log: dict[str, dict[int, float]],
) -> list[dict[str, Any]]:
    """Convert sidecars to normal metric rows; later train rows take priority."""

    by_step: dict[int, dict[str, Any]] = {}
    for source in (coverage, greedy_log):
        for metric, values in source.items():
            for step, value in values.items():
                record = by_step.setdefault(
                    step,
                    {"trainer/global_step": step},
                )
                record[metric] = value
    return [by_step[step] for step in sorted(by_step)]


def _series(
    records: list[dict[str, Any]],
    metric: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for record in records:
        step = _finite(record.get("trainer/global_step"))
        value = _finite(record.get(metric))
        if step is not None and value is not None:
            values[int(step)] = value
    return values


def _load_runs() -> list[dict[str, Any]]:
    if not IDENTITY.is_file() or not MANIFEST.is_file():
        return []
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    expected_ids = identity.get("job_ids", [])
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if [int(row["job_id"]) for row in rows] != expected_ids:
        raise RuntimeError("E64 plot refuses manifest/identity job mismatch")
    runs: list[dict[str, Any]] = []
    for row in rows:
        job_id = int(row["job_id"])
        run_dir = _run_dir(row["run_stamp"], job_id)
        train_records = (
            _read_metrics(run_dir / "train_metrics.jsonl")
            if run_dir is not None
            else []
        )
        coverage, parseability = (
            _coverage_sidecar(run_dir / "eval_mode_coverage_draws.jsonl")
            if run_dir is not None
            else ({}, {})
        )
        records = (
            _sidecar_records(coverage, _greedy_log_sidecar(job_id))
            + train_records
        )
        runs.append(
            {
                "arm": row["arm"],
                "seed": int(row["seed"]),
                "job_id": job_id,
                "records": records,
                "parseability": parseability,
            }
        )
    return runs


def load_e64_aggregate_points() -> list[dict[str, Any]]:
    """Expose E64 in the common E58-vs-GRPO monitor's point schema."""

    metric_sources = {
        "greedy": ("eval/math/accuracy",),
        "mean8": ("eval/math/sampled_mean_at_8",),
        "pass8": ("eval/math/sampled_any_correct_at_8",),
        (
            "semantic_shannon_success_conditioned_signed_"
            "open_set_entropy_ema"
        ): (
            "train/semantic_shannon_success_conditioned_signed_"
            "open_set_entropy_ema",
        ),
        (
            "semantic_shannon_success_conditioned_signed_"
            "open_set_next_coefficient"
        ): (
            "train/semantic_shannon_success_conditioned_signed_"
            "open_set_next_coefficient",
        ),
        "canonical_replay_mass_next_alpha": (
            "train/canonical_replay_mass_next_alpha",
        ),
        "canonical_replay_balance_loss": (
            "train/canonical_replay_balance_loss",
        ),
        "canonical_replay_available_modes": (
            "train/canonical_replay_available_modes",
        ),
        "online_canonical_new_outcome_row_fraction": (
            "train/online_canonical_new_outcome_row_fraction",
        ),
        "online_canonical_mean_support_per_prompt": (
            "train/verified_discovery_mean_support_per_prompt",
            "train/online_canonical_mean_support_per_prompt",
        ),
        "online_canonical_tracked_outcomes": (
            "train/verified_discovery_cumulative_outcomes",
            "train/online_canonical_tracked_outcomes",
        ),
    }
    points: list[dict[str, Any]] = []
    for run in _load_runs():
        by_step: dict[int, dict[str, Any]] = {}
        for record in run["records"]:
            step = _finite(record.get("trainer/global_step"))
            if step is None:
                continue
            step = int(step)
            point = by_step.setdefault(
                step,
                {
                    "arm": run["arm"],
                    "seed": run["seed"],
                    "step": step,
                    "passes": step * PASSES_PER_STEP,
                },
            )
            for output_metric, source_metrics in metric_sources.items():
                for source_metric in source_metrics:
                    value = _finite(record.get(source_metric))
                    if value is not None:
                        point[output_metric] = value
                        break
        points.extend(by_step[step] for step in sorted(by_step))
    return points


def _plot_aggregate(
    axis: plt.Axes,
    runs: list[dict[str, Any]],
    *,
    metric: str | None,
    parseability: bool = False,
    treatment_only: bool = False,
    ylabel: str,
) -> float:
    observed_max = 0.0
    arms = (TREATMENT,) if treatment_only else (CONTROL, TREATMENT)
    for arm in arms:
        arm_runs = [run for run in runs if run["arm"] == arm]
        by_x: dict[int, list[float]] = defaultdict(list)
        for run in arm_runs:
            values = (
                run["parseability"]
                if parseability
                else _series(run["records"], str(metric))
            )
            xs = np.array(sorted(values), dtype=float) * PASSES_PER_STEP
            ys = np.array([values[int(step)] for step in sorted(values)])
            if len(xs):
                observed_max = max(observed_max, float(xs[-1]))
                axis.plot(
                    xs,
                    ys,
                    color=COLORS[arm],
                    alpha=0.22,
                    linewidth=0.8,
                )
            for step, value in values.items():
                by_x[int(step)].append(float(value))
        if not by_x:
            continue
        steps = sorted(by_x)
        xs = np.array(steps, dtype=float) * PASSES_PER_STEP
        means = np.array([np.mean(by_x[step]) for step in steps])
        lows = np.array([np.min(by_x[step]) for step in steps])
        highs = np.array([np.max(by_x[step]) for step in steps])
        axis.fill_between(
            xs,
            lows,
            highs,
            color=COLORS[arm],
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(
            xs,
            means,
            color=COLORS[arm],
            linewidth=2.1,
            marker="o" if len(xs) <= 12 else None,
            markersize=3.0,
            label=ARM_LABELS[arm],
        )
    axis.set_ylabel(ylabel)
    return observed_max


def _plot_mass_balance_observations(
    axis: plt.Axes,
    runs: list[dict[str, Any]],
) -> float:
    observed = 0.0
    styles = (
        ("train/canonical_replay_mass_observations", "verified mass", "-"),
        ("train/canonical_replay_observations", "known-mode balance", "--"),
    )
    arm_runs = [run for run in runs if run["arm"] == TREATMENT]
    for metric, label, linestyle in styles:
        by_x: dict[int, list[float]] = defaultdict(list)
        for run in arm_runs:
            values = _series(run["records"], metric)
            for step, value in values.items():
                by_x[step].append(value)
        if not by_x:
            continue
        steps = sorted(by_x)
        xs = np.array(steps, dtype=float) * PASSES_PER_STEP
        ys = np.array([np.mean(by_x[step]) for step in steps])
        observed = max(observed, float(xs[-1]))
        axis.plot(
            xs,
            ys,
            color=COLORS[TREATMENT],
            linewidth=2.0,
            linestyle=linestyle,
            label=label,
        )
    axis.set_ylabel("controller observations")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles,
            labels,
            frameon=False,
            fontsize=7,
            loc="upper left",
        )
    return observed


def render(out: Path) -> None:
    runs = _load_runs()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8,
            "legend.fontsize": 7.5,
        }
    )
    figure, axes = plt.subplots(2, 6, figsize=(19, 6.9), squeeze=False)
    panels = (
        (
            "Held-out greedy pass@1",
            "eval/math/accuracy",
            False,
            False,
            "accuracy",
        ),
        (
            "Held-out sampled mean@8",
            "eval/math/sampled_mean_at_8",
            False,
            False,
            "mean correctness",
        ),
        (
            "Held-out sampled pass@8",
            "eval/math/sampled_any_correct_at_8",
            False,
            False,
            "any correct",
        ),
        (
            "Greedy response length",
            "eval/math/response_tok_len",
            False,
            False,
            "tokens",
        ),
        (
            "Sampled answer extraction@8",
            None,
            True,
            False,
            "parseable fraction",
        ),
        (
            "On-policy training reward",
            "actor/rewards",
            False,
            False,
            "mean reward",
        ),
        (
            "Discovered correct prompts",
            "train/verified_discovery_tracked_prompts",
            False,
            False,
            "prompts",
        ),
        (
            "Verified support per prompt",
            "train/verified_discovery_mean_support_per_prompt",
            False,
            False,
            "support (must be ≤1)",
        ),
        (
            "Verified-mass coefficient μ",
            "train/canonical_replay_mass_next_alpha",
            False,
            True,
            "next coefficient",
        ),
        (
            "Open-set coefficient β",
            (
                "train/semantic_shannon_success_conditioned_signed_"
                "open_set_next_coefficient"
            ),
            False,
            True,
            "next coefficient",
        ),
        (
            "Controller observations",
            None,
            False,
            True,
            "observations",
        ),
        (
            "Balance gradient norm (must be 0)",
            "train/canonical_replay_balance_score_gradient_l2",
            False,
            True,
            "raw score-gradient L2",
        ),
    )
    observed = 0.0
    for index, (title, metric, parseability, treatment_only, ylabel) in enumerate(
        panels
    ):
        axis = axes.flat[index]
        if index == 10:
            observed = max(
                observed,
                _plot_mass_balance_observations(axis, runs),
            )
        else:
            observed = max(
                observed,
                _plot_aggregate(
                    axis,
                    runs,
                    metric=metric,
                    parseability=parseability,
                    treatment_only=treatment_only,
                    ylabel=ylabel,
                ),
            )
        axis.set_title(title)
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.set_xlabel("training passes")
        if index in {0, 1, 2, 4, 5, 7}:
            axis.set_ylim(bottom=0)
        if index == 7:
            axis.axhline(1.0, color="#b91c1c", linewidth=0.8, linestyle=":")
        if index == 11:
            axis.axhline(0.0, color="#b91c1c", linewidth=0.8, linestyle=":")

    x_max = min(12.0, max(0.5, observed * 1.08))
    for axis in axes.flat:
        axis.set_xlim(0.0, x_max)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=2,
            frameon=False,
        )
    figure.suptitle(
        "E64 held-out MATH-500 realism track — correctness transfer, "
        "not reasoning-mode coverage\n"
        "Qwen2.5-0.5B · MATH12K-384 train · 3 matched seeds/arm · "
        "full MATH-500 evaluation",
        fontsize=13,
        y=0.995,
    )
    if not runs:
        figure.text(
            0.5,
            0.5,
            "Matched cohort pending smoke authorization",
            ha="center",
            va="center",
            fontsize=14,
            color="#666666",
        )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.91))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out.with_suffix(".png"), dpi=180)
    figure.savefig(out.with_suffix(".pdf"))
    plt.close(figure)
    print(
        f"[e64-plot] runs={len(runs)} observed_passes={observed:.3f} "
        f"png={out.with_suffix('.png')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    render(args.out)


if __name__ == "__main__":
    main()
