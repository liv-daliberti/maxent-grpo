#!/usr/bin/env python3
"""Sample exact answer-family coverage for graph-coloring checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _step_number(path: Path) -> int:
    match = re.search(r"step_(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _discover_run_dir(
    *,
    stamp_prefix: str,
    seed: int,
    variant: str,
    model_tag: str,
    run_data_root: Path,
) -> Path:
    candidates = sorted(
        run_data_root.glob(
            f"oat_zero_tiny_{model_tag}_{variant}_single_gpu_{stamp_prefix}_{variant}_s{seed}"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find run dir for "
            f"stamp={stamp_prefix} seed={seed} variant={variant} under {run_data_root}"
        )
    return candidates[-1]


def _discover_final_checkpoint(run_dir: Path) -> Path:
    candidates: list[Path] = []
    for root in [run_dir, *sorted(run_dir.glob("debug_*"))]:
        saved = root / "saved_models"
        if not saved.is_dir():
            continue
        candidates.extend(
            path
            for path in saved.glob("step_*")
            if path.is_dir() and (path / "config.json").is_file()
        )
    if not candidates:
        raise FileNotFoundError(f"No saved model checkpoints found under {run_dir}")
    return sorted(candidates, key=_step_number)[-1]


def _parse_checkpoint_specs(raw_specs: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Invalid checkpoint spec '{raw}'. Expected alias=path.")
        alias, path_text = raw.split("=", 1)
        alias = alias.strip()
        path = Path(path_text).expanduser().resolve()
        if not alias:
            raise ValueError(f"Checkpoint spec '{raw}' has an empty alias.")
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
        parsed.append((alias, path))
    return parsed


def _select_template(template: str):
    normalized = str(template).strip().lower().replace("-", "_")
    if normalized == "qwen_boxed":
        return lambda problem: (
            "<|im_start|>system\nReturn only the final answer inside \\boxed{}. "
            "Do not explain.<|im_end|>\n<|im_start|>user\n"
            + str(problem)
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    if normalized in {"raw", "no"}:
        return lambda problem: str(problem)
    raise ValueError(f"Unsupported template: {template}")


def _answer_mode_count(answer: Any, row: Mapping[str, Any]) -> int:
    if isinstance(row.get("answer_mode_count"), int):
        return max(int(row["answer_mode_count"]), 1)
    try:
        spec = json.loads(answer) if isinstance(answer, str) else answer
        return max(int(spec.get("num_completions", 1)), 1)
    except Exception:
        return 1


def compute_coverage_metrics(
    *,
    rewards: Sequence[float],
    answer_keys: Sequence[str | None],
    answer_mode_count: int,
) -> dict[str, float]:
    if len(rewards) != len(answer_keys):
        raise ValueError("rewards and answer_keys must have matching lengths.")
    if not rewards:
        raise ValueError("Need at least one sampled reward.")
    k = len(rewards)
    correct_keys = [
        str(answer_key)
        for reward, answer_key in zip(rewards, answer_keys)
        if float(reward) > 0.0 and answer_key is not None
    ]
    counts = Counter(correct_keys)
    distinct = len(counts)
    total_modes = max(int(answer_mode_count), 1)
    entropy = 0.0
    observed_entropy_norm = 0.0
    if counts:
        total_correct = float(sum(counts.values()))
        probs = [count / total_correct for count in counts.values()]
        entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0.0)
        if total_modes > 1:
            entropy /= math.log(float(total_modes))
        if distinct > 1:
            observed_entropy_norm = -sum(
                prob * math.log(prob) for prob in probs if prob > 0.0
            ) / math.log(float(distinct))
    return {
        "sampled_pass_at_1": float(float(rewards[0]) > 0.0),
        "mean_at_k": float(sum(float(value) for value in rewards) / float(k)),
        "any_correct_at_k": float(any(float(value) > 0.0 for value in rewards)),
        "distinct_correct_modes_at_k": float(distinct),
        "mode_coverage_at_k": float(distinct) / float(total_modes),
        "all_modes_covered_at_k": float(distinct >= total_modes),
        "correct_mode_entropy_total_norm": float(entropy),
        "correct_mode_entropy_observed_norm": float(observed_entropy_norm),
        "answer_key_extracted_frac": float(
            sum(answer_key is not None for answer_key in answer_keys)
        )
        / float(k),
        "correct_answer_key_extracted_frac": float(len(correct_keys))
        / max(float(sum(float(value) > 0.0 for value in rewards)), 1.0),
        "answer_mode_count": float(total_modes),
    }


def _mean_metric_dicts(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(sum(float(row.get(key, 0.0)) for row in rows) / float(len(rows)))
        for key in keys
    }


def _evaluate_checkpoint(
    *,
    alias: str,
    checkpoint_path: Path,
    datasets_by_split: Any,
    splits: Sequence[str],
    output_root: Path,
    template: str,
    sample_count: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_model_len: int,
    batch_size: int,
    seed: int,
    gpu_memory_utilization: float,
    swap_space: float,
    limit_per_split: int,
    include_text: bool,
) -> dict[str, Any]:
    import vllm
    from oat_drgrpo.math_grader import (
        boxed_reward_fn,
        extract_normalized_final_answer_for_clustering,
    )

    apply_template = _select_template(template)
    llm = vllm.LLM(
        model=str(checkpoint_path),
        dtype="bfloat16",
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
        swap_space=float(swap_space),
        enable_prefix_caching=True,
    )

    checkpoint_root = output_root / "checkpoints" / alias
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, Any] = {}
    for split in splits:
        dataset = datasets_by_split[split]
        row_count = min(len(dataset), int(limit_per_split)) if limit_per_split else len(dataset)
        rows = [dataset[index] for index in range(row_count)]
        prompts = [apply_template(row["problem"]) for row in rows]
        attempts: list[dict[str, Any]] = []
        prompt_metrics: list[dict[str, float]] = []
        params = vllm.SamplingParams(
            n=int(sample_count),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            seed=int(seed),
        )
        for batch_start in range(0, len(prompts), max(int(batch_size), 1)):
            batch_end = min(batch_start + max(int(batch_size), 1), len(prompts))
            outputs = llm.generate(prompts[batch_start:batch_end], params)
            for offset, output in enumerate(outputs):
                row_index = batch_start + offset
                row = rows[row_index]
                answer = row["answer"]
                rewards: list[float] = []
                answer_keys: list[str | None] = []
                prompt_attempts: list[dict[str, Any]] = []
                for sample_index, sample_output in enumerate(output.outputs, start=1):
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
                        "split": split,
                        "dataset_index": row_index,
                        "sample_index": sample_index,
                        "reward": float(reward),
                        "correct": bool(float(reward) > 0.0),
                        "answer_key": answer_key,
                        "token_length": len(sample_output.token_ids),
                    }
                    if include_text:
                        attempt["text"] = text
                    prompt_attempts.append(attempt)
                metrics = compute_coverage_metrics(
                    rewards=rewards,
                    answer_keys=answer_keys,
                    answer_mode_count=_answer_mode_count(answer, row),
                )
                metrics["dataset_index"] = float(row_index)
                metrics["split"] = split  # type: ignore[assignment]
                prompt_metrics.append(metrics)
                attempts.extend(prompt_attempts)

        split_root = checkpoint_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        attempts_path = split_root / "attempts.json"
        prompt_metrics_path = split_root / "prompt_metrics.json"
        summary_path = split_root / "summary.json"
        attempts_path.write_text(json.dumps(attempts, indent=2, sort_keys=True) + "\n")
        prompt_metrics_path.write_text(
            json.dumps(prompt_metrics, indent=2, sort_keys=True) + "\n"
        )
        numeric_prompt_metrics = [
            {key: value for key, value in row.items() if isinstance(value, (int, float))}
            for row in prompt_metrics
        ]
        summary = {
            "alias": alias,
            "checkpoint_path": str(checkpoint_path),
            "split": split,
            "sample_count": int(sample_count),
            "row_count": int(row_count),
            "metrics": _mean_metric_dicts(numeric_prompt_metrics),
            "attempts_path": str(attempts_path),
            "prompt_metrics_path": str(prompt_metrics_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        split_summaries[split] = summary
    return {
        "alias": alias,
        "checkpoint_path": str(checkpoint_path),
        "splits": split_summaries,
    }


def _parse_alias(alias: str) -> tuple[int | None, str]:
    match = re.search(r"_s(\d+)$", alias)
    seed = int(match.group(1)) if match else None
    variant = alias[: match.start()] if match else alias
    return seed, variant


def _pairwise_deltas(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_seed_variant: dict[tuple[int, str], Mapping[str, Any]] = {}
    for summary in summaries:
        seed, variant = _parse_alias(str(summary["alias"]))
        if seed is not None:
            by_seed_variant[(seed, variant)] = summary
    rows: list[dict[str, Any]] = []
    seeds = sorted({seed for seed, _variant in by_seed_variant})
    metrics = [
        "sampled_pass_at_1",
        "mean_at_k",
        "any_correct_at_k",
        "distinct_correct_modes_at_k",
        "mode_coverage_at_k",
        "all_modes_covered_at_k",
        "correct_mode_entropy_total_norm",
    ]
    for seed in seeds:
        baseline = by_seed_variant.get((seed, "grpo"))
        maxent = by_seed_variant.get((seed, "answer_maxent"))
        if baseline is None or maxent is None:
            continue
        for split, maxent_split in maxent["splits"].items():
            if split not in baseline["splits"]:
                continue
            base_metrics = baseline["splits"][split]["metrics"]
            maxent_metrics = maxent_split["metrics"]
            row: dict[str, Any] = {"seed": seed, "split": split}
            for metric in metrics:
                row[f"{metric}_delta"] = float(maxent_metrics.get(metric, 0.0)) - float(
                    base_metrics.get(metric, 0.0)
                )
            rows.append(row)
    return rows


def _write_markdown(
    *,
    path: Path,
    stamp_prefix: str,
    summaries: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Exact Answer-Mode Coverage",
        "",
        f"Stamp prefix: `{stamp_prefix}`.",
        "",
        "## Per Checkpoint",
        "",
        "| Alias | Split | Any@K | Mean@K | Distinct | Coverage | Entropy | All Modes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        for split, split_summary in summary["splits"].items():
            metrics = split_summary["metrics"]
            lines.append(
                "| {alias} | `{split}` | {any:.4f} | {mean:.4f} | {distinct:.4f} | "
                "{coverage:.4f} | {entropy:.4f} | {all_modes:.4f} |".format(
                    alias=summary["alias"],
                    split=split,
                    any=float(metrics.get("any_correct_at_k", 0.0)),
                    mean=float(metrics.get("mean_at_k", 0.0)),
                    distinct=float(metrics.get("distinct_correct_modes_at_k", 0.0)),
                    coverage=float(metrics.get("mode_coverage_at_k", 0.0)),
                    entropy=float(metrics.get("correct_mode_entropy_total_norm", 0.0)),
                    all_modes=float(metrics.get("all_modes_covered_at_k", 0.0)),
                )
            )
    if deltas:
        lines.extend(
            [
                "",
                "## Pairwise Deltas",
                "",
                "| Seed | Split | Any@K | Mean@K | Distinct | Coverage | Entropy |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in deltas:
            lines.append(
                "| {seed} | `{split}` | {any:.4f} | {mean:.4f} | {distinct:.4f} | "
                "{coverage:.4f} | {entropy:.4f} |".format(
                    seed=row["seed"],
                    split=row["split"],
                    any=float(row.get("any_correct_at_k_delta", 0.0)),
                    mean=float(row.get("mean_at_k_delta", 0.0)),
                    distinct=float(row.get("distinct_correct_modes_at_k_delta", 0.0)),
                    coverage=float(row.get("mode_coverage_at_k_delta", 0.0)),
                    entropy=float(
                        row.get("correct_mode_entropy_total_norm_delta", 0.0)
                    ),
                )
            )
        lines.extend(["", "## Aggregate Deltas", ""])
        for split in sorted({str(row["split"]) for row in deltas}):
            split_rows = [row for row in deltas if str(row["split"]) == split]
            coverage_delta = sum(
                float(row.get("mode_coverage_at_k_delta", 0.0)) for row in split_rows
            ) / float(len(split_rows))
            distinct_delta = sum(
                float(row.get("distinct_correct_modes_at_k_delta", 0.0))
                for row in split_rows
            ) / float(len(split_rows))
            lines.append(
                f"- `{split}`: mean coverage delta `{coverage_delta:.4f}`, "
                f"mean distinct-mode delta `{distinct_delta:.4f}`."
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-prefix", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("var/artifacts"))
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--seeds", default="43,44,45")
    parser.add_argument("--variants", default="grpo,answer_maxent")
    parser.add_argument("--model-tag", default="qwen25_0p5b_instruct")
    parser.add_argument("--splits", default="all")
    parser.add_argument("--template", default="qwen_boxed")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--swap-space", type=float, default=16.0)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()

    import sys

    repo_src = _repo_root() / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from datasets import load_from_disk

    eval_root = args.data_root / "eval"
    datasets_by_split = load_from_disk(str(eval_root))
    splits = list(datasets_by_split.keys()) if args.splits == "all" else _parse_csv(args.splits)
    checkpoints = _parse_checkpoint_specs(args.checkpoint)
    if not checkpoints:
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
                checkpoints.append((f"{variant}_s{seed}", _discover_final_checkpoint(run_dir)))

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for alias, checkpoint_path in checkpoints:
        print(f"[coverage-eval] start alias={alias} checkpoint={checkpoint_path}", flush=True)
        summaries.append(
            _evaluate_checkpoint(
                alias=alias,
                checkpoint_path=checkpoint_path,
                datasets_by_split=datasets_by_split,
                splits=splits,
                output_root=args.output_root / args.stamp_prefix,
                template=args.template,
                sample_count=args.sample_count,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_model_len=args.max_model_len,
                batch_size=args.batch_size,
                seed=args.seed,
                gpu_memory_utilization=args.gpu_memory_utilization,
                swap_space=args.swap_space,
                limit_per_split=args.limit_per_split,
                include_text=args.include_text,
            )
        )
        print(f"[coverage-eval] done alias={alias}", flush=True)

    deltas = _pairwise_deltas(summaries)
    payload = {
        "stamp_prefix": args.stamp_prefix,
        "data_root": str(args.data_root.resolve()),
        "sample_count": int(args.sample_count),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "splits": splits,
        "checkpoints": summaries,
        "pairwise_deltas": deltas,
    }
    json_path = args.output_root / f"{args.stamp_prefix}_coverage_summary.json"
    md_path = args.output_root / f"{args.stamp_prefix}_coverage_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(
        path=md_path,
        stamp_prefix=args.stamp_prefix,
        summaries=summaries,
        deltas=deltas,
    )
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
