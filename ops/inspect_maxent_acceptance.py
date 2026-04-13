#!/usr/bin/env python3
"""Summarize Selective Safe MaxEnt acceptance checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METRIC_RE = re.compile(r"'([^']+)':\s*([^,}]+)")


def _coerce_float(value: str) -> float | None:
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_log_steps(log_path: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    current: dict[str, float] = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        if "'" not in line or ":" not in line:
            continue
        for key, raw_value in METRIC_RE.findall(line):
            parsed = _coerce_float(raw_value)
            if parsed is None:
                continue
            current[key] = parsed
            if key == "misc/global_step":
                steps[int(round(parsed))] = dict(current)
                current = {}
    return steps


def parse_eval_averages(eval_dir: Path) -> dict[int, float]:
    scores_by_step: dict[int, list[float]] = defaultdict(list)
    for path in eval_dir.glob("*.json"):
        name = path.stem
        if "_" not in name:
            continue
        step_text, _ = name.split("_", 1)
        try:
            step = int(step_text)
        except ValueError:
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            continue
        problem_scores = []
        for row in payload:
            scores = row.get("scores", [])
            if not scores:
                continue
            problem_scores.append(sum(float(score) for score in scores) / len(scores))
        if problem_scores:
            scores_by_step[step].append(sum(problem_scores) / len(problem_scores))
    return {
        step: sum(values) / len(values)
        for step, values in scores_by_step.items()
        if values
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _metric(step_metrics: dict[str, float] | None, key: str) -> float | None:
    if step_metrics is None:
        return None
    return step_metrics.get(key)


def _first(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _delta(step_metrics: dict[str, float] | None, high_key: str, low_key: str) -> float | None:
    if step_metrics is None:
        return None
    if high_key not in step_metrics or low_key not in step_metrics:
        return None
    return step_metrics[high_key] - step_metrics[low_key]


def print_summary(
    *,
    label: str,
    steps: Iterable[int],
    log_steps: dict[int, dict[str, float]],
    eval_steps: dict[int, float],
) -> None:
    print(f"\n[{label}]")
    header = (
        "step  eval_avg  formatted  resp_len  moved_mass  selected  len_delta  format_delta"
    )
    print(header)
    for step in steps:
        metrics = log_steps.get(step)
        row = [
            f"{step:>4}",
            f"{_fmt(eval_steps.get(step)):>8}",
            f"{_fmt(_metric(metrics, 'actor/formatted')):>10}",
            f"{_fmt(_metric(metrics, 'actor/response_tok_len')):>9}",
            f"{_fmt(_first(_metric(metrics, 'train/listwise_semantic_moved_mass_l1'), _metric(metrics, 'listwise_semantic_moved_mass_l1'))):>11}",
            f"{_fmt(_first(_metric(metrics, 'train/listwise_semantic_prompt_selected_frac'), _metric(metrics, 'listwise_semantic_prompt_selected_frac'))):>8}",
            f"{_fmt(_first(_delta(metrics, 'train/listwise_semantic_expected_len_final_w', 'train/listwise_semantic_expected_len_q'), _delta(metrics, 'listwise_semantic_expected_len_final_w', 'listwise_semantic_expected_len_q'))):>9}",
            f"{_fmt(_first(_delta(metrics, 'train/listwise_semantic_expected_format_final_w', 'train/listwise_semantic_expected_format_q'), _delta(metrics, 'listwise_semantic_expected_format_final_w', 'listwise_semantic_expected_format_q'))):>12}",
        ]
        print("  ".join(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--candidate-eval-dir", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--baseline-eval-dir", type=Path, required=True)
    parser.add_argument("--previous-log", type=Path)
    parser.add_argument("--previous-eval-dir", type=Path)
    parser.add_argument("--steps", default="16,32,48,64,112,192")
    args = parser.parse_args()

    steps = [int(part) for part in args.steps.split(",") if part.strip()]
    print_summary(
        label="candidate",
        steps=steps,
        log_steps=parse_log_steps(args.candidate_log),
        eval_steps=parse_eval_averages(args.candidate_eval_dir),
    )
    print_summary(
        label="baseline",
        steps=steps,
        log_steps=parse_log_steps(args.baseline_log),
        eval_steps=parse_eval_averages(args.baseline_eval_dir),
    )
    if args.previous_log and args.previous_eval_dir:
        print_summary(
            label="previous",
            steps=steps,
            log_steps=parse_log_steps(args.previous_log),
            eval_steps=parse_eval_averages(args.previous_eval_dir),
        )


if __name__ == "__main__":
    main()
