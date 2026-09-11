#!/usr/bin/env python3
"""Gate and summarize E49B's matched toy/full MATH stages."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFIX = {
    "toy": "e49b_math_strategy_toy_05b_v1",
    "full": "e49b_math_strategy_full_05b_v1",
}
POOL = {"toy": 50, "full": 384}


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _run_complete(prefix: str, arm: str) -> tuple[bool, list[str]]:
    pattern = (
        "xdr_qwen25_0p5b_instruct_"
        f"{arm}_{prefix}_{arm}_s45"
    )
    matches = sorted((ROOT / "var/data").glob(pattern))
    complete = [
        str(path.relative_to(ROOT))
        for path in matches
        if (path / "TRAINING_COMPLETE.json").is_file()
    ]
    return bool(complete), complete


def _training_records(prefix: str, arm: str) -> list[dict]:
    pattern = (
        "xdr_qwen25_0p5b_instruct_"
        f"{arm}_{prefix}_{arm}_s45"
    )
    records = []
    for run_dir in sorted((ROOT / "var/data").glob(pattern)):
        for metrics_path in sorted(run_dir.glob("debug_*/train_metrics.jsonl")):
            for line_number, line in enumerate(
                metrics_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"non-object metric at {metrics_path}:{line_number}"
                    )
                record["__metrics_path"] = str(
                    metrics_path.relative_to(ROOT)
                )
                record["__line_number"] = line_number
                records.append(record)
    return records


def _support_never_decreases(records: list[dict]) -> bool:
    """Check every monotone-step attempt segment without crossing a rollback."""

    by_path: dict[str, list[dict]] = {}
    for record in records:
        by_path.setdefault(record["__metrics_path"], []).append(record)
    observed = False
    for path_records in by_path.values():
        previous_step = None
        previous_support = None
        for record in path_records:
            step = record.get("misc/global_step")
            support = record.get(
                "train/verified_discovery_cumulative_outcomes"
            )
            if not _finite(step):
                continue
            step = float(step)
            if previous_step is not None and step < previous_step:
                previous_support = None
            previous_step = step
            if not _finite(support):
                continue
            support = float(support)
            observed = True
            if (
                previous_support is not None
                and support < previous_support - 1e-8
            ):
                return False
            previous_support = support
    return observed


def analyze(stage: str, output: Path) -> dict:
    prefix = PREFIX[stage]
    curve_path = output.with_suffix(".curve.json")
    command = [
        sys.executable,
        str(ROOT / "ops/exp_scaling/parse_scaling_curve.py"),
        "--stamp-prefix",
        prefix,
        "--prompt-pool-size",
        str(POOL[stage]),
        "--num-samples",
        "16",
        "--max-training-passes",
        "3",
        "--eval-splits",
        "math",
        "--out",
        str(curve_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    rows = json.loads(curve_path.read_text(encoding="utf-8"))
    arms = ("grpo", "online_canonical_haarnoja")
    by_arm = {
        arm: sorted(
            (
                row
                for row in rows
                if row.get("arm") == arm
                and int(row.get("seed", -1)) == 45
                and row.get("split") == "math"
            ),
            key=lambda row: float(row.get("training_passes") or 0),
        )
        for arm in arms
    }
    complete = {}
    paths = {}
    for arm in arms:
        complete[arm], paths[arm] = _run_complete(prefix, arm)

    points = {}
    for arm, arm_rows in by_arm.items():
        points[arm] = {
            "count": len(arm_rows),
            "initial": arm_rows[0] if arm_rows else None,
            "terminal": arm_rows[-1] if arm_rows else None,
        }
    treatment = points["online_canonical_haarnoja"]
    baseline = points["grpo"]
    t0 = treatment["initial"] or {}
    t1 = treatment["terminal"] or {}
    b0 = baseline["initial"] or {}
    b1 = baseline["terminal"] or {}
    terminal_passes = float(t1.get("training_passes") or 0.0)
    treatment_rows = by_arm["online_canonical_haarnoja"]
    baseline_rows = by_arm["grpo"]
    treatment_train_rows = _training_records(
        prefix, "online_canonical_haarnoja"
    )
    baseline_train_rows = _training_records(prefix, "grpo")
    all_train_rows = baseline_train_rows + treatment_train_rows
    canonicalizer_rows = [
        row
        for row in all_train_rows
        if _finite(row.get("train/math_strategy_validator_positive_rows"))
    ]
    treatment_control_rows = [
        row
        for row in treatment_train_rows
        if _finite(row.get("train/online_canonical_dual_alpha_before"))
        and _finite(row.get("train/online_canonical_dual_next_alpha"))
        and _finite(row.get("train/online_canonical_dual_entropy_error"))
        and float(
            row.get("train/online_canonical_dual_observation_skipped") or 0
        )
        < 0.5
    ]

    mechanism_checks = {
        "matched_jobs_complete": all(complete.values()),
        "treatment_reached_three_passes": terminal_passes >= 3.0 - 1e-9,
        "four_or_more_eval_points_each": all(
            points[arm]["count"] >= 4 for arm in arms
        ),
        "validator_positive_coverage_nonzero": (
            any(
                float(
                    row.get(
                        "train/online_canonical_canonicalizable_correct_fraction"
                    )
                    or 0
                )
                > 0
                for row in treatment_train_rows
            )
        ),
        "support_reached_two": (
            any(
                float(
                    row.get(
                        "train/verified_discovery_mean_support_per_prompt"
                    )
                    or 0
                )
                > 1
                for row in treatment_train_rows
            )
        ),
        "terminal_support_at_least_two": (
            _finite(
                t1.get("verified_discovery_mean_support_per_prompt")
            )
            and float(
                t1["verified_discovery_mean_support_per_prompt"]
            )
            >= 2.0
        ),
        "treatment_exploration_is_live": (
            any(
                float(
                    row.get(
                        "train/online_canonical_combined_advantage_rms"
                    )
                    or 0
                )
                > 0
                for row in treatment_train_rows
            )
        ),
        "baseline_objective_influence_zero": (
            all(
                abs(
                    float(
                        row.get(
                            "train/online_canonical_combined_advantage_rms"
                        )
                        or 0
                    )
                )
                <= 1e-12
                for row in baseline_train_rows
            )
        ),
        "canonicalizer_accounting_is_exact": (
            bool(canonicalizer_rows)
            and all(
                abs(
                    float(
                        row.get("train/math_strategy_accepted_rows") or 0
                    )
                    + float(
                        row.get(
                            "train/math_strategy_rejected_integrity_rows"
                        )
                        or 0
                    )
                    + float(
                        row.get(
                            "train/math_strategy_rejected_ambiguous_rows"
                        )
                        or 0
                    )
                    + float(
                        row.get(
                            "train/math_strategy_rejected_disagreement_rows"
                        )
                        or 0
                    )
                    - float(
                        row.get(
                            "train/math_strategy_validator_positive_rows"
                        )
                        or 0
                    )
                )
                <= 1e-8
                for row in canonicalizer_rows
            )
        ),
        "judge_call_count_is_valid": (
            bool(canonicalizer_rows)
            and all(
                any(
                    abs(
                        float(
                            row.get("train/math_strategy_judge_calls") or 0
                        )
                        - expected
                    )
                    <= 1e-8
                    for expected in (0.0, 2.0, 4.0, 6.0)
                )
                for row in canonicalizer_rows
            )
        ),
        "normalized_entropy_eligible": (
            any(
                float(
                    row.get(
                        "train/online_canonical_normalized_entropy_ratio_eligible_fraction"
                    )
                    or 0
                )
                > 0
                for row in treatment_train_rows
            )
        ),
        "terminal_normalized_entropy_not_collapsed": (
            _finite(
                t1.get(
                    "online_canonical_dual_normalized_entropy_ema"
                )
            )
            and float(
                t1[
                    "online_canonical_dual_normalized_entropy_ema"
                ]
            )
            >= 0.50
        ),
        "haarnoja_observed": (
            float(t1.get("online_canonical_dual_observations") or 0) > 0
        ),
        "haarnoja_update_direction_is_correct": (
            bool(treatment_control_rows)
            and all(
                (
                    float(
                        row["online_canonical_dual_next_alpha"]
                        if "online_canonical_dual_next_alpha" in row
                        else row[
                            "train/online_canonical_dual_next_alpha"
                        ]
                    )
                    >= float(
                        row["train/online_canonical_dual_alpha_before"]
                    )
                    - 1e-10
                )
                if float(
                    row["train/online_canonical_dual_entropy_error"]
                )
                < 0
                else (
                    float(
                        row["train/online_canonical_dual_next_alpha"]
                    )
                    <= float(
                        row["train/online_canonical_dual_alpha_before"]
                    )
                    + 1e-10
                )
                for row in treatment_control_rows
            )
        ),
        "validated_support_never_decreases": (
            _support_never_decreases(treatment_train_rows)
        ),
        "alpha_finite_and_bounded": (
            _finite(t1.get("online_canonical_dual_next_alpha"))
            and 0.10 - 1e-8
            <= float(t1["online_canonical_dual_next_alpha"])
            <= 0.50 + 1e-8
        ),
        "no_nonfinite_quality": all(
            _finite(row.get(metric))
            for arm_rows in by_arm.values()
            for row in arm_rows
            for metric in ("greedy", "pass8")
        ),
    }
    treatment_greedy_delta = (
        float(t1.get("greedy")) - float(t0.get("greedy"))
        if _finite(t1.get("greedy")) and _finite(t0.get("greedy"))
        else math.nan
    )
    treatment_pass8_delta = (
        float(t1.get("pass8")) - float(t0.get("pass8"))
        if _finite(t1.get("pass8")) and _finite(t0.get("pass8"))
        else math.nan
    )
    baseline_pass8_delta = (
        float(b1.get("pass8")) - float(b0.get("pass8"))
        if _finite(b1.get("pass8")) and _finite(b0.get("pass8"))
        else math.nan
    )
    quality_checks = {
        "terminal_greedy_within_0_02_of_drgrpo": (
            _finite(t1.get("greedy"))
            and _finite(b1.get("greedy"))
            and float(t1["greedy"]) >= float(b1["greedy"]) - 0.02
        ),
        "treatment_quality_improves": (
            _finite(treatment_greedy_delta)
            and treatment_greedy_delta > 0
        )
        or (
            _finite(treatment_pass8_delta)
            and treatment_pass8_delta > 0
            and _finite(baseline_pass8_delta)
            and baseline_pass8_delta <= treatment_pass8_delta + 1e-12
        ),
    }
    passed = all(mechanism_checks.values()) and all(quality_checks.values())
    report = {
        "schema": "e49b_math_strategy_stage_report_v1",
        "stage": stage,
        "prefix": prefix,
        "run_complete": complete,
        "run_paths": paths,
        "points": points,
        "deltas": {
            "treatment_greedy": treatment_greedy_delta,
            "treatment_pass8": treatment_pass8_delta,
            "baseline_pass8": baseline_pass8_delta,
        },
        "mechanism_checks": mechanism_checks,
        "quality_checks": quality_checks,
        "pass": passed,
        "advance_to_full": bool(stage == "toy" and passed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("toy", "full"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        ROOT
        / "var/artifacts"
        / (
            "e49b_math_strategy_toy_advancement.json"
            if args.stage == "toy"
            else "e49b_math_strategy_full_report.json"
        )
    )
    report = analyze(args.stage, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
