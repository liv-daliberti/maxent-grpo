#!/usr/bin/env python3
"""Gate E49C's matched finite-action MATH toy and full stages."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
PREFIX = {
    "toy": "e49c_finite_action_math_toy_05b_v1",
    "full": "e49c_finite_action_math_full_05b_v1",
}
POOL = {"toy": 50, "full": 384}
TERMINAL_STEPS = {"toy": 150, "full": 1152}
ARMS = ("grpo", "online_canonical_haarnoja")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_name, path)


def _run_dirs(prefix: str, arm: str) -> list[pathlib.Path]:
    pattern = (
        "xdr_qwen25_0p5b_instruct_"
        f"{arm}_{prefix}_{arm}_s45"
    )
    return sorted((ROOT / "var/data").glob(pattern))


def _run_complete(prefix: str, arm: str) -> tuple[bool, list[str]]:
    complete = [
        str(path.relative_to(ROOT))
        for path in _run_dirs(prefix, arm)
        if (path / "TRAINING_COMPLETE.json").is_file()
    ]
    return len(complete) == 1, complete


def _training_records(prefix: str, arm: str) -> list[dict[str, Any]]:
    records = []
    sequence = 0
    for run_dir in _run_dirs(prefix, arm):
        for metrics_path in sorted(run_dir.glob("debug_*/train_metrics.jsonl")):
            for line_number, line in enumerate(
                metrics_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"non-object metric at {metrics_path}:{line_number}"
                    )
                sequence += 1
                row["__sequence"] = sequence
                row["__path"] = str(metrics_path.relative_to(ROOT))
                records.append(row)
    return records


def _latest(records: list[dict[str, Any]], key: str) -> float | None:
    candidates = [
        row
        for row in records
        if _finite(row.get("misc/global_step")) and _finite(row.get(key))
    ]
    if not candidates:
        return None
    row = max(
        candidates,
        key=lambda value: (
            float(value["misc/global_step"]),
            int(value["__sequence"]),
        ),
    )
    return float(row[key])


def _support_never_decreases(records: list[dict[str, Any]]) -> bool:
    by_path: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_path.setdefault(row["__path"], []).append(row)
    observed = False
    for path_rows in by_path.values():
        previous_step = None
        previous_support = None
        for row in path_rows:
            if not _finite(row.get("misc/global_step")):
                continue
            step = float(row["misc/global_step"])
            if previous_step is not None and step < previous_step:
                previous_support = None
            previous_step = step
            value = row.get("train/verified_discovery_cumulative_outcomes")
            if not _finite(value):
                continue
            value = float(value)
            observed = True
            if (
                previous_support is not None
                and value < previous_support - 1e-8
            ):
                return False
            previous_support = value
    return observed


def _parse_curve(stage: str, output: pathlib.Path) -> list[dict[str, Any]]:
    curve_path = output.with_suffix(".curve.json")
    command = [
        sys.executable,
        str(ROOT / "ops/exp_scaling/parse_scaling_curve.py"),
        "--stamp-prefix",
        PREFIX[stage],
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
    return json.loads(curve_path.read_text(encoding="utf-8"))


def _points(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
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
        for arm in ARMS
    }


def _eval_audit(stage: str) -> tuple[dict[str, Any] | None, str | None]:
    output = (
        ROOT
        / "var/artifacts"
        / f"e49c_{stage}_terminal_execution_audit.json"
    )
    if not output.is_file():
        return None, "terminal execution audit has not completed on node302"
    return json.loads(output.read_text(encoding="utf-8")), None


def _terminal_audit_metrics(
    audit: dict[str, Any],
    arm: str,
) -> dict[str, float]:
    sampled = audit["arms"][arm]["fixed_seed_sampled_k_neutral"]["metrics"]
    greedy = audit["arms"][arm][
        "deterministic_greedy_trace_neutral"
    ]["metrics"]
    return {
        "execution_gated_greedy": float(
            greedy["execution_gated_mean_correct"]
        ),
        "execution_gated_pass8": float(
            sampled["execution_gated_pass_at_k"]
        ),
        "audited_distinct8": float(
            sampled["audited_distinct_correct_strategies_mean"]
        ),
        "audited_support2_fraction": float(
            sampled["audited_support_at_least_two_prompt_fraction"]
        ),
        "raw_greedy_from_trace": float(greedy["raw_mean_correct"]),
        "raw_pass8_from_trace": float(sampled["raw_pass_at_k"]),
    }


def analyze(stage: str, output: pathlib.Path) -> dict[str, Any]:
    prefix = PREFIX[stage]
    complete = {}
    run_paths = {}
    for arm in ARMS:
        complete[arm], run_paths[arm] = _run_complete(prefix, arm)
    train = {
        arm: _training_records(prefix, arm)
        for arm in ARMS
    }
    if not all(complete.values()):
        report = {
            "schema": "e49c_finite_action_math_stage_report_v1",
            "stage": stage,
            "prefix": prefix,
            "run_complete": complete,
            "run_paths": run_paths,
            "terminal_failure": False,
            "pass": False,
            "advance_to_full": False,
            "status": "waiting_for_matched_jobs",
            "progress": {
                arm: _latest(rows, "misc/global_step")
                for arm, rows in train.items()
            },
        }
        _write_json(output, report)
        return report

    rows = _parse_curve(stage, output)
    by_arm = _points(rows)
    if any(len(by_arm[arm]) < 4 for arm in ARMS):
        report = {
            "schema": "e49c_finite_action_math_stage_report_v1",
            "stage": stage,
            "prefix": prefix,
            "run_complete": complete,
            "run_paths": run_paths,
            "terminal_failure": True,
            "pass": False,
            "advance_to_full": False,
            "status": "completed_jobs_missing_frozen_eval_points",
        }
        _write_json(output, report)
        return report

    audit, audit_error = _eval_audit(stage)
    if audit is None:
        report = {
            "schema": "e49c_finite_action_math_stage_report_v1",
            "stage": stage,
            "prefix": prefix,
            "run_complete": complete,
            "run_paths": run_paths,
            "terminal_failure": False,
            "pass": False,
            "advance_to_full": False,
            "status": "waiting_for_terminal_execution_audit",
            "audit_error": audit_error,
        }
        _write_json(output, report)
        return report

    initial = {arm: by_arm[arm][0] for arm in ARMS}
    terminal = {arm: by_arm[arm][-1] for arm in ARMS}
    baseline = terminal["grpo"]
    treatment = terminal["online_canonical_haarnoja"]
    audit_metrics = {
        arm: _terminal_audit_metrics(audit, arm) for arm in ARMS
    }
    canonicalizer_rows = [
        row
        for rows_for_arm in train.values()
        for row in rows_for_arm
        if _finite(row.get("train/math_strategy_validator_positive_rows"))
    ]
    control_rows = [
        row
        for row in train["online_canonical_haarnoja"]
        if _finite(row.get("train/online_canonical_dual_alpha_before"))
        and _finite(row.get("train/online_canonical_dual_next_alpha"))
        and _finite(row.get("train/online_canonical_dual_alpha_gradient"))
        and _finite(row.get("train/online_canonical_dual_entropy_error"))
        and float(
            row.get("train/online_canonical_dual_observation_skipped") or 0
        )
        < 0.5
    ]

    accounting_exact = bool(canonicalizer_rows) and all(
        abs(
            sum(
                float(row.get(key) or 0)
                for key in (
                    "train/math_strategy_accepted_rows",
                    "train/math_strategy_rejected_integrity_rows",
                    "train/math_strategy_rejected_ambiguous_rows",
                    "train/math_strategy_rejected_disagreement_rows",
                    "train/math_strategy_rejected_contract_rows",
                )
            )
            - float(
                row.get("train/math_strategy_validator_positive_rows") or 0
            )
        )
        <= 1e-8
        for row in canonicalizer_rows
    )
    support_mean = _latest(
        train["online_canonical_haarnoja"],
        "train/verified_discovery_mean_support_per_prompt",
    )
    support2_fraction = _latest(
        train["online_canonical_haarnoja"],
        "train/online_canonical_support_at_least_two_prompt_fraction",
    )
    final_alpha = _latest(
        train["online_canonical_haarnoja"],
        "train/online_canonical_dual_next_alpha",
    )
    mechanism_checks = {
        "matched_jobs_complete": all(complete.values()),
        "both_reached_exact_terminal_step": all(
            (_latest(train[arm], "misc/global_step") or -1)
            >= TERMINAL_STEPS[stage]
            for arm in ARMS
        ),
        "four_eval_points_each": all(
            len(by_arm[arm]) >= 4 for arm in ARMS
        ),
        "step0_raw_metrics_identical": all(
            abs(
                float(initial["grpo"][key])
                - float(initial["online_canonical_haarnoja"][key])
            )
            <= 1e-12
            for key in ("greedy", "pass8")
        ),
        "canonicalizer_accounting_exact": accounting_exact,
        "judge_calls_are_menu_bounded": (
            bool(canonicalizer_rows)
            and all(
                float(row.get("train/math_strategy_judge_calls") or 0)
                in {0.0, 2.0}
                for row in canonicalizer_rows
            )
        ),
        "no_judge_format_failures": all(
            float(
                row.get("train/math_strategy_judge_format_failure_rows") or 0
            )
            == 0
            for row in canonicalizer_rows
        ),
        "task_reward_gate_active": (
            bool(canonicalizer_rows)
            and all(
                float(
                    row.get(
                        "train/math_strategy_task_reward_gate_active"
                    )
                    or 0
                )
                == 1.0
                for row in canonicalizer_rows
            )
        ),
        "gated_reward_never_exceeds_raw": all(
            float(
                row.get("train/math_strategy_gated_task_reward_mean") or 0
            )
            <= float(
                row.get("train/math_strategy_raw_task_reward_mean") or 0
            )
            + 1e-12
            for row in canonicalizer_rows
        ),
        "baseline_exploration_influence_zero": all(
            abs(
                float(
                    row.get(
                        "train/online_canonical_combined_advantage_rms"
                    )
                    or 0
                )
            )
            <= 1e-12
            for row in train["grpo"]
        ),
        "treatment_exploration_live": any(
            float(
                row.get(
                    "train/online_canonical_combined_advantage_rms"
                )
                or 0
            )
            > 0
            for row in train["online_canonical_haarnoja"]
        ),
        "treatment_terminal_mean_support_at_least_two": (
            support_mean is not None and support_mean >= 2.0
        ),
        "treatment_terminal_support2_fraction_at_least_half": (
            support2_fraction is not None and support2_fraction >= 0.50
        ),
        "verified_support_never_decreases": _support_never_decreases(
            train["online_canonical_haarnoja"]
        ),
        "haarnoja_received_eligible_observations": bool(control_rows),
        "haarnoja_gradient_sign_correct": (
            bool(control_rows)
            and all(
                float(row["train/online_canonical_dual_alpha_gradient"])
                * float(row["train/online_canonical_dual_entropy_error"])
                >= -1e-12
                for row in control_rows
            )
        ),
        "alpha_finite_and_bounded": (
            final_alpha is not None
            and math.isfinite(final_alpha)
            and 0.10 - 1e-9 <= final_alpha <= 0.50 + 1e-9
        ),
        "terminal_trace_matches_curve": all(
            abs(
                audit_metrics[arm]["raw_greedy_from_trace"]
                - float(terminal[arm]["greedy"])
            )
            <= 1e-12
            and abs(
                audit_metrics[arm]["raw_pass8_from_trace"]
                - float(terminal[arm]["pass8"])
            )
            <= 1e-12
            for arm in ARMS
        ),
    }
    quality_checks = {
        "treatment_execution_gated_pass8_at_least_control": (
            audit_metrics["online_canonical_haarnoja"][
                "execution_gated_pass8"
            ]
            + 1e-12
            >= audit_metrics["grpo"]["execution_gated_pass8"]
        ),
        "treatment_raw_pass8_at_least_control": (
            float(treatment["pass8"]) + 1e-12
            >= float(baseline["pass8"])
        ),
        "treatment_raw_greedy_within_five_points": (
            float(treatment["greedy"]) + 0.05 + 1e-12
            >= float(baseline["greedy"])
        ),
    }
    if stage == "full":
        quality_checks[
            "treatment_eval_support2_fraction_plus_five_points"
        ] = (
            audit_metrics["online_canonical_haarnoja"][
                "audited_support2_fraction"
            ]
            >= audit_metrics["grpo"]["audited_support2_fraction"]
            + 0.05
            - 1e-12
        )
    passed = all(mechanism_checks.values()) and all(quality_checks.values())
    report = {
        "schema": "e49c_finite_action_math_stage_report_v1",
        "stage": stage,
        "prefix": prefix,
        "run_complete": complete,
        "run_paths": run_paths,
        "initial": initial,
        "terminal": terminal,
        "terminal_execution_audit": audit_metrics,
        "training_support": {
            "treatment_mean_support_per_prompt": support_mean,
            "treatment_support_at_least_two_prompt_fraction": support2_fraction,
            "treatment_final_alpha": final_alpha,
        },
        "mechanism_checks": mechanism_checks,
        "quality_checks": quality_checks,
        "terminal_failure": not passed,
        "pass": passed,
        "advance_to_full": bool(stage == "toy" and passed),
        "status": "pass" if passed else "terminal_gate_failed",
    }
    _write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(PREFIX), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = analyze(args.stage, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
