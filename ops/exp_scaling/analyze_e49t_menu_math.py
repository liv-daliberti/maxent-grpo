#!/usr/bin/env python3
"""Gate E49T-style natural-menu MATH runs and compare their mechanism to E46."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "ops/exp_scaling/analyze_e49d_maximal_support_math.py"
ARMS = ("grpo", "online_canonical_haarnoja")


def _load_base():
    spec = importlib.util.spec_from_file_location("e49t_e49d_analysis", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load analysis base: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _audit_metrics(audit: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        sampled = audit["arms"][arm][
            "fixed_seed_sampled_k_neutral"
        ]["metrics"]
        greedy = audit["arms"][arm][
            "deterministic_greedy_trace_neutral"
        ]["metrics"]
        result[arm] = {
            "execution_gated_greedy": float(
                greedy["execution_gated_mean_correct"]
            ),
            "execution_gated_pass8": float(
                sampled["execution_gated_pass_at_k"]
            ),
            "eligible_distinct8": float(
                sampled[
                    "eligible_audited_distinct_correct_strategies_mean"
                ]
            ),
            "eligible_support2_fraction": float(
                sampled[
                    "eligible_audited_support_at_least_two_prompt_fraction"
                ]
            ),
            "multi_route_eligible_fraction": float(
                sampled["multi_route_eligible_prompt_fraction"]
            ),
        }
    return result


def analyze(
    *,
    stage: str,
    prefix: str,
    pool_size: int,
    terminal_step: int,
    audit_path: pathlib.Path,
    output: pathlib.Path,
    required_multi_route_fraction: float,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema") != "e49t_terminal_natural_menu_eval_v1"
        or audit.get("identity", {}).get("prefix") != prefix
    ):
        raise RuntimeError("E49T terminal audit does not match requested run")

    base = _load_base()
    base.PREFIX = {stage: prefix}
    base.POOL = {stage: pool_size}
    base.TERMINAL_STEPS = {stage: terminal_step}
    base._eval_audit = lambda unused_stage: (audit, None)
    base_output = output.with_suffix(".base.json")
    raw = base.analyze(stage, base_output)
    if not raw.get("terminal"):
        report = {
            "schema": "e49t_menu_math_stage_report_v1",
            "stage": stage,
            "prefix": prefix,
            "status": raw.get("status"),
            "pass": False,
            "advance_to_full": False,
            "base_report": str(base_output.relative_to(ROOT)),
        }
        _write_json(output, report)
        return report

    terminal = raw["terminal"]
    training_support = raw["training_support"]
    audited = _audit_metrics(audit)
    mechanism_checks = {
        key: bool(value)
        for key, value in raw["mechanism_checks"].items()
        if key
        != "double_certified_multi_route_eval_fraction_at_least_twenty_percent"
    }
    mechanism_checks["certified_multi_route_eval_fraction_sufficient"] = all(
        audited[arm]["multi_route_eligible_fraction"]
        >= required_multi_route_fraction - 1e-12
        for arm in ARMS
    )
    mechanism_checks["normalized_entropy_eligible"] = mechanism_checks.get(
        "haarnoja_received_eligible_observations", False
    )
    mechanism_checks["e46_qualitative_chain_reproduced"] = all(
        mechanism_checks.get(key, False)
        for key in (
            "treatment_terminal_discovered_multiple_training_routes",
            "normalized_entropy_eligible",
            "haarnoja_received_eligible_observations",
            "haarnoja_gradient_sign_correct",
            "alpha_finite_and_bounded",
        )
    )

    control = terminal["grpo"]
    treatment = terminal["online_canonical_haarnoja"]
    tolerance = 0.05
    quality_checks = {
        "treatment_raw_greedy_no_more_than_five_points_below_control": (
            float(treatment["greedy"]) + tolerance + 1e-12
            >= float(control["greedy"])
        ),
        "treatment_raw_pass8_no_more_than_five_points_below_control": (
            float(treatment["pass8"]) + tolerance + 1e-12
            >= float(control["pass8"])
        ),
        "treatment_execution_gated_pass8_no_more_than_five_points_below_control": (
            audited["online_canonical_haarnoja"]["execution_gated_pass8"]
            + tolerance
            + 1e-12
            >= audited["grpo"]["execution_gated_pass8"]
        ),
        "treatment_preserves_or_improves_eligible_support2": (
            audited["online_canonical_haarnoja"][
                "eligible_support2_fraction"
            ]
            + 1e-12
            >= audited["grpo"]["eligible_support2_fraction"]
        ),
        "treatment_preserves_or_improves_eligible_distinct8": (
            audited["online_canonical_haarnoja"]["eligible_distinct8"]
            + 1e-12
            >= audited["grpo"]["eligible_distinct8"]
        ),
    }
    passed = all(mechanism_checks.values()) and all(quality_checks.values())
    report = {
        "schema": "e49t_menu_math_stage_report_v1",
        "stage": stage,
        "prefix": prefix,
        "run_complete": raw["run_complete"],
        "run_paths": raw["run_paths"],
        "initial": raw["initial"],
        "terminal": terminal,
        "terminal_execution_audit": audited,
        "training_support": training_support,
        "mechanism_checks": mechanism_checks,
        "quality_checks": quality_checks,
        "e46_reference_signature": {
            "countdown": {
                "first_eligible_steps": [26, 42],
                "max_bank_sizes": [3, 3],
                "controller_observations": [128, 82],
            },
            "graph_coloring": {
                "first_eligible_steps": [2, 3, 2],
                "max_bank_sizes": [11, 7, 6],
                "controller_observations": [1461, 556, 224],
                "observed_alpha_max": 0.23553691804409027,
            },
            "comparison_rule": (
                "validated discoveries create multi-support; normalized "
                "entropy becomes eligible; the signed Haarnoja error drives "
                "bounded alpha updates"
            ),
        },
        "base_report": str(base_output.relative_to(ROOT)),
        "terminal_failure": not passed,
        "pass": passed,
        "advance_to_full": bool(stage == "toy" and passed),
        "status": "pass" if passed else "terminal_gate_failed",
    }
    _write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("toy", "full"), required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--terminal-step", type=int, required=True)
    parser.add_argument("--audit", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--required-multi-route-fraction",
        type=float,
        required=True,
    )
    args = parser.parse_args()
    if (
        args.pool_size <= 0
        or args.terminal_step <= 0
        or not 0 < args.required_multi_route_fraction <= 1
    ):
        raise SystemExit("invalid E49T analysis bounds")
    report = analyze(
        stage=args.stage,
        prefix=args.prefix,
        pool_size=args.pool_size,
        terminal_step=args.terminal_step,
        audit_path=args.audit.resolve(),
        output=args.output.resolve(),
        required_multi_route_fraction=args.required_multi_route_fraction,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
