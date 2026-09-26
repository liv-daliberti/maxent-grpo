#!/usr/bin/env python3
"""Audit E49T's matched toy result and decide whether full scaling is allowed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PREFIX = "e49t_natural_menu_math_toy_05b_3ep_v1"
ROUTE_COVERAGE = (
    ROOT / "var/artifacts/e49t_toy_route_coverage_v1/result.json"
)
EVAL_KEYS = (
    "eval/math/accuracy",
    "eval/math/sampled_mean_at_8",
    "eval/math/sampled_any_correct_at_8",
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest_rows(prefix: str) -> list[dict[str, str]]:
    path = ROOT / "var/artifacts" / f"{prefix}_comparative_jobs.tsv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [
        dict(zip(header, line.split("\t"), strict=True))
        for line in lines[1:]
        if line.strip()
    ]
    if {row["arm"] for row in rows} != {
        "grpo",
        "online_canonical_haarnoja",
    }:
        raise RuntimeError("E49T manifest must contain exactly the matched arms")
    return rows


def _run_root(run_stamp: str) -> pathlib.Path:
    matches = sorted(
        path
        for path in (ROOT / "var/data").glob(
            "xdr_qwen25_0p5b_instruct_*"
        )
        if path.name.endswith(run_stamp)
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"could not resolve exactly one E49T run for {run_stamp}"
        )
    return matches[0]


def _metric_records(run_root: pathlib.Path, job_id: str) -> list[dict[str, Any]]:
    path = run_root / f"debug_job{job_id}" / "train_metrics.jsonl"
    if not path.is_file():
        raise RuntimeError(f"missing E49T training metrics: {path}")
    return _load_jsonl(path)


def _latest(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=lambda row: float(row["misc/global_step"]))


def _at_step(
    records: list[dict[str, Any]],
    step: int,
    *,
    require_eval: bool = True,
) -> dict[str, Any] | None:
    matches = [
        row
        for row in records
        if int(float(row.get("misc/global_step", -1))) == step
        and (
            not require_eval
            or all(key in row for key in EVAL_KEYS)
        )
    ]
    if not matches:
        return None
    return matches[-1]


def _max_metric(
    records: list[dict[str, Any]], key: str, default: float = 0.0
) -> float:
    return max(
        (float(row[key]) for row in records if key in row),
        default=default,
    )


def _max_abs_metric(
    records: list[dict[str, Any]], key: str, default: float = 0.0
) -> float:
    return max(
        (abs(float(row[key])) for row in records if key in row),
        default=default,
    )


def _last_metric(
    records: list[dict[str, Any]], key: str, default: float = 0.0
) -> float:
    matches = [row for row in records if key in row]
    if not matches:
        return default
    return float(
        max(matches, key=lambda row: float(row["misc/global_step"]))[key]
    )


def _sum_metric(records: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in records)


def _route_terminal(
    route_result: dict[str, Any], arm: str
) -> dict[str, Any] | None:
    evaluations = route_result.get("arms", {}).get(arm, {}).get(
        "evaluations", []
    )
    if not evaluations:
        return None
    return max(evaluations, key=lambda row: int(row["step"]))


def analyze(
    *,
    prefix: str,
    route_coverage_path: pathlib.Path,
    require_complete: bool,
    expected_updates: int = 150,
) -> dict[str, Any]:
    identity_path = ROOT / "var/artifacts" / f"{prefix}_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("schema") != prefix
        or identity.get("prompt_epochs") != 3
        or identity.get("num_samples") != 16
        or identity.get("method", {}).get("controller")
        != "E46 normalized canonical-bank Haarnoja"
        or identity.get("method", {}).get("policy_entropy_adaptation")
        is not False
    ):
        raise RuntimeError("E49T frozen identity does not match the protocol")

    arms: dict[str, Any] = {}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    for row in _manifest_rows(prefix):
        run_root = _run_root(row["run_stamp"])
        complete = run_root / "TRAINING_COMPLETE.json"
        if require_complete and not complete.is_file():
            raise RuntimeError(f"E49T arm is not complete: {row['arm']}")
        records = _metric_records(run_root, row["job_id"])
        raw_records[row["arm"]] = records
        latest = _latest(records)
        step = int(float(latest["misc/global_step"]))
        eval_records = [
            record
            for record in records
            if all(key in record for key in EVAL_KEYS)
        ]
        latest_eval = max(
            eval_records,
            key=lambda record: float(record["misc/global_step"]),
        )
        latest_eval_step = int(float(latest_eval["misc/global_step"]))
        eval_steps = sorted(
            {
                int(float(record["misc/global_step"]))
                for record in records
                if all(key in record for key in EVAL_KEYS)
            }
        )
        arms[row["arm"]] = {
            "job_id": row["job_id"],
            "run_stamp": row["run_stamp"],
            "run_root": str(run_root.relative_to(ROOT)),
            "training_complete": complete.is_file(),
            "terminal_step": step,
            "eval_steps": eval_steps,
            "latest_eval_step": latest_eval_step,
            "step0": {
                key: float((_at_step(records, 0) or {}).get(key, -1.0))
                for key in EVAL_KEYS
            },
            "terminal_eval": {
                key: float(latest_eval[key])
                for key in EVAL_KEYS
            },
            "gate": {
                "validator_positive_rows_total": _sum_metric(
                    records, "train/math_strategy_validator_positive_rows"
                ),
                "accepted_rows_total": _sum_metric(
                    records, "train/math_strategy_accepted_rows"
                ),
                "inferred_unstructured_rows_total": _sum_metric(
                    records,
                    "train/math_strategy_inferred_unstructured_rows",
                ),
                "rejected_strategy_inference_rows_total": _sum_metric(
                    records,
                    "train/math_strategy_rejected_strategy_inference_rows",
                ),
            },
            "bank": {
                "max_tracked_prompts": _max_metric(
                    records, "train/online_canonical_tracked_prompts"
                ),
                "max_tracked_outcomes": _max_metric(
                    records, "train/online_canonical_tracked_outcomes"
                ),
                "max_mean_support_per_prompt": _max_metric(
                    records,
                    "train/verified_discovery_mean_support_per_prompt",
                ),
                "max_support_at_least_two_prompt_fraction": _max_metric(
                    records,
                    "train/online_canonical_support_at_least_two_prompt_fraction",
                ),
            },
            "controller": {
                "max_observations": _max_metric(
                    records, "train/online_canonical_dual_observations"
                ),
                "max_optimizer_steps": _max_metric(
                    records, "train/online_canonical_dual_optimizer_steps"
                ),
                "max_abs_entropy_error": _max_abs_metric(
                    records, "train/online_canonical_dual_entropy_error"
                ),
                "max_abs_alpha_gradient": _max_abs_metric(
                    records, "train/online_canonical_dual_alpha_gradient"
                ),
                "max_normalized_entropy": _max_metric(
                    records,
                    "train/online_canonical_normalized_entropy_ratio_mean",
                ),
                "max_entropy_estimate": _max_metric(
                    records,
                    "train/online_canonical_entropy_estimate_mean",
                ),
                "initial_alpha": (_at_step(
                    records, 1, require_eval=False
                ) or {}).get(
                    "train/online_canonical_entropy_alpha_used", 0.0
                ),
                "terminal_alpha": _last_metric(
                    records,
                    "train/online_canonical_dual_next_alpha",
                ),
            },
        }

    control = arms["grpo"]
    treatment = arms["online_canonical_haarnoja"]
    step0_exact = control["step0"] == treatment["step0"]

    route_result: dict[str, Any] | None = None
    if route_coverage_path.is_file():
        route_result = json.loads(
            route_coverage_path.read_text(encoding="utf-8")
        )
        if (
            route_result.get("schema") != "e49t_toy_route_coverage_v1"
            or route_result.get("stamp_prefix") != prefix
            or route_result.get("identity_sha256") != _sha256(identity_path)
        ):
            raise RuntimeError("E49T route coverage identity mismatch")
    route_control = (
        _route_terminal(route_result, "grpo") if route_result else None
    )
    route_treatment = (
        _route_terminal(route_result, "online_canonical_haarnoja")
        if route_result
        else None
    )

    treatment_step = treatment["terminal_step"]
    control_step = control["terminal_step"]
    task_not_collapsed = (
        treatment["terminal_eval"]["eval/math/accuracy"]
        >= control["terminal_eval"]["eval/math/accuracy"] - 0.05
        and treatment["terminal_eval"]["eval/math/sampled_mean_at_8"]
        >= control["terminal_eval"]["eval/math/sampled_mean_at_8"] - 0.05
    )
    route_preserved = bool(
        route_control
        and route_treatment
        and route_treatment[
            "mean_normalized_strategy_coverage_all_prompts"
        ]
        >= route_control[
            "mean_normalized_strategy_coverage_all_prompts"
        ]
        and route_treatment["dual_full_coverage_rate"]
        >= route_control["dual_full_coverage_rate"]
    )
    alpha_changed = (
        treatment["controller"]["max_optimizer_steps"] > 0
        and treatment["controller"]["max_abs_entropy_error"] > 0.0
        and treatment["controller"]["max_abs_alpha_gradient"] > 0.0
        and (
            abs(
                treatment["controller"]["terminal_alpha"]
                - treatment["controller"]["initial_alpha"]
            )
            > 1e-8
            or treatment["controller"]["terminal_alpha"] <= 0.1000001
            or treatment["controller"]["terminal_alpha"] >= 0.4999999
        )
    )
    checks = {
        "matched_step0_exact": step0_exact,
        "both_completed_exactly_150_updates": (
            control_step == treatment_step == expected_updates
            and control["training_complete"]
            and treatment["training_complete"]
        ),
        "shared_gate_admits_natural_derivations": (
            control["gate"]["accepted_rows_total"] > 0
            and treatment["gate"]["accepted_rows_total"] > 0
            and control["gate"]["inferred_unstructured_rows_total"] > 0
            and treatment["gate"]["inferred_unstructured_rows_total"] > 0
        ),
        "treatment_bank_reaches_support_two": (
            treatment["bank"]["max_mean_support_per_prompt"] > 1.0
            and treatment["bank"][
                "max_support_at_least_two_prompt_fraction"
            ]
            > 0.0
        ),
        "haarnoja_receives_observations_and_steps": (
            treatment["controller"]["max_observations"] > 0
            and treatment["controller"]["max_optimizer_steps"] > 0
        ),
        "normalized_entropy_is_nonzero": (
            treatment["controller"]["max_normalized_entropy"] > 0.0
            or treatment["controller"]["max_entropy_estimate"] > 0.0
        ),
        "alpha_responds_to_entropy_error": alpha_changed,
        "task_quality_not_collapsed_vs_control": task_not_collapsed,
        "route_coverage_scored": bool(route_control and route_treatment),
        "route_coverage_preserved_or_improved": route_preserved,
    }
    complete_evidence = all(
        [
            checks["both_completed_exactly_150_updates"],
            checks["route_coverage_scored"],
        ]
    )
    advance = complete_evidence and all(checks.values())
    return {
        "schema": "e49t_natural_menu_math_toy_advancement_v1",
        "stamp_prefix": prefix,
        "identity_sha256": _sha256(identity_path),
        "route_coverage_sha256": (
            _sha256(route_coverage_path)
            if route_coverage_path.is_file()
            else None
        ),
        "arms": arms,
        "terminal_route_coverage": {
            "grpo": route_control,
            "online_canonical_haarnoja": route_treatment,
        },
        "checks": checks,
        "complete_evidence": complete_evidence,
        "advance_to_exact_oat_full": advance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--route-coverage", type=pathlib.Path, default=ROUTE_COVERAGE
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=(
            ROOT
            / "var/artifacts/"
            "e49t_natural_menu_math_toy_advancement_v1.json"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-updates", type=int, default=150)
    args = parser.parse_args()
    result = analyze(
        prefix=args.stamp_prefix,
        route_coverage_path=args.route_coverage.resolve(),
        require_complete=args.require_complete,
        expected_updates=args.expected_updates,
    )
    _write_json(args.out.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
