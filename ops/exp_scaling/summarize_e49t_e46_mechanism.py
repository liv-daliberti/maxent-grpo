#!/usr/bin/env python3
"""Compare E49T's causal mechanism trace with E46 graph/Countdown.

This is a reporting audit, not a new advancement criterion.  It makes the
qualitative comparison explicit: validated discoveries must create
multi-route support, normalized entropy must become eligible, and the signed
Haarnoja error must drive bounded controller updates without collapsing task
quality or route coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from collections import defaultdict
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_ADVANCEMENT = (
    ROOT / "var/artifacts/e49t_natural_menu_math_toy_advancement_v1.json"
)
DEFAULT_MATH_CURVE = (
    ROOT
    / "var/artifacts/"
    "e49t_natural_menu_math_toy_05b_3ep_v1_scaling_curve.json"
)
DEFAULT_GRAPH_CURVE = (
    ROOT
    / "var/artifacts/"
    "gce46_normalized_canonical_haarnoja_05b_v1_scaling_curve.json"
)
DEFAULT_COUNTDOWN_CURVE = (
    ROOT
    / "var/artifacts/"
    "cde46_normalized_canonical_haarnoja_05b_v1_scaling_curve.json"
)
DEFAULT_ROUTE_CALIBRATION = (
    ROOT / "var/artifacts/e49t_route_confusion_calibration_v1/result.json"
)
DEFAULT_DECLARATION_CALIBRATION = (
    ROOT / "var/artifacts/e49t_declaration_mismatch_calibration_v1/result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "var/artifacts/e49t_e46_mechanism_comparison_v1.json"
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _reference_signatures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("arm") == "online_canonical_haarnoja"
            and row.get("online_canonical_dual_observations") is not None
        ):
            grouped[int(row["seed"])].append(row)

    signatures = []
    for seed, seed_rows in sorted(grouped.items()):
        eligible = [
            row
            for row in seed_rows
            if float(row.get("online_canonical_dual_observations") or 0) > 0
        ]
        if not eligible:
            continue
        first = min(eligible, key=lambda row: int(row["step"]))
        last = max(
            eligible,
            key=lambda row: (
                int(row["step"]),
                bool(row.get("mechanism_only")),
            ),
        )
        task_rows = [
            row for row in seed_rows if row.get("greedy") is not None
        ]
        task_last = (
            max(task_rows, key=lambda row: int(row["step"]))
            if task_rows
            else None
        )
        signatures.append(
            {
                "seed": seed,
                "first_observed_step": int(first["step"]),
                "terminal_step": int(last["step"]),
                "terminal_training_passes": float(
                    last.get("training_passes") or 0.0
                ),
                "terminal_mean_support_per_prompt": float(
                    last["verified_discovery_mean_support_per_prompt"]
                ),
                "terminal_controller_observations": int(
                    last["online_canonical_dual_observations"]
                ),
                "terminal_controller_steps": int(
                    last["online_canonical_dual_optimizer_steps"]
                ),
                "terminal_normalized_entropy_ema": float(
                    last["online_canonical_dual_normalized_entropy_ema"]
                ),
                "target_ratio": float(
                    last["online_canonical_dual_target_ratio"]
                ),
                "terminal_alpha": float(
                    last["online_canonical_entropy_alpha_used"]
                ),
                "terminal_task": (
                    {
                        "greedy": float(task_last["greedy"]),
                        "mean8": float(task_last["mean8"]),
                        "pass8": float(task_last["pass8"]),
                        "coverage8": float(task_last["coverage8"]),
                    }
                    if task_last
                    else None
                ),
            }
        )
    if not signatures:
        raise RuntimeError("E46 reference curve has no controller signature")
    return signatures


def _math_trace(
    advancement: dict[str, Any],
    math_curve: list[dict[str, Any]],
) -> dict[str, Any]:
    arm = advancement["arms"]["online_canonical_haarnoja"]
    rows = [
        row
        for row in math_curve
        if row.get("arm") == "online_canonical_haarnoja"
    ]
    if not rows:
        raise RuntimeError("E49T curve lost its treatment arm")
    terminal = max(rows, key=lambda row: int(row["step"]))
    route = advancement["terminal_route_coverage"][
        "online_canonical_haarnoja"
    ]
    control_route = advancement["terminal_route_coverage"]["grpo"]

    def route_summary(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item
            for key, item in value.items()
            if key not in {"prompts"}
        }

    return {
        "terminal_step": int(arm["terminal_step"]),
        "terminal_training_passes": float(
            terminal.get("training_passes") or 0.0
        ),
        "terminal_task": {
            "greedy": float(
                arm["terminal_eval"]["eval/math/accuracy"]
            ),
            "mean8": float(
                arm["terminal_eval"]["eval/math/sampled_mean_at_8"]
            ),
            "pass8": float(
                arm["terminal_eval"][
                    "eval/math/sampled_any_correct_at_8"
                ]
            ),
        },
        "bank": arm["bank"],
        "controller": arm["controller"],
        "terminal_route_coverage": route_summary(route),
        "matched_control_route_coverage": route_summary(control_route),
    }


def summarize(
    *,
    advancement_path: pathlib.Path,
    math_curve_path: pathlib.Path,
    graph_curve_path: pathlib.Path,
    countdown_curve_path: pathlib.Path,
    route_calibration_path: pathlib.Path,
    declaration_calibration_path: pathlib.Path,
    expected_advancement_schema: str = (
        "e49t_natural_menu_math_toy_advancement_v1"
    ),
    output_schema: str = "e49t_e46_mechanism_comparison_v1",
) -> dict[str, Any]:
    advancement = _load(advancement_path)
    if (
        advancement.get("schema") != expected_advancement_schema
        or advancement.get("complete_evidence") is not True
    ):
        raise RuntimeError("E49T advancement evidence is not complete")
    math_curve = _load(math_curve_path)
    graph_curve = _load(graph_curve_path)
    countdown_curve = _load(countdown_curve_path)
    if not all(
        isinstance(curve, list)
        for curve in (math_curve, graph_curve, countdown_curve)
    ):
        raise RuntimeError("mechanism comparison requires curve arrays")

    route_calibration = _load(route_calibration_path)
    declaration_calibration = _load(declaration_calibration_path)
    math = _math_trace(advancement, math_curve)
    checks = {
        "frozen_72b_route_calibration_passed": (
            route_calibration.get("pass") is True
            and route_calibration.get("counts", {}).get(
                "misassigned_positive"
            )
            == 0
            and route_calibration.get("counts", {}).get(
                "duplicate_false_new_count"
            )
            == 0
            and route_calibration.get("counts", {}).get("accepted_negative")
            == 0
        ),
        "declared_route_mismatch_is_vetoed": (
            declaration_calibration.get("pass") is True
            and declaration_calibration.get("counts", {}).get(
                "mismatch_accepted"
            )
            == 0
            and declaration_calibration.get("counts", {}).get(
                "wrong_matched"
            )
            == 0
        ),
        "validated_discoveries_create_multi_route_support": (
            math["bank"]["max_mean_support_per_prompt"] > 1.0
            and math["bank"][
                "max_support_at_least_two_prompt_fraction"
            ]
            > 0.0
        ),
        "normalized_entropy_becomes_eligible": (
            math["controller"]["max_normalized_entropy"] > 0.0
            and math["controller"]["max_entropy_estimate"] > 0.0
        ),
        "signed_haarnoja_controller_updates": (
            math["controller"]["max_observations"] > 0
            and math["controller"]["max_optimizer_steps"] > 0
            and math["controller"]["max_abs_entropy_error"] > 0.0
            and math["controller"]["max_abs_alpha_gradient"] > 0.0
        ),
        "bounded_alpha_response_recorded": bool(
            advancement["checks"]["alpha_responds_to_entropy_error"]
        ),
        "matched_task_quality_not_collapsed": bool(
            advancement["checks"]["task_quality_not_collapsed_vs_control"]
        ),
        "terminal_route_coverage_preserved_or_improved": bool(
            advancement["checks"]["route_coverage_preserved_or_improved"]
        ),
        "terminal_natural_support_retained": bool(
            advancement["checks"].get(
                "treatment_retains_natural_support_on_at_least_eight",
                True,
            )
        ),
        "matched_initialization_exact": bool(
            advancement["checks"]["matched_step0_exact"]
        ),
    }
    return {
        "schema": output_schema,
        "interpretation": (
            "A qualitative E46 match requires the same causal chain, not the "
            "same support magnitude: validated discoveries grow finite "
            "support; normalized entropy becomes defined; signed Haarnoja "
            "updates respond within bounds; task and route quality survive."
        ),
        "inputs": {
            "advancement_sha256": _sha256(advancement_path),
            "math_curve_sha256": _sha256(math_curve_path),
            "graph_curve_sha256": _sha256(graph_curve_path),
            "countdown_curve_sha256": _sha256(countdown_curve_path),
            "route_calibration_sha256": _sha256(
                route_calibration_path
            ),
            "declaration_calibration_sha256": _sha256(
                declaration_calibration_path
            ),
        },
        "math": math,
        "e46_references": {
            "graph_coloring": _reference_signatures(graph_curve),
            "countdown": _reference_signatures(countdown_curve),
        },
        "checks": checks,
        "qualitative_mechanism_match": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--advancement", type=pathlib.Path, default=DEFAULT_ADVANCEMENT
    )
    parser.add_argument(
        "--math-curve", type=pathlib.Path, default=DEFAULT_MATH_CURVE
    )
    parser.add_argument(
        "--graph-curve", type=pathlib.Path, default=DEFAULT_GRAPH_CURVE
    )
    parser.add_argument(
        "--countdown-curve",
        type=pathlib.Path,
        default=DEFAULT_COUNTDOWN_CURVE,
    )
    parser.add_argument(
        "--route-calibration",
        type=pathlib.Path,
        default=DEFAULT_ROUTE_CALIBRATION,
    )
    parser.add_argument(
        "--declaration-calibration",
        type=pathlib.Path,
        default=DEFAULT_DECLARATION_CALIBRATION,
    )
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-advancement-schema",
        default="e49t_natural_menu_math_toy_advancement_v1",
    )
    parser.add_argument(
        "--output-schema",
        default="e49t_e46_mechanism_comparison_v1",
    )
    args = parser.parse_args()
    result = summarize(
        advancement_path=args.advancement.resolve(),
        math_curve_path=args.math_curve.resolve(),
        graph_curve_path=args.graph_curve.resolve(),
        countdown_curve_path=args.countdown_curve.resolve(),
        route_calibration_path=args.route_calibration.resolve(),
        declaration_calibration_path=(
            args.declaration_calibration.resolve()
        ),
        expected_advancement_schema=args.expected_advancement_schema,
        output_schema=args.output_schema,
    )
    _write_json(args.out.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
