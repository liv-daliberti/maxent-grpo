#!/usr/bin/env python3
"""Audit an E50 matched toy with independent base and terminal route probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_ANALYZER = (
    ROOT / "ops/exp_scaling/analyze_e49t_natural_menu_math.py"
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


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location(
        "e50_e49t_base_analyzer", BASE_ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load E49T matched analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe(path: pathlib.Path, expected_label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "e50_route_probe_result_v1"
        or payload.get("label") != expected_label
        or payload.get("status") != "pass"
        or payload.get("sample_count_per_prompt") != 64
        or payload.get("prompt_count") != 10
    ):
        raise RuntimeError(f"invalid E50 route probe: {path}")
    return payload


def _as_e49t_route_evaluation(
    probe: dict[str, Any], step: int
) -> dict[str, Any]:
    return {
        "step": step,
        "seed": 500722,
        "sample_count": 64,
        "prompt_count": 10,
        "mean_normalized_strategy_coverage_all_prompts": probe[
            "mean_normalized_route_coverage"
        ],
        "dual_prompt_count": 10,
        "dual_full_coverage_count": probe["dual_full_coverage_count"],
        "dual_full_coverage_rate": probe["dual_full_coverage_rate"],
        "accepted_correct_response_count": probe[
            "accepted_route_response_count"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("e50b", "e50d"), required=True)
    parser.add_argument("--stamp-prefix", required=True)
    parser.add_argument("--base-route-probe", type=pathlib.Path, required=True)
    parser.add_argument(
        "--control-route-probe", type=pathlib.Path, required=True
    )
    parser.add_argument(
        "--treatment-route-probe", type=pathlib.Path, required=True
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    identity_path = (
        ROOT / "var/artifacts" / f"{args.stamp_prefix}_identity.json"
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("schema") != args.stamp_prefix:
        raise RuntimeError("E50 matched identity mismatch")

    base_probe_path = args.base_route_probe.resolve()
    control_probe_path = args.control_route_probe.resolve()
    treatment_probe_path = args.treatment_route_probe.resolve()
    base_probe = _probe(base_probe_path, f"{args.variant}_baseline")
    control_probe = _probe(
        control_probe_path, f"{args.variant}_control_terminal"
    )
    treatment_probe = _probe(
        treatment_probe_path, f"{args.variant}_treatment_terminal"
    )
    expected_manifest_sha256 = identity["data_manifest_sha256"]
    probes = (base_probe, control_probe, treatment_probe)
    if any(
        probe.get("identity", {}).get("data_manifest_sha256")
        != expected_manifest_sha256
        for probe in probes
    ):
        raise RuntimeError("E50 route probe data identity mismatch")

    combined_path = (
        args.out.resolve().parent / "combined_terminal_route_coverage.json"
    )
    combined = {
        "schema": "e49t_toy_route_coverage_v1",
        "stamp_prefix": args.stamp_prefix,
        "identity_sha256": _sha256(identity_path),
        "endpoint_record_sha256": control_probe["identity"][
            "endpoint_record_sha256"
        ],
        "frozen_source_hash": identity["source_hash"],
        "judge_contract": {
            "sample_count": 64,
            "finite_menu_only": True,
            "task_reward_positive_required": True,
        },
        "arms": {
            "grpo": {
                "evaluations": [
                    _as_e49t_route_evaluation(control_probe, 150)
                ]
            },
            "online_canonical_haarnoja": {
                "evaluations": [
                    _as_e49t_route_evaluation(treatment_probe, 150)
                ]
            },
        },
    }
    _write_json(combined_path, combined)
    base = _load_base()
    result = base.analyze(
        prefix=args.stamp_prefix,
        route_coverage_path=combined_path,
        require_complete=True,
    )
    treatment_not_below_base = (
        float(treatment_probe["mean_normalized_route_coverage"])
        >= float(base_probe["mean_normalized_route_coverage"])
    )
    treatment_not_below_control = (
        float(treatment_probe["mean_normalized_route_coverage"])
        >= float(control_probe["mean_normalized_route_coverage"])
    )
    treatment_eight_dual = (
        int(treatment_probe["dual_full_coverage_count"]) >= 8
    )
    treatment_eight_natural = (
        int(treatment_probe["natural_support_prompt_count"]) >= 8
    )
    result["schema"] = "e50_matched_math_toy_advancement_v1"
    result["variant"] = args.variant
    result["route_probes"] = {
        "base": base_probe,
        "grpo_terminal": control_probe,
        "online_canonical_haarnoja_terminal": treatment_probe,
    }
    result["route_probe_sha256"] = {
        "base": _sha256(base_probe_path),
        "grpo_terminal": _sha256(control_probe_path),
        "online_canonical_haarnoja_terminal": _sha256(
            treatment_probe_path
        ),
        "combined": _sha256(combined_path),
    }
    result["checks"].update(
        {
            "base_route_probe_passed": (
                base_probe["baseline_gate"] is True
            ),
            "treatment_route_coverage_not_below_base": (
                treatment_not_below_base
            ),
            "treatment_route_coverage_not_below_control": (
                treatment_not_below_control
            ),
            "treatment_retains_both_routes_on_at_least_eight": (
                treatment_eight_dual
            ),
            "treatment_retains_natural_support_on_at_least_eight": (
                treatment_eight_natural
            ),
        }
    )
    result["complete_evidence"] = (
        result["complete_evidence"]
        and all(
            probe.get("status") == "pass"
            for probe in probes
        )
    )
    result["advance_to_exact_oat_full"] = (
        result["complete_evidence"] and all(result["checks"].values())
    )
    _write_json(args.out.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
