#!/usr/bin/env python3
"""Audit the exact E50E 384-train/MATH-500 matched run."""

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
        "e50e_e49t_base_analyzer", BASE_ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base matched analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe(path: pathlib.Path, label: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("schema") != "e50_route_probe_result_v1"
        or result.get("label") != label
        or result.get("status") != "pass"
        or result.get("sample_count_per_prompt") != 64
        or result.get("prompt_count") != 10
    ):
        raise RuntimeError(f"invalid E50E route probe: {path}")
    return result


def _route_eval(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": 1152,
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
    parser.add_argument("--route-variant", choices=("e50d",), required=True)
    parser.add_argument(
        "--stamp-prefix", default="e50e_exact_oat_math_05b_3ep_v1"
    )
    parser.add_argument("--base-route-probe", type=pathlib.Path, required=True)
    parser.add_argument(
        "--control-route-probe", type=pathlib.Path, required=True
    )
    parser.add_argument(
        "--treatment-route-probe", type=pathlib.Path, required=True
    )
    parser.add_argument(
        "--continuity-certificate", type=pathlib.Path, required=True
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    identity_path = (
        ROOT / "var/artifacts" / f"{args.stamp_prefix}_identity.json"
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("schema") != args.stamp_prefix
        or identity.get("optimizer_updates") != 1152
        or identity.get("train_rows") != 384
        or identity.get("eval_rows") != 500
    ):
        raise RuntimeError("E50E full-run identity mismatch")
    continuity_path = args.continuity_certificate.resolve()
    continuity = json.loads(continuity_path.read_text(encoding="utf-8"))
    continuity_ids = continuity.get("identities") or {}
    if (
        continuity.get("schema")
        != "e50e_qwen72_continuity_certificate_v1"
        or continuity.get("pass") is not True
        or not all((continuity.get("checks") or {}).values())
        or identity.get("continuity_certificate_sha256")
        != _sha256(continuity_path)
        or identity.get("judge_endpoint_record_sha256")
        != continuity_ids.get("endpoint_record_sha256")
        or identity.get("route_calibration_result_sha256")
        != continuity_ids.get("route_result_sha256")
        or identity.get("declaration_calibration_result_sha256")
        != continuity_ids.get("declaration_result_sha256")
    ):
        raise RuntimeError("E50E Qwen72 continuity identity mismatch")

    paths = {
        "base": args.base_route_probe.resolve(),
        "grpo_terminal": args.control_route_probe.resolve(),
        "online_canonical_haarnoja_terminal": (
            args.treatment_route_probe.resolve()
        ),
    }
    probes = {
        "base": _probe(paths["base"], f"{args.route_variant}_baseline"),
        "grpo_terminal": _probe(
            paths["grpo_terminal"],
            f"{args.route_variant}_full_control_terminal",
        ),
        "online_canonical_haarnoja_terminal": _probe(
            paths["online_canonical_haarnoja_terminal"],
            f"{args.route_variant}_full_treatment_terminal",
        ),
    }
    probe_manifest_hashes = {
        probe["identity"]["data_manifest_sha256"]
        for probe in probes.values()
    }
    if len(probe_manifest_hashes) != 1:
        raise RuntimeError("E50E route probes do not share one toy identity")
    terminal_endpoint_hashes = {
        probes[name]["identity"]["endpoint_record_sha256"]
        for name in (
            "grpo_terminal",
            "online_canonical_haarnoja_terminal",
        )
    }
    if terminal_endpoint_hashes != {
        identity["judge_endpoint_record_sha256"]
    }:
        raise RuntimeError(
            "E50E terminal probes did not use the training endpoint"
        )

    combined_path = (
        args.out.resolve().parent / "combined_terminal_route_coverage.json"
    )
    combined = {
        "schema": "e49t_toy_route_coverage_v1",
        "stamp_prefix": args.stamp_prefix,
        "identity_sha256": _sha256(identity_path),
        "endpoint_record_sha256": probes["grpo_terminal"]["identity"][
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
                "evaluations": [_route_eval(probes["grpo_terminal"])]
            },
            "online_canonical_haarnoja": {
                "evaluations": [
                    _route_eval(
                        probes["online_canonical_haarnoja_terminal"]
                    )
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
        expected_updates=1152,
    )
    base_route = probes["base"]
    control_route = probes["grpo_terminal"]
    treatment_route = probes["online_canonical_haarnoja_terminal"]
    result["schema"] = "e50e_exact_oat_math_result_v1"
    result["route_variant"] = args.route_variant
    result["route_probes"] = probes
    result["route_probe_sha256"] = {
        name: _sha256(path) for name, path in paths.items()
    }
    result["continuity_certificate_sha256"] = _sha256(continuity_path)
    result["checks"].update(
        {
            "base_route_probe_passed": (
                base_route["baseline_gate"] is True
            ),
            "treatment_route_coverage_not_below_base": (
                treatment_route["mean_normalized_route_coverage"]
                >= base_route["mean_normalized_route_coverage"]
            ),
            "treatment_route_coverage_not_below_control": (
                treatment_route["mean_normalized_route_coverage"]
                >= control_route["mean_normalized_route_coverage"]
            ),
            "treatment_retains_both_routes_on_at_least_eight": (
                treatment_route["dual_full_coverage_count"] >= 8
            ),
            "treatment_retains_natural_support_on_at_least_eight": (
                treatment_route["natural_support_prompt_count"] >= 8
            ),
        }
    )
    result["complete_evidence"] = (
        result["complete_evidence"]
        and all(probe["status"] == "pass" for probe in probes.values())
    )
    result["success"] = (
        result["complete_evidence"] and all(result["checks"].values())
    )
    _write_json(args.out.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
