#!/usr/bin/env python3
"""Certify a replacement Qwen72 endpoint against both frozen E49T controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
CANONICALIZER = ROOT / "src/oat_drgrpo/math_strategy_canonicalizer.py"
ROUTE_IDENTITY = (
    ROOT
    / "var/artifacts/e49t_route_confusion_calibration_v1/"
    "frozen_identity.json"
)
DECLARATION_IDENTITY = (
    ROOT
    / "var/artifacts/e49t_declaration_mismatch_calibration_v1/"
    "frozen_identity.json"
)
ORIGINAL_ROUTE_RESULT = (
    ROOT / "var/artifacts/e49t_route_confusion_calibration_v1/result.json"
)
ORIGINAL_DECLARATION_RESULT = (
    ROOT
    / "var/artifacts/e49t_declaration_mismatch_calibration_v1/result.json"
)
ROUTE_SCORER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "score_e49t_route_confusion_calibration.py"
)
DECLARATION_SCORER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "score_e49t_declaration_mismatch_calibration.py"
)
SCHEMA = "e50e_qwen72_continuity_certificate_v1"


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


def _load(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _passing_original(path: pathlib.Path, schema: str) -> None:
    result = _load(path)
    if (
        result.get("schema") != schema
        or result.get("pass") is not True
        or not all((result.get("checks") or {}).values())
    ):
        raise RuntimeError(f"original frozen calibration is not passing: {path}")


def build_certificate(
    *,
    endpoint_path: pathlib.Path,
    route_path: pathlib.Path,
    declaration_path: pathlib.Path,
    expected_node: str,
    expected_port: int,
) -> dict[str, Any]:
    _passing_original(
        ORIGINAL_ROUTE_RESULT,
        "e49t_route_confusion_calibration_result_v1",
    )
    _passing_original(
        ORIGINAL_DECLARATION_RESULT,
        "e49t_declaration_mismatch_result_v1",
    )
    endpoint = _load(endpoint_path)
    expected_endpoint = {
        "model": "qwen2.5-72b",
        "node": expected_node,
        "port": expected_port,
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "enforce_eager": True,
        "structured_output_backend": "guidance",
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(endpoint.get(key) != value for key, value in expected_endpoint.items()):
        raise RuntimeError("E50E continuity endpoint configuration mismatch")
    if not str(endpoint.get("job_id") or ""):
        raise RuntimeError("E50E continuity endpoint has no Slurm job identity")

    endpoint_sha256 = _sha256(endpoint_path)
    canonicalizer_sha256 = _sha256(CANONICALIZER)
    route = _load(route_path)
    route_counts = route.get("counts") or {}
    route_ids = route.get("identities") or {}
    route_checks = {
        "schema_and_pass": (
            route.get("schema")
            == "e49t_route_confusion_calibration_result_v1"
            and route.get("pass") is True
            and all((route.get("checks") or {}).values())
        ),
        "zero_false_new": route_counts.get("duplicate_false_new_count") == 0,
        "zero_misassignment": route_counts.get("misassigned_positive") == 0,
        "zero_answer_only_acceptance": route_counts.get("accepted_negative") == 0,
        "zero_open_set": route_counts.get("open_set_count") == 0,
        "positive_acceptance": (
            float(route_counts.get("positive_acceptance_fraction", 0.0))
            >= 0.75
        ),
        "dual_support": int(route_counts.get("dual_route_prompt_count", 0)) >= 10,
        "endpoint_bound": (
            route_ids.get("endpoint_record_sha256") == endpoint_sha256
        ),
        "cohort_bound": (
            route_ids.get("cohort_identity_sha256") == _sha256(ROUTE_IDENTITY)
        ),
        "canonicalizer_bound": (
            route_ids.get("canonicalizer_sha256") == canonicalizer_sha256
        ),
        "scorer_bound": (
            route_ids.get("scoring_script_sha256") == _sha256(ROUTE_SCORER)
        ),
    }

    declaration = _load(declaration_path)
    declaration_counts = declaration.get("counts") or {}
    declaration_ids = declaration.get("identities") or {}
    declaration_checks = {
        "schema_and_pass": (
            declaration.get("schema")
            == "e49t_declaration_mismatch_result_v1"
            and declaration.get("pass") is True
            and all((declaration.get("checks") or {}).values())
        ),
        "zero_mismatch_acceptance": (
            declaration_counts.get("mismatch_accepted") == 0
        ),
        "zero_wrong_assignment": declaration_counts.get("wrong_matched") == 0,
        "zero_open_set": declaration_counts.get("open_set") == 0,
        "matched_acceptance": (
            int(declaration_counts.get("correct_matched", 0)) >= 18
        ),
        "dual_support": (
            int(declaration_counts.get("dual_route_prompt_count", 0)) >= 10
        ),
        "endpoint_bound": (
            declaration_ids.get("endpoint_record_sha256") == endpoint_sha256
        ),
        "cohort_bound": (
            declaration_ids.get("cohort_identity_sha256")
            == _sha256(DECLARATION_IDENTITY)
        ),
        "canonicalizer_bound": (
            declaration_ids.get("canonicalizer_sha256")
            == canonicalizer_sha256
        ),
        "scorer_bound": (
            declaration_ids.get("scorer_sha256")
            == _sha256(DECLARATION_SCORER)
        ),
    }
    checks = {
        "route_confusion_control_passed": all(route_checks.values()),
        "declaration_mismatch_control_passed": all(
            declaration_checks.values()
        ),
        "same_endpoint_for_both_controls": (
            route_ids.get("endpoint_record_sha256")
            == declaration_ids.get("endpoint_record_sha256")
            == endpoint_sha256
        ),
        "same_frozen_canonicalizer": (
            route_ids.get("canonicalizer_sha256")
            == declaration_ids.get("canonicalizer_sha256")
            == canonicalizer_sha256
        ),
    }
    return {
        "schema": SCHEMA,
        "pass": all(checks.values()),
        "checks": checks,
        "route_checks": route_checks,
        "declaration_checks": declaration_checks,
        "endpoint": expected_endpoint,
        "endpoint_job_id": str(endpoint["job_id"]),
        "identities": {
            "endpoint_record_sha256": endpoint_sha256,
            "route_result_sha256": _sha256(route_path),
            "declaration_result_sha256": _sha256(declaration_path),
            "route_identity_sha256": _sha256(ROUTE_IDENTITY),
            "declaration_identity_sha256": _sha256(DECLARATION_IDENTITY),
            "canonicalizer_sha256": canonicalizer_sha256,
            "route_scorer_sha256": _sha256(ROUTE_SCORER),
            "declaration_scorer_sha256": _sha256(DECLARATION_SCORER),
            "original_route_result_sha256": _sha256(ORIGINAL_ROUTE_RESULT),
            "original_declaration_result_sha256": _sha256(
                ORIGINAL_DECLARATION_RESULT
            ),
            "certifier_sha256": _sha256(SCRIPT),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-record", type=pathlib.Path, required=True)
    parser.add_argument("--route-result", type=pathlib.Path, required=True)
    parser.add_argument(
        "--declaration-result", type=pathlib.Path, required=True
    )
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--expected-port", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    expected = build_certificate(
        endpoint_path=args.endpoint_record.resolve(),
        route_path=args.route_result.resolve(),
        declaration_path=args.declaration_result.resolve(),
        expected_node=args.expected_node,
        expected_port=args.expected_port,
    )
    output = args.output.resolve()
    if args.audit_only:
        if _load(output) != expected:
            raise RuntimeError("E50E continuity certificate identity drifted")
    else:
        if output.exists():
            raise RuntimeError(f"fresh E50E continuity certificate required: {output}")
        if expected["pass"] is not True:
            raise RuntimeError("E50E continuity controls did not pass")
        _write_json(output, expected)
    print(json.dumps(expected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
