#!/usr/bin/env python3
"""Compare legacy and E106 Python admission on prior step-0 responses.

This audit reads response and reference fields only. It deliberately ignores
stored scores and every E104/E106 evaluation artifact.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "var/artifacts/source_snapshots/e106_python_lambda_b853595e3b158046"
SNAPSHOT_SHA256 = "b853595e3b158046f73dee899a2c2a0558d4167e5b6971a36aa50565d9337d02"
sys.path.insert(0, str(SNAPSHOT / "src"))

from oat_drgrpo import math_grader  # noqa: E402


OUT = ROOT / "var/artifacts/e106_prior_python_surface_cross_scale_audit.json"
COHORTS = {
    "qwen05b": (
        "var/data/xdr_qwen25_0p5b_instruct_grpo_compute_matched_"
        "e78_replay_only_python_control_s*/debug_job*/eval_results/"
        "0_multi_answer.json",
        5,
    ),
    "falcon1b": (
        "var/data/xdr_falcon3_1b_instruct_grpo_compute_matched_"
        "e79_falcon_aligned_python_control_s*/debug_job*/eval_results/"
        "0_multi_answer.json",
        5,
    ),
    "qwen3b": (
        "var/data/xdr_qwen25_3b_instruct_grpo_compute_matched_"
        "e80r1_qwen3b_aligned_python_control_s70/debug_job*/eval_results/"
        "0_multi_answer.json",
        1,
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def legacy_python_candidate(response: str, reference: str) -> str | None:
    """Reproduce the pre-E106 Python-only candidate extraction surface."""

    spec = json.loads(reference)
    if spec.get("verifier") != "python_factor_function":
        return None
    if "\\boxed" in response:
        return math_grader.extract_answer(response)
    candidate = response.strip()
    if not candidate:
        return None
    fenced = re.fullmatch(
        r"```(?:python)?\s*(.*?)\s*```",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced is not None:
        candidate = fenced.group(1).strip()
    if (
        len(candidate) > 240
        or "\n" in candidate
        or "\r" in candidate
        or "<" in candidate
        or ">" in candidate
    ):
        return None
    candidate = re.sub(
        r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|program|function)\s*"
        r"(?:is|=|:)\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    return candidate or None


def validate_candidate(candidate: str | None, reference: str) -> str | None:
    if candidate is None:
        return None
    validation = math_grader.validate_python_factor_function_external(
        candidate,
        json.loads(reference),
    )
    return validation.canonical_key if validation is not None else None


def audit_file(path: Path) -> dict[str, Any]:
    records = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    repaired_keys: Counter[str] = Counter()
    recovered_keys: Counter[str] = Counter()
    for record in records:
        reference = str(record["reference"])
        outputs = record["output"]
        if not isinstance(outputs, list):
            outputs = [outputs]
        for raw_response in outputs:
            response = str(raw_response)
            counts["responses"] += 1
            legacy_candidate = legacy_python_candidate(response, reference)
            repaired_candidate = math_grader._extract_modebench_candidate(
                response,
                reference,
            )
            if legacy_candidate is not None:
                counts["legacy_candidates"] += 1
            if repaired_candidate is not None:
                counts["repaired_candidates"] += 1
            if (
                legacy_candidate is not None
                and re.match(r"^\\lambda(?:\s+|\\,\s*)n\s*:", legacy_candidate)
            ):
                counts["latex_lambda_candidates"] += 1
            repaired_key = validate_candidate(repaired_candidate, reference)
            if legacy_candidate == repaired_candidate:
                legacy_key = repaired_key
            else:
                legacy_key = validate_candidate(legacy_candidate, reference)
            if legacy_key is not None:
                counts["legacy_verified"] += 1
            if repaired_key is not None:
                counts["repaired_verified"] += 1
                repaired_keys[repaired_key] += 1
            if legacy_key is None and repaired_key is not None:
                counts["recovered_verified"] += 1
                recovered_keys[repaired_key] += 1
            if legacy_key is not None and repaired_key is None:
                counts["regressed_verified"] += 1
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        **dict(sorted(counts.items())),
        "repaired_distinct_endpoint_keys": len(repaired_keys),
        "repaired_endpoint_multiplicities": sorted(
            repaired_keys.values(), reverse=True
        ),
        "recovered_distinct_endpoint_keys": len(recovered_keys),
        "recovered_endpoint_multiplicities": sorted(
            recovered_keys.values(), reverse=True
        ),
    }


def main() -> int:
    scales: dict[str, Any] = {}
    violations: list[str] = []
    for scale, (pattern, expected_files) in COHORTS.items():
        files = sorted(ROOT.glob(pattern))
        if len(files) != expected_files:
            violations.append(
                f"{scale}: expected {expected_files} prior files, found {len(files)}"
            )
        reports = [audit_file(path) for path in files]
        totals: Counter[str] = Counter()
        for report in reports:
            for field in (
                "responses",
                "legacy_candidates",
                "repaired_candidates",
                "latex_lambda_candidates",
                "legacy_verified",
                "repaired_verified",
                "recovered_verified",
                "regressed_verified",
            ):
                totals[field] += int(report.get(field, 0))
        # Re-evaluate only the compact per-file endpoint counts is insufficient
        # for a cross-file union, so explicitly retain aggregate row counts and
        # the strongest per-file lower bound on endpoint breadth.
        max_distinct = max(
            (int(report["repaired_distinct_endpoint_keys"]) for report in reports),
            default=0,
        )
        if totals["regressed_verified"]:
            violations.append(f"{scale}: repaired parser regressed verified rows")
        scales[scale] = {
            "files": reports,
            "totals": dict(sorted(totals.items())),
            "max_repaired_distinct_endpoint_keys_in_one_seed": max_distinct,
        }
    payload = {
        "schema": "e106_prior_python_surface_cross_scale_audit_v1",
        "passed": not violations,
        "surface_version": math_grader.PYTHON_FACTOR_RESPONSE_SURFACE_VERSION,
        "snapshot_root": str(SNAPSHOT),
        "snapshot_sha256": SNAPSHOT_SHA256,
        "scope": "prior pre-E104 step-0 Python responses only",
        "fields_read": ["output", "reference"],
        "stored_scores_read": False,
        "post_e104_or_e106_update_outcomes_inspected": False,
        "pointmaze": "excluded",
        "scales": scales,
        "violations": violations,
    }
    atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
