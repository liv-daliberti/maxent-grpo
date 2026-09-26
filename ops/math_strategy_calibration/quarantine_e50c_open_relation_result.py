#!/usr/bin/env python3
"""Write the fail-closed terminal record for quarantined E50C."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
JOB_ID = "30124473"
OUTPUT = ROOT / "var/artifacts/e50c_72b_teacher_route_calibration_v1"
RESULT = OUTPUT / "result.json"
LOG = ROOT / f"var/artifacts/e50c-teacher-route-{JOB_ID}.out"
ERROR_LOG = ROOT / f"var/artifacts/e50c-teacher-route-{JOB_ID}.err"
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50c_72b_teacher_route_calibration_20260726.md"
)
QUARANTINE_PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50c_open_relation_quarantine_20260726.md"
)
E50C_SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50c_72b_teacher_route_calibration.py"
)
SCRIPT = pathlib.Path(__file__).resolve()
ENDPOINT = (
    ROOT / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
)
MANIFEST = (
    ROOT / "var/artifacts/e47_math_strategy_calibration_v1/manifest.json"
)
PROBLEMS = (
    ROOT / "var/artifacts/e47_math_strategy_calibration_v1/problems.jsonl"
)
FAILURES = (
    ROOT
    / "var/artifacts/e50f2_frozen_pairwise_relation_calibration_v1/"
    "result.json",
    ROOT
    / "var/artifacts/e50f4_pairwise_finite_family_calibration_v1/"
    "result.json",
)
PRIVATE_RESPONSES = OUTPUT / "private/teacher_responses.jsonl"


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


def main() -> None:
    if RESULT.exists():
        raise RuntimeError(f"E50C terminal result already exists: {RESULT}")
    records = [
        json.loads(line)
        for line in LOG.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError("E50C quarantine requires logged teacher progress")
    if any(
        row != {
            "phase": "teacher",
            "problem_order": index,
            "sample_count": 16,
        }
        for index, row in enumerate(records)
    ):
        raise RuntimeError("E50C teacher progress log is not sequential")
    error_text = ERROR_LOG.read_text(encoding="utf-8")
    if error_text and not (
        f"JOB {JOB_ID}" in error_text
        and "CANCELLED" in error_text
        and "SIGNAL Terminated" in error_text
    ):
        raise RuntimeError("E50C quarantine found an unexpected error log")

    failure_evidence = {}
    for path in FAILURES:
        result = json.loads(path.read_text(encoding="utf-8"))
        checks = result.get("checks") or {}
        zero_false_new = (
            checks.get("zero_false_new")
            if "zero_false_new" in checks
            else checks.get("zero_false_new_each_pass")
        )
        if result.get("pass") is not False or zero_false_new is not False:
            raise RuntimeError(f"false-new quarantine evidence drifted: {path}")
        failure_evidence[str(result["schema"])] = _sha256(path)

    private_count = 0
    private_sha256 = None
    if PRIVATE_RESPONSES.is_file():
        private_count = sum(
            bool(line.strip())
            for line in PRIVATE_RESPONSES.read_text(
                encoding="utf-8"
            ).splitlines()
        )
        private_sha256 = _sha256(PRIVATE_RESPONSES)
    payload = {
        "schema": "e50c_72b_teacher_route_calibration_v1",
        "pass": False,
        "failure": "open_relation_judge_quarantined_before_route_discovery",
        "claim_scope": (
            "terminal fail-closed cancellation record; no route or "
            "training claim"
        ),
        "job_id": JOB_ID,
        "teacher_completed_problem_count": len(records),
        "teacher_expected_problem_count": 50,
        "teacher_samples_per_completed_problem": 16,
        "teacher_logged_sample_count": 16 * len(records),
        "teacher_private_response_count": private_count,
        "teacher_private_responses_sha256": private_sha256,
        "candidate_count": 0,
        "cluster_eligible_count": 0,
        "double_audited_menu_count": 0,
        "bidirectionally_executable_count": 0,
        "selected_source_indices": [],
        "wrong_route_success_count": 0,
        "identity": {
            "protocol_sha256": _sha256(PROTOCOL),
            "quarantine_protocol_sha256": _sha256(QUARANTINE_PROTOCOL),
            "e50c_script_sha256": _sha256(E50C_SCRIPT),
            "quarantine_script_sha256": _sha256(SCRIPT),
            "progress_log_sha256": _sha256(LOG),
            "error_log_sha256": _sha256(ERROR_LOG),
            "endpoint_record_sha256": _sha256(ENDPOINT),
            "e47_manifest_sha256": _sha256(MANIFEST),
            "e47_problems_sha256": _sha256(PROBLEMS),
            "false_new_failure_sha256": failure_evidence,
        },
    }
    _write_json(RESULT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
