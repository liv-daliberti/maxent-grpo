#!/usr/bin/env python3
"""Validate the outcome-blind E111 proposal-retention resume recovery."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import install_e111_proposal_retention_resume_recovery as install  # noqa: E402


RECORD = install.RECORD


def validate() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not RECORD.is_file():
        return {"record": str(RECORD), "passed": False}, [
            "E111 proposal-retention resume recovery record is absent"
        ]
    try:
        payload = json.loads(RECORD.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"record": str(RECORD), "passed": False}, [
            f"E111 proposal-retention resume recovery record is invalid: {exc}"
        ]
    expected = {
        "schema": "e111_proposal_retention_resume_recovery_v1",
        "runtime_snapshot": str(install.RUNTIME),
        "files": list(install.FILES),
        "before_sha256": install.BEFORE_SHA256,
        "after_sha256": install.AFTER_SHA256,
        "root_after_sha256": install.AFTER_SHA256,
        "test_returncode": 0,
        "failure_marker": install.FAILURE_MARKER,
        "runtime_source_changed": True,
        "checkpoint_deserialization_only": True,
        "checkpoint_schema_changed": False,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "environment_changed": False,
        "jobs_signaled": False,
        "jobs_reset": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "installed": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            violations.append(f"E111 resume recovery has invalid {key}")
    for key, path in (
        ("ledger_sha256", install.LEDGER),
        ("protocol_sha256", install.PROTOCOL),
        ("runtime_snapshot_identity_sha256", install.RUNTIME / "SNAPSHOT_IDENTITY.json"),
        ("test_file_sha256", install.ROOT / "tests/test_online_canonical_bank.py"),
    ):
        if not path.is_file() or payload.get(key) != install.digest(path):
            violations.append(f"E111 resume recovery {key} mismatch")
    if Path(str(payload.get("ledger", ""))).resolve() != install.LEDGER.resolve():
        violations.append("E111 resume recovery names a different ledger")
    if Path(str(payload.get("protocol", ""))).resolve() != install.PROTOCOL.resolve():
        violations.append("E111 resume recovery names a different protocol")
    if install.EXPECTED_TEST_SUMMARY not in str(payload.get("test_stdout", "")):
        violations.append("E111 resume recovery lacks passing regression summary")

    ledger: dict[str, Any] = {}
    if install.LEDGER.is_file():
        try:
            ledger = json.loads(install.LEDGER.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            violations.append("E111 resume recovery ledger is invalid JSON")
    expected_ids = [int(run["job_id"]) for run in ledger.get("runs", [])]
    if payload.get("all_e111_job_ids") != expected_ids or len(expected_ids) != 15:
        violations.append("E111 resume recovery job set mismatch")
    failures = payload.get("trigger_failures")
    observed_ids = (
        {int(row.get("job_id")) for row in failures if isinstance(row, dict)}
        if isinstance(failures, list)
        else set()
    )
    if not install.EXPECTED_FAILURE_JOBS.issubset(observed_ids):
        violations.append("E111 resume recovery trigger job evidence mismatch")

    live_runtime = {
        relative: install.digest(install.RUNTIME / relative)
        for relative in install.FILES
        if (install.RUNTIME / relative).is_file()
    }
    live_root = {
        relative: install.digest(install.ROOT / relative)
        for relative in install.FILES
        if (install.ROOT / relative).is_file()
    }
    if live_runtime != install.AFTER_SHA256:
        violations.append("E111 resume recovery runtime source digest mismatch")
    if live_root != install.AFTER_SHA256:
        violations.append("E111 resume recovery root source digest mismatch")
    tracker_text = (install.ROOT / install.FILES[0]).read_text(encoding="utf-8")
    bank_text = (install.ROOT / install.FILES[1]).read_text(encoding="utf-8")
    test_text = (install.ROOT / "tests/test_online_canonical_bank.py").read_text(encoding="utf-8")
    for label, needle, value in (
        ("tracker", "def converted_pairs", tracker_text),
        ("bank", "proposal retention state has inconsistent", bank_text),
        ("bank", "observed_on_policy", bank_text),
        ("test", "test_admission_retention_resume_accepts_on_policy_converted_proposal", test_text),
    ):
        if needle not in value:
            violations.append(f"E111 resume recovery {label} lacks {needle}")

    report = {
        "record": str(RECORD),
        "record_sha256": install.digest(RECORD),
        "runtime_source_changed": True,
        "checkpoint_deserialization_only": True,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "environment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "trigger_job_ids": sorted(observed_ids),
        "trigger_occurrences": {
            str(int(row["job_id"])): int(row["occurrences"])
            for row in failures
            if isinstance(row, dict)
        } if isinstance(failures, list) else {},
        "passed": not violations,
    }
    return report, violations


def main() -> int:
    report, violations = validate()
    print(json.dumps(report | {"violations": violations}, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
