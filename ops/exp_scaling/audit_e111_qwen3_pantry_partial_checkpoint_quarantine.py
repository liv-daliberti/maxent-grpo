#!/usr/bin/env python3
"""Validate the outcome-blind E111 Pantry partial-checkpoint recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "var/artifacts/e111_qwen3_pantry_partial_checkpoint_quarantine.json"
CORRUPTION = "PytorchStreamReader failed reading zip archive: failed finding central directory"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zip_valid(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False
    return True


def validate() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not RECORD.is_file():
        return {"passed": False}, ["Pantry quarantine record is absent"]
    try:
        payload = json.loads(RECORD.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"passed": False}, [f"Pantry quarantine record is invalid: {exc}"]
    expected = {
        "schema": "e111_qwen3_pantry_partial_checkpoint_quarantine_v1",
        "job_id": 30674762,
        "latest_marker": "step_00002",
        "checkpoint_storage_only": True,
        "quarantined_bytes_recoverable": True,
        "bytes_deleted": False,
        "job_signaled_for_quarantine": False,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "stdout_reinitialized_on_requeue": True,
        "verified_after_requeue": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            violations.append(f"Pantry quarantine record has invalid {key}")
    protocol = Path(str(payload.get("protocol", "")))
    if not protocol.is_file() or digest(protocol) != payload.get("protocol_sha256"):
        violations.append("Pantry quarantine protocol digest drifted")
    quarantined = payload.get("quarantined_checkpoint_evidence", {})
    for name, should_be_valid in (
        ("mp_rank_00_model_states.pt", True),
        ("bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt", False),
    ):
        row = quarantined.get(name, {}) if isinstance(quarantined, dict) else {}
        path = Path(str(row.get("path", "")))
        if not path.is_file() or path.stat().st_size != row.get("size"):
            violations.append(f"quarantined Pantry archive drifted: {name}")
        elif zip_valid(path) is not should_be_valid:
            violations.append(f"quarantined Pantry ZIP status drifted: {name}")
    counts = payload.get("pre_recovery_failure_counts", {})
    if counts != {"tracebacks": 1, "partial_checkpoint": 1}:
        violations.append("Pantry archived pre-recovery failure evidence drifted")
    stdout = ROOT / "var/artifacts/logs/e111-q3-pantry-30674762.out"
    text = stdout.read_text(encoding="utf-8", errors="replace") if stdout.is_file() else ""
    observed = {
        "tracebacks": text.count("Traceback (most recent call last)"),
        "partial_checkpoint": text.count(CORRUPTION),
    }
    if payload.get("post_recovery_failure_counts") != observed:
        violations.append("Pantry post-requeue failure count changed")
    post = payload.get("post_recovery_valid_checkpoints")
    if not isinstance(post, list) or not post:
        violations.append("Pantry lacks valid post-recovery checkpoint evidence")
    report = {
        "record": str(RECORD),
        "record_sha256": digest(RECORD),
        "job_id": 30674762,
        "checkpoint_storage_only": payload.get("checkpoint_storage_only"),
        "quarantined_bytes_recoverable": payload.get("quarantined_bytes_recoverable"),
        "optimizer_update_changed": payload.get("optimizer_update_changed"),
        "treatment_changed": payload.get("treatment_changed"),
        "outcomes_inspected": payload.get("outcomes_inspected"),
        "pointmaze": payload.get("pointmaze"),
        "verified_after_requeue": payload.get("verified_after_requeue"),
        "allowed_traceback_occurrences": int(observed["tracebacks"]),
        "partial_checkpoint_occurrences": int(observed["partial_checkpoint"]),
        "violations": violations,
        "passed": not violations,
    }
    return report, violations


def main() -> int:
    report, violations = validate()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
