#!/usr/bin/env python3
"""Record the E111 Qwen-3B return from unavailable L40 to A6000."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e111_verified_support_discovery_mechanism_gate_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e111_qwen3_return_a6000_amendment_20260818.md"
RECORD = ROOT / "var/artifacts/e111_qwen3_return_a6000_amendment.json"
JOB_IDS = [30674758, 30674759, 30674760, 30674761, 30674762]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def job_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", str(job_id)],
        check=True,
        capture_output=True,
        text=True,
    )
    record = result.stdout.strip()
    if f"JobId={job_id}" not in record:
        raise RuntimeError(f"scheduler record mismatch for {job_id}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "after"))
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    runs = [run for run in ledger["runs"] if run["scale"] == "qwen3b"]
    if [int(run["job_id"]) for run in runs] != JOB_IDS:
        raise RuntimeError("E111 Qwen-3B job set changed")
    records = {str(job_id): job_record(job_id) for job_id in JOB_IDS}
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()
    if args.mode == "before":
        for job_id, record in records.items():
            for needle in (
                "JobState=PENDING",
                "Partition=lowprio",
                "ReqNodeList=node403",
                "TresPerNode=gres/gpu:l40:1",
                "TimeLimit=00:45:00",
            ):
                if needle not in record:
                    raise RuntimeError(f"{job_id} before record lacks {needle}")
        payload = {
            "schema": "e111_qwen3_return_a6000_amendment_v1",
            "recorded_before_at": now,
            "amendment": str(PROTOCOL),
            "amendment_sha256": digest(PROTOCOL),
            "ledger": str(LEDGER),
            "ledger_sha256": digest(LEDGER),
            "job_ids": JOB_IDS,
            "changed_fields": ["ReqNodeList", "Gres"],
            "old_required_nodes": "node403",
            "new_required_nodes": "node[103-104,205-208]",
            "old_gres": "gpu:l40:1",
            "new_gres": "gpu:a6000:1",
            "time_limit": "00:45:00",
            "l40_training_steps": 0,
            "scheduler_only": True,
            "environment_changed": False,
            "e112_changed": False,
            "endpoint_outcomes_inspected": False,
            "pointmaze": "excluded",
            "before_scheduler_records": records,
            "applied": False,
        }
    else:
        payload = json.loads(RECORD.read_text(encoding="utf-8"))
        if payload.get("applied") is not False:
            raise RuntimeError("before record is absent or already finalized")
        for job_id, record in records.items():
            for needle in (
                "Partition=lowprio",
                "ReqNodeList=node[103-104,205-208]",
                "TresPerNode=gres/gpu:a6000:1",
                "TimeLimit=00:45:00",
            ):
                if needle not in record:
                    raise RuntimeError(f"{job_id} after record lacks {needle}")
        payload.update(
            {
                "recorded_after_at": now,
                "after_scheduler_records": records,
                "applied": True,
            }
        )
    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(RECORD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
