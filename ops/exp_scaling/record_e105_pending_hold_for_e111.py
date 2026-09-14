#!/usr/bin/env python3
"""Capture and verify reversible holds on pending superseded E105 jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e105_group_centered_semantic_repair_full_three_scale_jobs.json"
SUPERSESSION = ROOT / "paper/preregistration/e105_v6_superseded_by_e111_20260818.md"
PROTOCOL = ROOT / "paper/preregistration/e105_pending_hold_for_e111_20260818.md"
RECORD = ROOT / "var/artifacts/e105_pending_hold_for_e111.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduler_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Completed jobs may age out of the live controller while remaining in
        # sacct. They are inactive and therefore never candidates for a hold.
        return ""
    return result.stdout.strip()


def field(record: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", record)
    return match.group(1) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "after"))
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    job_ids = [int(run["job_id"]) for run in ledger["runs"]]
    if len(job_ids) != 75 or len(set(job_ids)) != 75:
        raise RuntimeError("E105 ledger must contain 75 unique jobs")
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()

    if args.mode == "before":
        records = {str(job_id): scheduler_record(job_id) for job_id in job_ids}
        pending = [job_id for job_id in job_ids if field(records[str(job_id)], "JobState") == "PENDING"]
        running = [job_id for job_id in job_ids if field(records[str(job_id)], "JobState") == "RUNNING"]
        if not pending or set(pending) & set(running):
            raise RuntimeError("invalid E105 pending/running partition")
        payload = {
            "schema": "e105_pending_hold_for_e111_v1",
            "captured_before_at": now,
            "protocol": str(PROTOCOL),
            "protocol_sha256": digest(PROTOCOL),
            "ledger": str(LEDGER),
            "ledger_sha256": digest(LEDGER),
            "supersession": str(SUPERSESSION),
            "supersession_sha256": digest(SUPERSESSION),
            "pending_job_ids": pending,
            "running_job_ids": running,
            "held_job_ids": pending,
            "before_scheduler_records": {str(job_id): records[str(job_id)] for job_id in pending},
            "reversible": True,
            "jobs_signaled": False,
            "jobs_canceled": False,
            "endpoint_outcomes_inspected": False,
            "pointmaze": "excluded",
            "applied": False,
        }
    else:
        payload = json.loads(RECORD.read_text(encoding="utf-8"))
        held = [int(job_id) for job_id in payload["held_job_ids"]]
        records = {str(job_id): scheduler_record(job_id) for job_id in held}
        for job_id, record in records.items():
            state = field(record, "JobState")
            reason = field(record, "Reason")
            if state != "PENDING" or reason not in {"JobHeldUser", "JobHeldAdmin"}:
                raise RuntimeError(f"E105 {job_id} was not held: {state}/{reason}")
        payload.update(
            {
                "captured_after_at": now,
                "after_scheduler_records": records,
                "applied": True,
            }
        )
    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(RECORD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
