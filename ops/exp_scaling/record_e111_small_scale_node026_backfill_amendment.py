#!/usr/bin/env python3
"""Record E111 small-scale node026/45-minute scheduler amendment."""

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
PROTOCOL = ROOT / "paper/preregistration/e111_small_scale_node026_backfill_amendment_20260818.md"
RECORD = ROOT / "var/artifacts/e111_small_scale_node026_backfill_amendment.json"
JOB_IDS = [30674729, 30674733, 30674754]
OLD_NODES = "node[020-022,025,103-104,202-208,403]"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def job_record(job_id: int) -> str:
    record = command("scontrol", "show", "job", "-o", str(job_id))
    if f"JobId={job_id}" not in record:
        raise RuntimeError(f"scheduler record mismatch for {job_id}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "after"))
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    all_ids = {int(run["job_id"]) for run in ledger["runs"]}
    if not set(JOB_IDS) <= all_ids:
        raise RuntimeError("small-scale job set is not in E111 ledger")
    records = {str(job_id): job_record(job_id) for job_id in JOB_IDS}
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()
    if args.mode == "before":
        for job_id, record in records.items():
            for needle in (
                "JobState=PENDING",
                "Partition=lowprio",
                f"ReqNodeList={OLD_NODES}",
                "TresPerNode=gres/gpu:1",
                "MinMemoryNode=64G",
                "TimeLimit=08:00:00",
            ):
                if needle not in record:
                    raise RuntimeError(f"{job_id} before record lacks {needle}")
        payload = {
            "schema": "e111_small_scale_node026_backfill_amendment_v1",
            "recorded_before_at": now,
            "amendment": str(PROTOCOL),
            "amendment_sha256": digest(PROTOCOL),
            "ledger": str(LEDGER),
            "ledger_sha256": digest(LEDGER),
            "job_ids": JOB_IDS,
            "changed_fields": ["ReqNodeList", "TimeLimit"],
            "old_required_nodes": OLD_NODES,
            "new_required_nodes": "node026",
            "old_time_limit": "08:00:00",
            "new_time_limit": "00:45:00",
            "gres": "gpu:1",
            "scheduler_only": True,
            "environment_changed": False,
            "endpoint_outcomes_inspected": False,
            "pointmaze": "excluded",
            "node026_before": command("scontrol", "show", "node", "node026"),
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
                "ReqNodeList=node026",
                "TresPerNode=gres/gpu:1",
                "MinMemoryNode=64G",
                "TimeLimit=00:45:00",
            ):
                if needle not in record:
                    raise RuntimeError(f"{job_id} after record lacks {needle}")
        payload.update(
            {
                "recorded_after_at": now,
                "node026_after": command("scontrol", "show", "node", "node026"),
                "after_scheduler_records": records,
                "applied": True,
            }
        )
    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(RECORD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
