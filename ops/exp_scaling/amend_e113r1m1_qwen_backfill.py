#!/usr/bin/env python3
"""Shorten the pending M1 wall-time ceiling to admit scheduler backfill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e113_dapo_direct_baseline as e113  # noqa: E402
import launch_e113r1m1_qwen_memory_recovery as m1  # noqa: E402


PROTOCOL = "paper/preregistration/e113r1m1s2_qwen_backfill_20260819.md"
LEDGER = "var/artifacts/e113r1m1s2_qwen_backfill_amendment.json"
JOB_ID = 30790590


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def job_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SystemExit(f"cannot inspect M1 job {job_id}")
    return result.stdout.strip()


def validate_before(root: Path, ledger: dict[str, Any], record: str) -> None:
    smoke = ledger.get("smokes", {}).get("qwen05b", {})
    required = (
        "JobState=PENDING",
        "RunTime=00:00:00",
        "TimeLimit=08:00:00",
        "MinMemoryNode=64G",
        f"ReqNodeList={m1.NODELIST}",
        "TresPerNode=gres/gpu:a6000:1",
        "OAT_ZERO_MAX_TRAIN=32",
        "OAT_ZERO_MAX_QUERIES=5120",
        "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=16",
        "OAT_ZERO_ADAM_OFFLOAD=0",
        "OAT_ZERO_ACTIVATION_OFFLOADING=0",
    )
    if ledger.get("schema") != "e113r1m1_qwen_memory_recovery_jobs_v1":
        raise SystemExit("M1 ledger schema drifted")
    if ledger.get("released") is not True or int(smoke.get("job_id", -1)) != JOB_ID:
        raise SystemExit("M1 ledger does not bind the expected released job")
    run_dir = Path(str(smoke.get("run_dir", "")))
    if run_dir.exists():
        raise SystemExit(f"refusing amendment after M1 output appeared: {run_dir}")
    missing = [value for value in required if value not in record]
    if missing:
        raise SystemExit(f"M1 pre-amendment scheduler record lacks {missing}")


def main() -> int:
    root = e113.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source_path = root / m1.LEDGER
    protocol_path = root / PROTOCOL
    amendment_path = root / LEDGER
    for path in (source_path, protocol_path):
        if not path.is_file():
            raise SystemExit(f"required amendment input is absent: {path}")
    if amendment_path.exists():
        raise SystemExit(f"refusing duplicate amendment: {amendment_path}")

    source = load(source_path)
    before = job_record(JOB_ID)
    validate_before(root, source, before)
    if not args.apply:
        print(f"[e113r1m1s2] ready job={JOB_ID} TimeLimit=08:00:00->02:00:00")
        return 0

    subprocess.run(
        ["scontrol", "update", f"JobId={JOB_ID}", "TimeLimit=02:00:00"],
        check=True,
    )
    after = job_record(JOB_ID)
    required_after = (
        "JobState=PENDING",
        "RunTime=00:00:00",
        "TimeLimit=02:00:00",
        "MinMemoryNode=64G",
        f"ReqNodeList={m1.NODELIST}",
        "TresPerNode=gres/gpu:a6000:1",
    )
    missing = [value for value in required_after if value not in after]
    if missing:
        raise RuntimeError(f"M1 amended scheduler record lacks {missing}")
    payload = {
        "schema": "e113r1m1s2_qwen_backfill_amendment_v1",
        "installed": True,
        "outcomes_inspected": False,
        "job_id": JOB_ID,
        "same_job_id": True,
        "same_scientific_cells": True,
        "scientific_cells": 0,
        "change": {"TimeLimit": {"before": "08:00:00", "after": "02:00:00"}},
        "unchanged_host_memory": "64G",
        "m1_ledger": str(source_path),
        "m1_ledger_sha256": e113.e78.digest(source_path),
        "protocol": str(protocol_path),
        "protocol_sha256": e113.e78.digest(protocol_path),
        "amender_sha256": e113.e78.digest(Path(__file__)),
        "scheduler_record_before": before,
        "scheduler_record_after": after,
    }
    e113.e78.atomic_json(amendment_path, payload)
    print(f"[e113r1m1s2] amended pending job {JOB_ID} to TimeLimit=02:00:00")
    print(f"[e113r1m1s2] ledger {amendment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
