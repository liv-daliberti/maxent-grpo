#!/usr/bin/env python3
"""Move the pending zero-update M1 smoke from lowprio to all."""

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


PROTOCOL = "paper/preregistration/e113r1m1s3_qwen_partition_20260819.md"
S2_LEDGER = "var/artifacts/e113r1m1s2_qwen_backfill_amendment.json"
LEDGER = "var/artifacts/e113r1m1s3_qwen_partition_amendment.json"
JOB_ID = 30790590


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def job_record() -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(JOB_ID)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SystemExit(f"cannot inspect M1 job {JOB_ID}")
    return result.stdout.strip()


def main() -> int:
    root = e113.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    m1_path = root / m1.LEDGER
    s2_path = root / S2_LEDGER
    protocol_path = root / PROTOCOL
    output_path = root / LEDGER
    for path in (m1_path, s2_path, protocol_path):
        if not path.is_file():
            raise SystemExit(f"required amendment input is absent: {path}")
    if output_path.exists():
        raise SystemExit(f"refusing duplicate amendment: {output_path}")
    m1_payload = load(m1_path)
    s2_payload = load(s2_path)
    smoke = m1_payload.get("smokes", {}).get("qwen05b", {})
    if int(smoke.get("job_id", -1)) != JOB_ID:
        raise SystemExit("M1 job identity drifted")
    if s2_payload.get("schema") != "e113r1m1s2_qwen_backfill_amendment_v1":
        raise SystemExit("S2 amendment schema drifted")
    if Path(str(smoke.get("run_dir", ""))).exists():
        raise SystemExit("refusing partition amendment after M1 output appeared")

    before = job_record()
    required_before = (
        "JobState=PENDING",
        "RunTime=00:00:00",
        "Partition=lowprio",
        "Account=mltheory",
        "TimeLimit=02:00:00",
        "MinMemoryNode=64G",
        f"ReqNodeList={m1.NODELIST}",
        "TresPerNode=gres/gpu:a6000:1",
    )
    missing = [value for value in required_before if value not in before]
    if missing:
        raise SystemExit(f"M1 pre-amendment scheduler record lacks {missing}")
    if not args.apply:
        print(f"[e113r1m1s3] ready job={JOB_ID} Partition=lowprio->all")
        return 0

    subprocess.run(
        ["scontrol", "update", f"JobId={JOB_ID}", "Partition=all"],
        check=True,
    )
    after = job_record()
    required_after = (
        "JobState=PENDING",
        "RunTime=00:00:00",
        "Partition=all",
        "Account=mltheory",
        "TimeLimit=02:00:00",
        "MinMemoryNode=64G",
        f"ReqNodeList={m1.NODELIST}",
        "TresPerNode=gres/gpu:a6000:1",
    )
    missing = [value for value in required_after if value not in after]
    if missing:
        raise RuntimeError(f"M1 amended scheduler record lacks {missing}")
    payload = {
        "schema": "e113r1m1s3_qwen_partition_amendment_v1",
        "installed": True,
        "outcomes_inspected": False,
        "job_id": JOB_ID,
        "same_job_id": True,
        "same_scientific_cells": True,
        "scientific_cells": 0,
        "change": {"Partition": {"before": "lowprio", "after": "all"}},
        "unchanged_account": "mltheory",
        "unchanged_time_limit": "02:00:00",
        "unchanged_host_memory": "64G",
        "m1_ledger": str(m1_path),
        "m1_ledger_sha256": e113.e78.digest(m1_path),
        "s2_amendment": str(s2_path),
        "s2_amendment_sha256": e113.e78.digest(s2_path),
        "protocol": str(protocol_path),
        "protocol_sha256": e113.e78.digest(protocol_path),
        "amender_sha256": e113.e78.digest(Path(__file__)),
        "scheduler_record_before": before,
        "scheduler_record_after": after,
    }
    e113.e78.atomic_json(output_path, payload)
    print(f"[e113r1m1s3] moved pending job {JOB_ID} to Partition=all")
    print(f"[e113r1m1s3] ledger {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
