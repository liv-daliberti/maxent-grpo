#!/usr/bin/env python3
"""Record E111 Qwen-3B reduction from 8-step to 2-step durability."""

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
LEDGER = ROOT / "var/artifacts/e111_verified_support_discovery_mechanism_gate_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e111_qwen3_two_step_durability_amendment_20260818.md"
PRIOR_RECORD = ROOT / "var/artifacts/e111_qwen3_runtime_ops_durability_amendment.json"
RECORD = ROOT / "var/artifacts/e111_qwen3_two_step_durability_amendment.json"
RUNTIME_TRAIN = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops/train.sh"
JOB_IDS = [30674758, 30674759, 30674760, 30674761, 30674762]
BEGIN = "# BEGIN E111_QWEN3_RUNTIME_OPS_DURABILITY_AMENDMENT"
END = "# END E111_QWEN3_RUNTIME_OPS_DURABILITY_AMENDMENT"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def block(value: str) -> str:
    if value.count(BEGIN) != 1 or value.count(END) != 1:
        raise RuntimeError("runtime train script lacks one durability block")
    start = value.index(BEGIN)
    finish = value.index(END, start) + len(END)
    return value[start:finish]


def scheduler_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", str(job_id)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def restarts(record: str) -> int:
    match = re.search(r"(?:^| )Restarts=(\d+)(?: |$)", record)
    if match is None:
        raise RuntimeError("scheduler record lacks restart count")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "after"))
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_RECORD.read_text(encoding="utf-8"))
    records = {str(job_id): scheduler_record(job_id) for job_id in JOB_IDS}
    value = RUNTIME_TRAIN.read_text(encoding="utf-8")
    current_block = block(value)
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()
    if args.mode == "before":
        for needle in (
            "export OAT_ZERO_SAVE_STEPS=8",
            "export OAT_ZERO_SAVE_FROM=8",
            "export OAT_ZERO_RESUME_STEPS=8",
            "checkpoint_interval=8",
        ):
            if needle not in current_block:
                raise RuntimeError(f"prior runtime block lacks {needle}")
        if digest(RUNTIME_TRAIN) != prior["runtime_train_after_sha256"]:
            raise RuntimeError("prior runtime train digest chain broke")
        if digest_bytes(current_block.encode("utf-8")) != prior["amendment_block_sha256"]:
            raise RuntimeError("prior runtime block digest chain broke")
        payload = {
            "schema": "e111_qwen3_two_step_durability_amendment_v1",
            "recorded_before_at": now,
            "amendment": str(PROTOCOL),
            "amendment_sha256": digest(PROTOCOL),
            "ledger": str(LEDGER),
            "ledger_sha256": digest(LEDGER),
            "prior_record": str(PRIOR_RECORD),
            "prior_record_sha256": digest(PRIOR_RECORD),
            "runtime_train": str(RUNTIME_TRAIN),
            "runtime_train_before_sha256": digest(RUNTIME_TRAIN),
            "prior_block_sha256": digest_bytes(current_block.encode("utf-8")),
            "job_ids": JOB_IDS,
            "old_interval": 8,
            "new_interval": 2,
            "storage_only": True,
            "treatment_changed": False,
            "jobs_signaled": False,
            "jobs_reset": False,
            "endpoint_outcomes_inspected": False,
            "pointmaze": "excluded",
            "before_scheduler_records": records,
            "restart_counts_before": {key: restarts(record) for key, record in records.items()},
            "installed": False,
        }
    else:
        payload = json.loads(RECORD.read_text(encoding="utf-8"))
        if payload.get("installed") is not False:
            raise RuntimeError("before record is absent or finalized")
        for needle in (
            "export OAT_ZERO_SAVE_STEPS=2",
            "export OAT_ZERO_SAVE_FROM=2",
            "export OAT_ZERO_RESUME_STEPS=2",
            "checkpoint_interval=2",
        ):
            if needle not in current_block:
                raise RuntimeError(f"new runtime block lacks {needle}")
        payload.update(
            {
                "recorded_after_at": now,
                "runtime_train_after_sha256": digest(RUNTIME_TRAIN),
                "amendment_block_sha256": digest_bytes(current_block.encode("utf-8")),
                "after_scheduler_records": records,
                "restart_counts_after": {key: restarts(record) for key, record in records.items()},
                "installed": True,
            }
        )
    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(RECORD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
