#!/usr/bin/env python3
"""Record the prospective and installed E111 Qwen-3B durability amendment."""

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
PROTOCOL = ROOT / "paper/preregistration/e111_qwen3_restart_durability_amendment_20260818.md"
WRAPPER = ROOT / "ops/slurm/train_node302.slurm"
RECORD = ROOT / "var/artifacts/e111_qwen3_restart_durability_amendment.json"
JOB_IDS = [30674758, 30674759, 30674760, 30674761, 30674762]
SOURCE_ROOT = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/src"
OPS_ROOT = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops"
VARIANT = "verified_replay_semantic_maxent_verified_support_discovery"
BEGIN = "# BEGIN E111_QWEN3_RESTART_DURABILITY_AMENDMENT"
END = "# END E111_QWEN3_RESTART_DURABILITY_AMENDMENT"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def scheduler_record(job_id: int) -> str:
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


def restart_count(record: str) -> int:
    match = re.search(r"(?:^| )Restarts=(\d+)(?: |$)", record)
    if match is None:
        raise RuntimeError("scheduler record lacks Restarts")
    return int(match.group(1))


def amendment_block(wrapper: str) -> str:
    if wrapper.count(BEGIN) != 1 or wrapper.count(END) != 1:
        raise RuntimeError("wrapper does not contain exactly one amendment block")
    start = wrapper.index(BEGIN)
    finish = wrapper.index(END, start) + len(END)
    return wrapper[start:finish]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "after"))
    args = parser.parse_args()

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    runs = [run for run in ledger["runs"] if run["scale"] == "qwen3b"]
    if [int(run["job_id"]) for run in runs] != JOB_IDS:
        raise RuntimeError("E111 Qwen-3B job set changed")
    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    records = {str(job_id): scheduler_record(job_id) for job_id in JOB_IDS}
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()

    if args.mode == "before":
        if BEGIN in wrapper_text or END in wrapper_text:
            raise RuntimeError("durability block already installed")
        payload = {
            "schema": "e111_qwen3_restart_durability_amendment_v1",
            "recorded_before_at": now,
            "amendment": str(PROTOCOL),
            "amendment_sha256": digest(PROTOCOL),
            "ledger": str(LEDGER),
            "ledger_sha256": digest(LEDGER),
            "wrapper": str(WRAPPER),
            "wrapper_before_sha256": digest(WRAPPER),
            "job_ids": JOB_IDS,
            "source_root": str(SOURCE_ROOT),
            "ops_root": str(OPS_ROOT),
            "variant": VARIANT,
            "changed_fields": [
                "OAT_ZERO_SAVE_STEPS",
                "OAT_ZERO_SAVE_FROM",
                "OAT_ZERO_RESUME_STEPS",
            ],
            "checkpoint_before": {
                "OAT_ZERO_SAVE_STEPS": "32",
                "OAT_ZERO_SAVE_FROM": "32",
                "OAT_ZERO_RESUME_STEPS": "32",
            },
            "checkpoint_after": {
                "OAT_ZERO_SAVE_STEPS": "8",
                "OAT_ZERO_SAVE_FROM": "8",
                "OAT_ZERO_RESUME_STEPS": "8",
            },
            "scheduler_only": False,
            "storage_only": True,
            "treatment_environment_changed": False,
            "recovery_environment_changed": True,
            "endpoint_outcomes_inspected": False,
            "pointmaze": "excluded",
            "before_scheduler_records": records,
            "restart_counts_before": {
                key: restart_count(record) for key, record in records.items()
            },
            "installed": False,
        }
    else:
        payload = json.loads(RECORD.read_text(encoding="utf-8"))
        if payload.get("installed") is not False:
            raise RuntimeError("before record is absent or already finalized")
        if payload.get("wrapper_before_sha256") == digest(WRAPPER):
            raise RuntimeError("wrapper digest did not change")
        block = amendment_block(wrapper_text)
        payload.update(
            {
                "recorded_after_at": now,
                "wrapper_after_sha256": digest(WRAPPER),
                "amendment_block_sha256": digest_bytes(block.encode("utf-8")),
                "after_scheduler_records": records,
                "restart_counts_after": {
                    key: restart_count(record) for key, record in records.items()
                },
                "installed": True,
            }
        )

    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(RECORD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
