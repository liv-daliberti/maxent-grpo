#!/usr/bin/env python3
"""Record the effective E111 Qwen-3B runtime-ops durability amendment."""

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
PROTOCOL = ROOT / "paper/preregistration/e111_qwen3_runtime_ops_durability_amendment_20260818.md"
RUNTIME_TRAIN = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops/train.sh"
RECORD = ROOT / "var/artifacts/e111_qwen3_runtime_ops_durability_amendment.json"
JOB_IDS = [30674758, 30674759, 30674760, 30674761, 30674762]
SOURCE_ROOT = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/src"
OPS_ROOT = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops"
VARIANT = "verified_replay_semantic_maxent_verified_support_discovery"
BEGIN = "# BEGIN E111_QWEN3_RUNTIME_OPS_DURABILITY_AMENDMENT"
END = "# END E111_QWEN3_RUNTIME_OPS_DURABILITY_AMENDMENT"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def command(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def scheduler_record(job_id: int) -> str:
    record = command("scontrol", "show", "job", "-o", str(job_id))
    if f"JobId={job_id}" not in record:
        raise RuntimeError(f"scheduler record mismatch for {job_id}")
    return record


def restart_count(record: str) -> int:
    match = re.search(r"(?:^| )Restarts=(\d+)(?: |$)", record)
    if match is None:
        raise RuntimeError("scheduler record lacks Restarts")
    return int(match.group(1))


def amendment_block(value: str) -> str:
    if value.count(BEGIN) != 1 or value.count(END) != 1:
        raise RuntimeError("runtime train script lacks one exact amendment block")
    start = value.index(BEGIN)
    finish = value.index(END, start) + len(END)
    return value[start:finish]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "after"))
    args = parser.parse_args()

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    runs = [run for run in ledger["runs"] if run["scale"] == "qwen3b"]
    if [int(run["job_id"]) for run in runs] != JOB_IDS:
        raise RuntimeError("E111 Qwen-3B job set changed")
    runtime_text = RUNTIME_TRAIN.read_text(encoding="utf-8")
    records = {str(job_id): scheduler_record(job_id) for job_id in JOB_IDS}
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()

    if args.mode == "before":
        if BEGIN in runtime_text or END in runtime_text:
            raise RuntimeError("runtime-ops durability block already installed")
        stored = command("scontrol", "write", "batch_script", "30674760", "-")
        required_stored = (
            'RUNTIME_OPS_ROOT="${OAT_ZERO_OPS_SNAPSHOT_ROOT:-${ROOT_DIR}/ops}"',
            'cp "${RUNTIME_OPS_ROOT}/train.sh" "$RUNTIME_SCRIPT_DIR/train.sh"',
            'exec "$RUNTIME_SCRIPT_DIR/run_experiment.sh"',
        )
        for needle in required_stored:
            if needle not in stored:
                raise RuntimeError(f"stored batch script lacks {needle}")
        payload = {
            "schema": "e111_qwen3_runtime_ops_durability_amendment_v1",
            "recorded_before_at": now,
            "amendment": str(PROTOCOL),
            "amendment_sha256": digest(PROTOCOL),
            "ledger": str(LEDGER),
            "ledger_sha256": digest(LEDGER),
            "runtime_train": str(RUNTIME_TRAIN),
            "runtime_train_before_sha256": digest(RUNTIME_TRAIN),
            "stored_batch_script_job_id": 30674760,
            "stored_batch_script": stored,
            "stored_batch_script_sha256": digest_bytes(stored.encode("utf-8")),
            "stored_batch_script_rereads_runtime_train": True,
            "job_ids": JOB_IDS,
            "source_root": str(SOURCE_ROOT),
            "ops_root": str(OPS_ROOT),
            "variant": VARIANT,
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
            "storage_only": True,
            "python_source_changed": False,
            "treatment_changed": False,
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
        block = amendment_block(runtime_text)
        payload.update(
            {
                "recorded_after_at": now,
                "runtime_train_after_sha256": digest(RUNTIME_TRAIN),
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
