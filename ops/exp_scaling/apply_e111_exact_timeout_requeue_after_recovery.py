#!/usr/bin/env python3
"""Requeue the exact recovered E111 jobs after an ordinary wall-time timeout."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e111_verified_support_discovery_mechanism_gate_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e111_exact_timeout_requeue_after_recovery_20260818.md"
RECORD = ROOT / "var/artifacts/e111_exact_timeout_requeue_after_recovery.json"
JOB_IDS = (30674729, 30674733, 30674754, 30674762)
VARIANT = "verified_replay_semantic_maxent_verified_support_discovery"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def accounting(job_id: int) -> str:
    result = command(
        "sacct", "-n", "-P", "-j", str(job_id),
        "--format=JobIDRaw,State,Elapsed,ExitCode,Start,End",
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect accounting for {job_id}")
    rows = [line for line in result.stdout.splitlines() if line]
    if not rows or rows[0].split("|", 1)[0] != str(job_id):
        raise RuntimeError(f"accounting record mismatch for {job_id}")
    if rows[0].split("|")[1].split("+", 1)[0] != "TIMEOUT":
        raise RuntimeError(f"job {job_id} is not an exact TIMEOUT")
    return "\n".join(rows)


def scheduler(job_id: int) -> str:
    result = command("scontrol", "show", "job", "-dd", "-o", str(job_id))
    if result.returncode != 0 or f"JobId={job_id}" not in result.stdout:
        raise RuntimeError(f"cannot inspect requeued job {job_id}")
    record = result.stdout.strip()
    for required in (
        f"JobId={job_id}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        "OAT_ZERO_SOURCE_ROOT=" + str(
            ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/src"
        ),
    ):
        if required not in record:
            raise RuntimeError(f"requeued job {job_id} lacks {required}")
    if not any(f"JobState={state}" in record for state in ("PENDING", "RUNNING")):
        raise RuntimeError(f"requeued job {job_id} has unexpected state")
    return record


def main() -> int:
    if RECORD.exists():
        raise SystemExit(f"refusing duplicate recovery application: {RECORD}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    runs = {int(run["job_id"]): run for run in ledger.get("runs", [])}
    if set(JOB_IDS) - set(runs):
        raise RuntimeError("exact recovery jobs are absent from E111 ledger")
    pre = {str(job_id): accounting(job_id) for job_id in JOB_IDS}
    payload = {
        "schema": "e111_exact_timeout_requeue_after_recovery_v1",
        "recorded_before_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "ledger": str(LEDGER),
        "ledger_sha256": digest(LEDGER),
        "job_ids": list(JOB_IDS),
        "pre_action_accounting": pre,
        "commands": [["scontrol", "requeue", str(job_id)] for job_id in JOB_IDS],
        "same_job_ids": True,
        "replacement_jobs_submitted": False,
        "state_reset": False,
        "environment_changed": False,
        "treatment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "installed": False,
    }
    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    results: dict[str, dict[str, object]] = {}
    for job_id in JOB_IDS:
        result = command("scontrol", "requeue", str(job_id))
        results[str(job_id)] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode != 0:
            payload["requeue_results"] = results
            RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            raise RuntimeError(f"requeue failed for {job_id}: {result.stderr.strip()}")
    payload.update(
        {
            "recorded_after_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "requeue_results": results,
            "post_action_scheduler_records": {
                str(job_id): scheduler(job_id) for job_id in JOB_IDS
            },
            "installed": True,
        }
    )
    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[e111-timeout-requeue] installed=True jobs={len(JOB_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
