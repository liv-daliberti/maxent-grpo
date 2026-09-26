#!/usr/bin/env python3
"""Release an exact E113-R3 allocation whose DAPO learner is already dead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e113_dapo_direct_baseline as e113  # noqa: E402


LEDGER = "var/artifacts/e113r3_dapo_full_relaunch_jobs.json"
AMENDMENT = "var/artifacts/e113r3_terminal_failure_cleanup_amendment.json"
FATAL_PATTERN = re.compile(
    r"DAPO dynamic sampling could not produce a non-constant reward group "
    r"within 10 generation batches at learner step (?P<step>[0-9]+); "
    r"rejected=10, all_zero=(?P<all_zero>[0-9]+), "
    r"all_one=(?P<all_one>[0-9]+)"
)
MIN_LOG_STALE_SECONDS = 300


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scheduler_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", str(job_id), "-o"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def accounting_state(job_id: int) -> tuple[str, str]:
    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            str(job_id),
            "--format=State,ExitCode",
            "-P",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    fields = line.split("|")
    if len(fields) < 2:
        return "UNKNOWN", "UNKNOWN"
    return fields[0].split()[0], fields[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, type=int)
    args = parser.parse_args()
    job_id = args.job_id
    root = e113.repo_root()
    ledger_path = root / LEDGER
    amendment_path = root / AMENDMENT
    output_path = (
        root / f"var/artifacts/e113r3_failed_allocation_cleanup_{job_id}.json"
    )
    if output_path.exists():
        raise SystemExit(f"refusing duplicate cleanup record: {output_path}")
    ledger = load(ledger_path)
    amendment = load(amendment_path)
    job_ids = sorted(int(run["job_id"]) for run in ledger.get("runs", []))
    if (
        ledger.get("schema") != "e113r3_dapo_full_relaunch_jobs_v1"
        or ledger.get("released") is not True
        or job_ids != list(range(30790925, 30790975))
    ):
        raise SystemExit("R3 ledger is not the exact released 50-cell cohort")
    if amendment.get("schema") != "e113r3_terminal_failure_cleanup_amendment_v1":
        raise SystemExit("terminal cleanup amendment is absent or invalid")
    if amendment.get("exact_job_ids") != job_ids:
        raise SystemExit("terminal cleanup amendment job IDs drifted")

    if job_id not in job_ids:
        raise SystemExit(f"refusing cleanup: job {job_id} is not in the R3 ledger")
    run = next(row for row in ledger["runs"] if int(row["job_id"]) == job_id)
    if (Path(str(run["run_dir"])) / "TRAINING_COMPLETE.json").exists():
        raise SystemExit("refusing cleanup: completion receipt exists")
    metrics = list(
        Path(str(run["run_dir"])).glob(f"debug_job{job_id}/train_metrics.jsonl")
    )
    if len(metrics) != 1:
        raise SystemExit("refusing cleanup: metrics evidence is absent or ambiguous")
    rows = [
        json.loads(line)
        for line in metrics[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("refusing cleanup: metrics evidence is empty")
    accepted_updates = max(
        int(row.get("trainer/policy_sgd_step", 0)) for row in rows
    )

    logs = list((root / "var/artifacts/logs").glob(f"e113r3-*-{job_id}.out"))
    log_text = (
        logs[0].read_text(encoding="utf-8", errors="replace")
        if len(logs) == 1
        else ""
    )
    fatal_match = FATAL_PATTERN.search(log_text)
    if fatal_match is None:
        raise SystemExit("refusing cleanup: fatal evidence is absent or ambiguous")
    if int(fatal_match.group("all_zero")) + int(fatal_match.group("all_one")) != 10:
        raise SystemExit("refusing cleanup: fatal rejection accounting drifted")
    stale_seconds = time.time() - logs[0].stat().st_mtime
    if stale_seconds < MIN_LOG_STALE_SECONDS:
        raise SystemExit(
            f"refusing cleanup: fatal log is only {stale_seconds:.0f}s stale"
        )
    before = scheduler_record(job_id)
    if f"JobId={job_id} " not in before or "JobState=RUNNING" not in before:
        raise SystemExit("refusing cleanup: exact job is not still RUNNING")
    if "Restarts=0" not in before:
        raise SystemExit("refusing cleanup: restart count drifted")

    subprocess.run(["scancel", str(job_id)], check=True)
    state, exit_code = accounting_state(job_id)
    for _ in range(30):
        if state not in {"PENDING", "RUNNING", "COMPLETING", "UNKNOWN"}:
            break
        time.sleep(1)
        state, exit_code = accounting_state(job_id)
    if state not in {"CANCELLED", "FAILED"}:
        raise SystemExit(f"cleanup did not reach a terminal state: {state}/{exit_code}")

    payload = {
        "schema": "e113r3_failed_allocation_cleanup_v1",
        "job_id": job_id,
        "administrative_cleanup_only": True,
        "scientific_outcome_changed": False,
        "accepted_updates": accepted_updates,
        "completion_receipt": False,
        "fatal_signature": fatal_match.group(0),
        "fatal_learner_step": int(fatal_match.group("step")),
        "log": str(logs[0]),
        "log_sha256": e113.e78.digest(logs[0]),
        "log_stale_seconds_before_cleanup": round(stale_seconds, 3),
        "scheduler_record_before": before,
        "scheduler_state_after": state,
        "scheduler_exit_code_after": exit_code,
        "ledger": str(ledger_path),
        "ledger_sha256": e113.e78.digest(ledger_path),
        "amendment": str(amendment_path),
        "amendment_sha256": e113.e78.digest(amendment_path),
        "metrics": str(metrics[0]),
        "metrics_sha256": e113.e78.digest(metrics[0]),
    }
    e113.e78.atomic_json(output_path, payload)
    print(f"[e113r3s3] released dead allocation {job_id}: {state}/{exit_code}")
    print(f"[e113r3s3] record {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
