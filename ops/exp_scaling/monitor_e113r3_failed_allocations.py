#!/usr/bin/env python3
"""Reap learner-dead allocations in the exact released E113-R3 cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleanup_e113r3_failed_allocation as cleanup  # noqa: E402
import launch_e113_dapo_direct_baseline as e113  # noqa: E402


LEDGER = "var/artifacts/e113r3_dapo_full_relaunch_jobs.json"
REAPER_LEDGER = "var/artifacts/e113r3_failure_reaper_job.json"
ACTIVE_STATES = {"CONFIGURING", "COMPLETING", "PENDING", "RUNNING"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scheduler_states(job_ids: list[int]) -> dict[int, str]:
    result = subprocess.run(
        [
            "squeue",
            "-h",
            "-j",
            ",".join(str(job_id) for job_id in job_ids),
            "-o",
            "%i|%T",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    states: dict[int, str] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", maxsplit=1)
        if len(fields) == 2 and fields[0].isdigit():
            states[int(fields[0])] = fields[1]
    return states


def eligible_failure(root: Path, run: dict[str, Any], state: str) -> bool:
    job_id = int(run["job_id"])
    if state != "RUNNING":
        return False
    if (root / f"var/artifacts/e113r3_failed_allocation_cleanup_{job_id}.json").exists():
        return False
    if (Path(str(run["run_dir"])) / "TRAINING_COMPLETE.json").exists():
        return False
    logs = list((root / "var/artifacts/logs").glob(f"e113r3-*-{job_id}.out"))
    if len(logs) != 1:
        return False
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    if cleanup.FATAL_PATTERN.search(text) is None:
        return False
    return time.time() - logs[0].stat().st_mtime >= cleanup.MIN_LOG_STALE_SECONDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=601200)
    args = parser.parse_args()
    if args.poll_seconds < 15 or args.max_seconds < args.poll_seconds:
        raise SystemExit("invalid monitor timing")

    root = e113.repo_root()
    ledger = load(root / LEDGER)
    reaper = load(root / REAPER_LEDGER)
    runs = list(ledger.get("runs", []))
    job_ids = sorted(int(run["job_id"]) for run in runs)
    if (
        ledger.get("schema") != "e113r3_dapo_full_relaunch_jobs_v1"
        or ledger.get("released") is not True
        or job_ids != list(range(30790925, 30790975))
    ):
        raise SystemExit("R3 ledger is not the exact released 50-cell cohort")

    cleanup_script = root / "ops/exp_scaling/cleanup_e113r3_failed_allocation.py"
    monitor_script = Path(__file__).resolve()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if (
        reaper.get("schema") != "e113r3_failure_reaper_job_v1"
        or reaper.get("released") is not True
        or reaper.get("monitored_job_ids") != job_ids
        or str(reaper.get("job_id")) != slurm_job_id
        or reaper.get("monitor_sha256") != e113.e78.digest(monitor_script)
        or reaper.get("cleanup_sha256") != e113.e78.digest(cleanup_script)
    ):
        raise SystemExit("reaper launch identity or frozen code hashes drifted")
    expected_cleanup_sha256 = str(reaper["cleanup_sha256"])
    started = time.monotonic()
    print(
        f"[e113r3s4] monitoring {len(job_ids)} exact jobs; "
        f"poll={args.poll_seconds}s max={args.max_seconds}s",
        flush=True,
    )
    while time.monotonic() - started < args.max_seconds:
        states = scheduler_states(job_ids)
        active = {job_id: state for job_id, state in states.items() if state in ACTIVE_STATES}
        if not active:
            print("[e113r3s4] no active R3 jobs remain", flush=True)
            return 0
        for run in runs:
            job_id = int(run["job_id"])
            if not eligible_failure(root, run, states.get(job_id, "")):
                continue
            if e113.e78.digest(cleanup_script) != expected_cleanup_sha256:
                raise SystemExit("cleanup utility changed after reaper release")
            print(f"[e113r3s4] reaping learner-dead job {job_id}", flush=True)
            result = subprocess.run(
                [sys.executable, str(cleanup_script), "--job-id", str(job_id)],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.stdout:
                print(result.stdout.rstrip(), flush=True)
            if result.returncode and result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr, flush=True)
        time.sleep(args.poll_seconds)
    print("[e113r3s4] monitor reached its frozen time limit", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
