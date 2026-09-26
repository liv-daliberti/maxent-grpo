#!/usr/bin/env python3
"""Submit the CPU-only external failure reaper for exact E113-R3 jobs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e113_dapo_direct_baseline as e113  # noqa: E402


LEDGER = "var/artifacts/e113r3_dapo_full_relaunch_jobs.json"
PROTOCOL = "paper/preregistration/e113r3s4_external_failure_reaper_20260819.md"
MONITOR = "ops/exp_scaling/monitor_e113r3_failed_allocations.py"
CLEANUP = "ops/exp_scaling/cleanup_e113r3_failed_allocation.py"
SLURM_SCRIPT = "ops/slurm/e113r3_failure_reaper.slurm"
OUTPUT = "var/artifacts/e113r3_failure_reaper_job.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = e113.repo_root()
    paths = {
        "ledger": root / LEDGER,
        "protocol": root / PROTOCOL,
        "monitor": root / MONITOR,
        "cleanup": root / CLEANUP,
        "slurm_script": root / SLURM_SCRIPT,
    }
    output_path = root / OUTPUT
    if output_path.exists():
        raise SystemExit(f"refusing duplicate reaper launch: {output_path}")
    for path in paths.values():
        if not path.is_file():
            raise SystemExit(f"required reaper input is absent: {path}")
    ledger = load(paths["ledger"])
    job_ids = sorted(int(run["job_id"]) for run in ledger.get("runs", []))
    if (
        ledger.get("schema") != "e113r3_dapo_full_relaunch_jobs_v1"
        or ledger.get("released") is not True
        or job_ids != list(range(30790925, 30790975))
    ):
        raise SystemExit("R3 ledger is not the exact released 50-cell cohort")

    command = [
        "sbatch",
        "--parsable",
        "--hold",
        "--no-requeue",
        "--job-name=e113r3-failure-reaper",
        "--partition=all",
        "--account=mltheory",
        "--cpus-per-task=1",
        "--mem=1G",
        "--time=7-00:00:00",
        "--nice=100",
        f"--output={root}/var/artifacts/logs/e113r3-failure-reaper-%j.out",
        f"--error={root}/var/artifacts/logs/e113r3-failure-reaper-%j.err",
        str(paths["slurm_script"]),
    ]
    job_id = ""
    try:
        submitted = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = submitted.stdout.strip().split(";", maxsplit=1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"invalid sbatch response: {submitted.stdout!r}")
        inspected = subprocess.run(
            ["scontrol", "show", "job", job_id, "-o"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        required = (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "Dependency=(null)",
            "Requeue=0",
            "NumCPUs=1",
            "MinMemoryNode=1G",
        )
        missing = [value for value in required if value not in inspected]
        if missing or "gres/gpu" in inspected:
            raise RuntimeError(
                f"held reaper audit failed; missing={missing}, gpu={'gres/gpu' in inspected}"
            )
        payload = {
            "schema": "e113r3_failure_reaper_job_v1",
            "released": False,
            "scientific_cells": 0,
            "monitored_job_ids": job_ids,
            "job_id": int(job_id),
            "held_scheduler_record": inspected,
            "command": command,
            **{
                f"{name}_sha256": e113.e78.digest(path)
                for name, path in paths.items()
            },
        }
        e113.e78.atomic_json(output_path, payload)
        subprocess.run(["scontrol", "release", job_id], cwd=root, check=True)
        payload["released"] = True
        e113.e78.atomic_json(output_path, payload)
    except BaseException:
        if job_id:
            subprocess.run(["scancel", job_id], cwd=root, check=False)
        raise
    print(f"[e113r3s4] released CPU-only failure reaper job {job_id}")
    print(f"[e113r3s4] record {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
