#!/usr/bin/env python3
"""Collect parent-job Slurm accounting for this two-domain goal only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def account(root: Path) -> dict:
    jobs = {31433382: {"name": "initial_code_capability", "gpu_hour_ceiling": 1.0}}
    for submission in sorted(root.glob("*/submission.json")):
        record = json.loads(submission.read_text())
        if record.get("returncode") != 0:
            continue
        job_id = int(record.get("job_id") or record["stdout"].strip().split(";")[0])
        identity = json.loads((submission.parent / "identity.json").read_text())
        jobs[job_id] = {"name": submission.parent.name, "gpu_hour_ceiling": identity["allocated_gpu_hour_ceiling"], "directory": str(submission.parent.resolve())}
    fields = ["JobIDRaw", "State", "ExitCode", "ElapsedRaw", "Start", "End", "AllocTRES", "NodeList"]
    command = ["sacct", "-X", "-j", ",".join(map(str, jobs)), "-n", "-P", "--format=" + ",".join(fields)]
    proc = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = []
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
    observed = set()
    for line in proc.stdout.splitlines():
        values = line.split("|")
        if len(values) != len(fields):
            raise ValueError("unexpected Slurm accounting format")
        value = dict(zip(fields, values))
        job_id = int(value["JobIDRaw"])
        if job_id not in jobs or job_id in observed:
            raise ValueError("unexpected or duplicate parent job")
        observed.add(job_id)
        tres = dict(v.split("=", 1) for v in value["AllocTRES"].split(",") if "=" in v)
        # A generic count and a type-specific count describe the SAME GPU.
        gpu_count = int(tres.get("gres/gpu", "0"))
        if not gpu_count and any(k.startswith("gres/gpu:") for k in tres):
            gpu_count = sum(int(v) for k, v in tres.items() if k.startswith("gres/gpu:"))
        state = value["State"].split()[0].rstrip("+")
        elapsed = int(value["ElapsedRaw"])
        gpu_hours = gpu_count * elapsed / 3600
        row = {**jobs[job_id], "job_id": job_id, "state": state, "exit_code": value["ExitCode"], "elapsed_seconds": elapsed, "gpu_count": gpu_count, "allocated_gpu_hours": gpu_hours, "terminal": state in terminal, "raw_slurm": value}
        rows.append(row)
        if "directory" in row:
            (Path(row["directory"]) / "scheduler_receipt.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    missing = sorted(set(jobs) - observed)
    if missing:
        raise RuntimeError(f"Slurm has no parent record for submitted jobs {missing}; do not infer they stopped")
    actual = sum(r["allocated_gpu_hours"] for r in rows)
    remaining_allocations = sum(max(0, r["gpu_hour_ceiling"] - r["allocated_gpu_hours"]) for r in rows if not r["terminal"])
    result = {"schema": "real-domains-slurm-budget-20260921-v1", "updated_at": datetime.now(timezone.utc).isoformat(), "total_goal_gpu_hour_limit": 200, "accounted_gpu_hours": actual, "remaining_live_allocation_ceiling": remaining_allocations, "maximum_after_current_allocations": actual + remaining_allocations, "available_for_new_allocations": 200 - actual - remaining_allocations, "jobs": rows, "accounting_command": command}
    (root / "budget.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"jobs", "accounting_command"}}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    account(parser.parse_args().root)
