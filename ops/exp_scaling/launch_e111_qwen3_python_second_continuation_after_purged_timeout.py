#!/usr/bin/env python3
"""Continue E111 Qwen-3B Python from its purged first continuation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e111_qwen3_python_mathir_replacement as first_audit  # noqa: E402
import launch_e111_qwen3_python_mathir_replacement_after_purged_timeout as first  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = first.LEDGER
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e111_qwen3_python_second_continuation_after_purged_timeout_20260818.md"
)
RECORD = ROOT / "var/artifacts/e111_qwen3_python_second_continuation_jobs.json"
ORIGINAL_JOB_ID = 30674760
PRIOR_JOB_ID = 30739797
DOMAIN = "python_factors"


def now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def prior_accounting() -> str:
    result = command(
        "sacct",
        "-n",
        "-P",
        "-j",
        str(PRIOR_JOB_ID),
        "--format=JobIDRaw,State,Elapsed,ExitCode,Start,End",
    )
    if result.returncode != 0:
        raise RuntimeError("cannot inspect first-continuation accounting")
    rows = [line for line in result.stdout.splitlines() if line]
    if not rows or rows[0].split("|", 1)[0] != str(PRIOR_JOB_ID):
        raise RuntimeError("first-continuation accounting identity mismatch")
    if rows[0].split("|")[1].split("+", 1)[0] != "TIMEOUT":
        raise RuntimeError("first continuation is not an exact TIMEOUT")
    active = command("scontrol", "show", "job", "-o", str(PRIOR_JOB_ID))
    if active.returncode == 0 or "Invalid job id" not in active.stderr:
        raise RuntimeError("first continuation has not been purged as frozen")
    return "\n".join(rows)


def original_run(ledger: dict[str, Any]) -> dict[str, Any]:
    matches = [
        run
        for run in ledger.get("runs", [])
        if int(run.get("job_id", -1)) == ORIGINAL_JOB_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("original E111 Python cell is absent or duplicated")
    run = matches[0]
    expected = {"scale": "qwen3b", "domain": DOMAIN, "seed": 70}
    for key, value in expected.items():
        if run.get(key) != value:
            raise RuntimeError(f"original E111 Python cell has invalid {key}")
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    if args.submit and RECORD.exists():
        raise SystemExit(f"refusing duplicate continuation: {RECORD}")
    for path in (LEDGER, PROTOCOL, first.RECORD):
        if not path.is_file():
            raise SystemExit(f"required frozen input is absent: {path}")

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    first_report, first_violations = first_audit.validate(
        ledger, ledger.get("runs", [])
    )
    if first_violations or first_report.get("passed") is not True:
        raise RuntimeError(f"first continuation audit failed: {first_violations}")
    prior = first_report.get("continuation_by_original_job_id", {}).get(
        str(ORIGINAL_JOB_ID), {}
    )
    if int(prior.get("continuation_job_id", -1)) != PRIOR_JOB_ID:
        raise RuntimeError("first continuation chain identity drifted")
    accounting = prior_accounting()
    run = original_run(ledger)
    snapshot = Path(str(ledger["snapshot_root"])).resolve()
    first.e111.verify_snapshot(snapshot)
    sbatch, env, _template = first.continuation_command(run, snapshot)
    checkpoint = first.checkpoint_selection(run)
    if not checkpoint.endswith("debug_job30739797/checkpoints/step_00054"):
        raise RuntimeError(f"unexpected Python recovery checkpoint: {checkpoint}")
    if args.dry_run or not args.submit:
        print(" ".join(shlex.quote(token) for token in sbatch))
        print(f"# checkpoint={checkpoint}")
        print("[e111-python-second-continuation] dry_run=True cells=1")
        return 0

    payload: dict[str, Any] = {
        "schema": "e111_qwen3_python_second_continuation_jobs_v1",
        "recorded_before_at": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": first.e111.digest(PROTOCOL),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": first.e111.digest(Path(__file__).resolve()),
        "ledger": str(LEDGER),
        "ledger_sha256": first.e111.digest(LEDGER),
        "first_continuation_record": str(first.RECORD),
        "first_continuation_record_sha256": first.e111.digest(first.RECORD),
        "original_job_id": ORIGINAL_JOB_ID,
        "prior_continuation_job_id": PRIOR_JOB_ID,
        "domain": DOMAIN,
        "scale": "qwen3b",
        "seed": 70,
        "run_dir": str(run["run_dir"]),
        "run_stamp": str(run["run_stamp"]),
        "selected_checkpoint_before_submission": checkpoint,
        "prior_accounting": accounting,
        "checkpoint_interval": 2,
        "partition": first.PARTITION,
        "nodelist": first.NODELIST,
        "gres": first.GRES,
        "time_limit": first.TIME_LIMIT,
        "same_scientific_cell": True,
        "same_run_directory": True,
        "state_reset": False,
        "optimizer_update_changed": False,
        "environment_changed_except_storage_and_placement": False,
        "treatment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "released": False,
        "installed": False,
    }
    submitted: list[int] = []
    try:
        result = subprocess.run(sbatch, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sbatch failed")
        value = result.stdout.strip().split(";", 1)[0]
        if not value.isdigit():
            raise RuntimeError(f"invalid continuation job id: {result.stdout!r}")
        job_id = int(value)
        if job_id in (ORIGINAL_JOB_ID, PRIOR_JOB_ID):
            raise RuntimeError("second continuation did not receive a fresh job ID")
        submitted.append(job_id)
        held = first.held_job_audit(
            job_id, run=run, env=env, snapshot=snapshot
        )
        payload.update(
            {
                "continuation_job_id": job_id,
                "stdout": str(
                    ROOT
                    / "var/artifacts/logs"
                    / f"{first.e111.job_name('qwen3b', DOMAIN)}-{job_id}.out"
                ),
                "stderr": str(
                    ROOT
                    / "var/artifacts/logs"
                    / f"{first.e111.job_name('qwen3b', DOMAIN)}-{job_id}.err"
                ),
                "command": sbatch,
                "held_scheduler_record": held,
            }
        )
        first.e111.e81.atomic_json(RECORD, payload)
        released = command("scontrol", "release", str(job_id))
        if released.returncode != 0:
            raise RuntimeError(f"release failed for continuation {job_id}")
        payload.update(
            {
                "release_result": {
                    "returncode": released.returncode,
                    "stdout": released.stdout,
                    "stderr": released.stderr,
                },
                "recorded_after_at": now(),
                "released": True,
                "installed": True,
            }
        )
        first.e111.e81.atomic_json(RECORD, payload)
    except Exception as exc:
        first.cancel(submitted)
        payload.update(
            {
                "failed_at": now(),
                "error": str(exc),
                "released": False,
                "installed": False,
            }
        )
        first.e111.e81.atomic_json(RECORD, payload)
        raise
    print(
        f"[e111-python-second-continuation] installed=True job={submitted[0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
