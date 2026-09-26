#!/usr/bin/env python3
"""Continue the purged E111 Qwen-3B Pantry timeout, fail closed."""

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
import launch_e111_qwen3_python_mathir_replacement_after_purged_timeout as shared  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = shared.LEDGER
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e111_qwen3_pantry_continuation_after_purged_timeout_20260819.md"
)
RECORD = ROOT / "var/artifacts/e111_qwen3_pantry_continuation_jobs.json"
ORIGINAL_JOB_ID = 30674762
DOMAIN = "pantry_plan"


def now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def prior_accounting() -> str:
    result = command(
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        str(ORIGINAL_JOB_ID),
        "--format=JobIDRaw,State,Elapsed,ExitCode,Start,End",
    )
    if result.returncode != 0:
        raise RuntimeError("cannot inspect original Pantry accounting")
    rows = [line for line in result.stdout.splitlines() if line]
    if not rows or rows[0].split("|", 1)[0] != str(ORIGINAL_JOB_ID):
        raise RuntimeError("original Pantry accounting identity mismatch")
    if rows[0].split("|")[1].split("+", 1)[0] != "TIMEOUT":
        raise RuntimeError("original Pantry job is not an exact TIMEOUT")
    active = command("scontrol", "show", "job", "-o", str(ORIGINAL_JOB_ID))
    if active.returncode == 0 or "Invalid job id" not in active.stderr:
        raise RuntimeError("original Pantry job has not been purged as frozen")
    return "\n".join(rows)


def original_run(ledger: dict[str, Any]) -> dict[str, Any]:
    matches = [
        run
        for run in ledger.get("runs", [])
        if int(run.get("job_id", -1)) == ORIGINAL_JOB_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("original E111 Pantry cell is absent or duplicated")
    run = matches[0]
    expected = {"scale": "qwen3b", "domain": DOMAIN, "seed": 70}
    for key, value in expected.items():
        if run.get(key) != value:
            raise RuntimeError(f"original E111 Pantry cell has invalid {key}")
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
    for path in (LEDGER, PROTOCOL, shared.CHECKPOINT_VALIDATOR):
        if not path.is_file():
            raise SystemExit(f"required frozen input is absent: {path}")

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    accounting = prior_accounting()
    run = original_run(ledger)
    snapshot = Path(str(ledger["snapshot_root"])).resolve()
    shared.e111.verify_snapshot(snapshot)
    sbatch, env, _template = shared.continuation_command(run, snapshot)
    checkpoint = shared.checkpoint_selection(run)
    expected_checkpoint = (
        Path(str(run["run_dir"]))
        / f"debug_job{ORIGINAL_JOB_ID}"
        / "checkpoints/step_00054"
    )
    if Path(checkpoint).resolve() != expected_checkpoint.resolve():
        raise RuntimeError(f"unexpected Pantry recovery checkpoint: {checkpoint}")
    if args.dry_run or not args.submit:
        print(" ".join(shlex.quote(token) for token in sbatch))
        print(f"# checkpoint={checkpoint}")
        print("[e111-pantry-continuation] dry_run=True cells=1")
        return 0

    payload: dict[str, Any] = {
        "schema": "e111_qwen3_pantry_continuation_jobs_v1",
        "recorded_before_at": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": shared.e111.digest(PROTOCOL),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": shared.e111.digest(Path(__file__).resolve()),
        "ledger": str(LEDGER),
        "ledger_sha256": shared.e111.digest(LEDGER),
        "original_job_id": ORIGINAL_JOB_ID,
        "prior_job_id": ORIGINAL_JOB_ID,
        "domain": DOMAIN,
        "scale": "qwen3b",
        "seed": 70,
        "run_dir": str(run["run_dir"]),
        "run_stamp": str(run["run_stamp"]),
        "selected_checkpoint_before_submission": checkpoint,
        "prior_accounting": accounting,
        "checkpoint_interval": shared.CHECKPOINT_INTERVAL,
        "partition": shared.PARTITION,
        "nodelist": shared.NODELIST,
        "gres": shared.GRES,
        "time_limit": shared.TIME_LIMIT,
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
        if job_id == ORIGINAL_JOB_ID:
            raise RuntimeError("Pantry continuation did not receive a fresh job ID")
        submitted.append(job_id)
        held = shared.held_job_audit(
            job_id, run=run, env=env, snapshot=snapshot
        )
        payload.update(
            {
                "continuation_job_id": job_id,
                "stdout": str(
                    ROOT
                    / "var/artifacts/logs"
                    / f"{shared.e111.job_name('qwen3b', DOMAIN)}-{job_id}.out"
                ),
                "stderr": str(
                    ROOT
                    / "var/artifacts/logs"
                    / f"{shared.e111.job_name('qwen3b', DOMAIN)}-{job_id}.err"
                ),
                "command": sbatch,
                "held_scheduler_record": held,
            }
        )
        shared.e111.e81.atomic_json(RECORD, payload)
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
        shared.e111.e81.atomic_json(RECORD, payload)
    except Exception as exc:
        shared.cancel(submitted)
        payload.update(
            {
                "failed_at": now(),
                "error": str(exc),
                "released": False,
                "installed": False,
            }
        )
        shared.e111.e81.atomic_json(RECORD, payload)
        raise
    print(f"[e111-pantry-continuation] installed=True job={submitted[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
