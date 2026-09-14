#!/usr/bin/env python3
"""Resume the failed E119 MathIR Re:MaxRL seed-44 continuation.

The scientific cell, run directory, frozen source, objective, and seed remain
unchanged.  The replacement resumes from the latest durable checkpoint with a
two-hour evaluation-safe watchdog, node105/mltheory placement, and the full
campaign PVL exclusion.  It is submitted held, audited, recorded, and only
then released.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import launch_e119_level2_qwen05b_factorial as e119


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"
LEDGER = ROOT / e119.LEDGER
CONTINUATIONS = ARTIFACTS / "e119_level2_continuation_jobs.json"
AUDIT = ARTIFACTS / "e119_mathir_replaymax_s44_watchdog_recovery_20260904.json"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
ORIGINAL_JOB_ID = 31014445
FAILED_JOB_ID = 31041935
EXPECTED = ("mathir", "replay_maxrl", 44)


def atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def state(job_id: int) -> str:
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", str(job_id), "--format=State", "-P"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [
        line.split("|", 1)[0].strip().split()[0]
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return rows[0] if rows else "NOT_IN_QUEUE"


def latest_checkpoint(run_dir: Path) -> tuple[int, Path]:
    checkpoints: list[tuple[int, Path]] = []
    for candidate in run_dir.glob("debug_job*/checkpoints/step_*"):
        try:
            step = int(candidate.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        if candidate.is_dir():
            checkpoints.append((step, candidate))
    if not checkpoints:
        raise RuntimeError(f"no durable checkpoint under {run_dir}")
    return max(checkpoints)


def node105_command(command: list[str]) -> list[str]:
    rewritten = ["--mem=40G" if part == "--mem=64G" else part for part in command]
    return [*rewritten[:-1], f"--exclude={PVL}", rewritten[-1]]


def show(job_id: int) -> str:
    return subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def main() -> None:
    if AUDIT.exists():
        raise SystemExit(f"recovery already installed: {AUDIT}")

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    continuation = json.loads(CONTINUATIONS.read_text(encoding="utf-8"))
    continuation_before = json.loads(json.dumps(continuation))
    runs = {int(row["job_id"]): row for row in ledger["runs"]}
    records = {
        int(row["original_job_id"]): row
        for row in continuation["continuations"]
    }

    run = runs[ORIGINAL_JOB_ID]
    identity = (str(run["domain"]), str(run["arm"]), int(run["seed"]))
    if identity != EXPECTED:
        raise RuntimeError(f"identity drift for {ORIGINAL_JOB_ID}: {identity}")
    record = records[ORIGINAL_JOB_ID]
    if int(record["continuation_job_id"]) != FAILED_JOB_ID:
        raise RuntimeError(f"continuation lineage drift for {ORIGINAL_JOB_ID}")
    if state(FAILED_JOB_ID) != "FAILED":
        raise RuntimeError(f"continuation {FAILED_JOB_ID} is not FAILED")

    checkpoint_step, checkpoint = latest_checkpoint(Path(run["run_dir"]))
    templates = {
        (str(row["domain"]), int(row["seed"])): row
        for row in e119.templates(ROOT)
    }
    domain, arm, seed = EXPECTED
    template = templates[(domain, seed)]
    snapshot = Path(ledger["snapshot_root"])
    env, target = e119.environment(ROOT, template, arm, snapshot)
    if str(target) != str(run["run_dir"]):
        raise RuntimeError(f"run-directory drift for {ORIGINAL_JOB_ID}")
    env.update(
        {
            "OAT_ZERO_WATCHDOG_STALE_SECONDS": "7200",
            "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS": "3600",
            "OAT_ZERO_WATCHDOG_MAX_RESTARTS": "12",
        }
    )
    command = node105_command(e119.command(ROOT, template, arm, env))
    plan: dict[str, Any] = {
        "original_job_id": ORIGINAL_JOB_ID,
        "failed_job_id": FAILED_JOB_ID,
        "domain": domain,
        "arm": arm,
        "seed": seed,
        "run_dir": str(run["run_dir"]),
        "run_stamp": str(run["run_stamp"]),
        "checkpoint_step": checkpoint_step,
        "checkpoint": str(checkpoint),
    }

    submitted: list[int] = []
    committed = False
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "sbatch failed")
        new_id = int(result.stdout.strip().split(";", 1)[0])
        submitted.append(new_id)
        held = show(new_id)
        required = (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "Account=mltheory",
            "Partition=mltheory",
            "ReqNodeList=node105",
            "MinMemoryNode=40G",
            f"ExcNodeList={PVL}",
            "OAT_ZERO_WATCHDOG_STALE_SECONDS=7200",
            "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600",
            "OAT_ZERO_WATCHDOG_MAX_RESTARTS=12",
            "OAT_ZERO_AUTO_RESUME=1",
            f"RUN_STAMP={plan['run_stamp']}",
            f"SAVE_PATH={plan['run_dir']}",
        )
        missing = [token for token in required if token not in held]
        if missing:
            raise RuntimeError(f"held replacement {new_id} missing {missing}")
        plan["new_job_id"] = new_id
        plan["held_scheduler_record"] = held

        previous = [
            int(value) for value in record.get("previous_continuation_job_ids", [])
        ]
        record["previous_continuation_job_ids"] = [*previous, FAILED_JOB_ID]
        record["continuation_job_id"] = new_id
        record["repair_kind"] = "evaluation_safe_node105_watchdog_continuation"
        continuation["continuations"] = list(records.values())
        continuation["operational_change"] = (
            str(continuation.get("operational_change", "")).rstrip("; ")
            + "; 2026-09-04 MathIR Re:MaxRL seed 44 resumed with a "
            "two-hour evaluation-safe watchdog on node105/mltheory"
        ).lstrip("; ")
        continuation["released"] = False
        audit: dict[str, Any] = {
            "schema": "e119-mathir-replaymax-s44-watchdog-recovery-v1",
            "cause": "watchdog exit 75 during long evaluation",
            "same_scientific_cell": True,
            "same_run_directory": True,
            "automatic_checkpoint_resume": True,
            "optimizer_update_changed": False,
            "treatment_changed": False,
            "outcomes_inspected": False,
            "placement": "node105/mltheory",
            "watchdog_stale_seconds": 7200,
            "pvl_exclusion": PVL,
            "released": False,
            "replacement": plan,
        }
        atomic(CONTINUATIONS, continuation)
        atomic(AUDIT, audit)
        committed = True

        subprocess.run(["scontrol", "release", str(new_id)], check=True)
        continuation["released"] = True
        audit["released"] = True
        atomic(CONTINUATIONS, continuation)
        atomic(AUDIT, audit)
    except Exception:
        if submitted:
            subprocess.run(["scancel", *map(str, submitted)], check=False)
        if committed:
            atomic(CONTINUATIONS, continuation_before)
            AUDIT.unlink(missing_ok=True)
        raise

    print(
        "recovered E119 MathIR Re:MaxRL seed 44: "
        + ",".join(str(job_id) for job_id in submitted)
    )


if __name__ == "__main__":
    main()
