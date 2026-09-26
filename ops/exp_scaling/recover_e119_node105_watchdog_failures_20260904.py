#!/usr/bin/env python3
"""Resume four E119 node105 continuations killed by the short eval watchdog.

The scientific cells, run directories, frozen source, objectives, and seeds are
unchanged.  Replacements use the latest durable checkpoints, a two-hour stale
window for terminal evaluation, node105/mltheory, and the campaign-wide PVL
exclusion.  Jobs are submitted held, audited, recorded, and only then released.
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
AUDIT = ARTIFACTS / "e119_node105_watchdog_recovery_20260904.json"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
FAILURES = {
    31014399: (31041933, "countdown", "replay_drgrpo", 43),
    31014404: (31041936, "countdown", "maxrl", 44),
    31014398: (31041937, "countdown", "drgrpo", 43),
    31014403: (31041938, "countdown", "replay_drgrpo", 44),
}


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
    templates = {
        (str(row["domain"]), int(row["seed"])): row
        for row in e119.templates(ROOT)
    }
    snapshot = Path(ledger["snapshot_root"])

    plans: list[dict[str, Any]] = []
    for original_id, expected in FAILURES.items():
        failed_id, domain, arm, seed = expected
        run = runs[original_id]
        identity = (str(run["domain"]), str(run["arm"]), int(run["seed"]))
        if identity != (domain, arm, seed):
            raise RuntimeError(f"identity drift for {original_id}: {identity}")
        record = records[original_id]
        if int(record["continuation_job_id"]) != failed_id:
            raise RuntimeError(f"continuation lineage drift for {original_id}")
        if state(failed_id) != "FAILED":
            raise RuntimeError(f"continuation {failed_id} is not FAILED")
        checkpoint_step, checkpoint = latest_checkpoint(Path(run["run_dir"]))
        template = templates[(domain, seed)]
        env, target = e119.environment(ROOT, template, arm, snapshot)
        if str(target) != str(run["run_dir"]):
            raise RuntimeError(f"run-directory drift for {original_id}")
        env.update(
            {
                "OAT_ZERO_WATCHDOG_STALE_SECONDS": "7200",
                "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS": "3600",
                "OAT_ZERO_WATCHDOG_MAX_RESTARTS": "12",
            }
        )
        plans.append(
            {
                "original_job_id": original_id,
                "failed_job_id": failed_id,
                "domain": domain,
                "arm": arm,
                "seed": seed,
                "run_dir": str(run["run_dir"]),
                "run_stamp": str(run["run_stamp"]),
                "checkpoint_step": checkpoint_step,
                "checkpoint": str(checkpoint),
                "command": node105_command(e119.command(ROOT, template, arm, env)),
            }
        )

    submitted: list[int] = []
    audit: dict[str, Any] = {}
    committed = False
    try:
        for plan in plans:
            result = subprocess.run(plan.pop("command"), capture_output=True, text=True)
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

        for plan in plans:
            original_id = int(plan["original_job_id"])
            record = records[original_id]
            previous = [int(value) for value in record.get("previous_continuation_job_ids", [])]
            failed_id = int(plan["failed_job_id"])
            record["previous_continuation_job_ids"] = [*previous, failed_id]
            record["continuation_job_id"] = int(plan["new_job_id"])
            record["repair_kind"] = "evaluation_safe_node105_watchdog_continuation"

        continuation["continuations"] = list(records.values())
        continuation["operational_change"] = (
            str(continuation.get("operational_change", "")).rstrip("; ")
            + "; 2026-09-04 four Countdown continuations resumed with a "
            "two-hour evaluation-safe watchdog on node105/mltheory"
        ).lstrip("; ")
        continuation["released"] = False
        audit = {
            "schema": "e119-node105-watchdog-recovery-v1",
            "cause": "45-minute stale watchdog fired during long evaluation",
            "same_scientific_cells": True,
            "same_run_directories": True,
            "automatic_checkpoint_resume": True,
            "optimizer_update_changed": False,
            "treatment_changed": False,
            "outcomes_inspected": False,
            "placement": "node105/mltheory",
            "watchdog_stale_seconds": 7200,
            "pvl_exclusion": PVL,
            "released": False,
            "replacements": plans,
        }
        atomic(CONTINUATIONS, continuation)
        atomic(AUDIT, audit)
        committed = True

        for job_id in submitted:
            subprocess.run(["scontrol", "release", str(job_id)], check=True)
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
        "recovered E119 node105 watchdog continuations: "
        + ",".join(str(job_id) for job_id in submitted)
    )


if __name__ == "__main__":
    main()
