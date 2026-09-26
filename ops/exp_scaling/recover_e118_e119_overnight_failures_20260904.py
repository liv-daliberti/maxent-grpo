#!/usr/bin/env python3
"""Continue the six E118/E119 cells interrupted overnight on 2026-09-04.

The five E118 Qwen-3B Python cells resume their step-1920 checkpoints on the
safe allcs A6000/A100 pool.  The E119 Countdown Dr.GRPO cell resumes its
step-1728 checkpoint on node105/mltheory with an evaluation-safe watchdog.
Scientific cells, treatments, run directories, and optimizer settings remain
unchanged.  Every replacement is submitted held, audited, recorded durably,
and only then released.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import launch_e118q3_qwen3b_maxrl_extension as e118q3
import launch_e119_level2_qwen05b_factorial as e119


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"
E118_LEDGER = ROOT / e118q3.LEDGER
E119_LEDGER = ROOT / e119.LEDGER
E119_CONTINUATIONS = ARTIFACTS / "e119_level2_continuation_jobs.json"
AUDIT = ARTIFACTS / "e118_e119_overnight_failure_recovery_20260904.json"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
E118_FAILURES = {
    31010907: ("python_factors", "replay_maxrl", 72),
    31010908: ("python_factors", "maxrl", 73),
    31010909: ("python_factors", "replay_maxrl", 73),
    31010910: ("python_factors", "maxrl", 74),
    31010911: ("python_factors", "replay_maxrl", 74),
}
E119_FAILURE = 31014402


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


def add_exclusion(command: list[str]) -> list[str]:
    return [*command[:-1], f"--exclude={PVL}", command[-1]]


def node105_e119(command: list[str]) -> list[str]:
    replacements = {"--mem=64G": "--mem=40G"}
    return add_exclusion([replacements.get(part, part) for part in command])


def submit_held(
    command: list[str],
    *,
    run_stamp: str,
    required: tuple[str, ...],
) -> tuple[int, str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = int(result.stdout.strip().split(";", 1)[0])
    shown = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    common = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"ExcNodeList={PVL}",
        "OAT_ZERO_AUTO_RESUME=1",
        f"RUN_STAMP={run_stamp}",
    )
    missing = [token for token in (*common, *required) if token not in shown]
    if missing:
        raise RuntimeError(f"replacement {job_id} missing {missing}")
    return job_id, shown


def main() -> None:
    if AUDIT.exists():
        raise SystemExit(f"recovery already installed: {AUDIT}")

    e118_payload = json.loads(E118_LEDGER.read_text(encoding="utf-8"))
    e119_payload = json.loads(E119_LEDGER.read_text(encoding="utf-8"))
    continuation = json.loads(E119_CONTINUATIONS.read_text(encoding="utf-8"))
    e118_before = json.loads(json.dumps(e118_payload))
    continuation_before = json.loads(json.dumps(continuation))

    e118_by_id = {int(run["job_id"]): run for run in e118_payload["runs"]}
    if not set(E118_FAILURES).issubset(e118_by_id):
        raise RuntimeError("E118 failure identities drifted from the source ledger")
    for old_id, identity in E118_FAILURES.items():
        run = e118_by_id[old_id]
        observed = (str(run["domain"]), str(run["arm"]), int(run["seed"]))
        if observed != identity or state(old_id) != "TIMEOUT":
            raise RuntimeError(f"E118 cell {old_id} identity/state drift: {observed}")
        checkpoint = (
            Path(run["run_dir"])
            / f"debug_job{old_id}/checkpoints/step_01920"
        )
        if not checkpoint.is_dir():
            raise RuntimeError(f"missing E118 checkpoint: {checkpoint}")

    e119_run = next(
        run for run in e119_payload["runs"] if int(run["job_id"]) == E119_FAILURE
    )
    if (
        str(e119_run["domain"]),
        str(e119_run["arm"]),
        int(e119_run["seed"]),
    ) != ("countdown", "drgrpo", 44):
        raise RuntimeError("E119 failure identity drifted")
    existing_e119 = {
        int(row["original_job_id"]): row
        for row in continuation["continuations"]
    }
    if E119_FAILURE in existing_e119:
        raise RuntimeError("E119 failure already has a registered continuation")
    if state(E119_FAILURE) != "FAILED":
        raise RuntimeError(f"E119 cell {E119_FAILURE} is not FAILED")
    e119_checkpoint = (
        Path(e119_run["run_dir"])
        / f"debug_job{E119_FAILURE}/checkpoints/step_01728"
    )
    if not e119_checkpoint.is_dir():
        raise RuntimeError(f"missing E119 checkpoint: {e119_checkpoint}")

    e118_templates = {
        (str(row["domain"]), int(row["seed"])): row
        for row in e118q3.e80.references(ROOT)
    }
    e118_snapshot = Path(e118_payload["snapshot_root"])
    e118_model = e118q3.e80.model_root(ROOT)
    e119_template = next(
        row
        for row in e119.templates(ROOT)
        if str(row["domain"]) == "countdown" and int(row["seed"]) == 44
    )
    e119_snapshot = Path(e119_payload["snapshot_root"])

    submitted: list[int] = []
    records: list[dict[str, Any]] = []
    ledgers_written = False
    try:
        for old_id in sorted(E118_FAILURES):
            run = e118_by_id[old_id]
            template = e118_templates[(str(run["domain"]), int(run["seed"]))]
            env, target = e118q3.environment(
                ROOT,
                template,
                str(run["arm"]),
                e118_snapshot,
                e118_model,
            )
            if str(target) != str(run["run_dir"]):
                raise RuntimeError(f"E118 run-directory drift for {old_id}")
            job_id, shown = submit_held(
                add_exclusion(
                    e118q3.command(ROOT, template, str(run["arm"]), env)
                ),
                run_stamp=str(run["run_stamp"]),
                required=("Account=allcs", "MinMemoryNode=128G"),
            )
            submitted.append(job_id)
            records.append(
                {
                    "cohort": "e118",
                    "scale": "qwen3b",
                    "domain": run["domain"],
                    "arm": run["arm"],
                    "seed": int(run["seed"]),
                    "old_job_id": old_id,
                    "old_state": "TIMEOUT",
                    "new_job_id": job_id,
                    "resume_checkpoint": str(
                        Path(run["run_dir"])
                        / f"debug_job{old_id}/checkpoints/step_01920"
                    ),
                    "run_dir": run["run_dir"],
                }
            )
            run["previous_job_ids"] = [
                *run.get("previous_job_ids", []),
                old_id,
            ]
            run["job_id"] = job_id
            run["held_scheduler_record"] = shown

        e119_env, e119_target = e119.environment(
            ROOT,
            e119_template,
            str(e119_run["arm"]),
            e119_snapshot,
        )
        if str(e119_target) != str(e119_run["run_dir"]):
            raise RuntimeError("E119 run-directory drift")
        e119_env.update(
            {
                "OAT_ZERO_WATCHDOG_STALE_SECONDS": "7200",
                "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS": "3600",
                "OAT_ZERO_WATCHDOG_MAX_RESTARTS": "12",
            }
        )
        e119_new, _shown = submit_held(
            node105_e119(
                e119.command(
                    ROOT,
                    e119_template,
                    str(e119_run["arm"]),
                    e119_env,
                )
            ),
            run_stamp=str(e119_run["run_stamp"]),
            required=(
                "Account=mltheory",
                "Partition=mltheory",
                "ReqNodeList=node105",
                "MinMemoryNode=40G",
                "OAT_ZERO_WATCHDOG_STALE_SECONDS=7200",
                "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600",
                "OAT_ZERO_WATCHDOG_MAX_RESTARTS=12",
            ),
        )
        submitted.append(e119_new)
        e119_record = {
            "original_job_id": E119_FAILURE,
            "continuation_job_id": e119_new,
            "domain": e119_run["domain"],
            "arm": e119_run["arm"],
            "seed": int(e119_run["seed"]),
            "run_dir": e119_run["run_dir"],
            "run_stamp": e119_run["run_stamp"],
            "repair_kind": "evaluation_watchdog_node105_continuation",
            "previous_continuation_job_ids": [],
        }
        existing_e119[E119_FAILURE] = e119_record
        continuation["continuations"] = list(existing_e119.values())
        continuation["operational_change"] = (
            str(continuation.get("operational_change", "")).rstrip("; ")
            + "; 2026-09-04 Countdown s44 evaluation-watchdog continuation "
            "on node105/mltheory"
        ).lstrip("; ")
        continuation["released"] = False
        records.append(
            {
                "cohort": "e119",
                "scale": "qwen05b",
                "domain": e119_run["domain"],
                "arm": e119_run["arm"],
                "seed": int(e119_run["seed"]),
                "old_job_id": E119_FAILURE,
                "old_state": "FAILED",
                "new_job_id": e119_new,
                "resume_checkpoint": str(e119_checkpoint),
                "run_dir": e119_run["run_dir"],
            }
        )

        audit = {
            "schema": "e118-e119-overnight-failure-recovery-v1",
            "same_scientific_cells": True,
            "same_run_directories": True,
            "automatic_checkpoint_resume": True,
            "optimizer_update_changed": False,
            "treatment_changed": False,
            "outcomes_inspected": False,
            "pvl_exclusion": PVL,
            "e118_placement": "allcs safe-node pool",
            "e119_placement": "node105/mltheory",
            "released": False,
            "replacements": records,
        }
        atomic(E118_LEDGER, e118_payload)
        atomic(E119_CONTINUATIONS, continuation)
        atomic(AUDIT, audit)
        ledgers_written = True

        for job_id in submitted:
            subprocess.run(["scontrol", "release", str(job_id)], check=True)
        continuation["released"] = True
        audit["released"] = True
        atomic(E119_CONTINUATIONS, continuation)
        atomic(AUDIT, audit)
    except Exception:
        if submitted:
            subprocess.run(
                ["scancel", *map(str, submitted)],
                check=False,
            )
        if ledgers_written:
            atomic(E118_LEDGER, e118_before)
            atomic(E119_CONTINUATIONS, continuation_before)
            AUDIT.unlink(missing_ok=True)
        raise

    subprocess.run(
        [
            "python",
            str(ROOT / "ops/exp_scaling/build_e118_aggregate_ledger.py"),
        ],
        check=True,
    )
    print(
        "recovered 6 cells: "
        f"E118={','.join(str(row['new_job_id']) for row in records if row['cohort'] == 'e118')} "
        f"E119={e119_new}"
    )


if __name__ == "__main__":
    main()
