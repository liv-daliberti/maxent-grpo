#!/usr/bin/env python3
"""Recover one E118 timeout and one E119 exhausted-watchdog cell."""

from __future__ import annotations

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
AUDIT = ARTIFACTS / "e118_e119_targeted_failure_recovery_20260903.json"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
E118_OLD = 31010906
E119_ORIGINAL = 31014398


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


def safe_e119(command: list[str]) -> list[str]:
    replacements = {
        "--partition=mltheory": "--partition=all",
        "--account=mltheory": "--account=allcs",
        "--nodelist=node105": "--nodelist=node203,node204,node205,node207",
        "--gres=gpu:a5000:1": "--gres=gpu:1",
        "--mem=64G": "--mem=40G",
    }
    return add_exclusion([replacements.get(part, part) for part in command])


def submit(command: list[str], run_stamp: str) -> tuple[int, str]:
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
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Account=allcs",
        f"ExcNodeList={PVL}",
        "OAT_ZERO_AUTO_RESUME=1",
        f"RUN_STAMP={run_stamp}",
    )
    missing = [token for token in required if token not in shown]
    if missing:
        raise RuntimeError(f"replacement {job_id} missing {missing}")
    return job_id, shown


def main() -> None:
    if AUDIT.exists():
        raise SystemExit(f"recovery already installed: {AUDIT}")

    e118_payload = json.loads(E118_LEDGER.read_text(encoding="utf-8"))
    e119_payload = json.loads(E119_LEDGER.read_text(encoding="utf-8"))
    continuation = json.loads(E119_CONTINUATIONS.read_text(encoding="utf-8"))

    e118_runs = [run for run in e118_payload["runs"] if int(run["job_id"]) == E118_OLD]
    if len(e118_runs) != 1 or state(E118_OLD) != "TIMEOUT":
        raise RuntimeError("E118 failure identity or state drifted")
    e118_run = e118_runs[0]
    if (str(e118_run["domain"]), str(e118_run["arm"]), int(e118_run["seed"])) != (
        "python_factors",
        "maxrl",
        72,
    ):
        raise RuntimeError("E118 cell identity drifted")
    e118_checkpoint = (
        Path(e118_run["run_dir"])
        / f"debug_job{E118_OLD}/checkpoints/step_01920"
    )
    if not e118_checkpoint.is_dir():
        raise RuntimeError(f"missing E118 checkpoint: {e118_checkpoint}")

    e119_runs = [
        run for run in e119_payload["runs"] if int(run["job_id"]) == E119_ORIGINAL
    ]
    if len(e119_runs) != 1:
        raise RuntimeError("E119 original cell identity drifted")
    e119_run = e119_runs[0]
    prior_by_original = {
        int(row["original_job_id"]): row
        for row in continuation["continuations"]
    }
    prior = prior_by_original.get(E119_ORIGINAL)
    e119_old = int(prior["continuation_job_id"]) if prior else E119_ORIGINAL
    if state(e119_old) != "FAILED":
        raise RuntimeError(f"E119 effective job {e119_old} is not FAILED")
    if (str(e119_run["domain"]), str(e119_run["arm"]), int(e119_run["seed"])) != (
        "countdown",
        "drgrpo",
        43,
    ):
        raise RuntimeError("E119 cell identity drifted")
    e119_checkpoint = (
        Path(e119_run["run_dir"])
        / f"debug_job{e119_old}/checkpoints/step_01344"
    )
    if not e119_checkpoint.is_dir():
        raise RuntimeError(f"missing E119 checkpoint: {e119_checkpoint}")

    submitted: list[int] = []
    try:
        e118_templates = {
            (str(row["domain"]), int(row["seed"])): row
            for row in e118q3.e80.references(ROOT)
        }
        e118_template = e118_templates[("python_factors", 72)]
        e118_snapshot = Path(e118_payload["snapshot_root"])
        e118_model = e118q3.e80.model_root(ROOT)
        e118_env, e118_target = e118q3.environment(
            ROOT, e118_template, "maxrl", e118_snapshot, e118_model
        )
        if str(e118_target) != str(e118_run["run_dir"]):
            raise RuntimeError("E118 run-directory drift")
        e118_new, e118_shown = submit(
            add_exclusion(
                e118q3.command(ROOT, e118_template, "maxrl", e118_env)
            ),
            str(e118_run["run_stamp"]),
        )
        submitted.append(e118_new)

        e119_templates = {
            (str(row["domain"]), int(row["seed"])): row
            for row in e119.templates(ROOT)
        }
        e119_template = e119_templates[("countdown", 43)]
        e119_snapshot = Path(e119_payload["snapshot_root"])
        e119_env, e119_target = e119.environment(
            ROOT, e119_template, "drgrpo", e119_snapshot
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
        e119_new, e119_shown = submit(
            safe_e119(e119.command(ROOT, e119_template, "drgrpo", e119_env)),
            str(e119_run["run_stamp"]),
        )
        submitted.append(e119_new)
        for token in (
            "OAT_ZERO_WATCHDOG_STALE_SECONDS=7200",
            "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600",
            "OAT_ZERO_WATCHDOG_MAX_RESTARTS=12",
        ):
            if token not in e119_shown:
                raise RuntimeError(f"E119 replacement missing {token}")

        e118_run["previous_job_ids"] = [
            *e118_run.get("previous_job_ids", []),
            E118_OLD,
        ]
        e118_run["job_id"] = e118_new
        e118_run["held_scheduler_record"] = e118_shown

        prior_by_original[E119_ORIGINAL] = {
            "original_job_id": E119_ORIGINAL,
            "continuation_job_id": e119_new,
            "domain": e119_run["domain"],
            "arm": e119_run["arm"],
            "seed": int(e119_run["seed"]),
            "run_dir": e119_run["run_dir"],
            "run_stamp": e119_run["run_stamp"],
            "repair_kind": "evaluation_watchdog_continuation",
            "previous_continuation_job_ids": [
                *([] if not prior else prior.get("previous_continuation_job_ids", [])),
                *([] if e119_old == E119_ORIGINAL else [e119_old]),
            ],
        }
        continuation["continuations"] = list(prior_by_original.values())
        continuation["operational_change"] = (
            str(continuation.get("operational_change", ""))
            + "; targeted 2026-09-03 evaluation-watchdog recovery"
        ).lstrip("; ")
        continuation["released"] = False

        records = [
            {
                "cohort": "e118",
                "scale": "qwen3b",
                "domain": "python_factors",
                "arm": "maxrl",
                "seed": 72,
                "old_job_id": E118_OLD,
                "old_state": "TIMEOUT",
                "new_job_id": e118_new,
                "resume_checkpoint": str(e118_checkpoint),
                "run_dir": e118_run["run_dir"],
            },
            {
                "cohort": "e119",
                "scale": "qwen05b",
                "domain": "countdown",
                "arm": "drgrpo",
                "seed": 43,
                "old_job_id": e119_old,
                "old_state": "FAILED",
                "new_job_id": e119_new,
                "resume_checkpoint": str(e119_checkpoint),
                "run_dir": e119_run["run_dir"],
            },
        ]
        audit = {
            "schema": "e118-e119-targeted-failure-recovery-v1",
            "same_scientific_cells": True,
            "same_run_directories": True,
            "automatic_checkpoint_resume": True,
            "outcomes_inspected": False,
            "pvl_exclusion": PVL,
            "released": False,
            "replacements": records,
        }

        atomic(E118_LEDGER, e118_payload)
        atomic(E119_CONTINUATIONS, continuation)
        atomic(AUDIT, audit)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", str(job_id)], check=True)
        continuation["released"] = True
        audit["released"] = True
        atomic(E119_CONTINUATIONS, continuation)
        atomic(AUDIT, audit)
    except Exception:
        if submitted:
            subprocess.run(["scancel", *map(str, submitted)], check=False)
        raise

    subprocess.run(
        ["python", str(ROOT / "ops/exp_scaling/build_e118_aggregate_ledger.py")],
        check=True,
    )
    print(f"recovered E118={e118_new} E119={e119_new}")


if __name__ == "__main__":
    main()
