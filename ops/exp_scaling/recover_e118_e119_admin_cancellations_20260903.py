#!/usr/bin/env python3
"""Replace the 2026-09-03 root-cancelled E118/E119 allocations.

The replacement keeps each scientific cell and run directory unchanged.  The
standard runner's fail-closed checkpoint selector resumes the newest coherent
checkpoint when one exists; zero-step cancellations restart from initialization.
All replacements are submitted held, audited for the full PVL exclusion, made
durable in their lineage ledgers, and only then released.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import status_e78 as shared
import launch_e118_maxrl_verified_replay_factorial as e118f
import launch_e118q3_qwen3b_maxrl_extension as e118q3
import launch_e119_level2_qwen05b_factorial as e119


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
AUDIT = Path(
    os.environ.get(
        "E118_E119_RECOVERY_AUDIT",
        str(ARTIFACTS / "e118_e119_failure_recovery_r3_20260903.json"),
    )
)
RECOVERY_REASON = os.environ.get(
    "E118_E119_RECOVERY_REASON",
    "Slurm jobs were cancelled by uid 0 on 2026-09-03",
)
RELEASE_REPLACEMENTS = os.environ.get(
    "E118_E119_RECOVERY_RELEASE", "1"
).strip().lower() not in {"0", "false", "no"}
E119_CONT = ARTIFACTS / "e119_level2_continuation_jobs.json"
FAILURES = {
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "NOT_IN_QUEUE"
}


def atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def state(job_id: int) -> str:
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", str(job_id), "-o", "State", "-P"],
        capture_output=True, text=True, check=True,
    )
    rows = [line.split("|", 1)[0].strip().split()[0]
            for line in result.stdout.splitlines() if line.strip()]
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


def submit(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = int(result.stdout.strip().split(";", 1)[0])
    shown = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        capture_output=True, text=True, check=True,
    ).stdout
    required = ("JobState=PENDING", "Reason=JobHeldUser", f"ExcNodeList={PVL}",
                "OAT_ZERO_AUTO_RESUME=1")
    missing = [token for token in required if token not in shown]
    if missing:
        subprocess.run(["scancel", str(job_id)], check=False)
        raise RuntimeError(f"replacement {job_id} missing {missing}")
    return job_id, shown


def recover_e118() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    definitions = (
        ("falcon1b", ARTIFACTS / "e118f1_maxrl_verified_replay_extension_jobs.json", e118f),
        ("qwen3b", ARTIFACTS / "e118q3_maxrl_verified_replay_extension_jobs.json", e118q3),
    )
    for scale, ledger_path, launcher in definitions:
        ledger = json.loads(ledger_path.read_text())
        snapshot = Path(ledger["snapshot_root"])
        if scale == "falcon1b":
            templates = {(str(x["domain"]), int(x["seed"])): x
                         for x in launcher.source_runs(ROOT)}
        else:
            templates = {(str(x["domain"]), int(x["seed"])): x
                         for x in launcher.e80.references(ROOT)}
            model = launcher.e80.model_root(ROOT)
        changed = False
        for run in ledger["runs"]:
            old_id = int(run["job_id"])
            run_dir = Path(str(run["run_dir"]))
            target_steps = int(ledger["target_steps"])
            progress = min(shared.run_step(run_dir), target_steps)
            if shared.is_complete(run_dir, progress, target_steps):
                continue
            old_state = state(old_id)
            if old_state not in FAILURES:
                continue
            template = templates[(str(run["domain"]), int(run["seed"]))]
            if scale == "falcon1b":
                env, target = launcher.environment(ROOT, template, str(run["arm"]), snapshot)
                command = launcher.command(ROOT, template, str(run["arm"]), env)
            else:
                env, target = launcher.environment(ROOT, template, str(run["arm"]), snapshot, model)
                command = launcher.command(ROOT, template, str(run["arm"]), env)
            if str(target) != str(run["run_dir"]):
                raise RuntimeError(f"run-directory drift for {old_id}")
            new_id, shown = submit(add_exclusion(command))
            run["previous_job_ids"] = [*run.get("previous_job_ids", []), old_id]
            run["job_id"] = new_id
            run["held_scheduler_record"] = shown
            records.append({"cohort": "e118", "scale": scale,
                            "domain": run["domain"], "arm": run["arm"],
                            "seed": int(run["seed"]), "old_job_id": old_id,
                            "old_state": old_state, "new_job_id": new_id,
                            "run_dir": run["run_dir"]})
            changed = True
        if changed:
            atomic(ledger_path, ledger)
    return records


def recover_e119() -> list[dict[str, Any]]:
    ledger_path = ROOT / e119.LEDGER
    ledger = json.loads(ledger_path.read_text())
    continuation = json.loads(E119_CONT.read_text())
    by_original = {int(row["original_job_id"]): row
                   for row in continuation["continuations"]}
    templates = {(str(x["domain"]), int(x["seed"])): x for x in e119.templates(ROOT)}
    snapshot = Path(ledger["snapshot_root"])
    records: list[dict[str, Any]] = []
    for run in ledger["runs"]:
        original = int(run["job_id"])
        run_dir = Path(str(run["run_dir"]))
        target_steps = int(ledger["target_steps"])
        progress = min(shared.run_step(run_dir), target_steps)
        if shared.is_complete(run_dir, progress, target_steps):
            continue
        prior = by_original.get(original)
        old_id = int(prior["continuation_job_id"]) if prior else original
        old_state = state(old_id)
        if old_state not in FAILURES:
            continue
        template = templates[(str(run["domain"]), int(run["seed"]))]
        env, target = e119.environment(ROOT, template, str(run["arm"]), snapshot)
        if str(target) != str(run["run_dir"]):
            raise RuntimeError(f"run-directory drift for {old_id}")
        new_id, _shown = submit(safe_e119(e119.command(ROOT, template, str(run["arm"]), env)))
        lineage = {
            "original_job_id": original, "continuation_job_id": new_id,
            "domain": run["domain"], "arm": run["arm"], "seed": int(run["seed"]),
            "run_dir": run["run_dir"], "run_stamp": run["run_stamp"],
            "repair_kind": "admin_cancellation_auto_resume",
            "previous_continuation_job_ids": [
                *([] if not prior else prior.get("previous_continuation_job_ids", [])),
                *([] if old_id == original else [old_id]),
            ],
        }
        by_original[original] = lineage
        records.append({"cohort": "e119", "domain": run["domain"],
                        "arm": run["arm"], "seed": int(run["seed"]),
                        "old_job_id": old_id, "old_state": old_state,
                        "new_job_id": new_id, "run_dir": run["run_dir"]})
    continuation["continuations"] = list(by_original.values())
    continuation["operational_change"] = (
        "evaluation-safe watchdog, Level-2 Pantry repair, and "
        "2026-09-03 administrative-cancellation recovery"
    )
    continuation["released"] = False
    atomic(E119_CONT, continuation)
    return records


def main() -> None:
    if AUDIT.exists():
        raise SystemExit(f"recovery already installed: {AUDIT}")
    submitted: list[int] = []
    rollback_paths = (
        ARTIFACTS / "e118f1_maxrl_verified_replay_extension_jobs.json",
        ARTIFACTS / "e118q3_maxrl_verified_replay_extension_jobs.json",
        E119_CONT,
    )
    ledger_backups = {path: path.read_bytes() for path in rollback_paths}
    try:
        records = recover_e118() + recover_e119()
        submitted = [int(row["new_job_id"]) for row in records]
        audit = {
            "schema": "e118-e119-admin-cancellation-recovery-v1",
            "reason": RECOVERY_REASON,
            "same_scientific_cells": True, "same_run_directories": True,
            "automatic_checkpoint_resume": True, "pvl_exclusion": PVL,
            "released": False, "replacements": records,
        }
        atomic(AUDIT, audit)
        if RELEASE_REPLACEMENTS:
            for job_id in submitted:
                subprocess.run(["scontrol", "release", str(job_id)], check=True)
        if RELEASE_REPLACEMENTS and E119_CONT.is_file():
            continuation = json.loads(E119_CONT.read_text())
            continuation["released"] = True
            atomic(E119_CONT, continuation)
        audit["released"] = RELEASE_REPLACEMENTS
        atomic(AUDIT, audit)
    except Exception:
        if submitted:
            subprocess.run(["scancel", *map(str, submitted)], check=False)
        for path, payload in ledger_backups.items():
            temporary = path.with_suffix(path.suffix + ".rollback")
            temporary.write_bytes(payload)
            temporary.replace(path)
        subprocess.run(
            ["python", str(ROOT / "ops/exp_scaling/build_e118_aggregate_ledger.py")],
            check=False,
        )
        raise
    subprocess.run(["python", str(ROOT / "ops/exp_scaling/build_e118_aggregate_ledger.py")], check=True)
    print(f"recovered {len(records)} cells: E118={sum(r['cohort']=='e118' for r in records)} "
          f"E119={sum(r['cohort']=='e119' for r in records)}")


if __name__ == "__main__":
    main()
