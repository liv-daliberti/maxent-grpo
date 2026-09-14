#!/usr/bin/env python3
"""Install six scheduler-only E118 priority replacements for 2026-09-04."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import launch_e118q3_qwen3b_maxrl_extension as launch


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"
LEDGER = ROOT / launch.LEDGER
E120_LEDGER = ARTIFACTS / "e120r1_frequency_weighted_replay_jobs.json"
AUDIT = ARTIFACTS / "e118_400k_priority_replacements_20260904.json"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
# The original draft targeted 31010923, but that cell started on node205 before
# this amendment was installed. Preserve that useful allocation and promote
# the next three still-pending cells to node302 instead.
NODE302_OLD = (31010924, 31010925, 31010926)
NODE208_OLD = (31010927, 31010928, 31010930)
E120_QWEN3 = tuple(range(31033705, 31033715))
INACTIVE_E120_STATES = {
    "CANCELLED",
    "FAILED",
    "NODE_FAIL",
    "NOT_IN_QUEUE",
    "OUT_OF_MEMORY",
    "TIMEOUT",
}


def command_result(
    command: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=check)


def state(job_id: int) -> str:
    result = command_result(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"], check=False
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0].split()[0].upper()
    result = command_result(
        ["sacct", "-n", "-X", "-j", str(job_id), "-o", "State", "-P"]
    )
    rows = [line.split("|", 1)[0].strip().split()[0]
            for line in result.stdout.splitlines() if line.strip()]
    return rows[0] if rows else "NOT_IN_QUEUE"


def show(job_id: int) -> str:
    return command_result(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)]
    ).stdout


def atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def placed_command(
    base: list[str], *, account: str, partition: str, node: str
) -> list[str]:
    replacements = {
        "--account=allcs": f"--account={account}",
        "--partition=all": f"--partition={partition}",
        f"--nodelist={launch.NODES}": f"--nodelist={node}",
    }
    command = [replacements.get(part, part) for part in base]
    return [*command[:-1], f"--exclude={PVL}", command[-1]]


def submit_held(
    command: list[str], *, run_stamp: str, account: str, partition: str, node: str
) -> tuple[int, str]:
    result = command_result(command, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = int(result.stdout.strip().split(";", 1)[0])
    record = show(job_id)
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"Account={account}",
        # This cluster reports the public all alias internally as cs.
        f"Partition={'cs' if partition == 'all' else partition}",
        f"ReqNodeList={node}",
        f"ExcNodeList={PVL}",
        "MinMemoryNode=128G",
        "OAT_ZERO_AUTO_RESUME=1",
        f"RUN_STAMP={run_stamp}",
    )
    missing = [token for token in required if token not in record]
    if missing:
        command_result(["scancel", str(job_id)], check=False)
        raise RuntimeError(f"replacement {job_id} missing {missing}")
    return job_id, record


def main() -> None:
    if AUDIT.exists():
        raise SystemExit(f"priority replacements already installed: {AUDIT}")

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    ledger_before = json.loads(json.dumps(ledger))
    by_id = {int(row["job_id"]): row for row in ledger["runs"]}
    expected = {
        31010924: ("pantry_plan", "maxrl", 71),
        31010925: ("pantry_plan", "replay_maxrl", 71),
        31010926: ("pantry_plan", "maxrl", 72),
        31010927: ("pantry_plan", "replay_maxrl", 72),
        31010928: ("pantry_plan", "maxrl", 73),
        31010930: ("pantry_plan", "maxrl", 74),
    }
    old_states: dict[int, str] = {}
    for job_id, identity in expected.items():
        row = by_id.get(job_id)
        observed = None if row is None else (
            str(row["domain"]), str(row["arm"]), int(row["seed"])
        )
        old_state = state(job_id)
        old_states[job_id] = old_state
        allowed = {"PENDING", *INACTIVE_E120_STATES}
        if observed != identity or old_state not in allowed:
            raise RuntimeError(
                f"E118 target drift for {job_id}: {observed}, state={old_state}"
            )

    e120 = json.loads(E120_LEDGER.read_text(encoding="utf-8"))
    e120_by_id = {int(row["job_id"]): row for row in e120["runs"]}
    for job_id in E120_QWEN3:
        if str(e120_by_id.get(job_id, {}).get("model_key")) != "qwen3b":
            raise RuntimeError(f"E120 identity drift for {job_id}")

    templates = {
        (str(row["domain"]), int(row["seed"])): row
        for row in launch.e80.references(ROOT)
    }
    snapshot = Path(ledger["snapshot_root"])
    model = launch.e80.model_root(ROOT)
    submitted: list[int] = []
    records: list[dict[str, Any]] = []
    paused: list[int] = []
    already_inactive: list[int] = []
    discarded_precheckpoint_steps: dict[str, int] = {}
    ledger_written = False
    try:
        for old_id in (*NODE302_OLD, *NODE208_OLD):
            old = by_id[old_id]
            template = templates[(str(old["domain"]), int(old["seed"]))]
            env, target = launch.environment(
                ROOT, template, str(old["arm"]), snapshot, model
            )
            if str(target) != str(old["run_dir"]):
                raise RuntimeError(f"run-directory drift for {old_id}")
            if old_id in NODE302_OLD:
                account, partition, node = "mltheory", "mltheory", "node302"
            else:
                account, partition, node = "allcs", "all", "node208"
            new_id, scheduler_record = submit_held(
                placed_command(
                    launch.command(ROOT, template, str(old["arm"]), env),
                    account=account,
                    partition=partition,
                    node=node,
                ),
                run_stamp=str(old["run_stamp"]),
                account=account,
                partition=partition,
                node=node,
            )
            submitted.append(new_id)
            records.append(
                {
                    "old_job_id": old_id,
                    "new_job_id": new_id,
                    "domain": old["domain"],
                    "arm": old["arm"],
                    "seed": int(old["seed"]),
                    "run_dir": old["run_dir"],
                    "run_stamp": old["run_stamp"],
                    "account": account,
                    "partition": partition,
                    "node": node,
                    "held_scheduler_record": scheduler_record,
                }
            )

        # Keep every E120 Qwen-3B cell from refilling node302. The first three
        # may already be completing a checkpoint requeue from the fail-closed
        # account-change attempt; requeuehold is idempotent for this purpose.
        for job_id in E120_QWEN3:
            current = state(job_id)
            if current == "RUNNING":
                row = e120_by_id[job_id]
                run_dir = Path(str(row["run_dir"]))
                # These cells have not reached their first step-192 durable
                # checkpoint. Record the exact work sacrificed for immediate
                # E118 priority instead of calling it a checkpoint resume.
                metrics = sorted(run_dir.glob("debug_job*/train_metrics.jsonl"))
                latest = 0
                for path in metrics:
                    for line in path.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines():
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        for key in ("step", "global_step", "query_step"):
                            value = payload.get(key)
                            if isinstance(value, (int, float)):
                                latest = max(latest, int(value))
                discarded_precheckpoint_steps[str(job_id)] = latest
                command_result(["scontrol", "requeuehold", str(job_id)])
            elif current == "PENDING":
                command_result(["scontrol", "hold", str(job_id)])
            elif current == "COMPLETING":
                continue
            elif current in INACTIVE_E120_STATES:
                already_inactive.append(job_id)
                continue
            else:
                raise RuntimeError(f"cannot safely pause E120 job {job_id}: {current}")
            paused.append(job_id)

        for replacement in records:
            old = by_id[int(replacement["old_job_id"])]
            old["previous_job_ids"] = [
                *old.get("previous_job_ids", []),
                int(replacement["old_job_id"]),
            ]
            old["job_id"] = int(replacement["new_job_id"])
            old["held_scheduler_record"] = replacement["held_scheduler_record"]

        audit = {
            "schema": "e118-400k-priority-replacements-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "objective": "maximize E118 realized optimizer steps toward 400000 on 2026-09-04",
            "same_scientific_cells": True,
            "same_run_directories": True,
            "optimizer_update_changed": False,
            "treatment_changed": False,
            "pvl_compute_allowed": False,
            "pvl_exclusion": PVL,
            "e120_qwen3_paused_job_ids": list(E120_QWEN3),
            "e120_qwen3_already_inactive_job_ids": already_inactive,
            "e120_discarded_precheckpoint_steps": discarded_precheckpoint_steps,
            "replacements": [
                {key: value for key, value in row.items()
                 if key != "held_scheduler_record"}
                for row in records
            ],
            "old_job_states": {str(key): value for key, value in old_states.items()},
            "old_jobs_cancelled": [],
            "released": False,
            "installed": True,
            "outcomes_inspected": False,
        }
        atomic(LEDGER, ledger)
        atomic(AUDIT, audit)
        ledger_written = True
        cancellable = [
            job_id
            for job_id, old_state in old_states.items()
            if old_state in {"PENDING", "RUNNING", "COMPLETING"}
        ]
        if cancellable:
            command_result(["scancel", *map(str, cancellable)])
        audit["old_jobs_cancelled"] = cancellable
        atomic(AUDIT, audit)
        command_result(
            [
                "python",
                str(ROOT / "ops/exp_scaling/build_e118_aggregate_ledger.py"),
            ]
        )
        for job_id in submitted:
            command_result(["scontrol", "release", str(job_id)])
        audit["released"] = True
        atomic(AUDIT, audit)
    except Exception:
        if submitted:
            command_result(["scancel", *map(str, submitted)], check=False)
        if ledger_written:
            atomic(LEDGER, ledger_before)
            AUDIT.unlink(missing_ok=True)
            command_result(
                [
                    "python",
                    str(ROOT / "ops/exp_scaling/build_e118_aggregate_ledger.py"),
                ],
                check=False,
            )
        for job_id in paused:
            command_result(["scontrol", "release", str(job_id)], check=False)
        raise

    print(
        "E118 priority replacements installed: "
        + ",".join(
            f"{row['old_job_id']}->{row['new_job_id']}@{row['node']}"
            for row in records
        )
    )


if __name__ == "__main__":
    main()
