#!/usr/bin/env python3
"""Give E118 first claim on the user's safe Qwen-3B capacity for 2026-09-04.

This scheduler-only amendment pauses the ten E120-R1 Qwen-3B cells (running
cells are requeued-held from their durable checkpoints), assigns three oldest
pending E118 Pantry cells to node302/mltheory, and assigns three more to the
idle safe A6000 node208 through the all partition.  No scientific treatment,
run directory, checkpoint, or registered cell identity changes.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"
E118_LEDGER = ARTIFACTS / "e118_all_scales_maxrl_verified_replay_jobs.json"
E120_LEDGER = ARTIFACTS / "e120r1_frequency_weighted_replay_jobs.json"
AUDIT = ARTIFACTS / "e118_400k_priority_amendment_20260904.json"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
E118_NODE302 = (31010923, 31010924, 31010925)
E118_NODE208 = (31010926, 31010927, 31010928)
E120_QWEN3 = tuple(range(31033705, 31033715))


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=check)


def state(job_id: int) -> str:
    result = run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0].split()[0].upper()
    result = run(
        ["sacct", "-n", "-X", "-j", str(job_id), "-o", "State", "-P"]
    )
    rows = [line.split("|", 1)[0].strip().split()[0]
            for line in result.stdout.splitlines() if line.strip()]
    return rows[0] if rows else "NOT_IN_QUEUE"


def show(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout


def field(record: str, name: str) -> str | None:
    prefix = f"{name}="
    for token in record.split():
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    if AUDIT.exists():
        raise SystemExit(f"priority amendment already installed: {AUDIT}")

    e118 = json.loads(E118_LEDGER.read_text(encoding="utf-8"))
    e120 = json.loads(E120_LEDGER.read_text(encoding="utf-8"))
    e118_runs = {int(row["job_id"]): row for row in e118["runs"]}
    e120_runs = {int(row["job_id"]): row for row in e120["runs"]}

    expected_e118 = {
        31010923: ("pantry_plan", "replay_maxrl", 70),
        31010924: ("pantry_plan", "maxrl", 71),
        31010925: ("pantry_plan", "replay_maxrl", 71),
        31010926: ("pantry_plan", "maxrl", 72),
        31010927: ("pantry_plan", "replay_maxrl", 72),
        31010928: ("pantry_plan", "maxrl", 73),
    }
    for job_id, identity in expected_e118.items():
        row = e118_runs.get(job_id)
        observed = None if row is None else (
            str(row["domain"]), str(row["arm"]), int(row["seed"])
        )
        if observed != identity or state(job_id) != "PENDING":
            raise RuntimeError(
                f"E118 priority target {job_id} identity/state drift: {observed}"
            )
    for job_id in E120_QWEN3:
        row = e120_runs.get(job_id)
        if row is None or str(row.get("model_key")) != "qwen3b":
            raise RuntimeError(f"E120 Qwen-3B identity drift for {job_id}")
        if state(job_id) not in {"RUNNING", "PENDING"}:
            raise RuntimeError(f"E120 Qwen-3B state drift for {job_id}")

    before = {
        job_id: {"state": state(job_id), "record": show(job_id)}
        for job_id in (*E118_NODE302, *E118_NODE208, *E120_QWEN3)
    }
    paused: list[int] = []
    updated: list[int] = []
    try:
        # Hold every E120 Qwen-3B candidate first so a waiting sibling cannot
        # consume node302 as the three running allocations are requeued.
        for job_id in E120_QWEN3:
            if before[job_id]["state"] == "RUNNING":
                run(["scontrol", "requeuehold", str(job_id)])
            else:
                run(["scontrol", "hold", str(job_id)])
            paused.append(job_id)

        for job_id in E118_NODE302:
            run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    "Account=mltheory",
                    "Partition=mltheory",
                    "NodeList=node302",
                    f"ExcNodeList={PVL}",
                    "Nice=0",
                ]
            )
            updated.append(job_id)

        for job_id in E118_NODE208:
            run(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    "Account=allcs",
                    "Partition=all",
                    "NodeList=node208",
                    f"ExcNodeList={PVL}",
                    "Nice=0",
                ]
            )
            updated.append(job_id)

        after = {job_id: show(job_id) for job_id in updated}
        for job_id in E118_NODE302:
            required = (
                "Account=mltheory",
                "Partition=mltheory",
                "ReqNodeList=node302",
                f"ExcNodeList={PVL}",
                "MinMemoryNode=128G",
            )
            missing = [token for token in required if token not in after[job_id]]
            if missing:
                raise RuntimeError(f"node302 E118 job {job_id} missing {missing}")
        for job_id in E118_NODE208:
            required = (
                "Account=allcs",
                "Partition=all",
                "ReqNodeList=node208",
                f"ExcNodeList={PVL}",
                "MinMemoryNode=128G",
            )
            missing = [token for token in required if token not in after[job_id]]
            if missing:
                raise RuntimeError(f"node208 E118 job {job_id} missing {missing}")
        for job_id in E120_QWEN3:
            record = show(job_id)
            if field(record, "Reason") != "JobHeldUser":
                raise RuntimeError(f"E120 job {job_id} was not held")

        audit = {
            "schema": "e118-400k-priority-amendment-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "objective": "maximize E118 realized optimizer steps toward 400000 on 2026-09-04",
            "same_scientific_cells": True,
            "same_run_directories": True,
            "optimizer_update_changed": False,
            "treatment_changed": False,
            "pvl_compute_allowed": False,
            "pvl_exclusion": PVL,
            "e118_node302_job_ids": list(E118_NODE302),
            "e118_node208_job_ids": list(E118_NODE208),
            "e120_qwen3_paused_job_ids": list(E120_QWEN3),
            "e120_running_cells_requeued_from_checkpoint": [
                job_id
                for job_id in E120_QWEN3
                if before[job_id]["state"] == "RUNNING"
            ],
            "e120_pending_cells_held": [
                job_id
                for job_id in E120_QWEN3
                if before[job_id]["state"] == "PENDING"
            ],
            "outcomes_inspected": False,
            "installed": True,
        }
        atomic(AUDIT, audit)
    except Exception:
        for job_id in updated:
            record = before[job_id]["record"]
            command = [
                "scontrol",
                "update",
                f"JobId={job_id}",
                f"Account={field(record, 'Account')}",
                f"Partition={field(record, 'Partition')}",
                f"NodeList={field(record, 'ReqNodeList')}",
            ]
            run(command, check=False)
        for job_id in paused:
            run(["scontrol", "release", str(job_id)], check=False)
        raise

    print(
        "E118 priority installed: "
        f"node302={','.join(map(str, E118_NODE302))} "
        f"node208={','.join(map(str, E118_NODE208))}; "
        f"paused E120={','.join(map(str, E120_QWEN3))}"
    )


if __name__ == "__main__":
    main()
