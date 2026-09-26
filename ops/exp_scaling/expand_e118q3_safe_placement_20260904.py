#!/usr/bin/env python3
"""Add idle safe node202 capacity to pending E118 Qwen-3B jobs.

This is a scheduler-only placement repair: scientific cells, run directories,
optimizer settings, and all treatment variables remain unchanged.  Existing
PVL exclusions are required before and after every update.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e118q3_maxrl_verified_replay_extension_jobs.json"
AUDIT = ROOT / "var/artifacts/e118q3_safe_placement_expansion_20260904.json"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
SAFE_NODES = "node202,node205,node206,node207,node208,node302"


def command(parts: list[str]) -> str:
    result = subprocess.run(parts, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def show(job_id: int) -> str:
    return command(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def field(record: str, name: str) -> str:
    match = re.search(rf"(?:^| )(?P<name>{re.escape(name)})=(?P<value>[^ ]*)", record)
    if match is None:
        raise RuntimeError(f"job record lacks {name}")
    return match.group("value")


def atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if AUDIT.exists():
        raise SystemExit(f"placement expansion already recorded: {AUDIT}")
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if len(runs) != 50:
        raise RuntimeError(f"expected 50 E118 Qwen-3B cells, found {len(runs)}")

    records: list[dict[str, Any]] = []
    for run in runs:
        job_id = int(run["job_id"])
        try:
            current = show(job_id)
        except subprocess.CalledProcessError:
            continue
        current_state = field(current, "JobState")
        current_req_nodes = field(current, "ReqNodeList")
        recovered_running_update = (
            current_state == "RUNNING" and field(current, "NodeList") == "node202"
            and "202" in current_req_nodes
        )
        if current_state != "PENDING" and not recovered_running_update:
            continue
        required = {
            "Account": "allcs",
            "Partition": "cs",
            "ExcNodeList": PVL,
            "MinMemoryNode": "128G",
        }
        drift = {
            key: (expected, field(current, key))
            for key, expected in required.items()
            if field(current, key) != expected
        }
        if drift:
            raise RuntimeError(f"job {job_id} placement drift: {drift}")
        if "gres/gpu=1" not in field(current, "ReqTRES"):
            raise RuntimeError(f"job {job_id} no longer requests exactly one GPU")
        records.append(
            {
                "job_id": job_id,
                "domain": run["domain"],
                "arm": run["arm"],
                "seed": int(run["seed"]),
                "run_dir": run["run_dir"],
                "before_req_node_list": field(current, "ReqNodeList"),
                "state_at_reaudit": current_state,
                "needs_update": current_state == "PENDING",
            }
        )

    print(f"pending={len(records)} safe_nodes={SAFE_NODES} apply={args.apply}")
    if not args.apply:
        return
    if not records:
        raise RuntimeError("no pending E118 Qwen-3B jobs to update")

    for record in records:
        if not record["needs_update"]:
            continue
        subprocess.run(
            [
                "scontrol",
                "update",
                f"JobId={record['job_id']}",
                f"NodeList={SAFE_NODES}",
            ],
            check=True,
        )

    for record in records:
        updated = show(int(record["job_id"]))
        record["after_state"] = field(updated, "JobState")
        record["after_req_node_list"] = field(updated, "ReqNodeList")
        record["after_node_list"] = field(updated, "NodeList")
        if field(updated, "ExcNodeList") != PVL:
            raise RuntimeError(f"job {record['job_id']} lost the PVL exclusion")
        if "202" not in record["after_req_node_list"]:
            raise RuntimeError(f"job {record['job_id']} did not admit node202")

    audit = {
        "schema": "e118q3_safe_placement_expansion_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scheduler_only": True,
        "same_scientific_cells": True,
        "same_run_directories": True,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "pvl_exclusion": PVL,
        "safe_nodes": SAFE_NODES,
        "initial_update_count": 44,
        "verification_recovered_after_empty_nodelist_parse_error": True,
        "updated_jobs": records,
    }
    atomic(AUDIT, audit)
    print(f"updated={len(records)} audit={AUDIT}")


if __name__ == "__main__":
    main()
