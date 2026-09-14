#!/usr/bin/env python3
"""Apply the frozen scheduler-only E80-R1 completion repair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import status_e78 as shared  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e80r1_completion_scheduler_repair_20260821.md"
)
E80_LEDGER = ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"
E87_LEDGER = ROOT / "var/artifacts/e87_qwen3b_semantic_maxent_seed70_jobs.json"
PRIOR_PLACEMENT = ROOT / (
    "var/artifacts/e105_qwen3_paired_a6000_placement_amendment.json"
)
OUT = ROOT / "var/artifacts/e80r1_completion_scheduler_repair_20260821.json"

REGISTERED_NICE = "100"
TEMPORARY_NICE = "500"
ACCOUNT = "mltheory"
TIME_LIMIT = "3-00:00:00"
LIVE_A6000_POOL = "node[103-104,205-208]"
OLD_A6000_POOL = "node[103-104,205-208,805]"
LIVE_A6000_NODES = (
    "node103", "node104", "node205", "node206", "node207", "node208"
)

# job_id -> (domain, arm, seed)
PYTHON_TARGETS = {
    30277399: ("python_factors", "replay", 73),
    30277400: ("python_factors", "control", 74),
    30277401: ("python_factors", "replay", 74),
}
A6000_TARGETS = {
    30277404: ("mathir", "control", 71),
    30277405: ("mathir", "replay", 71),
    30277406: ("mathir", "control", 72),
    30277407: ("mathir", "replay", 72),
    30277408: ("mathir", "control", 73),
    30277409: ("mathir", "replay", 73),
    30277410: ("mathir", "control", 74),
    30277411: ("mathir", "replay", 74),
    30277414: ("pantry_plan", "control", 71),
    30277415: ("pantry_plan", "replay", 71),
    30277416: ("pantry_plan", "control", 72),
    30277417: ("pantry_plan", "replay", 72),
    30277418: ("pantry_plan", "control", 73),
    30277419: ("pantry_plan", "replay", 73),
    30277420: ("pantry_plan", "control", 74),
    30277421: ("pantry_plan", "replay", 74),
}
TARGETS = {**PYTHON_TARGETS, **A6000_TARGETS}


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}: {detail}")
    return result.stdout.strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def scheduler_record(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def field(record: str, key: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S*)", record)
    return match.group(1) if match else None


def export_map(record: str) -> dict[str, str]:
    match = re.search(r"--export=ALL,(\S+)", record)
    if not match:
        raise RuntimeError("scheduler record carries no frozen --export block")
    exports: dict[str, str] = {}
    for item in match.group(1).split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            exports[key] = value
    if not exports:
        raise RuntimeError("frozen --export block is empty")
    return exports


def load_ledgers() -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    e80 = json.loads(E80_LEDGER.read_text(encoding="utf-8"))
    e87 = json.loads(E87_LEDGER.read_text(encoding="utf-8"))
    if e80.get("schema") != "e80r1_qwen3b_aligned_verified_replay_jobs_v1":
        raise RuntimeError("unexpected E80-R1 ledger schema")
    if e80.get("released") is not True or int(e80.get("target_steps", 0)) != 3072:
        raise RuntimeError("E80-R1 is not the released 3,072-update cohort")
    if e87.get("schema") != "e87_qwen3b_semantic_maxent_seed70_jobs_v1":
        raise RuntimeError("unexpected E87 ledger schema")
    if e87.get("released") is not True:
        raise RuntimeError("E87 is not durably released")
    selected = {
        int(record["job_id"]): record
        for record in e80.get("runs", [])
        if int(record["job_id"]) in TARGETS
    }
    if set(selected) != set(TARGETS):
        raise RuntimeError(f"E80-R1 ledger lacks targets: {sorted(set(TARGETS) - set(selected))}")
    for job_id, expected in TARGETS.items():
        record = selected[job_id]
        actual = (str(record["domain"]), str(record["arm"]), int(record["seed"]))
        if actual != expected:
            raise RuntimeError(f"E80-R1 target identity drifted for {job_id}: {actual}")
    return e80, e87, selected


def require_e87_complete() -> dict[str, Any]:
    snapshot = shared.load_snapshot(E87_LEDGER)
    rows = list(snapshot["rows"])
    target = int(snapshot["target"])
    if len(rows) != 5 or any(int(row["step"]) < target for row in rows):
        raise RuntimeError("E87 is not 5/5 terminal; temporary E80 priority remains justified")
    return {
        "cells": len(rows),
        "terminal": sum(int(row["step"]) >= target for row in rows),
        "target_steps": target,
    }


def progress_snapshot() -> dict[int, dict[str, Any]]:
    snapshot = shared.load_snapshot(E80_LEDGER)
    rows = {int(row["job_id"]): dict(row) for row in snapshot["rows"]}
    if not set(TARGETS).issubset(rows):
        raise RuntimeError("E80-R1 progress snapshot lacks a target")
    return {job_id: rows[job_id] for job_id in TARGETS}


def inventory() -> dict[str, str]:
    nodes = run(
        [
            "sinfo", "-h", "-N", "-n",
            ",".join((*LIVE_A6000_NODES, "node805", "node302")),
            "-o", "%N|%P|%G|%m|%T",
        ]
    )
    lines = nodes.splitlines()
    for node in LIVE_A6000_NODES:
        matches = [
            line for line in lines
            if line.startswith(f"{node}|all|") and "gpu:a6000:" in line
        ]
        if len(matches) != 1:
            raise RuntimeError(f"live A6000 inventory drifted for {node}")
        pieces = matches[0].split("|")
        if int(pieces[3].rstrip("+")) < 128 * 1024:
            raise RuntimeError(f"A6000 node lacks 128 GiB: {node}")
        if pieces[4].lower().startswith(("down", "drain", "fail")):
            raise RuntimeError(f"authorized A6000 node is unavailable: {node}")
    node805 = [line for line in lines if line.startswith("node805|all|")]
    if len(node805) != 1 or not node805[0].split("|")[4].lower().startswith("down"):
        raise RuntimeError("node805 is no longer uniquely recorded as down; refreeze pool")
    node302 = [
        line for line in lines
        if line.startswith("node302|mltheory|") and "gpu:a100:" in line
    ]
    if len(node302) != 1 or int(node302[0].split("|")[3].rstrip("+")) < 128 * 1024:
        raise RuntimeError("node302 A100/mltheory inventory drifted")

    all_partition = run(["scontrol", "show", "partition", "all", "-o"])
    lowprio_partition = run(["scontrol", "show", "partition", "lowprio", "-o"])
    mltheory_partition = run(["scontrol", "show", "partition", "mltheory", "-o"])
    allowed = (field(all_partition, "AllowAccounts") or "").split(",")
    if ACCOUNT not in allowed or field(all_partition, "PreemptMode") != "OFF":
        raise RuntimeError("all partition no longer permits non-preempting mltheory work")
    if field(lowprio_partition, "PreemptMode") != "REQUEUE":
        raise RuntimeError("lowprio preemption contract drifted")
    if field(mltheory_partition, "AllowAccounts") != ACCOUNT:
        raise RuntimeError("mltheory partition account contract drifted")
    return {
        "nodes": nodes,
        "all_partition": all_partition,
        "lowprio_partition": lowprio_partition,
        "mltheory_partition": mltheory_partition,
    }


def require_same_environment(job_id: int, live: str, frozen: str) -> None:
    live_env = export_map(live)
    frozen_env = export_map(frozen)
    if live_env != frozen_env:
        drift = {
            key: (frozen_env.get(key), live_env.get(key))
            for key in sorted(set(frozen_env) | set(live_env))
            if frozen_env.get(key) != live_env.get(key)
        }
        raise RuntimeError(f"E80-R1 job {job_id} environment drifted: {drift}")


def validate_common(
    job_id: int, live: str, frozen: dict[str, Any], *, before: bool
) -> None:
    state = field(live, "JobState")
    allowed_states = {"PENDING"} if before else {"PENDING", "RUNNING"}
    if state not in allowed_states:
        raise RuntimeError(f"E80-R1 job {job_id} has unexpected state: {state}")
    expected = {
        "Account": ACCOUNT,
        "TimeLimit": TIME_LIMIT,
        "NumCPUs": "16",
        "MinMemoryNode": "128G",
        "Dependency": "(null)",
        "Nice": TEMPORARY_NICE if before else REGISTERED_NICE,
    }
    if before:
        expected["RunTime"] = "00:00:00"
    wrong = {
        key: (value, field(live, key))
        for key, value in expected.items()
        if field(live, key) != value
    }
    if wrong:
        raise RuntimeError(f"E80-R1 job {job_id} scheduler state drifted: {wrong}")
    require_same_environment(job_id, live, str(frozen["held_scheduler_record"]))
    if str(frozen["run_dir"]) != export_map(live).get("SAVE_PATH"):
        raise RuntimeError(f"E80-R1 job {job_id} output path drifted")


def validate_before(job_id: int, record: str, frozen: dict[str, Any]) -> None:
    validate_common(job_id, record, frozen, before=True)
    if job_id in PYTHON_TARGETS:
        expected = {
            "Partition": "mltheory",
            "ReqNodeList": "node302",
            "TresPerNode": "gres/gpu:a100:1",
        }
    else:
        expected = {
            "Partition": "lowprio",
            "ReqNodeList": OLD_A6000_POOL,
            "TresPerNode": "gres/gpu:a6000:1",
        }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if wrong:
        raise RuntimeError(f"E80-R1 job {job_id} placement drifted: {wrong}")


def validate_after(job_id: int, record: str, frozen: dict[str, Any]) -> None:
    validate_common(job_id, record, frozen, before=False)
    if job_id in PYTHON_TARGETS:
        expected = {
            "Partition": "mltheory",
            "ReqNodeList": "node302",
            "TresPerNode": "gres/gpu:a100:1",
        }
    else:
        expected = {
            "Partition": "all",
            "ReqNodeList": LIVE_A6000_POOL,
            "TresPerNode": "gres/gpu:a6000:1",
        }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if wrong:
        raise RuntimeError(f"E80-R1 job {job_id} postflight placement drifted: {wrong}")
    if job_id in A6000_TARGETS and "node805" in (field(record, "Reason") or ""):
        raise RuntimeError(f"E80-R1 job {job_id} still reports down node805")


def update(job_id: int, *, repaired: bool) -> None:
    command = ["scontrol", "update", f"JobId={job_id}"]
    if job_id in A6000_TARGETS:
        if repaired:
            command.extend(["Partition=all", f"NodeList={LIVE_A6000_POOL}"])
        else:
            command.extend(["Partition=lowprio", f"NodeList={OLD_A6000_POOL}"])
    command.append(f"Nice={REGISTERED_NICE if repaired else TEMPORARY_NICE}")
    run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    required = (PROTOCOL, E80_LEDGER, E87_LEDGER, PRIOR_PLACEMENT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"required repair inputs are absent: {missing}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate E80-R1 repair: {OUT}")

    prior = json.loads(PRIOR_PLACEMENT.read_text(encoding="utf-8"))
    if prior.get("schema") != "e105_qwen3_paired_a6000_placement_amendment_v1":
        raise SystemExit("prior E80-R1 paired A6000 placement is not authoritative")
    moved_ids = {
        int(pair[key])
        for pair in prior.get("moved_pairs", [])
        for key in ("control_job_id", "replay_job_id")
    }
    if moved_ids != set(A6000_TARGETS):
        raise SystemExit("prior E80-R1 A6000 moved-pair set drifted")

    _e80, _e87, frozen = load_ledgers()
    e87_completion = require_e87_complete()
    progress = progress_snapshot()
    node_inventory = inventory()
    before = {job_id: scheduler_record(job_id) for job_id in TARGETS}
    for job_id in TARGETS:
        validate_before(job_id, before[job_id], frozen[job_id])

    if not args.apply:
        for job_id in TARGETS:
            if job_id in A6000_TARGETS:
                print(
                    f"scontrol update JobId={job_id} Partition=all "
                    f"NodeList={LIVE_A6000_POOL} Nice={REGISTERED_NICE}"
                )
            else:
                print(f"scontrol update JobId={job_id} Nice={REGISTERED_NICE}")
        print(
            f"[e80r1-completion-repair] dry_run=True jobs={len(TARGETS)} "
            f"a6000={len(A6000_TARGETS)} python={len(PYTHON_TARGETS)}"
        )
        return 0

    changed: list[int] = []
    try:
        for job_id in TARGETS:
            update(job_id, repaired=True)
            changed.append(job_id)
        time.sleep(2)
        after = {job_id: scheduler_record(job_id) for job_id in TARGETS}
        for job_id in TARGETS:
            validate_after(job_id, after[job_id], frozen[job_id])
    except Exception:
        for job_id in reversed(changed):
            try:
                current = scheduler_record(job_id)
                if field(current, "JobState") == "PENDING":
                    update(job_id, repaired=False)
            except Exception:  # noqa: BLE001 - restore as much as possible
                pass
        raise

    payload = {
        "schema": "e80r1_completion_scheduler_repair_20260821_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "scheduler_only": True,
        "scientific_environment_changed": False,
        "gpu_type_changed": False,
        "stopping_rule_changed": False,
        "dependencies_changed": False,
        "jobs_cancelled_or_resubmitted": False,
        "run_directories_touched": False,
        "evaluation_outcomes_inspected": False,
        "progress_and_checkpoints_inspected": True,
        "pointmaze": "excluded",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": digest(Path(__file__)),
        "e80r1_ledger": str(E80_LEDGER.relative_to(ROOT)),
        "e80r1_ledger_sha256": digest(E80_LEDGER),
        "e87_ledger": str(E87_LEDGER.relative_to(ROOT)),
        "e87_ledger_sha256": digest(E87_LEDGER),
        "prior_placement": str(PRIOR_PLACEMENT.relative_to(ROOT)),
        "prior_placement_sha256": digest(PRIOR_PLACEMENT),
        "e87_completion": e87_completion,
        "inventory": node_inventory,
        "temporary_nice": int(TEMPORARY_NICE),
        "restored_registered_nice": int(REGISTERED_NICE),
        "old_a6000_pool": OLD_A6000_POOL,
        "new_a6000_pool": LIVE_A6000_POOL,
        "jobs": [
            {
                "job_id": job_id,
                "domain": TARGETS[job_id][0],
                "arm": TARGETS[job_id][1],
                "seed": TARGETS[job_id][2],
                "progress_before": progress[job_id],
                "restarts_before": field(before[job_id], "Restarts"),
                "reason_before": field(before[job_id], "Reason"),
                "reason_after": field(after[job_id], "Reason"),
                "priority_before": field(before[job_id], "Priority"),
                "priority_after": field(after[job_id], "Priority"),
                "before": before[job_id],
                "after": after[job_id],
            }
            for job_id in TARGETS
        ],
    }
    atomic_json(OUT, payload)
    print(
        f"[e80r1-completion-repair] applied=True jobs={len(TARGETS)} "
        f"a6000={len(A6000_TARGETS)} python={len(PYTHON_TARGETS)} "
        f"artifact={OUT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
