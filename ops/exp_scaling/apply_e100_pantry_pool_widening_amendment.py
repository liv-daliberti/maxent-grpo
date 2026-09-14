#!/usr/bin/env python3
"""Widen and unstall the five remaining E100 Falcon PantryPlan cells."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e78_verified_replay_only_05b as e78  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/e100_pantry_pool_widening_amendment_20260821.md"
)
PRIMARY_LEDGER = ROOT / "var/artifacts/e100_sparse_rlep_dr_falcon1b_jobs.json"
RECOVERY_LEDGER = ROOT / (
    "var/artifacts/e100_pantry_infrastructure_recovery_jobs.json"
)
PRIOR_POOL = ROOT / (
    "var/artifacts/e106_falcon_a6000_all_partition_pool_amendment.json"
)
OUT = ROOT / "var/artifacts/e100_pantry_pool_widening_amendment.json"
NEW_PARTITION = "all"
A6000_POOL = "node[103-104,205-208,805]"
AUTHORIZED_NODES = (
    "node103",
    "node104",
    "node205",
    "node206",
    "node207",
    "node208",
    "node805",
)
OLD_PARTITION = "cs"
TIME_LIMIT = "3-00:00:00"
GPU = "gres/gpu:a6000:1"
ACCOUNT = "allcs"

TARGETS: dict[int, dict[str, Any]] = {
    30572828: dict(kind="science", seed=56, node_list="node207", held=True,
                   dependency="(null)"),
    30572829: dict(kind="science", seed=57, node_list="node205", held=True,
                   dependency="(null)"),
    30572831: dict(kind="science", seed=59, node_list="node207", held=True,
                   dependency="(null)"),
    30790683: dict(kind="science", seed=55, node_list="node[205,207]",
                   held=False, dependency="(null)"),
    30790684: dict(kind="science", seed=58, node_list="node[205,207]",
                   held=False, dependency="afterok:30790681(unfulfilled)"),
    30790680: dict(kind="pool", seed=58, node_list="node[205,207]",
                   held=False, dependency="(null)"),
}
RELEASE_TIMEOUT_S = 90
RELEASE_POLL_S = 10


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}: {detail}")
    return result.stdout.strip()


def scheduler_record(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def field(record: str, key: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S*)", record)
    return match.group(1) if match else None


def export_map(record: str) -> dict[str, str]:
    match = re.search(r"--export=ALL,(\S+)", record)
    if not match:
        raise RuntimeError("scheduler record carries no frozen --export block")
    pairs: dict[str, str] = {}
    for item in match.group(1).split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            pairs[key] = value
    if not pairs:
        raise RuntimeError("frozen --export block is empty")
    return pairs


def inventory_record() -> str:
    return run(
        [
            "sinfo",
            "-h",
            "-N",
            "-n",
            ",".join(AUTHORIZED_NODES),
            "-o",
            "%N|%P|%G|%m|%T",
        ]
    )


def require_inventory(record: str) -> None:
    lines = record.splitlines()
    for node in AUTHORIZED_NODES:
        matches = [
            line
            for line in lines
            if line.startswith(f"{node}|{NEW_PARTITION}|")
            and "gpu:a6000:" in line
        ]
        if len(matches) != 1:
            raise RuntimeError(f"authorized A6000 inventory drifted: {node}")
        if int(matches[0].split("|")[3].rstrip("+")) < 64 * 1024:
            raise RuntimeError(f"authorized node lacks 64 GiB: {node}")


def frozen_records() -> dict[int, dict[str, Any]]:
    primary = json.loads(PRIMARY_LEDGER.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_LEDGER.read_text(encoding="utf-8"))
    if primary.get("released") is not True or recovery.get("released") is not True:
        raise RuntimeError("an E100 ledger was not durably released")
    found: dict[int, dict[str, Any]] = {}
    for record in primary.get("runs", []):
        job_id = int(record["job_id"])
        if job_id not in TARGETS:
            continue
        if str(record["domain"]) != "pantry_plan" or int(record["seed"]) != int(
            TARGETS[job_id]["seed"]
        ):
            raise RuntimeError(f"E100 science target drifted: {job_id}")
        found[job_id] = {
            "source": "primary",
            "record": str(record["held_scheduler_record"]),
            "run_dir": str(record["run_dir"]),
            "arm": str(record.get("arm", "")),
        }
    for entry in recovery.get("pantry_retries", []):
        frozen = str(entry.get("held_collection_scheduler_record", ""))
        match = re.match(r"JobId=(\d+)", frozen)
        if not match:
            continue
        job_id = int(match.group(1))
        if job_id in TARGETS:
            found[job_id] = {
                "source": "recovery",
                "record": frozen,
                "run_dir": "",
                "arm": "pool_collection",
            }
    if set(found) != set(TARGETS):
        missing = sorted(set(TARGETS) - set(found))
        raise RuntimeError(f"E100 ledgers lack frozen records for {missing}")
    return found


def validate(
    job_id: int,
    record: str,
    frozen: dict[str, Any],
    *,
    partition: str,
    node_list: str,
    held: bool,
) -> None:
    target = TARGETS[job_id]
    expected = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "TimeLimit": TIME_LIMIT,
        "Partition": partition,
        "ReqNodeList": node_list,
        "Account": ACCOUNT,
        "NumCPUs": "8",
        "MinMemoryNode": "64G",
        "TresPerNode": GPU,
        "Dependency": str(target["dependency"]),
    }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if wrong:
        raise RuntimeError(f"E100 job {job_id} scheduler state drifted: {wrong}")
    frozen_env = export_map(str(frozen["record"]))
    live_env = export_map(record)
    if frozen_env != live_env:
        drift = {
            key: (frozen_env.get(key), live_env.get(key))
            for key in set(frozen_env) | set(live_env)
            if frozen_env.get(key) != live_env.get(key)
        }
        raise RuntimeError(f"E100 job {job_id} environment drifted: {drift}")
    if frozen["run_dir"] and frozen["run_dir"] != live_env.get("SAVE_PATH"):
        raise RuntimeError(f"E100 job {job_id} output path drifted")
    if held:
        if field(record, "Reason") != "JobHeldUser" or field(
            record, "Priority"
        ) != "0":
            raise RuntimeError(f"E100 job {job_id} is not the held cell")
    elif field(record, "Reason") == "JobHeldUser":
        raise RuntimeError(f"E100 job {job_id} is unexpectedly held")


def update(job_id: int, *, partition: str, node_list: str) -> None:
    run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            f"Partition={partition}",
            f"NodeList={node_list}",
        ]
    )


def release(job_id: int) -> None:
    run(["scontrol", "release", str(job_id)])


def hold(job_id: int) -> None:
    run(["scontrol", "hold", str(job_id)])


def await_eligibility() -> dict[int, str]:
    deadline = time.monotonic() + RELEASE_TIMEOUT_S
    while True:
        records = {job_id: scheduler_record(job_id) for job_id in TARGETS}
        stuck = [
            job_id
            for job_id, record in records.items()
            if field(record, "Priority") == "0"
            and field(record, "Reason") == "JobHeldUser"
        ]
        if not stuck or time.monotonic() >= deadline:
            if stuck:
                raise RuntimeError(f"released E100 jobs stayed held: {stuck}")
            return records
        time.sleep(RELEASE_POLL_S)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"E100 widening protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate E100 widening amendment: {OUT}")
    prior = json.loads(PRIOR_POOL.read_text(encoding="utf-8"))
    if prior.get("schema") != "e106_falcon_a6000_all_partition_pool_amendment_v1":
        raise SystemExit("prior A6000 pool amendment is not authoritative")
    if {job["new_node_list"] for job in prior.get("jobs", [])} != {A6000_POOL}:
        raise SystemExit("registered A6000 pool drifted")

    inventory = inventory_record()
    require_inventory(inventory)
    frozen = frozen_records()
    before = {job_id: scheduler_record(job_id) for job_id in TARGETS}
    for job_id, target in TARGETS.items():
        validate(
            job_id,
            before[job_id],
            frozen[job_id],
            partition=OLD_PARTITION,
            node_list=str(target["node_list"]),
            held=bool(target["held"]),
        )

    if not args.apply:
        for job_id, target in TARGETS.items():
            print(
                f"scontrol update JobId={job_id} Partition={NEW_PARTITION} "
                f"NodeList={A6000_POOL}"
                + ("  # then: scontrol release" if target["held"] else "")
            )
        print(f"[e100-widen] dry_run=True jobs={len(TARGETS)}")
        return 0

    changed: list[int] = []
    released: list[int] = []
    try:
        for job_id in TARGETS:
            update(job_id, partition=NEW_PARTITION, node_list=A6000_POOL)
            changed.append(job_id)
        for job_id, target in TARGETS.items():
            if target["held"]:
                release(job_id)
                released.append(job_id)
        after = await_eligibility()
        for job_id in TARGETS:
            validate(
                job_id,
                after[job_id],
                frozen[job_id],
                partition=NEW_PARTITION,
                node_list=A6000_POOL,
                held=False,
            )
    except Exception:
        for job_id in released:
            try:
                hold(job_id)
            except Exception:  # noqa: BLE001 - restore as much as possible
                pass
        for job_id in changed:
            try:
                update(
                    job_id,
                    partition=OLD_PARTITION,
                    node_list=str(TARGETS[job_id]["node_list"]),
                )
            except Exception:  # noqa: BLE001 - restore as much as possible
                pass
        raise

    payload = {
        "schema": "e100_pantry_pool_widening_amendment_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": e78.digest(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": e78.digest(Path(__file__)),
        "primary_ledger": str(PRIMARY_LEDGER.relative_to(ROOT)),
        "primary_ledger_sha256": e78.digest(PRIMARY_LEDGER),
        "recovery_ledger": str(RECOVERY_LEDGER.relative_to(ROOT)),
        "recovery_ledger_sha256": e78.digest(RECOVERY_LEDGER),
        "prior_pool_amendment": str(PRIOR_POOL.relative_to(ROOT)),
        "prior_pool_amendment_sha256": e78.digest(PRIOR_POOL),
        "scheduler_only": True,
        "environment_changed": False,
        "gpu_type_changed": False,
        "walltime_changed": False,
        "account_changed": False,
        "dependencies_changed": False,
        "partition_changed": True,
        "run_directories_touched": False,
        "e100_outcomes_inspected": False,
        "pointmaze": "excluded",
        "authorized_nodes": list(AUTHORIZED_NODES),
        "inventory": inventory,
        "jobs": [
            {
                "job_id": job_id,
                "kind": str(TARGETS[job_id]["kind"]),
                "scale": "falcon1b",
                "domain": "pantry_plan",
                "seed": int(TARGETS[job_id]["seed"]),
                "arm": str(frozen[job_id]["arm"]),
                "old_partition": OLD_PARTITION,
                "new_partition": NEW_PARTITION,
                "old_node_list": str(TARGETS[job_id]["node_list"]),
                "new_node_list": A6000_POOL,
                "was_held": bool(TARGETS[job_id]["held"]),
                "dependency": str(TARGETS[job_id]["dependency"]),
                "restarts_before": field(before[job_id], "Restarts"),
                "reason_before": field(before[job_id], "Reason"),
                "reason_after": field(after[job_id], "Reason"),
                "priority_after": field(after[job_id], "Priority"),
                "gres": GPU,
                "before": before[job_id],
                "after": after[job_id],
            }
            for job_id in TARGETS
        ],
    }
    e78.atomic_json(OUT, payload)
    print(
        f"[e100-widen] jobs={len(TARGETS)} released={len(released)} "
        f"artifact={OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
