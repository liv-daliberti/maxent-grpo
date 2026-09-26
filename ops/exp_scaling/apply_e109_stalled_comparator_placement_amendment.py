#!/usr/bin/env python3
"""Unstall the six remaining E109 repaired Python Re:Dr comparators."""

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
    "paper/preregistration/"
    "e109_stalled_comparator_placement_amendment_20260821.md"
)
LEDGER = ROOT / (
    "var/artifacts/e109_repaired_python_replay_comparators_jobs.json"
)
OUT = ROOT / (
    "var/artifacts/e109_stalled_comparator_placement_amendment.json"
)
PRIOR_FALCON_POOL = ROOT / (
    "var/artifacts/e106_falcon_a6000_all_partition_pool_amendment.json"
)
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
TIME_LIMIT = "3-00:00:00"
GPU = "gres/gpu:a6000:1"

# job_id -> (scale, seed, old partition, old node list, account, cpus, memory,
#            held before the amendment)
TARGETS: dict[int, dict[str, Any]] = {
    30659546: dict(scale="falcon1b", seed=55, partition="cs",
                   node_list="node207", account="allcs", cpus="8",
                   memory="64G", held=True),
    30659547: dict(scale="falcon1b", seed=56, partition="cs",
                   node_list="node205", account="allcs", cpus="8",
                   memory="64G", held=True),
    30659549: dict(scale="falcon1b", seed=58, partition="cs",
                   node_list="node207", account="allcs", cpus="8",
                   memory="64G", held=True),
    30659550: dict(scale="falcon1b", seed=59, partition="cs",
                   node_list="node205", account="allcs", cpus="8",
                   memory="64G", held=True),
    30659554: dict(scale="qwen3b", seed=73, partition="lowprio",
                   node_list=A6000_POOL, account="mltheory", cpus="16",
                   memory="128G", held=False),
    30659555: dict(scale="qwen3b", seed=74, partition="lowprio",
                   node_list=A6000_POOL, account="mltheory", cpus="16",
                   memory="128G", held=False),
}
RELEASE_PRIORITY_TIMEOUT_S = 90
RELEASE_PRIORITY_POLL_S = 10


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
        memory = int(matches[0].split("|")[3].rstrip("+"))
        if memory < 128 * 1024:
            raise RuntimeError(f"authorized node lacks 128 GiB: {node}")


def ledger_runs() -> dict[int, dict[str, Any]]:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    if payload.get("released") is not True:
        raise RuntimeError("E109 was not durably released")
    if payload.get("domain") != "python_factors":
        raise RuntimeError("E109 ledger domain drifted")
    selected: dict[int, dict[str, Any]] = {}
    for record in payload.get("runs", []):
        job_id = int(record["job_id"])
        if job_id not in TARGETS:
            continue
        target = TARGETS[job_id]
        if (
            str(record["scale"]) != target["scale"]
            or int(record["seed"]) != target["seed"]
            or str(record["arm"]) != "replay"
            or str(record["domain"]) != "python_factors"
        ):
            raise RuntimeError(f"E109 target drifted in the ledger: {job_id}")
        selected[job_id] = record
    if set(selected) != set(TARGETS):
        raise RuntimeError("E109 ledger does not carry all six stalled cells")
    return selected


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
        "Account": target["account"],
        "NumCPUs": target["cpus"],
        "MinMemoryNode": target["memory"],
        "TresPerNode": GPU,
    }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if wrong:
        raise RuntimeError(f"E109 job {job_id} scheduler state drifted: {wrong}")
    frozen_env = export_map(str(frozen["held_scheduler_record"]))
    live_env = export_map(record)
    if frozen_env != live_env:
        drift = {
            key: (frozen_env.get(key), live_env.get(key))
            for key in set(frozen_env) | set(live_env)
            if frozen_env.get(key) != live_env.get(key)
        }
        raise RuntimeError(f"E109 job {job_id} environment drifted: {drift}")
    if str(frozen["run_dir"]) != live_env.get("SAVE_PATH"):
        raise RuntimeError(f"E109 job {job_id} output path drifted")
    if held:
        if field(record, "Reason") != "JobHeldUser" or field(
            record, "Priority"
        ) != "0":
            raise RuntimeError(f"E109 job {job_id} is not the held cell")
    elif field(record, "Reason") == "JobHeldUser":
        raise RuntimeError(f"E109 job {job_id} is unexpectedly held")


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


def await_eligibility(job_ids: list[int]) -> dict[int, str]:
    deadline = time.monotonic() + RELEASE_PRIORITY_TIMEOUT_S
    while True:
        records = {job_id: scheduler_record(job_id) for job_id in job_ids}
        stuck = [
            job_id
            for job_id, record in records.items()
            if field(record, "Priority") == "0"
            and field(record, "Reason") == "JobHeldUser"
        ]
        if not stuck or time.monotonic() >= deadline:
            if stuck:
                raise RuntimeError(f"released E109 jobs stayed held: {stuck}")
            return records
        time.sleep(RELEASE_PRIORITY_POLL_S)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"E109 unstall protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate E109 unstall amendment: {OUT}")
    if not PRIOR_FALCON_POOL.is_file():
        raise SystemExit(f"prior A6000 pool amendment is absent: {PRIOR_FALCON_POOL}")
    prior = json.loads(PRIOR_FALCON_POOL.read_text(encoding="utf-8"))
    if prior.get("schema") != "e106_falcon_a6000_all_partition_pool_amendment_v1":
        raise SystemExit("prior A6000 pool amendment is not authoritative")
    if {job["new_node_list"] for job in prior.get("jobs", [])} != {A6000_POOL}:
        raise SystemExit("registered A6000 pool drifted")

    inventory = inventory_record()
    require_inventory(inventory)
    frozen = ledger_runs()
    before = {job_id: scheduler_record(job_id) for job_id in TARGETS}
    for job_id, target in TARGETS.items():
        validate(
            job_id,
            before[job_id],
            frozen[job_id],
            partition=target["partition"],
            node_list=target["node_list"],
            held=bool(target["held"]),
        )

    if not args.apply:
        for job_id, target in TARGETS.items():
            print(
                f"scontrol update JobId={job_id} Partition={NEW_PARTITION} "
                f"NodeList={A6000_POOL}"
                + ("  # then: scontrol release" if target["held"] else "")
            )
        print(f"[e109-unstall] dry_run=True jobs={len(TARGETS)}")
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
        after = await_eligibility(list(TARGETS)) if released else {
            job_id: scheduler_record(job_id) for job_id in TARGETS
        }
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
                    partition=str(TARGETS[job_id]["partition"]),
                    node_list=str(TARGETS[job_id]["node_list"]),
                )
            except Exception:  # noqa: BLE001 - restore as much as possible
                pass
        raise

    payload = {
        "schema": "e109_stalled_comparator_placement_amendment_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": e78.digest(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": e78.digest(Path(__file__)),
        "ledger": str(LEDGER.relative_to(ROOT)),
        "ledger_sha256": e78.digest(LEDGER),
        "prior_pool_amendment": str(PRIOR_FALCON_POOL.relative_to(ROOT)),
        "prior_pool_amendment_sha256": e78.digest(PRIOR_FALCON_POOL),
        "scheduler_only": True,
        "environment_changed": False,
        "gpu_type_changed": False,
        "walltime_changed": False,
        "account_changed": False,
        "partition_changed": True,
        "run_directories_touched": False,
        "e105_e109_e112_outcomes_inspected": False,
        "pointmaze": "excluded",
        "authorized_nodes": list(AUTHORIZED_NODES),
        "inventory": inventory,
        "jobs": [
            {
                "job_id": job_id,
                "scale": str(TARGETS[job_id]["scale"]),
                "seed": int(TARGETS[job_id]["seed"]),
                "domain": "python_factors",
                "arm": "replay",
                "run_stamp": str(frozen[job_id]["run_stamp"]),
                "old_partition": str(TARGETS[job_id]["partition"]),
                "new_partition": NEW_PARTITION,
                "old_node_list": str(TARGETS[job_id]["node_list"]),
                "new_node_list": A6000_POOL,
                "was_held": bool(TARGETS[job_id]["held"]),
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
        f"[e109-unstall] jobs={len(TARGETS)} released={len(released)} "
        f"artifact={OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
