#!/usr/bin/env python3
"""Widen untouched Falcon A6000 gate jobs to the all-partition pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402
import launch_e106_python_lambda_normalization_three_scale as e106  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e106_falcon_a6000_all_partition_pool_amendment_20260817.md"
)
E104_LEDGER = ROOT / e104.LEDGER
E106_LEDGER = ROOT / e106.LEDGER
PRIOR_POOL = ROOT / (
    "var/artifacts/e106_falcon_same_gpu_pool_widening_amendment.json"
)
OUT = ROOT / (
    "var/artifacts/e106_falcon_a6000_all_partition_pool_amendment.json"
)
TARGETS = {
    30637794: ("e104", "pantry_plan"),
    30640330: ("e106", "python_factors"),
}
OLD_PARTITION = "cs"
NEW_PARTITION = "all"
OLD_NODE_LIST = "node[205-207]"
NEW_NODE_LIST = "node[103-104,205-208,805]"
AUTHORIZED_NODES = (
    "node103",
    "node104",
    "node205",
    "node206",
    "node207",
    "node208",
    "node805",
)
FALCON_CAPACITY_REFERENCES = (30516431, 30516432, 30516430)
QWEN3_CAPACITY_REFERENCE = 30638185


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required artifact is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {detail}")
    return result.stdout.strip()


def scheduler_record(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def accounting_record(job_id: int) -> dict[str, str]:
    output = run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State,NodeList,ReqTRES,Elapsed",
            "--parsable2",
        ]
    )
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    row = next((fields for fields in rows if fields[0] == str(job_id)), None)
    if row is None or not row[1].startswith("COMPLETED"):
        raise RuntimeError(f"capacity reference {job_id} did not complete")
    return {
        "job_id": row[0],
        "state": row[1],
        "node": row[2],
        "req_tres": row[3],
        "elapsed": row[4],
    }


def inventory_record() -> str:
    return run(
        [
            "sinfo",
            "-h",
            "-N",
            "-n",
            ",".join(AUTHORIZED_NODES),
            "-o",
            "%N|%P|%G|%m",
        ]
    )


def require_inventory(record: str) -> None:
    lines = record.splitlines()
    for node in AUTHORIZED_NODES:
        matches = [
            line
            for line in lines
            if line.startswith(f"{node}|all|")
            and "gpu:a6000:" in line
        ]
        if len(matches) != 1:
            raise RuntimeError(f"authorized A6000 inventory drifted: {node}")
        memory = int(matches[0].rsplit("|", maxsplit=1)[1])
        if memory < 64 * 1024:
            raise RuntimeError(f"authorized node lacks 64 GiB: {node}")


def target_runs() -> dict[int, dict[str, Any]]:
    ledgers = {"e104": load(E104_LEDGER), "e106": load(E106_LEDGER)}
    if any(payload.get("released") is not True for payload in ledgers.values()):
        raise RuntimeError("E104/E106 was not durably released")
    selected: dict[int, dict[str, Any]] = {}
    for job_id, (source, domain) in TARGETS.items():
        matches = [
            run_record
            for run_record in ledgers[source].get("runs", [])
            if int(run_record["job_id"]) == job_id
            and str(run_record["domain"]) == domain
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Falcon A6000 target drifted: {job_id}")
        selected[job_id] = {
            **matches[0],
            "source": source,
            "snapshot_root": ledgers[source]["snapshot_root"],
        }
    return selected


def validate_record(
    run_record: dict[str, Any],
    record: str,
    *,
    partition: str,
    node_list: str,
) -> None:
    required = (
        "JobState=PENDING",
        "RunTime=00:00:00",
        "TimeLimit=01:00:00",
        f"Partition={partition}",
        "Account=allcs",
        "NumCPUs=8",
        "MinMemoryNode=64G",
        f"ReqNodeList={node_list}",
        "TresPerNode=gres/gpu:a6000:1",
        f"OAT_ZERO_SOURCE_ROOT={run_record['snapshot_root']}/src",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={run_record['snapshot_root']}/ops",
        f"OAT_ZERO_SEED={run_record['seed']}",
        "OAT_ZERO_MAX_TRAIN=64",
        f"OAT_ZERO_VARIANT={e104.VARIANT}",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
    )
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(
            f"Falcon A6000 job {run_record['job_id']} lacks {missing}"
        )


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"Falcon all-partition protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate Falcon amendment: {OUT}")

    prior = load(PRIOR_POOL)
    if prior.get("schema") != (
        "e106_falcon_same_gpu_pool_widening_amendment_v1"
    ):
        raise RuntimeError("prior Falcon pool amendment is not authoritative")
    prior_jobs = {int(item["job_id"]): item for item in prior.get("jobs", [])}
    if prior_jobs[30637794].get("new_node_list") != OLD_NODE_LIST:
        raise RuntimeError("prior Falcon Pantry placement drifted")

    references = {
        "falcon_full_runs": [
            accounting_record(job_id)
            for job_id in FALCON_CAPACITY_REFERENCES
        ],
        "qwen3_update_only": accounting_record(QWEN3_CAPACITY_REFERENCE),
    }
    if {item["node"] for item in references["falcon_full_runs"]} != {
        "node205",
        "node206",
        "node207",
    }:
        raise RuntimeError("Falcon A6000 capacity nodes drifted")
    for item in references["falcon_full_runs"]:
        if "gres/gpu:a6000=1" not in item["req_tres"]:
            raise RuntimeError("Falcon A6000 capacity GPU drifted")
    qwen3_reference = references["qwen3_update_only"]
    if (
        qwen3_reference["node"] != "node208"
        or "gres/gpu:a6000=1" not in qwen3_reference["req_tres"]
        or "mem=128G" not in qwen3_reference["req_tres"]
    ):
        raise RuntimeError("Qwen-3B A6000 capacity reference drifted")

    inventory = inventory_record()
    require_inventory(inventory)
    targets = target_runs()
    before = {job_id: scheduler_record(job_id) for job_id in TARGETS}
    for job_id, run_record in targets.items():
        validate_record(
            run_record,
            before[job_id],
            partition=OLD_PARTITION,
            node_list=OLD_NODE_LIST,
        )
    if not args.apply:
        for job_id in TARGETS:
            print(
                f"scontrol update JobId={job_id} Partition={NEW_PARTITION} "
                f"NodeList={NEW_NODE_LIST}"
            )
        return 0

    changed: list[int] = []
    try:
        for job_id in TARGETS:
            update(
                job_id,
                partition=NEW_PARTITION,
                node_list=NEW_NODE_LIST,
            )
            changed.append(job_id)
        after = {job_id: scheduler_record(job_id) for job_id in TARGETS}
        for job_id, run_record in targets.items():
            validate_record(
                run_record,
                after[job_id],
                partition=NEW_PARTITION,
                node_list=NEW_NODE_LIST,
            )
    except Exception:
        for job_id in changed:
            update(
                job_id,
                partition=OLD_PARTITION,
                node_list=OLD_NODE_LIST,
            )
        raise

    payload = {
        "schema": "e106_falcon_a6000_all_partition_pool_amendment_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": e104.digest(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": e104.digest(Path(__file__)),
        "e104_ledger": str(E104_LEDGER.relative_to(ROOT)),
        "e104_ledger_sha256": e104.digest(E104_LEDGER),
        "e106_ledger": str(E106_LEDGER.relative_to(ROOT)),
        "e106_ledger_sha256": e104.digest(E106_LEDGER),
        "prior_pool_amendment": str(PRIOR_POOL.relative_to(ROOT)),
        "prior_pool_amendment_sha256": e104.digest(PRIOR_POOL),
        "scheduler_only": True,
        "environment_changed": False,
        "gpu_type_changed": False,
        "partition_changed": True,
        "account_changed": False,
        "post_e104_or_e106_update_outcomes_inspected": False,
        "pointmaze": "excluded",
        "authorized_nodes": list(AUTHORIZED_NODES),
        "inventory": inventory,
        "capacity_references": references,
        "jobs": [
            {
                "job_id": job_id,
                "scale": "falcon1b",
                "domain": str(targets[job_id]["domain"]),
                "source": str(targets[job_id]["source"]),
                "old_partition": OLD_PARTITION,
                "new_partition": NEW_PARTITION,
                "old_node_list": OLD_NODE_LIST,
                "new_node_list": NEW_NODE_LIST,
                "gres": "gpu:a6000:1",
                "before": before[job_id],
                "after": after[job_id],
            }
            for job_id in TARGETS
        ],
    }
    e104.e81.atomic_json(OUT, payload)
    print(f"[e106-falcon-all-pool] jobs=2 artifact={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
