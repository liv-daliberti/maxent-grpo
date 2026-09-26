#!/usr/bin/env python3
"""Widen untouched Falcon E104 jobs across proven same-GPU node pools."""

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


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e106_falcon_same_gpu_pool_widening_amendment_20260817.md"
)
LEDGER = ROOT / e104.LEDGER
ONE_HOUR = ROOT / "var/artifacts/e106_falcon_one_hour_backfill_amendment.json"
OUT = ROOT / "var/artifacts/e106_falcon_same_gpu_pool_widening_amendment.json"
PLACEMENTS = {
    30637790: {
        "old": "node202",
        "new": "node[202-204]",
        "gres": "gpu:a5000:1",
    },
    30637793: {
        "old": "node202",
        "new": "node[202-204]",
        "gres": "gpu:a5000:1",
    },
    30637794: {
        "old": "node206",
        "new": "node[205-207]",
        "gres": "gpu:a6000:1",
    },
}
A5000_REFERENCE = 30637791
A6000_REFERENCES = {
    "node205": 30516431,
    "node206": 30516432,
    "node207": 30516430,
}


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


def require(record: str, job_id: int, needles: tuple[str, ...]) -> None:
    missing = [needle for needle in needles if needle not in record]
    if missing:
        raise RuntimeError(f"Falcon job {job_id} lacks scheduler fields {missing}")


def target_runs() -> dict[int, dict[str, Any]]:
    ledger = load(LEDGER)
    if ledger.get("released") is not True:
        raise RuntimeError("E104 was not durably released")
    runs = {
        int(item["job_id"]): item
        for item in ledger.get("runs", [])
        if int(item["job_id"]) in PLACEMENTS
    }
    if set(runs) != set(PLACEMENTS):
        raise RuntimeError(f"Falcon pool target set drifted: {sorted(runs)}")
    return runs


def validate_record(
    run_record: dict[str, Any],
    record: str,
    *,
    node_list: str,
) -> None:
    job_id = int(run_record["job_id"])
    placement = PLACEMENTS[job_id]
    snapshot = Path(str(load(LEDGER)["snapshot_root"]))
    require(
        record,
        job_id,
        (
            "JobState=PENDING",
            "RunTime=00:00:00",
            "TimeLimit=01:00:00",
            "Partition=cs",
            "Account=allcs",
            "NumCPUs=8",
            "MinMemoryNode=64G",
            f"ReqNodeList={node_list}",
            f"TresPerNode=gres/{placement['gres']}",
            f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
            f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
            f"OAT_ZERO_SEED={run_record['seed']}",
            "OAT_ZERO_MAX_TRAIN=64",
            f"OAT_ZERO_VARIANT={e104.VARIANT}",
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        ),
    )


def update(job_id: int, node_list: str) -> None:
    run(["scontrol", "update", f"JobId={job_id}", f"NodeList={node_list}"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"Falcon pool protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate Falcon pool amendment: {OUT}")

    one_hour = load(ONE_HOUR)
    if one_hour.get("schema") != "e106_falcon_one_hour_backfill_amendment_v1":
        raise RuntimeError("one-hour Falcon amendment is not authoritative")
    if not set(PLACEMENTS).issubset(
        {int(item["job_id"]) for item in one_hour.get("jobs", [])}
    ):
        raise RuntimeError("one-hour Falcon amendment does not cover pool targets")

    references = {
        "a5000": accounting_record(A5000_REFERENCE),
        "a6000": {
            node: accounting_record(job_id)
            for node, job_id in A6000_REFERENCES.items()
        },
    }
    if references["a5000"]["node"] != "node203":
        raise RuntimeError("A5000 capacity reference node drifted")
    if "gres/gpu:a5000=1" not in references["a5000"]["req_tres"]:
        raise RuntimeError("A5000 capacity reference GPU drifted")
    for node, record in references["a6000"].items():
        if record["node"] != node or "gres/gpu:a6000=1" not in record["req_tres"]:
            raise RuntimeError(f"A6000 capacity reference drifted: {node}")

    targets = target_runs()
    before = {job_id: scheduler_record(job_id) for job_id in PLACEMENTS}
    for job_id, run_record in targets.items():
        validate_record(
            run_record,
            before[job_id],
            node_list=str(PLACEMENTS[job_id]["old"]),
        )
    if not args.apply:
        for job_id, placement in PLACEMENTS.items():
            print(f"scontrol update JobId={job_id} NodeList={placement['new']}")
        return 0

    changed: list[int] = []
    try:
        for job_id, placement in PLACEMENTS.items():
            update(job_id, str(placement["new"]))
            changed.append(job_id)
        after = {job_id: scheduler_record(job_id) for job_id in PLACEMENTS}
        for job_id, run_record in targets.items():
            validate_record(
                run_record,
                after[job_id],
                node_list=str(PLACEMENTS[job_id]["new"]),
            )
    except Exception:
        for job_id in changed:
            update(job_id, str(PLACEMENTS[job_id]["old"]))
        raise

    payload = {
        "schema": "e106_falcon_same_gpu_pool_widening_amendment_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": e104.digest(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": e104.digest(Path(__file__)),
        "e104_ledger": str(LEDGER.relative_to(ROOT)),
        "e104_ledger_sha256": e104.digest(LEDGER),
        "one_hour_amendment": str(ONE_HOUR.relative_to(ROOT)),
        "one_hour_amendment_sha256": e104.digest(ONE_HOUR),
        "scheduler_only": True,
        "environment_changed": False,
        "gpu_type_changed": False,
        "post_e104_or_e106_update_outcomes_inspected": False,
        "pointmaze": "excluded",
        "capacity_references": references,
        "jobs": [
            {
                "job_id": job_id,
                "scale": "falcon1b",
                "domain": str(targets[job_id]["domain"]),
                "old_node_list": placement["old"],
                "new_node_list": placement["new"],
                "gres": placement["gres"],
                "before": before[job_id],
                "after": after[job_id],
            }
            for job_id, placement in PLACEMENTS.items()
        ],
    }
    e104.e81.atomic_json(OUT, payload)
    print(f"[e106-falcon-pool] jobs=3 artifact={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
