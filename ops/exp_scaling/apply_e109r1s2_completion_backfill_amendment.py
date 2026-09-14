#!/usr/bin/env python3
"""Apply the E109-R1-S2 completion-backfill scheduler amendment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402
import launch_e109r1_qwen3_python_continuation as continuation  # noqa: E402
import launch_e117_same_plumbing_component_preflight as hashing  # noqa: E402
import validate_deepspeed_checkpoint as checkpoint  # noqa: E402


PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e109r1s2_completion_backfill_amendment_20260825.md"
)
CONTINUATION_ARTIFACT = continuation.ARTIFACT
ARTIFACT = ROOT / (
    "var/artifacts/e109r1s2_completion_backfill_amendment.json"
)
JOB_IDS = (30874012, 30874013)
OLD_NODE_LIST = "node[104,205-207,805]"
NEW_NODE_LIST = "node[103-104,205-207,805]"
INVENTORY_NODES = "node103,node104,node205,node206,node207,node805"
OLD_TIME_LIMIT = "3-00:00:00"
NEW_TIME_LIMIT = "12:00:00"


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed {command}: {detail}")
    return result.stdout.strip()


def inventory() -> str:
    return run(
        [
            "sinfo",
            "-h",
            "-N",
            "-n",
            INVENTORY_NODES,
            "-p",
            "all",
            "-o",
            "%N|%P|%T|%G|%C|%m|%E",
        ]
    )


def require_inventory(record: str) -> None:
    rows = {line.split("|", 1)[0]: line for line in record.splitlines()}
    if set(rows) != set(INVENTORY_NODES.split(",")):
        raise RuntimeError("E109-R1-S2 A6000 inventory is incomplete")
    for node, row in rows.items():
        fields = row.split("|")
        if fields[1] != "all" or "gpu:a6000:" not in fields[3]:
            raise RuntimeError(f"E109-R1-S2 GPU-class drift on {node}: {row}")
        state = fields[2].lower().rstrip("-+*~#")
        if state not in {"idle", "mixed"}:
            raise RuntimeError(f"E109-R1-S2 unhealthy node {node}: {row}")
        if int(fields[5].rstrip("+")) < 128 * 1024:
            raise RuntimeError(f"E109-R1-S2 memory shortfall on {node}: {row}")


def records_by_job(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if (
        payload.get("schema") != "e109r1_qwen3_python_continuation_jobs_v1"
        or payload.get("released") is not True
        or payload.get("outcomes_inspected") is not False
    ):
        raise RuntimeError("E109-R1 continuation artifact drifted")
    records = {
        int(row["continuation_job_id"]): row
        for row in payload.get("records", [])
    }
    if set(records) != set(JOB_IDS):
        raise RuntimeError("E109-R1-S2 requires exactly the two continuations")
    return records


def validate(
    job_id: int,
    record: str,
    frozen: dict[str, Any],
    *,
    node_list: str,
    time_limit: str,
    held: bool,
) -> None:
    expected = {
        "JobName": f"e109r1-q3-python-s{frozen['seed']}",
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Partition": "all",
        "Account": "allcs",
        "ReqNodeList": node_list,
        "NumCPUs": "16",
        "MinMemoryNode": "128G",
        "TimeLimit": time_limit,
        "Nice": "100",
        "TresPerNode": "gres/gpu:a6000:1",
    }
    failures = {
        field: (scheduler.field(record, field), value)
        for field, value in expected.items()
        if scheduler.field(record, field) != value
    }
    if held:
        if scheduler.field(record, "Reason") != "JobHeldUser":
            failures["Reason"] = (
                scheduler.field(record, "Reason"),
                "JobHeldUser",
            )
    elif scheduler.field(record, "Reason") == "JobHeldUser":
        failures["Reason"] = ("JobHeldUser", "released")
    environment = scheduler.environment(record)
    environment_hash = scheduler.sha256_text(environment)
    if environment_hash != frozen["environment_sha256"]:
        failures["environment_sha256"] = (
            environment_hash,
            frozen["environment_sha256"],
        )
    if f"SAVE_PATH={frozen['run_dir']}" not in environment:
        failures["SAVE_PATH"] = ("drifted", frozen["run_dir"])
    if failures:
        raise RuntimeError(f"E109-R1-S2 job {job_id} drifted: {failures}")


def validate_resume_checkpoint(frozen: dict[str, Any]) -> None:
    selected = Path(str(frozen["resume_checkpoint"]))
    expected_step = int(frozen["resume_checkpoint_step"])
    if (
        not selected.is_dir()
        or int(selected.name.removeprefix("step_")) != expected_step
        or checkpoint.validate_checkpoint(selected)
    ):
        raise RuntimeError(
            f"E109-R1-S2 seed {frozen['seed']} checkpoint drifted"
        )


def update(job_id: int, *, node_list: str, time_limit: str) -> None:
    run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            f"NodeList={node_list}",
            f"TimeLimit={time_limit}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing E109-R1-S2 protocol: {PROTOCOL}")
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate E109-R1-S2 amendment: {ARTIFACT}")

    payload = json.loads(CONTINUATION_ARTIFACT.read_text(encoding="utf-8"))
    frozen = records_by_job(payload)
    node_inventory = inventory()
    require_inventory(node_inventory)
    before = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
    for job_id in JOB_IDS:
        validate(
            job_id,
            before[job_id],
            frozen[job_id],
            node_list=OLD_NODE_LIST,
            time_limit=OLD_TIME_LIMIT,
            held=False,
        )
        validate_resume_checkpoint(frozen[job_id])

    if not args.apply:
        print(
            "scontrol hold " + " ".join(map(str, JOB_IDS))
            + "; transactionally set NodeList=" + NEW_NODE_LIST
            + " TimeLimit=" + NEW_TIME_LIMIT
            + "; release both"
        )
        print("[e109r1s2] dry_run=True jobs=2")
        return 0

    changed: list[int] = []
    try:
        run(["scontrol", "hold", *map(str, JOB_IDS)])
        held_before = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
        for job_id in JOB_IDS:
            validate(
                job_id,
                held_before[job_id],
                frozen[job_id],
                node_list=OLD_NODE_LIST,
                time_limit=OLD_TIME_LIMIT,
                held=True,
            )
        for job_id in JOB_IDS:
            update(job_id, node_list=NEW_NODE_LIST, time_limit=NEW_TIME_LIMIT)
            changed.append(job_id)
        held_after = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
        for job_id in JOB_IDS:
            validate(
                job_id,
                held_after[job_id],
                frozen[job_id],
                node_list=NEW_NODE_LIST,
                time_limit=NEW_TIME_LIMIT,
                held=True,
            )
        artifact = {
            "schema": "e109r1s2_completion_backfill_amendment_v1",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": hashing.digest(PROTOCOL),
            "application": str(Path(__file__).resolve().relative_to(ROOT)),
            "application_sha256": hashing.digest(Path(__file__).resolve()),
            "continuation_artifact": str(
                CONTINUATION_ARTIFACT.relative_to(ROOT)
            ),
            "continuation_artifact_sha256": hashing.digest(
                CONTINUATION_ARTIFACT
            ),
            "scheduler_only": True,
            "outcomes_inspected": False,
            "scientific_environment_changed": False,
            "gpu_type_changed": False,
            "run_directories_touched": False,
            "pointmaze": "excluded",
            "old_node_list": OLD_NODE_LIST,
            "new_node_list": NEW_NODE_LIST,
            "old_time_limit": OLD_TIME_LIMIT,
            "new_time_limit": NEW_TIME_LIMIT,
            "node_inventory": node_inventory,
            "released": False,
            "records": [
                {
                    "job_id": job_id,
                    "seed": int(frozen[job_id]["seed"]),
                    "run_dir": str(frozen[job_id]["run_dir"]),
                    "resume_checkpoint": str(
                        frozen[job_id]["resume_checkpoint"]
                    ),
                    "resume_checkpoint_step": int(
                        frozen[job_id]["resume_checkpoint_step"]
                    ),
                    "environment_sha256": str(
                        frozen[job_id]["environment_sha256"]
                    ),
                    "before_scheduler_record": before[job_id],
                    "held_before_scheduler_record": held_before[job_id],
                    "held_after_scheduler_record": held_after[job_id],
                }
                for job_id in JOB_IDS
            ],
        }
        continuation.atomic_json(ARTIFACT, artifact)
        run(["scontrol", "release", *map(str, JOB_IDS)])
        released = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
        for job_id in JOB_IDS:
            validate(
                job_id,
                released[job_id],
                frozen[job_id],
                node_list=NEW_NODE_LIST,
                time_limit=NEW_TIME_LIMIT,
                held=False,
            )
        for row in artifact["records"]:
            row["released_scheduler_record"] = released[row["job_id"]]
        artifact["released"] = True
        continuation.atomic_json(ARTIFACT, artifact)
    except Exception:
        for job_id in changed:
            try:
                update(
                    job_id,
                    node_list=OLD_NODE_LIST,
                    time_limit=OLD_TIME_LIMIT,
                )
            except Exception:  # noqa: BLE001 - best-effort rollback
                pass
        subprocess.run(
            ["scontrol", "release", *map(str, JOB_IDS)],
            capture_output=True,
            text=True,
            check=False,
        )
        if ARTIFACT.exists():
            ARTIFACT.unlink()
        raise

    print(
        f"[e109r1s2] jobs={list(JOB_IDS)} released=2 "
        f"nodes={NEW_NODE_LIST} time={NEW_TIME_LIMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
