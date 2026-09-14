#!/usr/bin/env python3
"""Apply the audited E109-R1-S3 / E114-R1 backfill acceleration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402
import launch_e109r1_qwen3_python_continuation as continuation  # noqa: E402


PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e109r1s3_e114r1_completion_backfill_acceleration_20260826.md"
)
ARTIFACT = ROOT / (
    "var/artifacts/"
    "e109r1s3_e114r1_completion_backfill_acceleration.json"
)
E109_LEDGER = ROOT / "var/artifacts/e109r1_qwen3_python_continuation_jobs.json"
E114_LEDGER = ROOT / "var/artifacts/e114_plain_grpo_qwen3b_extension_jobs.json"
E109_JOB_IDS = (30874012, 30874013)
E114_JOB_ID = 30790267
JOB_IDS = (*E109_JOB_IDS, E114_JOB_ID)
OLD_E109_NODES = "node[103-104,205-207,805]"
NEW_E109_NODES = "node[103-104,205-208,805]"
OLD_E114_TIME = "3-00:00:00"
NEW_E114_TIME = "1-00:00:00"
E114_CHECKPOINT = ROOT / (
    "var/data/xdr_qwen25_3b_instruct_e114_qwen3b_mathir_plain_grpo_s72/"
    "debug_job30790267/checkpoints/step_01728"
)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed {command}: {detail}")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def frozen_records() -> dict[int, dict[str, Any]]:
    e109 = json.loads(E109_LEDGER.read_text(encoding="utf-8"))
    if (
        e109.get("schema") != "e109r1_qwen3_python_continuation_jobs_v1"
        or e109.get("released") is not True
        or e109.get("outcomes_inspected") is not False
    ):
        raise RuntimeError("E109 continuation ledger drifted")
    records = {
        int(row["continuation_job_id"]): {
            "campaign": "E109",
            "seed": int(row["seed"]),
            "run_dir": str(row["run_dir"]),
            "checkpoint": str(row["resume_checkpoint"]),
            "checkpoint_step": int(row["resume_checkpoint_step"]),
            "environment": None,
            "environment_sha256": str(row["environment_sha256"]),
        }
        for row in e109.get("records", [])
    }
    if set(records) != set(E109_JOB_IDS):
        raise RuntimeError("E109 continuation ledger lacks the exact two jobs")

    e114 = json.loads(E114_LEDGER.read_text(encoding="utf-8"))
    if (
        e114.get("schema") != "e114_plain_grpo_qwen3b_extension_jobs_v1"
        or e114.get("released") is not True
        or int(e114.get("target_steps", -1)) != 3072
    ):
        raise RuntimeError("E114 release ledger drifted")
    matches = [
        row for row in e114.get("runs", [])
        if int(row.get("job_id", -1)) == E114_JOB_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("E114 release ledger lacks job 30790267")
    row = matches[0]
    expected_environment = scheduler.environment(str(row["held_scheduler_record"]))
    records[E114_JOB_ID] = {
        "campaign": "E114",
        "seed": int(row["seed"]),
        "run_dir": str(row["run_dir"]),
        "checkpoint": str(E114_CHECKPOINT),
        "checkpoint_step": 1728,
        "environment": expected_environment,
        "environment_sha256": scheduler.sha256_text(expected_environment),
    }
    return records


def validate_checkpoint(row: dict[str, Any]) -> None:
    checkpoint = Path(str(row["checkpoint"]))
    if not checkpoint.is_dir() or checkpoint.name != f"step_{row['checkpoint_step']:05d}":
        raise RuntimeError(f"checkpoint identity drifted: {checkpoint}")
    run(
        [
            sys.executable,
            str(ROOT / "ops/validate_deepspeed_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
        ]
    )


def validate_identity(job_id: int, record: str, frozen: dict[str, Any]) -> None:
    if job_id in E109_JOB_IDS:
        expected = {
            "JobName": f"e109r1-q3-python-s{frozen['seed']}",
            "Partition": "all",
            "Account": "allcs",
            "NumCPUs": "16",
            "MinMemoryNode": "128G",
            "TresPerNode": "gres/gpu:a6000:1",
        }
    else:
        expected = {
            "JobName": "e114-q3-mathir-s72",
            "Partition": "mltheory",
            "Account": "mltheory",
            "ReqNodeList": "node302",
            "NumCPUs": "16",
            "MinMemoryNode": "128G",
            "TresPerNode": "gres/gpu:a100:1",
        }
    failures = {
        key: (scheduler.field(record, key), value)
        for key, value in expected.items()
        if scheduler.field(record, key) != value
    }
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
        raise RuntimeError(f"job {job_id} scientific identity drifted: {failures}")


def validate_pending_before(job_id: int, record: str) -> None:
    expected = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Nice": "100",
    }
    if job_id in E109_JOB_IDS:
        expected.update(
            {
                "Restarts": "0",
                "ReqNodeList": OLD_E109_NODES,
                "TimeLimit": "12:00:00",
            }
        )
    else:
        expected.update({"Restarts": "1", "TimeLimit": OLD_E114_TIME})
    failures = {
        key: (scheduler.field(record, key), value)
        for key, value in expected.items()
        if scheduler.field(record, key) != value
    }
    if scheduler.field(record, "Reason") == "JobHeldUser":
        failures["Reason"] = ("JobHeldUser", "released")
    if failures:
        raise RuntimeError(f"job {job_id} is not safely amendable: {failures}")


def validate_amended(
    job_id: int,
    record: str,
    *,
    held: bool,
    nice: str,
) -> None:
    expected = {"Nice": nice}
    if job_id in E109_JOB_IDS:
        expected.update({"ReqNodeList": NEW_E109_NODES, "TimeLimit": "12:00:00"})
    else:
        expected.update({"ReqNodeList": "node302", "TimeLimit": NEW_E114_TIME})
    failures = {
        key: (scheduler.field(record, key), value)
        for key, value in expected.items()
        if scheduler.field(record, key) != value
    }
    reason = scheduler.field(record, "Reason")
    if held and reason != "JobHeldUser":
        failures["Reason"] = (reason, "JobHeldUser")
    if not held:
        if reason == "JobHeldUser":
            failures["Reason"] = (reason, "released")
        if scheduler.field(record, "JobState") not in {"PENDING", "RUNNING"}:
            failures["JobState"] = (
                scheduler.field(record, "JobState"),
                "PENDING or RUNNING",
            )
    if failures:
        raise RuntimeError(f"job {job_id} amendment drifted: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing protocol: {PROTOCOL}")
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate application: {ARTIFACT}")

    frozen = frozen_records()
    before = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
    for job_id in JOB_IDS:
        validate_identity(job_id, before[job_id], frozen[job_id])
        validate_pending_before(job_id, before[job_id])
        validate_checkpoint(frozen[job_id])

    inventory = run(
        [
            "sinfo", "-h", "-N", "-n",
            "node103,node104,node205,node206,node207,node208,node805,node302",
            "-o", "%N|%P|%T|%G|%C|%m|%E",
        ]
    ).stdout.strip()
    if not args.apply:
        print(
            f"[dry-run] E109 NodeList={NEW_E109_NODES}; "
            f"E114 TimeLimit={NEW_E114_TIME}; optional Nice=0; release all"
        )
        return 0

    changed: list[int] = []
    nice_results: dict[int, dict[str, Any]] = {}
    try:
        for job_id in JOB_IDS:
            run(["scontrol", "uhold", str(job_id)])
        held_before = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
        for job_id in JOB_IDS:
            validate_identity(job_id, held_before[job_id], frozen[job_id])
            if scheduler.field(held_before[job_id], "Reason") != "JobHeldUser":
                raise RuntimeError(f"job {job_id} did not enter a user hold")

        for job_id in E109_JOB_IDS:
            run(
                [
                    "scontrol", "update", f"JobId={job_id}",
                    f"NodeList={NEW_E109_NODES}",
                ]
            )
            changed.append(job_id)
        run(
            [
                "scontrol", "update", f"JobId={E114_JOB_ID}",
                f"TimeLimit={NEW_E114_TIME}",
            ]
        )
        changed.append(E114_JOB_ID)

        for job_id in JOB_IDS:
            result = run(
                ["scontrol", "update", f"JobId={job_id}", "Nice=0"],
                check=False,
            )
            nice_results[job_id] = {
                "attempted": True,
                "accepted": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }

        held_after = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
        for job_id in JOB_IDS:
            validate_identity(job_id, held_after[job_id], frozen[job_id])
            nice = "0" if nice_results[job_id]["accepted"] else "100"
            validate_amended(job_id, held_after[job_id], held=True, nice=nice)

        payload: dict[str, Any] = {
            "schema": "e109r1s3_e114r1_completion_backfill_acceleration_v1",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": digest(PROTOCOL),
            "application": str(Path(__file__).resolve().relative_to(ROOT)),
            "application_sha256": digest(Path(__file__).resolve()),
            "source_ledgers": {
                str(E109_LEDGER.relative_to(ROOT)): digest(E109_LEDGER),
                str(E114_LEDGER.relative_to(ROOT)): digest(E114_LEDGER),
            },
            "exact_job_ids": list(JOB_IDS),
            "scheduler_only": True,
            "scientific_environment_changed": False,
            "hardware_class_changed": False,
            "run_directories_touched": False,
            "outcomes_inspected": False,
            "e109_old_node_list": OLD_E109_NODES,
            "e109_new_node_list": NEW_E109_NODES,
            "e114_old_time_limit": OLD_E114_TIME,
            "e114_new_time_limit": NEW_E114_TIME,
            "node_inventory": inventory,
            "released": False,
            "records": [
                {
                    "job_id": job_id,
                    "campaign": frozen[job_id]["campaign"],
                    "seed": frozen[job_id]["seed"],
                    "run_dir": frozen[job_id]["run_dir"],
                    "resume_checkpoint": frozen[job_id]["checkpoint"],
                    "resume_checkpoint_step": frozen[job_id]["checkpoint_step"],
                    "checkpoint_valid": True,
                    "environment_sha256": frozen[job_id]["environment_sha256"],
                    "nice_update": nice_results[job_id],
                    "before_scheduler_record": before[job_id],
                    "held_before_scheduler_record": held_before[job_id],
                    "held_after_scheduler_record": held_after[job_id],
                }
                for job_id in JOB_IDS
            ],
        }
        atomic_json(ARTIFACT, payload)
        for job_id in JOB_IDS:
            run(["scontrol", "release", str(job_id)])
        released = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
        for row in payload["records"]:
            job_id = int(row["job_id"])
            validate_identity(job_id, released[job_id], frozen[job_id])
            nice = "0" if nice_results[job_id]["accepted"] else "100"
            validate_amended(job_id, released[job_id], held=False, nice=nice)
            row["released_scheduler_record"] = released[job_id]
        payload["released"] = True
        atomic_json(ARTIFACT, payload)
    except Exception:
        for job_id in changed:
            command = ["scontrol", "update", f"JobId={job_id}"]
            if job_id in E109_JOB_IDS:
                command.append(f"NodeList={OLD_E109_NODES}")
            else:
                command.append(f"TimeLimit={OLD_E114_TIME}")
            run(command, check=False)
        for job_id in JOB_IDS:
            run(["scontrol", "release", str(job_id)], check=False)
        if ARTIFACT.exists():
            ARTIFACT.unlink()
        raise

    print(
        f"[applied] E109 nodes={NEW_E109_NODES}; "
        f"E114 time={NEW_E114_TIME}; released={len(JOB_IDS)}; "
        f"artifact={ARTIFACT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
