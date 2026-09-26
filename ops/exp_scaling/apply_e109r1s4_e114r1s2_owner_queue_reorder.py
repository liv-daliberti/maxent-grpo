#!/usr/bin/env python3
"""Move the three approved E109/E114 completion jobs atop our own queues."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402


PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e109r1s4_e114r1s2_owner_queue_reorder_20260826.md"
)
PRECEDING = ROOT / (
    "var/artifacts/"
    "e109r1s3_e114r1_completion_backfill_acceleration.json"
)
ARTIFACT = ROOT / (
    "var/artifacts/e109r1s4_e114r1s2_owner_queue_reorder.json"
)
JOB_IDS = (30874013, 30874012, 30790267)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed {command}: {detail}")
    return result.stdout.strip()


def validate(job_id: int, record: str, environments: dict[int, str]) -> None:
    expected = {
        30874013: {
            "JobName": "e109r1-q3-python-s74",
            "Partition": "all",
            "Account": "allcs",
            "ReqNodeList": "node[103-104,205-208,805]",
            "TimeLimit": "12:00:00",
            "Nice": "0",
            "TresPerNode": "gres/gpu:a6000:1",
        },
        30874012: {
            "JobName": "e109r1-q3-python-s73",
            "Partition": "all",
            "Account": "allcs",
            "ReqNodeList": "node[103-104,205-208,805]",
            "TimeLimit": "12:00:00",
            "Nice": "0",
            "TresPerNode": "gres/gpu:a6000:1",
        },
        30790267: {
            "JobName": "e114-q3-mathir-s72",
            "Partition": "mltheory",
            "Account": "mltheory",
            "ReqNodeList": "node302",
            "TimeLimit": "1-00:00:00",
            "Nice": "0",
            "TresPerNode": "gres/gpu:a100:1",
        },
    }[job_id]
    failures = {
        key: (scheduler.field(record, key), value)
        for key, value in expected.items()
        if scheduler.field(record, key) != value
    }
    if scheduler.field(record, "JobState") not in {"PENDING", "RUNNING"}:
        failures["JobState"] = (
            scheduler.field(record, "JobState"),
            "PENDING or RUNNING",
        )
    environment = scheduler.environment(record)
    if scheduler.sha256_text(environment) != environments[job_id]:
        failures["environment_sha256"] = (
            scheduler.sha256_text(environment),
            environments[job_id],
        )
    if failures:
        raise RuntimeError(f"job {job_id} drifted: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate reorder: {ARTIFACT}")
    preceding = json.loads(PRECEDING.read_text(encoding="utf-8"))
    if preceding.get("released") is not True:
        raise SystemExit("preceding scheduler amendment is not released")
    environments = {
        int(row["job_id"]): str(row["environment_sha256"])
        for row in preceding["records"]
    }
    before = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
    for job_id in JOB_IDS:
        validate(job_id, before[job_id], environments)
    if not args.apply:
        print(f"[dry-run] scontrol top jobs={list(JOB_IDS)}")
        return 0

    applied: list[int] = []
    top_results: dict[int, dict[str, object]] = {}
    for job_id in JOB_IDS:
        if scheduler.field(before[job_id], "JobState") == "PENDING":
            result = subprocess.run(
                ["scontrol", "top", str(job_id)],
                capture_output=True,
                text=True,
                check=False,
            )
            top_results[job_id] = {
                "attempted": True,
                "accepted": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
            if result.returncode == 0:
                applied.append(job_id)
        else:
            top_results[job_id] = {
                "attempted": False,
                "accepted": False,
                "reason": "job started before reorder",
            }
    after = {job_id: scheduler.show(job_id) for job_id in JOB_IDS}
    for job_id in JOB_IDS:
        validate(job_id, after[job_id], environments)
    payload = {
        "schema": "e109r1s4_e114r1s2_owner_queue_reorder_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest(PROTOCOL),
        "application": str(Path(__file__).resolve().relative_to(ROOT)),
        "application_sha256": digest(Path(__file__).resolve()),
        "preceding_artifact": str(PRECEDING.relative_to(ROOT)),
        "preceding_artifact_sha256": digest(PRECEDING),
        "exact_job_ids": list(JOB_IDS),
        "top_applied_job_ids": applied,
        "top_results": {str(key): value for key, value in top_results.items()},
        "installed": len(applied) == len(JOB_IDS),
        "scheduler_only": True,
        "scientific_environment_changed": False,
        "outcomes_inspected": False,
        "records": [
            {
                "job_id": job_id,
                "environment_sha256": environments[job_id],
                "before_scheduler_record": before[job_id],
                "after_scheduler_record": after[job_id],
            }
            for job_id in JOB_IDS
        ],
    }
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(ARTIFACT)
    print(
        f"[recorded] owner-queue top accepted={applied} "
        f"artifact={ARTIFACT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
