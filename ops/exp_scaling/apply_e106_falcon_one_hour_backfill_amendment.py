#!/usr/bin/env python3
"""Shorten untouched Falcon E104/E106 smokes using observed runtime margin."""

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
    "e106_falcon_one_hour_backfill_amendment_20260817.md"
)
E104_LEDGER = ROOT / e104.LEDGER
E106_LEDGER = ROOT / e106.LEDGER
OUT = ROOT / "var/artifacts/e106_falcon_one_hour_backfill_amendment.json"
REFERENCE_JOB_ID = 30637791
TARGET_JOB_IDS = (30637790, 30637793, 30637794, 30640330)
OLD_TIME = "02:00:00"
NEW_TIME = "01:00:00"
MAX_REFERENCE_SECONDS = 21 * 60
MIN_RUNTIME_MARGIN = 2.8


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


def elapsed_seconds(value: str) -> int:
    fields = value.split(":")
    if len(fields) != 3:
        raise RuntimeError(f"unexpected elapsed time: {value!r}")
    hours_field, minutes, seconds = fields
    if "-" in hours_field:
        days, hours = hours_field.split("-", 1)
    else:
        days, hours = "0", hours_field
    return int(days) * 86_400 + int(hours) * 3_600 + int(minutes) * 60 + int(seconds)


def reference_runtime() -> dict[str, Any]:
    output = run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            str(REFERENCE_JOB_ID),
            "--format=JobIDRaw,State,Elapsed",
            "--parsable2",
        ]
    )
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    row = next((fields for fields in rows if fields[0] == str(REFERENCE_JOB_ID)), None)
    if row is None or not row[1].startswith("COMPLETED"):
        raise RuntimeError("Falcon runtime reference did not complete")
    seconds = elapsed_seconds(row[2])
    if seconds <= 0 or seconds > MAX_REFERENCE_SECONDS:
        raise RuntimeError("Falcon runtime reference exceeds registered bound")
    margin = 3_600 / seconds
    if margin < MIN_RUNTIME_MARGIN:
        raise RuntimeError("one-hour Falcon limit lacks registered runtime margin")
    return {
        "job_id": REFERENCE_JOB_ID,
        "state": row[1],
        "elapsed": row[2],
        "elapsed_seconds": seconds,
        "one_hour_margin": margin,
    }


def target_runs() -> dict[int, tuple[dict[str, Any], Path]]:
    e104_payload = load(E104_LEDGER)
    e106_payload = load(E106_LEDGER)
    if e104_payload.get("released") is not True or e106_payload.get("released") is not True:
        raise RuntimeError("E104/E106 was not durably released")
    runs = {
        int(run["job_id"]): (run, Path(str(e104_payload["snapshot_root"])))
        for run in e104_payload["runs"]
        if int(run["job_id"]) in TARGET_JOB_IDS
    }
    runs.update(
        {
            int(run["job_id"]): (run, Path(str(e106_payload["snapshot_root"])))
            for run in e106_payload["runs"]
            if int(run["job_id"]) in TARGET_JOB_IDS
        }
    )
    if set(runs) != set(TARGET_JOB_IDS):
        raise RuntimeError(f"Falcon amendment target set drifted: {sorted(runs)}")
    return runs


def require(record: str, job_id: int, needles: tuple[str, ...]) -> None:
    missing = [needle for needle in needles if needle not in record]
    if missing:
        raise RuntimeError(f"Falcon job {job_id} lacks scheduler fields {missing}")


def scientific_needles(run_record: dict[str, Any], snapshot: Path) -> tuple[str, ...]:
    return (
        "JobState=PENDING",
        "RunTime=00:00:00",
        "Account=allcs",
        "NumCPUs=8",
        "MinMemoryNode=64G",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
        f"OAT_ZERO_SEED={run_record['seed']}",
        "OAT_ZERO_MAX_TRAIN=64",
        f"OAT_ZERO_VARIANT={e104.VARIANT}",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
    )


def validate_record(
    run_record: dict[str, Any], snapshot: Path, record: str, *, time_limit: str
) -> None:
    job_id = int(run_record["job_id"])
    placement = (
        (
            "Partition=cs",
            "ReqNodeList=node[205-207]",
            "TresPerNode=gres/gpu:a6000:1",
        )
        if job_id == 30640330
        else (
            "Partition=cs",
            "TresPerNode=gres/gpu:a5000:1"
            if job_id in (30637790, 30637793)
            else "TresPerNode=gres/gpu:a6000:1",
        )
    )
    require(
        record,
        job_id,
        scientific_needles(run_record, snapshot)
        + placement
        + (f"TimeLimit={time_limit}",),
    )


def update(job_id: int, time_limit: str) -> None:
    run(["scontrol", "update", f"JobId={job_id}", f"TimeLimit={time_limit}"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"Falcon backfill protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate Falcon backfill amendment: {OUT}")

    reference = reference_runtime()
    targets = target_runs()
    before = {job_id: scheduler_record(job_id) for job_id in TARGET_JOB_IDS}
    for job_id, (record, snapshot) in targets.items():
        validate_record(record, snapshot, before[job_id], time_limit=OLD_TIME)
    if not args.apply:
        for job_id in TARGET_JOB_IDS:
            print(f"scontrol update JobId={job_id} TimeLimit={NEW_TIME}")
        return 0

    changed: list[int] = []
    try:
        for job_id in TARGET_JOB_IDS:
            update(job_id, NEW_TIME)
            changed.append(job_id)
        after = {job_id: scheduler_record(job_id) for job_id in TARGET_JOB_IDS}
        for job_id, (record, snapshot) in targets.items():
            validate_record(record, snapshot, after[job_id], time_limit=NEW_TIME)
    except Exception:
        for job_id in changed:
            update(job_id, OLD_TIME)
        raise

    payload = {
        "schema": "e106_falcon_one_hour_backfill_amendment_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": e104.digest(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": e104.digest(Path(__file__)),
        "e104_ledger": str(E104_LEDGER.relative_to(ROOT)),
        "e104_ledger_sha256": e104.digest(E104_LEDGER),
        "e106_ledger": str(E106_LEDGER.relative_to(ROOT)),
        "e106_ledger_sha256": e104.digest(E106_LEDGER),
        "scheduler_only": True,
        "environment_changed": False,
        "post_e104_or_e106_update_outcomes_inspected": False,
        "pointmaze": "excluded",
        "old_time_limit": OLD_TIME,
        "new_time_limit": NEW_TIME,
        "reference": reference,
        "jobs": [
            {
                "job_id": job_id,
                "scale": "falcon1b",
                "domain": str(targets[job_id][0]["domain"]),
                "before": before[job_id],
                "after": after[job_id],
            }
            for job_id in TARGET_JOB_IDS
        ],
    }
    e104.e81.atomic_json(OUT, payload)
    print(f"[e106-falcon-one-hour] jobs=4 artifact={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
