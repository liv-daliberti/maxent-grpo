#!/usr/bin/env python3
"""Reduce pending E110's scheduler limit to the proven two-hour bound."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402
import launch_e110_falcon_python_admission_horizon as e110  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/e110_two_hour_backfill_amendment_20260818.md"
)
LEDGER = ROOT / e110.LEDGER
OUT = ROOT / "var/artifacts/e110_two_hour_backfill_amendment.json"
OLD_LIMIT = "03:00:00"
NEW_LIMIT = "02:00:00"
SHORT_REFERENCE = 30640330
FULL_REFERENCE = 30269053


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required artifact is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def command(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {detail}")
    return result.stdout.strip()


def scheduler_record(job_id: int) -> str:
    return command(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def accounting_record(job_id: int) -> dict[str, str]:
    output = command(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,ReqTRES",
            "--parsable2",
        ]
    )
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    row = next((fields for fields in rows if fields[0] == str(job_id)), None)
    if row is None or not row[1].startswith("COMPLETED") or row[2] != "0:0":
        raise RuntimeError(f"runtime reference {job_id} did not complete")
    return {
        "job_id": row[0],
        "state": row[1],
        "exit_code": row[2],
        "elapsed": row[3],
        "node": row[4],
        "req_tres": row[5],
    }


def validate_record(record: str, *, limit: str, held: bool) -> None:
    required = [
        "JobState=PENDING",
        "RunTime=00:00:00",
        f"TimeLimit={limit}",
        "Partition=all",
        "Account=allcs",
        "ReqNodeList=node[103-104,205-208,805]",
        "TresPerNode=gres/gpu:a6000:1",
        "NumCPUs=8",
        "MinMemoryNode=64G",
        "OAT_ZERO_MAX_TRAIN=192",
        "OAT_ZERO_NUM_PROMPT_EPOCH=1",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=64",
        "OAT_ZERO_GENERATE_MAX_LENGTH=512",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
    ]
    if held:
        required.append("Reason=JobHeldUser")
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"E110 scheduler record lacks {missing}")


def update_limit(job_id: int, limit: str) -> None:
    command(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            f"TimeLimit={limit}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"E110 time-limit protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate E110 time amendment: {OUT}")
    ledger = load(LEDGER)
    if (
        ledger.get("schema") != "e110_falcon_python_admission_horizon_jobs_v1"
        or ledger.get("released") is not True
        or len(ledger.get("runs", [])) != 1
    ):
        raise RuntimeError("E110 ledger drifted")
    job_id = int(ledger["runs"][0]["job_id"])
    references = {
        "e106_v6_64_step": accounting_record(SHORT_REFERENCE),
        "e79_replay_3072_step": accounting_record(FULL_REFERENCE),
    }
    if (
        references["e106_v6_64_step"]["elapsed"] != "00:39:03"
        or references["e79_replay_3072_step"]["elapsed"] != "11:45:28"
    ):
        raise RuntimeError("E110 runtime references drifted")
    for reference in references.values():
        if (
            "gres/gpu:a6000=1" not in reference["req_tres"]
            or "mem=64G" not in reference["req_tres"]
        ):
            raise RuntimeError("E110 runtime reference hardware drifted")
    before = scheduler_record(job_id)
    validate_record(before, limit=OLD_LIMIT, held=False)
    if not args.apply:
        print(f"scontrol hold {job_id}")
        print(f"scontrol update JobId={job_id} TimeLimit={NEW_LIMIT}")
        print(f"scontrol release {job_id}")
        return 0

    changed = False
    released = False
    try:
        command(["scontrol", "hold", str(job_id)])
        held = scheduler_record(job_id)
        validate_record(held, limit=OLD_LIMIT, held=True)
        update_limit(job_id, NEW_LIMIT)
        changed = True
        amended = scheduler_record(job_id)
        validate_record(amended, limit=NEW_LIMIT, held=True)
        payload = {
            "schema": "e110_two_hour_backfill_amendment_v1",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": e81.digest(PROTOCOL),
            "script": str(Path(__file__).relative_to(ROOT)),
            "script_sha256": e81.digest(Path(__file__)),
            "e110_ledger": str(LEDGER.relative_to(ROOT)),
            "e110_ledger_sha256": e81.digest(LEDGER),
            "job_id": job_id,
            "old_time_limit": OLD_LIMIT,
            "new_time_limit": NEW_LIMIT,
            "scheduler_only": True,
            "environment_changed": False,
            "scientific_configuration_changed": False,
            "post_e104_or_e106_or_e110_update_outcomes_inspected": False,
            "pointmaze": "excluded",
            "runtime_references": references,
            "before": before,
            "held": held,
            "amended_held": amended,
            "released": False,
        }
        e81.atomic_json(OUT, payload)
        command(["scontrol", "release", str(job_id)])
        released = True
        released_record = scheduler_record(job_id)
        validate_record(released_record, limit=NEW_LIMIT, held=False)
        payload["released"] = True
        payload["released_at"] = datetime.now(timezone.utc).isoformat()
        payload["released_scheduler_record"] = released_record
        e81.atomic_json(OUT, payload)
    except Exception:
        if not released:
            if changed:
                update_limit(job_id, OLD_LIMIT)
            command(["scontrol", "release", str(job_id)])
        raise
    print(f"[e110-two-hour] job={job_id} artifact={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
