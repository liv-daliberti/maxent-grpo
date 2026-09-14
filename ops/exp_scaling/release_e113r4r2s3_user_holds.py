#!/usr/bin/env python3
"""Release the exact 46 zero-runtime E113-R4-R2 science user holds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e113r4_official_dapo_common as common  # noqa: E402
import launch_e113r4_official_dapo as launch  # noqa: E402


PROTOCOL = common.ROOT / (
    "paper/preregistration/e113r4r2s3_user_hold_release_20260829.md"
)
ARTIFACT = common.ROOT / "var/artifacts/e113r4r2s3_user_hold_release.json"
APPLICATION = Path(__file__).resolve()
EXPECTED_JOB_IDS = tuple(range(30869115, 30869161))
COMPLETED_JOB_IDS = tuple(range(30869111, 30869115))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result


def show(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout.strip()


def field(record: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]*)", record)
    if match is None:
        raise RuntimeError(f"scheduler record lacks {name}")
    return match.group(1)


def environment(record: str) -> str:
    match = re.search(r"(?:^| )SubmitLine=.*?--export=([^ ]+)", record)
    if match is None:
        raise RuntimeError("scheduler record lacks a SubmitLine export surface")
    return match.group(1)


def load_ledger() -> dict[str, Any]:
    payload = json.loads(common.LEDGER.read_text(encoding="utf-8"))
    if payload.get("schema") != "e113r4_official_verl_dapo_jobs_v1":
        raise RuntimeError("unexpected E113-R4 ledger schema")
    recovery = payload.get("vllm_scheduler_recovery")
    if not isinstance(recovery, dict):
        raise RuntimeError("E113-R4-R2 recovery is absent")
    if recovery.get("schema") != "e113r4_r4r2_vllm_scheduler_recovery_v1":
        raise RuntimeError("unexpected E113-R4-R2 recovery schema")
    if recovery.get("smoke_gate", {}).get("passed") is not True:
        raise RuntimeError("E113-R4-R2 smoke gate is not recorded as passed")
    expected_all = list(COMPLETED_JOB_IDS + EXPECTED_JOB_IDS)
    observed_all = [int(value) for value in recovery["replacement_science_job_ids"]]
    if observed_all != expected_all:
        raise RuntimeError("authoritative E113-R4-R2 science job IDs drifted")
    if payload.get("released") is not True:
        raise RuntimeError("authoritative E113-R4-R2 ledger is not released")
    return payload


def audit_held(record: str, *, expected_environment: str) -> None:
    required = {
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Partition": "all",
        "Account": "allcs",
        "QOS": "long",
        "Dependency": "(null)",
        "Requeue": "1",
        "ReqNodeList": "(null)",
        "ReqTRES": (
            "cpu=16,mem=128G,node=1,billing=44,gres/gpu=1,gres/gpu:a6000=1"
        ),
        "TresPerNode": "gres/gpu:a6000:1",
        "TimeLimit": "12:00:00",
    }
    failures: dict[str, Any] = {
        key: (field(record, key), expected)
        for key, expected in required.items()
        if field(record, key) != expected
    }
    if field(record, "NodeList") not in ("", "(null)"):
        failures["NodeList"] = field(record, "NodeList")
    if environment(record) != expected_environment:
        failures["environment"] = "scientific export drift"
    if failures:
        raise RuntimeError(f"E113-R4-R2 held job identity drifted: {failures}")


def audit_released(record: str, *, expected_environment: str) -> None:
    failures: dict[str, Any] = {}
    if field(record, "JobState") not in {"PENDING", "RUNNING"}:
        failures["JobState"] = field(record, "JobState")
    if field(record, "Reason") == "JobHeldUser":
        failures["Reason"] = "JobHeldUser"
    if field(record, "Restarts") != "0":
        failures["Restarts"] = field(record, "Restarts")
    for key, expected in {
        "Partition": "all",
        "Account": "allcs",
        "QOS": "long",
        "Dependency": "(null)",
        "Requeue": "1",
        "ReqTRES": (
            "cpu=16,mem=128G,node=1,billing=44,gres/gpu=1,gres/gpu:a6000=1"
        ),
        "TresPerNode": "gres/gpu:a6000:1",
        "TimeLimit": "12:00:00",
    }.items():
        if field(record, key) != expected:
            failures[key] = (field(record, key), expected)
    if environment(record) != expected_environment:
        failures["environment"] = "scientific export drift"
    if failures:
        raise RuntimeError(f"E113-R4-R2 released job identity drifted: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.check == args.apply:
        raise SystemExit("pass exactly one of --check or --apply")
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate E113-R4-R2-S3: {ARTIFACT}")
    if not PROTOCOL.is_file():
        raise SystemExit(f"release protocol is absent: {PROTOCOL}")

    ledger = load_ledger()
    rows = {int(row["job_id"]): row for row in ledger.get("runs", [])}
    if sorted(rows) != list(COMPLETED_JOB_IDS + EXPECTED_JOB_IDS):
        raise RuntimeError("E113-R4-R2 effective run ledger drifted")

    records: dict[int, dict[str, str]] = {}
    for job_id in EXPECTED_JOB_IDS:
        row = rows[job_id]
        run_dir = Path(str(row["run_dir"]))
        if run_dir.exists():
            raise RuntimeError(f"zero-runtime E113-R4 run directory exists: {run_dir}")
        expected_environment = environment(str(row["held_scheduler_record"]))
        before = show(job_id)
        audit_held(before, expected_environment=expected_environment)
        records[job_id] = {
            "environment": expected_environment,
            "before": before,
            "family": str(row["family"]),
            "domain": str(row["domain"]),
            "seed": str(row["seed"]),
            "run_dir": str(run_dir),
        }

    if args.check:
        print(f"[e113r4r2s3] preflight_passed=True jobs={len(EXPECTED_JOB_IDS)}")
        return 0

    run(["scontrol", "release", *[str(value) for value in EXPECTED_JOB_IDS]])
    after: dict[int, str] = {}
    for job_id in EXPECTED_JOB_IDS:
        current = show(job_id)
        audit_released(current, expected_environment=records[job_id]["environment"])
        after[job_id] = current

    payload = {
        "schema": "e113r4r2s3_user_hold_release_v1",
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "authorization": (
            "User explicitly requested that E113-R4 official-verl DAPO resume "
            "where possible."
        ),
        "protocol": str(PROTOCOL),
        "protocol_sha256": launch.digest(PROTOCOL),
        "application": str(APPLICATION),
        "application_sha256": launch.digest(APPLICATION),
        "ledger": str(common.LEDGER),
        "ledger_sha256": launch.digest(common.LEDGER),
        "completed_job_ids_untouched": list(COMPLETED_JOB_IDS),
        "released_job_ids": list(EXPECTED_JOB_IDS),
        "action": {
            "command": "scontrol release "
            + " ".join(str(value) for value in EXPECTED_JOB_IDS),
            "exit_code": 0,
        },
        "records": [
            {
                "job_id": job_id,
                "family": records[job_id]["family"],
                "domain": records[job_id]["domain"],
                "seed": int(records[job_id]["seed"]),
                "run_dir": records[job_id]["run_dir"],
                "before": records[job_id]["before"],
                "after": after[job_id],
            }
            for job_id in EXPECTED_JOB_IDS
        ],
        "scientific_configuration_changed": False,
        "incomplete_endpoint_values_inspected": False,
        "other_campaigns_modified": False,
    }
    launch.atomic_json(ARTIFACT, payload)
    states = {field(record, "JobState") for record in after.values()}
    print(
        f"[e113r4r2s3] released={len(EXPECTED_JOB_IDS)} "
        f"states={','.join(sorted(states))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
