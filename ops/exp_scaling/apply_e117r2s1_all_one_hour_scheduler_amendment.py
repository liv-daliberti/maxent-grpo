#!/usr/bin/env python3
"""Move untouched E117-R2 cells to all with a one-hour limit."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e117r2_same_plumbing_repair as e117r2  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / e117r2.LEDGER
PROTOCOL = ROOT / (
    "paper/preregistration/" "e117r2s1_all_one_hour_scheduler_amendment_20260830.md"
)
ARTIFACT = ROOT / ("var/artifacts/" "e117r2s1_all_one_hour_scheduler_amendment.json")
COMPLETED_JOB_ID = 30970803
PENDING_JOB_IDS = tuple(range(30970804, 30970815))
AUDIT_JOB_ID = 30970815
AUDIT_ARTIFACT = ROOT / (
    "var/artifacts/" "e117r2_same_plumbing_component_preflight_audit_job.json"
)
OLD_PARTITION = "cs"
OLD_ACCOUNT = "allcs"
OLD_LIMIT = "02:00:00"
NEW_PARTITION = "all"
NEW_ACCOUNT = "mltheory"
NEW_LIMIT = "01:00:00"
QOS = "none"
AMENDMENT_KEY = "scheduler_all_one_hour_amendment"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def show(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout.strip()


def field(record: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S*)", record)
    if match is None:
        raise RuntimeError(f"scheduler record lacks {name}")
    return match.group(1)


def environment(record: str) -> str:
    match = re.search(r"(?:^|\s)SubmitLine=.*?--export=([^ ]+)", record)
    if match is None:
        raise RuntimeError("scheduler record lacks exported environment")
    return match.group(1)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def accounting(job_id: int) -> dict[str, str]:
    result = run(
        [
            "sacct",
            "-X",
            "-j",
            str(job_id),
            "--starttime",
            "2026-08-29",
            "-n",
            "-P",
            "-o",
            (
                "JobIDRaw,State,ExitCode,Elapsed,Start,End,NodeList,"
                "Account,Partition,QOS,Timelimit,Restarts"
            ),
        ]
    )
    for line in result.stdout.splitlines():
        values = line.split("|")
        if len(values) == 12 and values[0] == str(job_id):
            keys = (
                "job_id",
                "state",
                "exit_code",
                "elapsed",
                "start",
                "end",
                "node",
                "account",
                "partition",
                "qos",
                "time_limit",
                "restarts",
            )
            return dict(zip(keys, values, strict=True))
    raise RuntimeError(f"Slurm accounting lacks E117-R2 job {job_id}")


def validate_completed_reference() -> dict[str, str]:
    state = accounting(COMPLETED_JOB_ID)
    expected = {
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:11:07",
        "node": "node202",
        "account": OLD_ACCOUNT,
        "partition": OLD_PARTITION,
        "qos": QOS,
        "time_limit": OLD_LIMIT,
        "restarts": "0",
    }
    wrong = {
        key: (value, state.get(key))
        for key, value in expected.items()
        if state.get(key) != value
    }
    if wrong:
        raise RuntimeError(f"completed runtime reference drifted: {wrong}")
    return state


def expected_environment(row: dict[str, Any]) -> str:
    value = environment(str(row["held_scheduler_record"]))
    if sha256_text(value) != str(row["scientific_environment_sha256"]):
        raise RuntimeError(f"ledger environment hash drifted for job {row['job_id']}")
    return value


def validate_pending_record(
    row: dict[str, Any],
    record: str,
    *,
    partition: str,
    account: str,
    limit: str,
    held: bool,
    allow_running: bool = False,
) -> None:
    job_id = int(row["job_id"])
    state = field(record, "JobState")
    if allow_running:
        if state not in {"PENDING", "RUNNING"}:
            raise RuntimeError(f"released job {job_id} has unexpected state {state}")
    elif state != "PENDING":
        raise RuntimeError(f"job {job_id} is no longer pending")

    expected = {
        "Partition": partition,
        "Account": account,
        "QOS": QOS,
        "TimeLimit": limit,
        "Requeue": "1",
        "Restarts": "0",
        "NumCPUs": "8",
        "MinMemoryNode": "64G",
        "ReqNodeList": str(row["node"]),
        "TresPerNode": "gres/gpu:1",
        "Dependency": "(null)",
    }
    if not allow_running:
        expected["RunTime"] = "00:00:00"
    if held:
        expected["Reason"] = "JobHeldUser"
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if not allow_running and field(record, "NodeList") not in {
        "",
        "(null)",
    }:
        wrong["NodeList"] = ("unassigned", field(record, "NodeList"))
    expected_export = expected_environment(row)
    actual_export = environment(record)
    if actual_export != expected_export:
        wrong["Environment"] = (
            sha256_text(expected_export),
            sha256_text(actual_export),
        )
    if wrong:
        raise RuntimeError(f"E117-R2 job {job_id} scheduler identity drifted: {wrong}")


def validate_audit_job() -> str:
    receipt = json.loads(AUDIT_ARTIFACT.read_text(encoding="utf-8"))
    expected_ids = list(range(COMPLETED_JOB_ID, PENDING_JOB_IDS[-1] + 1))
    if (
        receipt.get("schema") != "e117r2_same_plumbing_component_preflight_audit_job_v1"
        or int(receipt.get("audit_job_id", -1)) != AUDIT_JOB_ID
        or receipt.get("dependency_job_ids") != expected_ids
        or receipt.get("released") is not True
    ):
        raise RuntimeError("durable E117-R2 audit dependency drifted")

    record = show(AUDIT_JOB_ID)
    expected = {
        "JobState": "PENDING",
        "Reason": "Dependency",
        "Account": "allcs",
        "Partition": "all",
        "QOS": "none",
        "TimeLimit": "01:00:00",
        "Requeue": "0",
    }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    dependency = field(record, "Dependency")
    missing = [job_id for job_id in PENDING_JOB_IDS if str(job_id) not in dependency]
    if missing:
        wrong["Dependency"] = (
            "all 11 unresolved science jobs",
            missing,
        )
    if str(COMPLETED_JOB_ID) in dependency:
        wrong["CompletedDependency"] = (
            "fulfilled member pruned",
            dependency,
        )
    if wrong:
        raise RuntimeError(f"E117-R2 audit job drifted: {wrong}")
    return record


def load_and_validate() -> (
    tuple[
        dict[str, Any],
        dict[int, dict[str, Any]],
        dict[str, str],
        str,
    ]
):
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    if payload.get("schema") != ("e117_same_plumbing_component_preflight_jobs_v1"):
        raise RuntimeError("unexpected E117-R2 ledger schema")
    if payload.get("released") is not True:
        raise RuntimeError("E117-R2 ledger is not released")
    if payload.get(AMENDMENT_KEY) is not None:
        raise RuntimeError("E117-R2 one-hour amendment is already recorded")
    if ARTIFACT.exists():
        raise RuntimeError(f"amendment artifact already exists: {ARTIFACT}")
    if int(payload.get("audit_job_id", -1)) != AUDIT_JOB_ID:
        raise RuntimeError("E117-R2 audit job ID drifted")

    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise RuntimeError("E117-R2 run graph drifted")
    rows = {int(row["job_id"]): deepcopy(row) for row in runs}
    if set(rows) != set(range(COMPLETED_JOB_ID, PENDING_JOB_IDS[-1] + 1)):
        raise RuntimeError("E117-R2 authoritative job IDs drifted")

    completed = validate_completed_reference()
    before: dict[str, str] = {}
    for job_id in PENDING_JOB_IDS:
        record = show(job_id)
        validate_pending_record(
            rows[job_id],
            record,
            partition=OLD_PARTITION,
            account=OLD_ACCOUNT,
            limit=OLD_LIMIT,
            held=False,
        )
        run_dir = Path(str(rows[job_id]["run_dir"]))
        if run_dir.exists():
            raise RuntimeError(f"zero-runtime job {job_id} already created {run_dir}")
        before[str(job_id)] = record
    audit_record = validate_audit_job()
    return payload, rows, completed, audit_record


def update_route(job_id: int, *, partition: str, account: str, limit: str) -> None:
    run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            f"Partition={partition}",
            f"Account={account}",
            f"TimeLimit={limit}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not PROTOCOL.is_file():
        raise SystemExit(f"missing prospective amendment: {PROTOCOL}")
    ledger_sha256_before = e117r2.e117.digest(LEDGER)
    payload, rows, completed, audit_record = load_and_validate()
    before = {str(job_id): show(job_id) for job_id in PENDING_JOB_IDS}
    validation = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": e117r2.e117.digest(PROTOCOL),
        "application": str(Path(__file__).resolve()),
        "application_sha256": e117r2.e117.digest(Path(__file__)),
        "ledger": str(LEDGER),
        "ledger_sha256_before": ledger_sha256_before,
        "completed_runtime_reference": completed,
        "exact_pending_job_ids": list(PENDING_JOB_IDS),
        "excluded_completed_job_id": COMPLETED_JOB_ID,
        "excluded_audit_job_id": AUDIT_JOB_ID,
        "audit_scheduler_record": audit_record,
        "from_partition": OLD_PARTITION,
        "to_partition": NEW_PARTITION,
        "from_account": OLD_ACCOUNT,
        "to_account": NEW_ACCOUNT,
        "from_time_limit": OLD_LIMIT,
        "to_time_limit": NEW_LIMIT,
        "qos": QOS,
        "scientific_environment_changed": False,
        "common_source_block_changed": False,
        "outcomes_inspected_for_amendment": False,
    }
    if not args.apply:
        print(json.dumps(validation, indent=2, sort_keys=True))
        print(
            "dry-run only; pass --apply to user-hold, amend, record, "
            "and release the 11 zero-runtime jobs"
        )
        return 0

    held_ids: list[int] = []
    changed_ids: list[int] = []
    ledger_written = False
    try:
        for job_id in PENDING_JOB_IDS:
            run(["scontrol", "uhold", str(job_id)])
            held_ids.append(job_id)
        held_records: dict[str, str] = {}
        for job_id in PENDING_JOB_IDS:
            record = show(job_id)
            validate_pending_record(
                rows[job_id],
                record,
                partition=OLD_PARTITION,
                account=OLD_ACCOUNT,
                limit=OLD_LIMIT,
                held=True,
            )
            held_records[str(job_id)] = record

        for job_id in PENDING_JOB_IDS:
            update_route(
                job_id,
                partition=NEW_PARTITION,
                account=NEW_ACCOUNT,
                limit=NEW_LIMIT,
            )
            changed_ids.append(job_id)
        amended_records: dict[str, str] = {}
        for job_id in PENDING_JOB_IDS:
            record = show(job_id)
            validate_pending_record(
                rows[job_id],
                record,
                partition=NEW_PARTITION,
                account=NEW_ACCOUNT,
                limit=NEW_LIMIT,
                held=True,
            )
            amended_records[str(job_id)] = record

        now = datetime.now(timezone.utc).isoformat()
        amendment = {
            "schema": ("e117r2s1_all_one_hour_scheduler_amendment_v1"),
            "recorded_at": now,
            **validation,
            "before_scheduler_records": before,
            "held_scheduler_records": held_records,
            "amended_held_scheduler_records": amended_records,
            "all_jobs_zero_runtime_when_amended": True,
            "all_jobs_held_and_audited_before_ledger_write": True,
            "released": False,
        }
        payload[AMENDMENT_KEY] = amendment
        e117r2.e117.e111.e81.atomic_json(LEDGER, payload)
        e117r2.e117.e111.e81.atomic_json(ARTIFACT, amendment)
        ledger_written = True

        run(
            [
                "scontrol",
                "release",
                *(str(job_id) for job_id in PENDING_JOB_IDS),
            ]
        )
        released_records: dict[str, str] = {}
        for job_id in PENDING_JOB_IDS:
            record = show(job_id)
            validate_pending_record(
                rows[job_id],
                record,
                partition=NEW_PARTITION,
                account=NEW_ACCOUNT,
                limit=NEW_LIMIT,
                held=False,
                allow_running=True,
            )
            if field(record, "Reason") == "JobHeldUser":
                raise RuntimeError(f"job {job_id} remained user-held")
            released_records[str(job_id)] = record

        amendment["released"] = True
        amendment["released_at"] = datetime.now(timezone.utc).isoformat()
        amendment["released_scheduler_records"] = released_records
        payload[AMENDMENT_KEY] = amendment
        e117r2.e117.e111.e81.atomic_json(LEDGER, payload)
        receipt = deepcopy(amendment)
        receipt["ledger_sha256_after_release"] = e117r2.e117.digest(LEDGER)
        e117r2.e117.e111.e81.atomic_json(ARTIFACT, receipt)
    except BaseException:
        if not ledger_written:
            for job_id in changed_ids:
                update_route(
                    job_id,
                    partition=OLD_PARTITION,
                    account=OLD_ACCOUNT,
                    limit=OLD_LIMIT,
                )
            for job_id in held_ids:
                run(["scontrol", "release", str(job_id)], check=False)
        raise

    print(
        "[e117r2s1] released=11 partition=all account=mltheory " "time_limit=01:00:00"
    )
    print(f"ledger: {LEDGER}")
    print(f"artifact: {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
