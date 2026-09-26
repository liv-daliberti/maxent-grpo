#!/usr/bin/env python3
"""Repair the four E115 Qwen-0.5B cells routed away from node105."""

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
import launch_e78_verified_replay_only_05b as e78  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e115_qwen05b_badconstraints_partition_repair_20260821.md"
)
LEDGER = ROOT / (
    "var/artifacts/e115_ucpo_qwen05b_domain_extension_jobs.json"
)
PARENT_LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
OUT = ROOT / (
    "var/artifacts/e115_qwen05b_badconstraints_partition_repair.json"
)
OLD_PARTITION = "cs"
OLD_ACCOUNT = "allcs"
NEW_PARTITION = "mltheory"
NEW_ACCOUNT = "mltheory"
NODE = "node105"
GPU = "gres/gpu:a5000:1"
TIME_LIMIT = "1-12:00:00"
TARGETS = {
    30790285: ("countdown", 46),
    30790286: ("countdown", 47),
    30790290: ("mathir", 46),
    30790291: ("mathir", 47),
}


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


def require_partition_membership() -> str:
    record = run(["scontrol", "show", "partition", NEW_PARTITION])
    nodes = re.search(r"\sNodes=(\S+)", record)
    if not nodes:
        raise RuntimeError(f"cannot read {NEW_PARTITION} node membership")
    listed = run(["scontrol", "show", "hostnames", nodes.group(1)]).split()
    if NODE not in listed:
        raise RuntimeError(f"{NODE} is not a member of {NEW_PARTITION}")
    allowed = re.search(r"\sAllowAccounts=(\S+)", record)
    if not allowed or NEW_ACCOUNT not in allowed.group(1).split(","):
        raise RuntimeError(f"{NEW_ACCOUNT} may not use {NEW_PARTITION}")
    inventory = run(["sinfo", "-h", "-N", "-n", NODE, "-o", "%N|%P|%G|%m|%T"])
    if not any(
        line.startswith(f"{NODE}|{NEW_PARTITION}|") and "gpu:a5000:" in line
        for line in inventory.splitlines()
    ):
        raise RuntimeError(f"{NODE} exposes no A5000 in {NEW_PARTITION}")
    return inventory


def ledger_runs() -> dict[int, dict[str, Any]]:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    if payload.get("released") is not True:
        raise RuntimeError("E115 Qwen-0.5B was not durably released")
    if payload.get("ucpo_tau") != 0.2 or payload.get("variant") != "ucpo":
        raise RuntimeError("E115 objective drifted")
    selected: dict[int, dict[str, Any]] = {}
    for record in payload.get("runs", []):
        job_id = int(record["job_id"])
        if job_id not in TARGETS:
            continue
        domain, seed = TARGETS[job_id]
        if str(record["domain"]) != domain or int(record["seed"]) != seed:
            raise RuntimeError(f"E115 target drifted in the ledger: {job_id}")
        selected[job_id] = record
    if set(selected) != set(TARGETS):
        raise RuntimeError("E115 ledger does not carry all four routed cells")
    return selected


def parent_placement() -> dict[tuple[str, int], str]:
    payload = json.loads(PARENT_LEDGER.read_text(encoding="utf-8"))
    found: dict[tuple[str, int], str] = {}
    for record in payload.get("runs", []):
        key = (str(record.get("domain")), int(record.get("seed", -1)))
        if key not in {(domain, seed) for domain, seed in TARGETS.values()}:
            continue
        frozen = str(record.get("held_scheduler_record", ""))
        if f"--nodelist={NODE}" not in frozen or "--gres=gpu:a5000:1" not in frozen:
            raise RuntimeError(f"E78 parent placement drifted for {key}")
        found[key] = f"{NODE}/a5000"
    if set(found) != {(domain, seed) for domain, seed in TARGETS.values()}:
        raise RuntimeError("E78 parent cells for the four targets are absent")
    return found


def validate(
    job_id: int,
    record: str,
    frozen: dict[str, Any],
    *,
    partition: str,
    account: str,
    before: bool,
) -> None:
    expected = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "TimeLimit": TIME_LIMIT,
        "Partition": partition,
        "Account": account,
        "ReqNodeList": NODE,
        "NumCPUs": "8",
        "MinMemoryNode": "64G",
        "TresPerNode": GPU,
        "Restarts": "0",
    }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if wrong:
        raise RuntimeError(f"E115 job {job_id} scheduler state drifted: {wrong}")
    frozen_env = export_map(str(frozen["held_scheduler_record"]))
    live_env = export_map(record)
    if frozen_env != live_env:
        drift = {
            key: (frozen_env.get(key), live_env.get(key))
            for key in set(frozen_env) | set(live_env)
            if frozen_env.get(key) != live_env.get(key)
        }
        raise RuntimeError(f"E115 job {job_id} environment drifted: {drift}")
    if str(frozen["run_dir"]) != live_env.get("SAVE_PATH"):
        raise RuntimeError(f"E115 job {job_id} output path drifted")
    if Path(str(frozen["run_dir"])).exists():
        raise RuntimeError(f"E115 job {job_id} already has a run directory")
    reason = field(record, "Reason")
    if before and reason != "BadConstraints":
        raise RuntimeError(f"E115 job {job_id} is not the routed-away cell")
    if not before and reason in {"BadConstraints", "JobHeldUser"}:
        raise RuntimeError(f"E115 job {job_id} is still unschedulable: {reason}")


def update(job_id: int, *, partition: str, account: str) -> None:
    run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            f"Partition={partition}",
            f"Account={account}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"E115 repair protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate E115 repair: {OUT}")

    inventory = require_partition_membership()
    parents = parent_placement()
    frozen = ledger_runs()
    before = {job_id: scheduler_record(job_id) for job_id in TARGETS}
    for job_id in TARGETS:
        validate(
            job_id,
            before[job_id],
            frozen[job_id],
            partition=OLD_PARTITION,
            account=OLD_ACCOUNT,
            before=True,
        )

    if not args.apply:
        for job_id in TARGETS:
            print(
                f"scontrol update JobId={job_id} Partition={NEW_PARTITION} "
                f"Account={NEW_ACCOUNT}"
            )
        print(f"[e115-repair] dry_run=True jobs={len(TARGETS)}")
        return 0

    changed: list[int] = []
    try:
        for job_id in TARGETS:
            update(job_id, partition=NEW_PARTITION, account=NEW_ACCOUNT)
            changed.append(job_id)
        after = {job_id: scheduler_record(job_id) for job_id in TARGETS}
        for job_id in TARGETS:
            validate(
                job_id,
                after[job_id],
                frozen[job_id],
                partition=NEW_PARTITION,
                account=NEW_ACCOUNT,
                before=False,
            )
    except Exception:
        for job_id in changed:
            try:
                update(job_id, partition=OLD_PARTITION, account=OLD_ACCOUNT)
            except Exception:  # noqa: BLE001 - restore as much as possible
                pass
        raise

    payload = {
        "schema": "e115_qwen05b_badconstraints_partition_repair_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": e78.digest(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": e78.digest(Path(__file__)),
        "ledger": str(LEDGER.relative_to(ROOT)),
        "ledger_sha256": e78.digest(LEDGER),
        "parent_ledger": str(PARENT_LEDGER.relative_to(ROOT)),
        "parent_ledger_sha256": e78.digest(PARENT_LEDGER),
        "parent_placement": {f"{k[0]}/s{k[1]}": v for k, v in parents.items()},
        "scheduler_only": True,
        "environment_changed": False,
        "gpu_type_changed": False,
        "node_changed": False,
        "walltime_changed": False,
        "partition_changed": True,
        "account_changed": True,
        "run_directories_touched": False,
        "e115_outcomes_inspected": False,
        "pointmaze": "excluded",
        "inventory": inventory,
        "jobs": [
            {
                "job_id": job_id,
                "scale": "qwen05b",
                "domain": TARGETS[job_id][0],
                "seed": TARGETS[job_id][1],
                "arm": str(frozen[job_id].get("arm", "ucpo")),
                "run_stamp": str(frozen[job_id]["run_stamp"]),
                "old_partition": OLD_PARTITION,
                "new_partition": NEW_PARTITION,
                "old_account": OLD_ACCOUNT,
                "new_account": NEW_ACCOUNT,
                "node": NODE,
                "gres": GPU,
                "reason_before": field(before[job_id], "Reason"),
                "reason_after": field(after[job_id], "Reason"),
                "priority_after": field(after[job_id], "Priority"),
                "before": before[job_id],
                "after": after[job_id],
            }
            for job_id in TARGETS
        ],
    }
    e78.atomic_json(OUT, payload)
    print(f"[e115-repair] jobs={len(TARGETS)} artifact={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
