#!/usr/bin/env python3
"""Repair the submit-routed placement of four pending E116 science cells."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e116s1_qwen05b_science_route_repair_20260825.md"
)
LEDGER = ROOT / "var/artifacts/e116_sparse_rlep_qwen05b_domain_extension_jobs.json"
PRIOR_REPAIR = ROOT / "var/artifacts/scheduler_acceleration_package_20260821.json"
OUT = ROOT / "var/artifacts/e116s1_qwen05b_science_route_repair.json"

TARGETS = {
    30790408: {"domain": "countdown", "seed": 46, "pool_audit": 30790389},
    30790409: {"domain": "countdown", "seed": 47, "pool_audit": 30790391},
    30790413: {"domain": "mathir", "seed": 46, "pool_audit": 30790400},
    30790414: {"domain": "mathir", "seed": 47, "pool_audit": 30790402},
}
SMOKE_AUDIT = 30790404


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}: {detail}")
    return result.stdout.strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduler_record(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def field(record: str, key: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S*)", record)
    return match.group(1) if match else None


def export_map(record: str) -> dict[str, str]:
    match = re.search(r"--export=ALL,(\S+)", record)
    if not match:
        raise RuntimeError("scheduler record has no frozen --export block")
    result: dict[str, str] = {}
    for item in match.group(1).split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    if not result:
        raise RuntimeError("scheduler export block is empty")
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ledger_records() -> dict[int, dict[str, Any]]:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    if (
        payload.get("released") is not True
        or payload.get("schema") != "e116_sparse_rlep_completion_jobs_v1"
        or payload.get("cohort") != "e116_qwen05b"
        or payload.get("model") != "Qwen/Qwen2.5-0.5B-Instruct"
        or payload.get("domains") != ["countdown", "mathir"]
        or payload.get("seeds") != [43, 44, 45, 46, 47]
        or payload.get("variant") != "rlep"
    ):
        raise RuntimeError("E116 Qwen-0.5B ledger identity drifted")
    if int(payload.get("target_steps", -1)) != 3072:
        raise RuntimeError("E116 terminal horizon drifted")
    selected: dict[int, dict[str, Any]] = {}
    for record in payload.get("runs", []):
        job_id = int(record.get("job_id", -1))
        if job_id not in TARGETS:
            continue
        target = TARGETS[job_id]
        if (
            record.get("arm") != "rlep_dr_sparse"
            or record.get("domain") != target["domain"]
            or int(record.get("seed", -1)) != target["seed"]
            or int(record.get("pool_audit_dependency_job_id", -1))
            != target["pool_audit"]
            or int(record.get("smoke_audit_dependency_job_id", -1))
            != SMOKE_AUDIT
        ):
            raise RuntimeError(f"E116 ledger target drifted: {job_id}")
        selected[job_id] = record
    if set(selected) != set(TARGETS):
        raise RuntimeError("E116 ledger lacks the exact four route-repair targets")
    return selected


def require_inventory() -> dict[str, str]:
    node = run(["sinfo", "-h", "-N", "-n", "node105", "-o", "%N|%P|%G|%m|%T"])
    matches = [
        line for line in node.splitlines()
        if line.startswith("node105|mltheory|") and "gpu:a5000:" in line
    ]
    if len(matches) != 1 or int(matches[0].split("|")[3].rstrip("+")) < 64 * 1024:
        raise RuntimeError("node105 A5000/mltheory inventory drifted")
    partition = run(["scontrol", "show", "partition", "mltheory", "-o"])
    accounts = (field(partition, "AllowAccounts") or "").split(",")
    if "mltheory" not in accounts:
        raise RuntimeError("mltheory partition no longer authorizes node105/account")
    return {"node105": node, "mltheory": partition}


def require_audit(job_id: int) -> str:
    record = run([
        "sacct", "-X", "-n", "-P", "-j", str(job_id),
        "-o", "JobIDRaw,State,ExitCode",
    ])
    rows = [line for line in record.splitlines() if line]
    if rows != [f"{job_id}|COMPLETED|0:0"]:
        raise RuntimeError(f"E116 prerequisite audit {job_id} is not successful: {rows}")
    return rows[0]


def require_environment(job_id: int, live: str, frozen: str) -> None:
    live_env, frozen_env = export_map(live), export_map(frozen)
    if live_env != frozen_env:
        drift = {
            key: (frozen_env.get(key), live_env.get(key))
            for key in sorted(set(frozen_env) | set(live_env))
            if frozen_env.get(key) != live_env.get(key)
        }
        raise RuntimeError(f"E116 job {job_id} environment drifted: {drift}")


def validate_before(job_id: int, live: str, frozen: dict[str, Any]) -> None:
    expected = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Partition": "cs",
        "Account": "allcs",
        "ReqNodeList": "node105",
        "NumCPUs": "8",
        "MinMemoryNode": "64G",
        "TimeLimit": "1-12:00:00",
        "Nice": "100",
        "TresPerNode": "gres/gpu:a5000:1",
        "Dependency": "(null)",
    }
    wrong = {
        key: (value, field(live, key))
        for key, value in expected.items()
        if field(live, key) != value
    }
    if wrong:
        raise RuntimeError(f"E116 job {job_id} preflight drifted: {wrong}")
    require_environment(job_id, live, str(frozen["held_scheduler_record"]))
    if Path(str(frozen["run_dir"])).exists():
        raise RuntimeError(f"E116 zero-runtime target has a run directory: {job_id}")


def validate_after(job_id: int, live: str, frozen: dict[str, Any]) -> None:
    if field(live, "JobState") not in {"PENDING", "RUNNING"}:
        raise RuntimeError(f"E116 job {job_id} post-state drifted")
    expected = {
        "Partition": "mltheory",
        "Account": "mltheory",
        "ReqNodeList": "node105",
        "NumCPUs": "8",
        "MinMemoryNode": "64G",
        "TimeLimit": "1-12:00:00",
        "Nice": "100",
        "TresPerNode": "gres/gpu:a5000:1",
    }
    wrong = {
        key: (value, field(live, key))
        for key, value in expected.items()
        if field(live, key) != value
    }
    if wrong:
        raise RuntimeError(f"E116 job {job_id} postflight drifted: {wrong}")
    require_environment(job_id, live, str(frozen["held_scheduler_record"]))
    if field(live, "JobState") == "PENDING" and field(live, "Reason") in {
        "BadConstraints", "JobHeldUser"
    }:
        raise RuntimeError(f"E116 job {job_id} remains ineligible after repair")


def update(job_id: int, *, partition: str, account: str) -> None:
    run([
        "scontrol", "update", f"JobId={job_id}",
        f"Partition={partition}", f"Account={account}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for required in (PROTOCOL, LEDGER, PRIOR_REPAIR):
        if not required.is_file():
            raise SystemExit(f"required frozen input is absent: {required}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate E116-S1 application: {OUT}")

    frozen = ledger_records()
    inventory = require_inventory()
    audits = {
        str(job_id): require_audit(job_id)
        for job_id in sorted({SMOKE_AUDIT, *(row["pool_audit"] for row in TARGETS.values())})
    }
    before = {job_id: scheduler_record(job_id) for job_id in TARGETS}
    for job_id in TARGETS:
        validate_before(job_id, before[job_id], frozen[job_id])

    if not args.apply:
        for job_id in TARGETS:
            print(
                f"scontrol update JobId={job_id} "
                "Partition=mltheory Account=mltheory"
            )
        print(f"[e116-s1-route] dry_run=True jobs={len(TARGETS)}")
        return 0

    changed: list[int] = []
    try:
        for job_id in TARGETS:
            update(job_id, partition="mltheory", account="mltheory")
            changed.append(job_id)
        time.sleep(2)
        after = {job_id: scheduler_record(job_id) for job_id in TARGETS}
        for job_id in TARGETS:
            validate_after(job_id, after[job_id], frozen[job_id])
    except Exception:
        for job_id in reversed(changed):
            try:
                live = scheduler_record(job_id)
                if field(live, "JobState") == "PENDING":
                    update(job_id, partition="cs", account="allcs")
            except Exception:  # noqa: BLE001 - best-effort scheduler rollback
                pass
        raise

    payload = {
        "schema": "e116s1_qwen05b_science_route_repair_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "author_approved": True,
        "scheduler_only": True,
        "scientific_environment_changed": False,
        "gpu_type_changed": False,
        "walltime_changed": False,
        "dependencies_changed": False,
        "run_directories_touched": False,
        "outcomes_inspected": False,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": digest(Path(__file__)),
        "ledger": str(LEDGER.relative_to(ROOT)),
        "ledger_sha256": digest(LEDGER),
        "prior_repair": str(PRIOR_REPAIR.relative_to(ROOT)),
        "prior_repair_sha256": digest(PRIOR_REPAIR),
        "inventory": inventory,
        "prerequisite_audits": audits,
        "jobs": [
            {
                "job_id": job_id,
                "domain": TARGETS[job_id]["domain"],
                "seed": TARGETS[job_id]["seed"],
                "run_dir": frozen[job_id]["run_dir"],
                "before": before[job_id],
                "after": after[job_id],
            }
            for job_id in TARGETS
        ],
    }
    atomic_json(OUT, payload)
    print(
        f"[e116-s1-route] applied=True jobs={len(TARGETS)} "
        f"artifact={OUT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
