#!/usr/bin/env python3
"""Install or restore E114-R1-S3's owner-level next-slot priority fence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402


PROTOCOL = ROOT / "paper/preregistration/e114r1s3_next_slot_priority_fence_20260826.md"
ARTIFACT = ROOT / "var/artifacts/e114r1s3_next_slot_priority_fence.json"
E112_LEDGER = ROOT / "var/artifacts/e112r1_verified_support_discovery_full_three_scale_jobs.json"
E114_LEDGER = ROOT / "var/artifacts/e114_plain_grpo_qwen3b_extension_jobs.json"
TARGET = 30790267
STALE_COMPLETE = 30790272
COMPETITORS = (
    30791537, 30791538, 30791539, 30791540, 30791541, 30791542,
    30791543, 30791544, 30791545, 30791546, 30791549, 30791554,
)
TARGET_CHECKPOINT = ROOT / (
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


def atomic_json(payload: dict[str, Any]) -> None:
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(ARTIFACT)


def ledger_records(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, {int(row["job_id"]): row for row in payload.get("runs", [])}


def expected_environment(row: dict[str, Any]) -> str:
    return scheduler.environment(str(row["held_scheduler_record"]))


def validate_science(job_id: int, record: str, row: dict[str, Any]) -> None:
    environment = scheduler.environment(record)
    frozen = expected_environment(row)
    failures: dict[str, Any] = {}
    if scheduler.sha256_text(environment) != scheduler.sha256_text(frozen):
        failures["environment_sha256"] = (
            scheduler.sha256_text(environment), scheduler.sha256_text(frozen)
        )
    if f"SAVE_PATH={row['run_dir']}" not in environment:
        failures["SAVE_PATH"] = ("drifted", row["run_dir"])
    if failures:
        raise RuntimeError(f"job {job_id} scientific identity drifted: {failures}")


def validate_target(record: str, row: dict[str, Any]) -> None:
    expected = {
        "JobName": "e114-q3-mathir-s72",
        "JobState": "PENDING",
        "Partition": "mltheory",
        "Account": "mltheory",
        "ReqNodeList": "node302",
        "TimeLimit": "1-00:00:00",
        "Nice": "0",
        "TresPerNode": "gres/gpu:a100:1",
    }
    failures = {
        key: (scheduler.field(record, key), value)
        for key, value in expected.items()
        if scheduler.field(record, key) != value
    }
    if failures:
        raise RuntimeError(f"target E114 job drifted: {failures}")
    validate_science(TARGET, record, row)
    validation = run(
        [sys.executable, str(ROOT / "ops/validate_deepspeed_checkpoint.py"),
         "--checkpoint", str(TARGET_CHECKPOINT)],
        check=False,
    )
    if validation.returncode != 0:
        raise RuntimeError(f"target checkpoint invalid: {validation.stderr.strip()}")


def validate_competitor(
    job_id: int, record: str, row: dict[str, Any], *, nice: str
) -> None:
    expected = {
        "JobState": "PENDING",
        "Partition": "mltheory",
        "Account": "mltheory",
        "ReqNodeList": "node302",
        "TimeLimit": "3-00:00:00",
        "Nice": nice,
        "TresPerNode": "gres/gpu:a100:1",
    }
    failures = {
        key: (scheduler.field(record, key), value)
        for key, value in expected.items()
        if scheduler.field(record, key) != value
    }
    if scheduler.field(record, "Dependency") != "(null)":
        failures["Dependency"] = (scheduler.field(record, "Dependency"), "(null)")
    if failures:
        raise RuntimeError(f"E112 competitor {job_id} drifted: {failures}")
    validate_science(job_id, record, row)


def stale_receipt(row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    receipt = Path(str(row["run_dir"])) / "TRAINING_COMPLETE.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "oat_zero_training_complete_v1"
        or int(payload.get("terminal_step", -1)) < 3072
        or not Path(str(payload.get("terminal_export", ""))).is_dir()
    ):
        raise RuntimeError("stale Pantry job lacks a valid terminal receipt")
    return receipt, payload


def install() -> int:
    if ARTIFACT.exists():
        raise SystemExit(f"refusing duplicate priority fence: {ARTIFACT}")
    e112_payload, e112 = ledger_records(E112_LEDGER)
    e114_payload, e114 = ledger_records(E114_LEDGER)
    if e112_payload.get("released") is not True or e114_payload.get("released") is not True:
        raise RuntimeError("source ledgers are not released")
    if set(COMPETITORS) - set(e112) or {TARGET, STALE_COMPLETE} - set(e114):
        raise RuntimeError("source ledgers lack an exact priority-fence job")

    target_before = scheduler.show(TARGET)
    validate_target(target_before, e114[TARGET])
    stale_before = scheduler.show(STALE_COMPLETE)
    validate_competitor(STALE_COMPLETE, stale_before, e114[STALE_COMPLETE], nice="100")
    receipt_path, receipt = stale_receipt(e114[STALE_COMPLETE])
    before = {job_id: scheduler.show(job_id) for job_id in COMPETITORS}
    for job_id in COMPETITORS:
        validate_competitor(job_id, before[job_id], e112[job_id], nice="100")

    changed: list[int] = []
    try:
        for job_id in COMPETITORS:
            run(["scontrol", "update", f"JobId={job_id}", "Nice=1000"])
            changed.append(job_id)
        fenced = {job_id: scheduler.show(job_id) for job_id in COMPETITORS}
        for job_id in COMPETITORS:
            validate_competitor(job_id, fenced[job_id], e112[job_id], nice="1000")
        run(["scancel", str(STALE_COMPLETE)])
        stale_after = run(
            ["sacct", "-X", "-j", str(STALE_COMPLETE), "-n", "-P",
             "-o", "JobIDRaw,JobName,State,Elapsed,ExitCode,NodeList"]
        ).stdout.strip()
        if "CANCELLED" not in stale_after:
            raise RuntimeError(f"stale Pantry row did not cancel: {stale_after}")
    except Exception:
        for job_id in changed:
            run(["scontrol", "update", f"JobId={job_id}", "Nice=100"], check=False)
        raise

    payload = {
        "schema": "e114r1s3_next_slot_priority_fence_v1",
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest(PROTOCOL),
        "application": str(Path(__file__).resolve().relative_to(ROOT)),
        "application_sha256": digest(Path(__file__).resolve()),
        "source_ledgers": {
            str(E112_LEDGER.relative_to(ROOT)): digest(E112_LEDGER),
            str(E114_LEDGER.relative_to(ROOT)): digest(E114_LEDGER),
        },
        "target_job_id": TARGET,
        "target_checkpoint": str(TARGET_CHECKPOINT),
        "target_checkpoint_step": 1728,
        "target_before_scheduler_record": target_before,
        "stale_completed_job_id": STALE_COMPLETE,
        "stale_terminal_receipt": str(receipt_path),
        "stale_terminal_receipt_sha256": digest(receipt_path),
        "stale_terminal_step": int(receipt["terminal_step"]),
        "stale_before_scheduler_record": stale_before,
        "stale_after_accounting_record": stale_after,
        "competitor_job_ids": list(COMPETITORS),
        "old_nice": 100,
        "fence_nice": 1000,
        "scheduler_only": True,
        "scientific_environment_changed": False,
        "outcomes_inspected": False,
        "installed": True,
        "restored": False,
        "records": [
            {
                "job_id": job_id,
                "environment_sha256": scheduler.sha256_text(
                    expected_environment(e112[job_id])
                ),
                "before_scheduler_record": before[job_id],
                "fenced_scheduler_record": fenced[job_id],
            }
            for job_id in COMPETITORS
        ],
    }
    atomic_json(payload)
    print(
        f"[installed] target={TARGET} competitors={len(COMPETITORS)} "
        f"stale_cancelled={STALE_COMPLETE}"
    )
    return 0


def restore() -> int:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if payload.get("installed") is not True:
        raise RuntimeError("priority fence was not installed")
    if payload.get("restored") is True:
        print("[restore] already restored")
        return 0
    target = scheduler.show(TARGET)
    state = scheduler.field(target, "JobState")
    if state not in {"RUNNING", "COMPLETED"}:
        raise SystemExit(f"target {TARGET} is {state}; refusing early restoration")
    restored: list[int] = []
    after: dict[int, str] = {}
    for job_id in COMPETITORS:
        current = run(["scontrol", "show", "job", "-dd", "-o", str(job_id)], check=False)
        if current.returncode != 0:
            continue
        record = current.stdout.strip()
        if scheduler.field(record, "JobState") == "PENDING":
            run(["scontrol", "update", f"JobId={job_id}", "Nice=100"])
            restored.append(job_id)
            after[job_id] = scheduler.show(job_id)
    payload["target_at_restoration"] = target
    payload["restored_at"] = datetime.now(timezone.utc).isoformat()
    payload["restored_job_ids"] = restored
    payload["restored_scheduler_records"] = {
        str(job_id): record for job_id, record in after.items()
    }
    payload["restored"] = True
    atomic_json(payload)
    print(f"[restored] target_state={state} competitors={len(restored)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing protocol: {PROTOCOL}")
    return install() if args.apply else restore()


if __name__ == "__main__":
    raise SystemExit(main())
