#!/usr/bin/env python3
"""Prepare three E120 node105 continuations; --apply executes a reviewed plan."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prioritize_e118_capacity_20260905 import command, exports, field, show, submit_tokens

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
from validate_deepspeed_checkpoint import select_latest_checkpoint, validate_checkpoint

LEDGER = ROOT / "var/artifacts/e120r1_scheduler_continuation_jobs.json"
DIRECTORY = ROOT / "var/artifacts/campaign_health_capacity_20260905"
PLAN = DIRECTORY / "e120_node105_plan.json"
TRANSACTION = DIRECTORY / "e120_node105_transaction.json"
AMENDMENT = ROOT / "paper/preregistration/e120_node105_backfill_20260905.md"
PATCH = ROOT / "var/artifacts/e120_runtime_recovery_20260905/runtime-amendment.json"
TARGETS = (31048530, 31048528, 31048527)
EXPECTED = {31048530: ("pantry_plan", 57, 2688), 31048528: ("graph_coloring", 57, 2496),
            31048527: ("graph_coloring", 56, 2496)}
EXCLUSION = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_count(tres: str) -> int:
    return int(dict(item.split("=", 1) for item in tres.split(",")).get("gres/gpu", 0))


def capacity(*, allow_queue: bool = False) -> dict[str, Any]:
    record = command(["scontrol", "show", "node", "-o", "node105"]).stdout
    free_gpu = gpu_count(field(record, "CfgTRES")) - gpu_count(field(record, "AllocTRES"))
    free_mem = int(field(record, "RealMemory")) - int(field(record, "AllocMem"))
    free_cpu = int(field(record, "CPUEfctv")) - int(field(record, "CPUAlloc"))
    fits_now = free_gpu >= 4 and free_mem >= 232 * 1024 and free_cpu >= 32
    if not fits_now and not allow_queue:
        raise RuntimeError(f"joint 3-Falcon + 1-E119 headroom changed: {free_gpu} GPUs, {free_mem} MiB, {free_cpu} CPUs")
    if (gpu_count(field(record, "CfgTRES")) < 4
            or int(field(record, "RealMemory")) < 232 * 1024
            or int(field(record, "CPUEfctv")) < 32):
        raise RuntimeError("configured node cannot accommodate the joint plan")
    if any(word in field(record, "State") for word in ("DOWN", "DRAIN", "FAIL")):
        raise RuntimeError("node105 is unavailable")
    if "mltheory" not in field(record, "Partitions").split(","):
        raise RuntimeError("node105 lost mltheory membership")
    return {"record": record, "free_gpus": free_gpu, "free_mem_mib": free_mem,
            "free_cpus": free_cpu, "requested_joint_gpus": 4, "requested_joint_mem_mib": 232 * 1024,
            "allow_queue": allow_queue, "joint_fits_now": fits_now}


def placed(original: list[str], old_id: int) -> list[str]:
    changes = {"--account": "mltheory", "--partition": "mltheory", "--nodelist": "node105",
               "--gres": "gpu:a5000:1", "--comment": f"e120-node105-20260905-old{old_id}"}
    result = []
    for token in original[:-1]:
        key = token.split("=", 1)[0]
        if key in changes or token == "--hold":
            if key in changes and "=" not in token:
                raise RuntimeError(f"unsupported split option {key}")
            continue
        result.append(token)
    result.extend(f"{key}={value}" for key, value in changes.items())
    result.extend(["--hold", original[-1]])
    if [t for t in result if t.startswith("--export=")] != [t for t in original if t.startswith("--export=")]:
        raise RuntimeError("scientific/runtime export bytes changed")
    if "pvl" in " ".join(result).lower():
        raise RuntimeError("E120 refuses PVL in submission")
    return result


def audit(record: str, item: dict[str, Any]) -> None:
    if "pvl" in record.lower():
        raise RuntimeError("E120 refuses PVL scheduler record")
    expected = {"JobState": "PENDING", "Reason": "JobHeldUser", "RunTime": "00:00:00",
                "Restarts": "0", "Account": "mltheory", "Partition": "mltheory",
                "ReqNodeList": "node105", "MinMemoryNode": "64G", "NumCPUs": "8",
                "TimeLimit": "3-00:00:00", "Nice": "100", "ExcNodeList": EXCLUSION,
                "Dependency": "(null)", "WorkDir": str(ROOT), "Requeue": "1"}
    for key, value in expected.items():
        if field(record, key) != value:
            raise RuntimeError(f"new held job drifted: {key}={field(record, key)}; expected {value}")
    tres = dict(value.split("=", 1) for value in field(record, "ReqTRES").split(","))
    if tres.get("gres/gpu") != "1" or tres.get("gres/gpu:a5000") != "1":
        raise RuntimeError("new held job must request exactly one A5000")
    observed = submit_tokens(record)
    if exports(observed) != exports(item["original_command"]):
        raise RuntimeError("new held job changed scientific/runtime exports")
    if observed[-1] != item["original_command"][-1]:
        raise RuntimeError("new held job changed the frozen execution wrapper")
    if field(record, "StdOut") != str(ROOT / f"slurm-{field(record, 'JobId')}.out"):
        raise RuntimeError("new stdout does not match the repaired E120 watchdog")


def patch_valid() -> None:
    for row in json.loads(PATCH.read_text())["files"]:
        if sha(Path(row["path"]).read_bytes()) != row["after_sha256"]:
            raise RuntimeError("E120 repaired runtime hash changed")


def prepare(*, allow_queue: bool = False) -> dict[str, Any]:
    patch_valid()
    raw = LEDGER.read_bytes()
    ledger = json.loads(raw)
    by_id = {int(row["continuation_job_id"]): row for row in ledger["continuations"]}
    items = []
    for job_id in TARGETS:
        row = by_id[job_id]
        domain, seed, checkpoint_step = EXPECTED[job_id]
        if (row["model_key"], row["domain"], int(row["seed"])) != ("falcon1b", domain, seed):
            raise RuntimeError("registered Falcon cell changed")
        record = show(job_id)
        if field(record, "JobState") != "PENDING" or field(record, "Reason") != "Priority":
            raise RuntimeError(f"job {job_id} is no longer ordinary pending")
        if "pvl" in record.lower() or field(record, "ExcNodeList") != EXCLUSION:
            raise RuntimeError("old E120 resource fence drifted")
        old_command = submit_tokens(record)
        env = exports(old_command)
        if env.get("SAVE_PATH") != row["run_dir"] or env.get("RUN_STAMP") != row["run_stamp"]:
            raise RuntimeError("original job identity exports changed")
        if env.get("OAT_ZERO_AUTO_RESUME") != "1" or env.get("OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING") != "fresh_frequency":
            raise RuntimeError("resume or replay treatment changed")
        run_dir = Path(row["run_dir"])
        if (run_dir / "TRAINING_COMPLETE.json").exists():
            raise RuntimeError("candidate has completed")
        checkpoint, rejected = select_latest_checkpoint(run_dir)
        if checkpoint is None or int(checkpoint.name[5:]) != checkpoint_step or validate_checkpoint(checkpoint):
            raise RuntimeError("candidate checkpoint changed or is invalid")
        items.append({"old_job_id": job_id, "original_job_id": row["original_job_id"],
                      "domain": domain, "seed": seed, "run_dir": row["run_dir"], "run_stamp": row["run_stamp"],
                      "checkpoint": str(checkpoint), "checkpoint_step": checkpoint_step,
                      "rejected_checkpoints": rejected, "before_record": record,
                      "original_command": old_command, "command": placed(old_command, job_id)})
    return {"schema": "e120-node105-backfill-plan-v1", "created_at": now(), "applied": False,
            "ledger": str(LEDGER), "ledger_sha256": sha(raw), "amendment": str(AMENDMENT),
            "amendment_sha256": sha(AMENDMENT.read_bytes()), "runtime_amendment": str(PATCH),
            "capacity": capacity(allow_queue=allow_queue), "allow_queue": allow_queue, "same_scientific_cells": True, "same_run_directories": True,
            "scientific_exports_byte_identical": True, "qwen3_holds_changed": False,
            "joint_order": "Register E120 first, then E119; Slurm admits each allocation when its requested resources are available.",
            "items": items}


def apply(plan: dict[str, Any]) -> None:
    if TRANSACTION.exists():
        raise RuntimeError("transaction already exists; inspect its recorded state before any recovery")
    if sha(LEDGER.read_bytes()) != plan["ledger_sha256"] or sha(AMENDMENT.read_bytes()) != plan["amendment_sha256"]:
        raise RuntimeError("reviewed ledger or amendment changed")
    patch_valid()
    transaction = copy.deepcopy(plan)
    transaction.update(applied=True, started_at=now(), events=[], ledger_committed=False)
    transaction["capacity_at_start"] = capacity(allow_queue=plan.get("allow_queue", False))
    save(TRANSACTION, transaction)

    def event(message: str) -> None:
        transaction["events"].append({"at": now(), "message": message})
        save(TRANSACTION, transaction)

    try:
        for item in transaction["items"]:
            old = show(item["old_job_id"])
            if field(old, "JobState") != "PENDING" or exports(submit_tokens(old)) != exports(item["original_command"]):
                raise RuntimeError("old pending state or identity changed before hold")
            command(["scontrol", "hold", str(item["old_job_id"])])
            old = show(item["old_job_id"])
            if field(old, "JobState") != "PENDING" or field(old, "Reason") != "JobHeldUser":
                raise RuntimeError("old job did not become safely held")
            item["old_held_record"] = old
            event(f"held old pending job {item['old_job_id']}")
        for item in transaction["items"]:
            event(f"submitting held replacement for {item['old_job_id']}")
            output = command(item["command"]).stdout.strip()
            item["new_job_id"] = int(output.split(";")[0])
            event(f"submitted held replacement {item['new_job_id']}")
            item["new_held_record"] = show(item["new_job_id"])
            audit(item["new_held_record"], item)
            event(f"audited held replacement {item['new_job_id']}")
        transaction["capacity_before_commit"] = capacity(allow_queue=plan.get("allow_queue", False))
        if sha(LEDGER.read_bytes()) != plan["ledger_sha256"]:
            raise RuntimeError("continuation ledger changed before commit")
        original = LEDGER.read_bytes()
        ledger = json.loads(original)
        for item in transaction["items"]:
            old = show(item["old_job_id"])
            if field(old, "JobState") != "PENDING" or field(old, "Reason") != "JobHeldUser":
                raise RuntimeError("old writer is not held before ledger commit")
            audit(show(item["new_job_id"]), item)
            row = next(r for r in ledger["continuations"] if r["continuation_job_id"] == item["old_job_id"])
            row["intermediate_job_ids"] = [*row.get("intermediate_job_ids", []), item["old_job_id"]]
            row.update(continuation_job_id=item["new_job_id"], released=False,
                       repair_kind="node105_mltheory_capacity_backfill", placement_amendment=str(AMENDMENT),
                       held_scheduler_record=item["new_held_record"],
                       new_placement={"account": "mltheory", "partition": "mltheory", "node": "node105",
                                      "gpu": "a5000", "cpus": 8, "memory": "64G", "nice": 100})
        (DIRECTORY / "e120_continuations.before-node105.json").write_bytes(original)
        save(LEDGER, ledger)
        transaction["ledger_committed"] = True
        transaction["committed_ledger_sha256"] = sha(LEDGER.read_bytes())
        event("registered all three held replacements in authoritative continuation ledger")
        for item in transaction["items"]:
            command(["scancel", str(item["old_job_id"])])
            old = show(item["old_job_id"])
            if field(old, "JobState") not in {"CANCELLED", "COMPLETED", "FAILED", "TIMEOUT"}:
                raise RuntimeError("old job is not inactive; replacement remains held")
            item["old_inactive_record"] = old
            event(f"cancelled superseded pending job {item['old_job_id']}")
        for item in transaction["items"]:
            command(["scontrol", "release", str(item["new_job_id"])])
            item["released_record"] = show(item["new_job_id"])
            if field(item["released_record"], "JobState") == "PENDING" and field(item["released_record"], "Reason") == "JobHeldUser":
                raise RuntimeError("replacement did not release")
            next(r for r in ledger["continuations"] if r["continuation_job_id"] == item["new_job_id"])["released"] = True
            event(f"released replacement {item['new_job_id']}")
        save(LEDGER, ledger)
        transaction.update(completed_at=now(), status="released", final_ledger_sha256=sha(LEDGER.read_bytes()))
        event("transaction complete; optimizer startup still requires verification")
    except BaseException as exc:
        transaction.update(status="stopped_for_review", error=repr(exc))
        event("stopped; held jobs and ledger evidence retained, no blind retry")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-queue", action="store_true", help="Prepare unchanged resource requests that may wait for Slurm capacity")
    args = parser.parse_args()
    if args.apply:
        apply(json.loads(PLAN.read_text()))
        print(TRANSACTION)
    else:
        if TRANSACTION.exists():
            raise RuntimeError("execution record exists; do not replace its reviewed plan")
        plan = prepare(allow_queue=args.allow_queue)
        save(PLAN, plan)
        print(json.dumps({"plan": str(PLAN), "scheduler_mutated": False,
                          "old_job_ids": list(TARGETS), "capacity": {k: v for k, v in plan['capacity'].items() if k != 'record'}}))


if __name__ == "__main__":
    main()
