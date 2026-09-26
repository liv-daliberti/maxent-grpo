#!/usr/bin/env python3
"""Audited scheduler-only E118 timeout recovery and deferred Pantry successors."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

import prioritize_e118_capacity_20260905 as base

ROOT = base.ROOT
DIRECTORY = ROOT / "var/artifacts/e118_timeout_recovery_20260906"
AUDIT = DIRECTORY / "transaction.json"
LOCK = ROOT / "var/artifacts/e118_ledger_promotion.lock"
SCRIPT = Path(__file__).resolve()
WRAPPER = DIRECTORY / "pantry_successor.slurm"
SELECTED = ((31048117, "replacement", 2880), (31048116, "replacement", 2688),
            (31073912, "successor", None), (31073907, "successor", None),
            (31073908, "successor", None))
TERMINAL = {"COMPLETED", "TIMEOUT", "FAILED", "CANCELLED", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def command(parts: list[str]) -> str:
    return subprocess.run(parts, capture_output=True, text=True, check=True).stdout


def save(audit: dict, event: str) -> None:
    audit["updated_at_utc"] = base.now()
    audit.setdefault("events", []).append({"at": audit["updated_at_utc"], "event": event})
    base.atomic(AUDIT, audit)


def accounting(job: int) -> str:
    rows = command(["sacct", "-n", "-X", "-P", "-j", str(job), "-o", "JobID,State"])
    states = [line.split("|")[1].split()[0].rstrip("+") for line in rows.splitlines()
              if line.split("|")[0] == str(job)]
    if len(states) != 1:
        raise RuntimeError(f"ambiguous accounting for {job}: {rows}")
    return states[0]


def completed(run: Path) -> bool:
    receipt = run / "TRAINING_COMPLETE.json"
    if not receipt.exists():
        return False
    data = json.loads(receipt.read_text())
    if data.get("schema") != "oat_zero_training_complete_v1" or data.get("terminal_step", 0) < 3072:
        raise RuntimeError(f"invalid completion receipt: {receipt}")
    return True


def altered(original: list[str], updates: dict[str, str], target: Path) -> list[str]:
    result = [token for token in original[:-1]
              if token.split("=", 1)[0] not in updates and token != "--hold"]
    result.extend(f"{key}={value}" for key, value in updates.items())
    result.extend(["--hold", str(target)])
    if base.nonroot_exports(result) != base.nonroot_exports(original):
        raise RuntimeError("scientific/runtime exports changed")
    return result


def prepare() -> dict:
    if AUDIT.exists():
        raise RuntimeError(f"transaction already exists: {AUDIT}")
    DIRECTORY.mkdir(exist_ok=True)
    source_bytes = base.LEDGER.read_bytes()
    source = json.loads(source_bytes)
    by_id = {int(row["job_id"]): row for row in source["runs"]}
    (DIRECTORY / "source_ledger.before.json").write_bytes(source_bytes)
    rows = []
    for old_id, kind, step in SELECTED:
        row = by_id[old_id]
        state = accounting(old_id)
        if state != ("TIMEOUT" if kind == "replacement" else "RUNNING"):
            raise RuntimeError(f"unexpected predecessor state {old_id}: {state}")
        if completed(Path(row["run_dir"])):
            raise RuntimeError(f"cell already complete: {old_id}")
        record = row["held_scheduler_record"] if kind == "replacement" else base.show(old_id)
        original = base.submit_tokens(record)
        frozen = Path(original[-1])
        validator = Path(source["snapshot_root"]) / "ops/validate_deepspeed_checkpoint.py"
        checkpoint = command(["python3", str(validator), "--select-under", row["run_dir"]]).strip()
        if not checkpoint or (step is not None and Path(checkpoint).name != f"step_{step:05d}"):
            raise RuntimeError(f"unexpected checkpoint for {old_id}: {checkpoint}")
        item = {key: row[key] for key in base.IDENTITY}
        item.update(old_job_id=old_id, kind=kind, before_state=state,
                    original_command=original, resume_checkpoint=checkpoint,
                    frozen_launcher=str(frozen), frozen_launcher_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                    new_job_id=None, released=False)
        rows.append(item)
    # The wrapper promotes metadata only after afterany has released the allocation;
    # it then executes the exact frozen scientific/runtime launcher.
    WRAPPER.write_text("#!/bin/bash\nset -euo pipefail\n"
                       f"cd {ROOT}\nexec /usr/bin/python3 {SCRIPT} --start-successor\n")
    WRAPPER.chmod(0o755)
    amendment = ("# E118 scheduler recovery amendment, 2026-09-06\n\n"
        "Authorized: recover existing stranded cells and arrange follow-on allocations for "
        "healthy Pantry learners. No scientific cells, models, data, seeds, objectives, "
        "optimization settings, frozen training source, evaluation or checkpoint cadence change.\n\n"
        "Countdown MaxRL s70 and Graph Re:Max s74 receive audited 128 GiB, 16 CPU, "
        "one GPU, 12-hour mltheory/node302 continuations, in that order through afterany. "
        "The previous allocations are terminal; source and aggregate ledgers are replaced before release.\n\n"
        "Pantry running predecessors 31073912, 31073907 and 31073908 continue untouched. "
        "Their afterany successors request 36 hours on allcs/lowprio/node208, whose partition "
        "allows seven days. The longer scheduler walltime covers remaining compute. Lowprio "
        "preemption keeps Slurm automatic requeue eligibility and checkpoint resume. "
        "No completion ledger is replaced until a successor actually starts after its predecessor. "
        "A file lock serializes source+aggregate promotion. A valid completion receipt skips "
        "training and preserves the finished predecessor. Same-ID successor restarts are idempotent. "
        "All jobs retain the existing explicit PVL node exclusion. Future 36-hour walltime "
        "failures still require operational monitoring; this amendment does not change TERM handling.\n")
    (DIRECTORY / "scheduler_amendment.md").write_text(amendment)
    audit = {"schema": "e118-timeout-recovery-20260906-v1", "created_at_utc": base.now(),
             "status": "prepared", "source_ledger": str(base.LEDGER),
             "source_ledger_sha256_before": base.sha(source_bytes),
             "scheduler_only": True, "scientific_configuration_changed": False,
             "outcomes_inspected": False, "rows": rows, "events": []}
    save(audit, "prepared exact checkpoint recovery and deferred successor plan")
    return audit


def check_held(record: str, item: dict) -> None:
    successor = item["kind"] == "successor"
    expected = {"JobState": "PENDING", "Reason": "JobHeldUser",
                "Account": "allcs" if successor else "mltheory",
                "Partition": "lowprio" if successor else "mltheory",
                "ReqNodeList": "node208" if successor else "node302",
                "MinMemoryNode": "128G", "NumCPUs": "16", "Requeue": "1",
                "TimeLimit": "1-12:00:00" if successor else "12:00:00",
                "ExcNodeList": base.PVL}
    for key, value in expected.items():
        if base.field(record, key) != value:
            raise RuntimeError(f"held audit {item['new_job_id']} {key}: {base.field(record, key)} != {value}")
    if "gres/gpu=1" not in base.field(record, "ReqTRES").split(","):
        raise RuntimeError("GPU request drift")
    observed = base.submit_tokens(record)
    if base.nonroot_exports(observed) != base.nonroot_exports(item["original_command"]):
        raise RuntimeError("export drift")
    env = base.exports(observed)
    if any(env.get(key) != value for key, value in base.ROOT_EXPORTS.items()):
        raise RuntimeError("root export drift")
    dependency = item.get("dependency")
    actual = base.field(record, "Dependency")
    if dependency and not actual.startswith(dependency):
        raise RuntimeError(f"dependency drift: {actual} != {dependency}")


def apply(audit: dict) -> None:
    rows = audit["rows"]
    prior = None
    for item in rows:
        if item.get("submission_uncertain"):
            raise RuntimeError("uncertain prior submission; reconcile scheduler comment before retry")
        if item["new_job_id"] is None:
            successor = item["kind"] == "successor"
            dependency = f"afterany:{item['old_job_id']}" if successor else (f"afterany:{prior}" if prior else None)
            env = base.exports(item["original_command"])
            env.update(base.ROOT_EXPORTS)
            updates = {"--account": "allcs" if successor else "mltheory",
                       "--partition": "lowprio" if successor else "mltheory",
                       "--nodelist": "node208" if successor else "node302",
                       "--time": "36:00:00" if successor else "12:00:00",
                       "--output": str(ROOT / "var/artifacts/logs/%x-%j.out"),
                       "--error": str(ROOT / "var/artifacts/logs/%x-%j.err"),
                       "--nodes": "1", "--ntasks": "1", "--ntasks-per-node": "1",
                       "--comment": f"e118-timeout-20260906-old{item['old_job_id']}",
                       "--export": "ALL," + ",".join(f"{key}={value}" for key,value in env.items())}
            if dependency:
                updates["--dependency"] = dependency
            item["dependency"] = dependency
            item["command"] = altered(item["original_command"], updates, WRAPPER if successor else Path(item["frozen_launcher"]))
            item["submission_uncertain"] = True
            save(audit, f"submitting held {item['kind']} for {item['old_job_id']}")
            result = command(item["command"])
            item["new_job_id"] = int(result.strip().split(";", 1)[0])
            item["submission_uncertain"] = False
            save(audit, f"submitted held job {item['new_job_id']}")
        record = base.show(item["new_job_id"])
        if not item["released"]:
            check_held(record, item)
            item["held_scheduler_record"] = record
            save(audit, f"audited held job {item['new_job_id']}")
        if item["kind"] == "replacement":
            prior = item["new_job_id"]
    if not audit.get("ledger_committed"):
        with LOCK.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            source_bytes = base.LEDGER.read_bytes()
            if base.sha(source_bytes) != audit["source_ledger_sha256_before"]:
                raise RuntimeError("authoritative source ledger changed before commit")
            source = json.loads(source_bytes)
            for item in rows:
                if item["kind"] != "replacement":
                    continue
                if accounting(item["old_job_id"]) != "TIMEOUT" or item["old_job_id"] in base.queue():
                    raise RuntimeError("stranded predecessor unexpectedly live")
                row = next(r for r in source["runs"] if int(r["job_id"]) == item["old_job_id"])
                if any(row[key] != item[key] for key in base.IDENTITY):
                    raise RuntimeError("scientific cell identity drift")
                row["previous_job_ids"] = [*row.get("previous_job_ids", []), item["old_job_id"]]
                row["job_id"] = item["new_job_id"]
                row["held_scheduler_record"] = item["held_scheduler_record"]
            source.setdefault("repair_history", []).append({"at": base.now(), "audit": str(AUDIT),
                "scheduler_only": True, "replacements": [{"old_job_id": i["old_job_id"], "new_job_id": i["new_job_id"]} for i in rows if i["kind"] == "replacement"]})
            base.atomic(base.LEDGER, source)
            command(["python3", str(ROOT / "ops/exp_scaling/build_e118_aggregate_ledger.py")])
            audit["ledger_committed"] = True
            save(audit, "committed dead-cell replacement source and aggregate; active Pantry predecessors preserved")
    for item in rows:
        if not item["released"]:
            command(["scontrol", "release", str(item["new_job_id"])])
            item["released"] = True
            item["release_record"] = base.show(item["new_job_id"])
            save(audit, f"released job {item['new_job_id']}")
    audit["status"] = "released"
    save(audit, "all replacements and deferred successors released")


def start_successor() -> None:
    job = int(os.environ["SLURM_JOB_ID"])
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        audit = json.loads(AUDIT.read_text())
        item = next(i for i in audit["rows"] if i["new_job_id"] == job and i["kind"] == "successor")
        old = item["old_job_id"]
        if old in base.queue():
            raise RuntimeError(f"predecessor {old} is still active; refuse simultaneous writer")
        if accounting(old) not in TERMINAL:
            raise RuntimeError("predecessor is not terminal")
        if completed(Path(item["run_dir"])):
            save(audit, f"successor {job} skipped: predecessor cell has completion receipt")
            print(f"[e118-successor] {job}: completed cell; no retraining", flush=True)
            return
        frozen = Path(item["frozen_launcher"])
        if hashlib.sha256(frozen.read_bytes()).hexdigest() != item["frozen_launcher_sha256"]:
            raise RuntimeError("frozen launcher changed")
        validator = frozen.parents[1] / "validate_deepspeed_checkpoint.py"
        checkpoint = command(["python3", str(validator), "--select-under", item["run_dir"]]).strip()
        if not checkpoint:
            raise RuntimeError("successor lacks a valid checkpoint; refuse restart from zero")
        source = json.loads(base.LEDGER.read_text())
        row = next(r for r in source["runs"] if r["run_dir"] == item["run_dir"])
        if any(row[key] != item[key] for key in base.IDENTITY):
            raise RuntimeError("successor scientific cell identity drift")
        if int(row["job_id"]) not in {old, job}:
            raise RuntimeError("cell has another authoritative writer")
        if int(row["job_id"]) == old:
            row["previous_job_ids"] = [*row.get("previous_job_ids", []), old]
            row["job_id"] = job
            row["held_scheduler_record"] = item["held_scheduler_record"]
            source.setdefault("repair_history", []).append({"at": base.now(), "audit": str(AUDIT),
                "scheduler_only": True, "deferred_successor_started": {"old_job_id": old, "new_job_id": job}})
            base.atomic(base.LEDGER, source)
        # Always rebuild under lock, including if a previous restart committed only source.
        command(["python3", str(ROOT / "ops/exp_scaling/build_e118_aggregate_ledger.py")])
        item["promoted_at_utc"] = base.now()
        save(audit, f"promoted terminal predecessor {old} to successor {job}")
        print(f"[e118-successor] {job}: predecessor={old}, resume={checkpoint}", flush=True)
    os.chdir(ROOT)
    os.execv("/bin/bash", ["/bin/bash", str(frozen)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--start-successor", action="store_true")
    args = parser.parse_args()
    if args.start_successor:
        start_successor()
    elif args.prepare:
        prepare()
    else:
        apply(json.loads(AUDIT.read_text()))


if __name__ == "__main__":
    main()
