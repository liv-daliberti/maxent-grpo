#!/usr/bin/env python3
"""Reconcile the single audited E122 submission, then submit only cells 1..99 held."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e122_level3_factorial as original
import e122_slurm2511_adapter as compatibility

PLAN_SHA256 = "67c506a40b9a7fb7b984335d2ac47c859e01c9a62e5eddba5891aca9401cec2f"
INITIAL_LEDGER_SHA256 = "e02c8395be9ad2cd8f0246cd452b38687e8a480456e716ce18e1236f432997b6"
FIRST_JOB_ID = 31158645
SOURCE = Path(__file__).resolve()
TEST = original.ROOT / "tests/test_continue_e122_slurm2511.py"
require = original.require


def reconcile_journals(plan, ledger):
    """Only the exact known successful first submission can be resumed."""
    original.verify_initial_ledger(ledger, plan["cells"])
    require(original.digest(original.LEDGER) == INITIAL_LEDGER_SHA256, "prospective ledger changed")
    claim = original.read(original.CLAIM)
    require(claim["schema"] == "e122_once_only_submission_claim_v1"
            and claim["plan_sha256"] == PLAN_SHA256
            and claim["initial_ledger_sha256"] == INITIAL_LEDGER_SHA256
            and claim["jobs"] == 100, "original claim differs")
    expected = {"submission_000_intent.json", "submission_000_result.json"}
    actual = {p.name for pattern in ("submission_*_intent.json", "submission_*_result.json")
              for p in original.HERE.glob(pattern)}
    require(actual == expected and not list(original.HERE.glob("held_audit_*.json")),
            "partial execution differs from the reviewed single submission; never retry")
    for name in ("slurm2511_continuation_intent.json", "held_submission_complete.json",
                 "prospective_ledger_before_submission.json"):
        require(not (original.HERE / name).exists(), "continuation was already claimed or completed")
    intent = original.read(original.HERE / "submission_000_intent.json")
    result = original.read(original.HERE / "submission_000_result.json")
    require(intent["cell"] == plan["cells"][0]
            and intent["command_sha256"] == original.sha(plan["cells"][0]["command"]),
            "first submitted command differs from frozen plan")
    require(result.get("returncode") == 0
            and re.fullmatch(r"[1-9][0-9]*(?:;[^\s]+)?\s*", result.get("stdout", "")) is not None
            and int(result["stdout"].strip().split(";", 1)[0]) == FIRST_JOB_ID,
            "first submission is not the exact known success; never resubmit it")
    require(not any(Path(cell["run_dir"]).exists() for cell in plan["cells"]),
            "a training run directory already exists")
    return FIRST_JOB_ID


def reconcile_queue(first_job_id):
    result = subprocess.run(["squeue", "--noheader", "--user", str(os.getuid()),
                             "--format=%i|%j|%T|%r"], capture_output=True, text=True, timeout=45)
    require(result.returncode == 0, "cannot establish current E122 queue: " + result.stderr)
    rows = [line.strip().split("|", 3) for line in result.stdout.splitlines() if line.strip()]
    e122 = [row for row in rows if len(row) == 4 and row[1].startswith("e122_level3_")]
    require(e122 == [[str(first_job_id), "e122_level3_countdown_drgrpo_s43", "PENDING", "JobHeldUser"]],
            "unexpected or ambiguous E122 scheduler activity")
    # The caller holds the shared submission lock. The original submitter ran
    # on rinse; this process may run on wash, so local PID absence is not proof.
    return result.stdout


def record_for(cell, job_id, adapter):
    record = {key: cell[key] for key in
              ("domain", "dataset_domain", "arm", "seed", "run_stamp", "run_dir", "target_steps")}
    record.update(job_id=job_id, held_scheduler_record=adapter.audit_held(job_id, cell))
    return record


def execute(amendment_path, amendment_sha256, *, continue_held=False):
    amendment_path = Path(amendment_path).resolve()
    amendment = compatibility.authenticate_amendment(amendment_path, amendment_sha256)
    require(all(amendment["files_sha256"].get(str(p)) == original.digest(p) for p in (SOURCE, TEST)),
            "continuation implementation and tests must be pinned before execution")
    plan = original.verify_plan(expected_sha256=PLAN_SHA256, model_choice="05b")
    ledger = original.read(original.LEDGER)
    adapter = compatibility.LauncherAdapter(original)
    with (original.HERE / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first = reconcile_journals(plan, ledger)
        queue = reconcile_queue(first)
        first_record = record_for(plan["cells"][0], first, adapter)
        if not continue_held:
            return {"status": "single_successful_submission_reconciled", "existing_job_id": first,
                    "remaining_unsubmitted": 99, "scheduler_mutation": False}
        compatibility.authenticate_amendment(amendment_path, amendment_sha256)
        original.atomic_new(original.HERE / "slurm2511_continuation_intent.json", {
            "schema": "e122_slurm2511_continuation_intent_v1", "created_at": original.now(),
            "amendment_path": str(amendment_path), "amendment_sha256": amendment_sha256,
            "plan_sha256": PLAN_SHA256, "first_job_id": first, "existing_first_job_reused": True,
            "submit_indices": list(range(1, 100)), "queue_before": queue,
            "first_held_record": first_record, "source_sha256": original.digest(SOURCE)})
        original.atomic_new(original.HERE / "held_audit_000.json", first_record)
        records = [first_record]
        print(json.dumps({"event": "existing_first_job_reconciled", "job_id": first}), flush=True)
        for index, cell in enumerate(plan["cells"][1:], 1):
            job_id = original.submit_one_held(cell, index)
            require(job_id not in {row["job_id"] for row in records}, "duplicate scheduler job ID")
            record = record_for(cell, job_id, adapter)
            original.atomic_new(original.HERE / f"held_audit_{index:03d}.json", record)
            records.append(record)
            print(json.dumps({"event": "held_audited", "count": len(records), "job_id": job_id}), flush=True)
        original.verify_plan(expected_sha256=PLAN_SHA256, model_choice="05b")
        compatibility.authenticate_amendment(amendment_path, amendment_sha256)
        for record, cell in zip(records, plan["cells"]):
            adapter.audit_held(record["job_id"], cell)
        require(original.digest(original.LEDGER) == INITIAL_LEDGER_SHA256, "initial ledger changed")
        original.atomic_new(original.HERE / "prospective_ledger_before_submission.json", ledger)
        ledger.update(runs=records, status="held_audited", released=False,
            model_choice="05b", model_choice_pending=False, model=plan["model"],
            model_revision=plan["model_revision"], model_family="qwen05b",
            admission_proof=plan["admission_proof"], plan_path=str(original.PLAN), plan_sha256=PLAN_SHA256,
            snapshot_root=plan["snapshot_root"], held_audited_at=original.now(),
            slurm_display_amendment_path=str(amendment_path), slurm_display_amendment_sha256=amendment_sha256)
        original.e119.e78.atomic_json(original.LEDGER, ledger)
        original.atomic_new(original.HERE / "held_submission_complete.json", {
            "schema": "e122_held_submission_complete_v1", "created_at": original.now(),
            "plan_sha256": PLAN_SHA256, "ledger_sha256": original.digest(original.LEDGER),
            "amendment_sha256": amendment_sha256, "job_ids": [r["job_id"] for r in records],
            "released": False, "first_job_reused_without_resubmission": True})
    return {"held": 100, "released": 0, "ledger": str(original.LEDGER),
            "ledger_sha256": original.digest(original.LEDGER)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--continue-held", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(execute(args.amendment, args.amendment_sha256,
                                 continue_held=args.continue_held), sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"status": "stopped_for_reconciliation", "error": str(error),
                          "automatic_retry_permitted": False}), flush=True)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
