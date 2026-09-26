#!/usr/bin/env python3
"""One authorized same-ID repair: Pantry replay Dr.GRPO s44 memory throttling."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "ops"), str(ROOT / "ops/exp_scaling")]
import campaign_stats as campaign
from recover_e119_health_20260905 import atomic, call, field, show
from validate_deepspeed_checkpoint import select_latest_checkpoint

JID = 31037832
STAMP = "e119_level2_pantry_replay_drgrpo_s44"
ART = ROOT / "var/artifacts/campaign_health_capacity_20260905/e119_pantry_s44_memory_recovery"


def submitline(record: str) -> str:
    return record.split(" SubmitLine=", 1)[1].split(" WorkDir=", 1)[0]


def assert_held(record: str) -> None:
    assert field(record, "JobState") == "PENDING"
    assert field(record, "Priority") == "0"
    assert field(record, "Reason") in ("JobHeldUser", "JobHeldAdmin", "job_requeued_in_held_state")


def live_identity() -> tuple[str, dict, int]:
    mapping = campaign.e119_continuation_jobs(campaign.E119_LEDGER)
    originals = [original for original, effective in mapping.items() if effective == JID]
    assert len(originals) == 1, originals
    original = originals[0]
    ledger = json.loads(campaign.E119_LEDGER.read_text())
    run = next(row for row in ledger["runs"] if int(row["job_id"]) == original)
    assert run["run_stamp"] == STAMP
    record = show(JID)
    assert field(record, "JobName") == "e119-pantry-rd-s44"
    assert field(record, "JobState") == "RUNNING"
    assert field(record, "NodeList") == "node203"
    assert field(record, "MinMemoryNode") == "40G"
    assert field(record, "Partition") == "cs" and field(record, "Account") == "allcs"
    assert field(record, "NumCPUs") == "8"
    assert field(record, "TresPerNode") == "gres/gpu:1"
    assert field(record, "TimeLimit") == "1-12:00:00"
    live = call("squeue", "-h", "-u", "od2961", "-o", "%i|%j")
    assert [line for line in live.splitlines() if line.endswith("|e119-pantry-rd-s44")] == [f"{JID}|e119-pantry-rd-s44"]
    checkpoint, rejected = select_latest_checkpoint(Path(run["run_dir"]))
    assert checkpoint is None, f"New durable checkpoint requires a revised preservation plan: {checkpoint}"
    assert not rejected, rejected
    for node in ("node205", "node207"):
        assert "gpu:a6000:" in field(call("scontrol", "show", "node", "-o", node), "Gres")
    return record, run, original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume-held", action="store_true")
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    if args.resume_held:
        assert args.apply
        amendment = json.loads((ART / "recovery.json").read_text())
        assert amendment["job_id"] == JID and amendment["run_stamp"] == STAMP
        assert amendment["requeuehold_called"] and not amendment["applied"]
        held = show(JID)
        assert_held(held)
        assert field(held, "MinMemoryNode") == "40G"
        assert submitline(held) == amendment["original_submitline"]
        amendment["resumed_after_guard_pause"] = "Strict guard now recognizes the observed job_requeued_in_held_state reason, pending state and zero priority."
        finish_held(amendment, amendment["before"], amendment["original_job_id"], held)
        return
    assert not (ART / "recovery.json").exists(), "Inspect existing execution receipt before any retry"
    before, run, original = live_identity()
    metrics = Path(run["run_dir"]) / f"debug_job{JID}" / "train_metrics.jsonl"
    rows = []
    for line in metrics.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    assert rows
    amendment = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": JID, "original_job_id": original, "run_stamp": STAMP,
        "authorization": "User authorized fixing unhealthy E119; parent explicitly approved this one-cell memory recovery after disclosing loss of 74 unsaved updates.",
        "selection_basis": "Measured memory.high reclaim throttling, with current42.37GiB exceeding40GiB soft limit and11million high events; no numerical-outcome selection.",
        "before": before, "original_submitline": submitline(before),
        "run_dir": run["run_dir"], "checkpoint": None,
        "current_step": rows[-1].get("trainer/step"),
        "historical_highwater": max(row.get("trainer/step", 0) for row in rows),
        "resource_override": {"MinMemoryNode": "64G", "ReqNodeList": "node205,node207"},
        "science_exports_unchanged": True,
        "restart_from_initialization": True,
        "progress_tradeoff": "No durable checkpoint exists; current unsaved optimizer updates cannot survive restart. Existing optimizer/model/data/seed/replay/evaluation settings remain unchanged.",
        "applied": False,
    }
    atomic(ART / "operational_amendment.json", amendment)
    if not args.apply:
        print(json.dumps({"dry_run": "pass", "job_id": JID, "current_step": amendment["current_step"], "amendment": str(ART / "operational_amendment.json")}))
        return

    # Preserve diagnostics before Slurm overwrites current stdout on requeue.
    archives = []
    for source in [Path(field(before, "StdOut")), Path(field(before, "StdErr")), metrics]:
        destination = ART / (source.name + ".before-requeue")
        shutil.copy2(source, destination)
        archives.append({"source": str(source), "archive": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "bytes": destination.stat().st_size})
    amendment["archives"] = archives
    atomic(ART / "recovery.json", amendment)
    fresh_before, fresh_run, fresh_original = live_identity()
    assert fresh_original == original and fresh_run["run_dir"] == run["run_dir"]
    assert submitline(fresh_before) == submitline(before)
    call("scontrol", "requeuehold", str(JID))
    amendment["requeuehold_called"] = True
    atomic(ART / "recovery.json", amendment)
    deadline = time.monotonic() + 120
    while True:
        held = show(JID)
        if field(held, "JobState") == "PENDING":
            break
        assert field(held, "JobState") in ("RUNNING", "COMPLETING")
        assert time.monotonic() < deadline, "Timed out waiting for requeue; leave held for inspection"
        time.sleep(2)
    assert_held(held)
    finish_held(amendment, before, original, held)


def finish_held(amendment: dict, before: str, original: int, held: str) -> None:
    assert_held(held)
    amendment["held_before_update"] = held
    atomic(ART / "recovery.json", amendment)
    call("scontrol", "update", f"JobId={JID}", "MinMemoryNode=65536", "ReqNodeList=node205,node207")
    held = show(JID)
    assert_held(held)
    assert field(held, "MinMemoryNode") == "64G"
    assert field(held, "ReqNodeList") in ("node[205,207]", "node205,node207")
    assert submitline(held) == submitline(before), "Submitted science exports changed"
    for key in ("NumCPUs", "NumTasks", "CPUs/Task", "TresPerNode", "Partition", "Account", "ExcNodeList", "TimeLimit", "Command", "WorkDir", "StdOut", "StdErr", "JobName"):
        assert field(held, key) == field(before, key), key
    assert campaign.e119_continuation_jobs(campaign.E119_LEDGER)[original] == JID
    amendment["held_audit"] = held
    amendment["held_audit_passed"] = True
    atomic(ART / "recovery.json", amendment)
    call("scontrol", "release", str(JID))
    after = show(JID)
    assert field(after, "JobState") in ("PENDING", "RUNNING")
    assert field(after, "MinMemoryNode") == "64G"
    assert field(after, "ReqNodeList") in ("node[205,207]", "node205,node207")
    if field(after, "JobState") == "RUNNING":
        assert field(after, "NodeList") in ("node205", "node207")
    amendment["released"] = True
    amendment["applied"] = True
    amendment["after"] = after
    amendment["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic(ART / "recovery.json", amendment)
    print(json.dumps({"applied": True, "job_id": JID, "state": field(after, "JobState"), "reason": field(after, "Reason"), "memory": field(after, "MinMemoryNode"), "nodes": field(after, "ReqNodeList"), "audit": str(ART / "recovery.json")}))


if __name__ == "__main__":
    main()
