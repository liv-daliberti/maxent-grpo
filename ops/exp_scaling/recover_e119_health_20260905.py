#!/usr/bin/env python3
"""Audited, same-cell recovery of three E119 failures observed 2026-09-05."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickletools
import re
import shlex
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import campaign_stats as campaign
from validate_deepspeed_checkpoint import select_latest_checkpoint

ART = ROOT / "var/artifacts/e119_health_recovery_20260905"
AUDIT = ART / "recovery.json"
RUNTIME = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_50d36295558a8958/ops/run_experiment.sh"
FAILED = 31048378
ORIGINAL = 31014402
OOM = {31037827: (31014458, "drgrpo", "node205"), 31048178: (31014460, "maxrl", "node207")}


def call(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def show(jid: int) -> str:
    return call("scontrol", "show", "job", "-dd", "-o", str(jid))


def field(record: str, name: str) -> str:
    match = re.search(r"(?:^| )" + re.escape(name) + r"=([^ ]*)", record)
    if not match:
        raise RuntimeError(f"Missing scheduler field {name}")
    return match.group(1)


def checkpoint(run: dict) -> dict:
    selected, rejected = select_latest_checkpoint(Path(run["run_dir"]))
    assert selected is not None, run["run_dir"]
    model = selected / "mp_rank_00_model_states.pt"
    with zipfile.ZipFile(model) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith("data.pkl")))
    tokens = list(pickletools.genops(metadata))
    counters = {}
    for index, (_, arg, _) in enumerate(tokens):
        if arg in ("global_steps", "global_step", "prompt_batches_consumed_total"):
            values = [value for op, value, _ in tokens[index + 1:index + 4] if op.name.startswith("BININT")]
            assert values
            counters[arg] = values[0]
    step = int(selected.name.split("_")[1])
    assert counters == dict.fromkeys(("global_steps", "global_step", "prompt_batches_consumed_total"), step), counters
    return {"path": str(selected), "step": step, "saved_counters": counters, "rejected": rejected,
            "files": [{"name": p.name, "bytes": p.stat().st_size} for p in selected.glob("*.pt")]}


def prepare() -> tuple[dict, bytes, bytes, list[str]]:
    ledger = json.loads(campaign.E119_LEDGER.read_text())
    mapping = campaign.e119_continuation_jobs(campaign.E119_LEDGER)
    assert mapping[ORIGINAL] == FAILED
    runs = {int(r["job_id"]): r for r in ledger["runs"]}
    assert (runs[ORIGINAL]["domain"], runs[ORIGINAL]["arm"], runs[ORIGINAL]["seed"]) == ("countdown", "drgrpo", 44)
    assert call("sacct", "-n", "-X", "-P", "-j", str(FAILED), "--format=State").split("|")[0] == "FAILED"
    live = call("squeue", "-h", "-u", "od2961", "-o", "%i|%j")
    assert "e119-countd-d-s44" not in live
    failures = []
    for jid, (original, arm, node) in OOM.items():
        assert mapping[original] == jid
        run = runs[original]
        assert (run["domain"], run["arm"], run["seed"]) == ("pantry_plan", arm, 43)
        shown = show(jid)
        assert field(shown, "JobState") == "RUNNING"
        assert field(shown, "NodeList") == "node204"
        assert field(shown, "Partition") == "cs" and field(shown, "Account") == "allcs"
        assert field(shown, "MinMemoryNode") == "40G"
        out = Path(field(shown, "StdOut"))
        with out.open("rb") as handle:
            handle.seek(max(0, out.stat().st_size - 150000))
            tail = handle.read().decode(errors="replace")
        assert "torch.OutOfMemoryError: CUDA out of memory" in tail
        node_record = call("scontrol", "show", "node", "-o", node)
        assert "gpu:a6000:" in field(node_record, "Gres")
        failures.append({"job_id": jid, "original_job_id": original, "run_stamp": run["run_stamp"],
                         "target_node": node, "before": shown, "checkpoint": checkpoint(run)})
    raw = RUNTIME.read_bytes()
    text = raw.decode()
    begin, end = "# BEGIN E119_RUNTIME_DURABILITY_20260904", "# END E119_RUNTIME_DURABILITY_20260904"
    block = text[text.index(begin):text.index(end)]
    anchor = "|AttributeError:.*_infer_resume_step\""
    assert block.count(anchor) == 1 and "torch.OutOfMemoryError" not in block
    changed = text.replace(anchor, "|AttributeError:.*_infer_resume_step|torch.OutOfMemoryError: CUDA out of memory\"", 1).encode()
    subprocess.run(["bash", "-n"], input=changed, check=True)
    new_block = changed.decode().split(begin, 1)[1].split(end, 1)[0]
    guards = []
    for stamp, expected in [("e119_level2_pantry_drgrpo_s43", True), ("e119_level2_countdown_drgrpo_s44", True), ("e118q3_pantry_maxrl_s73", False)]:
        env = {"PATH": os.environ["PATH"], "RUN_STAMP": stamp, "ROOT_DIR": str(ROOT), "SLURM_JOB_ID": "123", "SLURM_JOB_NAME": "audit"}
        observed = subprocess.check_output(["bash", "-eu", "-c", new_block + '\nprintf "%s" "${OAT_ZERO_WATCHDOG_FATAL_PATTERN:-}"'], env=env, text=True)
        assert ("torch.OutOfMemoryError" in observed) == expected
        guards.append({"run_stamp": stamp, "oom_detection": expected})
    original_submit = call("sacct", "-n", "-X", "-P", "-j", str(FAILED), "--format=SubmitLine%20000").rstrip("|")
    command = shlex.split(original_submit)
    assert command[0] == "sbatch" and "--hold" in command and "--mem=64G" in command
    export_index = next(i for i, part in enumerate(command) if part.startswith("--export="))
    env_string = command[export_index]
    for name in ("OAT_ZERO_WATCHDOG_STALE_SECONDS", "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS", "OAT_ZERO_WATCHDOG_MAX_RESTARTS"):
        assert name + "=" not in env_string
    command[export_index] += ",OAT_ZERO_WATCHDOG_STALE_SECONDS=7200,OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS=3600,OAT_ZERO_WATCHDOG_MAX_RESTARTS=12"
    record = {"schema": "e119-health-recovery-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "same_scientific_cells": True, "same_run_directories": True, "treatment_changed": False,
              "optimizer_changed": False, "evaluation_changed": False, "selection_basis": "scheduler failure and learner CUDA OOM only",
              "runtime": {"path": str(RUNTIME), "before_sha256": sha(raw), "after_sha256": sha(changed), "guard_checks": guards},
              "pantry": failures, "countdown": {"failed_job_id": FAILED, "original_job_id": ORIGINAL,
              "checkpoint": checkpoint(runs[ORIGINAL]), "original_submitline": original_submit, "new_command": command},
              "applied": False}
    return record, raw, changed, command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if AUDIT.exists():
        raise SystemExit(f"Existing recovery requires explicit inspection: {AUDIT}")
    record, raw, changed, command = prepare()
    ART.mkdir(parents=True, exist_ok=True)
    atomic(ART / "plan.json", record)
    if not args.apply:
        print(json.dumps({"plan": str(ART / "plan.json"), "affected": list(OOM) + [FAILED], "checks_passed": True}))
        return
    assert RUNTIME.read_bytes() == raw
    (ART / "run_experiment.sh.before").write_bytes(raw)
    for plan in record["pantry"]:
        for kind in ("StdOut", "StdErr"):
            path = Path(field(plan["before"], kind))
            shutil.copy2(path, ART / (path.name + ".before-requeue"))
    RUNTIME.write_bytes(changed)
    record["runtime"]["applied"] = True
    atomic(AUDIT, record)
    # Requeue only learners already proven dead; hold them before node mutation.
    for plan in record["pantry"]:
        jid = plan["job_id"]
        call("scontrol", "requeuehold", str(jid))
        plan["held_after_requeue"] = show(jid)
        call("scontrol", "update", f"JobId={jid}", f"ReqNodeList={plan['target_node']}")
        held = show(jid)
        assert field(held, "ReqNodeList") == plan["target_node"]
        assert held.split(" SubmitLine=", 1)[1].split(" WorkDir=", 1)[0] == plan["before"].split(" SubmitLine=", 1)[1].split(" WorkDir=", 1)[0]
        for key in ("MinMemoryNode", "NumCPUs", "Partition", "Account", "ExcNodeList", "TimeLimit"):
            assert field(held, key) == field(plan["before"], key), key
        assert field(held, "JobState") in ("PENDING", "COMPLETING")
        plan["held_after_placement"] = held
        atomic(AUDIT, record)
        call("scontrol", "release", str(jid))
        plan["released"] = True
        plan["after"] = show(jid)
        atomic(AUDIT, record)
    # Original Countdown allocation has left the controller; use the established
    # held/audited/recorded continuation workflow with its exact prior exports.
    new_id = int(call(*command).split(";", 1)[0])
    record["countdown"]["new_job_id"] = new_id
    atomic(AUDIT, record)
    held = show(new_id)
    for token in ("JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=node105", "MinMemoryNode=64G", "OAT_ZERO_WATCHDOG_STALE_SECONDS=7200", "OAT_ZERO_WATCHDOG_MAX_RESTARTS=12", "RUN_STAMP=e119_level2_countdown_drgrpo_s44", "OAT_ZERO_AUTO_RESUME=1"):
        assert token in held, token
    record["countdown"]["held"] = held
    continuation = json.loads(campaign.E119_CONTINUATIONS.read_text())
    (ART / "continuation-ledger.before.json").write_text(json.dumps(continuation, indent=2) + "\n")
    target = next(r for r in continuation["continuations"] if int(r["original_job_id"]) == ORIGINAL)
    assert int(target["continuation_job_id"]) == FAILED
    target["previous_continuation_job_ids"] = [*target.get("previous_continuation_job_ids", []), FAILED]
    target["continuation_job_id"] = new_id
    target["repair_kind"] = "20260905_evaluation_safe_watchdog_continuation"
    continuation["operational_change"] = str(continuation.get("operational_change", "")) + "; 2026-09-05 Countdown Dr.GRPO s44 same-cell continuation with established two-hour watchdog"
    atomic(campaign.E119_CONTINUATIONS, continuation)
    assert campaign.e119_continuation_jobs(campaign.E119_LEDGER)[ORIGINAL] == new_id
    atomic(AUDIT, record)
    call("scontrol", "release", str(new_id))
    record["countdown"]["released"] = True
    record["countdown"]["after"] = show(new_id)
    record["applied"] = True
    atomic(AUDIT, record)
    print(json.dumps({"audit": str(AUDIT), "countdown_replacement": new_id, "pantry_requeued": list(OOM)}))


if __name__ == "__main__":
    main()
