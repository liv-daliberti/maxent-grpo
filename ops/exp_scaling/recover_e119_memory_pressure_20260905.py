#!/usr/bin/env python3
"""Plan E119 memory repairs; apply only explicitly selected reviewed actions.

Dry run reads scheduler, timing, checkpoint and prior independent cgroup evidence.
Apply rechecks live cgroups for each selected running job. Other cohorts and the
separately repaired job31037832 are excluded. No manuscript/runtime/ledger edits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "ops"), str(ROOT / "ops/exp_scaling")]
import campaign_stats as campaign
from recover_e119_health_20260905 import atomic, call, field, show, checkpoint as checked_checkpoint
from validate_deepspeed_checkpoint import select_latest_checkpoint

ART = ROOT / "var/artifacts/campaign_health_capacity_20260905/e119_memory_pressure_recovery"
PLAN = ART / "plan.json"
EXCLUDED = {31037832}
PRESERVE = ("JobName", "Account", "Partition", "NumCPUs", "NumTasks", "CPUs/Task",
            "TresPerNode", "TimeLimit", "ExcNodeList", "Command", "WorkDir",
            "StdOut", "StdErr", "Requeue", "Dependency")
OWN_HOLD_REASONS = {"JobHeldUser", "job_requeued_in_held_state"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ids(value: str) -> set[int]:
    return {int(part) for part in value.split(",") if part.strip()}


def submitline(record: str) -> str:
    return record.split(" SubmitLine=", 1)[1].split(" WorkDir=", 1)[0]


def exports(record: str) -> dict[str, str]:
    tokens = shlex.split(submitline(record))
    token = next(token for token in tokens if token.startswith("--export="))
    return dict(item.split("=", 1) for item in token[len("--export="):].split(",") if "=" in item)


def assert_own_hold(record: str) -> None:
    assert field(record, "JobState") == "PENDING"
    assert field(record, "Priority") == "0"
    assert field(record, "Reason") in OWN_HOLD_REASONS, "Never release an administrator hold"


def ordinary_pending(record: str) -> bool:
    return (field(record, "JobState") == "PENDING"
            and int(field(record, "Priority")) > 0
            and field(record, "Dependency") in {"(null)", ""}
            and "held" not in field(record, "Reason").lower()
            and "hold" not in field(record, "Reason").lower())


def pressure(memory: dict) -> bool:
    required = ("memory.current", "memory.high", "stat.anon", "stat.shmem", "events.high")
    if not all(isinstance(memory.get(k), int) for k in required):
        return False
    return (memory["memory.current"] > memory["memory.high"]
            and memory["stat.anon"] + memory["stat.shmem"] > memory["memory.high"]
            and memory["events.high"] > 0)


def identities() -> dict[int, dict]:
    mapping = campaign.e119_continuation_jobs(campaign.E119_LEDGER)
    runs = json.loads(campaign.E119_LEDGER.read_text())["runs"]
    result = {}
    for run in runs:
        original = int(run["job_id"])
        jid = mapping.get(original, original)
        assert jid not in result, "Duplicate effective cell mapping"
        result[jid] = {**run, "original_job_id": original, "effective_job_id": jid}
    assert len(result) == 100
    return result


def live_identity(jid: int, run: dict) -> str:
    assert jid not in EXCLUDED
    current = identities()[jid]
    assert current["original_job_id"] == run["original_job_id"]
    assert current["run_dir"] == run["run_dir"] and current["run_stamp"] == run["run_stamp"]
    record = show(jid)
    exp = exports(record)
    assert field(record, "JobName").startswith("e119-")
    assert exp["RUN_STAMP"] == run["run_stamp"]
    assert Path(exp["SAVE_PATH"]).resolve() == Path(run["run_dir"]).resolve()
    assert exp.get("OAT_ZERO_AUTO_RESUME") == "1"
    assert field(record, "NumCPUs") == "8"
    return record


def timing_and_checkpoint(jid: int, run: dict) -> dict:
    root = Path(run["run_dir"])
    metrics = root / f"debug_job{jid}" / "train_metrics.jsonl"
    rows = []
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrently appended final line may be incomplete.
    begin = 0
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if b.get("misc/elapse", 0) < a.get("misc/elapse", 0) or b.get("trainer/step", 0) < a.get("trainer/step", 0):
            begin = i
    segment = rows[begin:]
    tail = [{k: row.get(k) for k in ("trainer/step", "misc/global_step", "misc/elapse", "misc/weight_sync_elapse", "train/learn_batch_time")} for row in segment[-10:]]
    sync = [r["misc/weight_sync_elapse"] for r in tail if isinstance(r.get("misc/weight_sync_elapse"), (int, float))]
    checkpoint, rejected = select_latest_checkpoint(root)
    counter_validation = checked_checkpoint(run) if checkpoint else None
    if counter_validation:
        checkpoint = Path(counter_validation["path"])
        rejected = counter_validation["rejected"]
    step = segment[-1].get("trainer/step", segment[-1].get("misc/global_step")) if segment else None
    saved = int(checkpoint.name.split("_")[1]) if checkpoint else None
    return {"metric_file": str(metrics), "metric_age_seconds": time.time() - metrics.stat().st_mtime if metrics.exists() else None,
            "current_step": step, "last_10_metrics": tail,
            "median_weight_sync_seconds": statistics.median(sync) if sync else None,
            "checkpoint": str(checkpoint) if checkpoint else None, "checkpoint_step": saved,
            "rejected_checkpoints": rejected, "saved_counter_validation": counter_validation, "fresh_restart": checkpoint is None,
            "unsaved_steps": max(0, int(step) - (saved or 0)) if step is not None else None,
            "checkpoint_validation": "Same-cell model+optimizer ZIP directories and model metadata global_steps/global_step/prompt_batches_consumed_total agree with step tag; no tensor deserialization"}


def near_checkpoint(detail: dict, cadence: int) -> bool:
    step, saved = detail.get("current_step"), detail.get("checkpoint_step")
    if step is None or cadence <= 0 or (saved is not None and saved >= step):
        return False
    return int(step) > 0 and (-int(step)) % cadence <= 16


def checkpoint_cadence(record: str, run: dict) -> int:
    # The scoped E119 Pantry wrapper overrides submitted192 to rolling96.
    cadence = 96 if run["domain"] == "pantry_plan" else int(exports(record).get("OAT_ZERO_RESUME_STEPS", "192"))
    log = Path(field(record, "StdOut"))
    if log.is_file():
        text = log.read_text(errors="replace")
        values = re.findall(r"\[experiment\] storage_policy=[^\n]*?resume_steps=(\d+)", text)
        if values:
            cadence = min(cadence, int(values[-1]))
    return cadence


def proposed_nodes(record: str, run: dict) -> str:
    before = field(record, "ReqNodeList")
    if run["domain"] != "pantry_plan":
        return before
    if field(record, "Partition") == "cs" and field(record, "Account") == "allcs":
        allowed = set(call("scontrol", "show", "hostnames", before).splitlines())
        safe = [node for node in ("node205", "node207") if node in allowed]
        assert safe, "No already permitted Pantry A6000 route; needs separate placement review"
        return ",".join(safe)
    # Retain an already safe A100/A6000-only route without changing account/partition.
    assert before not in {"(null)", ""}, "Pantry route needs explicit review"
    nodes = call("scontrol", "show", "hostnames", before).splitlines()
    for node in nodes:
        assert any(kind in field(call("scontrol", "show", "node", "-o", node), "Gres") for kind in ("gpu:a6000:", "gpu:a100:"))
    return before


def audit_after(before: str, after: str, target_nodes: str) -> None:
    assert field(after, "MinMemoryNode") == "64G"
    assert submitline(after) == submitline(before), "Full submitted exports/arguments changed"
    for key in PRESERVE:
        assert field(after, key) == field(before, key), key
    assert field(before, "NumNodes") in {"1", "1-1"}
    assert field(after, "NumNodes") in {"1", "1-1"}
    actual = field(after, "ReqNodeList")
    if target_nodes == "node205,node207":
        assert actual in {"node205,node207", "node[205,207]"}
        if field(after, "JobState") == "RUNNING":
            assert field(after, "NodeList") in {"node205", "node207"}
    else:
        assert actual == target_nodes


def prior_evidence() -> dict[int, dict]:
    result = {}
    p = ART.parent / "e119_node203_memory_throughput.json"
    for row in json.loads(p.read_text())["jobs"]:
        complete = [m for m in row.get("cgroup_samples", []) if pressure(m)]
        if complete:
            result[int(row["job_id"])] = {"source": str(p), "memory": complete[-1]}
    p = ART.parent / "e119_other_nodes_memory_census.json"
    for row in json.loads(p.read_text())["jobs"]:
        raw = row.get("cgroup_second", {})
        memory = {}
        for key in ("memory.current", "memory.high"):
            if str(raw.get(key, "")).strip().isdigit():
                memory[key] = int(raw[key])
        for source, prefix in (("memory.stat", "stat."), ("memory.events", "events.")):
            for line in raw.get(source, "").splitlines():
                key, value = line.split()
                memory[prefix + key] = int(value)
        if pressure(memory):
            result[int(row["job_id"])] = {"source": str(p), "memory": memory}
    return result


def live_memory(jid: int, before: str) -> dict:
    node = field(before, "NodeList")
    peers = call("squeue", "-h", "-u", str(__import__("getpass").getuser()), "-w", node, "-t", "RUNNING", "-o", "%i|%j|%S").splitlines()
    eligible = [line.split("|") for line in peers if "|e119-" in line and int(line.split("|")[0]) not in EXCLUDED]
    probe_jid = int(max(eligible, key=lambda a: a[2])[0])
    script = f"""cg=/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{jid}
for fn in memory.current memory.high; do
  [[ -r "$cg/$fn" ]] || exit 4
  read -r value < "$cg/$fn"; printf '%s %s\\n' "$fn" "$value"
done
while read -r key value; do case "$key" in anon|shmem) printf 'stat.%s %s\\n' "$key" "$value";; esac; done < "$cg/memory.stat"
while read -r key value; do printf 'events.%s %s\\n' "$key" "$value"; done < "$cg/memory.events"
"""
    command = ["timeout", "-k", "3s", "20s", "srun", f"--jobid={probe_jid}", "--overlap", "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=0", "--gres=none", "/bin/bash", "-c", script]
    answer = subprocess.run(command, text=True, capture_output=True, timeout=28)
    memory = {}
    for line in answer.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            memory[parts[0]] = int(parts[1])
    assert answer.returncode == 0 and pressure(memory), f"Live non-cache pressure not confirmed: {answer.returncode}, {memory}, {answer.stderr[-500:]}"
    return {"checked_at_utc": now(), "probe_via_job": probe_jid, "memory": memory}


def prepare() -> dict:
    mapping, evidence = identities(), prior_evidence()
    live_ids = {int(line) for line in call("squeue", "-h", "-u", str(__import__("getpass").getuser()), "-o", "%i").splitlines() if line.isdigit()}
    plan = {"created_at_utc": now(), "read_only_plan": True, "excluded_ids": sorted(EXCLUDED), "pending": [], "running_candidates": [], "skipped": []}
    for jid, run in mapping.items():
        if jid in EXCLUDED or jid not in live_ids:
            continue
        record = live_identity(jid, run)
        if field(record, "MinMemoryNode") != "40G":
            continue
        entry = {"job_id": jid, "run": run, "before": record, "original_submitline": submitline(record)}
        if ordinary_pending(record):
            entry["target_nodes"] = proposed_nodes(record, run)
            plan["pending"].append(entry)
        elif field(record, "JobState") == "RUNNING" and jid in evidence:
            entry.update(timing_and_checkpoint(jid, run))
            entry["target_nodes"] = proposed_nodes(record, run)
            entry["pressure_evidence"] = evidence[jid]
            cadence = checkpoint_cadence(record, run)
            entry["effective_checkpoint_cadence"] = cadence
            entry["near_next_checkpoint"] = near_checkpoint(entry, cadence)
            plan["running_candidates"].append(entry)
        else:
            plan["skipped"].append({"job_id": jid, "state": field(record, "JobState"), "reason": "held/dependent or running without confirmed working-set pressure"})
    plan["pending_job_ids"] = [r["job_id"] for r in plan["pending"]]
    plan["pressure_candidate_ids"] = [r["job_id"] for r in plan["running_candidates"]]
    plan["fresh_restart_candidate_ids"] = [r["job_id"] for r in plan["running_candidates"] if r["fresh_restart"]]
    plan["near_checkpoint_candidate_ids"] = [r["job_id"] for r in plan["running_candidates"] if r["near_next_checkpoint"]]
    return plan


def archive(jid: int, record: str, detail: dict, directory: Path) -> list[dict]:
    result = []
    for label, name in (("stdout", field(record, "StdOut")), ("stderr", field(record, "StdErr")), ("metrics", detail["metric_file"])):
        path = Path(name)
        if not name or not path.is_file():
            continue
        target = directory / f"{jid}.{label}.before-requeue"
        shutil.copy2(path, target)
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result.append({"source": str(path), "archive": str(target), "bytes": target.stat().st_size, "sha256": digest.hexdigest()})
    return result


def apply_one(entry: dict, running: bool, args: argparse.Namespace) -> None:
    jid, run = entry["job_id"], entry["run"]
    directory = ART / str(jid)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "transaction.json"
    assert not receipt.exists(), f"Inspect prior receipt before retry: {receipt}"
    before = live_identity(jid, run)
    assert submitline(before) == entry["original_submitline"]
    assert field(before, "MinMemoryNode") == "40G"
    target = proposed_nodes(before, run)
    assert target == entry["target_nodes"]
    action = {"job_id": jid, "original_job_id": run["original_job_id"], "created_at_utc": now(), "before": before,
              "running_recovery": running, "applied": False, "target_memory": "64G", "target_nodes": target}
    if running:
        if field(before, "JobState") != "RUNNING":
            action["skipped"] = "Candidate no longer running; do not reinterpret approval as another action"
            atomic(receipt, action)
            return
        action["live_pressure"] = live_memory(jid, before)
        detail = timing_and_checkpoint(jid, run)
        action["progress_before_requeue"] = detail
        assert detail["checkpoint"] or jid in args.allow_fresh_restart, "Explicit fresh-restart allowance required: no valid checkpoint"
        cadence = checkpoint_cadence(before, run)
        if near_checkpoint(detail, cadence) and jid not in args.allow_near_checkpoint:
            action["skipped"] = "Within16steps of next save; await durable checkpoint or explicitly review rollback"
            atomic(receipt, action)
            return
        action["archives"] = archive(jid, before, detail, directory)
        latest = live_identity(jid, run)
        assert field(latest, "JobState") == "RUNNING"
        assert field(latest, "StartTime") == field(before, "StartTime")
        assert field(latest, "Restarts") == field(before, "Restarts")
        assert submitline(latest) == submitline(before)
        # Discover a checkpoint completed during archival before stopping the writer.
        action["progress_immediately_before_requeue"] = timing_and_checkpoint(jid, run)
        freshest = action["progress_immediately_before_requeue"]
        assert freshest["checkpoint"] or jid in args.allow_fresh_restart
        if near_checkpoint(freshest, cadence) and jid not in args.allow_near_checkpoint:
            action["skipped"] = "Reached checkpoint preservation window during archival; await durable save"
            atomic(receipt, action)
            return
        atomic(receipt, action)
        call("scontrol", "requeuehold", str(jid))
        action["hold_created"] = "requeuehold"
    else:
        if not ordinary_pending(before):
            action["skipped"] = "Pending job started, became held, or acquired a dependency"
            atomic(receipt, action)
            return
        atomic(receipt, action)
        call("scontrol", "hold", str(jid))
        action["hold_created"] = "hold"
    atomic(receipt, action)
    deadline = time.monotonic() + 120
    while True:
        held = show(jid)
        if field(held, "JobState") == "PENDING":
            break
        if not running:
            assert field(held, "JobState") == "RUNNING" and field(held, "Priority") == "0"
            call("scontrol", "release", str(jid))
            action["skipped"] = "Allocation raced pending hold; restored our hold without changing resources"
            atomic(receipt, action)
            return
        assert field(held, "JobState") in {"RUNNING", "COMPLETING"}
        assert time.monotonic() < deadline, "Leave held for inspection if Slurm cleanup does not finish"
        time.sleep(2)
    assert_own_hold(held)
    assert submitline(held) == submitline(before)
    action["held_before_update"] = held
    if running:
        action["resume_selection_after_writer_stopped"] = timing_and_checkpoint(jid, run)
        assert action["resume_selection_after_writer_stopped"]["checkpoint"] or jid in args.allow_fresh_restart
    atomic(receipt, action)
    update = ["scontrol", "update", f"JobId={jid}", "MinMemoryNode=65536"]
    if target != field(before, "ReqNodeList"):
        update.append(f"ReqNodeList={target}")
    call(*update)
    held = show(jid)
    assert_own_hold(held)
    audit_after(before, held, target)
    live_identity(jid, run)
    action["held_audit_passed"] = True
    action["held_after_update"] = held
    atomic(receipt, action)
    call("scontrol", "release", str(jid))
    after = show(jid)
    assert field(after, "JobState") in {"PENDING", "RUNNING"}
    audit_after(before, after, target)
    action.update({"released": True, "applied": True, "after": after, "completed_at_utc": now()})
    atomic(receipt, action)
    print(json.dumps({"job_id": jid, "applied": True, "running_recovery": running, "state": field(after, "JobState")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resize-pending", action="store_true")
    parser.add_argument("--recover-running", type=ids, default=set())
    parser.add_argument("--allow-fresh-restart", type=ids, default=set())
    parser.add_argument("--allow-near-checkpoint", type=ids, default=set())
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    if not args.apply:
        assert not args.resize_pending and not args.recover_running
        plan = prepare()
        atomic(PLAN, plan)
        print(json.dumps({k: plan[k] for k in ("pending_job_ids", "pressure_candidate_ids", "fresh_restart_candidate_ids", "near_checkpoint_candidate_ids")}))
        return
    plan = json.loads(PLAN.read_text())
    assert args.resize_pending or args.recover_running
    assert args.recover_running <= set(plan["pressure_candidate_ids"])
    assert args.allow_fresh_restart <= args.recover_running
    assert args.allow_near_checkpoint <= args.recover_running
    assert not (args.recover_running & EXCLUDED)
    if args.resize_pending:
        for entry in plan["pending"]:
            apply_one(entry, False, args)
    for entry in plan["running_candidates"]:
        if entry["job_id"] in args.recover_running:
            apply_one(entry, True, args)


if __name__ == "__main__":
    main()
