#!/usr/bin/env python3
"""Target-only post-save memory migration after documented checkpoint throttling.

Dry-run by default. Root may apply only after the original current-pressure
recovery declines because the save released memory. No scientific settings or
generic recovery source are changed by this process-local override.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "ops"), str(ROOT / "ops/exp_scaling")]
import recover_e119_memory_pressure_20260905 as recovery

TARGET, PEER, NODE, SAVE_STEP = 31048201, 31037832, "node205", 1152
BASE_SHA256 = "b254524f5b73b9c016ff272597ba1a1b361954baf428336300eeedc7d2f1ad15"
ENTRY_PATH = recovery.ART / "additional_final_candidates.json"
ENTRY_SHA256 = "eb87870d4167b9505c086101865bddc7768e7ed101d282dcfc3908218706d666"
EVIDENCE_PATH = recovery.ART / "final_six_cgroup_census.json"
EVIDENCE_SHA256 = "e5b0e9cb2309dc110a7938518c580fdeee557c034ea8255de0adc89bc666b73d"
ORIGINAL_TIMING = recovery.timing_and_checkpoint
APPLY_MODE = False
ENTRY = None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_saved_detail(detail: dict) -> None:
    assert detail["checkpoint_step"] == SAVE_STEP, "Wait for fully valid step1152."
    assert detail["checkpoint"] and not detail["fresh_restart"]
    counters = detail["saved_counter_validation"]["saved_counters"]
    assert all(counters.get(k) == SAVE_STEP for k in (
        "global_steps", "global_step", "prompt_batches_consumed_total"))
    assert detail["current_step"] is not None
    assert SAVE_STEP - 1 <= detail["current_step"] <= SAVE_STEP + 16
    assert 0 <= detail["unsaved_steps"] <= 16, "Re-review rollback above16 logged updates."


def bounded_timing(jid: int, run: dict) -> dict:
    assert jid == TARGET and run["run_stamp"] == "e119_level2_python_maxrl_s44"
    detail = ORIGINAL_TIMING(jid, run)
    check_saved_detail(detail)
    return detail


def same_attempt(record: str) -> None:
    assert ENTRY is not None
    assert recovery.field(record, "JobId") == str(TARGET)
    assert recovery.field(record, "UserId").startswith(getpass.getuser() + "(")
    assert recovery.field(record, "JobState") == "RUNNING"
    assert recovery.field(record, "MinMemoryNode") == "40G"
    for key in ("NodeList", "StartTime", "Restarts", "UserId"):
        assert recovery.field(record, key) == recovery.field(ENTRY["before"], key), key
    assert recovery.field(record, "NodeList") == NODE
    assert recovery.submitline(record) == ENTRY["original_submitline"]


def prior_pressure() -> dict:
    assert sha(EVIDENCE_PATH) == EVIDENCE_SHA256
    artifact = json.loads(EVIDENCE_PATH.read_text())
    row = next(row for row in artifact["jobs"] if row["job_id"] == TARGET)
    assert artifact["sample_interval_seconds"] == 5
    assert all(recovery.pressure(row["samples"][str(i)]) for i in (1, 2))
    assert row["samples"]["2"]["events.high"] > row["samples"]["1"]["events.high"]
    assert ENTRY["pressure_evidence"]["memory"] == row["samples"]["2"]
    observed = datetime.fromisoformat(artifact["asof"])
    assert 0 <= (datetime.now(timezone.utc) - observed).total_seconds() <= 5400
    return row["samples"]["2"]


def check_live_memory(memory: dict, old: dict) -> None:
    required = ("memory.current", "memory.high", "stat.anon", "stat.shmem",
                "events.high", "events.oom", "events.oom_kill")
    assert all(isinstance(memory.get(k), int) and memory[k] >= 0 for k in required)
    assert memory["memory.high"] == old["memory.high"] == 40 * 1024**3
    assert memory["events.high"] >= old["events.high"], "Cgroup counter reset or wrong attempt."
    for key in ("events.oom", "events.oom_kill"):
        assert memory[key] == old[key] == 0, "New OOM requires a fresh review."
    assert not recovery.pressure(memory), "Current pressure remains: use the original strict recovery."


def current_peer() -> str:
    # PEER is excluded from generic mutation scope because it was repaired
    # separately. This diagnostic identity check changes no exclusion.
    run = recovery.identities()[PEER]
    assert run["original_job_id"] == 31014463
    assert run["run_stamp"] == "e119_level2_pantry_replay_drgrpo_s44"
    record = recovery.show(PEER)
    exp = recovery.exports(record)
    assert recovery.field(record, "JobId") == str(PEER)
    assert recovery.field(record, "JobName").startswith("e119-")
    assert recovery.field(record, "UserId").startswith(getpass.getuser() + "(")
    assert recovery.field(record, "JobState") == "RUNNING"
    assert recovery.field(record, "NodeList") == NODE
    assert recovery.field(record, "MinMemoryNode") == "64G"
    assert recovery.field(record, "NumCPUs") == "8"
    assert exp["RUN_STAMP"] == run["run_stamp"]
    assert Path(exp["SAVE_PATH"]).resolve() == Path(run["run_dir"]).resolve()
    assert exp["OAT_ZERO_AUTO_RESUME"] == "1"
    return record


def post_save_basis(jid: int, before: str) -> dict:
    assert jid == TARGET
    same_attempt(before)
    old = prior_pressure()
    detail = bounded_timing(jid, ENTRY["run"])
    peer = current_peer()
    assert recovery.field(peer, "UserId").startswith(getpass.getuser() + "(")
    assert recovery.field(peer, "JobState") == "RUNNING"
    assert recovery.field(peer, "NodeList") == NODE
    assert recovery.field(peer, "MinMemoryNode") == "64G"
    script = f'''cg=/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{TARGET}
for fn in memory.current memory.high; do
  [[ -r "$cg/$fn" ]] || exit 4
  read -r value < "$cg/$fn"; printf '%s %s\\n' "$fn" "$value"
done
[[ -r "$cg/memory.stat" && -r "$cg/memory.events" ]] || exit 4
while read -r key value; do case "$key" in anon|shmem) printf 'stat.%s %s\\n' "$key" "$value";; esac; done < "$cg/memory.stat"
while read -r key value; do printf 'events.%s %s\\n' "$key" "$value"; done < "$cg/memory.events"
'''
    command = ["timeout", "-k", "3s", "20s", "srun", f"--jobid={PEER}",
               "--overlap", "--exact", "--nodes=1", "--ntasks=1",
               "--cpus-per-task=1", "--mem=0", "--gres=none",
               f"--nodelist={NODE}", "/bin/bash", "-c", script]
    result = subprocess.run(command, capture_output=True, text=True, timeout=28)
    memory = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            memory[parts[0]] = int(parts[1])
    report = {
        "checked_at_utc": recovery.now(), "target_job_id": TARGET,
        "basis": "Prevent recurrence of same-attempt checkpoint memory throttling after a validated save; current throttling is not asserted.",
        "current_pressure_confirmed": recovery.pressure(memory),
        "prior_two_sample_pressure_source": str(EVIDENCE_PATH),
        "prior_memory": old, "memory": memory, "checkpoint": detail,
        "probe_via_job": PEER, "probe_returncode": result.returncode,
        "probe_stdout": result.stdout, "probe_stderr": result.stderr,
        "target_before": before, "peer_before": peer,
        "script_sha256": sha(Path(__file__)), "base_script_sha256": BASE_SHA256,
    }
    dest = recovery.ART / ("python_s44_postsave_apply.json" if APPLY_MODE else "python_s44_postsave_dryrun.json")
    recovery.atomic(dest, report)
    assert result.returncode == 0, "Read-only live probe failed; no mutation."
    check_live_memory(memory, old)
    after = recovery.live_identity(TARGET, ENTRY["run"])
    same_attempt(after)
    peer_after = current_peer()
    for key in ("UserId", "JobState", "NodeList", "StartTime", "Restarts"):
        assert recovery.field(peer_after, key) == recovery.field(peer, key), key
    report.update({"target_after_probe": after, "peer_after_probe": peer_after,
                   "same_attempt_and_bounded_rollback_verified": True})
    recovery.atomic(dest, report)
    return {"checked_at_utc": report["checked_at_utc"], "probe_via_job": PEER,
            "memory": memory, "current_pressure_confirmed": False,
            "recovery_basis": report["basis"], "post_save_override_audit": str(dest)}


def main() -> None:
    global APPLY_MODE, ENTRY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    APPLY_MODE = args.apply
    assert sha(Path(recovery.__file__)) == BASE_SHA256
    assert sha(ENTRY_PATH) == ENTRY_SHA256
    entries = json.loads(ENTRY_PATH.read_text())["running_candidates"]
    ENTRY = next(e for e in entries if e["job_id"] == TARGET)
    assert ENTRY["run"]["run_stamp"] == "e119_level2_python_maxrl_s44"
    assert not (recovery.ART / str(TARGET) / "transaction.json").exists(), "Inspect an existing transaction before retrying."
    # These substitutions are restricted to this Python process. Reuse the
    # unchanged archive/requeuehold/held audit/resource update/release transaction.
    recovery.live_memory = post_save_basis
    recovery.timing_and_checkpoint = bounded_timing
    if not args.apply:
        before = recovery.live_identity(TARGET, ENTRY["run"])
        evidence = post_save_basis(TARGET, before)
        print(json.dumps({"dry_run": "pass", "job_id": TARGET,
                          "checkpoint_step": SAVE_STEP, "evidence": evidence}))
        return
    recovery.apply_one(ENTRY, True, argparse.Namespace(
        allow_fresh_restart=set(), allow_near_checkpoint=set()))


if __name__ == "__main__":
    main()
