#!/usr/bin/env python3
"""Use one validated fast peer to audit/recover only E119 Python Dr.GRPO s43.

This process imports the reviewed generic recovery and substitutes only its
read-only live memory probe. It never edits that script or any learner runtime.
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

TARGET = 31048192
PEER = 31048201
NODE = "node205"
EXPECTED_BASE_SHA256 = "b254524f5b73b9c016ff272597ba1a1b361954baf428336300eeedc7d2f1ad15"
ENTRY_PATH = recovery.ART / "additional_node205_candidate.json"
APPLY_MODE = False


def current_peer() -> str:
    runs = recovery.identities()
    record = recovery.live_identity(PEER, runs[PEER])
    assert recovery.field(record, "UserId").startswith(getpass.getuser() + "(")
    assert recovery.field(record, "JobState") == "RUNNING"
    assert recovery.field(record, "NodeList") == NODE
    return record


def live_memory_via_peer(jid: int, before: str) -> dict:
    assert jid == TARGET
    assert recovery.field(before, "JobId") == str(TARGET)
    assert recovery.field(before, "UserId").startswith(getpass.getuser() + "(")
    assert recovery.field(before, "JobState") == "RUNNING"
    assert recovery.field(before, "NodeList") == NODE
    peer_before = current_peer()
    # Bash builtins avoid process creation inside the diagnostic step. Read only
    # the explicitly authorized target cgroup, never change any cgroup limits.
    script = f"""cg=/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{TARGET}
for fn in memory.current memory.high; do
  [[ -r "$cg/$fn" ]] || exit 4
  read -r value < "$cg/$fn"; printf '%s %s\\n' "$fn" "$value"
done
[[ -r "$cg/memory.stat" && -r "$cg/memory.events" ]] || exit 4
while read -r key value; do case "$key" in anon|shmem) printf 'stat.%s %s\\n' "$key" "$value";; esac; done < "$cg/memory.stat"
while read -r key value; do printf 'events.%s %s\\n' "$key" "$value"; done < "$cg/memory.events"
exit 0
"""
    command = ["timeout", "-k", "3s", "20s", "srun", f"--jobid={PEER}",
               "--overlap", "--exact", "--nodes=1", "--ntasks=1",
               "--cpus-per-task=1", "--mem=0", "--gres=none",
               f"--nodelist={NODE}", "/bin/bash", "-c", script]
    result = subprocess.run(command, text=True, capture_output=True, timeout=28)
    memory = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            memory[fields[0]] = int(fields[1])
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_job_id": TARGET, "probe_via_job": PEER, "node": NODE,
        "probe_policy": "Explicit owned fast peer on same node; one CPU, no GPU, existing allocation,20second timeout plus3second kill grace.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_script_sha256": EXPECTED_BASE_SHA256,
        "peer_before": peer_before, "target_before": before,
        "memory": memory, "probe_returncode": result.returncode,
        "probe_stdout": result.stdout, "probe_stderr": result.stderr,
        "same_complete_pressure_predicate": recovery.pressure(memory),
    }
    destination = recovery.ART / ("python_s43_probe_override_apply.json" if APPLY_MODE else "python_s43_probe_override_dryrun.json")
    recovery.atomic(destination, report)
    assert result.returncode == 0 and recovery.pressure(memory), "Complete current target pressure was not confirmed; no target mutation is authorized by this probe."
    peer_after = current_peer()
    for key in ("UserId", "NodeList", "StartTime", "Restarts"):
        assert recovery.field(peer_after, key) == recovery.field(peer_before, key), key
    report["peer_after"] = peer_after
    report["peer_identity_verified_after_probe"] = True
    recovery.atomic(destination, report)
    return {"checked_at_utc": report["checked_at_utc"], "probe_via_job": PEER,
            "memory": memory, "probe_override_audit": str(destination)}


def main() -> None:
    global APPLY_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    APPLY_MODE = args.apply
    assert hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest() == EXPECTED_BASE_SHA256
    entries = json.loads(ENTRY_PATH.read_text())["running_candidates"]
    assert len(entries) == 1 and entries[0]["job_id"] == TARGET
    entry = entries[0]
    assert entry["run"]["run_stamp"] == "e119_level2_python_drgrpo_s43"
    assert not (recovery.ART / str(TARGET) / "transaction.json").exists(), "Inspect an existing transaction before retrying."
    # The substitution is local to this Python process. The generic active
    # recovery processes and source file retain their original implementation.
    recovery.live_memory = live_memory_via_peer
    if not args.apply:
        before = recovery.live_identity(TARGET, entry["run"])
        assert recovery.submitline(before) == entry["original_submitline"]
        assert recovery.field(before, "MinMemoryNode") == "40G"
        evidence = live_memory_via_peer(TARGET, before)
        detail = recovery.timing_and_checkpoint(TARGET, entry["run"])
        assert detail["checkpoint"] and not detail["fresh_restart"]
        assert not recovery.near_checkpoint(detail, recovery.checkpoint_cadence(before, entry["run"]))
        print(json.dumps({"dry_run": "pass", "job_id": TARGET, "probe_job": PEER,
                          "current_step": detail["current_step"], "checkpoint_step": detail["checkpoint_step"],
                          "unsaved_steps": detail["unsaved_steps"], "probe_audit": evidence["probe_override_audit"]}))
        return
    recovery.apply_one(entry, True, argparse.Namespace(allow_fresh_restart=set(), allow_near_checkpoint=set()))


if __name__ == "__main__":
    main()
