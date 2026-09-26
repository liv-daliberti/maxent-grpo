#!/usr/bin/env python3
"""Migrate exactly two reviewed E119 cells to64GiB at their next durable save.

This preventive baseline change does not claim either learner currently failed.
Dry-run by default. Process-local wrappers preserve the generic held transaction.
"""
from __future__ import annotations
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'ops/exp_scaling')]
import recover_e119_memory_pressure_20260905 as recovery

BASE_SHA = 'b254524f5b73b9c016ff272597ba1a1b361954baf428336300eeedc7d2f1ad15'
PLAN = recovery.ART / 'final_four_migration_candidates.json'
PLAN_SHA = '0aecce2953cf057dd43b34272c9aa9278e349ede2462c0614136365d4e70b06e'
CONFIG = {
    31048196: {'step': 1920, 'peer': 31048166, 'node': 'node202', 'stamp': 'e119_level2_python_drgrpo_s44'},
    31075763: {'step': 1344, 'peer': 31045874, 'node': 'node105', 'stamp': 'e119_level2_python_replay_drgrpo_s45'},
}
ORIGINAL_TIMING = recovery.timing_and_checkpoint
TARGET = None
ENTRY = None
APPLY_MODE = False


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_detail_valid(detail: dict, target: int) -> None:
    expected = CONFIG[target]['step']
    assert detail['checkpoint_step'] == expected, f'Wait for fully valid step{expected}; review other checkpoints.'
    assert detail['checkpoint'] and not detail['fresh_restart']
    counters = detail['saved_counter_validation']['saved_counters']
    assert all(counters.get(k) == expected for k in ('global_steps', 'global_step', 'prompt_batches_consumed_total'))
    assert detail['current_step'] is not None and expected - 1 <= detail['current_step'] <= expected + 16
    assert 0 <= detail['unsaved_steps'] <= 16, 'Preventive rollback allowance is at most16 updates.'


def bounded_timing(jid: int, run: dict) -> dict:
    assert jid == TARGET and run['run_stamp'] == CONFIG[jid]['stamp']
    detail = ORIGINAL_TIMING(jid, run)
    saved_detail_valid(detail, jid)
    return detail


def same_attempt(record: str) -> None:
    assert TARGET in CONFIG and ENTRY is not None
    assert recovery.field(record, 'JobId') == str(TARGET)
    assert recovery.field(record, 'UserId').startswith(getpass.getuser() + '(')
    assert recovery.field(record, 'JobState') == 'RUNNING'
    assert recovery.field(record, 'MinMemoryNode') == '40G'
    for key in ('NodeList', 'StartTime', 'Restarts', 'UserId'):
        assert recovery.field(record, key) == recovery.field(ENTRY['before'], key), key
    assert recovery.field(record, 'NodeList') == CONFIG[TARGET]['node']
    assert recovery.submitline(record) == ENTRY['original_submitline']


def complete_memory(memory: dict, old: dict) -> None:
    required = ('memory.current', 'memory.high', 'stat.anon', 'stat.shmem', 'events.high', 'events.oom', 'events.oom_kill')
    assert all(isinstance(memory.get(k), int) and memory[k] >= 0 for k in required)
    assert memory['memory.high'] == old['memory.high'] == 40 * 1024**3
    assert memory['events.high'] >= old['events.high'], 'Cgroup counters must belong to the same attempt.'
    assert memory['events.oom'] == old['events.oom'] == 0
    assert memory['events.oom_kill'] == old['events.oom_kill'] == 0


def preventive_basis(jid: int, before: str) -> dict:
    assert jid == TARGET
    same_attempt(before)
    detail = bounded_timing(jid, ENTRY['run'])
    config = CONFIG[jid]
    peer_id = config['peer']
    peer = recovery.live_identity(peer_id, recovery.identities()[peer_id])
    assert recovery.field(peer, 'UserId').startswith(getpass.getuser() + '(')
    assert recovery.field(peer, 'JobState') == 'RUNNING'
    assert recovery.field(peer, 'NodeList') == config['node']
    assert recovery.field(peer, 'MinMemoryNode') == '64G'
    script = f'''cg=/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{jid}
for fn in memory.current memory.high; do
  [[ -r "$cg/$fn" ]] || exit 4
  read -r value < "$cg/$fn"; printf '%s %s\\n' "$fn" "$value"
done
[[ -r "$cg/memory.stat" && -r "$cg/memory.events" ]] || exit 4
while read -r key value; do case "$key" in anon|shmem) printf 'stat.%s %s\\n' "$key" "$value";; esac; done < "$cg/memory.stat"
while read -r key value; do printf 'events.%s %s\\n' "$key" "$value"; done < "$cg/memory.events"
'''
    cmd = ['timeout', '-k', '3s', '20s', 'srun', f'--jobid={peer_id}', '--overlap', '--exact', '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=0', '--gres=none', f"--nodelist={config['node']}", '/bin/bash', '-c', script]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=28)
    memory = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            memory[parts[0]] = int(parts[1])
    report = {
        'checked_at_utc': recovery.now(), 'target_job_id': jid,
        'basis': 'PREVENTIVE: migrate the remaining40GiB E119 baseline after repeated checkpoint memory throttling across this cohort; this target is not claimed currently failed.',
        'current_pressure_confirmed': recovery.pressure(memory),
        'memory': memory, 'prior_memory': ENTRY['pressure_evidence']['memory'],
        'checkpoint': detail, 'probe_via_job': peer_id,
        'probe_returncode': result.returncode, 'probe_stdout': result.stdout,
        'probe_stderr': result.stderr, 'target_before': before, 'peer_before': peer,
        'script_sha256': sha(Path(__file__)), 'base_sha256': BASE_SHA,
    }
    dest = recovery.ART / f"preventive_{jid}_{'apply' if APPLY_MODE else 'dryrun'}.json"
    recovery.atomic(dest, report)
    assert result.returncode == 0, 'Read-only probe failed; no mutation.'
    complete_memory(memory, ENTRY['pressure_evidence']['memory'])
    same_attempt(recovery.live_identity(jid, ENTRY['run']))
    peer_after = recovery.live_identity(peer_id, recovery.identities()[peer_id])
    for key in ('UserId', 'JobState', 'NodeList', 'StartTime', 'Restarts'):
        assert recovery.field(peer_after, key) == recovery.field(peer, key), key
    report['identity_and_bounded_rollback_verified'] = True
    recovery.atomic(dest, report)
    return {'checked_at_utc': report['checked_at_utc'], 'probe_via_job': peer_id,
            'memory': memory, 'current_pressure_confirmed': report['current_pressure_confirmed'],
            'recovery_basis': report['basis'], 'preventive_audit': str(dest)}


def main() -> None:
    global TARGET, ENTRY, APPLY_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True, type=int, choices=sorted(CONFIG))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    TARGET, APPLY_MODE = args.job_id, args.apply
    assert sha(Path(recovery.__file__)) == BASE_SHA
    assert sha(PLAN) == PLAN_SHA
    ENTRY = next(e for e in json.loads(PLAN.read_text())['running_candidates'] if e['job_id'] == TARGET)
    assert ENTRY['classification'] == 'PREVENTIVE_MEMORY_BASELINE'
    assert ENTRY['run']['run_stamp'] == CONFIG[TARGET]['stamp']
    assert not (recovery.ART / str(TARGET) / 'transaction.json').exists(), 'Inspect an existing transaction before retrying.'
    recovery.live_memory = preventive_basis
    recovery.timing_and_checkpoint = bounded_timing
    if not args.apply:
        evidence = preventive_basis(TARGET, recovery.live_identity(TARGET, ENTRY['run']))
        print(json.dumps({'dry_run': 'pass', 'job_id': TARGET, 'checkpoint_step': CONFIG[TARGET]['step'], 'evidence': evidence}))
        return
    recovery.apply_one(ENTRY, True, argparse.Namespace(allow_fresh_restart=set(), allow_near_checkpoint=set()))


if __name__ == '__main__':
    main()
