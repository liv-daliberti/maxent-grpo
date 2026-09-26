#!/usr/bin/env python3
"""Same-ID CPU-only repair for the observed E119 hourly supervisor memory stall.

This independent receipt does not edit any protected guard, scientific job,
ledger, retry count, deadline, or frozen helper. Prepare never mutates Slurm.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import guard_e119_lowprio_20260909 as guard

ROOT = guard.ROOT
ART = ROOT / 'var/artifacts/e119_hourly_guard_memory_recovery_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
LOCK = ART / 'operation.lock'
CPU = 31159699
SCIENCE = 31159682
MEMORY = '4G'
EVIDENCE = Path('/tmp/e119_node915_lock_audit_20260909.json')
CPU_FIELDS = ('UserId', 'JobName', 'Account', 'Partition', 'ReqNodeList', 'ExcNodeList',
              'TimeLimit', 'Requeue', 'Nice', 'QOS', 'Dependency', 'Features', 'WorkDir',
              'NumTasks', 'CPUs/Task', 'Comment', 'StdOut', 'StdErr')
SCIENCE_STATE = ('started_at_utc', 'deadline_utc', 'cleanup_deadline_utc', 'last_resume_step',
                 'attempts', 'transition_errors')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def command(args, *, check=True, timeout=45):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def event(tx, label):
    tx.setdefault('events', []).append({'at': datetime.now(timezone.utc).isoformat(), 'event': label})
    write(TX, tx)


def show(job, *, timeout=45):
    return command(['scontrol', 'show', 'job', '-dd', '-o', str(job)], timeout=timeout).stdout


def field(record, key):
    return guard.base.field(record, key)


def cpu_profile(plan, record, *, memory=None):
    require(field(record, 'JobId') == str(CPU), 'CPU job ID differs')
    require(field(record, 'UserId').endswith(f'({os.getuid()})'), 'CPU ownership differs')
    require('gres/gpu' not in field(record, 'ReqTRES'), 'CPU supervisor unexpectedly requests GPUs')
    require(guard.base.submit_tokens(record) == plan['cpu_submit_tokens'], 'CPU original submission changed')
    for key, value in plan['cpu_resources'].items():
        require(field(record, key) == value, 'CPU immutable resource changed: ' + key)
    require(field(record, 'NumNodes') in {'1', '1-1'}, 'CPU node count differs')
    if memory is not None:
        require(field(record, 'MinMemoryNode') == memory, 'CPU memory differs')
    return record


def science_snapshot():
    tx = read(guard.TX)
    require(tx['status'] == 'watching' and not tx.get('memory_failure'), 'Scientific guard is not in a healthy persisted state')
    require(all(x.get('released') for x in tx['attempts']), 'Unfinished scientific mutation must be reconciled first')
    return {key: tx[key] for key in SCIENCE_STATE}


def audit_science(plan):
    for path, expected in plan['files_sha256'].items():
        require(digest(path) == expected, 'Frozen file changed: ' + path)
    gplan = read(guard.PLAN)
    require(gplan['new_job_id'] == SCIENCE, 'Science scope differs')
    guard.frozen(gplan)
    guard.identity(gplan)
    guard.stable(gplan, show(SCIENCE))
    guard.no_other_writer(gplan)
    require(science_snapshot() == plan['science_state'], 'Science retry counters or absolute deadlines changed')


def held(plan, record):
    cpu_profile(plan, record)
    require(field(record, 'JobState') == 'PENDING' and field(record, 'Reason') == 'job_requeued_in_held_state'
            and field(record, 'Priority') == '0', 'Exact owned CPU requeue hold is missing')
    require(int(field(record, 'Restarts')) == plan['cpu_restarts'] + 1, 'CPU requeue must increment restarts exactly once')


def await_owned_hold(plan, record):
    deadline = time.monotonic() + 30
    while field(record, 'JobState') == 'COMPLETING':
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'CPU is still completing; poll and rerun without repeating requeue')
        time.sleep(min(1, remaining))
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'CPU is still completing; poll and rerun without repeating requeue')
        record = show(CPU, timeout=min(5, remaining))
    held(plan, record)
    return record


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'Existing immutable CPU recovery preparation')
    cpu = show(CPU)
    require(field(cpu, 'JobState') == 'RUNNING' and field(cpu, 'MinMemoryNode') == '512M', 'Expected stalled512MiB running supervisor missing')
    proof = read(EVIDENCE)
    observation = json.loads(proof['stdout'])
    require(proof['returncode'] == 0 and any(row.get('wchan') == 'mem_cgroup_handle_over_high'
            and row.get('global_lock_fds') and 'guard_e119_lowprio_20260909.py' in row.get('scripts', [])
            for row in observation['guards']), 'Observed owned-lock memory stall proof missing')
    ready = read(guard.CAPACITY_ART / 'guard_ready.json')
    require(ready['guard_job_id'] == CPU and ready['new_job_id'] == SCIENCE, 'Registered CPU/science pair differs')
    gplan = read(guard.PLAN)
    paths = {str(Path(__file__).resolve()), str(guard.PLAN), str(EVIDENCE), str(Path(guard.__file__).resolve()),
             str(guard.PROTOCOL), str(guard.CAPACITY_PLAN), *gplan['helper_sha256']}
    plan = {'schema': 'e119-hourly-cpu-memory-recovery-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
            'cpu_job_id': CPU, 'science_job_id': SCIENCE, 'memory_before': '512M', 'memory_after': MEMORY,
            'cpu_before': cpu, 'cpu_restarts': int(field(cpu, 'Restarts')),
            'cpu_submit_tokens': guard.base.submit_tokens(cpu),
            'cpu_resources': {key: field(cpu, key) for key in CPU_FIELDS},
            'science_state': science_snapshot(), 'files_sha256': {p: digest(p) for p in paths}}
    cpu_profile(plan, cpu, memory='512M'); audit_science(plan)
    write(PLAN, plan)
    print(json.dumps({'prepared': True, 'plan_sha256': digest(PLAN), 'scheduler_mutations': False}))


def apply():
    plan = read(PLAN)
    tx = read(TX) if TX.exists() else {'plan_sha256': digest(PLAN), 'events': []}
    require(tx['plan_sha256'] == digest(PLAN), 'Recovery plan changed')
    if tx.get('released'):
        cpu_profile(plan, show(CPU), memory=MEMORY)
        print(json.dumps({'already_released': CPU})); return
    audit_science(plan)
    record = show(CPU); cpu_profile(plan, record)
    if tx.get('release_intent') and field(record, 'Priority') != '0':
        cpu_profile(plan, record, memory=MEMORY)
        require(field(record, 'JobState') in {'RUNNING', 'PENDING', 'CONFIGURING'}
                and int(field(record, 'Restarts')) == plan['cpu_restarts'] + 1,
                'Unexpected state while reconciling CPU release')
        tx.update(released=True, cpu_after=record)
        event(tx, 'Reconciled exact CPU release without repeating any mutation')
        print(json.dumps({'released': CPU, 'reconciled': True})); return
    if not tx.get('requeue_intent'):
        require(field(record, 'JobState') == 'RUNNING' and int(field(record, 'Restarts')) == plan['cpu_restarts'], 'CPU changed before requeue intent')
        require(field(record, 'MinMemoryNode') == '512M', 'Unexpected initial CPU memory')
        tx['requeue_intent'] = True
        event(tx, 'Persisted CPU-only requeuehold intent; uncertain calls are never repeated')
        command(['scontrol', 'requeuehold', str(CPU)])
    # The only safe response to an ambiguous requeue is to inspect its exact hold.
    # Never resend requeuehold after a persisted intent, even if the old job is R.
    record = await_owned_hold(plan, show(CPU))
    require(field(record, 'MinMemoryNode') in {'512M', MEMORY}, 'Unreviewed held CPU memory')
    if field(record, 'MinMemoryNode') == '512M':
        if tx.get('memory_intent'):
            raise RuntimeError('Unacknowledged memory update remains unchanged; inspect manually')
        tx['memory_intent'] = True
        event(tx, 'Persisted held CPU-only memory amendment from512MiB to4GiB')
        command(['scontrol', 'update', f'JobId={CPU}', 'MinMemoryNode=4096'])
    record = show(CPU); held(plan, record); cpu_profile(plan, record, memory=MEMORY)
    audit_science(plan)
    tx['held_after'] = record
    if not tx.get('release_intent'):
        tx['release_intent'] = True
        event(tx, 'Persisted CPU release intent after resource/science/deadline audits')
        command(['scontrol', 'release', str(CPU)])
    record = show(CPU); cpu_profile(plan, record, memory=MEMORY)
    require(field(record, 'JobState') in {'RUNNING', 'PENDING', 'CONFIGURING'} and field(record, 'Priority') != '0', 'CPU release unacknowledged; do not repeat')
    require(int(field(record, 'Restarts')) == plan['cpu_restarts'] + 1, 'Unexpected CPU restart count after release')
    tx.update(released=True, cpu_after=record)
    event(tx, 'Same CPU ID released with4GiB; science identities/retries/absolute deadline preserved')
    print(json.dumps({'released': CPU, 'state': field(record, 'JobState'), 'memory': MEMORY}))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=['prepare', 'apply'])
    args = parser.parse_args(); guard.base.command = command
    ART.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'prepare': prepare()
        else: apply()


if __name__ == '__main__': main()
