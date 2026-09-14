#!/usr/bin/env python3
"""Bounded read-only startup observer for six exact existing science jobs."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import prioritize_e118_capacity_20260905 as base

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/campaign_startup_observer_20260909'
PLAN = ART / 'plan.json'
STATE = ART / 'latest.json'
LOCK = ART / 'observer.lock'
PROTOCOL = ROOT / 'paper/preregistration/campaign_startup_observer_20260909.md'
TRANSACTION = ROOT / 'var/artifacts/campaign_a5000_completion_20260909/transaction.json'
IDS = (31158503, 31158504, 31158505, 31158506, 31158507, 31151411)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(parts, timeout=10):
    return subprocess.run(parts, capture_output=True, text=True, timeout=timeout)


def prepare():
    assert not PLAN.exists() and not STATE.exists()
    tx = json.loads(TRANSACTION.read_text()); assert tx['status'] == 'complete'
    rows = [{'job_id': x['new_job_id'], 'run_dir': x['run_dir'],
             'checkpoint_step': x['checkpoint']['step'], 'expected_exports': base.exports(x['command']),
             'launcher': x['original_command'][-1]} for x in tx['items']]
    record = base.show(31151411); tokens = base.submit_tokens(record); env = base.exports(tokens)
    rows.append({'job_id': 31151411, 'run_dir': env['SAVE_PATH'], 'checkpoint_step': 1920,
                 'expected_exports': env, 'launcher': tokens[-1]})
    assert tuple(r['job_id'] for r in rows) == IDS
    for row in rows:
        row['launcher_sha256'] = digest(row['launcher'])
    plan = {'schema': 'campaign-startup-observer-20260909-v1', 'prepared_at_utc': base.now(),
            'rows': rows, 'source_transaction': str(TRANSACTION), 'source_transaction_sha256': digest(TRANSACTION),
            'controller_sha256': digest(__file__), 'helper_sha256': digest(base.__file__),
            'protocol_sha256': digest(PROTOCOL), 'maximum_watch_hours': 24, 'poll_seconds': 60,
            'read_only_science_jobs': True, 'scheduler_mutations_by_observer': False}
    base.atomic(PLAN, plan)
    print(json.dumps({'prepared': True, 'job_ids': IDS, 'plan': str(PLAN)}), flush=True)


def metrics(path):
    if not path.is_file():
        return None
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 512000))
        lines = handle.read().decode('utf8', 'replace').splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if 'trainer/policy_sgd_step' in value or 'misc/global_step' in value:
            return value
    return None


def inspect_log(path, previous, attempt_key):
    info = previous.get('log', {})
    if not path.is_file():
        return info
    st = path.stat()
    if info.get('attempt_key') != attempt_key or info.get('inode') != st.st_ino or info.get('offset', 0) > st.st_size:
        info = {'attempt_key': attempt_key, 'inode': st.st_ino, 'offset': 0,
                'restore_messages': [], 'fatal_messages': [], 'selected_checkpoint_step': None}
    with path.open('r', errors='replace') as handle:
        handle.seek(info['offset'])
        for line in handle:
            line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
            if '[slurm] host=' in line:
                info.update(restore_messages=[], fatal_messages=[], selected_checkpoint_step=None)
            if 'auto_resume=' in line:
                match = re.search(r'step_(\d+)', line)
                if match:
                    info['selected_checkpoint_step'] = int(match.group(1))
            if any(token in line for token in ('auto_resume=', 'Loaded checkpoint', 'Successfully loaded', 'Restored optimizer')):
                info['restore_messages'] = (info['restore_messages'] + [line[:1800]])[-12:]
            if any(token in line for token in ('Fatal Python error', 'Traceback (most recent call last)', 'CUDA out of memory', 'OutOfMemoryError', 'EngineDeadError')):
                info['fatal_messages'] = (info['fatal_messages'] + [line[:1800]])[-12:]
        info['offset'] = handle.tell()
    return info


def cgroup_probe(job):
    code = '''from pathlib import Path
import json
p=Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_JOB')
s={k:int(v) for k,v in (line.split() for line in (p/'memory.stat').read_text().splitlines())}
e={k:int(v) for k,v in (line.split() for line in (p/'memory.events').read_text().splitlines())}
r={'noncache_gib':sum(s.get(k,0) for k in ['anon','shmem','kernel'])/2**30,'events':e}
for k in ['memory.current','memory.high','memory.max','memory.peak']:
 if (p/k).exists():r[k]=(p/k).read_text().strip()
print(json.dumps(r))'''.replace('JOB', str(job))
    try:
        result = execute(['srun', '--overlap', f'--jobid={job}', '--nodes=1', '--ntasks=1',
                          '--cpus-per-task=1', '--mem=64M', '--time=00:01:00', 'python', '-c', code], timeout=8)
        return {'observed_at_utc': base.now(), 'returncode': result.returncode,
                'value': json.loads(result.stdout) if result.returncode == 0 else None,
                'stderr': result.stderr[-1500:]}
    except Exception as exc:
        return {'observed_at_utc': base.now(), 'probe_error': repr(exc), 'optional_probe': True}


def observe(item, previous):
    job = item['job_id']
    result = execute(['scontrol', 'show', 'job', '-dd', '-o', str(job)])
    if result.returncode or not result.stdout.strip():
        return {'job_id': job, 'status': 'controller_unavailable', 'error': result.stderr[-1500:],
                'previous_verified': previous.get('fresh_optimizer_verified', False)}
    record = result.stdout; tokens = base.submit_tokens(record)
    row = {'job_id': job, 'observed_at_utc': base.now(), 'scheduler_record': record,
           'scheduler': {key: base.field(record, key) for key in ('JobState', 'Reason', 'NodeList', 'RunTime', 'Restarts', 'StartTime', 'TimeLimit')},
           'fresh_optimizer_verified': False}
    if base.exports(tokens) != item['expected_exports'] or tokens[-1] != item['launcher'] or digest(item['launcher']) != item['launcher_sha256']:
        row['status'] = 'identity_or_source_changed'; return row
    state = row['scheduler']['JobState']
    attempt_key = row['scheduler']['StartTime'] + '/' + row['scheduler']['Restarts']
    row['log'] = inspect_log(Path(base.field(record, 'StdOut')), previous, attempt_key)
    row['status'] = state.lower()
    run = Path(item['run_dir']); receipt = run / 'TRAINING_COMPLETE.json'
    if receipt.is_file():
        data = json.loads(receipt.read_text())
        if data.get('terminal_step', 0) >= 3072:
            row.update(status='completed', fresh_optimizer_verified=True, terminal_receipt=data)
            return row
    path = run / f'debug_job{job}' / 'train_metrics.jsonl'
    latest = metrics(path)
    if latest:
        threshold = row['log'].get('selected_checkpoint_step') or item['checkpoint_step']
        stamp = row['scheduler']['StartTime']
        start_epoch = datetime.fromisoformat(stamp).timestamp() if stamp not in ('Unknown', 'N/A') else float('inf')
        row.update(latest_record_step=latest.get('trainer/policy_sgd_step', latest.get('misc/global_step')),
                   selected_checkpoint_step=threshold, metrics_path=str(path),
                   metrics_mtime=path.stat().st_mtime, metrics_age_seconds=time.time()-path.stat().st_mtime,
                   training_times={k: latest.get(k) for k in ('train/total_time', 'actor/total_time', 'misc/vllm_go_sleep_time', 'misc/vllm_wake_up_time')})
        row['fresh_optimizer_verified'] = (state == 'RUNNING' and row['latest_record_step'] > threshold
            and row['metrics_mtime'] >= start_epoch and row['metrics_age_seconds'] < 300
            and float(latest.get('train/total_time', 0)) > 0 and not row['log'].get('fatal_messages'))
        if row['fresh_optimizer_verified']:
            row['status'] = 'fresh_optimizer_verified'
            old_probe = previous.get('cgroup') if previous.get('log', {}).get('attempt_key') == attempt_key else None
            row['cgroup'] = old_probe or cgroup_probe(job)
    if row['log'].get('fatal_messages'):
        row.update(status='current_attempt_fatal', fresh_optimizer_verified=False)
    return row


def run(watch):
    plan = json.loads(PLAN.read_text())
    assert digest(__file__) == plan['controller_sha256'] and digest(base.__file__) == plan['helper_sha256']
    assert digest(PROTOCOL) == plan['protocol_sha256']
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        'schema': plan['schema'], 'plan_sha256': digest(PLAN), 'started_at_utc': base.now(),
        'deadline_utc': (datetime.now(timezone.utc)+timedelta(hours=24)).isoformat(), 'jobs': {}}
    assert state['plan_sha256'] == digest(PLAN)
    while True:
        if datetime.now(timezone.utc) >= datetime.fromisoformat(state['deadline_utc']):
            state['status'] = 'deadline_reached'; base.atomic(STATE, state); return
        for item in plan['rows']:
            previous = state['jobs'].get(str(item['job_id']), {})
            try:
                row = observe(item, previous)
            except Exception as exc:
                row = {'job_id': item['job_id'], 'status': 'observation_error', 'error': repr(exc)}
            state['jobs'][str(item['job_id'])] = row
            transition = (row.get('status'), row.get('scheduler')) != (previous.get('status'), previous.get('scheduler'))
            with (ART / 'observations.jsonl').open('a') as handle:
                compact = {k:v for k,v in row.items() if k not in ('scheduler_record','log')}
                if transition:
                    compact.update(scheduler_record=row.get('scheduler_record'), log=row.get('log'))
                handle.write(json.dumps(compact)+'\n')
        state['updated_at_utc'] = base.now()
        state['status'] = 'all_verified' if all(r.get('fresh_optimizer_verified') for r in state['jobs'].values()) and len(state['jobs']) == 6 else 'watching'
        base.atomic(STATE, state)
        print(json.dumps({'at': state['updated_at_utc'], 'status': state['status'],
            'jobs': [{'job_id': r['job_id'], 'status': r['status'], 'step': r.get('latest_record_step')} for r in state['jobs'].values()]}), flush=True)
        if state['status'] == 'all_verified' or not watch:
            return
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','once','watch'))
    args = parser.parse_args(); ART.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if args.phase == 'prepare' else run(args.phase == 'watch')
