#!/usr/bin/env python3
"""Observe current node208 MathIR startup without reading evaluation outcomes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

import backfill_e118_node208_20260908 as placement

ART = placement.c.ART
OUTPUT = ART / 'node208_startup_verified.json'


def observe(old):
    audit_path = ART / f'{old}.json'
    if not audit_path.exists():
        return dict(old_job_id=old, first_optimizer_verified=False, status='not_submitted')
    audit = json.loads(audit_path.read_text())
    item = audit['replacements'][0]
    job = item.get('new_job_id')
    row = dict(old_job_id=old, job_id=job, arm=item['arm'], seed=item['seed'],
               transaction=str(audit_path), run_dir=item['run_dir'],
               vllm_gpu_ratio=0.25, first_optimizer_verified=False)
    if not job:
        return row
    record = placement.c.base.show(job)
    keys = ['JobState', 'NodeList', 'MinMemoryNode', 'NumCPUs', 'TresPerNode',
            'TimeLimit', 'Restarts', 'StartTime', 'Account', 'Partition']
    row['scheduler'] = {key: placement.c.base.field(record, key) for key in keys}
    if row['scheduler']['JobState'] == 'RUNNING' and row['scheduler']['NodeList'] == 'node208':
        remote = """from pathlib import Path
import json
p=Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_JOB')
s={k:int(v) for k,v in (line.split() for line in (p/'memory.stat').read_text().splitlines())}
e={k:int(v) for k,v in (line.split() for line in (p/'memory.events').read_text().splitlines())}
r={'events':e,'noncache_gib':sum(s.get(k,0) for k in ['anon','shmem','kernel'])/2**30,
   'stat':{k:s.get(k,0) for k in ['anon','shmem','kernel','file','inactive_file','active_file','file_dirty','file_writeback']}}
for key in ['memory.current','memory.peak','memory.high','memory.max']:
 r[key]=(p/key).read_text().strip()
print(json.dumps(r))""".replace('job_JOB', f'job_{job}')
        probe = subprocess.run(['timeout', '-k', '3s', '25s', 'srun', f'--jobid={job}',
                                '--overlap', '--exact', '--nodes=1', '--ntasks=1',
                                '--cpus-per-task=1', '--mem=0', '--gres=none',
                                '/usr/bin/python3', '-c', remote],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            row['cgroup'] = json.loads(probe.stdout)
        else:
            row['probe_error'] = probe.stderr[-1500:]
    attempt = Path(item['run_dir']) / f'debug_job{job}'
    metrics = attempt / 'train_metrics.jsonl'
    row['metrics_path'] = str(metrics)
    if metrics.exists():
        parsed = []
        for line in metrics.read_text().splitlines():
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        row['metrics_age_seconds'] = time.time() - metrics.stat().st_mtime
        if parsed:
            last = parsed[-1]
            row['fresh_step'] = last.get('trainer/global_step', 0)
            row['sleep_wake_metrics'] = {k: v for k, v in last.items() if 'sleep' in k or 'wake' in k}
    clean = []
    for key in ['StdOut', 'StdErr']:
        log = Path(placement.c.base.field(record, key))
        row[key.lower() + '_path'] = str(log)
        text = log.read_text(errors='replace') if log.exists() else ''
        if '[slurm] host=' in text:
            text = text[text.rfind('[slurm] host='):]
        clean.extend(re.sub(r'\x1b\[[0-9;]*m', '', line) for line in text.splitlines())
    row['fatal_errors'] = [x for x in clean if re.search(r'Traceback|Error:|Fatal Python|CUDA out of memory|OutOfMemoryError|oom-kill', x)][-10:]
    row['initialization_evidence'] = [x for x in clean if re.search(r'# GPU blocks:|Maximum concurrency|CPU Offload:|After initializing ZeRO optimizer', x)][-6:]
    events = row.get('cgroup', {}).get('events', {})
    scheduler_ok = all(row['scheduler'][k] == v for k, v in dict(
        JobState='RUNNING', NodeList='node208', MinMemoryNode='116G', NumCPUs='16',
        TresPerNode='gres/gpu:a6000:1', TimeLimit='3-00:00:00', Account='allcs',
        Partition='lowprio', Restarts='0').items())
    row['first_optimizer_verified'] = bool(
        scheduler_ok and row.get('fresh_step', 0) > 0 and row.get('metrics_age_seconds', 999999) < 180
        and not row['fatal_errors'] and all(events.get(k, -1) == 0 for k in ['high', 'oom', 'oom_kill'])
        and row.get('cgroup', {}).get('noncache_gib', 999999) < 116)
    return row


def check():
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(observe, placement.PROFILE_IDS))
    result = dict(schema='e118-node208-first-optimizer-monitor-v1',
                  checked_at_utc=datetime.now(timezone.utc).isoformat(), jobs=rows,
                  all_verified=len(rows) == 4 and all(row['first_optimizer_verified'] for row in rows))
    placement.c.base.atomic(OUTPUT, result)
    print(json.dumps(dict(checked_at_utc=result['checked_at_utc'], all_verified=result['all_verified'],
                          jobs=[{k: row.get(k) for k in ['job_id', 'fresh_step', 'first_optimizer_verified', 'fatal_errors', 'probe_error']} for row in rows])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-seconds', type=int, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.watch_seconds
    while True:
        result = check()
        terminal = any(row.get('scheduler', {}).get('JobState') in ['FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'TIMEOUT'] for row in result['jobs'])
        if result['all_verified'] or terminal or any(row.get('fatal_errors') for row in result['jobs']) or time.monotonic() >= deadline:
            break
        time.sleep(min(30, max(0, deadline-time.monotonic())))
