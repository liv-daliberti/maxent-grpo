#!/usr/bin/env python3
"""Bounded, read-only verification of the current Countdown replay attempt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

import backfill_e118_countdown_replay_s74_20260908 as placement

ART = placement.c.ART
OUTPUT = ART / 'node203_countdown_replay_startup.json'
RECEIPT = ART / 'node203_countdown_replay_first_optimizer_verified.json'
REPORT = ART / 'node203_countdown_replay_report.md'
LOCAL_TZ = ZoneInfo('America/New_York')


def observe():
    tx = json.loads((ART / f'{placement.OLD_JOB}.json').read_text())
    item = tx['replacements'][0]
    job = item.get('new_job_id')
    row = dict(old_job_id=placement.OLD_JOB, job_id=job, profile=item['profile'],
               transaction_status=tx['status'], run_dir=item['run_dir'],
               checkpoint_step=item['checkpoint_step'], first_optimizer_verified=False)
    if not job:
        return row
    record = placement.c.base.show(job)
    keys = ['JobState', 'NodeList', 'StartTime', 'Restarts', 'MinMemoryNode',
            'NumCPUs', 'TresPerNode', 'TimeLimit', 'Account', 'Partition', 'StdOut', 'StdErr']
    scheduler = {k: placement.c.base.field(record, k) for k in keys}
    row['scheduler'] = scheduler
    if scheduler['JobState'] != 'RUNNING' or scheduler['NodeList'] != 'node203':
        return row
    start_local = datetime.fromisoformat(scheduler['StartTime']).replace(tzinfo=LOCAL_TZ)
    start = start_local.astimezone(timezone.utc)
    row['current_start_time_utc'] = start.isoformat()
    remote = """from pathlib import Path
import json
p=Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_JOB')
s={k:int(v) for k,v in (line.split() for line in (p/'memory.stat').read_text().splitlines())}
e={k:int(v) for k,v in (line.split() for line in (p/'memory.events').read_text().splitlines())}
print(json.dumps({'noncache_gib':sum(s.get(k,0) for k in ['anon','shmem','kernel'])/2**30,
 'events':e,'memory_current':int((p/'memory.current').read_text()),'memory_high':int((p/'memory.high').read_text()),'memory_peak':int((p/'memory.peak').read_text())}))""".replace('job_JOB', f'job_{job}')
    probe = subprocess.run(['timeout', '-k', '3s', '25s', 'srun', f'--jobid={job}',
                            '--overlap', '--exact', '--nodes=1', '--ntasks=1',
                            '--cpus-per-task=1', '--mem=0', '--gres=none',
                            '/usr/bin/python3', '-c', remote], capture_output=True, text=True)
    if probe.returncode == 0:
        row['cgroup'] = json.loads(probe.stdout)
    else:
        row['probe_error'] = probe.stderr[-1500:]
    metrics = Path(item['run_dir']) / f'debug_job{job}' / 'train_metrics.jsonl'
    row['metrics_path'] = str(metrics)
    if metrics.exists():
        parsed = []
        for line in metrics.read_text().splitlines():
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        mtime = metrics.stat().st_mtime
        row.update(metrics_mtime_utc=datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
                   metrics_age_seconds=time.time() - mtime,
                   metric_mtime_after_current_start=mtime >= start.timestamp())
        if parsed:
            last = parsed[-1]
            row['fresh_step'] = last.get('trainer/global_step', 0)
            row['sleep_wake_metrics'] = {k: v for k, v in last.items() if 'sleep' in k or 'wake' in k}
    clean = []
    for key in ['StdOut', 'StdErr']:
        path = Path(scheduler[key])
        text = path.read_text(errors='replace') if path.exists() else ''
        if '[slurm] host=' in text:
            text = text[text.rfind('[slurm] host='):]
        clean.extend(re.sub(r'\x1b\[[0-9;]*m', '', line) for line in text.splitlines())
    row['fatal_errors'] = [s for s in clean if re.search(r'Traceback|Error:|Fatal Python|CUDA out of memory|OutOfMemoryError|oom-kill', s)][-10:]
    row['initialization_evidence'] = [s for s in clean if re.search(r'CPU Offload:|After initializing ZeRO optimizer|GPU blocks:|Maximum concurrency', s)][-6:]
    completion = []
    for line in clean:
        match = re.search(r'I(\d{2})(\d{2}) (\d{2}:\d{2}:\d{2}\.\d+).*post-learning done step=(\d+)', line)
        if match:
            stamp = datetime.fromisoformat(f'{start_local.year}-{match[1]}-{match[2]}T{match[3]}').replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc)
            if stamp >= start and int(match[4]) > item['checkpoint_step']:
                completion.append(dict(at_utc=stamp.isoformat(), step=int(match[4]), line=line))
    row['current_attempt_optimizer_completions'] = completion[-5:]
    after = placement.c.base.show(job)
    stable = all(placement.c.base.field(after, k) == scheduler[k] for k in ['JobState', 'NodeList', 'StartTime', 'Restarts'])
    timing = row.get('sleep_wake_metrics', {})
    events = row.get('cgroup', {}).get('events', {})
    expected = dict(JobState='RUNNING', NodeList='node203', MinMemoryNode='116G', NumCPUs='16',
                    TresPerNode='gres/gpu:a5000:1', TimeLimit='3-00:00:00', Account='allcs', Partition='lowprio')
    row['checks'] = dict(
        scheduler_matches=all(scheduler[k] == v for k, v in expected.items()),
        attempt_stable_during_probe=stable,
        metric_mtime_after_current_start=row.get('metric_mtime_after_current_start', False),
        positive_step_above_checkpoint_baseline=row.get('fresh_step', 0) > item['checkpoint_step'],
        timestamped_current_attempt_optimizer_completion=bool(completion),
        metrics_recent=row.get('metrics_age_seconds', 999999) < 180,
        sleep_wake_positive=any('sleep' in k and float(v) > 0 for k, v in timing.items()) and any('wake' in k and float(v) > 0 for k, v in timing.items()),
        no_fatal_errors=not row['fatal_errors'],
        no_high_oom_events=all(events.get(k, -1) == 0 for k in ['high', 'oom', 'oom_kill']),
        noncache_below_reservation=row.get('cgroup', {}).get('noncache_gib', 999999) < 116)
    row['first_optimizer_verified'] = all(row['checks'].values())
    return row


def check():
    row = observe()
    result = dict(schema='e118-node203-countdown-replay-current-attempt-v1',
                  checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  row=row, all_verified=row['first_optimizer_verified'])
    placement.c.base.atomic(OUTPUT, result)
    if result['all_verified']:
        if not RECEIPT.exists():
            result.update(observer_path=str(Path(__file__).resolve()),
                          observer_sha256=placement.c.recovery.digest(__file__))
            placement.c.base.atomic(RECEIPT, result)
        saved = json.loads(RECEIPT.read_text())
        assert saved['row']['job_id'] == row['job_id']
        r = saved['row']
        REPORT.write_text(f"# E118 Countdown replay seed-74 startup, 2026-09-08\n\n"
                          f"Verified: {saved['checked_at_utc']}\n\n"
                          f"Existing job {placement.OLD_JOB} was replaced by {r['job_id']} on node203. "
                          f"The current attempt is RUNNING and reached optimizer step {int(r['fresh_step'])} "
                          f"above checkpoint baseline {r['checkpoint_step']}. "
                          f"Noncache memory is {r['cgroup']['noncache_gib']:.2f} GiB, with positive sleep/wake "
                          f"timings, no fatal errors, and zero cgroup high/OOM events.\n\n"
                          f"The job reserves one A5000, 16 CPUs, 116 GiB, and 72 hours on allcs/lowprio. "
                          f"VLLM GPU ratio 0.40 is the reviewed runtime allocation amendment; the frozen "
                          f"launcher, scientific settings, run identity, and PVL exclusion are preserved.\n\n"
                          f"Current StartTime: {r['current_start_time_utc']}. Metrics mtime: "
                          f"{r['metrics_mtime_utc']}. Timestamped post-learning completion confirms "
                          f"optimizer work after the current start; the scheduler attempt stayed stable "
                          f"during the bounded read-only probe.\n\n"
                          f"Receipt: {RECEIPT}\n\nTransaction: {ART / str(placement.OLD_JOB)}.json\n\n"
                          f"Wrapper: {placement.WRAPPER}\n\nProtocol: {placement.PROTOCOL}\n")
    print(json.dumps(dict(at=result['checked_at_utc'], job=row.get('job_id'),
                          scheduler=row.get('scheduler', {}).get('JobState'),
                          step=row.get('fresh_step'), all_verified=result['all_verified'],
                          fatal_errors=row.get('fatal_errors'), probe_error=row.get('probe_error'))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-seconds', type=int, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.watch_seconds
    while True:
        result = check()
        row = result['row']
        terminal = row.get('scheduler', {}).get('JobState') in ['FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'TIMEOUT']
        if result['all_verified'] or terminal or row.get('fatal_errors') or time.monotonic() >= deadline:
            break
        time.sleep(min(30, max(0, deadline - time.monotonic())))
