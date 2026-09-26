#!/usr/bin/env python3
"""Render audited coding128 results and terminal accounting; never launch GPU work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'var/seed_paper_eval/paper310/bin/python'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def parse_accounting(raw, jobs, historical_ids):
    rows, seen = [], set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        job, state, exit_code, elapsed, tres, start, end, node = line.split('|')
        job = int(job)
        if job not in jobs or job in seen:
            raise ValueError('unexpected or duplicate scheduler parent job')
        seen.add(job)
        state = state.split()[0].rstrip('+')
        if state not in TERMINAL:
            raise ValueError(f'job {job} is not terminal')
        if job not in historical_ids and (state != 'COMPLETED' or exit_code != '0:0'):
            raise ValueError(f'active campaign job {job} did not complete successfully')
        allocation = dict(x.split('=', 1) for x in tres.split(',') if '=' in x)
        count = int(allocation['gres/gpu']) if 'gres/gpu' in allocation else sum(int(v) for k, v in allocation.items() if k.startswith('gres/gpu:'))
        if count != 1:
            raise ValueError('expected one allocated GPU for every submitted campaign job')
        if int(elapsed) < 0:
            raise ValueError('negative elapsed allocation')
        rows.append({'job_id': job, 'run': jobs[job], 'state': state, 'exit_code': exit_code,
                     'elapsed_seconds': int(elapsed), 'allocated_gpu_count': count,
                     'allocated_gpu_hours': int(elapsed) * count / 3600,
                     'historical_attempt': job in historical_ids, 'allocation': tres,
                     'scheduler_start': start, 'scheduler_end': end, 'node': node})
    if seen != set(jobs):
        raise ValueError('scheduler omitted submitted campaign jobs')
    return rows


def finish(execution):
    freeze = read(execution / 'protocol_freeze.json')
    config_path = execution / 'campaign_config.json'
    if freeze['status'] != 'FROZEN_AUTHORIZED_READY' or sha(config_path) != freeze['campaign_config_sha256']:
        raise ValueError('campaign configuration differs from its freeze')
    import drive_corrected_code128_20260922 as driver
    config, schedule = driver.verify_frozen_config(execution)
    summary_path = execution / 'summary.json'
    summary = read(summary_path)
    if summary.get('status') != 'pass' or summary.get('schema') != 'independent-native-hf-code128-audit-20260922-v1' or summary.get('kind') != 'complete_fixed_native_code128_comparison':
        raise ValueError('independent aggregate did not pass')
    if summary.get('schedule_sha256') != sha(config['schedule_path']) or summary.get('primary_checkpoint') != 128 or summary.get('samples') != 23040 or len(summary.get('shards', [])) != 13 or summary.get('paired_training_seeds') != 1:
        raise ValueError('aggregate is not the fixed complete campaign')
    paths = [Path(p) for p in config['validation_runs'] + config.get('historical_runs', [])]
    paths += [Path(p) for p in config['training_runs'].values()]
    paths += [Path(schedule['run_parent']) / s['name'] for s in schedule['shards']]
    jobs = {}
    for path in paths:
        receipt = read(path / 'submission.json')
        if receipt.get('returncode') != 0:
            raise ValueError('missing successful submission receipt')
        job = int(receipt['job_id'])
        if job in jobs:
            raise ValueError('duplicate submission in campaign ledger')
        jobs[job] = str(path)
    historical = {int(read(Path(p) / 'submission.json')['job_id']) for p in config.get('historical_runs', [])}
    command = ['sacct', '-X', '-n', '-P', '-j', ','.join(map(str, sorted(jobs))), '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,Start,End,NodeList']
    proc = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = parse_accounting(proc.stdout, jobs, historical)
    previous_path = execution.parent / 'diagnosis_accounting_final_v2.json'
    previous = read(previous_path)
    if not math.isfinite(previous['combined_gpu_hours']) or previous['combined_gpu_hours'] < 0 or not math.isclose(previous['combined_gpu_hours'], config['prior_completed_gpu_hours'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('prior accounting differs from frozen prior total')
    if set(jobs) & {int(r['job_id']) for r in previous['jobs']}:
        raise ValueError('new campaign overlaps prior terminal accounting')
    accounting = {'schema': 'corrected-code128-terminal-accounting-20260922-v1', 'status': 'complete', 'created_at': now(),
                  'accounting_rule': 'Parent-job elapsed seconds times allocated GPU count, including all recorded unsuccessful attempts; generic and typed TRES describe the same GPU.',
                  'campaign_gpu_hours': sum(r['allocated_gpu_hours'] for r in rows),
                  'historical_attempt_gpu_hours': sum(r['allocated_gpu_hours'] for r in rows if r['historical_attempt']),
                  'prior_gpu_hours': previous['combined_gpu_hours'],
                  'combined_gpu_hours': previous['combined_gpu_hours'] + sum(r['allocated_gpu_hours'] for r in rows),
                  'prior_accounting': {'path': str(previous_path), 'sha256': sha(previous_path)},
                  'summary': {'path': str(summary_path), 'sha256': sha(summary_path)},
                  'campaign_config_sha256': sha(config_path), 'protocol_freeze_sha256': sha(execution / 'protocol_freeze.json'),
                  'jobs': rows, 'scheduler_query': command, 'scheduler_raw_output': proc.stdout}
    accounting_path = execution / 'terminal_accounting.json'
    write(accounting_path, accounting)
    renderer = ROOT / 'ops/render_corrected_code128_report_20260922.py'
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(PYTHON.parent.parent / 'lib')
    env['MPLCONFIGDIR'] = str(execution / 'report_matplotlib_cache')
    subprocess.run([str(PYTHON), str(renderer), '--summary', str(summary_path), '--output-dir', str(execution / 'report'), '--accounting', str(accounting_path), '--maxrl-training', str(Path(config['training_runs']['maxrl']) / 'training'), '--remax-training', str(Path(config['training_runs']['remax']) / 'training')], env=env, check=True)
    artifacts = [{'path': str(p), 'sha256': sha(p)} for p in sorted((execution / 'report').iterdir()) if p.is_file()]
    return {'schema': 'corrected-code128-finalization-20260922-v1', 'status': 'complete', 'updated_at': now(),
            'summary_sha256': sha(summary_path), 'accounting_path': str(accounting_path), 'accounting_sha256': sha(accounting_path),
            'renderer_sha256': sha(renderer), 'finalizer_sha256': sha(Path(__file__)), 'artifacts': artifacts,
            'scope': 'Derived diagnostic report only; no paper edit, inference, submission, checkpoint selection or protocol change.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    execution = args.execution.resolve()
    with (execution / 'finalization.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                failure = execution / 'campaign_failure.json'
                if failure.exists():
                    raise RuntimeError('campaign controller requires review; see campaign_failure.json')
                prior_result = execution / 'finalization_status.json'
                if prior_result.exists() and read(prior_result).get('status') == 'complete':
                    result = read(prior_result)
                    for item in result['artifacts'] + [{'path': result['accounting_path'], 'sha256': result['accounting_sha256']}]:
                        if sha(item['path']) != item['sha256']:
                            raise ValueError('completed report artifact changed')
                    if sha(execution / 'summary.json') != result['summary_sha256']:
                        raise ValueError('completed summary changed')
                    print(json.dumps(result), flush=True)
                    return
                status_path = execution / 'campaign_status.json'
                status = read(status_path).get('status') if status_path.exists() else 'not_started'
                if status == 'complete':
                    result = finish(execution)
                else:
                    controller = read(execution / 'controller_process.json')
                    proc_path = Path('/proc') / str(controller['pid']) / 'cmdline'
                    cmdline = proc_path.read_bytes().split(b'\0') if proc_path.exists() else []
                    if str(ROOT / 'ops/drive_corrected_code128_20260922.py').encode() not in cmdline or str(execution).encode() not in cmdline:
                        if status_path.exists() and read(status_path).get('status') == 'complete':
                            continue
                        raise RuntimeError('campaign controller is no longer running; review before resuming')
                    result = {'status': 'waiting_for_audited_completion', 'campaign_status': status, 'updated_at': now()}
                write(execution / 'finalization_status.json', result)
                if result['status'] == 'complete' or not args.watch:
                    print(json.dumps(result), flush=True)
                    return
            except Exception as exc:
                write(execution / 'finalization_status.json', {'status': 'needs_review', 'updated_at': now(), 'error': f'{type(exc).__name__}: {exc}'})
                raise
            time.sleep(45)


if __name__ == '__main__':
    main()
