#!/usr/bin/env python3
"""Collect the authorized frozen temperature conditions with hidden credentials.

Each child uses its original frozen native runner and saves every request,
response, receipt and grade. Preflights are the first registered sample of
each condition, and therefore count toward its 960-response budget.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/frontier_temperature_20260911'
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_chat_frontier_modebench import atomic, file_sha, now, verify_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'full'])
    parser.add_argument('--slug', action='append', choices=['grok43', 'kimi_k3'], required=True)
    args = parser.parse_args()
    slugs = list(dict.fromkeys(args.slug))
    runs = [BASE / (slug + '_t' + temperature) for slug in slugs for temperature in ['1p0', '1p5']]
    if args.stage == 'full':
        gate = json.loads((BASE / 'collection_gate.json').read_text())
        if gate.get('status') != 'authorized_supported_controls' or any(slug not in gate['admitted_slugs'] for slug in slugs):
            raise ValueError('Full collection requires a documented control-support gate')
        for run in runs:
            marker = json.loads((run / 'preflight_result.json').read_text())
            if marker.get('exit_code') != 0 or marker.get('terminal_samples') != 1:
                raise ValueError('Missing successful frozen first-sample preflight')
    for run in runs:
        verify_snapshot(run, json.loads((run / 'manifest.json').read_text()))
    key = os.environ.get('AZURE_OPENAI_API_KEY') or getpass.getpass('Azure API key (hidden): ')
    if not key:
        raise ValueError('Missing API credential')
    child_env = os.environ.copy()
    child_env['AZURE_OPENAI_API_KEY'] = key
    started = now()
    processes = []
    log_root = BASE / 'collection_logs' / (args.stage + '_' + started.replace(':', '-'))
    log_root.mkdir(parents=True)
    for run in runs:
        manifest = json.loads((run / 'manifest.json').read_text())
        marker_path = run / 'preflight_result.json'
        if args.stage == 'preflight' and marker_path.exists():
            marker = json.loads(marker_path.read_text())
            if marker.get('exit_code') == 0:
                print(json.dumps({'condition': run.name, 'status': 'existing_preflight_retained'}), flush=True)
                continue
            raise ValueError('A failed preflight needs explicit review before retrying')
        command = [sys.executable, str(run / 'code/ops/evaluate_chat_frontier_modebench.py'),
                   'run', '--model', manifest['model'], '--output', str(run),
                   '--profile', str(run / 'model_profile.json'),
                   '--workers', '1' if args.stage == 'preflight' else '64', '--request-timeout', '600']
        if args.stage == 'preflight':
            command += ['--max-new', '1', '--max-attempts', '2']
        if manifest['model'] == 'FW-Kimi-K3':
            command += ['--rpm', '45']  # Both temperature conditions total the original 90-RPM limit.
        log = log_root / (run.name + '.log')
        handle = log.open('w')
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT)
        processes.append((run, process, handle, log, command))
    results = []
    while processes:
        remaining = []
        for run, process, handle, log, command in processes:
            exit_code = process.poll()
            if exit_code is None:
                remaining.append((run, process, handle, log, command))
                continue
            handle.close()
            status = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
            result = {'stage': args.stage, 'condition': run.name, 'exit_code': exit_code,
                      'started_at_utc': started, 'completed_at_utc': now(),
                      'terminal_samples': status.get('completed_samples', 0),
                      'command': command, 'log': str(log.relative_to(BASE)), 'log_sha256': file_sha(log)}
            atomic(run / ('preflight_result.json' if args.stage == 'preflight' else 'collection_result.json'), result)
            results.append(result)
            print(json.dumps(result), flush=True)
        processes = remaining
        if processes:
            time.sleep(2)
    atomic(log_root / 'results.json', results)
    return int(any(result['exit_code'] != 0 for result in results))


if __name__ == '__main__':
    raise SystemExit(main())
