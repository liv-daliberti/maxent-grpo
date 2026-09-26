#!/usr/bin/env python3
"""Read-only live status of the five-domain Level 3 calibration campaign."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def status():
    artifact = ROOT / 'var/artifacts/modebench_level3_v1'
    jobs = []
    # New candidate revisions have their own immutable ledgers. Discover them
    # instead of silently omitting revisions added after this script was written.
    for path in sorted(set(artifact.glob('*jobs.json')) | {artifact / 'guided_interface_amendment.json'}):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            continue
        for row in payload.get('jobs', []):
            if isinstance(row, dict) and 'job_id' in row:
                jobs.append({'ledger': str(path.relative_to(ROOT)), **row})
    ids = sorted({str(job['job_id']) for job in jobs})
    scheduler = subprocess.run(['squeue', '-h', '-j', ','.join(ids), '-o', '%i|%T|%M|%R'], capture_output=True, text=True) if ids else None
    live = {}
    if scheduler and scheduler.returncode == 0:
        for line in scheduler.stdout.splitlines():
            job, state, elapsed, reason = line.split('|', 3)
            live[job] = {'state': state, 'elapsed': elapsed, 'node_or_reason': reason}
    pending_tasks = {}
    for job in jobs:
        tasks_path = Path(job.get('tasks', ''))
        if not tasks_path.is_file():
            continue
        for task in json.loads(tasks_path.read_text()):
            output = Path(task['output'])
            if output.exists() or str(output) in pending_tasks:
                continue
            batches = Path(str(output) + '.batches')
            manifest = batches / 'run.json'
            if not manifest.exists():
                continue
            identity = json.loads(manifest.read_text())['identity']
            pending_tasks[str(output)] = {
                'output': str(output), 'domain': task['domain'],
                'completed_batches': len(list(batches.glob('seed-*.json'))),
                'batch_size': identity['batch_size'],
                'seeds': identity['seeds'],
            }
    recipes = {}
    for directory in sorted(artifact.glob('recipes_v*')):
        if not directory.is_dir():
            continue
        cells = {}
        for path in sorted(directory.glob('*.json')):
            recipe = json.loads(path.read_text())
            if not isinstance(recipe, dict) or 'development' not in recipe:
                continue
            cells[recipe['domain']] = {
                'path': str(path.relative_to(ROOT)), 'decision': recipe['decision'],
                'selected_metrics': recipe['development']['selected_metrics'],
                'differences': recipe['development']['differences'],
                'gates': recipe['development'].get('gates'),
            }
        recipes[directory.name] = cells
    results = []
    for path in sorted((ROOT / 'var/results/modebench_level3_v1').glob('*.json')):
        row = json.loads(path.read_text())
        if row.get('status') == 'complete' and 'metrics' in row:
            results.append({'receipt': str(path.relative_to(ROOT)), 'model': row['model_label'],
                            'domain': row['domain'], 'level': row['level'], 'split': row['split'],
                            'interface': row['sampling']['name'],
                            'draws': len(row['sampling']['seeds']), 'metrics': row['metrics']})
    return {'observed_at': datetime.now(timezone.utc).isoformat(), 'jobs': [{**job, 'live': live.get(str(job['job_id']))} for job in jobs],
            'scheduler_read_error': scheduler.stderr if scheduler and scheduler.returncode else None,
            'receipts': results, 'pending_tasks': list(pending_tasks.values()),
            'recorded_recipes': recipes,
            'recipe_status_note': 'Recorded decisions only; run advance_modebench_level3.py to authenticate current source and receipt hashes.',
            'completion': 'unproven_until_five_domain_heldout_confirmation_passes'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    report = status()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report['observed_at'])
        for job in report['jobs']:
            if job['live']:
                print('JOB', job['job_id'], job['live']['state'], job['live']['elapsed'], job['live']['node_or_reason'])
        for task in report['pending_tasks']:
            print('SAVED BATCHES', Path(task['output']).name, task['completed_batches'])
        for version, recipes in report['recorded_recipes'].items():
            for domain, recipe in recipes.items():
                print('RECORDED RECIPE', version, domain, recipe['decision'])
        for row in report['receipts']:
            metrics = row['metrics']
            print('RESULT', Path(row['receipt']).name, 'rows', metrics['rows'], 'draws', row['draws'],
                  'pass1', f"{metrics['pass1']:.4f}", 'pass8', f"{metrics['pass8']:.4f}")
        if report['scheduler_read_error']:
            print('SCHEDULER READ ERROR', report['scheduler_read_error'])
        print(report['completion'])


if __name__ == '__main__':
    main()
