#!/usr/bin/env python3
"""Inspect v2 calibration; optionally fit complete authenticated domain receipts.

Read-only by default. --fit-complete publishes one immutable fit per completed
five-receipt domain. Failed selected-development gates remain failed; this
program never changes weight grids, row-order seeds, pools, or GPU jobs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'ops', ROOT / 'ops/exp_scaling'):
    sys.path.insert(0, str(path))
from fit_modebench_level3_independent import fit_recipe
from evaluate_modebench_level3 import atomic_new, file_sha

CAMPAIGN = ROOT / 'var/artifacts/modebench_level3_v2'
SEAL_SHA = 'e03ffa74a476638401ddadb49b950f3d74377457114bf13b630ed8834be34736'
AMENDMENT_SHA = 'cd8168785e9d469d7d715123ff0120170e9091cd6ce7aad3d3c4ceb3aff93dd6'
RECOVERY_LEDGER = CAMPAIGN / 'regular_queue_recovery/development_jobs.json'
RECOVERY_LEDGER_SHA = 'bfcfc53886ecc0b44e52a8c5c8657a3e09d399d7b85f29bc5d6ed71c0516035d'


def inspect(fit_complete=False):
    spec = importlib.util.spec_from_file_location('v2_sealed_launch', CAMPAIGN / 'launch.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    seal = launcher.authenticate_saved_seal(SEAL_SHA)
    if file_sha(CAMPAIGN / 'confirmation_control_amendment.json') != AMENDMENT_SHA:
        raise ValueError('prospective confirmation amendment changed')
    ledger = json.loads((CAMPAIGN / 'development_jobs.json').read_text())
    plan = json.loads((CAMPAIGN / 'protocol.json').read_text())
    if ledger['implementation_seal_sha256'] != SEAL_SHA or ledger['confirmation_control_amendment_sha256'] != AMENDMENT_SHA:
        raise ValueError('submission ledger does not bind this sealed revision')
    if file_sha(RECOVERY_LEDGER) != RECOVERY_LEDGER_SHA:
        raise ValueError('registered execution recovery ledger changed')
    recovery = json.loads(RECOVERY_LEDGER.read_text())
    transfers = {job['old_job_id']: job for job in recovery['jobs']}
    if len(transfers) != 11 or recovery['status'] != 'all_eleven_submitted':
        raise ValueError('incomplete execution recovery ledger')
    results = {}
    for domain in ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry'):
        jobs = [job for job in ledger['jobs'] if job['domain'] == domain]
        if len(jobs) != 5:
            raise ValueError('one baseline and four candidates required per domain')
        cells = []
        for job in jobs:
            output = Path(job['output'])
            count = len(list(Path(str(output) + '.batches').glob('seed-*.json')))
            transferred = transfers.get(job['job_id'])
            if transferred and any(transferred[key] != job[key] for key in ('name', 'tasks', 'output', 'domain', 'model_label')):
                raise ValueError('recovery transfer differs from original scientific cell')
            cells.append({'name': job['name'],
                          'original_job_id': job['job_id'],
                          'job_id': transferred['new_job_id'] if transferred else job['job_id'],
                          'execution_recovery_ledger_sha256': RECOVERY_LEDGER_SHA if transferred else None,
                          'receipt_exists': output.is_file(), 'saved_batches': count,
                          'expected_batches': 32 if job['model_label'] == '05b' and domain == 'pantry' else 64,
                          'output': str(output)})
        record = {'cells': cells, 'status': 'sampling_pending_or_running'}
        if all(cell['receipt_exists'] for cell in cells):
            baseline = next(job['output'] for job in jobs if job['model_label'] == '05b')
            candidates = [next(job['output'] for job in jobs if job['difficulty'] == tier) for tier in range(4)]
            path = CAMPAIGN / 'recipes' / f'{domain}.json'
            if path.exists():
                recipe = json.loads(path.read_text())
                reproduced = fit_recipe(baseline, candidates, domain)
                if recipe != json.loads(json.dumps(reproduced, allow_nan=False)):
                    raise ValueError(f'{domain}: saved independent fit does not reproduce')
            elif fit_complete:
                recipe = fit_recipe(baseline, candidates, domain, path)
            else:
                recipe = None
            if recipe is None:
                record['status'] = 'complete_receipts_ready_to_fit'
            else:
                record.update(status=recipe['decision'], recipe=str(path),
                              recipe_sha256=file_sha(path), weights=recipe['weights'],
                              development=recipe['development'])
        results[domain] = record
    return {'schema': 'modebench_level3_independent_progress_v2',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'implementation_seal_sha256': SEAL_SHA, 'confirmation_amendment_sha256': AMENDMENT_SHA,
            'fit_complete_requested': fit_complete,
            'scope': 'original_candidate_ranges; accepted revised ranges are tracked separately',
            'execution_recovery_ledger': str(RECOVERY_LEDGER),
            'execution_recovery_ledger_sha256': RECOVERY_LEDGER_SHA,
            'all_development_fits_pass': all(record['status'] == 'development_fit_pass_pending_confirmation' for record in results.values()),
            'confirmation_match_verified': False, 'domains': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-complete', action='store_true')
    parser.add_argument('--output', type=Path, help='optional new immutable progress snapshot')
    args = parser.parse_args()
    result = inspect(args.fit_complete)
    if args.output:
        atomic_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
