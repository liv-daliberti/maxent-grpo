#!/usr/bin/env python3
"""Snapshot the recovery implementation and submit its cells to Slurm.

The worker runs from an immutable copy of the code, so a later edit in the
working tree cannot change what a running or resumed cell does. Each cell is one
checkpoint and writes its own receipts, so resubmitting an index is a no-op once
that cell is complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from followup_metrics import atomic_new, file_sha  # noqa: E402

BASE = ROOT / 'artifacts/modebench_recovery_five_domain_20260917'
OPS = ('run_recovery_five_domain.py', 'portfolio_withdrawals.py', 'followup_metrics.py',
       'frontier_modebench_contract.py', 'prepare_recovery_five_domain.py',
       'make_pantry_plan_mode_data.py', 'make_modebench_data.py',
       'build_paper_decoding_objection.py', 'prepare_portfolio_withdrawals.py',
       'evaluate_modebench_level2_viability.py', 'repo_env.sh')
#: Every cell shares one GPU model, as the PantryPlan recovery experiment used
#: and as the retained draws that supply the ordinary portfolio were produced on.
RESOURCES = ('--partition=mltheory,lowprio', '--account=mltheory', '--gres=gpu:a5000:1',
             '--cpus-per-task=6', '--mem=40G', '--time=02:00:00')


def snapshot():
    code = BASE / 'code'
    files = [ROOT / 'ops' / name for name in OPS] + sorted((ROOT / 'src/oat_drgrpo').rglob('*.py'))
    for path in files:
        dest = code / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if file_sha(dest) != file_sha(path):
                raise ValueError('frozen code copy differs from the working tree: ' + str(path))
        else:
            shutil.copyfile(path, dest)
    inputs = json.loads((BASE / 'inputs.json').read_text())
    for name, digest in inputs['code_sha256'].items():
        if file_sha(code / name) != digest:
            raise ValueError('snapshot does not match the frozen protocol: ' + name)
    return code, inputs


def submit(array, dry_run):
    code, inputs = snapshot()
    worker = BASE / 'worker.slurm'
    text = ('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + str(ROOT) + '\n'
            'source ' + str(code / 'ops/repo_env.sh') + '\n'
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0\n'
            'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false '
            'VLLM_ATTENTION_BACKEND=XFORMERS\n'
            'exec ' + str(ROOT / 'var/seed_paper_eval/paper310/bin/python') + ' -B '
            + str(code / 'ops/run_recovery_five_domain.py')
            + ' --inputs ' + str(BASE / 'inputs.json') + ' --index "${SLURM_ARRAY_TASK_ID:?}"\n')
    if worker.exists():
        if worker.read_text() != text:
            raise ValueError('an existing worker script differs; inspect before resubmitting')
    else:
        worker.write_text(text)
    (BASE / 'logs').mkdir(exist_ok=True)
    command = ['sbatch', '--parsable', '--job-name=recovery-5domain', *RESOURCES,
               '--array=' + array, '--chdir=' + str(ROOT),
               '--output=' + str(BASE / 'logs/%A_%a.out'),
               '--error=' + str(BASE / 'logs/%A_%a.err'), str(worker)]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    record = {'command': command, 'array': array, 'cells': len(inputs['checkpoints']),
              'worker_sha256': file_sha(worker), 'inputs_sha256': file_sha(BASE / 'inputs.json'),
              'created_at_utc': datetime.now(timezone.utc).isoformat()}
    if dry_run:
        print(json.dumps({'dry_run': True, **record}))
        return
    atomic_new(BASE / f'submissions/intent_{stamp}.json', record)
    result = subprocess.run(command, text=True, capture_output=True)
    receipt = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
               'job_id': result.stdout.strip().split(';')[0] if result.returncode == 0 else None,
               **record}
    atomic_new(BASE / f'submissions/result_{stamp}.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('returncode', 'job_id', 'array', 'stderr')}))
    result.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array', required=True, help='Slurm array spec, e.g. 0-0 or 1-24%%3')
    parser.add_argument('--dry-run', action='store_true')
    submit(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
