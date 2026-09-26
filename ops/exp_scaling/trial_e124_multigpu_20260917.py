#!/usr/bin/env python3
"""Run one E124 cell across several GPUs, with the optimizer kept on-device.

E124 pins OAT_ZERO_N_GPU=1, which forces OAT_ZERO_ADAM_OFFLOAD=1: a 7B policy plus
fp32 Adam state is ~112 GB and does not fit one 80 GB card, so the whole optimizer
state (~84 GB) sits in host RAM. That is why a cell asks for 256G, and why exactly
one runs on a 503G node while four a100s idle. Sharding the optimizer over four
GPUs under ZeRO-2 puts it back on-device (~21 GB each), frees the host, and gives
roughly four times the compute per cell.

The recipe sanctions this: OAT_ZERO_N_GPU, NUM_GPUS_PER_ACTOR, ADAM_OFFLOAD,
ACTIVATION_OFFLOADING, ZERO_STAGE, VLLM_GPU_RATIO and TRAIN_BATCH_SIZE_PER_DEVICE
are all in PHYSICAL_ENV_KEYS. TRAIN_BATCH_SIZE and ROLLOUT_BATCH_SIZE are not --
they are scientific and untouched here, so the global batch and the update math are
unchanged.

This edits nothing in the plan. It rewrites one job's submitted export list, so the
trial is entirely contained in a single job and the frozen plan still describes the
sweep. Roll it into the plan only once a cell has crossed the evaluation boundary
at step 192, which is where the reduced-memory single-GPU attempt froze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(os.environ.get('OAT_ZERO_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
ART = ROOT / 'var/artifacts/e124_qwen7b_three_level'
PHYSICAL = {'OAT_ZERO_N_GPU', 'OAT_ZERO_NUM_GPUS_PER_ACTOR', 'OAT_ZERO_ADAM_OFFLOAD',
            'OAT_ZERO_ACTIVATION_OFFLOADING', 'OAT_ZERO_ZERO_STAGE', 'OAT_ZERO_VLLM_GPU_RATIO',
            'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS'}


def require(condition, message):
    if not condition:
        raise SystemExit('refusing: ' + message)


def launcher():
    spec = importlib.util.spec_from_file_location(
        'e124_launcher', ROOT / 'ops/exp_scaling/launch_e124_qwen7b_three_level.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--mem', default='96G')
    parser.add_argument('--cpus', type=int, default=32)
    parser.add_argument('--vllm-ratio', default='0.25')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    m = launcher()
    plan = json.loads((ART / 'plan.json').read_text())
    tx_path = ART / 'transaction.json'
    tx = json.loads(tx_path.read_text())
    cell = next((c for c in plan['cells'] if c['cell_id'] == args.cell), None)
    require(cell is not None, f'unknown cell {args.cell}')
    row = tx['rows'].get(args.cell)
    require(row and row.get('job_id'), f'{args.cell} has no recorded job')
    old = int(row['job_id'])
    state = subprocess.run(['squeue', '-h', '-j', str(old), '-o', '%T'],
                           capture_output=True, text=True, timeout=60).stdout.strip()
    require(state == 'PENDING', f'{args.cell}: job {old} is {state or "gone"}; only a pending cell is retried')

    overrides = {'OAT_ZERO_N_GPU': str(args.gpus),
                 'OAT_ZERO_NUM_GPUS_PER_ACTOR': str(args.gpus),
                 'OAT_ZERO_ADAM_OFFLOAD': '0',
                 'OAT_ZERO_ACTIVATION_OFFLOADING': '0',
                 'OAT_ZERO_VLLM_GPU_RATIO': args.vllm_ratio}
    require(set(overrides) <= PHYSICAL, 'refusing to change a non-physical setting')

    command = []
    for token in m.job_command(plan, cell):
        if token.startswith('--export=ALL,'):
            items = token[len('--export=ALL,'):].split(',')
            seen = set()
            out = []
            for item in items:
                k, _, v = item.partition('=')
                if k in overrides:
                    v = overrides[k]; seen.add(k)
                out.append(f'{k}={v}')
            require(seen == set(overrides), f'plan lacks expected physical keys: {set(overrides) - seen}')
            token = '--export=ALL,' + ','.join(out)
        elif token.startswith('--gres='):
            token = f'--gres=gpu:a100:{args.gpus}'
        elif token.startswith('--mem='):
            token = f'--mem={args.mem}'
        elif token.startswith('--cpus-per-task='):
            token = f'--cpus-per-task={args.cpus}'
        command.append(token)

    print(json.dumps({'cell': args.cell, 'job': old, 'gpus': args.gpus,
                      'memory': args.mem, 'cpus': args.cpus, 'overrides': overrides}, indent=2))
    if not args.apply:
        print('\ndry run; pass --apply')
        return 0

    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                         timeout=180, env=m.clean_submit_environment())
    require(out.returncode == 0, 'sbatch failed: ' + out.stderr.strip())
    new = out.stdout.strip().split(';')[0]
    require(new.isdigit(), f'ambiguous sbatch response {out.stdout!r}')
    cancelled = subprocess.run(['scancel', str(old)], capture_output=True, text=True, timeout=60)
    require(cancelled.returncode == 0, f'scancel {old} failed: {cancelled.stderr.strip()}')
    subprocess.run(['scontrol', 'release', str(new)], capture_output=True, text=True, timeout=60)

    row.update(status='released', job_id=int(new), previous_job_id=str(old),
               physical_trial=f'{args.gpus} GPUs, optimizer on-device, --mem={args.mem}')
    tx.setdefault('physical_trials', []).append({
        'at': datetime.now(timezone.utc).isoformat(), 'cell': args.cell,
        'from_job': old, 'to_job': int(new), 'gpus': args.gpus, 'memory': args.mem,
        'overrides': overrides, 'scientific_settings_changed': False,
        'why': 'single-GPU forced the optimizer into host RAM; sharding it over GPUs frees the host and multiplies compute'})
    tx_path.write_text(json.dumps(tx, indent=2, sort_keys=True) + '\n')
    m.sync_ledger(plan, tx)
    print(json.dumps({'cell': args.cell, 'new_job': new, 'gpus': args.gpus, 'memory': args.mem}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
