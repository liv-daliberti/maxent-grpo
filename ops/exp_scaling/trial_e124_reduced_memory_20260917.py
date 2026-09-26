#!/usr/bin/env python3
"""Resubmit one E124 cell with a smaller host-memory request, as a measured trial.

Every cell asks for 256G on a 503G node, so exactly one runs at a time while five
a100s sit idle -- the request, not the hardware, is the throughput limit. The 256G
was never measured: the systems benchmark computes the figure that would justify it
(host_required = max(peak_nonreclaimable_dirty x 1.2, peak + 8 GiB)) and has never
passed. Meanwhile a running cell reports MaxRSS exactly equal to its own cap, which
is cgroup memory.current including page cache, inflated by ~114 GiB of file I/O --
memory touched, not memory needed.

This changes one cell and nothing else. If the smaller request holds through a
checkpoint cycle, several cells fit per node; if it is too small the cell is
OOM-killed and the watchdog requeues it from its last checkpoint, which is written
every 96 steps. Scientific settings are untouched: only --mem differs.
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
    parser.add_argument('--mem', default='160G')
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

    command = m.job_command(plan, cell)
    before = [x for x in command if x.startswith('--mem=')]
    require(before == ['--mem=256G'], f'unexpected memory request {before}')
    command = [f'--mem={args.mem}' if x.startswith('--mem=') else x for x in command]

    print(json.dumps({'cell': args.cell, 'job': old, 'memory': f'256G -> {args.mem}'}, indent=2))
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
               memory_trial=f'--mem={args.mem}, reduced from 256G pending a measured figure')
    tx.setdefault('memory_trials', []).append({
        'at': datetime.now(timezone.utc).isoformat(), 'cell': args.cell,
        'from_job': old, 'to_job': int(new), 'memory': args.mem,
        'why': '256G was never measured and is the binding limit on concurrency; scientific settings unchanged'})
    tx_path.write_text(json.dumps(tx, indent=2, sort_keys=True) + '\n')
    m.sync_ledger(plan, tx)  # campaign_stats reads the ledger, not the transaction
    print(json.dumps({'cell': args.cell, 'new_job': new, 'memory': args.mem}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
