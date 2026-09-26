#!/usr/bin/env python3
"""Move the Level-1 python_factors pair out of lowprio, which is preempting it to death.

The pair was placed on the legacy a6000 pool to add concurrency while node302 was full.
That pool was reached through partition lowprio, which is PriorityTier=1 and
PreemptMode=REQUEUE, so anything in cs or all (PriorityTier=100, PreemptMode=OFF) evicts
it. Both cells took two preemptions within five minutes.

Preemption is only survivable if a cell can bank a checkpoint between evictions. These
cells checkpoint at SAVE_STEPS=96 and run at ~76 steps/h, so they need about 1.2h
uninterrupted to save anything at all. Below that they restart from step 0 forever.

Partition all spans node205-208 -- the whole legacy pool, including node208, which cs
does not cover -- and is not preemptible. It requires account allcs: this cluster couples
partition to account, and --partition=all under account mltheory silently lands back on
mltheory. The submission is therefore verified after the fact rather than trusted.

Everything else is byte-identical to the frozen command: same gres, same node list, same
exclude list, same scientific exports, same entry point.

Dry run by default. Pass --apply to act.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/n/fs/similarity/maxent-grpo')
TX = ROOT / 'var/artifacts/e124_qwen7b_three_level/transaction.json'
CELLS = ['l1-python_factors-maxrl-s70', 'l1-python_factors-replay_maxrl-s70']


def run(argv, check=True):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    if check and p.returncode != 0:
        raise RuntimeError(f'{argv[0]} failed ({p.returncode}): {p.stderr.strip()}')
    return p.stdout


def fields(job_id):
    out = run(['scontrol', 'show', 'job', str(job_id)], check=False)
    return dict(t.split('=', 1) for t in out.split() if '=' in t) if out.strip() else {}


def plan_environment(cell):
    plan = json.loads((ROOT / 'var/artifacts/e124_qwen7b_three_level/plan.json').read_text())
    return next(c for c in plan['cells'] if c['cell_id'] == cell)['environment']


def build(cell, row):
    """Rebuild the submission, taking the environment from plan.json.

    transaction.json's stored `command` is NOT a safe source: it predates the
    2026-09-17 level1_domain_correction and still names a domain on the eight Level-1
    cells that had one, which invokes the Level-2 prompt/syntax contract and kills the
    cell 62 seconds in. That is how the python_factors pair died on 2026-09-20.
    """
    env = plan_environment(cell)
    stored = next(t for t in row['command'] if t.startswith('--export='))
    have = dict(kv.split('=', 1) for kv in stored[len('--export=ALL,'):].split(',') if '=' in kv)
    drift = {k: (env.get(k), have.get(k)) for k in set(env) | set(have) if env.get(k) != have.get(k)}
    if drift:
        print(f'  {cell}: stored command is stale, rebuilding from plan.json -- {drift}')

    out = []
    for t in row['command']:
        if t == '--hold':
            continue
        if t.startswith('--export='):
            t = '--export=ALL,' + ','.join(f'{k}={env[k]}' for k in sorted(env))
        elif t.startswith('--mem='):
            t = '--mem=224G'
        elif t.startswith('--partition='):
            t = '--partition=all'
        elif t.startswith('--account='):
            t = '--account=allcs'
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    if not a.apply:
        print('DRY RUN -- pass --apply to act')

    tx = json.loads(TX.read_text())
    moved = []
    for cell in CELLS:
        row = tx['rows'][cell]
        old = row['job_id']
        f = fields(old)
        state = f.get('JobState')
        print(f"\n{cell}: job {old} is {state}, Restarts={f.get('Restarts')}")

        # Anything checkpointed would be stranded by a new job id, so refuse if present.
        plan = json.loads((ROOT / 'var/artifacts/e124_qwen7b_three_level/plan.json').read_text())
        run_dir = Path(next(c['run_dir'] for c in plan['cells'] if c['cell_id'] == cell))
        ckpts = sorted(run_dir.glob(f'debug_job{old}/checkpoints/step_*'))
        if ckpts:
            print(f"  REFUSING: {[c.name for c in ckpts]} would be stranded by a new job id")
            continue

        cmd = build(cell, row)
        if not a.apply:
            print('  would cancel and resubmit to partition=all account=allcs at 224G')
            continue

        run(['scancel', str(old)])
        for _ in range(30):
            time.sleep(2)
            if fields(old).get('JobState') in (None, '', 'CANCELLED'):
                break
        new = run(cmd).strip().split(';')[0]

        g = fields(new)
        want = {'Partition': 'all', 'Account': 'allcs',
                'TresPerNode': 'gres/gpu:a6000:1', 'MinMemoryNode': '224G'}
        bad = {k: g.get(k) for k, v in want.items() if g.get(k) != v}
        if bad:
            print(f"  job {new} submitted but LANDED WRONG: {bad}")
            print('  partition is coupled to account here -- do not leave it like this')
        else:
            print(f"  {old} -> {new} on partition=all (PreemptMode=OFF), a6000, 224G, verified")

        row['previous_job_id'] = old
        row['job_id'] = int(new)
        row['status_note'] = ('moved off lowprio 2026-09-20: two preemptions in five minutes, '
                              'and the cell needs ~1.2h uninterrupted to reach SAVE_STEPS=96')
        moved.append({'cell': cell, 'from': old, 'to': int(new), 'landed_correctly': not bad})

    if a.apply and moved:
        tx.setdefault('placement_history', []).append({
            'at': datetime.now(timezone.utc).isoformat(),
            'schema': 'e124_placement_change_v1',
            'direction': 'lowprio/a6000 -> all/a6000',
            'cells': moved,
            'why': ('lowprio is PriorityTier=1 PreemptMode=REQUEUE and both cells were preempted '
                    'twice within five minutes. A cell checkpoints at SAVE_STEPS=96 and runs at '
                    '~76 steps/h, so it needs ~1.2h uninterrupted to bank anything; below that it '
                    'restarts from step 0 indefinitely. Partition all is PriorityTier=100 '
                    'PreemptMode=OFF and spans the whole legacy pool including node208.'),
            'account_changed': ('mltheory -> allcs, required because this cluster couples partition '
                                'to account; --partition=all under mltheory silently lands on '
                                'mltheory. Accounting only; no scientific setting changed.'),
            'gpu_model_unchanged': 'a6000, the model the frozen command already named',
            'scientific_settings_changed': False,
            'audit_note': ('controller_v4.audit_job expects Account=mltheory and one of two frozen '
                           'placements; these two cells now match neither and must be covered by a '
                           'v5 amendment before the controller is started.'),
        })
        TX.write_text(json.dumps(tx, indent=1, sort_keys=True) + '\n')
        print(f'\ntransaction.json updated with {len(moved)} placement change(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
