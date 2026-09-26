#!/usr/bin/env python3
"""Cancel the E124 Level-2 and Level-3 cells that carry no progress.

Level 1 is the focus; L2/L3 were already held at priority 0 and could not start, so
this is queue hygiene rather than a resource decision.

Five Level-2 cells are deliberately NOT cancelled. They hold the 2,784 steps rescued on
2026-09-19, and those checkpoints live in directories named after the cells' current job
ids. Cancelling would mean a new job id on any later resubmission, a new debug_job
directory, and AUTO_RESUME unable to see the checkpoints -- re-creating the exact
stranding the rescue undid. They stay held.

Every cancellation is checked twice before it is sent: the job must be PENDING at
priority 0, and its run directory must contain no step_* checkpoint under any attempt.
A cell that has acquired progress since this script was written is skipped, not
cancelled.

Dry run by default. Pass --apply to act.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/n/fs/similarity/maxent-grpo')
ART = ROOT / 'var/artifacts/e124_qwen7b_three_level'

# Held because they carry rescued progress; see the module docstring.
KEEP = {
    'l2-countdown-replay_maxrl-s70': 864,
    'l2-pantry_plan-maxrl-s70': 576,
    'l2-pantry_plan-replay_maxrl-s70': 480,
    'l2-python_factors-maxrl-s70': 480,
    'l2-mathir-replay_maxrl-s70': 384,
}


def run(argv, check=True):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if check and p.returncode != 0:
        raise RuntimeError(f'{argv[0]} failed ({p.returncode}): {p.stderr.strip()}')
    return p.stdout


def fields(job_id):
    out = run(['scontrol', 'show', 'job', str(job_id)], check=False)
    return dict(t.split('=', 1) for t in out.split() if '=' in t) if out.strip() else {}


def checkpoints(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, 'debug_job*', 'checkpoints', 'step_*')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    if not a.apply:
        print('DRY RUN -- pass --apply to act')

    plan = json.loads((ART / 'plan.json').read_text())
    tx = json.loads((ART / 'transaction.json').read_text())
    cells = {c['cell_id']: c for c in plan['cells']}

    cancel, kept, skipped = [], [], []
    for cid, row in sorted(tx['rows'].items()):
        if not (cid.startswith('l2-') or cid.startswith('l3-')):
            continue
        jid = row['job_id']
        f = fields(jid)
        ck = checkpoints(cells[cid]['run_dir'])

        if cid in KEEP:
            kept.append((cid, jid, KEEP[cid]))
            continue
        if ck:
            skipped.append((cid, jid, 'HAS CHECKPOINTS ' + str([Path(c).name for c in ck])))
            continue
        if not f:
            skipped.append((cid, jid, 'not in scheduler'))
            continue
        if f.get('JobState') != 'PENDING' or f.get('Priority') != '0':
            skipped.append((cid, jid, f"state={f.get('JobState')} prio={f.get('Priority')}"))
            continue
        cancel.append((cid, jid))

    print(f'\n=== keeping {len(kept)} cells held (rescued progress) ===')
    for cid, jid, steps in kept:
        print(f'  {cid:<40} {jid}  {steps} steps')

    if skipped:
        print(f'\n=== skipped {len(skipped)} ===')
        for cid, jid, why in skipped:
            print(f'  {cid:<40} {jid}  {why}')

    print(f'\n=== cancelling {len(cancel)} empty cells ===')
    done = []
    for cid, jid in cancel:
        if not a.apply:
            print(f'  {cid:<40} {jid}  would cancel')
            continue
        run(['scancel', str(jid)])
        after = fields(jid)
        state = after.get('JobState', 'gone')
        print(f'  {cid:<40} {jid}  -> {state}')
        done.append({'cell': cid, 'job_id': jid})

    if a.apply and done:
        tx.setdefault('retirements', []).append({
            'at': datetime.now(timezone.utc).isoformat(),
            'schema': 'e124_level_retirement_v1',
            'cancelled': done,
            'why': 'operator refocused the campaign on Level 1; these cells carried no progress',
            'kept_held': [{'cell': c, 'job_id': j, 'steps': s} for c, j, s in kept],
            'why_kept': ('their rescued checkpoints live in debug_job<current job id>; cancelling '
                         'would force a new job id and a new directory on resubmission, stranding '
                         '2784 steps again'),
            'reversible': ('cancelled cells can be resubmitted from plan.json per-cell environment '
                           'with ops/exp_scaling/repair_e124_python_factors_20260921.py as the '
                           'pattern -- never from the stored command, which is stale'),
        })
        (ART / 'transaction.json').write_text(json.dumps(tx, indent=1, sort_keys=True) + '\n')
        print(f'\ntransaction.json updated: {len(done)} retirement(s) recorded')
    return 0


if __name__ == '__main__':
    sys.exit(main())
