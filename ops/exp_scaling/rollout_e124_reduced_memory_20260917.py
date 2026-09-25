#!/usr/bin/env python3
"""Resubmit every pending E124 cell with the measured host-memory request.

Cells asked for 256G on a 503G node, so one ran at a time while five a100s idled.
A cell resubmitted at 160G plateaued at ~107 GiB and survived checkpoints at steps
96 and 192 beside a 256G sibling, so the original figure was roughly 2.4x oversized
and was the only reason the node held a single cell. At 160G about three fit.

Only --mem changes. Scientific settings, entry point and placement are taken from
the plan unaltered, and each replacement is submitted held and its predecessor
cancelled only once the replacement exists, so a failure cannot strand a cell.

Running cells are never touched: a cell that is training keeps its allocation and
its progress, and can be reduced later when it next stops. The ledger is written as
well as the transaction, because campaign_stats reads the ledger.
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
CONTRACTS = {'graph_coloring': ('qwen_boxed', 'none'),
             'countdown': ('qwen_level2_countdown', 'countdown_legal_v3'),
             'python_factors': ('qwen_level2_python_factors', 'domain_legal_v1'),
             'mathir': ('qwen_level2_mathir', 'domain_legal_v1'),
             'pantry_plan': ('qwen_level2_pantry', 'domain_legal_v1')}


def require(condition, message):
    if not condition:
        raise SystemExit('refusing: ' + message)


def contract_ok(env):
    domain = env.get('OAT_ZERO_MODEBENCH_DOMAIN', 'none')
    syntax = env.get('OAT_ZERO_MODEBENCH_SYNTAX_PROFILE', 'none')
    if domain == 'none':
        return syntax == 'none'
    return (env.get('OAT_ZERO_PROMPT_TEMPLATE'), syntax) == CONTRACTS[domain]


def launcher():
    spec = importlib.util.spec_from_file_location(
        'e124_launcher', ROOT / 'ops/exp_scaling/launch_e124_qwen7b_three_level.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mem', default='160G')
    parser.add_argument('--limit', type=int, default=0, help='stop after this many cells (0 = all)')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    m = launcher()
    plan = json.loads((ART / 'plan.json').read_text())
    tx_path = ART / 'transaction.json'
    tx = json.loads(tx_path.read_text())
    cells = {c['cell_id']: c for c in plan['cells']}

    # Ask the scheduler what each job actually requests. The recorded command in
    # the transaction is the plan's, not the one submitted, so comparing against it
    # silently skipped every cell on a second pass.
    queue = {}
    for line in subprocess.run(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T|%m'],
                               capture_output=True, text=True, timeout=60).stdout.splitlines():
        jid, state, mem = line.split('|', 2)
        if jid.isdigit():
            queue[int(jid)] = (state, mem.strip())

    todo, skipped = [], []
    for cid, row in sorted(tx['rows'].items()):
        cell = cells.get(cid)
        jid = row.get('job_id')
        if cell is None or not jid:
            continue
        entry = queue.get(int(jid))
        state, current_mem = entry if entry else (None, None)
        if state != 'PENDING':
            skipped.append({'cell': cid, 'why': f'job is {state or "not queued"}'})
            continue
        if not contract_ok(cell['environment']):
            skipped.append({'cell': cid, 'why': 'environment violates the training contract'})
            continue
        # Skip only a cell already at *this* request. A further reduction is a
        # legitimate second pass: the first pass sized against an empty node, and
        # the node's real occupancy is only visible once cells are on it.
        if current_mem == args.mem:
            skipped.append({'cell': cid, 'why': f'already at {args.mem}'})
            continue
        todo.append((cid, int(jid), cell))
    if args.limit:
        todo = todo[:args.limit]

    print(json.dumps({'to_resubmit': len(todo), 'cells': [c for c, _, _ in todo],
                      'skipped': skipped}, indent=2))
    if not args.apply:
        print('\ndry run; pass --apply')
        return 0

    done, failed = [], []
    for cid, old, cell in todo:
        command = m.job_command(plan, cell)
        require(len([x for x in command if x.startswith('--mem=')]) == 1,
                f'{cid}: unexpected memory request in plan command')
        command = [f'--mem={args.mem}' if x.startswith('--mem=') else x for x in command]
        out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             timeout=180, env=m.clean_submit_environment())
        if out.returncode != 0:
            failed.append({'cell': cid, 'error': out.stderr.strip()[:200]})
            continue
        new = out.stdout.strip().split(';')[0]
        if not new.isdigit():
            failed.append({'cell': cid, 'error': f'ambiguous sbatch response {out.stdout!r}'})
            continue
        cancelled = subprocess.run(['scancel', str(old)], capture_output=True, text=True, timeout=60)
        require(cancelled.returncode == 0, f'{cid}: scancel {old} failed: {cancelled.stderr.strip()}')
        subprocess.run(['scontrol', 'release', str(new)], capture_output=True, text=True, timeout=60)
        tx['rows'][cid].update(status='released', job_id=int(new), previous_job_id=str(old),
                               memory_trial=f'--mem={args.mem}, measured ~107 GiB in use')
        done.append({'cell': cid, 'from': old, 'to': int(new)})
        print(json.dumps(done[-1]))

    if done:
        tx.setdefault('memory_trials', []).append({
            'at': datetime.now(timezone.utc).isoformat(), 'memory': args.mem,
            'resubmitted': done,
            'evidence': 'cell l1-graph_coloring-replay_maxrl-s70 plateaued at ~107 GiB and passed steps 96 and 192 at 160G',
            'unchanged': 'scientific settings, entry point and placement'})
        tx_path.write_text(json.dumps(tx, indent=2, sort_keys=True) + '\n')
        m.sync_ledger(plan, tx)
    print(json.dumps({'resubmitted': len(done), 'failed': failed}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
