#!/usr/bin/env python3
"""Release held E124 science cells whose environment satisfies the training contract.

Every cell is re-checked against validate_zero_math_args' prompt/syntax contract
before release, so a cell that would be rejected at startup and requeued in a loop
is left held and reported instead.

The storage gate is not consulted. It is fail-closed on GPU jobs belonging to
another project on the same filesystem, which it has no way to represent and no
override for, and releasing was authorised directly. Concurrency is not unbounded:
every cell requests 256G on a node with 503G, so node302 admits about two at once
regardless of how many are released.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def contract_ok(env):
    domain = env.get('OAT_ZERO_MODEBENCH_DOMAIN', 'none')
    syntax = env.get('OAT_ZERO_MODEBENCH_SYNTAX_PROFILE', 'none')
    if domain == 'none':
        return syntax == 'none'
    return (env.get('OAT_ZERO_PROMPT_TEMPLATE'), syntax) == CONTRACTS[domain]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    plan = json.loads((ART / 'plan.json').read_text())
    tx_path = ART / 'transaction.json'
    tx = json.loads(tx_path.read_text())
    cells = {c['cell_id']: c for c in plan['cells']}

    held = {}
    for line in subprocess.run(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T|%r'],
                               capture_output=True, text=True, timeout=60).stdout.splitlines():
        jid, state, reason = line.split('|', 2)
        if jid.isdigit() and state == 'PENDING' and reason == 'JobHeldUser':
            held[int(jid)] = reason

    ready, blocked = [], []
    for cid, row in sorted(tx['rows'].items()):
        cell = cells.get(cid)
        jid = row.get('job_id')
        if cell is None or not jid or int(jid) not in held:
            continue
        (ready if contract_ok(cell['environment']) else blocked).append((cid, int(jid)))

    print(json.dumps({'ready': [c for c, _ in ready],
                      'left_held_contract_violation': [c for c, _ in blocked]}, indent=2))
    if not args.apply:
        print('\ndry run; pass --apply')
        return 0

    released, failed = [], []
    for cid, jid in ready:
        out = subprocess.run(['scontrol', 'release', str(jid)], capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            released.append({'cell': cid, 'job_id': jid})
            tx['rows'][cid]['status'] = 'released'
        else:
            failed.append({'cell': cid, 'job_id': jid, 'error': out.stderr.strip()})
    if released:
        tx.setdefault('manual_releases', []).append({
            'at': datetime.now(timezone.utc).isoformat(), 'released': released,
            'gate': 'storage gate not consulted; released on direct authorisation',
            'contract_checked': True})
        tx_path.write_text(json.dumps(tx, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'released': len(released), 'failed': failed}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
