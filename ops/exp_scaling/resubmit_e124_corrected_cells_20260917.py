#!/usr/bin/env python3
"""Resubmit the E124 cells whose environment was corrected, then release them.

repair_e124_level1_domain_20260917.py cleared the ModeBench domain on eight
Level-1 cells that the training entry point rejected at startup. Their queued jobs
still carry the rejected environment, and Slurm cannot rewrite a submitted job's
exports, so each is resubmitted from the corrected plan and its predecessor
cancelled.

The replacement is submitted held and verified to satisfy the prompt/syntax
contract before the predecessor is cancelled, so a bad correction cannot strand a
cell. Released afterwards only if --release is given.
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
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()

    m = launcher()
    plan = json.loads((ART / 'plan.json').read_text())
    tx_path = ART / 'transaction.json'
    tx = json.loads(tx_path.read_text())
    cells = {c['cell_id']: c for c in plan['cells']}

    queued = {}
    for line in subprocess.run(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T'],
                               capture_output=True, text=True, timeout=60).stdout.splitlines():
        jid, _, state = line.partition('|')
        if jid.isdigit():
            queued[int(jid)] = state

    todo = []
    for cid, row in sorted(tx['rows'].items()):
        if row.get('status') != 'needs_resubmission' or cid not in cells:
            continue
        cell = cells[cid]
        require(contract_ok(cell['environment']), f'{cid}: corrected environment still violates the contract')
        jid = row.get('job_id')
        require(jid and queued.get(int(jid)) == 'PENDING', f'{cid}: predecessor {jid} is not pending; resolve by hand')
        todo.append((cid, int(jid), cell))

    print(json.dumps({'to_resubmit': [c for c, _, _ in todo]}, indent=2))
    if not args.apply:
        print('\ndry run; pass --apply')
        return 0

    done = []
    for cid, old, cell in todo:
        out = subprocess.run(m.job_command(plan, cell), cwd=ROOT, capture_output=True,
                             text=True, timeout=180, env=m.clean_submit_environment())
        require(out.returncode == 0, f'{cid}: sbatch failed: {out.stderr.strip()}')
        new = out.stdout.strip().split(';')[0]
        require(new.isdigit(), f'{cid}: ambiguous sbatch response {out.stdout!r}')
        cancelled = subprocess.run(['scancel', str(old)], capture_output=True, text=True, timeout=60)
        require(cancelled.returncode == 0, f'{cid}: scancel {old} failed: {cancelled.stderr.strip()}')
        tx['rows'][cid].update(status='held', job_id=int(new), previous_job_id=str(old))
        if args.release:
            released = subprocess.run(['scontrol', 'release', str(new)], capture_output=True, text=True, timeout=60)
            if released.returncode == 0:
                tx['rows'][cid]['status'] = 'released'
        done.append({'cell': cid, 'from': old, 'to': int(new)})
        print(json.dumps(done[-1]))

    tx.setdefault('corrections', []).append({
        'at': datetime.now(timezone.utc).isoformat(),
        'reason': 'Level-1 ModeBench domain cleared; predecessors carried the rejected environment',
        'resubmitted': done, 'released': bool(args.release)})
    tx_path.write_text(json.dumps(tx, indent=2, sort_keys=True) + '\n')
    m.sync_ledger(plan, tx)  # campaign_stats reads the ledger, not the transaction
    print(json.dumps({'resubmitted': len(done)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
