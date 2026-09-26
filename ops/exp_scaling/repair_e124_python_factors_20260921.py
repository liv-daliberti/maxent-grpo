#!/usr/bin/env python3
"""Resubmit the Level-1 python_factors pair, which failed on a stale environment.

Both cells died 62 seconds in on 2026-09-20:

    ValueError: Level-2 prompt/syntax contract mismatch:
      domain=python_factors requires (qwen_level2_python_factors, domain_legal_v1)

They were resubmitted from the `command` stored in transaction.json. That command
predates the 2026-09-17 level1_domain_correction, which set OAT_ZERO_MODEBENCH_DOMAIN
to 'none' on the eight Level-1 cells that named a domain -- naming one invokes the
Level-2 prompt/syntax contract, which Level 1 does not satisfy. The correction lives in
plan.json's per-cell `environment`; nothing rewrote the transaction's command.

Diffing the two for l1-python_factors-maxrl-s70: 147 keys each, and exactly one differs.

    OAT_ZERO_MODEBENCH_DOMAIN   plan='none'   stored command='python_factors'

So this script builds the export block from plan.json's environment, which is the
authority, and takes every placement flag verbatim from a live sibling Level-1 job
rather than reconstructing it -- that keeps partition, account, nodelist, exclude list,
gres, memory, time limit and entry point exactly as the rest of Level 1 has them,
including the current plan hash in --comment.

Cells are submitted HELD. mathir Re:Max is already queued for the next free slot on
node302 and these must not race it; release them when you want them.

Dry run by default. Pass --apply to act.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/n/fs/similarity/maxent-grpo')
ART = ROOT / 'var/artifacts/e124_qwen7b_three_level'
PLAN, TX = ART / 'plan.json', ART / 'transaction.json'

CELLS = ['l1-python_factors-maxrl-s70', 'l1-python_factors-replay_maxrl-s70']
TEMPLATE_JOB = 31343992          # l1-mathir-replay_maxrl-s70: same level, correct placement

# Flags copied from the template; everything else is rebuilt per cell.
PLACEMENT_FLAGS = ('--partition=', '--account=', '--nodelist=', '--exclude=', '--gres=',
                   '--mem=', '--cpus-per-task=', '--nodes=', '--ntasks=', '--time=',
                   '--nice=', '--chdir=', '--output=', '--error=')


def run(argv, check=True):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    if check and p.returncode != 0:
        raise RuntimeError(f'{argv[0]} failed ({p.returncode}): {p.stderr.strip()}')
    return p.stdout


def fields(job_id):
    out = run(['scontrol', 'show', 'job', str(job_id)], check=False)
    return dict(t.split('=', 1) for t in out.split() if '=' in t) if out.strip() else {}


def template():
    out = run(['scontrol', 'show', 'job', str(TEMPLATE_JOB), '-d'])
    m = re.search(r'SubmitLine=(.*?)\s+WorkDir=', out, re.S)
    if not m:
        raise RuntimeError(f'could not read SubmitLine of template job {TEMPLATE_JOB}')
    toks = shlex.split(m.group(1))
    flags = [t for t in toks if t.startswith(PLACEMENT_FLAGS)]
    entry = toks[-1]
    if not entry.endswith('.slurm'):
        raise RuntimeError(f'template entry point looks wrong: {entry}')
    if '--requeue' not in toks:
        raise RuntimeError('template lost --requeue; refusing to submit without it')
    return flags, entry


def build(cell, plan, flags, entry):
    c = next(x for x in plan['cells'] if x['cell_id'] == cell)
    env = c['environment']
    if env.get('OAT_ZERO_MODEBENCH_DOMAIN') != 'none':
        raise RuntimeError(f'{cell}: plan environment still names a domain -- '
                           f'{env.get("OAT_ZERO_MODEBENCH_DOMAIN")!r}; the correction is missing')
    export = 'ALL,' + ','.join(f'{k}={env[k]}' for k in sorted(env))
    name = {'l1-python_factors-maxrl-s70': 'e124-l1-python-m-s70',
            'l1-python_factors-replay_maxrl-s70': 'e124-l1-python-rm-s70'}[cell]
    comment = f"e124:{plan['plan_sha256'][:16]}:{cell}"
    return (['sbatch', '--parsable', '--hold', f'--job-name={name}', f'--export={export}']
            + flags + ['--requeue', f'--comment={comment}', entry]), env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    if not a.apply:
        print('DRY RUN -- pass --apply to act')

    plan = json.loads(PLAN.read_text())
    tx = json.loads(TX.read_text())
    flags, entry = template()
    print(f'\ntemplate job {TEMPLATE_JOB}: ' + ' '.join(f for f in flags if f.startswith(
        ('--partition', '--account', '--nodelist', '--gres', '--mem', '--time'))))
    print(f'entry point: {entry}')

    submitted = []
    for cell in CELLS:
        row = tx['rows'][cell]
        old = row['job_id']
        f = fields(old)
        state = f.get('JobState')
        if state in ('RUNNING', 'PENDING', 'COMPLETING'):
            print(f'\n{cell}: job {old} is {state} -- refusing to submit a duplicate')
            continue
        cmd, env = build(cell, plan, flags, entry)
        print(f'\n{cell}')
        print(f'  previous job {old}: {state or "not in scheduler"} '
              f'({run(["sacct","-j",str(old),"-X","-n","-o","State"],check=False).split()[:1]})')
        print(f'  MODEBENCH_DOMAIN={env["OAT_ZERO_MODEBENCH_DOMAIN"]}  ({len(env)} exports)')
        if not a.apply:
            print('  would submit HELD to ' + next(x for x in flags if x.startswith('--partition=')))
            continue

        new = run(cmd).strip().split(';')[0]
        g = fields(new)
        want = {'Partition': 'mltheory', 'Account': 'mltheory',
                'TresPerNode': 'gres/gpu:a100:1', 'MinMemoryNode': '224G',
                'JobState': 'PENDING', 'Reason': 'JobHeldUser'}
        bad = {k: g.get(k) for k, v in want.items() if g.get(k) != v}
        # Re-read the domain straight back out of the scheduler, not out of our own dict.
        d = run(['scontrol', 'show', 'job', str(new), '-d'], check=False)
        dom = re.search(r'OAT_ZERO_MODEBENCH_DOMAIN=([^,\s]*)', d)
        if bad or not dom or dom.group(1) != 'none':
            print(f'  job {new} SUBMITTED BUT WRONG: {bad} domain={dom.group(1) if dom else "?"}')
        else:
            print(f'  {old} -> {new}, held, mltheory/a100/224G, MODEBENCH_DOMAIN=none, verified')

        row['previous_job_id'] = old
        row['job_id'] = int(new)
        row['status'] = 'held'
        row['status_note'] = ('resubmitted 2026-09-21 from plan.json environment after the '
                              'previous pair failed on the pre-correction domain value')
        submitted.append({'cell': cell, 'from': old, 'to': int(new), 'verified': not bad})

    if a.apply and submitted:
        tx.setdefault('corrections', []).append({
            'at': datetime.now(timezone.utc).isoformat(),
            'schema': 'e124_cell_resubmission_v1',
            'cells': submitted,
            'why': ('the 2026-09-20 resubmission of this pair was built from the command stored in '
                    'transaction.json, which predates the 2026-09-17 level1_domain_correction. It '
                    'carried OAT_ZERO_MODEBENCH_DOMAIN=python_factors, which invokes the Level-2 '
                    'prompt/syntax contract on a Level-1 cell, and both jobs raised ValueError 62 '
                    'seconds in.'),
            'source_of_truth': ('plan.json per-cell environment; the stored command is stale and must '
                                'not be used to rebuild a submission. The two differ in exactly one of '
                                '147 keys.'),
            'placement': f'copied verbatim from live sibling job {TEMPLATE_JOB}',
            'submitted_held': 'so they cannot race mathir Re:Max for the next node302 slot',
            'scientific_settings_changed': False,
        })
        TX.write_text(json.dumps(tx, indent=1, sort_keys=True) + '\n')
        print(f'\ntransaction.json updated with {len(submitted)} resubmission(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
