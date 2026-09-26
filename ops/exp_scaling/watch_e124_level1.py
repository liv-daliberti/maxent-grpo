#!/usr/bin/env python3
"""Level-1 progress board for the E124 Qwen-7B three-level sweep.

Ten cells: five domains x {maxrl, replay_maxrl}, seed 70, 3072 steps each.

Step counts come from the cell's *current* job directory only -- a checkpoint written
under a superseded job id cannot be resumed, so counting it would overstate progress.
Throughput is measured from consecutive eval_results mtimes, which land every
OAT_ZERO_EVAL_PROMPT_INTERVAL (192) steps, so it reflects real recent wall-clock rather
than an average dragged down by queue time or a stall.

  ./.venv/bin/python ops/exp_scaling/watch_e124_level1.py
  ./.venv/bin/python ops/exp_scaling/watch_e124_level1.py --watch 300
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

ROOT = '/n/fs/similarity/maxent-grpo'
ART = f'{ROOT}/var/artifacts/e124_qwen7b_three_level'
TARGET = 3072
WIDTH = 24

DOMAINS = ['countdown', 'graph_coloring', 'mathir', 'pantry_plan', 'python_factors']
ARMS = [('maxrl', 'MaxRL'), ('replay_maxrl', 'Re:Max')]


def scontrol(job_id):
    p = subprocess.run(['scontrol', 'show', 'job', str(job_id)],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0 or not p.stdout.strip():
        return {}
    return dict(t.split('=', 1) for t in p.stdout.split() if '=' in t)


def cell_state(run_dir, job_id):
    """Checkpoint, live step and recent rate from the current job's directory."""
    d = os.path.join(run_dir, f'debug_job{job_id}')
    ckpt = step = 0
    rate = None
    steps = [int(x.rsplit('_', 1)[1]) for x in glob.glob(os.path.join(d, 'checkpoints', 'step_*'))]
    if steps:
        ckpt = max(steps)
    metrics = os.path.join(d, 'train_metrics.jsonl')
    if os.path.exists(metrics):
        try:
            with open(metrics) as fh:
                last = None
                for last in fh:
                    pass
            if last:
                step = int(json.loads(last)['misc/global_step'])
        except Exception:
            pass
    evals = []
    for f in glob.glob(os.path.join(d, 'eval_results', '*_multi_answer.json')):
        try:
            evals.append((int(os.path.basename(f).split('_')[0]), os.path.getmtime(f)))
        except ValueError:
            continue
    evals.sort()
    if len(evals) >= 2:
        (s0, t0), (s1, t1) = evals[-2], evals[-1]
        if t1 > t0 and s1 > s0:
            rate = (s1 - s0) / ((t1 - t0) / 3600.0)
    return ckpt, step, rate


def bar(frac, width=WIDTH):
    full = int(round(frac * width))
    return '#' * full + '.' * (width - full)


def board():
    plan = json.load(open(f'{ART}/plan.json'))
    tx = json.load(open(f'{ART}/transaction.json'))
    cells = {c['cell_id']: c for c in plan['cells']}
    rows = tx['rows']

    print(f'E124 Level 1 -- Qwen-7B, seed 70, {TARGET} steps/cell      '
          f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    print(f'{"domain":<16}{"arm":<8}{"job":>9} {"state":<9}{"progress":<26}'
          f'{"step":>11}{"pct":>7}{"st/h":>7}  eta')
    print('-' * 104)

    done = live_rates = 0
    total = 0
    for domain in DOMAINS:
        for arm_key, arm_label in ARMS:
            cid = f'l1-{domain}-{arm_key}-s70'
            c, row = cells.get(cid), rows.get(cid, {})
            if not c:
                continue
            job = row.get('job_id')
            f = scontrol(job)
            ckpt, step, rate = cell_state(c['run_dir'], job)
            total += step

            state = f.get('JobState', 'GONE')
            reason = f.get('Reason', '')
            if state == 'PENDING':
                tag = 'queued' if reason == 'Resources' else 'waiting'
                if reason == 'JobHeldUser':
                    tag = 'HELD'
            elif state == 'RUNNING':
                tag = 'running'
            elif step >= TARGET:
                tag = 'DONE'
            else:
                tag = state.lower()[:9]

            if step >= TARGET:
                done += 1

            eta = ''
            if step >= TARGET:
                # A finished cell has nothing to resume. Its checkpoints are pruned on
                # success (PRUNE_RESUME_ON_SUCCESS), so ckpt reads 0, and a held one is
                # still PENDING -- the resume branch below would print "resumes from 0".
                eta = ''
            elif state == 'RUNNING' and rate:
                hrs = (TARGET - step) / rate
                eta = (datetime.now() + timedelta(hours=hrs)).strftime('%m-%d %H:%M')
                eta = f'{hrs:4.1f}h -> {eta}'
                live_rates += rate
            elif state == 'PENDING' and ckpt > 0:
                eta = f'resumes from {ckpt}'
            elif state == 'PENDING':
                eta = 'not started'

            print(f'{domain:<16}{arm_label:<8}{job:>9} {tag:<9}'
                  f'[{bar(min(step / TARGET, 1.0))}] {step:>6}/{TARGET}'
                  f'{100 * step / TARGET:>6.1f}%'
                  f'{(f"{rate:.0f}" if rate else "-"):>7}  {eta}')
        print()

    grand = TARGET * 10
    print('-' * 104)
    print(f'{"LEVEL 1":<16}{"":<8}{"":>9} {done}/10 done  '
          f'[{bar(total / grand, 40)}] {total:>6}/{grand}{100 * total / grand:>6.1f}%'
          f'   {live_rates:.0f} st/h aggregate')
    if live_rates:
        print(f'{"":<16}remaining {grand - total} steps; at the current aggregate rate '
              f'that is {(grand - total) / live_rates / 24:.1f} days if concurrency holds')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', type=int, metavar='SECONDS',
                    help='redraw every SECONDS (>=60; the scheduler rate-limits polling)')
    a = ap.parse_args()
    if a.watch:
        if a.watch < 60:
            ap.error('use >=60s: the controller enforces per-user RPC rate limiting')
        while True:
            os.system('clear')
            board()
            time.sleep(a.watch)
    return board()


if __name__ == '__main__':
    sys.exit(main())
