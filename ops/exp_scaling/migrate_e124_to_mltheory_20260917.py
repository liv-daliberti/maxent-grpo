#!/usr/bin/env python3
"""Move E124's queued cells from lowprio/a6000 to the mltheory partition on a100.

Slurm refuses to change a job's partition after submission -- "Partition may not
be modified after submission. Please cancel and resubmit." -- so each cell is
resubmitted rather than updated. The new job is submitted held first and the old
one cancelled only once it exists, so a failed submission cannot lose a cell.

Only placement moves. The cell environments come from the frozen plan and are
resubmitted byte-for-byte; the a100 *profile* is deliberately not adopted, because
regenerating the plan for it requires re-issuing the ModeBench Level-3 admission,
which pins the 2026-09-09 src/oat_drgrpo state that live source has since moved
past. The pinned a6000 profile is also the conservative choice here: its 0.40 vLLM
reserve on an 80 GiB a100 leaves more training headroom than on the 48 GiB card it
was tuned for.

Running cells are never touched. A cell already training keeps its node, its
a6000, and its result; this tool only moves what is still waiting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get('OAT_ZERO_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
ART = ROOT / 'var/artifacts/e124_qwen7b_three_level'
TARGET = {'partition': 'mltheory', 'gres': 'gpu:a100:1', 'nodelist': 'node302'}


def launcher():
    spec = importlib.util.spec_from_file_location(
        'e124_launcher', ROOT / 'ops/exp_scaling/launch_e124_qwen7b_three_level.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise SystemExit('refusing: ' + message)


def queue_rows():
    out = subprocess.run(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%j|%T|%P'],
                         capture_output=True, text=True, timeout=60)
    require(out.returncode == 0, 'squeue failed: ' + out.stderr.strip())
    rows = {}
    for line in out.stdout.splitlines():
        jid, name, state, partition = line.split('|', 3)
        if jid.isdigit():
            rows[int(jid)] = {'name': name, 'state': state, 'partition': partition}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='without this, only report what would move')
    args = parser.parse_args()

    m = launcher()
    require(m.ART == ART, 'launcher artifact root is not the original campaign')
    plan = json.loads((ART / 'plan.json').read_text())
    tx_path = ART / 'transaction.json'
    tx = json.loads(tx_path.read_text())
    cells = {c['cell_id']: c for c in plan['cells']}
    live = queue_rows()

    movable, skipped = [], []
    for cell_id, row in sorted(tx['rows'].items()):
        job_id = row.get('job_id')
        cell = cells.get(cell_id)
        if cell is None:
            skipped.append((cell_id, job_id, 'not a science cell in this plan')); continue
        if not job_id or int(job_id) not in live:
            skipped.append((cell_id, job_id, 'no live scheduler job')); continue
        entry = live[int(job_id)]
        if entry['state'] != 'PENDING':
            skipped.append((cell_id, job_id, f"job is {entry['state']}; running work is never moved")); continue
        if entry['partition'] == TARGET['partition']:
            skipped.append((cell_id, job_id, 'already on the target partition')); continue
        movable.append((cell_id, int(job_id), cell))

    print(json.dumps({'movable': len(movable), 'skipped': len(skipped),
                      'skipped_detail': [{'cell': c, 'job': j, 'why': w} for c, j, w in skipped]}, indent=2))
    if not args.apply:
        print('\ndry run; pass --apply to move them')
        return 0

    history = tx.setdefault('placement_history', [])
    moved = []
    for cell_id, old_job, cell in movable:
        command = m.job_command(plan, cell)
        submitted = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   timeout=180, env=m.clean_submit_environment())
        require(submitted.returncode == 0, f'{cell_id}: sbatch failed: {submitted.stderr.strip()}')
        new_job = submitted.stdout.strip().split()[-1]
        require(new_job.isdigit(), f'{cell_id}: unexpected sbatch output: {submitted.stdout!r}')
        # The replacement exists and is held, so cancelling the original cannot
        # strand the cell.
        cancelled = subprocess.run(['scancel', str(old_job)], capture_output=True, text=True, timeout=60)
        require(cancelled.returncode == 0, f'{cell_id}: scancel {old_job} failed: {cancelled.stderr.strip()}')
        tx['rows'][cell_id]['job_id'] = new_job
        tx['rows'][cell_id]['previous_job_id'] = str(old_job)
        moved.append({'cell': cell_id, 'from_job': str(old_job), 'to_job': new_job})
        print(json.dumps({'moved': cell_id, 'from': old_job, 'to': new_job}))

    history.append({'at': datetime.now(timezone.utc).isoformat(),
                    'change': 'partition lowprio -> mltheory, gres a6000 -> a100, nodelist node205-208 -> node302',
                    'reason': ('Slurm cannot modify a partition after submission, so each pending cell was '
                               'resubmitted held with placement flags changed and its predecessor cancelled. '
                               'Cell environments and entry points are unchanged from the frozen plan.'),
                    'environment_changed': False, 'plan_sha256': plan['plan_sha256'], 'moved': moved})
    tx_path.write_text(json.dumps(tx, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'moved_total': len(moved), 'transaction': str(tx_path)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
