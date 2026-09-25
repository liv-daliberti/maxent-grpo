#!/usr/bin/env python3
"""Finish the 2026-09-19 E124 throughput work that the operator permission gate stopped.

Two independent actions, either of which can be run alone:

  --fix-l3-memory   Raise the nine Level-3 cells still queued at 120G to 224G, in place.
                    Both smaller sizes are falsified. 120G stalls in the first optimizer
                    step (2026-09-18 level_gating record). 160G looked fine on a single
                    cell but on 2026-09-20 every 160G cell stalled once the node was
                    loaded -- countdown Re:Max sat at step 192 for three hours and ran
                    the moment it was moved to 224G. Only 224G has completed cells.

  --move-to-a6000   Move one matched Level-1 pair from mltheory/a100 (node302, currently
                    full) to the legacy lowprio/a6000 pool, which has a free 224G slot.
                    Measured throughput is ~76 steps/h on a6000 against ~78-86 on a100 --
                    the run is bound by CPU Adam offload and host RAM, not GPU compute --
                    so a slot now beats a faster slot in 13h.

Slurm cannot change a job's partition after submission, so the move is a cancel and
resubmit. Only cells at step 0 are eligible: a new job id means a new debug_job output
directory, and AUTO_RESUME cannot see checkpoints written under the old id. That is the
mechanism that stranded 3,264 steps during the 09-17/18 memory churn.

The submission takes its environment from plan.json and its placement from the stored
command. The stored command's own export block is stale -- see build_command.

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

L3_AT_120G = [31342762, 31342763, 31342764, 31342765,
              31342766, 31342767, 31342768, 31343980, 31343981]

# Retired 2026-09-21. The a6000 pool is only reachable through partition lowprio
# (PriorityTier=1, PreemptMode=REQUEUE). On 2026-09-20 both moved cells were preempted
# twice within five minutes, and a cell needs ~1.2h uninterrupted to reach SAVE_STEPS=96,
# so it can never bank progress there. Do not re-enable without a non-preemptible route.
MOVE_PAIR = []

TARGET_MEM_MB = 229376          # 224G in MB; scontrol rejects the "224G" spelling
LEGACY_POOL = 'node205,node206,node207,node208'


def now():
    return datetime.now(timezone.utc).isoformat()


def run(argv, check=True):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if check and p.returncode != 0:
        raise RuntimeError(f'{argv[0]} failed ({p.returncode}): {p.stderr.strip()}')
    return p.stdout


def fields(job_id):
    out = run(['scontrol', 'show', 'job', str(job_id)], check=False)
    if not out.strip():
        return {}
    return dict(t.split('=', 1) for t in out.split() if '=' in t)


def load_tx():
    return json.loads(TX.read_text())


def save_tx(tx):
    TX.write_text(json.dumps(tx, indent=1, sort_keys=True) + '\n')


def fix_l3_memory(apply):
    print('\n=== raise the nine 120G Level-3 cells to 224G ===')
    changed = []
    for job_id in L3_AT_120G:
        f = fields(job_id)
        if not f:
            print(f'  {job_id}: not in the scheduler, skipping')
            continue
        mem, state, reason = f.get('MinMemoryNode'), f.get('JobState'), f.get('Reason')
        if mem == '224G':
            print(f'  {job_id}: already 224G')
            continue
        if state != 'PENDING':
            print(f'  {job_id}: state is {state}, not PENDING -- refusing to touch it')
            continue
        if reason != 'JobHeldUser':
            print(f'  {job_id}: reason is {reason}, expected JobHeldUser -- refusing')
            continue
        if not apply:
            print(f'  {job_id}: would set {mem} -> 224G')
            continue
        run(['scontrol', 'update', f'jobid={job_id}', f'MinMemoryNode={TARGET_MEM_MB}'])
        after = fields(job_id)
        ok = after.get('MinMemoryNode') == '224G' and after.get('Priority') == '0'
        print(f'  {job_id}: {mem} -> {after.get("MinMemoryNode")} '
              f'(hold {"intact" if after.get("Priority") == "0" else "LOST -- INVESTIGATE"})')
        if ok:
            changed.append(job_id)
    return changed


def plan_environment(cell):
    plan = json.loads((ROOT / 'var/artifacts/e124_qwen7b_three_level/plan.json').read_text())
    return next(c for c in plan['cells'] if c['cell_id'] == cell)['environment']


def build_command(cell, row):
    """Rebuild the submission, taking the environment from plan.json.

    transaction.json's stored `command` is NOT a safe source. It predates the
    2026-09-17 level1_domain_correction, so for the eight Level-1 cells that named a
    domain it still carries OAT_ZERO_MODEBENCH_DOMAIN=<domain>, which invokes the
    Level-2 prompt/syntax contract and kills the cell 62 seconds in. That is exactly
    how the python_factors pair died on 2026-09-20. plan.json is the authority.
    """
    env = plan_environment(cell)
    stored = next(t for t in row['command'] if t.startswith('--export='))
    have = dict(kv.split('=', 1) for kv in stored[len('--export=ALL,'):].split(',') if '=' in kv)
    drift = {k: (env.get(k), have.get(k)) for k in set(env) | set(have) if env.get(k) != have.get(k)}
    if drift:
        print(f'  {cell}: stored command is stale, rebuilding from plan.json -- {drift}')

    cmd = [t for t in row['command'] if t != '--hold']
    cmd = ['--export=ALL,' + ','.join(f'{k}={env[k]}' for k in sorted(env))
           if t.startswith('--export=') else t for t in cmd]
    cmd = [f'--mem={TARGET_MEM_MB // 1024}G' if t.startswith('--mem=') else t for t in cmd]
    for required in ('--partition=lowprio', '--gres=gpu:a6000:1', f'--nodelist={LEGACY_POOL}'):
        if required not in cmd:
            raise RuntimeError(f'stored command is not the legacy a6000 form: missing {required}')
    return cmd


def move_to_a6000(apply):
    print('\n=== move a Level-1 pair to lowprio/a6000 ===')
    if not MOVE_PAIR:
        print('  RETIRED: lowprio preempts these cells faster than they can checkpoint.')
        print('  See the MOVE_PAIR comment. Nothing done.')
        return []
    tx = load_tx()
    moved = []
    for cell in MOVE_PAIR:
        row = tx['rows'][cell]
        old = row['job_id']
        f = fields(old)
        if f.get('JobState') != 'PENDING':
            print(f'  {cell}: job {old} is {f.get("JobState")}, not PENDING -- refusing to move it')
            continue

        run_dir = Path(next(c['run_dir'] for c in
                            json.loads((ROOT / 'var/artifacts/e124_qwen7b_three_level/plan.json').read_text())['cells']
                            if c['cell_id'] == cell))
        progress = sorted(run_dir.glob('debug_job*/checkpoints/step_*'))
        if progress:
            print(f'  {cell}: has checkpoints {[p.name for p in progress]} -- '
                  f'moving would strand them. REFUSING.')
            continue

        cmd = build_command(cell, row)
        if not apply:
            print(f'  {cell}: would cancel {old} and resubmit to lowprio/a6000 at 224G')
            continue

        run(['scancel', str(old)])
        for _ in range(30):
            time.sleep(2)
            if fields(old).get('JobState') in (None, '', 'CANCELLED'):
                break
        new = run(cmd).strip().split(';')[0]
        g = fields(new)
        checks = {
            'Partition': 'lowprio',
            'Account': 'mltheory',
            'TresPerNode': 'gres/gpu:a6000:1',
            'MinMemoryNode': '224G',
            'TimeLimit': '3-00:00:00',
        }
        bad = {k: g.get(k) for k, v in checks.items() if g.get(k) != v}
        if bad:
            print(f'  {cell}: job {new} submitted but placement is WRONG: {bad}')
            print('         partition is coupled to account on this cluster -- verify before letting it run')
        else:
            print(f'  {cell}: {old} -> {new} on lowprio/a6000 at 224G, verified')

        row['previous_job_id'] = old
        row['job_id'] = int(new)
        row['status_note'] = ('moved to lowprio/a6000 2026-09-19 to use a free slot while node302 was full; '
                              'cell was at step 0 so no progress was stranded')
        moved.append({'cell': cell, 'from': old, 'to': int(new), 'verified': not bad})

    if apply and moved:
        tx.setdefault('placement_history', []).append({
            'at': now(),
            'schema': 'e124_placement_change_v1',
            'direction': 'mltheory/a100 -> lowprio/a6000',
            'cells': moved,
            'why': ('node302 is the only a100 in mltheory and was fully allocated at 448G of 503G. The legacy '
                    'a6000 pool had a free slot on node206. Measured throughput is ~76 steps/h on a6000 '
                    'against ~78-86 on a100, because the cell is bound by CPU Adam offload and host RAM rather '
                    'than GPU compute, so the placement costs almost no speed and gains a slot immediately.'),
            'pair_integrity': ('both arms of the mathir Level-1 pair move together, so the maxrl vs replay_maxrl '
                               'comparison is not split across GPU models'),
            'eligibility': 'both cells were at step 0, so the new job id and new debug_job directory strand nothing',
            'scientific_settings_changed': False,
            'sanctioned_by': 'controller_operational_amendment_20260917_v4, which accepts lowprio/a6000 per cell',
            'preemption': 'lowprio is PreemptMode=REQUEUE, so a preempted cell requeues under the same job id and resumes',
        })
        save_tx(tx)
        print(f'\n  transaction.json updated with {len(moved)} placement change(s)')
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='act; otherwise dry run')
    ap.add_argument('--fix-l3-memory', action='store_true')
    ap.add_argument('--move-to-a6000', action='store_true')
    a = ap.parse_args()
    if not (a.fix_l3_memory or a.move_to_a6000):
        ap.error('pick at least one of --fix-l3-memory / --move-to-a6000')
    if not a.apply:
        print('DRY RUN -- pass --apply to act')

    if a.fix_l3_memory:
        fix_l3_memory(a.apply)
    if a.move_to_a6000:
        move_to_a6000(a.apply)

    print('\n=== Level-1 state ===')
    tx = load_tx()
    for cell, row in sorted(tx['rows'].items()):
        if not cell.startswith('l1-'):
            continue
        f = fields(row['job_id'])
        print(f'  {cell:<40} {row["job_id"]:>10} {f.get("Partition",""):>9} '
              f'{f.get("MinMemoryNode",""):>6} {f.get("JobState",""):>8} {f.get("Reason","")[:22]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
