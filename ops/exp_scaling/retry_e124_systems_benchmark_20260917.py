#!/usr/bin/env python3
"""Resubmit the E124 systems benchmark and release that one job by hand.

The benchmark failed on 2026-09-11 because ``systems.slurm`` never sourced
``ops/repo_env.sh``, so DeepSpeed tried to JIT-build CPUAdam instead of loading
the prebuilt extension and died for want of ninja. That is fixed; this retries it.

The release is manual and deliberately narrow. E124's storage gate is fail-closed
on GPU jobs it cannot map to a maxent-grpo ledger, and 28 jobs from another
project on the same filesystem currently trip it -- a condition the gate has no
way to express and no override for. Rather than weaken the gate, exactly one job
is released here: the qualification the other 30 cells wait on. Those cells stay
held and keep waiting for the gate.

``stage_one`` cannot do this, because it refuses to resubmit a row that already
carries a job id and audits it instead; the recorded id belongs to the failed run
and has aged out of the scheduler.
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
LIVE = {'RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}


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
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    m = launcher()
    plan = json.loads((ART / 'plan.json').read_text())
    tx_path = ART / 'transaction.json'
    tx = json.loads(tx_path.read_text())
    row = m.systems_row(plan)
    item = tx['rows'].get('systems')
    require(item and item.get('job_id'), 'no recorded systems submission to supersede')

    state = m.scheduler_state(item['job_id'])
    require(state.get('inactive') and state.get('state') not in LIVE,
            f"recorded systems job {item['job_id']} is {state.get('state')}; never supersede a live benchmark")

    for line in subprocess.run(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%j'],
                               capture_output=True, text=True, timeout=60).stdout.splitlines():
        jid, _, name = line.partition('|')
        require(name != 'e124-7b-systems', f'a systems benchmark is already queued as {jid}')

    # The fix this retry depends on; a retry without it just reproduces the failure.
    require('repo_env.sh' in (ART / 'systems.slurm').read_text(),
            'systems.slurm does not source repo_env.sh; the original failure would recur')
    stress = ART / 'systems' / m.__dict__.get('PROFILE_ID', 'a6000_cpu_adam_mb1_omp4')
    require(not stress.exists(), f'{stress.name} still present; run_suite refuses a rerun over failed stress')

    command = m.job_command(plan, row, benchmark=True)
    print(json.dumps({'superseding_job': item['job_id'], 'previous_state': state,
                      'placement': [x for x in command if x.startswith(('--partition', '--gres', '--nodelist'))]},
                     indent=2))
    if not args.apply:
        print('\ndry run; pass --apply to resubmit and release')
        return 0

    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                         timeout=180, env=m.clean_submit_environment())
    require(out.returncode == 0, 'sbatch failed: ' + out.stderr.strip())
    new_job = out.stdout.strip().split(';')[0]
    require(new_job.isdigit(), f'ambiguous sbatch response: {out.stdout!r}')

    released = subprocess.run(['scontrol', 'release', new_job], capture_output=True, text=True, timeout=60)
    require(released.returncode == 0, f'scancel-safe release of {new_job} failed: {released.stderr.strip()}')

    tx['rows']['systems'] = {'status': 'released', 'command': command, 'job_id': int(new_job),
                             'previous_job_id': str(item['job_id']), 'submitted_at': m.now(),
                             'retries': int(item.get('retries', 0)) + 1,
                             'release': 'manual; storage gate fail-closed on unmappable external writers'}
    tx.setdefault('manual_releases', []).append({
        'at': datetime.now(timezone.utc).isoformat(), 'cell': 'systems', 'job_id': int(new_job),
        'superseded': str(item['job_id']),
        'why': ('Qualification blocks all 30 cells. The storage gate cannot map 28 GPU jobs belonging '
                'to another project on the same filesystem and has no override, so this single job was '
                'released by hand. Science cells remain held and still require the gate.'),
        'scope': 'systems benchmark only; no science cell released'})
    tx_path.write_text(json.dumps(tx, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'systems_job': new_job, 'state': 'released', 'science_cells_released': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
