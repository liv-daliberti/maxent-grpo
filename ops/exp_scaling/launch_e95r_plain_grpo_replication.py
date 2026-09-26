#!/usr/bin/env python3
"""Re-run the 55 plain-GRPO cells whose trained policies no longer exist.

On 2026-08-24 the weight files of all 55 E95 runs were removed in a single
sub-second pass, from ``var/checkpoints`` -- a root both retention scripts are
scoped to skip -- and they had never been uploaded. Their metrics survive as
recorded numbers, but the policies are gone, so the plain-GRPO half of the
manuscript's concentration claim can never be re-measured at Qwen-0.5B or
Falcon-1B.

This is a replication, not a recovery, and the distinction matters for what the
results may be used for. The original runtime snapshots are gone too, so these
run on the current source tree, and training is not bit-reproducible in any
case. The policies produced here are new ones trained under the same recipe;
they cannot be substituted into the E95 cells as though those had been
re-verified.

Two things make the replication better than what it replaces. The evaluation
seeds draw blocks disjointly now (``eval_mode_coverage_disjoint_draws``), so
these cells are born with thirty-two independent streams per prompt rather than
eleven. And each run is registered for archival on completion, so a repeat of
the original loss requires a repeat of the original mistake.

The recipe is not reconstructed by inference: every one of the 55 runs preserved
its complete submitted environment in the E95 ledgers, and this replays exactly
that, changing only the output location, the job name, and the source root that
no longer exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LEDGERS = ('e95_plain_grpo_Qwen25-05B_jobs.json',
           'e95_plain_grpo_Falcon3-1B_jobs.json',
           'e95_plain_grpo_Qwen25-3B_jobs.json')
OUT_LEDGER = ROOT / 'var/artifacts/e95r_plain_grpo_replication_jobs.json'
RUN_ROOT = ROOT / 'var/data'
SLURM = ROOT / 'ops/slurm/train_node302.slurm'
EXPORT = re.compile(r'--export=([^\s]+)')
# Replaced per run; everything else is replayed byte-for-byte from the ledger.
REDIRECTED = ('SAVE_PATH', 'RUN_STAMP', 'OAT_ZERO_SOURCE_ROOT', 'OAT_ZERO_OPS_SNAPSHOT_ROOT')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def recipes() -> list[dict]:
    """Every E95 cell, with the exact environment its job was submitted under."""
    found = []
    for name in LEDGERS:
        payload = json.loads((ROOT / 'var/artifacts' / name).read_text())
        for run in payload['runs']:
            record = str(run.get('held_scheduler_record', ''))
            hit = EXPORT.search(record)
            require(hit is not None, f'{run["run_stamp"]}: no submitted environment preserved')
            env = dict(item.split('=', 1) for item in hit.group(1).split(',') if '=' in item)
            require(env.get('OAT_ZERO_VARIANT') == 'grpo_plain_control',
                    f'{run["run_stamp"]}: unexpected variant {env.get("OAT_ZERO_VARIANT")!r}')
            for key in ('OAT_ZERO_PROMPT_DATA', 'OAT_ZERO_EVAL_DATA', 'OAT_ZERO_PRETRAIN'):
                require(Path(env[key]).exists(), f'{run["run_stamp"]}: missing {key}={env[key]}')
            found.append({'ledger': name, 'original_stamp': run['run_stamp'],
                          'original_job_id': run.get('job_id'),
                          'original_run_dir': run.get('run_dir'),
                          'domain': run['domain'], 'seed': int(run['seed']), 'env': env})
    require(len(found) == 55, f'expected the 55 E95 cells, found {len(found)}')
    return sorted(found, key=lambda r: (r['ledger'], r['domain'], r['seed']))


def replication_env(recipe: dict, stamp_prefix: str = 'e95r') -> tuple[dict, Path, str]:
    env = dict(recipe['env'])
    stamp = f'{stamp_prefix}_' + recipe['original_stamp'].removeprefix('e95_')
    model_tag = Path(env['OAT_ZERO_PRETRAIN']).parts[-3].replace('models--', '').replace('--', '_').lower()
    save = RUN_ROOT / f'xdr_{model_tag}_grpo_plain_control_{stamp}'
    env['SAVE_PATH'] = str(save)
    env['RUN_STAMP'] = stamp
    # The snapshots these ran from no longer exist; the plain-GRPO overlay they
    # carried has since been merged into the tree, so the live source is used.
    env['OAT_ZERO_SOURCE_ROOT'] = str(ROOT / 'src')
    env['OAT_ZERO_OPS_SNAPSHOT_ROOT'] = str(ROOT / 'ops')
    env['OAT_ZERO_REPO_ROOT'] = str(ROOT)
    return env, save, stamp


def sbatch_command(env: dict, stamp: str, nice: int, gpu: str | None) -> list[str]:
    exports = ','.join(f'{k}={v}' for k, v in sorted(env.items()))
    command = ['sbatch', f'--job-name={stamp}', f'--nice={nice}', '--hold',
               f'--export=ALL,{exports}']
    if gpu:
        command.append(f'--gres=gpu:{gpu}:1')
    command.append(str(SLURM.relative_to(ROOT)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('all', 'Qwen2.5-0.5B', 'Falcon3-1B', 'Qwen2.5-3B'),
                        default='all')
    parser.add_argument('--nice', type=int, default=200,
                        help='submit below ordinary priority; this is a replication, '
                             'not something anyone is waiting on')
    parser.add_argument('--gpu', default=None, help='pin a GPU model, e.g. a100')
    parser.add_argument('--submit', action='store_true')
    # A later wave must not write over an earlier one's ledger. The archive plan
    # pins each ledger by hash and re-checks it before every upload and every
    # deletion, so overwriting the file a published cohort was planned from
    # aborts the archive pass rather than corrupting it quietly.
    parser.add_argument('--out-ledger', type=Path, default=OUT_LEDGER)
    parser.add_argument('--stamp-prefix', default='e95r',
                        help='run-stamp and run-directory prefix; give a later wave its own')
    parser.add_argument('--write-plan', action='store_true',
                        help='write the ledger without submitting, so the plan can be reviewed first')
    args = parser.parse_args()
    require(re.fullmatch(r'[a-z0-9]+', args.stamp_prefix), 'stamp prefix must be lowercase alphanumeric')
    out_ledger = Path(args.out_ledger).resolve()
    require(not out_ledger.exists(), f'ledger already exists, choose another: {out_ledger}')

    wanted = {'Qwen2.5-0.5B': 'Qwen25-05B', 'Falcon3-1B': 'Falcon3-1B', 'Qwen2.5-3B': 'Qwen25-3B'}
    plan = []
    for recipe in recipes():
        if args.family != 'all' and wanted[args.family] not in recipe['ledger']:
            continue
        env, save, stamp = replication_env(recipe, args.stamp_prefix)
        require(not save.exists(), f'{stamp}: output directory already exists: {save}')
        plan.append({'stamp': stamp, 'save_path': str(save), 'domain': recipe['domain'],
                     'seed': recipe['seed'], 'replicates': recipe['original_stamp'],
                     'original_job_id': recipe['original_job_id'],
                     'command': sbatch_command(env, stamp, args.nice, args.gpu), 'env': env})

    ledger = {
        'schema': 'e95r-plain-grpo-replication-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        # Shape fields the shared campaign reader needs to report progress; they
        # mirror the E95 ledgers this replicates, so the cohort shows up in
        # campaign_stats.py like any other.
        'experiment': args.stamp_prefix.upper().replace('E95R', 'E95-R'), 'family': args.family if args.family != 'all' else None,
        'variant': 'grpo_plain_control', 'arms': ['grpo_plain_control'],
        'target_steps': 3072, 'train_rows': 384, 'passes': 8,
        'checkpoint_interval_steps': 192, 'released': True,
        'objective': 'plain_GRPO_reward_only_control',
        'domains': sorted({row['domain'] for row in plan}),
        'seeds': sorted({row['seed'] for row in plan}),
        'replicates': 'E95 plain-GRPO control suite, whose weights were removed 2026-08-24',
        'is_recovery': False,
        'caveat': ('new policies trained under the recorded recipe on the current source tree; '
                   'not the lost checkpoints and not substitutable for them'),
        'evaluation_seeding': 'eval_mode_coverage_disjoint_draws default True: 32 disjoint streams per prompt',
        'runs': [{**{k: v for k, v in row.items() if k != 'env'},
                   'run_dir': row['save_path'], 'arm': 'grpo_plain_control',
                   'run_stamp': row['stamp']} for row in plan],
        'run_count': len(plan),
    }
    if args.submit:
        for row in plan:
            result = subprocess.run(row['command'], cwd=ROOT, capture_output=True, text=True)
            require(result.returncode == 0, f'{row["stamp"]}: sbatch failed: {result.stderr.strip()}')
            row['job_id'] = result.stdout.strip().split()[-1]
        ledger['runs'] = [{**{k: v for k, v in row.items() if k != 'env'},
                           'run_dir': row['save_path'], 'arm': 'grpo_plain_control',
                           'run_stamp': row['stamp']} for row in plan]
    if args.submit or args.write_plan:
        ledger['released'] = bool(args.submit)
        out_ledger.parent.mkdir(parents=True, exist_ok=True)
        out_ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'runs': len(plan), 'submitted': args.submit,
                      'held': args.submit and 'yes -- release with scontrol release',
                      'ledger': str(out_ledger) if (args.submit or args.write_plan) else None,
                      'example': plan[0]['command'][:4] if plan else None}, indent=2))


if __name__ == '__main__':
    main()
