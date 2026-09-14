#!/usr/bin/env python3
"""Seal or submit the fresh 64-draw local panel; never modify previous runs."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_modebench_sampling_budget_local import SCHEMA, SETTINGS, validate_plan, verify_checkpoint
from evaluate_modebench_level3 import atomic_new, file_sha
BASE = ROOT / 'artifacts/modebench_discovery_curves_20260911'
PRIOR = ROOT / 'artifacts/modebench_prompt_ablation_20260911'
MANIFEST_SHA = '1d42de03313dc571329cbc14b4339d124c0994aae36eb6d2a8f4ea9d5b6b05e3'
PRIOR_PLAN_SHA = 'f82f028dab3c2dfd2590db91297e0f6eac43101afde561f2ce328d0b493576f5'


def prepare():
    local = BASE / 'local'
    local.mkdir(exist_ok=True)
    if (local / 'plan.json').exists():
        raise FileExistsError('local plan already sealed')
    assert file_sha(BASE / 'manifest.json') == MANIFEST_SHA
    assert file_sha(PRIOR / 'local/plan_v2.json') == PRIOR_PLAN_SHA
    root_manifest = json.loads((BASE / 'manifest.json').read_text())
    for name, digest in root_manifest['artifact_sha256'].items():
        assert file_sha(BASE / name) == digest
    for name, digest in root_manifest['code_sha256'].items():
        assert file_sha(BASE / 'code' / name) == digest
    old = json.loads((PRIOR / 'local/plan_v2.json').read_text())
    snapshot = local / 'code'
    snapshot.mkdir(exist_ok=True)
    pins = {}
    for source, digest in old['code_sha256'].items():
        source = Path(source)
        assert file_sha(source) == digest
        target = snapshot / source.relative_to(PRIOR / 'local/code_v2')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert file_sha(target) == digest
        else:
            shutil.copyfile(source, target)
        pins[str(target)] = digest
    for name in ('evaluate_modebench_sampling_budget_local.py', 'prepare_modebench_sampling_budget_local.py',
                 'test_modebench_sampling_budget_local.py'):
        source, target = ROOT / 'ops' / name, snapshot / 'ops' / name
        if target.exists():
            assert file_sha(target) == file_sha(source)
        else:
            shutil.copyfile(source, target)
        pins[str(target)] = file_sha(target)
    checkpoints = copy.deepcopy(old['checkpoints'])
    for checkpoint in checkpoints:
        checkpoint['expected_draws'] *= 4
    inputs = [BASE / name for name in ('manifest.json', 'protocol.json', 'ANALYSIS_PLAN.md',
                                      'support_reference.json', 'rows.jsonl', 'prompts.jsonl', 'selection.json')]
    inputs += [local / 'preparation_tests.json', PRIOR / 'local/plan_v2.json', PRIOR / 'selection.json', PRIOR / 'rows.jsonl', PRIOR / 'prompts.jsonl']
    if any(not p.is_file() for p in inputs):
        raise ValueError('root frozen contract missing required files: ' + str([str(p) for p in inputs if not p.is_file()]))
    rng_keys = [(level, domain, index) for level in (2, 3) for domain in ('python_factors', 'mathir', 'pantry') for index in range(128)]
    from evaluate_modebench_sampling_budget_local import problem_seed
    from evaluate_modebench_prompt_ablation_local_v2 import problem_seed as old_seed
    child = [problem_seed(key, block) + i for key in rng_keys for block in range(8) for i in range(8)]
    previous = {old_seed(key) + i for key in rng_keys for i in range(8)}
    assert len(set(child)) == 49152 and not set(child).intersection(previous) and max(child) < 2**31
    plan = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
            'rows_path': str(BASE / 'rows.jsonl'), 'prompts_path': str(BASE / 'prompts.jsonl'),
            'prior_plan_path': str(PRIOR / 'local/plan_v2.json'), 'prior_selection_path': str(PRIOR / 'selection.json'),
            'input_sha256': {str(p): file_sha(p) for p in inputs}, 'code_sha256': pins,
            'settings': SETTINGS, 'checkpoints': checkpoints, 'output_root': str(local / 'results'),
            'expected_draws': 110592, 'expected_checkpoints': 25,
            'selection': 'Existing outcome-independent selection rank_in_cell<=16, first16 of prior32 in each cell.',
            'fresh_collection': 'All64 draws are new; previous n8 outcomes are neither reused nor pooled.',
            'rng_audit': {'all_candidate_problems': 768, 'expanded_child_seeds': len(child),
                          'unique_child_seeds': len(set(child)), 'overlap_with_previous_n8_child_seeds': 0,
                          'minimum_seed': min(child), 'maximum_seed': max(child)},
            'transfer_interpretation': 'Exact E119 L2 checkpoints; evaluation L3 is transfer.',
            'model_cache': 'Read-only reuse of complete checksum-pinned local checkpoint cache from prior ablation.',
            'scheduler': {'partition': 'lowprio', 'account': 'mltheory', 'gres': 'gpu:a5000:1',
                          'cpus': 6, 'memory': '40G', 'time': '02:00:00', 'array': '0-24%8',
                          'maximum_concurrent_owned_evaluation_gpus': 8},
            'free_bytes_at_preparation': shutil.disk_usage(local).free}
    validate_plan(plan)
    atomic_new(local / 'plan.json', plan)
    (local / 'logs').mkdir(exist_ok=True)
    q = shlex.quote
    worker = '\n'.join(['#!/usr/bin/env bash', 'set -euo pipefail', f'cd {q(str(ROOT))}', f'export OAT_ZERO_REPO_ROOT={q(str(ROOT))}',
       f'source {q(str(snapshot / "ops/repo_env.sh"))}',
       'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0',
       'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false',
       'export VLLM_ATTENTION_BACKEND=XFORMERS',
       f'exec {q(str(ROOT / "var/seed_paper_eval/paper310/bin/python"))} -B {q(str(snapshot / "ops/evaluate_modebench_sampling_budget_local.py"))} --plan {q(str(local / "plan.json"))} --task-index "${{SLURM_ARRAY_TASK_ID:?}}"', ''])
    with (local / 'worker.slurm').open('x') as handle:
        handle.write(worker)
    print(json.dumps({'status': 'prepared', 'plan': str(local / 'plan.json'), 'sha256': file_sha(local / 'plan.json'),
                      'checkpoints': 25, 'draws': 110592}), flush=True)


def submit():
    local = BASE / 'local'
    plan_path = local / 'plan.json'
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if file_sha(BASE / 'manifest.json') != MANIFEST_SHA:
        raise ValueError('frozen root contract changed')
    for checkpoint in plan['checkpoints']:
        verify_checkpoint(checkpoint)
    with (local / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (local / 'submission_intent.json').exists():
            raise ValueError('submission already attempted; inspect immutable receipt')
        capacity = subprocess.run(['sinfo', '-N', '-p', 'lowprio', '--format=%N %t %G'], text=True, capture_output=True, check=True)
        idle = [line for line in capacity.stdout.splitlines() if ' idle ' in line and 'gpu:a5000:' in line]
        idle_count = sum(int(line.split('gpu:a5000:')[1].split('(')[0].split()[0]) for line in idle)
        if idle_count < 8:
            raise ValueError('fresh scheduler check does not show eight idle eligible A5000 GPUs')
        command = ['sbatch', '--parsable', '--job-name=mb-discovery64', '--partition=lowprio', '--account=mltheory',
                   '--gres=gpu:a5000:1', '--cpus-per-task=6', '--mem=40G', '--time=02:00:00', '--array=0-24%8',
                   '--chdir=' + str(ROOT), '--output=' + str(local / 'logs/%A_%a.out'),
                   '--error=' + str(local / 'logs/%A_%a.err'), str(local / 'worker.slurm')]
        atomic_new(local / 'submission_intent.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                   'command': command, 'checkpoint_indices': list(range(25)), 'plan_sha256': file_sha(plan_path),
                   'worker_sha256': file_sha(local / 'worker.slurm'), 'launcher_sha256': file_sha(__file__),
                   'scheduler_capacity_before_submit': capacity.stdout, 'idle_eligible_a5000_count': idle_count,
                   'global_overlap_proof': 'Single new array 0-24%8, one GPU per task; no other discovery-curve array submitted. Prior paired panel is complete.'})
        result = subprocess.run(command, text=True, capture_output=True)
        job_id = result.stdout.strip().split(';')[0] if not result.returncode else None
        atomic_new(local / 'submission_result.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                   'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'job_id': job_id})
        if result.returncode:
            raise RuntimeError(result.stderr)
        readback = subprocess.run(['scontrol', 'show', 'job', job_id], text=True, capture_output=True, check=True)
        atomic_new(local / 'submission_readback.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                   'job_id': job_id, 'readback': readback.stdout, 'plan_sha256': file_sha(plan_path)})
        print(json.dumps({'status': 'submitted', 'job_id': job_id, 'checkpoints': 25, 'draws': 110592}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit'))
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else submit()
