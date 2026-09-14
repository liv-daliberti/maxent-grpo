#!/usr/bin/env python3
"""Freeze, stage exact archived weights, and submit the local prompt ablation."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_modebench_prompt_ablation_local import SCHEMA, SETTINGS, validate_inputs, read_jsonl, verify_checkpoint
from evaluate_modebench_level3 import atomic_new, file_sha, sha
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_prompt_ablation_20260911'
INITIAL = ROOT / 'var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
SEED_PANEL = {'python_factors': (43,44,45,46,47), 'mathir': (43,44,45,46,47), 'pantry': (43,46)}


def prepare(base):
    local = base / 'local'
    if (local / 'plan.json').exists():
        raise FileExistsError('local scientific panel already frozen')
    rows, prompts = base / 'rows.jsonl', base / 'prompts.jsonl'
    validate_inputs(read_jsonl(rows), read_jsonl(prompts))
    ledger_path = ROOT / 'var/artifacts/e119_level2_qwen05b_factorial_jobs.json'
    ledger = json.loads(ledger_path.read_text())
    checkpoints = [{'label': 'qwen05b_initial', 'model_path': str(INITIAL),
                    'training_method': 'initial', 'training_seed': None,
                    'trained_on_level': None, 'domain': None, 'expected_draws': 3072,
                    'source': {'kind': 'frozen_initial_snapshot', 'model': 'Qwen/Qwen2.5-0.5B-Instruct',
                               'snapshot': INITIAL.name},
                    'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': file_sha(p)}
                              for p in sorted(INITIAL.iterdir()) if p.is_file() and p.name != 'README.md']}]
    archive_inputs = {}
    for domain, seeds in SEED_PANEL.items():
        source_domain = 'pantry_plan' if domain == 'pantry' else domain
        for seed in seeds:
            for method in ('drgrpo', 'replay_drgrpo'):
                matched = [r for r in ledger['runs'] if (r['domain'],r['seed'],r['arm']) == (source_domain,seed,method)]
                if len(matched) != 1:
                    raise ValueError('training run identity ambiguous')
                run = matched[0]
                receipt_path = Path(run['run_dir']) / 'MODEL_ARCHIVE.json'
                receipt = json.loads(receipt_path.read_text())
                manifest_path = Path(receipt['manifest_path'])
                if file_sha(manifest_path) != receipt['manifest_sha256']:
                    raise ValueError('archive manifest differs')
                verification_path = Path(receipt['verification_path'])
                if file_sha(verification_path) != receipt['verification_sha256']:
                    raise ValueError('archive verification differs')
                manifest = json.loads(manifest_path.read_text())
                completion_path = Path(run['run_dir']) / 'TRAINING_COMPLETE.json'
                if file_sha(completion_path) != receipt['completion_receipt_sha256']:
                    raise ValueError('original training completion receipt differs')
                completion = json.loads(completion_path.read_text())
                if completion.get('terminal_step') != 3073 or completion.get('terminal_export') != receipt['original_terminal_export']:
                    raise ValueError('training completion endpoint differs')
                verification = json.loads(verification_path.read_text())
                if (verification.get('commit_sha') != receipt['commit_sha'] or
                    verification.get('manifest_sha256') != receipt['manifest_sha256'] or
                    verification.get('remote_bytes', {}).get('status') != 'verified'):
                    raise ValueError('archive lacks verified exact remote bytes')
                terminal = Path(receipt['original_terminal_export'])
                if (terminal.name != 'step_03073' or str(terminal) != manifest['export_dir'] or
                    not terminal.is_relative_to(Path(run['run_dir'])) or receipt['status'] != 'retired'):
                    raise ValueError('archive is not the complete recorded endpoint')
                label = f'qwen05b_E119_{domain}_{method}_s{seed}'
                checkpoints.append({'label': label, 'model_path': str(local / 'models' / label),
                    'training_method': method, 'training_seed': seed, 'trained_on_level': 2,
                    'domain': domain, 'expected_draws': 1024,
                    'source': {'kind': 'archived_terminal_checkpoint', 'training_run': run,
                               'archive_receipt': str(receipt_path), 'archive_receipt_sha256': file_sha(receipt_path),
                               'archive_manifest': str(manifest_path), 'archive_manifest_sha256': file_sha(manifest_path),
                               'repo_id': receipt['repo_id'], 'commit_sha': receipt['commit_sha'],
                               'repo_prefix': receipt['repo_prefix'], 'original_terminal_export': str(terminal),
                               'completion_receipt_sha256': receipt['completion_receipt_sha256'],
                               'export_step_label': 3073, 'audited_evaluation_optimizer_step': 3072},
                    'files': [{'name': f['relative_path'], 'bytes': f['size'], 'sha256': f['sha256'],
                               'source_local_path': f['local_path'], 'path_in_repo': f['path_in_repo']}
                              for f in manifest['files']]})
                archive_inputs.update({str(p): file_sha(p) for p in (receipt_path,manifest_path,verification_path,completion_path)})
    if len(checkpoints) != 25 or sum(c['expected_draws'] for c in checkpoints) != 27648:
        raise ValueError('panel size differs')
    local.mkdir(parents=True, exist_ok=True)
    snapshot = local / 'code'
    sources = [ROOT / 'ops' / name for name in ('evaluate_modebench_prompt_ablation_local.py',
               'launch_modebench_prompt_ablation_local.py', 'evaluate_modebench_level3.py',
               'evaluate_modebench_level2_viability.py', 'frontier_modebench_contract.py', 'repo_env.sh')]
    sources += sorted((ROOT / 'src').rglob('*.py'))
    source_pins = {}
    for source in sources:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(source, target)
        source_pins[str(target)] = file_sha(target)
    inputs = {str(p): file_sha(p) for p in (rows, prompts, ledger_path)}
    for p in base.glob('*manifest*.json'):
        inputs[str(p)] = file_sha(p)
    plan = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
            'rows_path': str(rows), 'prompts_path': str(prompts),
            'input_sha256': {**inputs, **archive_inputs}, 'code_sha256': source_pins,
            'settings': SETTINGS, 'checkpoints': checkpoints, 'output_root': str(local / 'results'),
            'checkpoint_selection': 'Every available completed matched DrGRPO/ReplayDrGRPO seed pair at freeze time; no performance selection.',
            'transfer_interpretation': 'E119 checkpoints trained on Level2; Level3 evaluation measures transfer.',
            'seed_panel': SEED_PANEL, 'expected_draws': 27648,
            'storage': {'isolated_model_bytes': sum(f['bytes'] for c in checkpoints[1:] for f in c['files']),
                        'free_bytes_at_preparation': shutil.disk_usage(local).free},
            'scheduler': {'partition': 'lowprio', 'account': 'mltheory', 'gres': 'gpu:a5000:1',
                          'cpus': 6, 'memory': '40G', 'time': '01:00:00', 'array': '0-24%2'}}
    atomic_new(local / 'plan.json', plan)
    (local / 'logs').mkdir(exist_ok=True)
    q = shlex.quote
    worker = '\n'.join(['#!/usr/bin/env bash', 'set -euo pipefail', f'cd {q(str(ROOT))}',
       f'source {q(str(ROOT / "ops/repo_env.sh"))}',
       'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0',
       'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false',
       'export VLLM_ATTENTION_BACKEND=XFORMERS',
       f'exec {q(str(ROOT / "var/seed_paper_eval/paper310/bin/python"))} -B {q(str(snapshot / "ops/evaluate_modebench_prompt_ablation_local.py"))} --plan {q(str(local / "plan.json"))} --task-index "${{SLURM_ARRAY_TASK_ID:?}}"', ''])
    with (local / 'worker.slurm').open('x') as handle:
        handle.write(worker)
    print(json.dumps({'status': 'prepared', 'plan': str(local/'plan.json'), 'sha256': file_sha(local/'plan.json'),
                      'checkpoints': len(checkpoints), 'draws': plan['expected_draws']}), flush=True)


def stage(local):
    plan = json.loads((local / 'plan.json').read_text())
    reserve = plan['storage']['isolated_model_bytes'] + 50 * 1024**3
    if shutil.disk_usage(local).free < reserve:
        raise RuntimeError('insufficient free space for isolated staging plus 50GiB reserve')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ.pop('HF_HUB_OFFLINE', None)
    from huggingface_hub import hf_hub_download
    def one(checkpoint):
        if checkpoint['training_method'] == 'initial':
            verify_checkpoint(checkpoint)
            return checkpoint['label']
        target = Path(checkpoint['model_path'])
        target.mkdir(parents=True, exist_ok=True)
        for item in checkpoint['files']:
            dest = target / item['name']
            if dest.exists():
                if file_sha(dest) != item['sha256']:
                    raise ValueError(f'existing staged weight differs: {dest}')
                continue
            source = Path(item['source_local_path'])
            if source.exists():
                if file_sha(source) != item['sha256']:
                    raise ValueError(f'source export changed: {source}')
                shutil.copyfile(source, dest)
            else:
                downloaded = hf_hub_download(repo_id=checkpoint['source']['repo_id'],
                    filename=item['path_in_repo'], revision=checkpoint['source']['commit_sha'],
                    token=False, cache_dir=str(local/'download_cache'))
                if file_sha(Path(downloaded)) != item['sha256']:
                    raise ValueError('commit-pinned archive download checksum differs')
                os.link(Path(downloaded).resolve(), dest)
        verify_checkpoint(checkpoint)
        print(json.dumps({'event': 'checkpoint_staged', 'checkpoint': checkpoint['label']}), flush=True)
        return checkpoint['label']
    with ThreadPoolExecutor(max_workers=2) as executor:
        staged = list(executor.map(one, plan['checkpoints']))
    receipt = local / 'staging_complete.json'
    if not receipt.exists():
        atomic_new(receipt, {'status': 'complete', 'plan_sha256': file_sha(local/'plan.json'),
                   'checkpoints': staged, 'free_bytes_after_staging': shutil.disk_usage(local).free})


def submit(local):
    plan_path = local / 'plan.json'
    plan = json.loads(plan_path.read_text())
    from evaluate_modebench_prompt_ablation_local import validate_plan
    validate_plan(plan)
    for c in plan['checkpoints']:
        verify_checkpoint(c)
    if not (local/'staging_complete.json').exists():
        raise ValueError('staging not complete')
    intent = local / 'submission_intent.json'
    command = ['sbatch', '--parsable', '--job-name=mb-prompt-local', '--partition=lowprio', '--account=mltheory',
               '--gres=gpu:a5000:1', '--cpus-per-task=6', '--mem=40G', '--time=01:00:00', '--array=0-24%2',
               '--chdir='+str(ROOT), '--output='+str(local/'logs/%A_%a.out'),
               '--error='+str(local/'logs/%A_%a.err'), str(local/'worker.slurm')]
    atomic_new(intent, {'command': command, 'plan_sha256': file_sha(plan_path),
                       'worker_sha256': file_sha(local/'worker.slurm'), 'created_at': datetime.now(timezone.utc).isoformat()})
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    atomic_new(local/'submission_result.json', {'returncode': result.returncode, 'stdout': result.stdout,
               'stderr': result.stderr, 'job_id': result.stdout.strip().split(';')[0] if result.returncode==0 else None})
    if result.returncode:
        raise RuntimeError(result.stderr)
    print(result.stdout.strip(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare','stage','submit'))
    p.add_argument('--base', type=Path, default=BASE)
    args = p.parse_args()
    if args.action == 'prepare': prepare(args.base.resolve())
    elif args.action == 'stage': stage(args.base.resolve()/'local')
    else: submit(args.base.resolve()/'local')
if __name__ == '__main__': main()
