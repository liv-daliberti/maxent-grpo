#!/usr/bin/env python3
"""Prepare or submit a bounded continuation using original frozen job paths.

No training configuration or source is rewritten. The original source bundle,
resolved training identity, sealed checkpoint and every adapter byte are checked
before submission and again inside the new allocation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_original(original_run: Path, checkpoint: Path, arm: str) -> dict:
    original_run, checkpoint = original_run.resolve(), checkpoint.resolve()
    identity_path, config_path = original_run / 'identity.json', original_run / 'config.json'
    frozen = json.loads(identity_path.read_text())
    if frozen.get('schema') != 'real-domains-frozen-job-20260921-v1' or frozen['request'].get('entrypoint') != 'train_real_domains_pilot_20260921.py':
        raise ValueError('resume requires an original frozen training job')
    if arm not in ('maxrl', 'remax') or frozen['request'].get('arm') != arm:
        raise ValueError('resume arm differs from original training arm')
    if digest(config_path) != frozen['config_sha256']:
        raise ValueError('original frozen configuration drift')
    config = json.loads(config_path.read_text())
    frozen_by_path = {}
    for row in frozen['files']:
        path = Path(row['snapshot']).resolve()
        if not path.is_relative_to(original_run / 'bundle') or digest(path) != row['sha256']:
            raise ValueError(f'original frozen dependency drift: {path}')
        frozen_by_path[str(path)] = row['sha256']
    runner = original_run / 'bundle/ops/train_real_domains_pilot_20260921.py'
    if str(runner) not in frozen_by_path:
        raise ValueError('original trainer is absent from frozen identity')
    training_identity_path = checkpoint.parent / 'identity.json'
    training = json.loads(training_identity_path.read_text())
    if training.get('arm') != arm or training.get('input_config_sha256') != digest(config_path):
        raise ValueError('checkpoint training identity differs from original configuration or arm')
    resolved = training['config']
    if canonical_hash(resolved) != training['config_sha256'] or any(resolved.get(k) != v for k, v in config.items()):
        raise ValueError('resolved training configuration drift')
    if training.get('runner_sha256') != frozen_by_path[str(runner)]:
        raise ValueError('checkpoint trainer differs from original frozen source')
    adapter_path = Path(training['adapter_module']).resolve()
    if frozen_by_path.get(str(adapter_path)) != training['adapter_module_sha256']:
        raise ValueError('checkpoint adapter differs from original frozen source')
    for name, expected in training.get('production_source_sha256', {}).items():
        path = original_run / 'bundle/src' / (name.replace('.', '/') + '.py')
        if frozen_by_path.get(str(path)) != expected:
            raise ValueError('production learner source differs from original frozen source')
    model_path = Path(resolved['model'])
    if model_path.name != resolved['model_revision'] or digest(model_path / 'config.json') != training['model_config_sha256']:
        raise ValueError('base model revision or config drift')
    seal_path = checkpoint / 'complete.json'
    seal = json.loads(seal_path.read_text())
    if seal.get('arm') != arm or seal.get('config_sha256') != training['config_sha256']:
        raise ValueError('checkpoint seal configuration or arm drift')
    completed = seal['completed_updates']
    if type(completed) is not int or not 0 < completed < resolved['updates']:
        raise ValueError('checkpoint must precede the original target update count')
    if digest(checkpoint / 'training.pt') != seal['training_state_sha256'] or digest(checkpoint / 'bank.json') != seal['bank_sha256']:
        raise ValueError('checkpoint training state or bank drift')
    files = {p.relative_to(checkpoint / 'adapter').as_posix(): p for p in (checkpoint / 'adapter').rglob('*') if p.is_file()}
    if not files or set(files) != set(seal['adapter_files']):
        raise ValueError('checkpoint adapter file set differs from seal')
    for relative, path in files.items():
        if digest(path) != seal['adapter_files'][relative]:
            raise ValueError('checkpoint adapter bytes differ from seal')
    return {
        'original_run': str(original_run), 'original_identity_sha256': digest(identity_path),
        'config_path': str(config_path), 'config_sha256': digest(config_path),
        'resolved_config_sha256': training['config_sha256'], 'bundle': str(original_run / 'bundle'),
        'training_identity_path': str(training_identity_path), 'training_identity_sha256': digest(training_identity_path),
        'checkpoint': str(checkpoint), 'checkpoint_seal_sha256': digest(seal_path),
        'completed_updates_before_resume': completed, 'original_target_updates': resolved['updates'],
        'arm': arm, 'model_revision': resolved['model_revision'], 'runner_sha256': training['runner_sha256'],
    }


def verify_prepared(identity_path: Path) -> dict:
    identity = json.loads(identity_path.read_text())
    actual = validate_original(Path(identity['request']['original_run']), Path(identity['request']['checkpoint']), identity['request']['arm'])
    if actual != identity['resume_provenance']:
        raise ValueError('resume provenance changed after preparation')
    if digest(Path(identity['guard_path'])) != identity['guard_sha256']:
        raise ValueError('resume validation source changed')
    return actual


def prepare(request_path: Path, output: Path, submit: bool) -> dict:
    request = json.loads(request_path.read_text())
    output = output.resolve()
    if output.exists():
        raise FileExistsError('resume output directory must be new')
    minutes = request['time_limit_minutes']
    if type(minutes) is not int or not 1 <= minutes <= 240 or request.get('gpu_count', 1) != 1:
        raise ValueError('resume allocation must be one GPU for1..240minutes')
    partition = request.get('partition', 'all')
    if partition not in {'all', 'lowprio', 'mltheory'}:
        raise ValueError('unsupported resume partition')
    provenance = validate_original(Path(request['original_run']), Path(request['checkpoint']), request['arm'])
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True)
    guard = output / Path(__file__).name
    shutil.copyfile(Path(__file__), guard)
    identity = {'schema': 'real-domains-resume-job-20260921-v1',
                'prepared_at': datetime.now(timezone.utc).isoformat(),
                'request': request, 'request_sha256': digest(request_path),
                'resume_provenance': provenance, 'job_id': None,
                'allocated_gpu_hour_ceiling': minutes / 60,
                'guard_path': str(guard), 'guard_sha256': digest(guard),
                'output_training': str(output / 'training')}
    identity_path = output / 'identity.json'
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    bundle = Path(provenance['bundle'])
    python = root / 'var/seed_paper_eval/paper310/bin/python'
    q = shlex.quote
    args = [str(python), str(bundle / 'ops/train_real_domains_pilot_20260921.py'),
            '--config', provenance['config_path'], '--output', str(output / 'training'),
            '--arm', request['arm'], '--resume', provenance['checkpoint']]
    script = '\n'.join([
        '#!/usr/bin/env bash', 'set -euo pipefail',
        f'export OAT_ZERO_REPO_ROOT={q(str(root))}',
        f'source {q(str(bundle / "repo_env.sh"))}',
        f'export OAT_ZERO_SOURCE_ROOT={q(str(bundle / "src"))}',
        f'export OAT_ZERO_TESTLIB_ROOT={q(str(bundle / "testlib"))}',
        f'export OAT_ZERO_SANDBOX_SOURCE={q(str(bundle / "ops/constructive_code_sandbox.c"))}',
        f'export PYTHONPATH={q(str(bundle / "ops") + ":" + str(bundle / "src"))}',
        f'export LD_LIBRARY_PATH={q(str(python.parent.parent / "lib"))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0',
        'export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4',
        f'{q(str(python))} - {q(str(identity_path))} {digest(identity_path)} {q(str(guard))} {digest(guard)} <<\'PYVERIFY\'',
        'import hashlib,pathlib,sys',
        'for path, expected in ((sys.argv[1],sys.argv[2]),(sys.argv[3],sys.argv[4])):',
        '    assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==expected, "resume binding drift"',
        'PYVERIFY',
        shlex.join([str(python), str(guard), '--verify-identity', str(identity_path)]),
        'exec ' + shlex.join(args), '',
    ])
    script_path = output / 'run.slurm'
    script_path.write_text(script)
    submission = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1',
                  '--gres=gpu:a6000:1', f'--cpus-per-task={int(request.get("cpus", 8))}',
                  f'--mem={int(request.get("memory_gb", 64))}G', f'--time={minutes}',
                  '--account=mltheory', f'--partition={partition}',
                  f'--job-name={request.get("job_name", "real-domains-resume")}',
                  f'--output={output}/slurm-%j.out', f'--error={output}/slurm-%j.err', str(script_path)]
    (output / 'submission_intent.json').write_text(json.dumps({'argv': submission, 'authorized_gpu_hour_ceiling': minutes / 60}, indent=2) + '\n')
    job_id = None
    if submit:
        proc = subprocess.run(submission, text=True, capture_output=True, check=False)
        receipt = {'argv': submission, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
        if proc.returncode == 0:
            job_id = int(proc.stdout.strip().split(';')[0])
            receipt['job_id'] = job_id
        (output / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if proc.returncode:
            raise RuntimeError(f'sbatch resume failed: {proc.stderr}')
    return {'output': str(output), 'job_id': job_id, 'gpu_hour_ceiling': minutes / 60, **provenance}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--verify-identity', type=Path)
    args = parser.parse_args()
    if args.verify_identity:
        result = verify_prepared(args.verify_identity)
    else:
        if args.request is None or args.output is None:
            parser.error('--request and --output are required')
        result = prepare(args.request, args.output, args.submit)
    print(json.dumps(result, sort_keys=True))
