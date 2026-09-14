#!/usr/bin/env python3
"""Prepare, audit and stage the user-authorized thirty-cell E124 sweep.

Scientific jobs remain held until the isolated systems suite and storage gate
pass. Mutations have durable intents; this tool never modifies another campaign.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import math
import pickletools
import zipfile
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get('OAT_ZERO_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
SOURCE = Path(__file__).resolve()
ART = ROOT / 'var/artifacts/e124_qwen7b_three_level'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
LEDGER = ROOT / 'var/artifacts/e124_qwen7b_three_level_jobs.json'
PROTOCOL = ROOT / 'paper/preregistration/e124_qwen7b_three_level_20260909.md'
PYTHON = ROOT / 'var/seed_paper_eval/paper310/bin/python'
HEALTHY_OPS = ROOT / 'var/artifacts/source_snapshots/e76_tuned_scale_50d36295558a8958/ops'
MODEL = ROOT / 'var/cache/huggingface/transformers/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
PVL = 'node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]'
POOL = 'node205,node206,node207,node208'
USER_HELD_REASONS = frozenset({'JobHeldUser', 'job_requeued_in_held_state', 'job requeued in held state'})
LIVE = {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'PENDING'}
TERMINAL = {'COMPLETED', 'TIMEOUT', 'PREEMPTED', 'FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value, *, new=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if new:
        with path.open('x') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(path.parent)
        return
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def lock():
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'controller.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def clean_submit_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(('OAT_ZERO_', 'SBATCH_', 'SLURM_'))
            and key not in ('SAVE_PATH', 'RUN_STAMP', 'ROOT_DIR', 'PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}


def command(args, *, timeout=120):
    kwargs = {'env': clean_submit_environment()} if Path(str(args[0])).name == 'sbatch' else {}
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True, timeout=timeout, **kwargs)
    require(result.returncode == 0, f'command failed ({result.returncode}): {args[0]}: {result.stderr[-1000:]}')
    return result.stdout


def fields(record):
    return {k: v for k, v in re.findall(r'([A-Za-z][A-Za-z0-9_/]*)=([^ ]+)', record)}


def show(job_id):
    return command(['scontrol', 'show', 'job', '-o', str(int(job_id))])


def submit_tokens(record):
    require('SubmitLine=' in record, 'scheduler record lacks original submit command')
    # Slurm appends WorkDir/StdErr/StdOut after SubmitLine in its one-line record.
    return shlex.split(record.split('SubmitLine=', 1)[1].split(' WorkDir=', 1)[0])


def exports(tokens):
    values = [x.split('=', 1)[1] for x in tokens if x.startswith('--export=')]
    require(len(values) == 1 and values[0].startswith('ALL,'), 'one complete export map required')
    parts = values[0][4:].split(',')
    result = dict(part.split('=', 1) for part in parts)
    require(len(result) == len(parts), 'duplicate environment key')
    return result


def runtime_bytes(relative, original):
    if relative not in {'ops/run_experiment.sh', 'ops/train.sh', 'ops/slurm/train_node302.slurm'}:
        return original
    text = original.decode()
    text = text.replace('e119_level2_*', 'e124_l[123]_*')
    text = text.replace('^e119_level2_pantry_(drgrpo|replay_drgrpo|maxrl|replay_maxrl)_s(43|44|45|46|47)$',
                        '^e124_l[23]_pantry_(maxrl|replay_maxrl)_s70$')
    if relative == 'ops/slurm/train_node302.slurm':
        text = text.replace('$ROOT_DIR/var/artifacts/logs/xdr_train-${SLURM_JOB_ID}.out',
                            '$ROOT_DIR/var/artifacts/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out')
    return text.encode()


def runtime_description():
    sources = {}
    for directory, prefix in ((ROOT / 'src/oat_drgrpo', 'src/oat_drgrpo'), (HEALTHY_OPS, 'ops')):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                sources[prefix + '/' + str(path.relative_to(directory))] = path
    for name in ('launch_e124_qwen7b_three_level.py', 'e124_qwen7b_recipes.py',
                 'e124_storage_admission.py', 'benchmark_e124_qwen7b.py', 'benchmark_e123_a100_runtime.py'):
        sources['control/' + name] = ROOT / 'ops/exp_scaling' / name
    inventory = {}
    for relative, path in sources.items():
        content = runtime_bytes(relative, path.read_bytes())
        inventory[relative] = {'source': str(path), 'source_sha256': digest(path),
                               'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    identity = seal(inventory)
    return {'root': str(ROOT / 'var/artifacts/source_snapshots' / ('e124_qwen7b_' + identity[:20])),
            'sha256': identity, 'inventory': inventory}


def publish_snapshot(snapshot):
    target = Path(snapshot['root'])
    require(not target.exists(), 'snapshot already exists; inspect it instead of overwriting')
    target.mkdir()
    try:
        for relative, item in snapshot['inventory'].items():
            source = Path(item['source'])
            require(digest(source) == item['source_sha256'], 'source changed while publishing: ' + str(source))
            payload = runtime_bytes(relative, source.read_bytes())
            require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'runtime transformation drift')
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(payload)
            destination.chmod(0o555 if source.stat().st_mode & 0o111 else 0o444)
        write(target / 'identity.json', snapshot, new=True)
    except BaseException:
        # An incomplete content-addressed snapshot is evidence, never reused.
        raise


def model_identity():
    require(MODEL.is_dir(), 'exact pinned pretrained model snapshot is missing')
    index = read(MODEL / 'model.safetensors.index.json')
    names = sorted(set(index['weight_map'].values()))
    require(names and all(Path(n).name == n and n.endswith('.safetensors') for n in names), 'invalid weight shard inventory')
    pins = {}
    for name in names + ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'model.safetensors.index.json']:
        path = MODEL / name
        stat = path.stat()
        require(stat.st_size > 0, 'empty model file: ' + name)
        pins[str(path)] = {'sha256': digest(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                           'resolved': str(path.resolve())}
    return {'path': str(MODEL), 'revision': MODEL.name, 'model': 'Qwen/Qwen2.5-7B-Instruct',
            'total_weight_bytes': index['metadata']['total_size'], 'files': pins}


def operational_env(environment):
    env = dict(environment)
    env.update({'OAT_ZERO_REPO_ROOT': str(ROOT), 'MAXENT_GRPO_ROOT': str(ROOT),
                'MAXENT_GRPO_VAR_ROOT': str(ROOT / 'var'), 'CUDA_HOME': str(ROOT / 'var/cuda124_toolkit'),
                'OAT_ZERO_PYTHON': str(PYTHON), 'OAT_ZERO_PYTHON_LIB_DIR': str(PYTHON.parent.parent / 'lib'),
                'PYTHONDONTWRITEBYTECODE': '1', 'OAT_ZERO_REQUIRE_EXISTING_DATA': '1'})
    return dict(sorted(env.items()))


def job_command(plan, row, *, benchmark=False):
    name = 'e124-7b-systems' if benchmark else f"e124-l{row['level']}-{row['domain'][:6]}-{'rm' if row['arm']=='replay_maxrl' else 'm'}-s70"
    env = row['environment']
    require(all(',' not in str(k) and ',' not in str(v) and '\n' not in str(v) for k, v in env.items()), 'unsafe Slurm export value')
    comment = 'e124:' + plan['plan_sha256'][:16] + ':' + ('systems' if benchmark else row['cell_id'])
    script = ART / 'systems.slurm' if benchmark else Path(plan['snapshot']['root']) / 'ops/slurm/train_node302.slurm'
    return ['sbatch', '--parsable', '--hold', '--job-name=' + name,
            '--export=ALL,' + ','.join(k + '=' + v for k, v in env.items()),
            '--partition=lowprio', '--account=mltheory', '--nodelist=' + POOL,
            '--exclude=' + PVL, '--gres=gpu:a6000:1', '--mem=256G', '--cpus-per-task=8',
            '--nodes=1', '--ntasks=1', '--time=3-00:00:00', '--nice=100', '--requeue',
            '--chdir=' + str(ROOT), '--comment=' + comment,
            '--output=' + str(ART / 'logs/%x-%j.out'), '--error=' + str(ART / 'logs/%x-%j.err'), str(script)]



def preflight():
    require(not (ART / 'input_preflight.json').exists(), 'input preflight already exists')
    sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
    import e124_qwen7b_recipes as recipes
    proof = {'schema': 'e124_input_preflight_v1', 'at': now(),
             'recipe_sha256': digest(ROOT / 'ops/exp_scaling/e124_qwen7b_recipes.py'),
             'admission': recipes.dataset_admission(), 'model': model_identity()}
    proof['sha256'] = seal(proof)
    write(ART / 'input_preflight.json', proof, new=True)
    return {'status': 'input_preflight_verified', 'path': str(ART / 'input_preflight.json')}

def prepare():
    require(not PLAN.exists() and not TX.exists() and not LEDGER.exists(), 'E124 already prepared or claimed')
    sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
    import e124_qwen7b_recipes as recipes
    import benchmark_e124_qwen7b as benchmark
    require((ROOT / 'var/cuda124_toolkit/bin/nvcc').is_file(), 'real CUDA compiler missing')
    require(PROTOCOL.is_file(), 'prospective E124 protocol missing')
    if (ART / 'input_preflight.json').is_file():
        inputs = read(ART / 'input_preflight.json')
        require(seal({k: v for k, v in inputs.items() if k != 'sha256'}) == inputs['sha256'], 'input proof changed')
        require(inputs['recipe_sha256'] == digest(ROOT / 'ops/exp_scaling/e124_qwen7b_recipes.py'), 'recipe changed after input proof')
        admission, model = inputs['admission'], inputs['model']
        for path, expected in admission['files_sha256'].items():
            require(digest(path) == expected, 'admitted input changed after preflight: ' + path)
        for path, expected in model['files'].items():
            stat = Path(path).stat()
            require(stat.st_size == expected['size'] and stat.st_mtime_ns == expected['mtime_ns'], 'pretrained model changed after preflight')
    else:
        admission = recipes.dataset_admission()
        model = model_identity()
    snapshot = runtime_description()
    cells = recipes.build_cells(snapshot['root'], MODEL, gpu_class='a6000')
    require(len(cells) == 30, 'exactly thirty cells required')
    for row in cells:
        row['cell_id'] = f"l{row['level']}-{row['domain']}-{row['arm']}-s70"
        row['model_size'] = '7b'
        row['environment'] = operational_env(row['environment'])
        row['environment_sha256'] = seal(row['environment'])
        require(not Path(row['run_dir']).exists(), 'fresh science namespace required')
    require(len({r['cell_id'] for r in cells}) == len({r['run_dir'] for r in cells}) == 30, 'duplicate scientific cell or output')
    plan = {'schema': 'e124_qwen7b_three_level_plan_v1', 'created_at': now(), 'root': str(ROOT),
            'model': model, 'model_revision': MODEL.name, 'snapshot': snapshot,
            'protocol': str(PROTOCOL), 'protocol_sha256': digest(PROTOCOL), 'admission': admission,
            'cells': cells, 'target_steps': 3072, 'passes': 8, 'seed': 70,
            'benchmark_cells': [{'cell_id': 'systems', 'model_size': '7b', 'run_dir': str(ART / 'systems')}],
            'systems_gpu_class': 'a6000', 'max_active': 1, 'checkpoint_peak_reserve_gib': 220,
            'shared_headroom_gib': 64, 'max_retries_per_cell': 64,
            'deadline_utc': (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()}
    plan['plan_sha256'] = seal(plan)
    publish_snapshot(snapshot)
    write(PLAN, plan, new=True)
    benchmark.prepare(PLAN, ART / 'systems')
    benchmark_plan = ART / 'systems/plan.json'
    require(benchmark_plan.is_file(), 'systems plan was not generated')
    control = Path(snapshot['root']) / 'control'
    (ART / 'logs').mkdir(exist_ok=True)
    (ART / 'systems.slurm').write_text('#!/bin/bash\nset -euo pipefail\n' +
        'export OAT_ZERO_REPO_ROOT=' + shlex.quote(str(ROOT)) + '\n' +
        'export MAXENT_GRPO_ROOT=' + shlex.quote(str(ROOT)) + '\n' +
        'export CUDA_HOME=' + shlex.quote(str(ROOT / 'var/cuda124_toolkit')) + '\n' +
        'cd ' + shlex.quote(str(ROOT)) + '\nexec ' + shlex.join([str(PYTHON), str(control / 'benchmark_e124_qwen7b.py'), 'run-suite', '--plan', str(benchmark_plan)]) + '\n')
    (ART / 'controller.slurm').write_text('#!/bin/bash\nset -euo pipefail\nexport PATH=/usr/bin:/bin\n' +
        'export PYTHONDONTWRITEBYTECODE=1\nexport OAT_ZERO_REPO_ROOT=' + shlex.quote(str(ROOT)) + '\ncd ' + shlex.quote(str(ROOT)) + '\n' +
        'exec /usr/local/anaconda3/2024.02/bin/python3 ' + shlex.quote(str(control / SOURCE.name)) + ' watch\n')
    tx = {'schema': 'e124_qwen7b_three_level_transaction_v1', 'plan_sha256': plan['plan_sha256'],
          'status': 'prepared', 'created_at': now(), 'rows': {}, 'events': [],
          'auxiliary_pins': {str(p): digest(p) for p in (benchmark_plan, ART / 'systems.slurm', ART / 'controller.slurm')}}
    write(TX, tx, new=True)
    write(LEDGER, {'schema': 'e124_qwen7b_three_level_jobs_v1', 'plan': str(PLAN), 'plan_sha256': plan['plan_sha256'],
                   'model_size': '7b', 'target_steps': 3072, 'passes': 8, 'planned_runs': cells,
                   'runs': [], 'status': 'prepared', 'released': False}, new=True)
    return {'status': 'prepared', 'plan': str(PLAN), 'plan_sha256': plan['plan_sha256'], 'cells': 30}


def verify(plan, tx, *, full=False):
    require(seal({k: v for k, v in plan.items() if k != 'plan_sha256'}) == plan['plan_sha256'] == tx['plan_sha256'], 'plan identity changed')
    require(digest(PROTOCOL) == plan['protocol_sha256'], 'protocol changed')
    for path, expected in plan['admission']['files_sha256'].items():
        require(digest(path) == expected, 'frozen dataset/admission input changed: ' + path)
    for directory, expected in plan['admission']['directory_files'].items():
        actual = sorted(str(p) for p in Path(directory).rglob('*') if p.is_file())
        require(actual == sorted(expected), 'frozen dataset inventory changed: ' + directory)
    for path, expected in tx['auxiliary_pins'].items():
        require(digest(path) == expected, 'launch input changed: ' + path)
    for path, pin in plan['model']['files'].items():
        stat = Path(path).stat()
        require(stat.st_size == pin['size'] and stat.st_mtime_ns == pin['mtime_ns'] and str(Path(path).resolve()) == pin['resolved'], 'model file identity changed: ' + path)
        if full:
            require(digest(path) == pin['sha256'], 'model content changed: ' + path)
    for relative, pin in plan['snapshot']['inventory'].items():
        require(digest(Path(plan['snapshot']['root']) / relative) == pin['sha256'], 'frozen runtime changed: ' + relative)


def event(tx, label, **values):
    tx['events'].append({'at': now(), 'event': label, **values})
    write(TX, tx)


def audit_job(plan, row, item, *, held=None):
    record = show(item['job_id'])
    f = fields(record)
    require(f.get('JobId') == str(item['job_id']), 'scheduler returned a different job')
    require(f.get('UserId', '').endswith(f'({os.getuid()})'), 'job ownership changed')
    expected_name = next(x.split('=', 1)[1] for x in item['command'] if x.startswith('--job-name='))
    require(f.get('JobName') == expected_name, 'job name changed')
    expected = {'Account': 'mltheory', 'Partition': 'lowprio', 'MinMemoryNode': '256G',
                'NumCPUs': '8', 'CPUs/Task': '8', 'TresPerNode': 'gres/gpu:a6000:1',
                'TimeLimit': '3-00:00:00', 'Nice': '100', 'Requeue': '1', 'Dependency': '(null)',
                'Comment': next(x.split('=', 1)[1] for x in item['command'] if x.startswith('--comment='))}
    for key, value in expected.items():
        require(f.get(key) == value, f"owned job {item['job_id']} changed {key}: {f.get(key)}")
    require(set(command(['scontrol', 'show', 'hostnames', f['ReqNodeList']]).split()) == set(POOL.split(',')), 'node pool changed')
    require(f.get('ExcNodeList') == PVL and f.get('WorkDir') == str(ROOT), 'placement or workdir changed')
    require(exports(submit_tokens(record)) == row['environment'], 'scientific/runtime exports changed')
    require(submit_tokens(record)[-1] == item['command'][-1], 'entry point changed')
    if held is True:
        require(f['JobState'] == 'PENDING' and f['Reason'] in USER_HELD_REASONS and f['Priority'] == '0', 'exact owned user hold required')
        if f['Reason'] != 'JobHeldUser':
            previous = item.get('requeue_intent', {})
            require(item.get('status') in ('held_retry', 'release_intent')
                    and isinstance(previous.get('restarts_before'), int)
                    and int(f.get('Restarts', '-1')) == previous['restarts_before'] + 1,
                    'held requeue lacks exact owned recovery provenance')
    elif held is False and f['JobState'] == 'PENDING':
        require(f['Reason'] not in USER_HELD_REASONS and f['Priority'] != '0', 'released job unexpectedly held')
    return f


def systems_row(plan):
    return {'cell_id': 'systems', 'model_size': '7b', 'run_dir': str(ART / 'systems'),
            'environment': operational_env({'OAT_ZERO_REPO_ROOT': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'})}


def stage_one(plan, tx, row, *, benchmark=False):
    key = row['cell_id']
    old = tx['rows'].get(key)
    if old:
        require(old.get('job_id') is not None, 'unresolved submission intent; never blindly resubmit ' + key)
        audit = audit_job(plan, row, old, held=True)
        old.update(status='held', held_audit=audit)
        event(tx, 'held_submission_reconciled', cell=key)
        return
    args = job_command(plan, row, benchmark=benchmark)
    tx['rows'][key] = {'status': 'submit_intent', 'command': args, 'submitted_at': now(), 'retries': 0}
    event(tx, 'held_submit_intent', cell=key)
    output = command(args).strip()
    require(re.fullmatch(r'\d+(;[^\s]+)?', output) is not None, 'ambiguous submission response; inspect durable intent')
    item = tx['rows'][key]
    item['job_id'] = int(output.split(';')[0])
    item['status'] = 'held_unverified'
    event(tx, 'scheduler_id_recorded', cell=key, job_id=item['job_id'])
    audit = audit_job(plan, row, item, held=True)
    item.update(status='held', held_audit=audit)
    event(tx, 'held_audit_passed', cell=key)


def sync_ledger(plan, tx):
    runs = []
    for row in plan['cells']:
        item = tx['rows'].get(row['cell_id'], {})
        if item.get('job_id'):
            runs.append({k: row[k] for k in ('cell_id', 'level', 'domain', 'arm', 'seed', 'run_dir', 'run_stamp', 'model_size')}
                        | {'job_id': item['job_id'], 'controller_status': item['status']})
    write(LEDGER, {'schema': 'e124_qwen7b_three_level_jobs_v1', 'plan': str(PLAN), 'plan_sha256': plan['plan_sha256'],
                   'model_size': '7b', 'target_steps': 3072, 'passes': 8, 'planned_cells': 30, 'runs': runs,
                   'status': tx['status'], 'released': any(r['controller_status'] not in ('held', 'held_unverified') for r in runs)})


def stage(plan, tx):
    verify(plan, tx)
    reject_untracked_jobs(tx)
    stage_one(plan, tx, systems_row(plan), benchmark=True)
    for row in plan['cells']:
        stage_one(plan, tx, row)
        sync_ledger(plan, tx)
    tx['status'] = 'staged'
    event(tx, 'all_31_gpu_jobs_held_and_audited')
    sync_ledger(plan, tx)
    return {'status': 'staged', 'systems_job_id': tx['rows']['systems']['job_id'],
            'science_job_ids': [tx['rows'][r['cell_id']]['job_id'] for r in plan['cells']]}



def reject_untracked_jobs(tx):
    known = {int(v['job_id']) for v in tx['rows'].values() if v.get('job_id')}
    if tx.get('controller', {}).get('job_id'):
        known.add(int(tx['controller']['job_id']))
    for line in command(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%j']).splitlines():
        jid, name = line.split('|', 1)
        if name.startswith('e124-'):
            require(jid.isdigit() and int(jid) in known, 'untracked E124 scheduler job; reconcile before mutation')


def scheduler_state(job_id):
    lines = command(['squeue', '-h', '-j', str(job_id), '-o', '%i|%T|%r']).splitlines()
    matches = [x.split('|', 2) for x in lines if x.split('|', 1)[0] == str(job_id)]
    require(len(matches) <= 1, 'duplicate scheduler identity')
    if matches:
        return {'state': matches[0][1], 'reason': matches[0][2], 'inactive': False}
    output = command(['sacct', '-n', '-X', '-j', str(job_id), '--format=JobIDRaw,State,ExitCode', '-P'])
    records = [line.split('|') for line in output.splitlines() if line.split('|')[0] == str(job_id)]
    require(records, 'scheduler accounting is not visible yet')
    latest = records[-1]
    state = latest[1].split()[0].rstrip('+')
    require(state in TERMINAL, 'inactive job has no terminal accounting record yet')
    return {'state': state, 'exit_code': latest[2], 'inactive': True}


def other_writers(row, own_id):
    target = str(Path(row['run_dir']).resolve())
    found = []
    queue = command(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T|%r'])
    for line in queue.splitlines():
        jid, state, reason = line.split('|', 2)
        if not jid.isdigit() or int(jid) == own_id or (state == 'PENDING' and reason in USER_HELD_REASONS):
            continue
        tokens = submit_tokens(show(int(jid)))
        if not any(x.startswith('--export=ALL,') for x in tokens):
            continue
        env = exports(tokens)
        if env.get('SAVE_PATH') and str(Path(env['SAVE_PATH']).resolve()) == target:
            found.append(int(jid))
    require(not found, 'other active or released-pending writer for cell: ' + str(found))


def release_owned(plan, tx, row, item, *, retry=False):
    if item['status'] == 'release_intent':
        observed = scheduler_state(item['job_id'])
        if observed['state'] in LIVE and not (observed['state'] == 'PENDING' and observed['reason'] in USER_HELD_REASONS):
            audit_job(plan, row, item, held=False)
            item['status'] = 'released'
            event(tx, 'release_acknowledged_after_observation', cell=row['cell_id'])
            return True
        require(not observed['inactive'], 'release outcome requires accounting review')
    audit_job(plan, row, item, held=True)
    other_writers(row, item['job_id'])
    if not retry and item['status'] != 'release_intent' and row['cell_id'] != 'systems':
        require(not Path(row['run_dir']).exists(), 'fresh scientific namespace acquired artifacts before first release')
    admission = storage_report(plan, tx, releasing=row['cell_id'])
    if not admission['allowed']:
        tx['status'] = 'waiting_storage'
        write(ART / 'release_storage_wait.json', {'at': now(), 'cell': row['cell_id'], 'storage': admission})
        write(TX, tx)
        return False
    item['status'] = 'release_intent'
    event(tx, 'release_intent', cell=row['cell_id'], retry=retry)
    command(['scontrol', 'release', str(item['job_id'])])
    audit_job(plan, row, item, held=False)
    item.update(status='released', released_at=now())
    event(tx, 'owned_job_released', cell=row['cell_id'])
    return True


def metadata_scalars(raw, keys):
    result = {key: [] for key in keys}
    pending, previous, bank = None, None, False
    memo = {}
    for op, value, _ in pickletools.genops(raw):
        if op.name in {'BINPUT', 'LONG_BINPUT', 'PUT', 'MEMOIZE'}:
            memo[len(memo) if op.name == 'MEMOIZE' else int(value)] = previous
            continue
        semantic = value if op.name in {'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8', 'STRING'} else memo.get(int(value)) if op.name in {'GET', 'BINGET', 'LONG_BINGET'} else None
        if pending is not None:
            require(op.name in {'BININT', 'BININT1', 'BININT2', 'INT', 'LONG', 'LONG1', 'LONG4', 'BINFLOAT', 'FLOAT'}
                    and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), 'invalid checkpoint scalar')
            result[pending].append(value)
            pending = None
        previous = semantic
        bank |= semantic == 'online_canonical_bank_state'
        if isinstance(semantic, str) and semantic in result:
            pending = semantic
    require(pending is None, 'truncated checkpoint scalar')
    return result, bank


def checkpoint(row):
    root = Path(row['run_dir']).resolve()
    grid = int(row['environment']['OAT_ZERO_RESUME_STEPS'])
    require(grid == 96, 'E124 checkpoint cadence changed')
    candidates = []
    for path in root.glob('debug*/checkpoints/step_*'):
        if re.fullmatch(r'step_\d+', path.name) and path.is_dir():
            candidates.append((int(path.name[5:]), path))
    errors = []
    for step, path in sorted(candidates, reverse=True):
        try:
            require(path.resolve().is_relative_to(root) and 0 < step <= 3072 and step % grid == 0, 'checkpoint outside registered grid/path')
            files, model, optimizer = [], False, False
            for archive_path in sorted(path.glob('*.pt')):
                require(archive_path.resolve().is_relative_to(root), 'checkpoint archive escapes run')
                with zipfile.ZipFile(archive_path) as archive:
                    names = [n for n in archive.namelist() if n == 'data.pkl' or n.endswith('/data.pkl')]
                    require(len(names) == 1 and archive.getinfo(names[0]).file_size <= 16 * 1024**2, 'invalid checkpoint metadata inventory')
                    raw = archive.read(names[0])
                if archive_path.name.endswith('_model_states.pt'):
                    keys = ('global_steps', 'global_step', 'policy_sgd_step', 'prompt_batches_consumed_total')
                    values, bank = metadata_scalars(raw, keys)
                    require(bank and all(values[k] and all(v == step for v in values[k]) for k in keys), 'model/prompt/replay counters differ')
                    model = True
                elif archive_path.name.endswith('_optim_states.pt'):
                    values, _ = metadata_scalars(raw, ('step',))
                    require(values['step'] and all(v == step for v in values['step']), 'optimizer step differs')
                    optimizer = True
                stat = archive_path.stat()
                files.append({'name': archive_path.name, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
            require(model and optimizer, 'complete model and optimizer required')
            return {'path': str(path), 'step': step, 'files': files, 'rejected': errors}
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    raise RuntimeError('no validated checkpoint: ' + str(errors))


def completed(row):
    root = Path(row['run_dir']).resolve()
    marker = root / 'TRAINING_COMPLETE.json'
    if not marker.is_file():
        return None
    data = read(marker)
    export = Path(data.get('terminal_export', '')).resolve()
    step = data.get('terminal_step')
    require(data.get('schema') == 'oat_zero_training_complete_v1' and isinstance(step, int) and 3072 <= step <= 3073,
            'invalid terminal completion receipt')
    require(export.is_relative_to(root) and export.parent.name == 'saved_models' and export.name == f'step_{step:05d}', 'terminal export path differs')
    index_path = export / 'model.safetensors.index.json'
    if index_path.is_file():
        index = read(index_path)
        shards = sorted(set(index['weight_map'].values()))
        require(shards and all(Path(name).name == name for name in shards), 'terminal shard inventory invalid')
        require(all((export / name).is_file() and (export / name).stat().st_size > 0 for name in shards), 'terminal shard missing')
        expected = read(MODEL / 'model.safetensors.index.json')['metadata']['total_size']
        require(index['metadata']['total_size'] == expected and sum((export / name).stat().st_size for name in shards) >= expected, 'terminal model size differs')
    else:
        weights = export / 'model.safetensors'
        require(weights.is_file() and weights.stat().st_size >= 15231233024, 'complete terminal weights missing')
    return {'step': step, 'marker_sha256': digest(marker), 'terminal_export': str(export)}


def recover_timeout(plan, tx, row, item, observed):
    """Never repeat a requeue after an uncertain acknowledgement."""
    jid = item['job_id']
    if item['status'] == 'held_retry':
        release_owned(plan, tx, row, item, retry=True)
        return
    if item['status'] == 'requeue_intent':
        record = audit_job(plan, row, item, held=None)
        previous = item['requeue_intent']
        require(int(record.get('Restarts', '-1')) == previous['restarts_before'] + 1, 'unconfirmed requeue transition; manual review required')
        require(record['JobState'] == 'PENDING' and record['Reason'] in USER_HELD_REASONS and record['Priority'] == '0', 'requeue did not retain exact owned hold')
        detail = checkpoint(row)
        require(detail['path'] == previous['checkpoint']['path'] and detail['files'] == previous['checkpoint']['files'], 'checkpoint changed during inactive retry')
        item['retries'] = previous['retry_number']
        item['last_resume_step'] = detail['step']
        command(['scontrol', 'hold', str(jid)])
        item['status'] = 'held_retry'
        event(tx, 'same_id_retry_hold_verified', cell=row['cell_id'])
        release_owned(plan, tx, row, item, retry=True)
        return
    require(observed['inactive'] and observed['state'] in ('TIMEOUT', 'PREEMPTED'), 'only accounted inactive timeout/preemption may retry')
    require(item.get('retries', 0) < plan['max_retries_per_cell'], 'bounded retry cap reached')
    require(completed(row) is None, 'completed cell must not retry')
    other_writers(row, jid)
    detail = checkpoint(row)
    require(detail['step'] > item.get('last_resume_step', 0), 'no durable progress since previous attempt')
    record = audit_job(plan, row, item, held=None)
    item['requeue_intent'] = {'at': now(), 'checkpoint': detail,
                              'restarts_before': int(record.get('Restarts', '0')), 'retry_number': item.get('retries', 0) + 1}
    item['status'] = 'requeue_intent'
    event(tx, 'same_id_requeue_intent', cell=row['cell_id'])
    command(['scontrol', 'requeuehold', str(jid)])


def storage_report(plan, tx, *, releasing=None):
    sys.path.insert(0, str(Path(plan['snapshot']['root']) / 'control'))
    import e124_storage_admission as storage
    own = []
    for row in [systems_row(plan), *plan['cells']]:
        item = tx['rows'].get(row['cell_id'], {})
        if row['cell_id'] != releasing and item.get('status') in ('released', 'release_intent', 'requeue_intent', 'held_retry'):
            own.append(row | {'job_id': item['job_id']})
    return storage.storage_report(plan, own_live_runs=own)


def qualification(plan):
    profile_path = ART / 'systems/qualified_profile.json'
    require(profile_path.is_file(), 'systems job completed without qualification')
    profile = read(profile_path)
    require(profile.get('status') == 'passed', 'systems qualification did not pass')
    require(profile.get('manifest_sha256') == digest(PLAN), 'systems evidence is not bound to this exact manifest')
    require(profile.get('plan_sha256') == read(ART / 'systems/plan.json')['plan_sha256'], 'systems benchmark identity differs')
    for path, expected in profile['evidence']['files'].items():
        require(digest(path) == expected, 'systems qualification evidence changed: ' + path)
    return {'path': str(profile_path), 'sha256': digest(profile_path)}


def observe_pass(plan, tx):
    verify(plan, tx)
    reject_untracked_jobs(tx)
    require(datetime.now(timezone.utc) < datetime.fromisoformat(plan['deadline_utc']), 'recorded controller deadline reached')
    summary = []
    for row in [systems_row(plan), *plan['cells']]:
        item = tx['rows'][row['cell_id']]
        if item['status'] in ('held', 'completed', 'qualified'):
            continue
        if item['status'] == 'release_intent':
            if not release_owned(plan, tx, row, item):
                summary.append({'cell': row['cell_id'], 'job_id': item['job_id'], 'state': 'PENDING', 'reason': 'owned_hold_waiting_storage'})
                continue
        observed = scheduler_state(item['job_id'])
        summary.append({'cell': row['cell_id'], 'job_id': item['job_id'], **observed})
        if row['cell_id'] == 'systems':
            if observed['inactive']:
                require(observed['state'] == 'COMPLETED' and observed['exit_code'] == '0:0', 'systems suite stopped; inspect evidence before further launch')
                tx['qualification'] = qualification(plan)
                item['status'] = 'qualified'
                event(tx, 'systems_profile_qualified')
        elif item['status'] in ('requeue_intent', 'held_retry'):
            recover_timeout(plan, tx, row, item, observed)
        elif observed['inactive']:
            if observed['state'] == 'COMPLETED' and observed['exit_code'] == '0:0':
                endpoint = completed(row)
                require(endpoint is not None, 'scheduler completion without full scientific endpoint')
                item.update(status='completed', endpoint=endpoint)
                event(tx, 'scientific_cell_completed', cell=row['cell_id'])
            else:
                recover_timeout(plan, tx, row, item, observed)
        else:
            audit_job(plan, row, item, held=False)
    if all(tx['rows'][r['cell_id']]['status'] == 'completed' for r in plan['cells']):
        tx['status'] = 'completed'
        event(tx, 'all_thirty_cells_completed')
        sync_ledger(plan, tx)
        return True
    ready = tx['rows']['systems']['status'] == 'qualified'
    if ready:
        require(digest(tx['qualification']['path']) == tx['qualification']['sha256'], 'qualification receipt changed')
    candidates = [r for r in (plan['cells'] if ready else [systems_row(plan)]) if tx['rows'][r['cell_id']]['status'] == 'held']
    report = storage_report(plan, tx)
    if candidates and report['allowed']:
        row = candidates[0]
        if release_owned(plan, tx, row, tx['rows'][row['cell_id']]):
            tx['status'] = 'training' if ready else 'systems_queued'
    elif candidates:
        tx['status'] = 'waiting_storage' if not ready else 'staged_training'
    write(TX, tx)
    write(ART / 'status.json', {'schema': 'e124_live_status_v1', 'at': now(), 'status': tx['status'],
                               'qualified': ready, 'completed': sum(tx['rows'][r['cell_id']]['status'] == 'completed' for r in plan['cells']),
                               'held': sum(tx['rows'][r['cell_id']]['status'] == 'held' for r in plan['cells']),
                               'storage': report, 'observations': summary})
    sync_ledger(plan, tx)
    return False


def start_controller(plan, tx):
    verify(plan, tx)
    require(len(tx['rows']) == 31 and all(r['status'] == 'held' for r in tx['rows'].values()), 'all systems/science jobs must be audited held before controller start')
    require('controller' not in tx, 'existing or ambiguous CPU controller submission; inspect it')
    args = ['sbatch', '--parsable', '--hold', '--job-name=e124-controller', '--account=mltheory', '--partition=lowprio',
            '--nodelist=node916,node917', '--nodes=1', '--ntasks=1', '--gres=none', '--cpus-per-task=1', '--mem=2G', '--time=1-01:10:00', '--requeue',
            '--comment=e124-controller:' + plan['plan_sha256'][:16], '--chdir=' + str(ROOT),
            '--output=' + str(ART / 'logs/controller-%j.out'), '--error=' + str(ART / 'logs/controller-%j.err'), str(ART / 'controller.slurm')]
    tx['controller'] = {'status': 'submit_intent', 'command': args, 'at': now()}
    event(tx, 'controller_submit_intent')
    out = command(args).strip()
    require(re.fullmatch(r'\d+(;[^\s]+)?', out) is not None, 'ambiguous controller submission')
    jid = int(out.split(';')[0])
    tx['controller'].update(job_id=jid, status='held_unverified')
    event(tx, 'controller_id_recorded')
    f = fields(show(jid))
    require(f.get('JobState') == 'PENDING' and f.get('Reason') == 'JobHeldUser' and f.get('Priority') == '0', 'controller is not held')
    require(f.get('Account') == 'mltheory' and f.get('Partition') == 'lowprio', 'controller placement differs')
    require(set(command(['scontrol', 'show', 'hostnames', f['ReqNodeList']]).split()) == {'node916', 'node917'}, 'controller node pool differs')
    require('gres/gpu' not in f.get('ReqTRES', '') and f.get('MinMemoryNode') == '2G', 'CPU-only controller resources differ')
    require(f.get('TimeLimit') == '1-01:10:00' and f.get('UserId', '').endswith(f'({os.getuid()})'), 'controller ownership/time differs')
    require(f.get('JobName') == 'e124-controller' and f.get('JobId') == str(jid), 'controller scheduler identity differs')
    tx['controller']['status'] = 'release_intent'
    event(tx, 'controller_release_intent')
    command(['scontrol', 'release', str(jid)])
    tx['controller']['status'] = 'released'
    event(tx, 'controller_released')
    return {'status': 'controller_queued', 'job_id': jid, 'science_cells': 30}


def watch():
    started = time.monotonic()
    with (ART / 'supervisor.lock').open('a') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        errors = 0
        while True:
            try:
                with lock():
                    plan, tx = read(PLAN), read(TX)
                    require(str(tx['controller']['job_id']) == os.environ.get('SLURM_JOB_ID'), 'watcher must run in its exact recorded CPU allocation')
                    if tx.get('status') == 'needs_review':
                        return 2
                    write(ART / 'controller_progress.json', {'at': now(), 'job_id': int(os.environ['SLURM_JOB_ID']),
                                                           'phase': 'checking_frozen_inputs_and_scheduler', 'ready': False})
                    done = observe_pass(plan, tx)
                    write(ART / 'controller_ready.json', {'at': now(), 'job_id': int(os.environ['SLURM_JOB_ID']),
                                                         'plan_sha256': plan['plan_sha256'], 'status': tx['status'], 'singleton': True})
                    if done:
                        return 0
                    if time.monotonic() - started > 86400:
                        tx['controller']['self_requeues'] = tx['controller'].get('self_requeues', 0) + 1
                        require(tx['controller']['self_requeues'] <= 90, 'CPU controller requeue cap reached')
                        event(tx, 'controller_self_requeue_intent')
                        command(['scontrol', 'requeue', os.environ['SLURM_JOB_ID']])
                        return 0
                    errors = 0
            except Exception as exc:
                errors += 1
                write(ART / 'controller_error.json', {'at': now(), 'consecutive_errors': errors, 'error': repr(exc)})
                if errors >= 8:
                    with lock():
                        tx = read(TX)
                        tx.update(status='needs_review', error=repr(exc))
                        event(tx, 'controller_stopped_for_review')
                    return 2
            time.sleep(60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'prepare', 'stage', 'status', 'watch', 'start-controller'])
    args = parser.parse_args(argv)
    if args.action == 'status':
        print(json.dumps(read(TX), indent=2)); return 0
    if args.action == 'watch':
        return watch()
    with lock():
        if args.action == 'preflight':
            result = preflight()
        elif args.action == 'prepare':
            result = prepare()
        elif args.action == 'stage':
            result = stage(read(PLAN), read(TX))
        else:
            result = start_controller(read(PLAN), read(TX))
    print(json.dumps(result, indent=2)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
