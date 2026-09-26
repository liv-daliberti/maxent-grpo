#!/usr/bin/env python3
"""Prepare and submit isolated, bounded Level 4/5 calibration arrays.

Preparation is local. Submission records an exclusive durable intent before
calling sbatch once; an interrupted/ambiguous submission must be reconciled
instead of retried. No existing experiment jobs are modified.
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
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
PYTHON = ROOT / 'var/seed_paper_eval/paper310/bin/python'
DOMAINS = ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry')
LEVEL_MODELS = {'level4': '7b', 'level5': '14b'}
SCHEMA = 'modebench-scale-array-v1'
HARDWARE = {'partition': 'lowprio', 'account': 'mltheory',
            'nodes': 'node205,node206,node207,node208', 'gres': 'gpu:a6000:1',
            'cpus': 6, 'memory': '64G', 'time': '08:00:00',
            'tensor_parallel_size': 1, 'attention_backend': 'XFORMERS',
            'vllm_version': '0.8.4', 'engine': 'V0', 'dtype': 'float16'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_new(path, payload):
    """Create an immutable JSON record; concurrent writers cannot replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def evaluator():
    sys.path.insert(0, str(ROOT / 'ops'))
    import evaluate_modebench_scale
    return evaluate_modebench_scale


def checkpoint_identity(path, label):
    require(path is not None, f'--model-{label} is required for this level')
    path = Path(path).resolve()
    require(path.is_dir() and (path / 'config.json').is_file(), f'local {label} checkpoint missing: {path}')
    config = read(path / 'config.json')
    expected = {'7b': (3584, 28), '14b': (5120, 48)}[label]
    require(config.get('architectures') == ['Qwen2ForCausalLM']
            and (config.get('hidden_size'), config.get('num_hidden_layers')) == expected,
            f'checkpoint architecture does not match Qwen2.5 {label}')
    index = path / 'model.safetensors.index.json'
    require(index.is_file(), f'shard index missing: {path}')
    names = sorted(set(read(index)['weight_map'].values()))
    require(names and all(Path(name).name == name for name in names), 'invalid model shard index')
    require(all((path / name).is_file() and (path / name).stat().st_size > 0 for name in names), 'model shard missing')
    require((path / 'tokenizer_config.json').is_file() and (path / 'tokenizer.json').is_file(), 'tokenizer missing')
    return {'label': label, 'path': str(path), 'revision': path.name,
            'configuration_sha256': {p.name: digest(p) for p in sorted(path.glob('*.json'))},
            'weights': [{'name': name, 'resolved_path': str((path / name).resolve()),
                         'bytes': (path / name).stat().st_size,
                         'mtime_ns': (path / name).stat().st_mtime_ns} for name in names],
            'weight_identity_method': 'complete indexed shard inventory; resolved path, bytes and mtime'}


def validate_inputs(inputs, phase):
    """Each input is a level/domain/tier and an explicit JSONL source path."""
    require(isinstance(inputs, list) and inputs, 'nonempty input manifest required')
    seen = set()
    for item in inputs:
        require(item.get('level') in LEVEL_MODELS and item.get('domain') in DOMAINS, 'unknown level/domain')
        tier = item.get('tier')
        require(type(tier) is int and tier in range(4) if phase == 'dev' else tier is None,
                'development needs tier 0..3; confirmation must omit tier')
        key = (item['level'], item['domain'], tier)
        require(key not in seen, 'duplicate input cell')
        seen.add(key)
        require(Path(item['rows_jsonl']).is_file(), f"source missing: {item['rows_jsonl']}")
    levels = {x['level'] for x in inputs}
    tiers = range(4) if phase == 'dev' else (None,)
    require(seen == {(level, domain, tier) for level in levels for domain in DOMAINS for tier in tiers},
            'each included level needs all five domains and every registered tier')


def draw_labels(level, phase):
    require(level in LEVEL_MODELS and phase in ('dev', 'eval'), 'unknown draw label cell')
    start = (6548000 if level == 'level4' else 6558000) + (1000 if phase == 'eval' else 0)
    return list(range(start, start + 4))


def discover_inputs(data_root, levels, phase):
    base = Path(data_root).resolve()
    require(levels and set(levels) <= set(LEVEL_MODELS) and len(levels) == len(set(levels)), 'invalid levels')
    return [{'level': level, 'domain': domain, **({'tier': tier} if phase == 'dev' else {}),
             'rows_jsonl': str(base / level / ('pools' if phase == 'dev' else 'dataset') / domain /
                              (f'difficulty_{tier}.jsonl' if phase == 'dev' else 'eval.jsonl'))}
            for level in levels for domain in DOMAINS for tier in (range(4) if phase == 'dev' else (None,))]


def confirmation_gate(data_root, inputs):
    """Reproduce each passing recipe and authenticate its frozen held-out rows."""
    require(data_root is not None, 'confirmation requires --data-root with frozen datasets and passing recipes')
    base = Path(data_root).resolve()
    sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
    from fit_modebench_scale import fit_domain
    from materialize_modebench_scale import authenticate
    authenticate(base / 'protocol.json')
    e = evaluator()
    pins = {str(base / 'protocol.json'): digest(base / 'protocol.json')}
    for item in inputs:
        level, domain = item['level'], item['domain']
        source = base / level / 'dataset' / domain / 'eval.jsonl'
        require(Path(item['rows_jsonl']).resolve() == source, 'confirmation must use the frozen eval.jsonl')
        identity_path = source.parent / 'identity.json'
        identity = read(identity_path)
        recipe_path = base / level / 'recipes' / (domain + '.json')
        recipe = read(recipe_path)
        require(recipe.get('schema') == 'modebench_scale_development_recipe_v1'
                and recipe.get('development_fit_pass') is True
                and recipe == fit_domain(base, level, domain, publish=False), 'passing reproducible recipe required')
        require(identity.get('schema') == 'modebench_scale_frozen_domain_v1'
                and identity.get('status') == 'frozen_pending_heldout_confirmation'
                and identity.get('level') == level and identity.get('domain') == domain
                and identity.get('recipe_sha256') == digest(recipe_path)
                and identity.get('protocol_sha256') == digest(base / 'protocol.json'), 'frozen dataset identity changed')
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        require(len(rows) == 128 and identity['splits']['eval']['rows'] == len(rows)
                and identity['splits']['eval']['rows_sha256'] == e.sha(rows), 'frozen eval rows changed')
        for evidence, expected in recipe['input_sha256'].items():
            require(digest(evidence) == expected, f'development evidence changed: {evidence}')
            pins[evidence] = expected
        for source_path in (source, identity_path, recipe_path):
            pins[str(source_path)] = digest(source_path)
    return pins


def development_pool_identity(data_root, level, domain, protocol_sha256):
    """Bind each candidate tier to the materializer's verified pool inventory."""
    path = Path(data_root).resolve() / level / 'pools' / domain / 'identity.json'
    identity = read(path)
    require(identity.get('schema') == 'modebench_scale_development_pools_v1'
            and identity.get('status') == 'verified_candidates_pending_model_calibration'
            and identity.get('level') == level and identity.get('domain') == domain
            and identity.get('protocol_sha256') == protocol_sha256,
            'development pool identity differs from registered level/domain/protocol')
    require(isinstance(identity.get('tiers'), dict) and set(identity['tiers']) == {'0', '1', '2', '3'},
            'development pool identity needs exactly four tiers')
    return path, identity


def job_command(plan, cell):
    argv = [plan['python'], str(ROOT / 'ops/evaluate_modebench_scale.py'),
            '--model', plan['models'][cell['model_label']]['path'], '--model-label', cell['model_label'],
            '--tasks-json', cell['tasks'], '--resume']
    if plan['phase'] == 'eval':
        argv.append('--confirm-eval')
    return argv


def validate_dependency_ids(values):
    require(isinstance(values, (list, tuple))
            and all(type(value) is int and value > 0 for value in values)
            and len(values) == len(set(values)),
            'dependencies must be distinct positive integer Slurm array job IDs')
    return sorted(values)


def scheduler_command(plan_path, plan):
    path = Path(plan_path).resolve()
    hw = plan['hardware']
    dependencies = validate_dependency_ids(plan['dependency_ids'])
    command = ['sbatch', '--parsable', '--partition=' + hw['partition'], '--account=' + hw['account'],
            '--nodelist=' + hw['nodes'], '--nodes=1', '--ntasks=1', '--gres=' + hw['gres'],
            '--cpus-per-task=' + str(hw['cpus']), '--mem=' + hw['memory'], '--time=' + hw['time'],
            '--array=0-' + str(len(plan['cells']) - 1) + '%' + str(plan['concurrency']),
            '--job-name=modebench-scale-' + plan['phase'], '--no-requeue',
            '--output=' + str(path.parent / 'logs/%A_%a.out'),
            '--error=' + str(path.parent / 'logs/%A_%a.err'),
            '--chdir=' + str(ROOT), '--export=ALL']
    if dependencies:
        command.append('--dependency=afterany:' + ':'.join(str(job) for job in dependencies))
    return command + [str(path.parent / 'worker.slurm'), str(path)]


def prepare(inputs_path, campaign_root, models, *, phase='dev', concurrency=4, data_root=None,
            levels=('level4', 'level5'), dependency_ids=()):
    require(phase in ('dev', 'eval'), 'phase must be dev or eval')
    require(type(concurrency) is int and 1 <= concurrency <= 4, 'concurrency must be 1..4')
    dependency_ids = validate_dependency_ids(dependency_ids)
    target = Path(campaign_root).resolve()
    require(not target.exists(), 'fresh campaign directory required')
    require(inputs_path is not None or data_root is not None, '--inputs or --data-root is required')
    inputs_path = Path(inputs_path).resolve() if inputs_path is not None else None
    inputs = read(inputs_path) if inputs_path else discover_inputs(data_root, levels, phase)
    validate_inputs(inputs, phase)
    e = evaluator()
    pins = {str(SOURCE): digest(SOURCE),
            str(ROOT / 'ops/repo_env.sh'): digest(ROOT / 'ops/repo_env.sh')}
    if inputs_path:
        pins[str(inputs_path)] = digest(inputs_path)
    pins.update({str((ROOT / p).resolve()): h for p, h in e.code_identity().items()})
    if phase == 'eval':
        pins.update(confirmation_gate(data_root, inputs))
    if data_root is not None:
        protocol_path = Path(data_root).resolve() / 'protocol.json'
        protocol = read(protocol_path)
        require(protocol.get('schema') == 'modebench_scale_protocol_v1', 'scale protocol missing')
        for source, expected in protocol['files_sha256'].items():
            require(digest(source) == expected, f'protocol input changed: {source}')
            pins[source] = expected
        pins[str(protocol_path)] = digest(protocol_path)
        for item in inputs:
            require(protocol['draw_labels'][item['level']][phase] == draw_labels(item['level'], phase),
                    'registered sampling draw labels changed')
    labels = {LEVEL_MODELS[item['level']] for item in inputs}
    identities = {label: checkpoint_identity(models[label], label) for label in sorted(labels)}
    if data_root is not None:
        for label, identity in identities.items():
            require(e.model_identity(Path(identity['path']), label) == protocol['models'][label],
                    f'checkpoint does not match registered protocol: {label}')
    require(PYTHON.is_file(), 'calibration Python runtime missing')
    plan = {'schema': SCHEMA, 'created_at': now(), 'phase': phase, 'interface': e.INTERFACE,
            'python': str(PYTHON), 'concurrency': concurrency, 'hardware': dict(HARDWARE),
            'dependency_ids': dependency_ids,
            'models': identities, 'cells': [], 'immutable_inputs_sha256': pins,
            'data_root': str(Path(data_root).resolve()) if data_root else None,
            'treatment_training_started': False}
    pending = []
    seen_blocks, schedule_hashes = set(), {}
    for level in sorted({item['level'] for item in inputs}):
        for domain in DOMAINS:
            group = sorted([i for i in inputs if i['level'] == level and i['domain'] == domain],
                           key=lambda item: item.get('tier', -1))
            cell_id = f'{level}_{domain}'
            tasks_path = target / 'tasks' / (cell_id + '.json')
            tasks = []
            pool_identity = None
            if data_root is not None and phase == 'dev':
                identity_path, pool_identity = development_pool_identity(
                    data_root, level, domain, pins[str(protocol_path)])
                pins[str(identity_path)] = digest(identity_path)
            for item in group:
                source = Path(item['rows_jsonl']).resolve()
                suffix = '_d' + str(item['tier']) if phase == 'dev' else ''
                result_root = Path(data_root).resolve() / level / 'results' if data_root else target / 'receipts'
                output = ((result_root / 'development' / domain / f"difficulty_{item['tier']}.json")
                          if phase == 'dev' else result_root / 'confirmation' / (domain + '.json')) if data_root else (
                              target / 'receipts' / (cell_id + suffix + '.json'))
                task = {'level': level, 'domain': domain, 'split': phase, 'interface': e.INTERFACE,
                        'rows_jsonl': str(source), 'seeds': draw_labels(level, phase), 'batch_size': 8,
                        'row_limit': 0, 'row_offset': 0,
                        'output': str(output)}
                e.validate_task(task, phase == 'eval')
                rows, _ = e.load_rows(task)
                if pool_identity is not None:
                    expected_path = identity_path.parent / f"difficulty_{item['tier']}.jsonl"
                    require(source == expected_path, 'development source is outside its registered pool')
                    tier_identity = pool_identity['tiers'][str(item['tier'])]
                    require(type(tier_identity.get('rows')) is int and tier_identity['rows'] == len(rows)
                            and tier_identity.get('rows_sha256') == e.sha(rows),
                            'development pool tier row count/hash differs from verified identity')
                schedule = e.frozen.schedule_record(domain, rows, task['seeds'])
                blocks = {base for row in schedule['request_seeds'] for base in row}
                require(not seen_blocks & blocks, 'RNG block collision across calibration tasks; refuse submission')
                seen_blocks.update(blocks)
                schedule_hashes[str(output)] = e.sha(schedule)
                tasks.append(task)
                pins[str(source)] = digest(source)
            cell = {'id': cell_id, 'model_label': LEVEL_MODELS[level], 'tasks': str(tasks_path)}
            cell['command'] = job_command(plan, cell)
            cell['shell_command'] = shlex.join(cell['command'])
            plan['cells'].append(cell)
            pending.append((tasks_path, tasks))
    plan['rng_admission'] = {'policy': e.frozen.POLICY, 'distinct_request_blocks': len(seen_blocks),
                             'distinct_child_seeds': len(seen_blocks) * 8,
                             'task_seed_schedule_sha256': schedule_hashes}
    # Validate all inputs before creating a campaign claim or writable outputs.
    target.mkdir(parents=True, exist_ok=False)
    for directory in ('tasks', 'receipts', 'logs', 'runtime'):
        (target / directory).mkdir()
    for path, tasks in pending:
        atomic_new(path, tasks)
        pins[str(path)] = digest(path)
    script = ('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n'
              'source ops/repo_env.sh\nexport HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
              'export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS OMP_NUM_THREADS=4\n'
              'export PYTHONDONTWRITEBYTECODE=1\nexec ' + shlex.join([str(PYTHON), str(SOURCE), 'worker']) +
              ' --plan "$1" --index "${SLURM_ARRAY_TASK_ID:?}"\n')
    (target / 'worker.slurm').write_text(script)
    pins[str(target / 'worker.slurm')] = digest(target / 'worker.slurm')
    plan['submit_command'] = scheduler_command(target / 'plan.json', plan)
    plan['submit_shell_command'] = shlex.join(plan['submit_command'])
    atomic_new(target / 'plan.json', plan)
    atomic_new(target / 'plan.sha256.json', {'sha256': digest(target / 'plan.json')})
    return plan


def verify(plan_path, *, fresh=False):
    path = Path(plan_path).resolve()
    require(digest(path) == read(path.parent / 'plan.sha256.json')['sha256'], 'plan changed')
    plan = read(path)
    require(plan.get('schema') == SCHEMA and plan['hardware'] == HARDWARE, 'runtime contract changed')
    require(plan['submit_command'] == scheduler_command(path, plan), 'scheduler command changed')
    for source, expected in plan['immutable_inputs_sha256'].items():
        require(digest(source) == expected, f'prepared input changed: {source}')
    for label, expected in plan['models'].items():
        require(checkpoint_identity(expected['path'], label) == expected, f'checkpoint changed: {label}')
    for cell in plan['cells']:
        require(cell['command'] == job_command(plan, cell), 'worker command changed')
        if fresh:
            for task in read(cell['tasks']):
                output = Path(task['output'])
                require(not output.exists() and not Path(str(output) + '.batches').exists(), 'preexisting calibration outputs')
    return plan


def submit(plan_path, runner=subprocess.run):
    path = Path(plan_path).resolve()
    plan = verify(path, fresh=True)
    intent = path.parent / 'submission_intent.json'
    require(not intent.exists(), 'submission already attempted; reconcile its durable intent without resubmitting')
    atomic_new(intent, {'at': now(), 'plan_sha256': digest(path), 'command': plan['submit_command'],
                        'status': 'submission_attempt_started'})
    result = runner(plan['submit_command'], text=True, capture_output=True, check=False, timeout=60)
    output = result.stdout.strip()
    evidence = {'at': now(), 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', output)
    if result.returncode != 0 or match is None:
        atomic_new(path.parent / 'submission_ambiguous.json', evidence)
        raise RuntimeError('sbatch failed or returned an ambiguous job id; durable intent prevents duplicate submission')
    evidence.update(status='submitted', array_job_id=int(match.group(1)),
                    cells=[{'array_index': i, 'cell': cell['id']} for i, cell in enumerate(plan['cells'])])
    atomic_new(path.parent / 'submission_result.json', evidence)
    return evidence


def worker(plan_path, index):
    path = Path(plan_path).resolve()
    plan = verify(path)
    require(type(index) is int and 0 <= index < len(plan['cells']), 'invalid array index')
    require(os.environ.get('VLLM_USE_V1') == '0' and os.environ.get('VLLM_ATTENTION_BACKEND') == 'XFORMERS',
            'worker requires vLLM V0 and XFORMERS')
    import importlib.metadata
    require(importlib.metadata.version('vllm') == HARDWARE['vllm_version'], 'vLLM version changed')
    # An immediately scheduled array can start before sbatch's parent publishes
    # its result. Close that race without another submission attempt.
    result_path = path.parent / 'submission_result.json'
    deadline = time.monotonic() + 60
    while not result_path.exists() and time.monotonic() < deadline:
        time.sleep(1)
    require(result_path.exists(), 'submission result absent; reconcile the durable intent')
    submission = read(result_path)
    require(os.environ.get('SLURM_ARRAY_JOB_ID') == str(submission['array_job_id'])
            and os.environ.get('SLURM_ARRAY_TASK_ID') == str(index), 'worker is outside its submitted array')
    atomic_new(path.parent / 'runtime' / f'{index}.json',
               {'at': now(), 'cell': plan['cells'][index]['id'], 'plan_sha256': digest(path),
                'array_job_id': submission['array_job_id'], 'hostname': os.uname().nodename,
                'command': plan['cells'][index]['command']})
    os.execv(plan['python'], plan['cells'][index]['command'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--inputs', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--levels', choices=tuple(LEVEL_MODELS), nargs='+', default=list(LEVEL_MODELS))
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--model-7b', type=Path)
    p.add_argument('--model-14b', type=Path)
    p.add_argument('--phase', choices=('dev', 'eval'), default='dev')
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--dependency-ids', type=int, nargs='*', default=[])
    for name in ('verify', 'submit', 'worker'):
        p = commands.add_parser(name)
        p.add_argument('--plan', type=Path, required=True)
        if name == 'worker':
            p.add_argument('--index', type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.inputs, args.campaign_root, {'7b': args.model_7b, '14b': args.model_14b},
                         phase=args.phase, concurrency=args.concurrency, data_root=args.data_root, levels=args.levels,
                         dependency_ids=args.dependency_ids)
        print(json.dumps({'status': 'prepared', 'cells': len(result['cells']),
                          'plan': str(args.campaign_root.resolve() / 'plan.json'),
                          'submit_command': result['submit_command']}))
    elif args.command == 'verify':
        result = verify(args.plan)
        print(json.dumps({'status': 'verified', 'cells': len(result['cells'])}))
    elif args.command == 'submit':
        print(json.dumps(submit(args.plan)))
    else:
        worker(args.plan, args.index)


if __name__ == '__main__':
    main()
