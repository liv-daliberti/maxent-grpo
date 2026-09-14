#!/usr/bin/env python3
"""Prepare immutable partial-domain arrays across explicitly declared sources.

The caller authenticates each source's scientific protocol, passing recipe and
freeze gates before supplying cells and pins. This launcher authenticates those
files, registered models, full-row evaluator tasks, and disjoint RNG schedules.
It retains the qualified v3 runtime and exactly-once submission behavior. It
never chooses a recipe, creates data, or admits a difficulty match.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'ops/exp_scaling') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import launch_modebench_scale_runtime_v3 as runtime

SOURCE = Path(__file__).resolve()
SCHEMA = 'modebench-scale-domain-array-v1'
ORIGIN_SHA256 = '54b0548485d17e2688cd3ef452571a575f35d31435cf5aeb471ad855f76eba4e'
ORIGIN_CHAIN = {**runtime.ORIGIN_CHAIN, str(runtime.SOURCE): ORIGIN_SHA256}
HARDWARE = dict(runtime.HARDWARE)
RUNTIME_PROFILE = dict(runtime.RUNTIME_PROFILE)
PYTHON = runtime.PYTHON
DOMAINS = runtime.DOMAINS
LEVEL_MODELS = runtime.LEVEL_MODELS
CONCURRENCY = 1
require, read, digest, now, atomic_new = runtime.require, runtime.read, runtime.digest, runtime.now, runtime.atomic_new
checkpoint_identity, evaluator = runtime.checkpoint_identity, runtime.evaluator
validate_dependency_ids = runtime.validate_dependency_ids


def source_pins():
    return {str(SOURCE): digest(SOURCE), **ORIGIN_CHAIN,
            str(ROOT / 'ops/repo_env.sh'): digest(ROOT / 'ops/repo_env.sh'),
            **{str((ROOT / path).resolve()): value for path, value in evaluator().code_identity().items()}}


def check_pins(pins):
    require(isinstance(pins, dict) and pins, 'nonempty caller input pins required')
    for source, expected in pins.items():
        require(isinstance(source, str) and Path(source).is_absolute()
                and str(Path(source).resolve()) == source and Path(source).is_file()
                and isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
                'input pins require canonical absolute file paths and SHA256 digests')
        require(digest(source) == expected, 'pinned input changed: ' + source)


def job_command(plan, cell):
    return runtime.job_command({**plan, 'phase': cell['phase']}, cell)


def scheduler_command(plan_path, plan):
    require(plan.get('concurrency') == CONCURRENCY, 'domain arrays require concurrency one')
    command = runtime.scheduler_command(plan_path, plan)
    return ['--job-name=modebench-scale-domains-' + plan['phase'] if value.startswith('--job-name=') else value
            for value in command]


def worker_script():
    return ('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n'
            'source ops/repo_env.sh\nexport HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
            'export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1\n'
            'export PYTHONDONTWRITEBYTECODE=1\nexec ' + shlex.join([str(PYTHON), str(SOURCE), 'worker']) +
            ' --plan "$1" --index "${SLURM_ARRAY_TASK_ID:?}"\n')


def inspect_cells(cells, pins, models, *, fresh):
    """Validate caller-supplied task lists without interpreting scientific gates."""
    require(isinstance(cells, list) and cells, 'nonempty explicit domain cells required')
    e = evaluator()
    ids, logical, outputs, seen_blocks, schedules = set(), set(), set(), set(), {}
    for cell in cells:
        identifier = cell.get('id')
        require(isinstance(identifier, str) and re.fullmatch('[a-z][a-z0-9_]*', identifier), 'unsafe cell id')
        require(identifier not in ids, 'duplicate cell id')
        ids.add(identifier)
        level, domain, phase = cell.get('level'), cell.get('domain'), cell.get('phase')
        require(level in LEVEL_MODELS and domain in DOMAINS and phase in ('dev', 'eval'), 'unknown level/domain/phase')
        key = (level, domain, phase)
        require(key not in logical, 'duplicate logical domain cell')
        logical.add(key)
        require(isinstance(cell.get('source_kind'), str) and cell['source_kind'].strip(), 'explicit source kind required')
        root = Path(cell.get('source_root', ''))
        require(root.is_absolute() and root.is_dir() and str(root.resolve()) == str(root), 'canonical source root required')
        protocol_path = root / 'protocol.json'
        require(str(protocol_path) in pins, 'caller must pin each source protocol')
        protocol = read(protocol_path)
        label = LEVEL_MODELS[level]
        require(label in models and protocol.get('models', {}).get(label) ==
                e.model_identity(Path(models[label]['path']), label), 'checkpoint differs from source protocol')
        tasks = cell.get('tasks')
        require(isinstance(tasks, list) and len(tasks) == (4 if phase == 'dev' else 1),
                'each domain needs four development tasks or one confirmation task')
        base = root / level / ('pools' if phase == 'dev' else 'dataset') / domain
        require(str(base / 'identity.json') in pins, 'caller must pin pool/frozen identity certificates')
        if phase == 'eval':
            require(str(root / level / 'recipes' / (domain + '.json')) in pins,
                    'caller must pin the confirmation recipe')
        inputs, labels = set(), None
        for task in tasks:
            require(task.get('level') == level and task.get('domain') == domain and task.get('split') == phase,
                    'task differs from its domain cell')
            e.validate_task(task, phase == 'eval')
            require(task.get('batch_size', 8) == HARDWARE['batch_size']
                    and task.get('row_offset', 0) == 0 and task.get('row_limit', 0) == 0,
                    'tasks require qualified batching and complete source rows')
            require(task.get('interface', e.INTERFACE) == e.INTERFACE, 'task interface changed')
            if labels is None:
                labels = task['seeds']
            require(task['seeds'] == labels, 'candidate tiers require common registered draw labels')
            field = 'rows_jsonl' if task.get('rows_jsonl') else 'dataset'
            source = Path(task[field])
            require(source.is_absolute() and str(source.resolve()) == str(source), 'canonical task source required')
            require(source not in inputs, 'duplicate task source')
            inputs.add(source)
            if field == 'rows_jsonl':
                required_inputs = [source]
            else:
                require(phase == 'eval' and source == base / 'eval' and source.is_dir(),
                        'only frozen eval datasets may be scored as Arrow inputs')
                required_inputs = [path for path in source.rglob('*') if path.is_file()]
                require(required_inputs, 'nonempty frozen dataset required')
            require(all(str(path.resolve()) in pins for path in required_inputs), 'caller must pin every task input file')
            output = Path(task['output'])
            require(output.is_absolute() and str(output.resolve()) == str(output), 'canonical task output required')
            require(str(output) not in outputs and str(output) not in pins, 'duplicate or input-overlapping output')
            outputs.add(str(output))
            if fresh:
                require(not output.exists() and not Path(str(output) + '.batches').exists(), 'preexisting calibration outputs')
            rows, identity = e.load_rows(task)
            require(rows and identity['row_offset'] == 0 and identity['row_limit'] == 0,
                    'nonempty full source rows required')
            schedule = e.frozen.schedule_record(domain, rows, task['seeds'])
            blocks = {block for row in schedule['request_seeds'] for block in row}
            require(len(blocks) == len(rows) * 4 and not seen_blocks & blocks,
                    'RNG block collision across calibration tasks')
            seen_blocks.update(blocks)
            schedules[str(output)] = e.sha(schedule)
        expected = {base / f'difficulty_{tier}.jsonl' for tier in range(4)} if phase == 'dev' else {
            base / ('eval.jsonl' if tasks[0].get('rows_jsonl') else 'eval')}
        require(inputs == expected, 'task sources differ from declared domain pool/frozen paths')
    return {'policy': e.frozen.POLICY, 'distinct_request_blocks': len(seen_blocks),
            'distinct_child_seeds': len(seen_blocks) * 8, 'task_seed_schedule_sha256': schedules}


def prepare(campaign_root, cells, models, *, dependency_ids=(), pins=None):
    """Prepare only the supplied cells; caller owns source-specific science gates."""
    target = Path(campaign_root).resolve()
    require(not target.exists(), 'fresh campaign directory required')
    check_pins(pins)
    immutable = source_pins()
    check_pins(immutable)
    for source, expected in pins.items():
        require(source not in immutable or immutable[source] == expected, 'conflicting source pin')
        immutable[source] = expected
    require(isinstance(cells, list) and cells, 'nonempty explicit domain cells required')
    labels = {LEVEL_MODELS.get(cell.get('level')) for cell in cells}
    require(None not in labels, 'unknown level')
    identities = {label: checkpoint_identity(models.get(label), label) for label in sorted(labels)}
    copies = json.loads(json.dumps(cells))
    rng = inspect_cells(copies, pins, identities, fresh=True)
    require(PYTHON.is_file(), 'calibration Python runtime missing')
    phases = {cell['phase'] for cell in copies}
    plan = {'schema': SCHEMA, 'created_at': now(), 'phase': next(iter(phases)) if len(phases) == 1 else 'mixed',
            'python': str(PYTHON), 'concurrency': CONCURRENCY, 'hardware': HARDWARE,
            'runtime_profile': RUNTIME_PROFILE, 'origin_chain': ORIGIN_CHAIN,
            'dependency_ids': validate_dependency_ids(dependency_ids), 'models': identities,
            'scientific_inputs_sha256': dict(pins), 'immutable_inputs_sha256': immutable,
            'rng_admission': rng, 'cells': [], 'treatment_training_started': False}
    task_files = []
    for cell in copies:
        tasks = cell.pop('tasks')
        cell['tasks'] = str(target / 'tasks' / (cell['id'] + '.json'))
        cell['model_label'] = LEVEL_MODELS[cell['level']]
        cell['command'] = job_command(plan, cell)
        cell['shell_command'] = shlex.join(cell['command'])
        task_files.append((Path(cell['tasks']), tasks))
        plan['cells'].append(cell)
    target.mkdir(parents=True, exist_ok=False)
    for name in ('tasks', 'logs', 'runtime'):
        (target / name).mkdir()
    for path, tasks in task_files:
        atomic_new(path, tasks)
        immutable[str(path)] = digest(path)
    script = target / 'worker.slurm'
    script.write_text(worker_script())
    immutable[str(script)] = digest(script)
    plan['submit_command'] = scheduler_command(target / 'plan.json', plan)
    plan['submit_shell_command'] = shlex.join(plan['submit_command'])
    atomic_new(target / 'plan.json', plan)
    atomic_new(target / 'plan.sha256.json', {'sha256': digest(target / 'plan.json')})
    return plan


def verify(plan_path, fresh=False):
    path = Path(plan_path).resolve()
    require(digest(path) == read(path.parent / 'plan.sha256.json')['sha256'], 'plan changed')
    plan = read(path)
    require(plan.get('schema') == SCHEMA and plan.get('hardware') == HARDWARE
            and plan.get('runtime_profile') == RUNTIME_PROFILE and plan.get('concurrency') == CONCURRENCY
            and plan.get('python') == str(PYTHON) and plan.get('origin_chain') == ORIGIN_CHAIN,
            'domain array runtime contract changed')
    check_pins(plan['immutable_inputs_sha256'])
    check_pins(plan['scientific_inputs_sha256'])
    require(all(plan['immutable_inputs_sha256'].get(source) == expected
                for source, expected in {**source_pins(), **plan['scientific_inputs_sha256']}.items()),
            'plan omits required immutable input pins')
    require(plan['submit_command'] == scheduler_command(path, plan), 'scheduler command changed')
    require((path.parent / 'worker.slurm').read_text() == worker_script(), 'worker script changed')
    for label, expected in plan['models'].items():
        require(checkpoint_identity(expected['path'], label) == expected, 'checkpoint changed: ' + label)
    cells = []
    for cell in plan['cells']:
        require(cell.get('model_label') == LEVEL_MODELS.get(cell.get('level'))
                and cell['command'] == job_command(plan, cell), 'worker command changed')
        task_path = Path(cell['tasks'])
        require(task_path == path.parent / 'tasks' / (cell['id'] + '.json')
                and plan['immutable_inputs_sha256'].get(str(task_path)) == digest(task_path), 'task manifest changed')
        cells.append({**cell, 'tasks': read(task_path)})
    require(inspect_cells(cells, plan['scientific_inputs_sha256'], plan['models'], fresh=fresh) == plan['rng_admission'],
            'RNG schedule admission changed')
    return plan


def submit(plan_path, runner=subprocess.run):
    path = Path(plan_path).resolve()
    plan = verify(path, fresh=True)
    intent = path.parent / 'submission_intent.json'
    require(not intent.exists(), 'submission already attempted; reconcile its durable intent without resubmitting')
    atomic_new(intent, {'at': now(), 'plan_sha256': digest(path), 'command': plan['submit_command'],
                        'status': 'submission_attempt_started'})
    result = runner(plan['submit_command'], text=True, capture_output=True, check=False, timeout=60)
    evidence = {'at': now(), 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', result.stdout.strip())
    if result.returncode != 0 or match is None:
        atomic_new(path.parent / 'submission_ambiguous.json', evidence)
        raise RuntimeError('sbatch failed or returned an ambiguous job id; durable intent prevents duplicate submission')
    evidence.update(status='submitted', array_job_id=int(match.group(1)),
                    cells=[{'array_index': index, 'cell': cell['id']} for index, cell in enumerate(plan['cells'])])
    atomic_new(path.parent / 'submission_result.json', evidence)
    return evidence


def worker(plan_path, index):
    path = Path(plan_path).resolve()
    plan = verify(path)
    require(type(index) is int and 0 <= index < len(plan['cells']), 'invalid array index')
    require(os.environ.get('VLLM_USE_V1') == '0' and os.environ.get('VLLM_ATTENTION_BACKEND') == 'XFORMERS',
            'worker requires vLLM V0 and XFORMERS')
    require(all(os.environ.get(key) == value for key, value in RUNTIME_PROFILE['thread_environment'].items()),
            'worker CPU thread environment differs from runtime profile')
    require(importlib.metadata.version('vllm') == HARDWARE['vllm_version'], 'vLLM version changed')
    result_path = path.parent / 'submission_result.json'
    deadline = time.monotonic() + 60
    while not result_path.exists() and time.monotonic() < deadline:
        time.sleep(1)
    require(result_path.exists(), 'submission result absent; reconcile the durable intent')
    submission = read(result_path)
    require(submission.get('status') == 'submitted' and type(submission.get('array_job_id')) is int
            and os.environ.get('SLURM_ARRAY_JOB_ID') == str(submission['array_job_id'])
            and os.environ.get('SLURM_ARRAY_TASK_ID') == str(index), 'worker is outside its submitted array')
    intent = read(path.parent / 'submission_intent.json')
    require(intent.get('plan_sha256') == digest(path) and intent.get('command') == plan['submit_command'],
            'submission intent differs from prepared plan')
    atomic_new(path.parent / 'runtime' / f'{index}.json',
               {'at': now(), 'cell': plan['cells'][index]['id'], 'plan_sha256': digest(path),
                'array_job_id': submission['array_job_id'], 'hostname': os.uname().nodename,
                'command': plan['cells'][index]['command']})
    os.execv(plan['python'], plan['cells'][index]['command'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'submit', 'worker'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args(argv)
    if args.action == 'worker':
        worker(args.plan, args.index)
    else:
        result = verify(args.plan) if args.action == 'verify' else submit(args.plan)
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
