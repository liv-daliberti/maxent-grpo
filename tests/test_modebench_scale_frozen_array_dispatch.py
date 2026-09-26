"""Scratch operational tests; no scientific model, scheduler, or production writes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

REPO = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    d = module(REPO / 'artifacts/dispatch_modebench_scale_frozen_arrays_20260912.py', 'scratch_dispatch')
    r = module(REPO / 'artifacts/recover_modebench_scale_pantry_frozen_view_20260912.py', 'scratch_atomic')
    root = tmp_path / 'workspace'
    root.mkdir()
    def file(name, text='scratch-only'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path
    composite = root / 'composite'
    plan_path = file('composite/revision_development/plan.json')
    worker = file('composite/revision_development/worker.slurm', 'UNCHANGED ORIGINAL WORKER')
    source = file('artifacts/dispatch.py')
    launcher_path = file('ops/launch.py')
    recovery_path = file('artifacts/recovery.py')
    runner = file('artifacts/view_runner.py')
    manifest_path = file('artifacts/view/manifest.json', '{}')
    file('ops/repo_env.sh')
    copy = file('artifacts/view/copy.py')
    proot = file('proot')
    ready = file('completed_recovery.json', '{"job_id": 555}')
    python = root / 'python'
    plan = {'cells': [{'id': 'level4_graph_coloring', 'command': [str(python), 'evaluator', '--resume']}],
            'submit_command': ['sbatch', '--parsable', '--partition=mltheory', '--account=mltheory',
             '--nodelist=node105', '--nodes=1', '--ntasks=1', '--gres=gpu:a5000:2',
             '--cpus-per-task=6', '--mem=60G', '--time=08:00:00', '--array=0-0%1',
             '--job-name=modebench-scale-domains-dev', '--no-requeue', '--export=ALL',
             '--dependency=afterany:123:456', str(worker), str(plan_path)]}
    plan_path.write_text(json.dumps(plan))
    file('composite/revision_development/plan.sha256.json', json.dumps({'sha256': d.sha(plan_path)}))
    for key, value in {'ROOT': root, 'SOURCE': source, 'LAUNCHER': launcher_path,
                       'RECOVERY_HELPER': recovery_path, 'VIEW_RUNNER': runner,
                       'VIEW_MANIFEST': manifest_path, 'PYTHON': python, 'COMPOSITE': composite,
                       'DISPATCH_ROOT': root / 'artifacts/dispatch'}.items():
        monkeypatch.setattr(d, key, value)
    checks, workers, calls = [], [], []
    recovery = SimpleNamespace(atomic_new=r.atomic_new)
    def verify(path, fresh=False):
        assert path == plan_path
        checks.append(fresh)
        assert d.sha(path) == json.loads((path.parent / 'plan.sha256.json').read_text())['sha256']
        return json.loads(path.read_text())
    def submit(path, runner):
        verify(path, fresh=True)
        r.atomic_new(path.parent / 'submission_intent.json',
                     {'plan_sha256': d.sha(path), 'command': plan['submit_command']})
        result = runner(plan['submit_command'], text=True, capture_output=True, check=False, timeout=60)
        import re
        match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', result.stdout.strip())
        if result.returncode != 0 or match is None:
            r.atomic_new(path.parent / 'submission_ambiguous.json', {'stdout': result.stdout})
            raise RuntimeError('original ambiguous result')
        value = {'status': 'submitted', 'array_job_id': int(match[1])}
        r.atomic_new(path.parent / 'submission_result.json', value)
        return value
    launcher = SimpleNamespace(verify=verify, submit=submit,
                               worker=lambda path, index: workers.append((path, index)))
    manifest = {'mappings': [{'source': str(copy)}], 'proot': {'path': str(proot)}}
    monkeypatch.setattr(d, 'modules', lambda path: (recovery, launcher, manifest))
    completion = {'job_id': 555, 'files_sha256': {str(ready): d.sha(ready)}}
    monkeypatch.setattr(d, 'recovery_complete', lambda recovery: completion)
    def run(command, **kwargs):
        assert (d.DISPATCH_ROOT / 'revision_development/submission_intent.json').is_file()
        assert (plan_path.parent / 'submission_intent.json').is_file()
        assert kwargs == {'text': True, 'capture_output': True, 'check': False,
                          'timeout': 60, 'close_fds': True}
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='789\n', stderr='')
    monkeypatch.setattr(d.subprocess, 'run', run)
    return SimpleNamespace(d=d, recovery=recovery, launcher=launcher, path=plan_path, plan=plan,
                           manifest=manifest_path, calls=calls, checks=checks, workers=workers,
                           directory=d.DISPATCH_ROOT / 'revision_development', completion=completion)


def submitted(c):
    return c.d.submit(c.path, c.manifest)


def test_exact_hook_preserves_plan_script_resources_args_dependencies(dispatch):
    c = dispatch
    before = {p: p.read_bytes() for p in [c.path, c.path.parent / 'worker.slurm',
                                        c.path.parent / 'plan.sha256.json']}
    assert submitted(c)['array_job_id'] == 789
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(c.calls) == 1
    actual = c.calls[0]
    assert actual[:-3] == c.plan['submit_command'][:-2]
    assert actual[-3].startswith('--comment=mb-frozen-revision_development-')
    assert actual[-2:] == [str(c.directory / 'outer_worker.slurm'), str(c.path)]
    script = (c.directory / 'outer_worker.slurm').read_text()
    assert ' exec --manifest ' + str(c.manifest) in script
    assert ' -- ' + str(c.d.PYTHON) + ' -B ' + str(c.d.SOURCE) in script
    assert '--index "${SLURM_ARRAY_TASK_ID:?}"' in script
    for key, value in c.d.ENVIRONMENT.items():
        assert key + '=' + value in script
    assert c.d.read(c.path.parent / 'submission_intent.json')['command'] == c.plan['submit_command']
    assert c.d.read(c.directory / 'scheduler_result.json')['actual_command'] == actual
    assert c.d.verify(c.path, c.manifest)


def test_duplicate_submit_never_calls_scheduler_twice(dispatch):
    c = dispatch
    submitted(c)
    with pytest.raises(ValueError, match='already attempted'):
        submitted(c)
    assert len(c.calls) == 1


@pytest.mark.parametrize('name', ['submission_intent.json', 'submission_result.json', 'submission_ambiguous.json'])
def test_preexisting_original_submission_blocks_dispatch(dispatch, name):
    c = dispatch
    (c.path.parent / name).write_text('{}')
    with pytest.raises(ValueError, match='original submission evidence'):
        submitted(c)
    assert not c.calls


@pytest.mark.parametrize('kind', ['timeout', 'invalid', 'nonzero'])
def test_ambiguous_submission_keeps_both_intents_and_never_retries(dispatch, monkeypatch, kind):
    c = dispatch
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if kind == 'timeout':
            raise subprocess.TimeoutExpired(argv, 60)
        return SimpleNamespace(returncode=1 if kind == 'nonzero' else 0, stdout='bad', stderr='error')
    monkeypatch.setattr(c.d.subprocess, 'run', run)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        submitted(c)
    assert (c.directory / 'submission_intent.json').exists()
    assert (c.path.parent / 'submission_intent.json').exists()
    assert (c.directory / 'submission_ambiguous.json').exists()
    with pytest.raises(ValueError, match='already attempted'):
        submitted(c)
    assert len(calls) == 1


def test_incomplete_recovery_blocks_before_dispatch_artifacts(dispatch, monkeypatch):
    c = dispatch
    def incomplete(recovery):
        raise ValueError('Pantry recovery incomplete')
    monkeypatch.setattr(c.d, 'recovery_complete', incomplete)
    with pytest.raises(ValueError, match='incomplete'):
        submitted(c)
    assert not c.directory.exists() and not c.calls


@pytest.mark.parametrize('changed', ['source', 'wrapper', 'plan', 'completion'])
def test_changed_dispatch_pins_reject_worker(dispatch, changed):
    c = dispatch
    submitted(c)
    path = {'source': c.d.SOURCE, 'wrapper': c.directory / 'outer_worker.slurm',
            'plan': c.path, 'completion': Path(next(iter(c.completion['files_sha256'])))}[changed]
    path.chmod(0o644)
    path.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        c.d.worker(c.path, 0)
    assert not c.workers


@pytest.mark.parametrize('name', ['other', 'revision_development_extra'])
def test_unapproved_stage_rejected(dispatch, name):
    c = dispatch
    with pytest.raises(ValueError, match='only exact'):
        c.d.submit(c.path.parent.parent / name / 'plan.json', c.manifest)


def worker_env(c, monkeypatch):
    submitted(c)
    for key, value in c.d.ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    for key, value in {'SLURM_ARRAY_JOB_ID': '789', 'SLURM_ARRAY_TASK_ID': '0',
                       'SLURMD_NODENAME': 'node105', 'SLURM_CPUS_PER_TASK': '6',
                       'SLURM_MEM_PER_NODE': '61440'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(c.d.socket, 'gethostname', lambda: 'node105.ionic.cs.princeton.edu')
    monkeypatch.setattr(c.d.importlib.metadata, 'version', lambda name: '0.8.4')
    cuda = SimpleNamespace(device_count=lambda: 2, get_device_name=lambda index: 'NVIDIA RTX A5000')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    return cuda


def test_worker_records_actual_runtime_then_original_worker_once(dispatch, monkeypatch):
    c = dispatch
    worker_env(c, monkeypatch)
    c.d.worker(c.path, 0)
    assert c.workers == [(c.path, 0)]
    runtime = c.d.read(c.directory / 'runtime/0.json')
    assert runtime['hostname'] == 'node105.ionic.cs.princeton.edu'
    assert runtime['command'] == c.plan['cells'][0]['command']
    assert runtime['array_job_id'] == 789 and runtime['visible_gpu_names'] == ['NVIDIA RTX A5000'] * 2
    with pytest.raises(ValueError, match='already started'):
        c.d.worker(c.path, 0)
    assert len(c.workers) == 1


@pytest.mark.parametrize('key,value', [('SLURM_MEM_PER_NODE', '60416'), ('SLURM_ARRAY_JOB_ID', '123'),
                                      ('SLURM_ARRAY_TASK_ID', '1'), ('SLURM_CPUS_PER_TASK', '4'),
                                      ('SLURMD_NODENAME', 'node106'), ('VLLM_USE_V1', '1')])
def test_worker_wrong_environment_never_calls_original(dispatch, monkeypatch, key, value):
    c = dispatch
    worker_env(c, monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        c.d.worker(c.path, 0)
    assert not c.workers and not (c.directory / 'runtime/0.json').exists()


def test_worker_wrong_actual_host_gpu_or_version(dispatch, monkeypatch):
    c = dispatch
    cuda = worker_env(c, monkeypatch)
    monkeypatch.setattr(c.d.socket, 'gethostname', lambda: 'wash.cs.princeton.edu')
    with pytest.raises(ValueError, match='hostname'):
        c.d.worker_environment(789, 0)
    monkeypatch.setattr(c.d.socket, 'gethostname', lambda: 'node105')
    monkeypatch.setattr(cuda, 'device_count', lambda: 1)
    with pytest.raises(ValueError, match='A5000'):
        c.d.worker_environment(789, 0)
    monkeypatch.setattr(cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(c.d.importlib.metadata, 'version', lambda name: 'wrong')
    with pytest.raises(ValueError, match='version'):
        c.d.worker_environment(789, 0)


def test_worker_waits_for_both_results_without_holding_submit_lock(dispatch, monkeypatch):
    c = dispatch
    worker_env(c, monkeypatch)
    result = c.directory / 'submission_result.json'
    saved = result.read_bytes()
    result.unlink()
    monkeypatch.setattr(c.d.time, 'sleep', lambda seconds: result.write_bytes(saved))
    with c.d.lock(c.d.DISPATCH_ROOT / 'revision_development.lock'):
        c.d.worker(c.path, 0)
    assert len(c.workers) == 1


def test_worker_missing_result_times_out_without_runtime(dispatch, monkeypatch):
    c = dispatch
    worker_env(c, monkeypatch)
    (c.directory / 'submission_result.json').unlink()
    times = iter([0, 61])
    monkeypatch.setattr(c.d.time, 'monotonic', lambda: next(times))
    with pytest.raises(ValueError, match='results missing'):
        c.d.worker(c.path, 0)
    assert not c.workers and not (c.directory / 'runtime/0.json').exists()


def test_worker_conflicting_array_results_fail_closed(dispatch, monkeypatch):
    c = dispatch
    worker_env(c, monkeypatch)
    path = c.directory / 'submission_result.json'
    value = c.d.read(path)
    value['array_job_id'] = 790
    path.chmod(0o644)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='result linkage'):
        c.d.worker(c.path, 0)
    assert not c.workers


def test_hook_rejects_added_parameters(dispatch, monkeypatch):
    c = dispatch
    def bad_submit(path, runner):
        c.recovery.atomic_new(path.parent / 'submission_intent.json',
            {'plan_sha256': c.d.sha(path), 'command': c.plan['submit_command']})
        return runner(c.plan['submit_command'], text=True, capture_output=True,
                      check=False, timeout=60, pass_fds=(9,))
    monkeypatch.setattr(c.launcher, 'submit', bad_submit)
    with pytest.raises(ValueError, match='unexpected'):
        submitted(c)
    assert not c.calls


@pytest.fixture
def readiness(tmp_path):
    d = module(REPO / 'artifacts/dispatch_modebench_scale_frozen_arrays_20260912.py', 'readiness_dispatch')
    root = tmp_path / 'recovery'
    root.mkdir()
    results = root / 'receipts'
    results.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value))
        return path
    plan = {'command': ['exact', 'original', 'evaluator'], 'submit_command': ['exact', 'sbatch']}
    write(root / 'plan.json', plan)
    digest = d.sha(root / 'plan.json')
    write(root / 'plan.sha256.json', {'sha256': digest})
    write(root / 'submission_intent.json', {'command': plan['submit_command'], 'plan_sha256': digest})
    write(root / 'submission_result.json', {'status': 'submitted', 'job_id': 555,
        'plan_sha256': digest, 'intent_sha256': d.sha(root / 'submission_intent.json')})
    write(root / 'runtime.json', {'job_id': 555, 'status': 'validated_before_exact_evaluator_exec',
        'command': plan['command'], 'view_manifest_sha256': d.VIEW_MANIFEST_SHA, 'plan_sha256': digest})
    model = {'label': '14b', 'snapshot': 'fixed'}
    runtime = {'max_model_len': 2048, 'tensor_parallel_size': 2, 'swap_space': 4.0,
               'gpu_memory_utilization': 0.82, 'enable_prefix_caching': True}
    tasks = []
    for tier in range(4):
        path = results / f'difficulty_{tier}.json'
        task = {'output': str(path), 'rows_jsonl': str(root / f'rows_{tier}'),
                'seeds': [11, 12, 13, 14], 'batch_size': 8}
        tasks.append(task)
        write(path, {'status': 'complete', 'level': 'level5', 'domain': 'pantry', 'model_label': '14b',
             'identity': {'split': 'dev', 'model': {**model, 'vllm_version': '0.8.4'}, 'runtime': runtime,
                          'source': {'path': task['rows_jsonl']}, 'seeds': task['seeds'], 'batch_size': 8}})
    tasks_path = write(root / 'tasks.json', tasks)
    checks = []
    def validate(receipt, rows):
        checks.append(receipt)
        return {'rows': len(rows), 'status': 'validated_by_fixture'}
    evaluator = SimpleNamespace(ENGINE_CONTRACT={'vllm_version': '0.8.4'},
        model_identity=lambda path, label: model,
        load_rows=lambda task: ([{'row': 'fixture'}], {'path': task['rows_jsonl']}),
        runtime_settings=lambda **kwargs: kwargs, validate_seed_receipt=validate)
    output = ['555|555|COMPLETED|0:0|2026-09-12T03:00:00|node105\n']
    r = SimpleNamespace(RECOVERY_ROOT=root, RESULTS=results, LAUNCHER=root / 'launcher',
        verify=lambda root, require_original_inventory: plan,
        original=lambda: ({'models': {'14b': {'path': '/fixed/model'}}}, {'tasks': str(tasks_path)}),
        module=lambda path, name: SimpleNamespace(evaluator=lambda: evaluator),
        run_read=lambda command: output[0])
    return SimpleNamespace(d=d, r=r, output=output, checks=checks, root=root, results=results)


def test_completed_recovery_requires_all_four_original_validators(readiness):
    c = readiness
    evidence = c.d.recovery_complete(c.r)
    assert evidence['job_id'] == 555 and len(c.checks) == 4
    assert len(evidence['receipt_checks']) == 4 and len(evidence['files_sha256']) == 9
    assert evidence['command'][:5] == ['sacct', '-X', '-j', '555', '-n']


@pytest.mark.parametrize('column,value', [(0, '999'), (1, '555_0'), (2, 'RUNNING'), (3, '1:0'),
                                        (4, 'Unknown'), (5, 'node106')])
def test_recovery_not_exactly_completed_rejected(readiness, column, value):
    c = readiness
    row = c.output[0].strip().split('|')
    row[column] = value
    c.output[0] = '|'.join(row)
    with pytest.raises(ValueError, match='authoritatively completed'):
        c.d.recovery_complete(c.r)


@pytest.mark.parametrize('key,value', [('split', 'eval'), ('model', {'label': '7b'}),
                                      ('seeds', [1, 2, 3, 4]), ('runtime', {}),
                                      ('source', {}), ('batch_size', 1)])
def test_recovery_receipt_must_match_exact_original_task(readiness, key, value):
    c = readiness
    path = c.results / 'difficulty_3.json'
    receipt = c.d.read(path)
    receipt['identity'][key] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='exact frozen tasks'):
        c.d.recovery_complete(c.r)


def test_recovery_validator_failure_blocks_dispatch_readiness(readiness):
    c = readiness
    original = c.r.module
    def broken(path, name):
        launcher = original(path, name)
        evaluator = launcher.evaluator()
        def reject(*args):
            raise ValueError('original seed receipt rejection')
        evaluator.validate_seed_receipt = reject
        return launcher
    c.r.module = broken
    with pytest.raises(ValueError, match='seed receipt'):
        c.d.recovery_complete(c.r)


def test_duplicate_hook_invocation_records_ambiguity_but_submits_once(dispatch, monkeypatch):
    c = dispatch
    def twice(path, runner):
        c.recovery.atomic_new(path.parent / 'submission_intent.json',
            {'plan_sha256': c.d.sha(path), 'command': c.plan['submit_command']})
        kwargs = {'text': True, 'capture_output': True, 'check': False, 'timeout': 60}
        runner(c.plan['submit_command'], **kwargs)
        return runner(c.plan['submit_command'], **kwargs)
    monkeypatch.setattr(c.launcher, 'submit', twice)
    with pytest.raises(ValueError, match='duplicate'):
        submitted(c)
    assert len(c.calls) == 1
    assert (c.directory / 'scheduler_result.json').exists()
    assert (c.directory / 'submission_ambiguous.json').exists()


@pytest.mark.parametrize('field', ['actual_command', 'stdout', 'returncode'])
def test_scheduler_receipt_drift_rejected_even_with_updated_result_hash(dispatch, monkeypatch, field):
    c = dispatch
    worker_env(c, monkeypatch)
    path = c.directory / 'scheduler_result.json'
    scheduler = c.d.read(path)
    if field == 'actual_command':
        scheduler[field][-3] = '--comment=wrong-token'
    elif field == 'stdout':
        scheduler[field] = '790\n'
    else:
        scheduler[field] = 1
    path.chmod(0o644)
    path.write_text(json.dumps(scheduler))
    result = c.directory / 'submission_result.json'
    value = c.d.read(result)
    value['scheduler_result_sha256'] = c.d.sha(path)
    result.chmod(0o644)
    result.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='scheduler receipt'):
        c.d.worker(c.path, 0)
    assert not c.workers


def test_dispatch_reconciliation_token_drift_rejected(dispatch):
    c = dispatch
    submitted(c)
    path = c.directory / 'dispatch.json'
    value = c.d.read(path)
    value['reconciliation_token'] = 'mb-frozen-revision_development-' + '0' * 24
    path.chmod(0o644)
    path.write_text(json.dumps(value))
    checksum = c.directory / 'dispatch.sha256.json'
    checksum.chmod(0o644)
    checksum.write_text(json.dumps({'sha256': c.d.sha(path)}))
    with pytest.raises(ValueError, match='mapping changed'):
        c.d.verify(c.path, c.manifest)
