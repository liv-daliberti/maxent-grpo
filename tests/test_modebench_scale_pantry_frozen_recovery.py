"""Scratch-only operational tests. No model, scientific grading, or real Slurm calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    r = load(ROOT / 'artifacts/recover_modebench_scale_pantry_frozen_view_20260912.py', 'scratch_recovery')
    adapter = load(ROOT / 'artifacts/run_modebench_scale_frozen_view_20260912.py', 'scratch_adapter')
    base = tmp_path / 'scratch_only'
    base.mkdir()
    for folder in ('artifacts', 'src', 'ops', 'actions', 'original/runtime', 'original/logs', 'results'):
        (base / folder).mkdir(parents=True, exist_ok=True)
    def file(name, content='scratch-only; never scientific evidence'):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
    source = file('artifacts/recovery.py')
    runner = file('artifacts/view_runner.py')
    canonical = file('src/templates.py', 'neutral')
    preserved = file('artifacts/preserved.py', 'old')
    proot = file('proot')
    proot.chmod(0o755)
    python = file('python')
    launcher = file('ops/launcher.py')
    file('ops/repo_env.sh')
    for key, value in {'ROOT': base, 'RUNNER': runner, 'CANONICAL': canonical,
                       'PRESERVED': preserved, 'PROOT': proot,
                       'OLD_SHA256': r.sha(preserved), 'NEUTRAL_SHA256': r.sha(canonical),
                       'PROOT_SHA256': r.sha(proot)}.items():
        monkeypatch.setattr(adapter, key, value)
    view = base / 'artifacts/frozen'
    adapter.prepare(view)
    # A scratch source change simulates what the mapped child observes. The real
    # host source is never changed, and no PRoot/model/Slurm process is started.
    canonical.write_text('old')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    monkeypatch.setenv('PYTHONPYCACHEPREFIX', '/tmp/NONPRODUCTION-scale-frozen-pycache-scratch/pycache')
    original_path = base / 'original/plan.json'
    protocol = file('protocol.json', '{}')
    task = file('original/tasks.json', '[]')
    command = [str(python), str(base / 'ops/evaluate.py'), '--tasks-json', str(task), '--resume']
    cell = {'id': 'level5_pantry', 'domain': 'pantry', 'model_label': '14b', 'command': command}
    plan = {'cells': [None] * 9 + [cell], 'immutable_inputs_sha256': {str(task): r.sha(task)}}
    original_path.write_text(json.dumps(plan))
    file('original/plan.sha256.json', json.dumps({'sha256': r.sha(original_path)}))
    file('original/runtime/9.json', json.dumps({'array_job_id': 31243495, 'cell': 'level5_pantry',
          'command': command, 'plan_sha256': r.sha(original_path)}))
    file('original/submission_result.json', '{}')
    file('original/submission_intent.json', '{}')
    file('original/logs/31243495_9.out')
    file('original/logs/31243495_9.err', 'validate_seed_receipt\n'
         'ValueError: receipt evaluator/helper code hashes missing or changed\n')
    for tier in range(3):
        directory = base / f'results/difficulty_{tier}.json.batches'
        directory.mkdir()
        file(str(directory.relative_to(base) / 'identity.json'), '{"scratch_only": true}')
        for batch in range(116):
            file(str(directory.relative_to(base) / f'seed-0__rows-{batch}.json'), '{"scratch_only": true}')
    for tier in (0, 1):
        file(f'results/difficulty_{tier}.json', '{"scratch_only": true}')
    constants = {'ROOT': base, 'SOURCE': source, 'VIEW_RUNNER': runner, 'VIEW_RUNNER_SHA': r.sha(runner),
                 'LAUNCHER': launcher,
                 'ORIGINAL_PLAN': original_path, 'ORIGINAL_PLAN_SHA': r.sha(original_path),
                 'PROTOCOL': protocol, 'PROTOCOL_SHA': r.sha(protocol), 'RESULTS': base / 'results',
                 'RECOVERY_ROOT': base / 'recovery', 'ACTION_ROOT': base / 'actions',
                 'CLAIM': base / 'actions/claim.json', 'LOCK': base / 'actions/action.lock',
                 'PYTHON': python, 'EXTRA_PINS': {}}
    for key, value in constants.items():
        monkeypatch.setattr(r, key, value)
    verified = []
    def verify_original(path):
        verified.append(path)
        return plan
    def module(path, name):
        if path == runner:
            return adapter
        assert path == launcher
        return SimpleNamespace(verify=verify_original)
    monkeypatch.setattr(r, 'module', module)
    calls = []
    def read_scheduler(command):
        calls.append(command)
        if command[0] == 'squeue':
            return ''
        assert command[0] == 'sacct'
        return '|'.join(r.ORIGINAL_JOB) + '\n'
    monkeypatch.setattr(r, 'run_read', read_scheduler)
    # Every test must explicitly replace this guard before any mocked submission.
    def no_real_process(*args, **kwargs):
        pytest.fail('unexpected process launch in scratch test')
    monkeypatch.setattr(r.subprocess, 'run', no_real_process)
    return SimpleNamespace(r=r, root=r.RECOVERY_ROOT, manifest=view / 'manifest.json',
                           adapter=adapter, canonical=canonical, verified=verified,
                           scheduler_reads=calls, command=command)


def prepared(c):
    c.r.prepare(c.root, c.manifest)
    return c.r.read(c.root / 'plan.json')


def mock_submit(c, monkeypatch, *, stdout='123456\n', returncode=0):
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        assert c.r.read(c.root / 'submission_intent.json')['command'] == command
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr='')
    monkeypatch.setattr(c.r.subprocess, 'run', invoke)
    return calls


def test_prepare_preserves_every_original_byte_and_runtime(campaign):
    c = campaign
    before = {str(p): p.read_bytes() for p in c.r.ORIGINAL_PLAN.parent.rglob('*') if p.is_file()}
    before.update({str(p): p.read_bytes() for p in c.r.RESULTS.rglob('*') if p.is_file()})
    result = c.r.prepare(c.root, c.manifest)
    assert result['status'] == 'prepared_only'
    assert all(Path(path).read_bytes() == data for path, data in before.items())
    plan = c.r.verify(c.root)
    assert len(plan['preserved_outputs_sha256']) == 353
    assert plan['command'] == c.command
    assert c.verified == [c.r.ORIGINAL_PLAN, c.r.ORIGINAL_PLAN]
    assert not (c.root / 'submission_intent.json').exists()
    assert not (c.root / 'runtime.json').exists()


def test_remote_wrapper_explicit_view_and_exact_resources(campaign):
    c = campaign
    plan = prepared(c)
    script = (c.root / 'worker.slurm').read_text()
    assert str(c.r.VIEW_RUNNER) + ' exec --manifest ' + str(c.manifest) in script
    assert ' -- ' + str(c.r.PYTHON) + ' -B ' + str(c.r.SOURCE) + ' worker --root ' in script
    for key, value in c.r.ENVIRONMENT.items():
        assert key + '=' + value in script
    command = plan['submit_command']
    assert '--mem=59G' in command and '--gres=gpu:a5000:2' in command
    assert '--no-requeue' in command and '--cpus-per-task=6' in command
    assert not any(arg.startswith('--array') for arg in command)
    assert not any('controller' in arg for arg in command)


def test_double_prepare_and_alternative_root_refused(campaign):
    c = campaign
    prepared(c)
    with pytest.raises(ValueError, match='already exists'):
        c.r.prepare(c.root, c.manifest)
    with pytest.raises(ValueError, match='one fixed'):
        c.r.prepare(c.root.with_name('another'), c.manifest)


@pytest.mark.parametrize('which', ['helper', 'batch', 'runtime', 'binary', 'manifest'])
def test_changed_pins_refuse_before_submission(campaign, monkeypatch, which):
    c = campaign
    prepared(c)
    path = {'helper': c.r.SOURCE, 'batch': next(c.r.RESULTS.rglob('seed-*')),
            'runtime': c.r.ORIGINAL_PLAN.parent / 'runtime/9.json',
            'binary': c.adapter.PROOT, 'manifest': c.manifest}[which]
    path.chmod(0o644)
    path.write_text(path.read_text() + '\nchanged')
    with pytest.raises(ValueError, match='changed'):
        c.r.submit(c.root)
    assert not (c.root / 'submission_intent.json').exists()


def test_new_output_after_preparation_refused(campaign):
    c = campaign
    prepared(c)
    (c.r.RESULTS / 'unexpected.json').write_text('{}')
    with pytest.raises(ValueError, match='inventory changed'):
        c.r.submit(c.root)


def test_still_live_original_rejected_before_claim(campaign, monkeypatch):
    c = campaign
    monkeypatch.setattr(c.r, 'run_read', lambda command: '31243495_9|RUNNING\n')
    with pytest.raises(ValueError, match='still live'):
        c.r.prepare(c.root, c.manifest)
    assert not c.root.exists() and not c.r.CLAIM.exists()


@pytest.mark.parametrize('column,value', [(0, '999'), (1, '31243495_8'), (2, 'RUNNING'),
                                        (3, '0:0'), (4, 'Unknown'), (5, 'node106')])
def test_wrong_terminal_identity_refused(campaign, monkeypatch, column, value):
    c = campaign
    row = c.r.ORIGINAL_JOB.copy()
    row[column] = value
    monkeypatch.setattr(c.r, 'run_read', lambda command: '' if command[0] == 'squeue' else '|'.join(row))
    with pytest.raises(ValueError, match='terminal failure'):
        c.r.prepare(c.root, c.manifest)
    assert not c.r.CLAIM.exists()


def test_missing_source_failure_refused(campaign):
    c = campaign
    (c.r.ORIGINAL_PLAN.parent / 'logs/31243495_9.err').write_text('unrelated failure')
    with pytest.raises(ValueError, match='traceback missing'):
        c.r.prepare(c.root, c.manifest)


def test_outside_view_refused(campaign):
    c = campaign
    c.canonical.write_text('neutral')
    with pytest.raises(ValueError, match='inside the explicit'):
        c.r.prepare(c.root, c.manifest)


def test_original_plan_pin_drift_refused(campaign):
    c = campaign
    c.r.ORIGINAL_PLAN.write_text('{}')
    with pytest.raises(ValueError, match='pinned input changed'):
        c.r.prepare(c.root, c.manifest)
    assert not c.verified


def test_submit_once_with_durable_intent(campaign, monkeypatch):
    c = campaign
    prepared(c)
    calls = mock_submit(c, monkeypatch)
    result = c.r.submit(c.root)
    assert result['job_id'] == 123456 and len(calls) == 1
    with pytest.raises(ValueError, match='already attempted'):
        c.r.submit(c.root)
    assert len(calls) == 1


@pytest.mark.parametrize('stdout,code', [('not-a-job\n', 0), ('123456\n', 1), ('123456\n654321\n', 0)])
def test_bad_submission_output_preserved_never_retried(campaign, monkeypatch, stdout, code):
    c = campaign
    prepared(c)
    calls = mock_submit(c, monkeypatch, stdout=stdout, returncode=code)
    with pytest.raises(RuntimeError, match='ambiguous'):
        c.r.submit(c.root)
    assert (c.root / 'submission_ambiguous.json').exists()
    with pytest.raises(ValueError, match='already attempted'):
        c.r.submit(c.root)
    assert len(calls) == 1


def test_submission_timeout_keeps_intent_and_failure(campaign, monkeypatch):
    c = campaign
    prepared(c)
    def timed_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(c.r.subprocess, 'run', timed_out)
    with pytest.raises(RuntimeError, match='unknown'):
        c.r.submit(c.root)
    assert c.r.read(c.root / 'submission_ambiguous.json')['exception'] == 'TimeoutExpired'
    with pytest.raises(ValueError, match='already attempted'):
        c.r.submit(c.root)


def test_reconcile_matches_unique_token_and_never_submits(campaign, monkeypatch):
    c = campaign
    plan = prepared(c)
    mock_submit(c, monkeypatch, stdout='ambiguous')
    with pytest.raises(RuntimeError):
        c.r.submit(c.root)
    row = ['123456', plan['job_name'], plan['recovery_token'], plan['user'], 'mltheory', 'mltheory',
           'cpu=6,mem=59G,node=1,gres/gpu:a5000=2', '480']
    monkeypatch.setattr(c.r, 'run_read', lambda command: '|'.join(row))
    def forbidden(*args, **kwargs):
        pytest.fail('reconcile must not submit')
    monkeypatch.setattr(c.r.subprocess, 'run', forbidden)
    result = c.r.reconcile(c.root)
    assert result['job_id'] == 123456 and result['status'] == 'reconciled_from_accounting'


def test_reconcile_ambiguous_multiple_jobs_refuses(campaign, monkeypatch):
    c = campaign
    prepared(c)
    mock_submit(c, monkeypatch, stdout='ambiguous')
    with pytest.raises(RuntimeError):
        c.r.submit(c.root)
    monkeypatch.setattr(c.r, 'run_read', lambda command: '123|other\n456|other\n')
    with pytest.raises(ValueError, match='unique submitted job'):
        c.r.reconcile(c.root)
    assert not (c.root / 'submission_result.json').exists()


def setup_worker(c, monkeypatch):
    prepared(c)
    mock_submit(c, monkeypatch)
    submission = c.r.submit(c.root)
    for key, value in c.r.ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    for key, value in {'SLURM_JOB_ID': str(submission['job_id']), 'SLURMD_NODENAME': 'node105',
                       'SLURM_CPUS_PER_TASK': '6', 'SLURM_MEM_PER_NODE': '60416'}.items():
        monkeypatch.setenv(key, value)
    for key in ('SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(c.r.socket, 'gethostname', lambda: 'node105.ionic.cs.princeton.edu')
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.8.4')
    cuda = SimpleNamespace(device_count=lambda: 2, get_device_name=lambda index: 'NVIDIA RTX A5000')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    return submission, cuda


@pytest.mark.parametrize('key,value', [('OMP_NUM_THREADS', '1'), ('VLLM_USE_V1', '1'),
                                      ('HF_HUB_OFFLINE', '0'), ('SLURM_JOB_ID', '31243495'),
                                      ('SLURM_MEM_PER_NODE', '61440'), ('SLURMD_NODENAME', 'node106'),
                                      ('SLURM_CPUS_PER_TASK', '4'), ('SLURM_ARRAY_TASK_ID', '9')])
def test_worker_bad_environment_never_executes(campaign, monkeypatch, key, value):
    c = campaign
    setup_worker(c, monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        c.r.worker(c.root)
    assert not (c.root / 'runtime.json').exists()


def test_worker_checks_actual_hostname_gpu_and_vllm(campaign, monkeypatch):
    c = campaign
    submission, cuda = setup_worker(c, monkeypatch)
    monkeypatch.setattr(c.r.socket, 'gethostname', lambda: 'wash.cs.princeton.edu')
    with pytest.raises(ValueError, match='hostname'):
        c.r.worker_environment(submission)
    monkeypatch.setattr(c.r.socket, 'gethostname', lambda: 'node105.ionic.cs.princeton.edu')
    monkeypatch.setattr(cuda, 'device_count', lambda: 1)
    with pytest.raises(ValueError, match='two actual A5000'):
        c.r.worker_environment(submission)
    monkeypatch.setattr(cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(cuda, 'get_device_name', lambda index: 'NVIDIA A100')
    with pytest.raises(ValueError, match='two actual A5000'):
        c.r.worker_environment(submission)
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.9.0')
    with pytest.raises(ValueError, match='vLLM version'):
        c.r.worker_environment(submission)


def test_worker_exact_exec_preserves_runtime9_and_refuses_repeat(campaign, monkeypatch):
    c = campaign
    setup_worker(c, monkeypatch)
    original_runtime = c.r.ORIGINAL_PLAN.parent / 'runtime/9.json'
    before = original_runtime.read_bytes()
    executed = []
    monkeypatch.setattr(c.r.os, 'execv', lambda binary, argv: executed.append((binary, argv)))
    c.r.worker(c.root)
    assert executed == [(c.command[0], c.command)]
    assert original_runtime.read_bytes() == before
    runtime = c.r.read(c.root / 'runtime.json')
    assert runtime['job_id'] == 123456 and runtime['original_array_index'] == 9
    assert runtime['hostname'] == 'node105.ionic.cs.princeton.edu'
    assert runtime['visible_gpu_names'] == ['NVIDIA RTX A5000'] * 2
    with pytest.raises(ValueError, match='already started'):
        c.r.worker(c.root)
    assert len(executed) == 1


def test_worker_distinct_lock_avoids_immediate_submission_race(campaign, monkeypatch):
    c = campaign
    setup_worker(c, monkeypatch)
    path = c.root / 'submission_result.json'
    contents = path.read_bytes()
    path.unlink()
    monkeypatch.setattr(c.r.time, 'sleep', lambda seconds: path.write_bytes(contents))
    executed = []
    monkeypatch.setattr(c.r.os, 'execv', lambda *argv: executed.append(argv))
    with c.r.action_lock():
        c.r.worker(c.root)
    assert len(executed) == 1


def test_worker_wait_timeout_has_no_runtime_or_exec(campaign, monkeypatch):
    c = campaign
    setup_worker(c, monkeypatch)
    (c.root / 'submission_result.json').unlink()
    clock = iter([0, 61])
    monkeypatch.setattr(c.r.time, 'monotonic', lambda: next(clock))
    with pytest.raises(ValueError, match='submission result absent'):
        c.r.worker(c.root)
    assert not (c.root / 'runtime.json').exists()


@pytest.mark.parametrize('name', ['submission_result.json', 'submission_ambiguous.json', 'runtime.json'])
def test_orphan_execution_record_prevents_submission(campaign, name):
    c = campaign
    prepared(c)
    (c.root / name).write_text('{}')
    with pytest.raises(ValueError, match='preexisting execution evidence'):
        c.r.submit(c.root)
    assert not (c.root / 'submission_intent.json').exists()


@pytest.mark.parametrize('job_id', ['31243495', '31243495_9', '31243495_[8-9]'])
def test_user_wide_queue_matches_original_array_ids(campaign, monkeypatch, job_id):
    c = campaign
    calls = []
    def queued(command):
        calls.append(command)
        assert command[0] == 'squeue' and '-u' in command and '-j' not in command
        return job_id + '|RUNNING\n'
    monkeypatch.setattr(c.r, 'run_read', queued)
    with pytest.raises(ValueError, match='still live'):
        c.r.original_terminal()
    assert len(calls) == 1


def test_successful_user_queue_absence_accepts_unrelated_jobs(campaign, monkeypatch):
    c = campaign
    calls = []
    def inspect(command):
        calls.append(command)
        if command[0] == 'squeue':
            assert '-u' in command and '-j' not in command
            return '31252690|RUNNING\n312434950_9|PENDING\n'
        return '|'.join(c.r.ORIGINAL_JOB) + '\n'
    monkeypatch.setattr(c.r, 'run_read', inspect)
    result = c.r.original_terminal()
    assert result['original_row'] == c.r.ORIGINAL_JOB
    assert result['squeue_command'] == calls[0]
    assert len(calls) == 2


def test_user_queue_failure_is_not_absence(campaign, monkeypatch):
    c = campaign
    def unavailable(command):
        raise ValueError('scheduler inspection failed: connection error')
    monkeypatch.setattr(c.r, 'run_read', unavailable)
    with pytest.raises(ValueError, match='scheduler inspection failed'):
        c.r.original_terminal()
