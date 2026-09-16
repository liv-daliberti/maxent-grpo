"""Scratch-only tests of explicit transport; no scheduler or scientific process."""
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import importlib.util
import json
import os
import pwd
from pathlib import Path
import re
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

REAL_RUN = subprocess.run
SOURCE = Path(__file__).resolve().parents[1] / 'artifacts/modebench_scale_frozen_stage_transport_v2_20260912.py'
spec = importlib.util.spec_from_file_location('frozen_stage_transport_tested', SOURCE)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else None
    if mode is not None: path.chmod(0o600)
    path.write_text(json.dumps(value, sort_keys=True))
    if mode is not None: path.chmod(mode)


def mutate(path, change):
    value = transport.read(path); change(value); write(path, value)


def completed_gate_fixture(workspace, authority_root, activation_path, runtime_path, monkeypatch):
    """Ten original cells and forty pinned receipts, wholly inside scratch."""
    original_plan = workspace / 'original_development/plan.json'
    cells = []; receipts = []; task_paths = []
    for cell in range(10):
        tasks = original_plan.parent / f'tasks_{cell}.json'
        values = []
        for tier in range(4):
            receipt = original_plan.parent / f'receipt_{cell}_{tier}.json'
            write(receipt, {'status': 'complete', 'cell': cell, 'tier': tier})
            receipts.append(receipt); values.append({'output': str(receipt)})
        write(tasks, values); task_paths.append(tasks); cells.append({'tasks': str(tasks)})
    write(original_plan, {'cells': cells})
    monkeypatch.setattr(transport, 'ORIGINAL_PLAN', original_plan)
    reconciliation=workspace/'execution_reconciliation.json';verifier=workspace/'reconciliation_verifier.py'
    verifier.write_text('# reviewed fixture verifier')
    monkeypatch.setattr(transport,'RECONCILIATION',reconciliation)
    monkeypatch.setattr(transport,'RECONCILIATION_VERIFIER',verifier)
    monkeypatch.setattr(transport,'RECONCILIATION_VERIFIER_SHA',transport.digest(verifier))
    write(reconciliation,{'schema':'modebench_scale_frozen_recovery_execution_reconciliation_v2',
        'status':'verified_scientific_outputs_with_failed_execution','scheduler_success':False,
        'scientific_outputs_complete':True,'exit_cause':'unknown',
        'recovery_execution':{'job_id':31253118,'state':'FAILED','exit_code':'1:0','end_utc':'2026-09-12T01:59:59'},
        'files_sha256':{str(p):transport.digest(p) for p in receipts}})
    activation = transport.read(activation_path)
    activation['inputs_sha256'].update({str(path): transport.digest(path) for path in [original_plan, *task_paths,reconciliation,verifier]})
    write(activation_path, activation)
    write(authority_root / 'activation.sha256.json', {'sha256': transport.digest(activation_path)})
    completion = authority_root / 'recovery_completion.json'
    write(completion, {'schema':'modebench_scale_wash_reconciled_completion_v2','scheduler_success':False,'exit_cause':'unknown',
        'reconciliation_path':str(reconciliation),'reconciliation_sha256':transport.digest(reconciliation),
        'at_utc': '2026-09-12T02:00:00+00:00',
        'commands': [['squeue', '-u', transport.pwd.getpwuid(os.getuid()).pw_name, '-r', '-h', '-o', '%i|%T'],
            ['sacct', '-j', '31253118', '-X', '-n', '-P', '--format=JobIDRaw,State,ExitCode,End,NodeList']],
        'stdout': ['', '31253118|FAILED|1:0|2026-09-12T01:59:59|node105\n'],
        'receipts_sha256': {str(path): transport.digest(path) for path in receipts}})
    runtime = transport.read(runtime_path)
    runtime.update(authority_sha256=transport.digest(activation_path),
                   recovery_completion_path=str(completion), recovery_completion_sha256=transport.digest(completion))
    write(runtime_path, runtime)
    return SimpleNamespace(path=completion, receipts=receipts, original_plan=original_plan, tasks=task_paths)


class ScientificStub:
    """Imitate only local prepare/verify/submit; explicit runner owns transport."""
    def __init__(self, launcher_path):
        self.__file__ = str(launcher_path)
        self.calls = []

    def validate_dependency_ids(self, values):
        assert isinstance(values, (tuple, list))
        assert all(type(value) is int and value > 0 for value in values)
        assert len(values) == len(set(values))
        self.calls.append(('validate_dependencies', list(values)))
        return sorted(values)

    def prepare(self, campaign_root, cells, models, *, dependency_ids=(), pins=None):
        self.calls.append(('prepare', campaign_root, cells, models, dependency_ids, pins))
        return {'delegated': True, 'dependency_ids': dependency_ids}

    def verify(self, path, *, fresh=False):
        self.calls.append(('verify', path, fresh))
        plan = transport.read(path)
        assert transport.read(path.parent / 'plan.sha256.json')['sha256'] == transport.digest(path)
        assert (path.parent / 'worker.slurm').read_text() == '# original scientific worker\n'
        if fresh:
            assert not any((path.parent / 'runtime').iterdir())
        return plan

    def submit(self, path, runner):
        self.calls.append(('submit', path))
        plan = self.verify(path, fresh=True)
        transport.atomic_new(path.parent / 'submission_intent.json', {
            'at': transport.now(), 'plan_sha256': transport.digest(path),
            'command': plan['submit_command'], 'status': 'submission_attempt_started'})
        result = runner(plan['submit_command'], text=True, capture_output=True, check=False, timeout=60)
        evidence = {'at': transport.now(), 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', result.stdout.strip())
        if result.returncode != 0 or match is None:
            transport.atomic_new(path.parent / 'submission_ambiguous.json', evidence)
            raise RuntimeError('durable original submission requires reconciliation')
        evidence.update(status='submitted', array_job_id=int(match[1]),
                        cells=[{'array_index': i, 'cell': cell['id']} for i, cell in enumerate(plan['cells'])])
        transport.atomic_new(path.parent / 'submission_result.json', evidence)
        return evidence


@pytest.mark.parametrize('dependencies,expected', [([], [31253118]), ([7], [7, 31253118]),
    ([31253118, 7], [7, 31253118])])
def test_prepare_delegates_science_unchanged_and_only_adds_recovery(fixture, dependencies, expected):
    original = fixture.original
    proxy = fixture.proxy
    cells = [{'id': 'fixture', 'tasks': [{'must': 'remain identical'}]}]
    models = {'14b': '/model'}; pins = {'/source': 'digest'}
    before = copy.deepcopy((cells, models, pins, dependencies))
    result = proxy.prepare(fixture.stage, cells, models, dependency_ids=dependencies, pins=pins)
    assert result['dependency_ids'] == expected
    _, root, actual_cells, actual_models, actual_dependencies, actual_pins = original.calls[-1]
    assert root == fixture.stage and actual_dependencies == expected
    assert actual_cells is cells and actual_models is models and actual_pins is pins
    assert (cells, models, pins, dependencies) == before
    assert original.calls[0] == ('validate_dependencies', dependencies)


@pytest.mark.parametrize('dependencies', [[True], [-1], [4, 4], ['4']])
def test_invalid_dependencies_never_reach_scientific_prepare(fixture, dependencies):
    original = fixture.original
    with pytest.raises(AssertionError):
        fixture.proxy.prepare(fixture.stage, [], {}, dependency_ids=dependencies)
    assert not any(call[0] == 'prepare' for call in original.calls)


def test_effective_command_changes_only_script_and_keeps_plan_argument(tmp_path):
    path = tmp_path / 'plan.json'
    canonical = ['sbatch', '--parsable', '--array=0-3%2', '--dependency=afterany:7:31253118',
                 '--gres=gpu:a5000:2', str(path.parent / 'worker.slurm'), str(path)]
    before = list(canonical)
    effective = transport.effective_command(path, canonical)
    assert effective[:-2] == canonical[:-2]
    assert effective[-2:] == [str(path.parent / 'frozen_transport/worker.slurm'), str(path)]
    assert canonical == before


@pytest.mark.parametrize('command', [[], ['sbatch'], ['other', '--parsable', 'worker', 'plan'],
    ['sbatch', '--parsable', '/other/worker.slurm', '/other/plan.json']])
def test_wrong_canonical_command_route_rejected(tmp_path, command):
    with pytest.raises(ValueError): transport.effective_command(tmp_path / 'plan.json', command)


def test_atomic_ledger_never_overwrites(tmp_path):
    path = tmp_path / 'record.json'; transport.atomic_new(path, {'first': True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError): transport.atomic_new(path, {'second': True})
    assert path.read_bytes() == before


class WouldExec(RuntimeError):
    pass



@pytest.fixture
def fixture(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    artifacts = workspace / 'composite'; artifacts.mkdir()
    authority_root = workspace / 'authority'; authority_root.mkdir()
    paths = {}
    for key in ('SOURCE', 'LAUNCHER', 'RECOVERY_HELPER', 'RUNNER', 'PYTHON'):
        path = workspace / (key.lower() + '.py'); path.write_text('# ' + key + ' fixture\n'); paths[key] = path
        monkeypatch.setattr(transport, key, path)
    monkeypatch.setattr(transport, 'ROOT', workspace)
    monkeypatch.setattr(transport, 'ARTIFACTS', artifacts)
    monkeypatch.setattr(transport, 'AUTHORITY_ROOT', authority_root)
    for key in ('LAUNCHER', 'RECOVERY_HELPER', 'RUNNER'):
        monkeypatch.setattr(transport, key + '_SHA', transport.digest(paths[key]))
    view_dir = workspace / 'view'; view_dir.mkdir()
    proot = view_dir / 'proot'; proot.write_text('# proot fixture\n')
    preserved = view_dir / 'preserved.py'; preserved.write_text('original\n')
    copy_path = view_dir / 'copy.py'; copy_path.write_text('original\n')
    canonical = workspace / 'templates.py'; canonical.write_text('original\n')
    manifest = view_dir / 'manifest.json'
    manifest_value = {'proot': {'path': str(proot), 'sha256': transport.digest(proot)},
        'preserved_source': {'path': str(preserved), 'sha256': transport.digest(preserved)},
        'mappings': [{'source': str(copy_path), 'destination': str(canonical), 'sha256': transport.digest(copy_path)}]}
    write(manifest, manifest_value)
    (view_dir / 'manifest.sha256').write_text(transport.digest(manifest) + '  manifest.json\n')
    lock = artifacts / 'controller.lock'; fence_fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fence_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    info = os.fstat(fence_fd); pid = os.getpid()
    ticks = int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19])
    command = [part.decode() for part in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if part]
    identity = {'host': 'wash.cs.princeton.edu', 'uid': os.getuid(), 'lock_path': str(lock),
                'lock_inode': info.st_ino, 'lock_device': info.st_dev, 'fence_fd': fence_fd,
                'owner_pid': pid, 'owner_start_ticks': ticks}
    activation_path = authority_root / 'activation.json'
    activation = {'schema': 'modebench_scale_wash_authority_v2', 'root': str(authority_root),
        **{key: identity[key] for key in ('host', 'uid', 'lock_path', 'lock_inode')},
        'transport': str(paths['SOURCE']), 'transport_sha256': transport.digest(paths['SOURCE']),
        'recovery_job_id': transport.RECOVERY_JOB, 'view_manifest': str(manifest),
        'inputs_sha256': {str(manifest): transport.digest(manifest)}}
    write(activation_path, activation)
    write(authority_root / 'activation.sha256.json', {'sha256': transport.digest(activation_path)})
    runtime_path = authority_root / 'outer_runtime.json'
    runtime = {'schema': 'modebench_scale_wash_authority_runtime_v2', **identity,
               'owner_command': command, 'authority_path': str(activation_path),
               'authority_sha256': transport.digest(activation_path)}
    write(runtime_path, runtime)
    gate = completed_gate_fixture(workspace, authority_root, activation_path, runtime_path, monkeypatch)
    authority = {**identity, 'authority_path': str(activation_path), 'authority_sha256': transport.digest(activation_path),
        'authority_runtime_path': str(runtime_path), 'authority_runtime_sha256': transport.digest(runtime_path),
        'view_manifest': str(manifest), 'view_manifest_sha256': transport.digest(manifest)}
    stage = artifacts / 'revision_development'; stage.mkdir(); (stage / 'runtime').mkdir()
    plan_path = stage / 'plan.json'
    plan = {'dependency_ids': [7, transport.RECOVERY_JOB], 'python': str(paths['PYTHON']),
        'cells': [{'id': 'level5_graph_coloring', 'command': [str(paths['PYTHON']), '/sealed/evaluator.py', '--original']},
                  {'id': 'level5_mathir', 'command': [str(paths['PYTHON']), '/sealed/evaluator.py', '--mathir']}],
        'hardware': {'nodes': 'node105', 'cpus': 6, 'memory': '60G'},
        'runtime_profile': {'thread_environment': {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'}},
        'submit_command': ['sbatch', '--parsable', '--array=0-1%2', '--dependency=afterany:7:31253118',
                           '--gres=gpu:a5000:2', str(stage / 'worker.slurm'), str(plan_path)]}
    write(plan_path, plan); write(stage / 'plan.sha256.json', {'sha256': transport.digest(plan_path)})
    (stage / 'worker.slurm').write_text('# original scientific worker\n')
    original = ScientificStub(paths['LAUNCHER'])
    process_calls = []; exec_calls = []
    def forbid_process(*args, **kwargs):
        raise AssertionError('unexpected real subprocess route')
    def forbid_exec(*args, **kwargs):
        raise AssertionError('unexpected real exec route')
    monkeypatch.setattr(transport.subprocess, 'run', forbid_process)
    monkeypatch.setattr(transport.os, 'execv', forbid_exec)
    monkeypatch.setattr(transport.socket, 'gethostname', lambda: 'wash.cs.princeton.edu')
    def checked_view(path):
        assert path == manifest
        value = transport.read(path)
        assert transport.digest(path) + '  manifest.json\n' == (path.parent / 'manifest.sha256').read_text()
        for item in (value['proot'], value['preserved_source']):
            transport.require(transport.digest(item['path']) == item['sha256'], 'view dependency changed')
        mapping = value['mappings'][0]
        transport.require(transport.digest(mapping['source']) == transport.digest(mapping['destination']) == mapping['sha256'],
                          'outside the frozen view')
        return value
    def fake_module(path, expected, name):
        transport.require(transport.digest(path) == expected, 'sealed adapter dependency changed')
        if path == transport.RECONCILIATION_VERIFIER:
            return SimpleNamespace(verify_reconciliation=lambda:transport.read(transport.RECONCILIATION))
        if path == paths['LAUNCHER']: return original
        assert path == paths['RECOVERY_HELPER']
        return SimpleNamespace(frozen_view=checked_view)
    monkeypatch.setattr(transport, 'module', fake_module)
    def dispatch(result=None, exception=None):
        def run(command, **kwargs):
            process_calls.append((list(command), dict(kwargs)))
            if exception is not None: raise exception
            return result or SimpleNamespace(returncode=0, stdout='987654\n', stderr='')
        monkeypatch.setattr(transport.subprocess, 'run', run)
    def worker_environment(index=0):
        mock_gpu_runtime(monkeypatch)
        monkeypatch.setattr(transport.socket, 'gethostname', lambda: 'node105.ionic.cs.princeton.edu')
        for key, value in {'SLURM_JOB_ID': '987655', 'SLURM_ARRAY_JOB_ID': '987654', 'SLURM_ARRAY_TASK_ID': str(index),
            'SLURMD_NODENAME': 'node105', 'SLURM_CPUS_PER_TASK': '6', 'SLURM_MEM_PER_NODE': '61440',
            'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'VLLM_USE_V1': '0',
            'VLLM_ATTENTION_BACKEND': 'XFORMERS', 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPYCACHEPREFIX': '/tmp/NONPRODUCTION-scale-frozen-pycache-fixture/pycache'}.items():
            monkeypatch.setenv(key, value)
        def capture(executable, argv):
            exec_calls.append((executable, list(argv))); raise WouldExec('mocked scientific worker exec')
        monkeypatch.setattr(transport.os, 'execv', capture)
    proxy = transport.LauncherProxy(original, authority, manifest)
    yield SimpleNamespace(root=workspace, stage=stage, path=plan_path, plan=plan, original=original, proxy=proxy,
        authority=authority, activation=activation_path, authority_runtime=runtime_path, manifest=manifest,
        view_dir=view_dir, canonical=canonical, paths=paths, lock=lock, fence_fd=fence_fd, gate=gate,
        dispatch=dispatch, process_calls=process_calls, worker_environment=worker_environment, exec_calls=exec_calls)
    try: os.close(fence_fd)
    except OSError: pass


def prepare(fixture):
    return transport.prepare_transport(fixture.original, fixture.path, fixture.authority, fixture.manifest)


def submit(fixture):
    fixture.dispatch(); return fixture.proxy.submit(fixture.path)


def test_install_checks_authority_and_does_not_launch(fixture):
    controller = SimpleNamespace(launch=fixture.original)
    assert transport.install(controller, fixture.authority, fixture.manifest) is fixture.original
    assert isinstance(controller.launch, transport.LauncherProxy)
    assert fixture.process_calls == []
    with pytest.raises(ValueError, match='already installed'):
        transport.install(controller, fixture.authority, fixture.manifest)


def test_transport_preserves_original_plan_worker_command_and_pins(fixture):
    originals = {path: path.read_bytes() for path in [fixture.path, fixture.stage / 'plan.sha256.json', fixture.stage / 'worker.slurm']}
    tx = prepare(fixture)
    assert originals == {path: path.read_bytes() for path in originals}
    assert tx['canonical_submit_command'] == fixture.plan['submit_command']
    assert tx['effective_submit_command'] == transport.effective_command(fixture.path, fixture.plan['submit_command'])
    assert tx['original_worker_preserved'] is True and tx['scientific_plan_or_evaluator_argv_changed'] is False
    _, verified = transport.verify_transport(fixture.original, fixture.path)
    assert verified == tx and fixture.process_calls == []
    wrapper = (fixture.stage / 'frozen_transport/worker.slurm').read_text()
    assert str(fixture.paths['RUNNER']) + ' exec --manifest ' + str(fixture.manifest) in wrapper
    assert ' -- ' + str(fixture.paths['PYTHON']) + ' -B ' + str(fixture.paths['SOURCE']) + ' worker --plan' in wrapper
    assert '"$1" --index "${SLURM_ARRAY_TASK_ID:?}"' in wrapper
    for path in [fixture.activation, fixture.authority_runtime, fixture.activation.parent / 'activation.sha256.json',
                 fixture.manifest.parent / 'manifest.sha256']:
        assert tx['inputs_sha256'][str(path)] == transport.digest(path)


def test_original_submit_delegates_and_ledgers_both_commands(fixture):
    submitted = submit(fixture)
    txroot = fixture.stage / 'frozen_transport'; tx = transport.read(txroot / 'plan.json')
    intent = transport.read(fixture.stage / 'submission_intent.json')
    effective_intent = transport.read(txroot / 'submission_intent.json')
    effective_result = transport.read(txroot / 'submission_result.json')
    assert submitted['array_job_id'] == 987654
    assert fixture.process_calls == [(tx['effective_submit_command'], {'text': True, 'capture_output': True, 'check': False, 'timeout': 60, 'close_fds': True})]
    assert intent['command'] == fixture.plan['submit_command']
    assert effective_intent['canonical_command'] == intent['command']
    assert effective_intent['effective_command'] == tx['effective_submit_command']
    assert effective_intent['canonical_intent_sha256'] == transport.digest(fixture.stage / 'submission_intent.json')
    assert effective_result['intent_sha256'] == transport.digest(txroot / 'submission_intent.json')
    assert ('submit', fixture.path) in fixture.original.calls


@pytest.mark.parametrize('failure', ['timeout', 'oserror', 'nonzero', 'bad_stdout'])
def test_ambiguous_submission_keeps_intents_and_refuses_retry(fixture, failure):
    if failure == 'timeout': fixture.dispatch(exception=subprocess.TimeoutExpired(['sbatch'], 60))
    elif failure == 'oserror': fixture.dispatch(exception=OSError('transport unavailable'))
    else: fixture.dispatch(result=SimpleNamespace(returncode=17 if failure == 'nonzero' else 0,
                                                  stdout='' if failure == 'nonzero' else 'ambiguous', stderr='failure'))
    with pytest.raises((RuntimeError, OSError, subprocess.TimeoutExpired)):
        fixture.proxy.submit(fixture.path)
    assert (fixture.stage / 'submission_intent.json').is_file()
    assert (fixture.stage / 'frozen_transport/submission_intent.json').is_file()
    assert len(fixture.process_calls) == 1
    with pytest.raises((ValueError, FileExistsError)):
        fixture.proxy.submit(fixture.path)
    assert len(fixture.process_calls) == 1


def test_double_execute_does_not_dispatch_twice(fixture):
    submit(fixture)
    with pytest.raises(FileExistsError): fixture.proxy.execute(fixture.path, fixture.plan['submit_command'])
    assert len(fixture.process_calls) == 1


def test_concurrent_submitters_dispatch_at_most_once(fixture):
    fixture.dispatch()
    def attempt():
        try: return fixture.proxy.submit(fixture.path)
        except (ValueError, FileExistsError): return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(fixture.process_calls) == 1


@pytest.mark.parametrize('name', ['plan.sha256.json', 'worker.slurm'])
def test_missing_original_artifacts_refuse_transport(fixture, name):
    (fixture.stage / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError, AssertionError)): prepare(fixture)
    assert fixture.process_calls == []


@pytest.mark.parametrize('name', ['SOURCE', 'LAUNCHER', 'RECOVERY_HELPER', 'RUNNER'])
def test_pinned_dependency_drift_refuses_verification(fixture, name):
    prepare(fixture); fixture.paths[name].write_text('changed')
    with pytest.raises(ValueError): transport.verify_transport(fixture.original, fixture.path)


def test_unknown_or_symlink_stage_root_rejected(fixture):
    with pytest.raises(ValueError): transport.stage_path(fixture.root / 'elsewhere/plan.json')
    link = fixture.root / 'linked'; link.symlink_to(fixture.stage, target_is_directory=True)
    with pytest.raises(ValueError): transport.stage_path(link / 'plan.json')


def test_wrong_view_argument_and_neutral_source_refused(fixture):
    other = fixture.view_dir / 'other.json'; other.write_bytes(fixture.manifest.read_bytes())
    with pytest.raises(ValueError, match='view differs'): transport.prepare_transport(fixture.original, fixture.path, fixture.authority, other)
    fixture.canonical.write_text('neutral')
    with pytest.raises(ValueError, match='frozen view'): prepare(fixture)
    assert fixture.process_calls == []


def test_missing_operational_pin_cannot_be_hidden_by_rehash(fixture):
    prepare(fixture); txroot = fixture.stage / 'frozen_transport'
    mutate(txroot / 'plan.json', lambda tx: tx['inputs_sha256'].pop(str(fixture.authority_runtime)))
    write(txroot / 'plan.sha256.json', {'sha256': transport.digest(txroot / 'plan.json')})
    with pytest.raises(ValueError, match='omits required'): transport.verify_transport(fixture.original, fixture.path)


def test_worker_records_transport_then_execs_only_original_scientific_worker(fixture):
    submit(fixture); os.close(fixture.fence_fd)  # Historical worker verification must not require live owner FD.
    fixture.worker_environment(index=1)
    with pytest.raises(WouldExec): transport.worker(fixture.path, 1)
    record = transport.read(fixture.stage / 'frozen_transport/runtime/1.json')
    assert record['evaluator_command'] == fixture.plan['cells'][1]['command']
    assert fixture.exec_calls == [(str(fixture.paths['PYTHON']), [str(fixture.paths['PYTHON']), '-B',
        str(fixture.paths['LAUNCHER']), 'worker', '--plan', str(fixture.path), '--index', '1'])]
    assert not (fixture.stage / 'runtime/1.json').exists()
    with pytest.raises(ValueError, match='already attempted'): transport.worker(fixture.path, 1)
    assert len(fixture.exec_calls) == 1


@pytest.mark.parametrize('key,value', [('SLURM_ARRAY_JOB_ID', 'different'), ('SLURM_ARRAY_TASK_ID', '1'),
    ('SLURMD_NODENAME', 'node007'), ('SLURM_CPUS_PER_TASK', '8'), ('SLURM_MEM_PER_NODE', '60416'),
    ('OMP_NUM_THREADS', '1'), ('VLLM_USE_V1', '1'), ('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')])
def test_wrong_worker_identity_or_runtime_refuses_exec(fixture, monkeypatch, key, value):
    submit(fixture); fixture.worker_environment(); monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match='worker identity'): transport.worker(fixture.path, 0)
    assert fixture.exec_calls == []


@pytest.mark.parametrize('target,change', [
    ('submission_result.json', lambda r: r.update(array_job_id=12)),
    ('frozen_transport/submission_result.json', lambda r: r.update(returncode=1)),
    ('frozen_transport/submission_result.json', lambda r: r.update(intent_sha256='bad')),
    ('frozen_transport/submission_intent.json', lambda r: r.update(transport_plan_sha256='bad')),
    ('frozen_transport/submission_intent.json', lambda r: r.update(status='not attempted')),
    ('submission_intent.json', lambda r: r.update(plan_sha256='bad')),
])
def test_worker_refuses_mislinked_canonical_and_effective_ledgers(fixture, target, change):
    submit(fixture); mutate(fixture.stage / target, change)
    if target == 'frozen_transport/submission_intent.json':
        # Preserve envelope self-consistency; semantic linkage still must reject.
        mutate(fixture.stage / 'frozen_transport/submission_result.json',
               lambda r: r.update(intent_sha256=transport.digest(fixture.stage / target)))
    fixture.worker_environment()
    with pytest.raises(ValueError, match='submission linkage'): transport.worker(fixture.path, 0)
    assert fixture.exec_calls == []


def test_worker_missing_submission_evidence_fails_without_sleeping(fixture, monkeypatch):
    prepare(fixture); fixture.worker_environment()
    counter = iter([0, 61]); monkeypatch.setattr(transport.time, 'monotonic', lambda: next(counter))
    monkeypatch.setattr(transport.time, 'sleep', lambda _: pytest.fail('unexpected wait'))
    with pytest.raises(ValueError, match='submission results absent'): transport.worker(fixture.path, 0)
    assert fixture.exec_calls == []


@pytest.mark.parametrize('field,value', [('host', 'soak.cs.princeton.edu'), ('owner_pid', 1),
    ('owner_start_ticks', '123'), ('lock_inode', 1), ('lock_device', 1), ('uid', -1)])
def test_declared_authority_must_match_hashed_records(fixture, field, value):
    authority = dict(fixture.authority); authority[field] = value
    controller = SimpleNamespace(launch=fixture.original)
    with pytest.raises(ValueError): transport.install(controller, authority, fixture.manifest)
    assert controller.launch is fixture.original and fixture.process_calls == []


def test_actual_host_is_checked_before_scientific_prepare(fixture, monkeypatch):
    monkeypatch.setattr(transport.socket, 'gethostname', lambda: 'soak.cs.princeton.edu')
    with pytest.raises(ValueError, match='actual wash authority'):
        fixture.proxy.prepare(fixture.stage, [], {})
    assert not any(call[0] == 'prepare' for call in fixture.original.calls)


@pytest.mark.parametrize('field,value', [('owner_start_ticks', '123'), ('owner_command', ['/different/owner'])])
def test_actual_proc_owner_identity_checked_after_consistent_record_rehash(fixture, field, value):
    mutate(fixture.authority_runtime, lambda record: record.update({field: value}))
    fixture.authority['authority_runtime_sha256'] = transport.digest(fixture.authority_runtime)
    if field in fixture.authority: fixture.authority[field] = value
    proxy = transport.LauncherProxy(fixture.original, fixture.authority, fixture.manifest)
    with pytest.raises(ValueError, match='owner is absent, replaced'):
        proxy.prepare(fixture.stage, [], {})
    assert not any(call[0] == 'prepare' for call in fixture.original.calls)


def test_named_fence_replacement_blocks_prepare_before_delegation(fixture):
    fixture.lock.unlink(); fixture.lock.write_text('new unrelated lock')
    with pytest.raises(ValueError, match='fence descriptor/path changed'):
        fixture.proxy.prepare(fixture.stage, [], {})
    assert not any(call[0] == 'prepare' for call in fixture.original.calls)


def test_closed_fence_blocks_prepare_before_delegation(fixture):
    os.close(fixture.fence_fd)
    with pytest.raises(OSError): fixture.proxy.prepare(fixture.stage, [], {})
    assert not any(call[0] == 'prepare' for call in fixture.original.calls)


def test_same_inode_with_unrelated_open_description_is_not_owned_fence(fixture):
    second = os.open(fixture.lock, os.O_RDWR)
    try:
        mutate(fixture.authority_runtime, lambda record: record.update(fence_fd=second))
        authority = {**fixture.authority, 'fence_fd': second,
                     'authority_runtime_sha256': transport.digest(fixture.authority_runtime)}
        proxy = transport.LauncherProxy(fixture.original, authority, fixture.manifest)
        with pytest.raises((ValueError, BlockingIOError)): proxy.prepare(fixture.stage, [], {})
        assert not any(call[0] == 'prepare' for call in fixture.original.calls)
    finally: os.close(second)


def test_missing_or_changed_authority_record_rejected(fixture):
    fixture.authority_runtime.unlink()
    with pytest.raises(ValueError, match='canonical authority record'): prepare(fixture)
    assert fixture.process_calls == []


def test_concurrent_workers_can_exec_original_worker_at_most_once(fixture):
    submit(fixture); fixture.worker_environment()
    def attempt():
        try: transport.worker(fixture.path, 0)
        except WouldExec: return 'exec'
        except (ValueError, FileExistsError): return 'refused'
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == ['exec', 'refused']
    assert len(fixture.exec_calls) == 1


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'; root.mkdir()
    def file(name, content='fixture'):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content); return path
    artifacts = root/'composite'; authority_root = root/'authority'
    paths = {
        'ROOT':root, 'SOURCE':file('transport.py'), 'LAUNCHER':file('launcher.py'),
        'RECOVERY_HELPER':file('recovery.py'), 'RUNNER':file('runner.py'),
        'PYTHON':file('literal-venv/python'), 'ARTIFACTS':artifacts, 'AUTHORITY_ROOT':authority_root,
    }
    for name, value in paths.items(): monkeypatch.setattr(transport, name, value)
    monkeypatch.setattr(transport, 'LAUNCHER_SHA', transport.digest(paths['LAUNCHER']))
    manifest = file('view/manifest.json', '{}'); file('view/manifest.sha256')
    preserved = file('view/preserved.py'); mapped = file('view/mapped.py'); proot = file('proot')
    view = {'proot':{'path':str(proot)}, 'preserved_source':{'path':str(preserved)},
            'mappings':[{'source':str(mapped)}]}
    monkeypatch.setattr(transport, 'view', lambda path: view)
    lock = file('composite/controller.lock'); fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    info = os.fstat(fd)
    proc = Path('/proc')/str(os.getpid())
    ticks = (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]
    command = [part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
    driver = file('driver.py')
    activation = {'schema':'modebench_scale_wash_authority_v2', 'root':str(authority_root),
        'host':'wash.cs.princeton.edu', 'uid':os.getuid(), 'lock_path':str(lock), 'lock_inode':info.st_ino,
        'transport':str(paths['SOURCE']), 'transport_sha256':transport.digest(paths['SOURCE']),
        'recovery_job_id':31253118, 'view_manifest':str(manifest),
        'inputs_sha256':{str(p):transport.digest(p) for p in [driver, manifest, paths['SOURCE']]}}
    activation_path = file('authority/activation.json', json.dumps(activation))
    file('authority/activation.sha256.json', json.dumps({'sha256':transport.digest(activation_path)}))
    runtime = {'schema':'modebench_scale_wash_authority_runtime_v2', 'owner_pid':os.getpid(),
        'owner_start_ticks':ticks, 'owner_command':command, 'host':'wash.cs.princeton.edu',
        'uid':os.getuid(), 'lock_path':str(lock), 'lock_inode':info.st_ino, 'lock_device':info.st_dev,
        'fence_fd':fd, 'authority_path':str(activation_path), 'authority_sha256':transport.digest(activation_path)}
    runtime_path = file('authority/outer_runtime.json', json.dumps(runtime))
    gate = completed_gate_fixture(root, authority_root, activation_path, runtime_path, monkeypatch)
    receipt, completion = gate.receipts[0], gate.path
    runtime = transport.read(runtime_path)
    authority = {k:runtime[k] for k in ('owner_pid','owner_start_ticks','host','uid','lock_path',
        'lock_inode','lock_device','fence_fd','authority_path','authority_sha256')}
    authority.update(authority_runtime_path=str(runtime_path), authority_runtime_sha256=transport.digest(runtime_path),
                     view_manifest=str(manifest), view_manifest_sha256=transport.digest(manifest))
    monkeypatch.setattr(transport.socket, 'gethostname', lambda: 'wash.cs.princeton.edu')
    plan_path = file('composite/revision_development/plan.json')
    (plan_path.parent/'runtime').mkdir()
    original_worker = file('composite/revision_development/worker.slurm', '# original scientific worker\n')
    plan = {'cells':[{'id':'level4_graph_coloring', 'command':[str(paths['PYTHON']), 'evaluator', '--resume']}],
        'dependency_ids':[123,31253118], 'hardware':{'memory':'60G','nodes':'node105','cpus':6},
        'runtime_profile':{'thread_environment':{'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}},
        'submit_command':['sbatch','--parsable','--partition=mltheory','--account=mltheory',
             '--nodelist=node105','--nodes=1','--ntasks=1','--gres=gpu:a5000:2',
             '--cpus-per-task=6','--mem=60G','--time=08:00:00','--array=0-0%1',
             '--job-name=modebench-scale-domains-dev','--no-requeue','--export=ALL',
             '--dependency=afterany:123:31253118',str(original_worker),str(plan_path)]}
    write(plan_path, plan); write(plan_path.parent/'plan.sha256.json', {'sha256':transport.digest(plan_path)})
    original = ScientificStub(paths['LAUNCHER'])
    proxy = transport.LauncherProxy(original, authority, manifest)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs == {'text':True,'capture_output':True,'check':False,'timeout':60,'close_fds':True}
        assert (plan_path.parent/'submission_intent.json').is_file()
        assert (plan_path.parent/'frozen_transport/submission_intent.json').is_file()
        return SimpleNamespace(returncode=0, stdout='789\n', stderr='')
    monkeypatch.setattr(transport.subprocess, 'run', run)
    monkeypatch.setattr(transport, 'module', lambda path,*args:
        SimpleNamespace(verify_reconciliation=lambda:transport.read(transport.RECONCILIATION))
        if path==transport.RECONCILIATION_VERIFIER else original)
    try:
        yield SimpleNamespace(authority=authority, fd=fd, original=original, proxy=proxy, path=plan_path, plan=plan,
            root=root, manifest=manifest, directory=plan_path.parent/'frozen_transport', calls=calls,
            driver=driver, receipt=receipt, completion=completion, runtime=runtime_path, activation=activation_path)
    finally:
        os.close(fd)


def test_authority_rechecks_operational_and_completed_receipt_closure(scratch):
    c = scratch
    pins = transport.authority_pins(c.authority, live=True)
    assert pins[str(c.driver)] == transport.digest(c.driver)
    assert pins[str(c.receipt)] == transport.digest(c.receipt)
    assert pins[str(c.completion)] == transport.digest(c.completion)


@pytest.mark.parametrize('name', ['driver', 'receipt', 'activation', 'runtime'])
def test_changed_authority_input_fails_before_any_transport(scratch, name):
    c = scratch; getattr(c,name).write_text('changed')
    with pytest.raises(ValueError): c.proxy.submit(c.path)
    assert not c.directory.exists() and not c.calls


def test_released_fence_is_rejected_without_reacquiring(scratch):
    c = scratch
    fcntl.flock(c.fd, fcntl.LOCK_UN)
    with pytest.raises(ValueError, match='no longer owns'): transport.authority_pins(c.authority, live=True)
    with open(c.authority['lock_path'], 'r+') as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not c.directory.exists()


def test_other_owner_does_not_prove_inherited_fd_ownership(scratch):
    c = scratch; fcntl.flock(c.fd, fcntl.LOCK_UN)
    with open(c.authority['lock_path'], 'r+') as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='no longer owns'): transport.assert_fence(c.authority)


def test_authority_verification_retains_lock_and_rejects_replaced_path(scratch):
    c = scratch
    transport.assert_fence(c.authority)
    with open(c.authority['lock_path'], 'r+') as other:
        with pytest.raises(BlockingIOError): fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path=Path(c.authority['lock_path']); path.unlink(); path.touch()
    with pytest.raises(ValueError, match='changed'): transport.assert_fence(c.authority)


def test_authority_owner_start_ticks_and_record_linkage_checked(scratch):
    c = scratch
    changed = dict(c.authority); changed['owner_start_ticks']='1'
    with pytest.raises(ValueError, match='recorded owner'): transport.authority_pins(changed,live=True)
    mutate(c.runtime, lambda value:value.update(owner_start_ticks='1'))
    changed['authority_runtime_sha256']=transport.digest(c.runtime)
    with pytest.raises(ValueError, match='absent, replaced'): transport.authority_pins(changed,live=True)


def test_submit_retains_scientific_plan_command_resources_and_records_actual_transport(scratch):
    c = scratch
    inputs=[c.path,c.path.parent/'worker.slurm',c.path.parent/'plan.sha256.json']
    before={p:p.read_bytes() for p in inputs}
    assert c.proxy.submit(c.path)['array_job_id']==789
    assert all(p.read_bytes()==data for p,data in before.items())
    assert len(c.calls)==1
    actual,kwargs=c.calls[0]
    assert actual[:-2]==c.plan['submit_command'][:-2]
    assert actual[-2:]==[str(c.directory/'worker.slurm'),str(c.path)]
    tx=transport.read(c.directory/'plan.json')
    assert tx['canonical_submit_command']==c.plan['submit_command']
    assert tx['effective_submit_command']==actual
    assert transport.verify_transport(c.original,c.path)[0]==c.plan
    wrapper=(c.directory/'worker.slurm').read_text()
    assert ' exec --manifest '+str(c.manifest)+' -- ' in wrapper
    assert str(transport.PYTHON)+' -B '+str(transport.SOURCE)+' worker --plan' in wrapper
    transport.assert_fence(c.authority)


def test_duplicate_submit_guard_before_second_scheduler_call(scratch):
    c=scratch; c.proxy.submit(c.path)
    with pytest.raises(ValueError,match='reconciliation'): c.proxy.submit(c.path)
    assert len(c.calls)==1


@pytest.mark.parametrize('name',['submission_intent.json','submission_result.json','submission_ambiguous.json'])
def test_any_existing_canonical_submission_evidence_blocks_new_transport(scratch,name):
    c=scratch; write(c.path.parent/name,{})
    with pytest.raises(ValueError,match='reconciliation'): c.proxy.submit(c.path)
    assert not c.directory.exists() and not c.calls


@pytest.mark.parametrize('kind',['timeout','oserror','invalid','nonzero'])
def test_ambiguous_submission_preserves_intents_and_never_retries(scratch,monkeypatch,kind):
    c=scratch; calls=[]
    def failed(command,**kwargs):
        calls.append(command)
        if kind=='timeout': raise subprocess.TimeoutExpired(command,60)
        if kind=='oserror': raise OSError('fixture failure')
        return SimpleNamespace(returncode=1 if kind=='nonzero' else 0,stdout='bad',stderr='fixture')
    monkeypatch.setattr(transport.subprocess,'run',failed)
    with pytest.raises((OSError,subprocess.TimeoutExpired,RuntimeError)): c.proxy.submit(c.path)
    assert (c.path.parent/'submission_intent.json').exists()
    assert (c.directory/'submission_intent.json').exists()
    assert (c.directory/('submission_ambiguous.json' if kind in ('timeout','oserror') else 'submission_result.json')).exists()
    with pytest.raises(ValueError,match='reconciliation'): c.proxy.submit(c.path)
    assert len(calls)==1


def worker_env(c,monkeypatch):
    mock_gpu_runtime(monkeypatch)
    c.proxy.submit(c.path)
    monkeypatch.setattr(transport.socket,'gethostname',lambda:'node105.ionic.cs.princeton.edu')
    for name,value in {'SLURM_ARRAY_JOB_ID':'789','SLURM_ARRAY_TASK_ID':'0','SLURMD_NODENAME':'node105',
        'SLURM_CPUS_PER_TASK':'6','SLURM_MEM_PER_NODE':'61440','OMP_NUM_THREADS':'4',
        'OPENBLAS_NUM_THREADS':'1','VLLM_USE_V1':'0','VLLM_ATTENTION_BACKEND':'XFORMERS'}.items():
        monkeypatch.setenv(name,value)
    executed=[]
    monkeypatch.setattr(transport.os,'execv',lambda executable,argv:executed.append((executable,argv)))
    return executed


def test_worker_authenticates_both_ledgers_and_executes_original_scientific_worker_once(scratch,monkeypatch):
    c=scratch; executed=worker_env(c,monkeypatch)
    transport.worker(c.path,0)
    assert executed==[(str(transport.PYTHON),[str(transport.PYTHON),'-B',str(transport.LAUNCHER),
        'worker','--plan',str(c.path),'--index','0'])]
    runtime=transport.read(c.directory/'runtime/0.json')
    assert runtime['evaluator_command']==c.plan['cells'][0]['command']
    assert runtime['hostname']=='node105.ionic.cs.princeton.edu'
    assert runtime['status']=='validated_before_unchanged_scientific_worker_exec'
    with pytest.raises(ValueError,match='already attempted'): transport.worker(c.path,0)
    assert len(executed)==1


@pytest.mark.parametrize('name,value',[('SLURM_ARRAY_JOB_ID','790'),('SLURM_ARRAY_TASK_ID','1'),
    ('SLURMD_NODENAME','node106'),('SLURM_CPUS_PER_TASK','4'),('SLURM_MEM_PER_NODE','60416'),
    ('VLLM_USE_V1','1'),('VLLM_ATTENTION_BACKEND','FLASH_ATTN'),('OMP_NUM_THREADS','6')])
def test_worker_rejects_wrong_runtime_before_original_worker(scratch,monkeypatch,name,value):
    c=scratch; executed=worker_env(c,monkeypatch); monkeypatch.setenv(name,value)
    with pytest.raises(ValueError,match='identity/backend/environment'): transport.worker(c.path,0)
    assert not executed and not (c.directory/'runtime/0.json').exists()


@pytest.mark.parametrize('field,value',[('stdout','790\n'),('returncode',1),('intent_sha256','0'*64),
    ('transport_plan_sha256','0'*64),('stderr','unexpected')])
def test_worker_rejects_mismatched_effective_submission(scratch,monkeypatch,field,value):
    c=scratch; executed=worker_env(c,monkeypatch)
    mutate(c.directory/'submission_result.json',lambda data:data.update({field:value}))
    with pytest.raises(ValueError,match='submission linkage'): transport.worker(c.path,0)
    assert not executed


@pytest.mark.parametrize('relative',['runtime/0.json','frozen_transport/runtime/0.json'])
def test_either_prior_runtime_blocks_worker_replay(scratch,monkeypatch,relative):
    c=scratch; executed=worker_env(c,monkeypatch); write(c.path.parent/relative,{})
    with pytest.raises(ValueError,match='already attempted'): transport.worker(c.path,0)
    assert not executed


def test_missing_operational_pin_rejected_even_when_transport_checksum_resealed(scratch):
    c=scratch; c.proxy.submit(c.path)
    mutate(c.directory/'plan.json',lambda data:data['inputs_sha256'].pop(str(c.driver)))
    write(c.directory/'plan.sha256.json',{'sha256':transport.digest(c.directory/'plan.json')})
    with pytest.raises(ValueError,match='omits required'): transport.verify_transport(c.original,c.path)


def test_install_uses_explicit_launcher_hook_and_rejects_second_install(scratch):
    c=scratch; controller=SimpleNamespace(launch=c.original)
    assert transport.install(controller,c.authority,c.manifest) is c.original
    assert isinstance(controller.launch,transport.LauncherProxy)
    with pytest.raises(ValueError,match='already installed'): transport.install(controller,c.authority,c.manifest)


def test_real_proot_preserves_descriptor_specific_fence_and_source_mapping(scratch):
    c = scratch
    proot = Path('/usr/libexec/apptainer/bin/proot')
    python = SOURCE.parents[1]/'var/seed_paper_eval/paper310/bin/python'
    canonical = c.root/'canonical.txt'; canonical.write_text('neutral host')
    old = c.root/'frozen.txt'; old.write_text('original guest')
    code = """import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('transport_child',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
authority=json.loads(sys.argv[2]);module.assert_fence(authority)
assert Path(sys.argv[3]).read_text()=='original guest'
print('inherited_fence_and_source_mapping_verified')
"""
    result = REAL_RUN([str(proot),'-r','/','-b',str(old)+':'+str(canonical),'-w',str(c.root),
        str(python),'-B','-c',code,str(SOURCE),json.dumps(c.authority),str(canonical)],
        pass_fds=(c.fd,),text=True,capture_output=True,check=False,timeout=30)
    assert result.returncode==0, result.stderr
    assert result.stdout.strip()=='inherited_fence_and_source_mapping_verified'
    assert canonical.read_text()=='neutral host'
    transport.assert_fence(c.authority)


def mock_gpu_runtime(monkeypatch):
    monkeypatch.setattr(transport.importlib.metadata, 'version', lambda name:'0.8.4')
    cuda = SimpleNamespace(device_count=lambda:2, get_device_name=lambda i:'NVIDIA RTX A5000')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    return cuda


@pytest.mark.parametrize('failure',['version','count','type'])
def test_worker_rejects_unqualified_actual_gpu_runtime(scratch,monkeypatch,failure):
    c=scratch; executed=worker_env(c,monkeypatch)
    if failure=='version': monkeypatch.setattr(transport.importlib.metadata,'version',lambda name:'0.9.0')
    else:
        cuda=sys.modules['torch'].cuda
        if failure=='count': cuda.device_count=lambda:1
        else: cuda.get_device_name=lambda i:'NVIDIA A100'
    with pytest.raises(ValueError,match='vLLM version|actual A5000'): transport.worker(c.path,0)
    assert not executed and not (c.directory/'runtime/0.json').exists()


def reseal_gate_runtime(fixture):
    """Corruptions may reseal envelopes, but cannot bypass semantic validation."""
    mutate(fixture.authority_runtime, lambda value: value.update(
        recovery_completion_sha256=transport.digest(fixture.gate.path)))
    fixture.authority['authority_runtime_sha256'] = transport.digest(fixture.authority_runtime)
    fixture.proxy.authority.update(fixture.authority)


@pytest.mark.parametrize('field,value', [
    ('recovery_completion_path', '/unrelated/recovery_completion.json'),
    ('recovery_completion_sha256', '0' * 64),
])
def test_completion_must_be_bound_to_recorded_runtime(fixture, field, value):
    mutate(fixture.authority_runtime, lambda record: record.update({field: value}))
    fixture.authority['authority_runtime_sha256'] = transport.digest(fixture.authority_runtime)
    fixture.proxy.authority.update(fixture.authority)
    with pytest.raises(ValueError, match='completion/runtime linkage'): prepare(fixture)
    assert not (fixture.stage / 'frozen_transport').exists() and fixture.process_calls == []


def test_completion_record_missing_is_refused(fixture):
    fixture.gate.path.unlink()
    with pytest.raises(ValueError, match='completion/runtime linkage'): prepare(fixture)
    assert not (fixture.stage / 'frozen_transport').exists()


@pytest.mark.parametrize('change,error', [
    (lambda r: r['stdout'].__setitem__(0, '31253118|RUNNING\n'), 'active job'),
    (lambda r: r['stdout'].__setitem__(1, '31253118|COMPLETED|0:0|2026-09-12T01:59:59|node105\n'), 'terminal recovery'),
    (lambda r: r['stdout'].__setitem__(1, '31253118|COMPLETED|1:0|2026-09-12T01:59:59|node105\n'), 'terminal recovery'),
    (lambda r: r['stdout'].__setitem__(1, '31253119|COMPLETED|0:0|2026-09-12T01:59:59|node105\n'), 'terminal recovery'),
    (lambda r: r['stdout'].__setitem__(1, '31253118|COMPLETED|0:0|Unknown|node105\n'), 'terminal recovery'),
    (lambda r: r['stdout'].__setitem__(1, '31253118|COMPLETED|0:0|2026-09-12T01:59:59|node007\n'), 'terminal recovery'),
    (lambda r: r['commands'][1].__setitem__(2, '31253119'), 'scheduler evidence'),
    (lambda r: r['receipts_sha256'].pop(next(iter(r['receipts_sha256']))), 'exact forty'),
])
def test_resealed_completion_requires_exact_failed_reconciliation_and_receipts(fixture, change, error):
    mutate(fixture.gate.path, change); reseal_gate_runtime(fixture)
    with pytest.raises(ValueError, match=error): prepare(fixture)
    assert not (fixture.stage / 'frozen_transport').exists() and fixture.process_calls == []


def test_forty_receipt_count_cannot_replace_an_original_receipt(fixture):
    unrelated = fixture.root / 'unrelated_receipt.json'; write(unrelated, {'status': 'complete'})
    def replace(record):
        record['receipts_sha256'].pop(next(iter(record['receipts_sha256'])))
        record['receipts_sha256'][str(unrelated)] = transport.digest(unrelated)
    mutate(fixture.gate.path, replace); reseal_gate_runtime(fixture)
    with pytest.raises(ValueError, match='exact forty'): prepare(fixture)
    assert not (fixture.stage / 'frozen_transport').exists()


@pytest.mark.parametrize('missing', ['plan', 'tasks'])
def test_recovery_receipt_membership_depends_on_pinned_original_inputs(fixture, missing):
    path = fixture.gate.original_plan if missing == 'plan' else fixture.gate.tasks[0]
    mutate(fixture.activation, lambda r: r['inputs_sha256'].pop(str(path)))
    activation_sha = transport.digest(fixture.activation)
    write(fixture.activation.parent / 'activation.sha256.json', {'sha256': activation_sha})
    mutate(fixture.authority_runtime, lambda r: r.update(authority_sha256=activation_sha))
    fixture.authority.update(authority_sha256=activation_sha,
        authority_runtime_sha256=transport.digest(fixture.authority_runtime))
    fixture.proxy.authority.update(fixture.authority)
    with pytest.raises(ValueError, match='original plan pin|task pin changed'): prepare(fixture)
    assert not (fixture.stage / 'frozen_transport').exists()


@pytest.mark.parametrize('mutation',['runtime_link','digest','active','failed','missing_receipt','wrong_output','wrong_query'])
def test_recorded_recovery_gate_cannot_be_detached_or_narrowed(scratch,mutation):
    c=scratch
    if mutation=='runtime_link': mutate(c.runtime,lambda value:value.pop('recovery_completion_path'))
    elif mutation=='digest': mutate(c.runtime,lambda value:value.update(recovery_completion_sha256='0'*64))
    else:
        def change(value):
            if mutation=='active':value['stdout'][0]='31253118|RUNNING'
            elif mutation=='failed':value['stdout'][1]=value['stdout'][1].replace('FAILED','CANCELLED')
            elif mutation=='missing_receipt':value['receipts_sha256'].pop(str(c.receipt))
            elif mutation=='wrong_output':
                digest=value['receipts_sha256'].pop(str(c.receipt));value['receipts_sha256'][str(c.driver)]=digest
            else:value['commands'][1][2]='31253119'
        mutate(c.completion,change)
        mutate(c.runtime,lambda value:value.update(recovery_completion_sha256=transport.digest(c.completion)))
    c.authority['authority_runtime_sha256']=transport.digest(c.runtime)
    with pytest.raises(ValueError):transport.authority_pins(c.authority,live=True)
    assert not c.directory.exists() and not c.calls


@pytest.mark.parametrize('field,value',[('scheduler_success',True),('scientific_outputs_complete',False),
    ('exit_cause','cleanup_failure'),('status','verified_recovery_only'),('schema','wrong')])
def test_reconciliation_cannot_claim_success_or_established_cause(fixture,field,value):
    mutate(transport.RECONCILIATION,lambda record:record.update({field:value}))
    with pytest.raises(ValueError,match='exact failed recovery'):transport.reconciliation()


@pytest.mark.parametrize('field,value',[('job_id',31253119),('state','COMPLETED'),('exit_code','0:0')])
def test_reconciliation_is_limited_to_exact_failed_recovery(fixture,field,value):
    mutate(transport.RECONCILIATION,lambda record:record['recovery_execution'].update({field:value}))
    with pytest.raises(ValueError,match='exact failed recovery'):transport.reconciliation()


@pytest.mark.parametrize('field,value',[('schema','wrong'),('reconciliation_path','/different/proof.json'),
    ('reconciliation_sha256','0'*64)])
def test_completion_cannot_detach_reviewed_reconciliation(fixture,field,value):
    mutate(fixture.gate.path,lambda record:record.update({field:value}));reseal_gate_runtime(fixture)
    with pytest.raises(ValueError,match='reviewed reconciliation'):prepare(fixture)


def test_v2_keeps_scientific_launcher_dispatch_and_worker_functions_unchanged():
    import ast
    previous=SOURCE.parent/'modebench_scale_frozen_stage_transport_20260912.py'
    def functions(path):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body
        if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    old,new=functions(previous),functions(SOURCE)
    for name in ['script','effective_command','prepare_transport','verify_transport','LauncherProxy','install','worker']:
        assert old[name]==new[name], name
