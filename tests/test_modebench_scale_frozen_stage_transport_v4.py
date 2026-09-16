"""Synthetic confirmation transport only; no scheduler/model/scientific calls."""
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_modebench_scale_frozen_stage_transport_v2 import ScientificStub, WouldExec, write, mutate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/modebench_scale_frozen_stage_transport_v4_20260912.py'


def load_transport():
    spec = importlib.util.spec_from_file_location('scratch_confirmation_v4', SOURCE)
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    return a


def command(a, path, dependencies=(7, 31253118, 39999001)):
    return ['sbatch', '--parsable', '--partition=mltheory', '--account=mltheory',
        '--nodelist=node105', '--nodes=1', '--ntasks=1', '--gres=gpu:a5000:2',
        '--cpus-per-task=6', '--mem=60G', '--time=08:00:00', '--array=0-9%1',
        '--job-name=modebench-scale-domains-eval', '--no-requeue',
        '--output='+str(path.parent/'logs/%A_%a.out'), '--error='+str(path.parent/'logs/%A_%a.err'),
        '--chdir='+str(a.ROOT), '--export=ALL', '--dependency=afterany:'+':'.join(map(str, dependencies)),
        str(path.parent/'worker.slurm'), str(path)]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    a = load_transport(); b = a._base
    workspace = tmp_path/'workspace'; workspace.mkdir()
    artifacts = workspace/'composite'; artifacts.mkdir()
    authority_root = workspace/'authority'; authority_root.mkdir()
    driver_path = workspace/'driver.py'; driver_path.write_text('# synthetic authority verifier\n')
    launcher_path = workspace/'launcher.py'; launcher_path.write_text('# original launcher fixture\n')
    for key, value in [('ROOT', workspace), ('ARTIFACTS', artifacts), ('AUTHORITY_ROOT', authority_root),
                       ('LAUNCHER', launcher_path), ('LAUNCHER_SHA', a.digest(launcher_path))]:
        monkeypatch.setattr(a, key, value); monkeypatch.setattr(b, key, value)
    monkeypatch.setattr(a, 'DRIVER', driver_path)
    for key in ('RECOVERY_HELPER', 'RUNNER'):
        path = workspace/(key.lower()+'.py'); path.write_text('# '+key+' fixture\n')
        monkeypatch.setattr(b, key, path)
    manifest = workspace/'view/manifest.json'; manifest.parent.mkdir()
    paths = []
    for name in ('proot', 'preserved.py', 'copy.py'):
        path = manifest.parent/name; path.write_text('fixture original'); paths.append(path)
    value = {'proot': {'path': str(paths[0])}, 'preserved_source': {'path': str(paths[1])},
             'mappings': [{'source': str(paths[2])}]}
    write(manifest, value); (manifest.parent/'manifest.sha256').write_text(a.digest(manifest))
    monkeypatch.setattr(a, 'view', lambda path: a.read(path)); monkeypatch.setattr(b, 'view', a.view)
    lock = artifacts/'controller.lock'; fd = os.open(lock, os.O_CREAT|os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
    stat = os.fstat(fd); pid = os.getpid()
    ticks = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    argv = [part.decode() for part in Path('/proc/self/cmdline').read_bytes().split(b'\0') if part]
    identity = {'host': a.HOST, 'uid': os.getuid(), 'lock_path': str(lock), 'lock_inode': stat.st_ino,
                'lock_device': stat.st_dev, 'fence_fd': fd, 'owner_pid': pid, 'owner_start_ticks': ticks}
    completion = authority_root/'revised_recovery_completion.json'
    write(completion, {'schema': 'scratch_reviewed_completion', 'revised_recovery_job_id': 39999001,
                       'all_outputs_complete': True})
    recovery = {'revised_recovery_job_id': 39999001, 'revised_recovery_completion_path': str(completion),
                'revised_recovery_completion_sha256': a.digest(completion)}
    activation_path = authority_root/'activation.json'
    activation = {'schema': 'modebench_scale_spin_authority_v1', 'root': str(authority_root),
        **{key: identity[key] for key in ('host', 'uid', 'lock_path', 'lock_inode')},
        'driver': str(driver_path), 'driver_sha256': a.digest(driver_path),
        'transport': str(a.SOURCE), 'transport_sha256': a.digest(a.SOURCE),
        'recovery_job_id': a.RECOVERY_JOB, 'view_manifest': str(manifest), **recovery,
        'inputs_sha256': {str(path): a.digest(path) for path in
            (driver_path, a.SOURCE, a.BASE_SOURCE, a.PREVIOUS_SOURCE, completion, manifest)}}
    write(activation_path, activation)
    write(authority_root/'activation.sha256.json', {'sha256': a.digest(activation_path)})
    runtime_path = authority_root/'outer_runtime.json'
    write(runtime_path, {'schema': 'modebench_scale_spin_authority_runtime_v1', **identity, **recovery,
        'owner_command': argv, 'authority_path': str(activation_path), 'authority_sha256': a.digest(activation_path)})
    authority = {**identity, **recovery, 'authority_path': str(activation_path),
        'authority_sha256': a.digest(activation_path), 'authority_runtime_path': str(runtime_path),
        'authority_runtime_sha256': a.digest(runtime_path), 'view_manifest': str(manifest),
        'view_manifest_sha256': a.digest(manifest)}
    verifier_calls = []
    def verify(root, *, guest):
        verifier_calls.append((root, guest))
        assert root == authority_root and guest is True
        assert a.read(completion)['all_outputs_complete'] is True
        return a.read(activation_path)
    fence_calls = []
    driver = SimpleNamespace(SOURCE=driver_path, HOST=a.HOST, SCHEMA='modebench_scale_spin_authority_v1',
        verify=verify, assert_fence=lambda actual_fd: fence_calls.append(actual_fd))
    original = ScientificStub(launcher_path)
    def module(path, expected, name):
        a.require(a.digest(path) == expected, 'sealed module changed')
        if path == driver_path: return driver
        assert path == launcher_path
        return original
    monkeypatch.setattr(a, 'module', module); monkeypatch.setattr(b, 'module', module)
    monkeypatch.setattr(a.socket, 'gethostname', lambda: a.HOST)
    stage = artifacts/'confirmation'; stage.mkdir(); (stage/'runtime').mkdir()
    path = stage/'plan.json'
    cells = [{'id': level+'_'+domain, 'level': level, 'domain': domain, 'phase': 'eval',
              'command': ['/literal/python', '/original/evaluator', '--'+domain]}
             for level in ('level4', 'level5') for domain in a.DOMAINS]
    hardware = {'partition': 'mltheory', 'account': 'mltheory', 'nodes': 'node105', 'cpus': 6,
        'memory': '60G', 'gres': 'gpu:a5000:2', 'tensor_parallel_size': 2, 'dtype': 'float16',
        'engine': 'V0', 'attention_backend': 'XFORMERS', 'vllm_version': '0.8.4'}
    plan = {'phase': 'eval', 'concurrency': 1, 'cells': cells, 'hardware': hardware,
        'dependency_ids': [7, 31253118, 39999001], 'submit_command': command(a, path),
        'runtime_profile': {'thread_environment': {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'}}}
    write(path, plan); write(stage/'plan.sha256.json', {'sha256': a.digest(path)})
    (stage/'worker.slurm').write_text('# original scientific worker\n')
    process_calls = []; exec_calls = []
    def no_process(*args, **kw): raise AssertionError('unregistered scheduler route')
    monkeypatch.setattr(b.subprocess, 'run', no_process)
    def dispatch(result=None, exception=None):
        def run(command, **kwargs):
            process_calls.append((list(command), dict(kwargs)))
            if exception is not None: raise exception
            return result or SimpleNamespace(returncode=0, stdout='987654\n', stderr='')
        monkeypatch.setattr(b.subprocess, 'run', run)
    def worker_environment(node='node202', index=0):
        monkeypatch.setattr(a.socket, 'gethostname', lambda: node+'.ionic.cs.princeton.edu')
        monkeypatch.setattr(a.importlib.metadata, 'version', lambda name: '0.8.4')
        cuda = SimpleNamespace(device_count=lambda: 2, get_device_name=lambda i: 'NVIDIA RTX A5000')
        monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
        for key, value in {'SLURM_JOB_ID': '987655', 'SLURM_JOB_PARTITION': 'cs', 'SLURM_JOB_ACCOUNT': 'allcs',
            'SLURM_ARRAY_JOB_ID': '987654', 'SLURM_ARRAY_TASK_ID': str(index),
            'SLURMD_NODENAME': node, 'SLURM_CPUS_PER_TASK': '6', 'SLURM_MEM_PER_NODE': '61440',
            'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'VLLM_USE_V1': '0',
            'VLLM_ATTENTION_BACKEND': 'XFORMERS', 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPYCACHEPREFIX': '/tmp/NONPRODUCTION-scale-frozen-pycache-fixture/pycache'}.items():
            monkeypatch.setenv(key, value)
        def capture(executable, argv):
            exec_calls.append((executable, list(argv))); raise WouldExec('original worker would execute')
        monkeypatch.setattr(a.os, 'execv', capture)
        return cuda
    proxy = a.LauncherProxy(original, authority, manifest)
    yield SimpleNamespace(a=a, b=b, path=path, plan=plan, stage=stage, original=original, proxy=proxy,
        authority=authority, activation=activation_path, runtime=runtime_path, completion=completion,
        driver=driver, verifier_calls=verifier_calls, fence_calls=fence_calls, manifest=manifest,
        lock=lock, fd=fd, dispatch=dispatch, process_calls=process_calls,
        worker_environment=worker_environment, exec_calls=exec_calls)
    try: os.close(fd)
    except OSError: pass


def prepare(f):
    return f.a.prepare_transport(f.original, f.path, f.authority, f.manifest)


def submit(f):
    f.dispatch(); return f.proxy.submit(f.path)


def repin_authority(f):
    write(f.activation.parent/'activation.sha256.json', {'sha256': f.a.digest(f.activation)})
    f.authority['authority_sha256'] = f.a.digest(f.activation)
    mutate(f.runtime, lambda value: value.update(authority_sha256=f.authority['authority_sha256']))
    f.authority['authority_runtime_sha256'] = f.a.digest(f.runtime)


def test_effective_command_changes_only_four_declared_items(fixture):
    f = fixture; before = list(f.plan['submit_command'])
    effective = f.a.effective_command(f.path, before)
    restored = [value.replace('--partition=cs', '--partition=mltheory').replace('--account=allcs', '--account=mltheory') for value in effective]
    restored.insert(4, '--nodelist=node105'); restored[-2] = str(f.stage/'worker.slurm')
    assert restored == before == f.plan['submit_command']
    assert '--account=allcs' in effective and '--partition=cs' in effective and not any(v.startswith('--nodelist') for v in effective)


@pytest.mark.parametrize('old,new', [('--partition=mltheory', '--partition=all'), ('--account=mltheory', '--account=other'),
    ('--nodelist=node105', '--nodelist=node202'), ('--gres=gpu:a5000:2', '--gres=gpu:a100:2'),
    ('--cpus-per-task=6', '--cpus-per-task=8'), ('--mem=60G', '--mem=59G'),
    ('--array=0-9%1', '--array=0-9%2'), ('--no-requeue', '--requeue'), ('--ntasks=1', '--ntasks=2')])
def test_canonical_command_cannot_smuggle_operational_changes(fixture, old, new):
    f = fixture; values = [new if value==old else value for value in f.plan['submit_command']]
    with pytest.raises(ValueError, match='exact canonical'):
        f.a.effective_command(f.path, values)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'worker', 'dependency'])
def test_command_shape_routing_dependencies_are_exact(fixture, mutation):
    f = fixture; values = list(f.plan['submit_command'])
    if mutation == 'duplicate': values.insert(3, '--partition=mltheory')
    elif mutation == 'missing': values.remove('--account=mltheory')
    elif mutation == 'worker': values[-2] = '/other/worker.slurm'
    else: values[-3] = '--dependency=afterany:7:7:39999001'
    with pytest.raises(ValueError): f.a.effective_command(f.path, values)


def test_original_development_is_out_of_scope(fixture):
    f = fixture
    with pytest.raises(ValueError, match='only covers future confirmation'):
        f.a.stage_path(f.path.parent.with_name('revision_development')/'plan.json')


def test_prepare_adds_both_recoveries_without_mutating_science(fixture):
    f = fixture; cells = copy.deepcopy(f.plan['cells']); models = {'7b':'frozen', '14b':'frozen'}; pins = {'/input':'sha'}
    before = copy.deepcopy((cells, models, pins))
    value = f.proxy.prepare(f.stage, cells, models, dependency_ids=[7], pins=pins)
    assert value['dependency_ids'] == [7, 31253118, 39999001]
    call = f.original.calls[-1]
    assert call[2] is cells and call[3] is models and call[5] is pins
    assert before == (cells, models, pins)


def test_transport_preserves_canonical_bytes_and_pins_both_sources(fixture):
    f = fixture; paths = [f.path, f.stage/'plan.sha256.json', f.stage/'worker.slurm']
    before = {path:path.read_bytes() for path in paths}
    value = prepare(f)
    assert value['schema'] == f.a.SCHEMA
    assert value['scientific_plan_or_evaluator_argv_changed'] is False
    assert before == {path:path.read_bytes() for path in paths}
    for path in (f.a.SOURCE, f.a.BASE_SOURCE, f.a.PREVIOUS_SOURCE, f.a.DRIVER, f.completion):
        assert value['inputs_sha256'][str(path)] == f.a.digest(path)
    assert f.a.verify_transport(f.original, f.path)[1] == value
    assert f.process_calls == [] and f.verifier_calls and f.fence_calls


@pytest.mark.parametrize('field,value', [('host', 'wash.cs.princeton.edu'), ('owner_start_ticks','0'),
    ('lock_inode',1), ('owner_pid',999999999), ('uid',-1)])
def test_changed_live_owner_cannot_prepare(fixture, field, value):
    f = fixture; authority = {**f.authority, field:value}
    with pytest.raises((ValueError, FileNotFoundError)):
        f.a.authority_pins(authority, live=True)
    assert f.process_calls == []


def test_static_authority_verification_works_on_worker_but_live_requires_spin(fixture, monkeypatch):
    f = fixture; monkeypatch.setattr(f.a.socket, 'gethostname', lambda:'node202.ionic.cs.princeton.edu')
    assert f.a.authority_pins(f.authority, live=False)
    with pytest.raises(ValueError, match='actual spin'):
        f.a.authority_pins(f.authority, live=True)


@pytest.mark.parametrize('omission', ['driver', 'transport', 'base', 'previous', 'completion', 'view'])
def test_activation_cannot_omit_required_closure(fixture, omission):
    f = fixture
    paths = {'driver':f.a.DRIVER, 'transport':f.a.SOURCE, 'base':f.a.BASE_SOURCE, 'previous':f.a.PREVIOUS_SOURCE,
             'completion':f.completion, 'view':f.manifest}
    mutate(f.activation, lambda value:value['inputs_sha256'].pop(str(paths[omission])))
    repin_authority(f)
    with pytest.raises(ValueError, match='omits driver'):
        f.a.authority_pins(f.authority)


def test_driver_recovery_verification_is_mandatory(fixture):
    f = fixture
    def failed(*args, **kwargs): raise ValueError('recovery proof invalid')
    f.driver.verify = failed
    with pytest.raises(ValueError, match='recovery proof invalid'): prepare(f)
    assert not (f.stage/'frozen_transport').exists()


def test_completion_drift_refuses_preparation(fixture):
    f = fixture; mutate(f.completion, lambda value:value.update(all_outputs_complete=False))
    with pytest.raises(ValueError, match='completion'): prepare(f)
    assert not (f.stage/'frozen_transport').exists()


@pytest.mark.parametrize('kind', ['missing_domain', 'phase', 'tp', 'node', 'dependency'])
def test_confirmation_scientific_scope_must_stay_exact(fixture, kind):
    f = fixture; plan = copy.deepcopy(f.plan)
    if kind == 'missing_domain': plan['cells'].pop()
    elif kind == 'phase': plan['cells'][0]['phase'] = 'dev'
    elif kind == 'tp': plan['hardware']['tensor_parallel_size'] = 1
    elif kind == 'node': plan['hardware']['nodes'] = 'node202'
    else: plan['dependency_ids'].remove(39999001)
    with pytest.raises(ValueError): f.a.confirmation_scope(plan, f.authority)


def test_lost_fence_is_never_reacquired(fixture):
    f = fixture; fcntl.flock(f.fd, fcntl.LOCK_UN)
    with pytest.raises(ValueError, match='no longer owns'): prepare(f)
    assert not (f.stage/'frozen_transport').exists()
    probe = os.open(f.lock, os.O_RDWR)
    try: fcntl.flock(probe, fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally: os.close(probe)


def test_effective_ledgers_are_exact_and_dispatch_once(fixture):
    f = fixture; submit(f)
    root = f.stage/'frozen_transport'
    tx = f.a.read(root/'plan.json'); intent = f.a.read(root/'submission_intent.json')
    assert intent['canonical_command'] == f.plan['submit_command']
    assert intent['effective_command'] == tx['effective_submit_command']
    assert f.process_calls[0][0] == tx['effective_submit_command']
    assert f.process_calls[0][1]['close_fds'] is True
    with pytest.raises((ValueError, FileExistsError)): f.proxy.submit(f.path)
    with pytest.raises(FileExistsError): f.proxy.execute(f.path, f.plan['submit_command'])
    assert len(f.process_calls) == 1


def test_concurrent_submissions_cannot_dispatch_twice(fixture):
    f = fixture; f.dispatch()
    def attempt():
        try: return f.proxy.submit(f.path)
        except (ValueError, FileExistsError): return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _:attempt(), range(2)))
    assert sum(value is not None for value in results) == 1 and len(f.process_calls) == 1


@pytest.mark.parametrize('kind', ['timeout', 'nonzero'])
def test_ambiguous_attempt_is_retained_without_retry(fixture, kind):
    f = fixture
    if kind == 'timeout': f.dispatch(exception=subprocess.TimeoutExpired(['sbatch'], 60))
    else: f.dispatch(result=SimpleNamespace(returncode=1, stdout='', stderr='failed'))
    with pytest.raises((subprocess.TimeoutExpired, RuntimeError)): f.proxy.submit(f.path)
    with pytest.raises((ValueError, FileExistsError)): f.proxy.submit(f.path)
    assert len(f.process_calls) == 1 and (f.stage/'submission_intent.json').exists()


@pytest.mark.parametrize('node', ['node202', 'node203', 'node204'])
def test_allowed_actual_worker_nodes_are_truthfully_recorded(fixture, node):
    f = fixture; submit(f); f.worker_environment(node)
    with pytest.raises(WouldExec): f.a.worker(f.path, 0)
    record = f.a.read(f.stage/'frozen_transport/runtime/0.json')
    assert record['hostname'] == node+'.ionic.cs.princeton.edu'
    assert record['environment']['SLURMD_NODENAME'] == node
    assert record['environment']['SLURM_JOB_PARTITION'] == 'cs'
    assert record['environment']['SLURM_JOB_ACCOUNT'] == 'allcs'
    assert record['environment']['SLURM_JOB_ID'] == '987655'
    assert record['canonical_node_constraint'] == f.plan['hardware']['nodes'] == 'node105'
    assert record['transport_schema'] == f.a.SCHEMA and record['effective_partition'] == 'cs' and record['effective_account'] == 'allcs'
    assert record['evaluator_command'] == f.plan['cells'][0]['command']
    assert f.exec_calls == [(str(f.a.PYTHON), [str(f.a.PYTHON), '-B', str(f.a.LAUNCHER),
        'worker', '--plan', str(f.path), '--index', '0'])]
    with pytest.raises(ValueError, match='already attempted'): f.a.worker(f.path, 0)
    assert len(f.exec_calls) == 1


@pytest.mark.parametrize('field,value', [('SLURMD_NODENAME','node205'), ('SLURM_CPUS_PER_TASK','8'),
    ('SLURM_MEM_PER_NODE','60416'), ('SLURM_ARRAY_JOB_ID','1'), ('SLURM_ARRAY_TASK_ID','1'),
    ('OMP_NUM_THREADS','1'), ('VLLM_USE_V1','1'), ('VLLM_ATTENTION_BACKEND','FLASH_ATTN')])
def test_worker_identity_backend_and_resources_cannot_change(fixture, monkeypatch, field, value):
    f = fixture; submit(f); f.worker_environment()
    monkeypatch.setenv(field, value)
    with pytest.raises(ValueError, match='actual worker identity'): f.a.worker(f.path, 0)
    assert not (f.stage/'frozen_transport/runtime/0.json').exists() and f.exec_calls == []


@pytest.mark.parametrize('kind', ['node', 'count', 'gpu', 'version'])
def test_worker_refuses_unqualified_actual_hardware(fixture, monkeypatch, kind):
    f = fixture; submit(f); cuda = f.worker_environment('node205' if kind=='node' else 'node202')
    if kind == 'count': cuda.device_count = lambda:1
    elif kind == 'gpu': cuda.get_device_name = lambda i:'NVIDIA A100'
    elif kind == 'version': monkeypatch.setattr(f.a.importlib.metadata, 'version', lambda name:'0.9.0')
    with pytest.raises(ValueError): f.a.worker(f.path, 0)
    assert f.exec_calls == [] and not (f.stage/'frozen_transport/runtime/0.json').exists()


@pytest.mark.parametrize('kind', ['effective_command', 'canonical_command', 'result_job', 'result_intent'])
def test_worker_rejects_cross_ledger_tampering(fixture, kind):
    f = fixture; submit(f); root = f.stage/'frozen_transport'
    if kind == 'effective_command':
        mutate(root/'submission_intent.json', lambda value:value['effective_command'].append('--extra'))
    elif kind == 'canonical_command':
        mutate(f.stage/'submission_intent.json', lambda value:value['command'].append('--extra'))
    elif kind == 'result_job':
        mutate(root/'submission_result.json', lambda value:value.update(stdout='987656\n'))
    else:
        mutate(root/'submission_result.json', lambda value:value.update(intent_sha256='0'*64))
    f.worker_environment()
    with pytest.raises(ValueError, match='submission linkage'): f.a.worker(f.path, 0)
    assert f.exec_calls == []


def test_install_and_inherited_dispatch_have_new_scope(fixture):
    f = fixture; controller = SimpleNamespace(launch=f.original)
    f.a.install(controller, f.authority, f.manifest)
    assert isinstance(controller.launch, f.a.LauncherProxy)
    assert f.a.install.__globals__['authority_pins'] is f.a.authority_pins
    assert f.a.LauncherProxy.execute.__globals__['verify_transport'] is f.a.verify_transport
    assert f.a.LauncherProxy.execute.__globals__['SOURCE'] == f.a.SOURCE
    with pytest.raises(ValueError, match='already installed'): f.a.install(controller, f.authority, f.manifest)


@pytest.mark.parametrize('field,value', [('SLURM_JOB_PARTITION', 'mltheory'), ('SLURM_JOB_PARTITION', None),
    ('SLURM_JOB_ACCOUNT', 'other'), ('SLURM_JOB_ACCOUNT', None), ('SLURM_JOB_ID', None),
    ('SLURM_JOB_ID', ''), ('SLURM_JOB_ID', '0'), ('SLURM_JOB_ID', '987654_0'), ('SLURM_JOB_ID', '-1')])
def test_actual_allocation_identity_is_required_before_runtime(fixture, monkeypatch, field, value):
    f = fixture; submit(f); f.worker_environment()
    if value is None: monkeypatch.delenv(field, raising=False)
    else: monkeypatch.setenv(field, value)
    with pytest.raises(ValueError, match='actual worker identity'): f.a.worker(f.path, 0)
    assert f.exec_calls == [] and not (f.stage/'frozen_transport/runtime/0.json').exists()


def test_reversed_account_partition_and_previous_all_route_are_rejected(fixture, monkeypatch):
    f = fixture; submit(f); f.worker_environment()
    for account, partition in [('cs', 'allcs'), ('mltheory', 'all'), ('mltheory', 'mltheory')]:
        monkeypatch.setenv('SLURM_JOB_ACCOUNT', account)
        monkeypatch.setenv('SLURM_JOB_PARTITION', partition)
        with pytest.raises(ValueError, match='actual worker identity'): f.a.worker(f.path, 0)
    assert f.exec_calls == [] and not (f.stage/'frozen_transport/runtime/0.json').exists()


def test_old_node105_cannot_execute_new_cs_transport(fixture):
    f = fixture; submit(f); f.worker_environment('node105')
    with pytest.raises(ValueError, match='actual worker identity'): f.a.worker(f.path, 0)
    assert f.exec_calls == []
