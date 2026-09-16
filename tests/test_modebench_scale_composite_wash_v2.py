"""Operational scratch tests; no production authority, scheduler, grader, or LLM."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def authority(tmp_path, monkeypatch):
    a = module(ROOT / 'artifacts/continue_modebench_scale_composite_wash_v2_20260912.py', 'scratch_wash')
    base = tmp_path / 'scratch_only'
    base.mkdir()
    def file(name, contents='scratch_only'):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path
    paths = {
        'ROOT': base, 'SOURCE': file('adapter.py'), 'TRANSPORT': file('transport.py'),
        'ALTERNATIVE_DRIVER': file('unused_adapter.py'), 'ALTERNATIVE_AUTHORITY': base / 'unused_authority',
        'AUTHORIZATION': base / 'wash_authorization.json', 'LOCK_QUALIFICATION': base / 'lock_qualification.json',
        'RUNNER': file('runner.py'), 'PYTHON': Path('/usr/bin/python3'),
        'PREPARED_V1': base / 'prepared_v1', 'PREPARED_V1_SOURCE': file('prepared_v1_driver.py'),
        'RECONCILIATION_VERIFIER': file('reconciliation_verifier.py'),
        'RECONCILIATION': base / 'recovery/execution_reconciliation.json',
        'AUTHORITY': base / 'authority', 'ARTIFACTS': base / 'old_authority',
        'LOCK': file('old_authority/controller.lock'), 'CLAIM': base / 'old_authority/new_claim.json',
        'CONTROLLER': file('controller.py'), 'PREDECESSOR': file('previous.py'),
        'ORIGINAL_PLAN': file('original/plan.json'), 'RECOVERY': base / 'recovery',
        'CANONICAL': file('templates.py', 'neutral')}
    for key, value in paths.items():
        monkeypatch.setattr(a, key, value)
    v1_activation = file('prepared_v1/activation.json', '{}')
    v1_sidecar = file('prepared_v1/activation.sha256.json', json.dumps({'sha256': a.sha(v1_activation)}))
    (a.PREPARED_V1 / 'sweeps').mkdir()
    monkeypatch.setattr(a, 'PREPARED_V1_PINS', {path: a.sha(path) for path in (a.PREPARED_V1_SOURCE, v1_activation, v1_sidecar)})
    monkeypatch.setattr(a, 'RECONCILIATION_VERIFIER_SHA', a.sha(a.RECONCILIATION_VERIFIER))
    monkeypatch.setattr(a, 'LOCK_INODE', a.LOCK.stat().st_ino)
    monkeypatch.setattr(a, 'NEUTRAL_SHA', a.sha(a.CANONICAL))
    old = file('preserved.py', 'old')
    monkeypatch.setattr(a, 'OLD_SHA', a.sha(old))
    a.AUTHORIZATION.write_text(json.dumps({
        'schema': 'modebench_scale_wash_continuation_user_direction_v1',
        'authorized_controller_host': a.HOST, 'historical_controller_host': 'soak.cs.princeton.edu',
        'historical_process_state': 'not directly observed', 'neutral_level3_remains_main': True}))
    phases = [{'returncode': 0, 'stdout': json.dumps({'observed': state,
               'host': 'node105.ionic.cs.princeton.edu', 'job': '31253118', 'inode': 123})}
              for state in ('blocked', 'acquired')]
    a.LOCK_QUALIFICATION.write_text(json.dumps({
        'schema': 'modebench_scale_scratch_cross_host_lock_qualification_v1',
        'status': 'passed', 'host': a.HOST, 'allocation_job_id': 31253118,
        'neutral_source_before': a.NEUTRAL_SHA, 'neutral_source_after': a.NEUTRAL_SHA, 'phases': phases}))
    monkeypatch.setattr(a, 'TRANSFER_PINS', {path: a.sha(path) for path in (a.AUTHORIZATION, a.LOCK_QUALIFICATION)})
    monkeypatch.setattr(a, 'RUNNER_SHA', a.sha(a.RUNNER))
    monkeypatch.setattr(a, 'TRANSPORT_SHA', a.sha(a.TRANSPORT))
    monkeypatch.setattr(a.os, 'uname', lambda: SimpleNamespace(nodename=a.HOST))
    monkeypatch.setattr(a, 'local_controllers', lambda: [])
    activation = file('old_authority/controller_activation.json', json.dumps({
        'host': 'soak.cs.princeton.edu', 'files_sha256': {str(a.CONTROLLER): a.sha(a.CONTROLLER)}}))
    arm = file('old_authority/controller_arm_result_v2.json', json.dumps({
        'pid': 3256866, 'start_ticks': '88049820', 'activation_sha256': a.sha(activation),
        'command': [str(a.PYTHON), '-B', str(a.CONTROLLER), '--advance', '--watch', '--interval', '60']}))
    error = file('old_authority/controller_v2.stderr',
                 'in <module>\nin main\nin sweep\noriginal.authenticate\nregistered source/input changed: ' + str(a.CANONICAL))
    file('old_authority/controller_identity.json', '{}')
    monkeypatch.setattr(a, 'HISTORICAL_PINS', {path: a.sha(path) for path in (a.CONTROLLER, a.PREDECESSOR, activation, arm, error)})
    manifest = file('view/manifest.json', '{}')
    file('view/manifest.sha256', a.sha(manifest))
    copied = file('view/copy.py', 'old')
    proot = file('proot')
    view = {'proot': {'path': str(proot)}, 'preserved_source': {'path': str(old)},
            'mappings': [{'source': str(copied)}]}
    manifest.write_text(json.dumps(view))
    (manifest.parent / 'manifest.sha256').write_text(a.sha(manifest))
    file('recovery/plan.json', json.dumps({'view_manifest': str(manifest),
         'inputs_sha256': {str(a.CANONICAL): a.OLD_SHA, str(old): a.OLD_SHA}}))
    file('recovery/plan.sha256.json', json.dumps({'sha256': a.sha(a.RECOVERY / 'plan.json')}))
    file('recovery/submission_intent.json', '{}')
    file('recovery/submission_result.json', json.dumps({'job_id': 31253118, 'status': 'submitted',
         'plan_sha256': a.sha(a.RECOVERY / 'plan.json')}))
    file('recovery/runtime.json', '{"scratch_only": true}')
    certificate = {'schema': a.RECONCILIATION_SCHEMA, 'status': a.RECONCILIATION_STATUS,
        'scientific_outputs_complete': True, 'scheduler_success': False, 'exit_cause': 'unknown',
        'recovery_execution': {'state': 'FAILED', 'exit_code': '1:0', 'end_utc': a.RECOVERY_END},
        'files_sha256': {str(a.CANONICAL): a.OLD_SHA}}
    file('recovery/execution_reconciliation.json', json.dumps(certificate))
    def reconciliation_verify(manifest_path):
        a.reconciliation_inputs()
        return {'command': a.reconciliation_command(manifest_path), 'returncode': 0,
                'observation': {'schema': a.RECONCILIATION_SCHEMA, 'status': a.RECONCILIATION_STATUS,
                  'scientific_outputs_complete': True, 'scheduler_success': False, 'exit_cause': 'unknown',
                  'certificate_path': str(a.RECONCILIATION), 'certificate_sha256': a.sha(a.RECONCILIATION)}}
    monkeypatch.setattr(a, 'verify_reconciliation_subprocess', reconciliation_verify)
    cells = []
    for cell in range(10):
        tasks = [{'output': str(file(f'receipts/{cell}_{tier}.json', '{"scratch_only": true}'))} for tier in range(4)]
        path = file(f'tasks/{cell}.json', json.dumps(tasks))
        cells.append({'tasks': str(path)})
    a.ORIGINAL_PLAN.write_text(json.dumps({'cells': cells}))
    def load(path, name):
        assert path == a.RUNNER
        return SimpleNamespace(authenticate=lambda path: view)
    monkeypatch.setattr(a, 'load_module', load)
    def probe(owner_pid=0):
        return {'command': [str(a.PYTHON), '-B', str(a.SOURCE), 'host-probe', '--owner-pid', str(owner_pid)],
                'returncode': 0, 'observation': a.host_probe(owner_pid)}
    monkeypatch.setattr(a, 'probe_subprocess', probe)
    def no_process(*args, **kwargs):
        pytest.fail('no real process or scheduler allowed in scratch unit tests')
    monkeypatch.setattr(a.subprocess, 'run', no_process)
    return SimpleNamespace(a=a, root=a.AUTHORITY, manifest=manifest, old=old)


def prepare(c):
    c.a.prepare(c.root, c.manifest, c.a.sha(c.a.TRANSPORT))
    return c.a.read(c.root / 'activation.json')


def scheduler(c, monkeypatch, *, live=False, state='FAILED', code='1:0', returncode=0):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == 'squeue':
            stdout = ('31253118|RUNNING\n' if live else '999999|RUNNING\n')
        else:
            assert command[0] == 'sacct'
            stdout = f'31253118|{state}|{code}|2026-09-12T02:04:21|node105\n'
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr='failure' if returncode else '')
    monkeypatch.setattr(c.a.subprocess, 'run', run)
    return calls


def test_prepare_additive_preserves_old_activation_identity_and_lock(authority):
    c = authority
    old_paths = [c.a.ARTIFACTS / name for name in ('controller_activation.json', 'controller_identity.json', 'controller.lock')]
    before = {str(path): path.read_bytes() for path in old_paths}
    value = prepare(c)
    assert value['host'] == c.a.HOST and value['historical_soak_os_state'] == 'unverified'
    assert value['lock_inode'] == c.a.LOCK_INODE
    assert value['transport_sha256'] == c.a.sha(c.a.TRANSPORT)
    assert value['transfer_evidence']['historical_soak_os_state'] == 'unverified'
    assert 'soak configuration unobserved' in value['transfer_evidence']['lock_qualification']['scope']
    assert 'observational' in value['transfer_evidence']['historical_log_association']
    assert all(value['inputs_sha256'][str(path)] == digest for path, digest in c.a.TRANSFER_PINS.items())
    assert all(Path(path).read_bytes() == data for path, data in before.items())
    assert not (c.root / 'outer_runtime.json').exists()
    assert c.a.verify(c.root) == value


def test_prepare_double_claim_refused(authority):
    c = authority
    prepare(c)
    with pytest.raises(ValueError, match='already claimed'):
        prepare(c)


def test_prepare_does_not_weaken_historical_pins(authority):
    c = authority
    c.a.CONTROLLER.write_text('changed')
    with pytest.raises(ValueError, match='pinned input changed'):
        prepare(c)
    assert not c.a.CLAIM.exists()


def test_wrong_host_and_wrong_source_view_refuse(authority, monkeypatch):
    c = authority
    monkeypatch.setattr(c.a.os, 'uname', lambda: SimpleNamespace(nodename='soak.cs.princeton.edu'))
    with pytest.raises(ValueError, match='actual wash'):
        c.a.host_probe()
    monkeypatch.setattr(c.a.os, 'uname', lambda: SimpleNamespace(nodename=c.a.HOST))
    c.a.CANONICAL.write_text('old')
    with pytest.raises(ValueError, match='outside the guest'):
        c.a.host_probe()


def test_local_duplicate_refused_owner_allowed(authority, monkeypatch):
    c = authority
    monkeypatch.setattr(c.a, 'local_controllers', lambda: [{'pid': 123}])
    with pytest.raises(ValueError, match='another local'):
        c.a.host_probe()
    assert c.a.host_probe(123)['local_controllers'] == [{'pid': 123}]


def test_python_c_string_not_controller_invocation(authority):
    a = authority.a
    assert a.actual_script(['/bin/python3', '-c', str(a.CONTROLLER), '--advance']) == (None, [])
    assert a.actual_script(['/bin/python3', '-B', str(a.CONTROLLER), '--advance']) == (str(a.CONTROLLER), ['--advance'])


@pytest.mark.parametrize('name', ['TRANSPORT', 'SOURCE', 'RUNNER'])
def test_changed_new_operational_inputs_rejected(authority, name):
    c = authority
    prepare(c)
    getattr(c.a, name).write_text('changed after registration')
    with pytest.raises(ValueError, match='chang'):
        c.a.verify(c.root)


def test_host_and_guest_pins_are_explicit_distinct_contracts(authority):
    c = authority
    prepare(c)
    with pytest.raises(ValueError, match='fence changed'):
        c.a.verify(c.root, guest=True)
    c.a.CANONICAL.write_text('old')
    c.a.verify(c.root, guest=True)
    with pytest.raises(ValueError, match='fence changed'):
        c.a.verify(c.root)


@pytest.mark.parametrize('live,state,code', [(True, 'RUNNING', '0:0'), (False, 'COMPLETED', '0:0'), (False, 'FAILED', '0:0')])
def test_no_advance_without_exact_audited_failed_recovery(authority, monkeypatch, live, state, code):
    c = authority
    prepare(c)
    scheduler(c, monkeypatch, live=live, state=state, code=code)
    with pytest.raises(ValueError):
        c.a.watch(c.root)
    assert not (c.root / 'start_intent.json').exists()
    assert not (c.root / 'outer_runtime.json').exists()


def test_scheduler_failure_is_not_terminal_success(authority, monkeypatch):
    c = authority
    scheduler(c, monkeypatch, returncode=1)
    with pytest.raises(ValueError, match='scheduler query failed'):
        c.a.recovery_gate()


def test_all_forty_receipts_required(authority, monkeypatch):
    c = authority
    scheduler(c, monkeypatch)
    completed = c.a.recovery_gate()
    assert len(completed['receipts_sha256']) == 40
    Path(next(iter(completed['receipts_sha256']))).unlink()
    with pytest.raises(ValueError, match='all forty'):
        c.a.recovery_gate()


def test_continuous_existing_fence_and_no_unlink(authority):
    c = authority
    inode = c.a.LOCK.stat().st_ino
    with c.a.lifetime_fence() as fd:
        c.a.assert_fence(fd)
        second = os.open(c.a.LOCK, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(second)
    assert c.a.LOCK.stat().st_ino == inode
    with c.a.lifetime_fence():
        pass


def test_lock_inode_replacement_fails_closed(authority):
    c = authority
    with c.a.lifetime_fence() as fd:
        c.a.LOCK.rename(c.a.LOCK.with_name('preserved_old.lock'))
        c.a.LOCK.write_text('new unrelated inode')
        with pytest.raises(ValueError, match='inode changed'):
            c.a.assert_fence(fd)


def test_guest_argv_uses_explicit_runner_and_inherited_fd(authority):
    c = authority
    activation = prepare(c)
    command = c.a.guest_command(c.root, activation, 2, 15)
    assert command[:7] == [str(c.a.PYTHON), '-B', str(c.a.RUNNER), 'exec', '--manifest', str(c.manifest), '--']
    assert command[-4:] == ['--sweep', '2', '--fence-fd', '15']
    assert '--advance' not in command  # only guarded internal _sweep entrypoint


def test_failed_guest_durable_record_blocks_retry(authority, monkeypatch):
    c = authority
    prepare(c)
    scheduler(c, monkeypatch)
    gate = c.a.recovery_gate()
    monkeypatch.setattr(c.a, 'recovery_gate', lambda: gate)
    calls = []
    def failed(command, **kwargs):
        calls.append(command)
        fd = kwargs['pass_fds'][0]
        c.a.assert_fence(fd)
        assert (c.root / 'sweeps/000001/intent.json').exists()
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(c.a.subprocess, 'run', failed)
    with pytest.raises(ValueError, match='never auto-retry'):
        c.a.watch(c.root)
    assert c.a.read(c.root / 'failure.json')['status'] == 'explicit_reconciliation_required'
    assert not (c.root / 'sweeps/000001/result.json').exists()
    with pytest.raises(ValueError, match='already attempted'):
        c.a.watch(c.root)
    assert len(calls) == 1


def test_successful_guest_terminal_result_and_runtime_linkage(authority, monkeypatch):
    c = authority
    activation = prepare(c)
    scheduler(c, monkeypatch)
    gate = c.a.recovery_gate()
    monkeypatch.setattr(c.a, 'recovery_gate', lambda: gate)
    def complete(command, **kwargs):
        fd = kwargs['pass_fds'][0]
        runtime = c.a.read(c.root / 'outer_runtime.json')
        record = c.a.authority_record(c.root, activation, runtime, fd)
        assert record['owner_pid'] == os.getpid()
        assert record['fence_fd'] == fd and record['uid'] == os.getuid()
        assert runtime['host'] == c.a.HOST and runtime['lock_inode'] == c.a.LOCK_INODE
        assert runtime['recovery_completion_path'] == str(c.root / 'recovery_completion.json')
        assert runtime['recovery_completion_sha256'] == c.a.sha(c.root / 'recovery_completion.json')
        assert c.a.runtime_completion(c.root, runtime) == gate
        c.a.atomic_new(c.root / 'sweeps/000001/guest_runtime.json', {'scratch_only': True, 'authority': record})
        c.a.atomic_new(c.root / 'sweeps/000001/result.json', {'authority': record,
            'guest_runtime_sha256': c.a.sha(c.root / 'sweeps/000001/guest_runtime.json'),
            'result': {'status': 'needs_new_development_revision'}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.a.subprocess, 'run', complete)
    result = c.a.watch(c.root)
    assert result['status'] == 'needs_new_development_revision'
    assert c.a.read(c.root / 'terminal.json')['last_sweep'] == 1
    assert not (c.root / 'failure.json').exists()


@pytest.fixture(params=['local', 'shared_nfs'])
def fence_workspace(request, tmp_path):
    if request.param == 'local':
        yield tmp_path
    else:
        with tempfile.TemporaryDirectory(prefix='NONPRODUCTION-proot-inherited-fence-', dir=ROOT / 'artifacts') as directory:
            yield Path(directory)


def test_real_proot_child_retains_same_flock_after_parent_closes_fd(fence_workspace, monkeypatch):
    """Local FD-inheritance qualification only; NOT a soak/NFS cross-host claim."""
    adapter = module(ROOT / 'artifacts/run_modebench_scale_frozen_view_20260912.py', 'real_proot_adapter')
    base = fence_workspace / 'real_proot_fixture'
    base.mkdir()
    artifacts = base / 'artifacts'; artifacts.mkdir()
    canonical = base / 'templates.py'; canonical.write_text('neutral')
    preserved = artifacts / 'preserved.py'; preserved.write_text('old')
    runner = artifacts / 'runner.py'; runner.write_bytes((ROOT / 'artifacts/run_modebench_scale_frozen_view_20260912.py').read_bytes())
    for key, value in {'ROOT': base, 'RUNNER': runner, 'CANONICAL': canonical, 'PRESERVED': preserved,
                       'OLD_SHA256': adapter.file_sha(preserved), 'NEUTRAL_SHA256': adapter.file_sha(canonical)}.items():
        monkeypatch.setattr(adapter, key, value)
    view = artifacts / 'view'
    adapter.prepare(view)
    lock = base / 'scratch.lock'; lock.write_text('scratch_only')
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    driver_path = ROOT / 'artifacts/continue_modebench_scale_composite_wash_v2_20260912.py'
    code = ('import os,sys,json,importlib.util; from pathlib import Path; '
            f'spec=importlib.util.spec_from_file_location("scratch_fence", {str(driver_path)!r}); '
            'a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a); '
            f'a.LOCK=Path({str(lock)!r});a.LOCK_INODE=a.LOCK.stat().st_ino; '
            f'assert Path({str(canonical)!r}).read_text()=="old"; '
            'fd=int(sys.argv[1]);a.assert_fence(fd); '
            'print(json.dumps({"inode":os.fstat(fd).st_ino}),flush=True);sys.stdin.readline();a.assert_fence(fd)')
    python = ROOT / 'var/seed_paper_eval/paper310/bin/python'
    executable, argv, environment = adapter.execution(view / 'manifest.json', [str(python), '-c', code, str(fd)])
    child = subprocess.Popen(argv, executable=executable, env=environment, pass_fds=(fd,),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([child.stdout], [], [], 10)[0], 'child readiness timeout'
        line = child.stdout.readline()
        assert line, child.stderr.read()
        assert json.loads(line)['inode'] == lock.stat().st_ino
        assert canonical.read_text() == 'neutral'  # Genuine outside-view observation.
        os.close(fd); fd = None
        contender = os.open(lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child.communicate('\n', timeout=10)
            assert child.returncode == 0
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
    finally:
        if fd is not None:
            os.close(fd)
        if child.poll() is None:
            child.kill(); child.communicate(timeout=10)


def test_relative_script_resolves_using_own_process_cwd(authority, tmp_path):
    c = authority
    proc = tmp_path / 'proc_fixture'
    directory = proc / '123'
    directory.mkdir(parents=True)
    (directory / 'cwd').symlink_to(c.a.SOURCE.parent)
    value = {'pid': 123, 'command': ['/usr/bin/python3', '-B', c.a.SOURCE.name, 'watch']}
    assert c.a.controller_invocation(value, proc)
    value['command'] = ['/usr/bin/python3', '-B', c.a.CONTROLLER.name, '--advance']
    assert c.a.controller_invocation(value, proc)
    value['command'] = ['/usr/bin/python3', '-c', c.a.SOURCE.name, 'watch']
    assert not c.a.controller_invocation(value, proc)


def test_pinned_trace_must_show_top_level_authentication_failure(authority):
    c = authority
    path = c.a.ARTIFACTS / 'controller_v2.stderr'
    path.write_text('A pinned but unrelated exception is not the required source fence evidence.')
    c.a.HISTORICAL_PINS[path] = c.a.sha(path)
    with pytest.raises(ValueError, match='top-level authentication failure missing'):
        prepare(c)
    assert not c.a.CLAIM.exists()


def test_historical_owner_must_have_been_plain_python(authority):
    c = authority
    path = c.a.ARTIFACTS / 'controller_arm_result_v2.json'
    arm = c.a.read(path)
    arm['command'] = ['/usr/bin/proot', *arm['command']]
    path.write_text(json.dumps(arm))
    c.a.HISTORICAL_PINS[path] = c.a.sha(path)
    with pytest.raises(ValueError, match='plain-Python owner identity changed'):
        prepare(c)


def test_guest_calls_only_unchanged_internal_sweep_under_fence(authority, monkeypatch):
    c = authority
    activation = prepare(c)
    scheduler(c, monkeypatch)
    gate = c.a.recovery_gate()
    monkeypatch.setattr(c.a, 'recovery_gate', lambda: gate)
    called = []
    composite = SimpleNamespace(DEFAULT_PARENT=c.a.ROOT / 'data', DEFAULT_RELEASE=c.a.ROOT / 'release',
                                DEFAULT_ARTIFACTS=c.a.ARTIFACTS)
    composite.original = SimpleNamespace(authenticate=lambda path: called.append('authenticate'))
    composite.original_plan = lambda parent: called.append('original_plan')
    def forbidden(*args, **kwargs):
        pytest.fail('do not call host-bound sweep or validate_activation')
    composite.sweep = forbidden
    composite.validate_activation = forbidden
    def science(parent, release, artifacts, *, advance):
        assert advance is True
        assert (parent, release, artifacts) == (composite.DEFAULT_PARENT, composite.DEFAULT_RELEASE, composite.DEFAULT_ARTIFACTS)
        called.append('unchanged_internal_sweep')
        return {'status': 'needs_new_development_revision'}
    composite._sweep = science
    def install(controller, record, manifest):
        assert controller is composite and manifest == c.manifest
        c.a.assert_fence(record['fence_fd'])
        assert record['owner_pid'] == os.getpid()
        called.append('install_transport')
    def load(path, name):
        if path == c.a.RECONCILIATION_VERIFIER:
            return SimpleNamespace(verify_reconciliation=lambda root: c.a.read(c.a.RECONCILIATION))
        if path == c.a.CONTROLLER:
            return composite
        assert path == c.a.TRANSPORT
        return SimpleNamespace(install=install)
    monkeypatch.setattr(c.a, 'load_module', load)
    def run_guest(command, **kwargs):
        fd = kwargs['pass_fds'][0]
        c.a.CANONICAL.write_text('old')  # Scratch simulation of the guest mapping only.
        try:
            c.a.guest_sweep(c.root, 1, fd)
            with pytest.raises(ValueError, match='already attempted'):
                c.a.guest_sweep(c.root, 1, fd)
        finally:
            c.a.CANONICAL.write_text('neutral')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.a.subprocess, 'run', run_guest)
    c.a.watch(c.root)
    assert called == ['authenticate', 'original_plan', 'install_transport', 'unchanged_internal_sweep']
    runtime = c.a.read(c.root / 'sweeps/000001/guest_runtime.json')
    assert runtime['authority']['authority_runtime_sha256'] == c.a.sha(c.root / 'outer_runtime.json')


@pytest.mark.parametrize('omitted', ['SOURCE', 'TRANSPORT', 'RUNNER', 'CLAIM', 'CONTROLLER'])
def test_even_resealed_activation_cannot_omit_mandatory_pins(authority, omitted):
    c = authority
    value = prepare(c)
    del value['inputs_sha256'][str(getattr(c.a, omitted))]
    path = c.root / 'activation.json'
    path.chmod(0o644); path.write_text(json.dumps(value))
    sidecar = c.root / 'activation.sha256.json'
    sidecar.chmod(0o644); sidecar.write_text(json.dumps({'sha256': c.a.sha(path)}))
    with pytest.raises(ValueError, match='mandatory input pins'):
        c.a.verify(c.root)


@pytest.mark.parametrize('field,value', [('historical_soak_pid', 1), ('historical_soak_start_ticks', '1'),
    ('historical_soak_os_state', 'dead'), ('host_source_sha256', '0' * 64),
    ('guest_source_sha256', '0' * 64), ('scientific_entrypoint', 'alternate:fit')])
def test_wrong_declared_authority_scope_refused(authority, field, value):
    c = authority
    activation = prepare(c)
    activation[field] = value
    path = c.root / 'activation.json'
    path.chmod(0o644); path.write_text(json.dumps(activation))
    sidecar = c.root / 'activation.sha256.json'
    sidecar.chmod(0o644); sidecar.write_text(json.dumps({'sha256': c.a.sha(path)}))
    with pytest.raises(ValueError, match='authority contract changed'):
        c.a.verify(c.root)


def test_lost_lock_is_not_silently_reacquired(authority):
    c = authority
    with c.a.lifetime_fence() as fd:
        fcntl.flock(fd, fcntl.LOCK_UN)
        with pytest.raises(ValueError, match='no longer owns its lock'):
            c.a.assert_fence(fd)
        contender = os.open(c.a.LOCK, os.O_RDWR)
        try:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)


@pytest.mark.parametrize('name', ['AUTHORIZATION', 'LOCK_QUALIFICATION'])
def test_transfer_evidence_pinned_before_claim_and_during_watch(authority, name):
    c = authority
    path = getattr(c.a, name)
    before = path.read_bytes()
    path.write_text('changed evidence')
    with pytest.raises(ValueError, match='pinned input changed'):
        prepare(c)
    assert not c.a.CLAIM.exists()
    path.write_bytes(before)
    prepare(c)
    path.write_text('changed evidence after prepare')
    with pytest.raises(ValueError, match='pinned input changed'):
        c.a.verify(c.root)


@pytest.mark.parametrize('state', ['missing_block', 'different_inode', 'wrong_host'])
def test_lock_qualification_does_not_overclaim_or_accept_wrong_observation(authority, state):
    c = authority
    qualification = c.a.read(c.a.LOCK_QUALIFICATION)
    first = json.loads(qualification['phases'][0]['stdout'])
    first[{'missing_block': 'observed', 'different_inode': 'inode', 'wrong_host': 'host'}[state]] = {
        'missing_block': 'acquired', 'different_inode': 456, 'wrong_host': 'soak.cs.princeton.edu'}[state]
    qualification['phases'][0]['stdout'] = json.dumps(first)
    c.a.LOCK_QUALIFICATION.write_text(json.dumps(qualification))
    c.a.TRANSFER_PINS[c.a.LOCK_QUALIFICATION] = c.a.sha(c.a.LOCK_QUALIFICATION)
    with pytest.raises(ValueError, match='limited cross-host exclusion'):
        prepare(c)
    assert not c.a.CLAIM.exists()


@pytest.mark.parametrize('kind', ['root', 'claim', 'watch', 'broken_symlink'])
def test_competing_unused_authority_refused_before_claim_and_after_prepare(authority, kind):
    c = authority
    prepare(c)
    if kind == 'broken_symlink':
        c.a.ALTERNATIVE_AUTHORITY.symlink_to(c.a.ROOT / 'absent')
    else:
        c.a.ALTERNATIVE_AUTHORITY.mkdir()
        if kind != 'root':
            (c.a.ALTERNATIVE_AUTHORITY / (kind + '.json')).write_text('{}')
    with pytest.raises(ValueError, match='alternative frozen-view authority exists'):
        c.a.host_probe()
    with pytest.raises(ValueError, match='alternative frozen-view authority exists'):
        c.a.verify(c.root)
    assert not (c.root / 'outer_runtime.json').exists()


def test_competing_authority_refused_before_preparation(authority):
    c = authority
    c.a.ALTERNATIVE_AUTHORITY.mkdir()
    with pytest.raises(ValueError, match='alternative frozen-view authority exists'):
        prepare(c)
    assert not c.a.CLAIM.exists()


@pytest.mark.parametrize('driver,action', [('SOURCE', 'watch'), ('SOURCE', 'guest-sweep'),
    ('ALTERNATIVE_DRIVER', 'watch'), ('ALTERNATIVE_DRIVER', 'advance-once'),
    ('CONTROLLER', '--advance'), ('PREDECESSOR', '--advance')])
def test_all_advancing_driver_invocations_recognized(authority, driver, action):
    a = authority.a
    value = {'pid': 123, 'command': [str(a.PYTHON), '-B', str(getattr(a, driver)), action]}
    assert a.controller_invocation(value)


def test_other_description_holding_lock_does_not_authenticate_wrong_fd(authority):
    a = authority.a
    with a.lifetime_fence():
        wrong = os.open(a.LOCK, os.O_RDWR)
        try:
            with pytest.raises(ValueError, match='no longer owns its lock'):
                a.assert_fence(wrong)
        finally:
            os.close(wrong)


def test_assert_fence_never_flocks_inherited_descriptor(authority, monkeypatch):
    a = authority.a
    original = fcntl.flock
    with a.lifetime_fence() as fd:
        def guarded(other, operation):
            assert other != fd, 'validation must never reacquire inherited lock'
            return original(other, operation)
        monkeypatch.setattr(a.fcntl, 'flock', guarded)
        a.assert_fence(fd)


def test_real_nfs_scratch_exclusion_across_local_processes(monkeypatch):
    """The shared filesystem is real; this does not qualify the remote soak mount."""
    a = module(ROOT / 'artifacts/continue_modebench_scale_composite_wash_v2_20260912.py', 'scratch_nfs_authority')
    python = ROOT / 'var/seed_paper_eval/paper310/bin/python'
    with tempfile.TemporaryDirectory(prefix='NONPRODUCTION-wash-fence-', dir=ROOT / 'artifacts') as directory:
        lock = Path(directory) / 'scratch.lock'
        lock.write_text('scratch only; not the controller lock')
        monkeypatch.setattr(a, 'LOCK', lock)
        monkeypatch.setattr(a, 'LOCK_INODE', lock.stat().st_ino)
        code = ('import os,fcntl,sys;fd=os.open(sys.argv[1],os.O_RDWR);'
                '\ntry:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);print("acquired")'
                '\nexcept BlockingIOError:print("blocked")'
                '\nfinally:os.close(fd)')
        def observe():
            result = subprocess.run([str(python), '-B', '-c', code, str(lock)],
                                    text=True, capture_output=True, check=True, timeout=10)
            return result.stdout.strip()
        with a.lifetime_fence() as fd:
            a.assert_fence(fd)
            assert observe() == 'blocked'
        assert observe() == 'acquired'
        assert lock.stat().st_ino == a.LOCK_INODE


@pytest.mark.parametrize('name', ['AUTHORIZATION', 'LOCK_QUALIFICATION'])
def test_resealed_activation_cannot_omit_explicit_transfer_evidence(authority, name):
    c = authority
    value = prepare(c)
    del value['inputs_sha256'][str(getattr(c.a, name))]
    path = c.root / 'activation.json'
    path.chmod(0o644); path.write_text(json.dumps(value))
    sidecar = c.root / 'activation.sha256.json'
    sidecar.chmod(0o644); sidecar.write_text(json.dumps({'sha256': c.a.sha(path)}))
    with pytest.raises(ValueError, match='wash transfer evidence omitted or changed'):
        c.a.verify(c.root)


def test_unreviewed_transport_digest_refused_before_claim(authority):
    c = authority
    c.a.TRANSPORT.write_text('different source even when provided digest matches')
    with pytest.raises(ValueError, match='reviewed transport digest required'):
        prepare(c)
    assert not c.a.CLAIM.exists()


@pytest.mark.parametrize('change', ['command', 'queue_shape', 'live_array', 'failed', 'nonzero',
                                   'node', 'unknown_end', 'invalid_end', 'duplicate_row',
                                   'missing_receipt', 'extra_receipt', 'changed_receipt'])
def test_retained_completion_revalidates_exact_failed_state_and_exact_receipts(authority, monkeypatch, change):
    c = authority
    scheduler(c, monkeypatch)
    completed = c.a.recovery_gate()
    if change == 'command':
        completed['commands'][0][2] = 'different_user'
    elif change == 'queue_shape':
        completed['stdout'][0] = 'uninterpretable queue output'
    elif change == 'live_array':
        completed['stdout'][0] = '31253118_9|RUNNING\n'
    elif change in ('failed', 'nonzero', 'node', 'unknown_end', 'invalid_end'):
        before, after = {'failed': ('FAILED', 'COMPLETED'), 'nonzero': ('1:0', '0:0'),
                         'node': ('node105', 'node106'), 'unknown_end': ('2026-09-12T02:04:21', 'Unknown'),
                         'invalid_end': ('2026-09-12T02:04:21', 'not-a-time')}[change]
        completed['stdout'][1] = completed['stdout'][1].replace(before, after)
    elif change == 'duplicate_row':
        completed['stdout'][1] *= 2
    elif change == 'missing_receipt':
        completed['receipts_sha256'].pop(next(iter(completed['receipts_sha256'])))
    elif change == 'extra_receipt':
        completed['receipts_sha256'][str(c.a.ROOT / 'unrelated.json')] = '0' * 64
    else:
        Path(next(iter(completed['receipts_sha256']))).write_text('{"scratch_only": "changed"}')
    with pytest.raises(ValueError):
        c.a.validate_completion(completed)


@pytest.mark.parametrize('change', ['missing_path', 'different_path', 'missing_sha', 'changed_record'])
def test_runtime_binds_exact_completion_path_and_bytes(authority, monkeypatch, change):
    c = authority
    prepare(c)
    scheduler(c, monkeypatch)
    path = c.root / 'recovery_completion.json'
    completed = c.a.recovery_gate()
    c.a.atomic_new(path, completed)
    runtime = {'recovery_completion_path': str(path), 'recovery_completion_sha256': c.a.sha(path)}
    assert c.a.runtime_completion(c.root, runtime) == completed
    if change == 'missing_path':
        del runtime['recovery_completion_path']
    elif change == 'different_path':
        runtime['recovery_completion_path'] = str(c.root / 'other.json')
    elif change == 'missing_sha':
        del runtime['recovery_completion_sha256']
    else:
        completed['receipts_sha256'].clear()
        path.chmod(0o644); path.write_text(json.dumps(completed))
    with pytest.raises(ValueError, match='completion linkage changed'):
        c.a.runtime_completion(c.root, runtime)


def test_guest_rejects_changed_completion_before_import_or_sweep(authority, monkeypatch):
    c = authority
    prepare(c)
    scheduler(c, monkeypatch)
    completed = c.a.recovery_gate()
    monkeypatch.setattr(c.a, 'recovery_gate', lambda: completed)
    def no_import(*args, **kwargs):
        pytest.fail('completion guard must run before importing scientific controller or transport')
    monkeypatch.setattr(c.a, 'load_module', no_import)
    def run_guest(command, **kwargs):
        path = c.root / 'recovery_completion.json'
        changed = c.a.read(path)
        changed['receipts_sha256'].clear()
        path.chmod(0o644); path.write_text(json.dumps(changed))
        c.a.CANONICAL.write_text('old')
        try:
            with pytest.raises(ValueError, match='completion linkage changed'):
                c.a.guest_sweep(c.root, 1, kwargs['pass_fds'][0])
        finally:
            c.a.CANONICAL.write_text('neutral')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(c.a.subprocess, 'run', run_guest)
    with pytest.raises(ValueError, match='never auto-retry'):
        c.a.watch(c.root)
    assert not (c.root / 'sweeps/000001/guest_runtime.json').exists()
    assert not (c.root / 'sweeps/000001/result.json').exists()


def test_forty_outputs_must_be_four_per_original_domain(authority, monkeypatch):
    c = authority
    scheduler(c, monkeypatch)
    plan = c.a.read(c.a.ORIGINAL_PLAN)
    first, second = [Path(cell['tasks']) for cell in plan['cells'][:2]]
    one, two = c.a.read(first), c.a.read(second)
    two.append(one.pop())
    first.write_text(json.dumps(one)); second.write_text(json.dumps(two))
    with pytest.raises(ValueError, match='four original tiers per domain'):
        c.a.recovery_gate()


@pytest.mark.parametrize('name', ['start_intent.json', 'outer_runtime.json', 'recovery_completion.json',
                                  'failure.json', 'terminal.json', 'sweeps/000001'])
def test_v1_prepared_presence_allowed_but_any_attempt_refuses(authority, name):
    c = authority
    assert c.a.prepared_v1_unstarted()
    (c.a.PREPARED_V1 / name).write_text('{}')
    with pytest.raises(ValueError, match='prepared v1 authority'):
        c.a.host_probe()
    assert not c.a.CLAIM.exists()


def test_v1_prepared_pins_preserved_and_advancing_invocation_recognized(authority):
    c = authority
    value = prepare(c)
    assert all(value['inputs_sha256'][str(path)] == digest for path, digest in c.a.PREPARED_V1_PINS.items())
    assert c.a.controller_invocation({'pid': 123, 'command': ['/usr/bin/python3', '-B', str(c.a.PREPARED_V1_SOURCE), 'watch']})
    c.a.PREPARED_V1_SOURCE.write_text('changed')
    with pytest.raises(ValueError, match='pinned input changed'):
        c.a.verify(c.root)


@pytest.mark.parametrize('field,value', [('schema', 'other'), ('status', 'verified'),
    ('scientific_outputs_complete', False), ('scheduler_success', True), ('exit_cause', 'guessed')])
def test_reconciliation_truthful_scope_required_before_claim(authority, field, value):
    c = authority
    saved = c.a.read(c.a.RECONCILIATION); saved[field] = value
    c.a.RECONCILIATION.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='reconciliation scope changed'):
        prepare(c)
    assert not c.a.CLAIM.exists()


@pytest.mark.parametrize('field,value', [('state', 'COMPLETED'), ('exit_code', '0:0'), ('end_utc', 'Unknown')])
def test_reconciliation_requires_actual_failed_terminal(authority, field, value):
    c = authority
    saved = c.a.read(c.a.RECONCILIATION); saved['recovery_execution'][field] = value
    c.a.RECONCILIATION.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='actual failed recovery terminal changed'):
        prepare(c)
    assert not c.a.CLAIM.exists()


@pytest.mark.parametrize('field', ['reconciliation_path', 'reconciliation_sha256', 'scheduler_success', 'exit_cause'])
def test_resealed_activation_cannot_weaken_reconciliation(authority, field):
    c = authority
    saved = prepare(c)
    saved[field] = True if field == 'scheduler_success' else 'changed'
    activation = c.root / 'activation.json'; activation.chmod(0o644); activation.write_text(json.dumps(saved))
    sidecar = c.root / 'activation.sha256.json'; sidecar.chmod(0o644)
    sidecar.write_text(json.dumps({'sha256': c.a.sha(activation)}))
    with pytest.raises(ValueError, match='wash authority contract changed'):
        c.a.verify(c.root)


def test_readonly_reconciliation_transport_is_explicit_frozen_guest(authority, monkeypatch):
    c = authority
    actual = module(ROOT / 'artifacts/continue_modebench_scale_composite_wash_v2_20260912.py', 'readonly_reconciliation_original')
    method = actual.verify_reconciliation_subprocess
    method = type(method)(method.__code__, c.a.__dict__, method.__name__, method.__defaults__, method.__closure__)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert '--verify-existing' in command and '--publish-reconciliation' not in command
        assert command == c.a.reconciliation_command(c.manifest)
        value = {'schema': c.a.RECONCILIATION_SCHEMA, 'status': c.a.RECONCILIATION_STATUS,
                 'scientific_outputs_complete': True, 'scheduler_success': False, 'exit_cause': 'unknown',
                 'certificate_path': str(c.a.RECONCILIATION), 'certificate_sha256': c.a.sha(c.a.RECONCILIATION)}
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr='')
    monkeypatch.setattr(c.a.subprocess, 'run', run)
    assert method(c.manifest)['returncode'] == 0
    assert len(calls) == 1
    assert not c.a.CLAIM.exists()
