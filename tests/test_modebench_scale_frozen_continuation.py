"""Scratch authority/coordination tests; no canonical controller lock or jobs.

Real NFS and PRoot tests use only newly created NONPRODUCTION scratch files.
Scientific branch fixtures substitute expensive generation, never a real model.
"""
import ast
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest

from test_continue_modebench_scale_composite import campaign

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/continue_modebench_scale_frozen_view_20260912.py'


def load(path=SOURCE, name='scratch_frozen_authority'):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class ScienceNormalization(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.name = {'advance_sweep': '_sweep'}.get(node.name, node.name)
        pairs = [(arg, default) for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                 if arg.arg != 'view_manifest']
        node.args.kwonlyargs = [arg for arg, _ in pairs]
        node.args.kw_defaults = [default for _, default in pairs]
        return self.generic_visit(node)

    def visit_Call(self, node):
        if any(kw.arg == 'view_manifest' for kw in node.keywords):
            assert isinstance(node.func, ast.Name) and node.func.id == 'submitted_stage'
        node.keywords = [kw for kw in node.keywords if kw.arg != 'view_manifest']
        if isinstance(node.func, ast.Name) and node.func.id == 'dispatch_submit':
            assert len(node.args) == 2 and isinstance(node.args[1], ast.Name) and node.args[1].id == 'view_manifest'
            node.func = ast.Attribute(value=ast.Name(id='launch', ctx=ast.Load()), attr='submit', ctx=ast.Load())
            node.args = node.args[:1]
        return self.generic_visit(node)


@pytest.mark.parametrize('new,old', [('advance_sweep', '_sweep'), ('submitted_stage', 'submitted_stage')])
def test_scientific_coordination_ast_parity(new, old):
    original = ast.parse((ROOT / 'ops/exp_scaling/continue_modebench_scale_composite.py').read_text())
    revised = ast.parse(SOURCE.read_text())
    get = lambda tree, name: next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    expected = get(original, old)
    actual = ScienceNormalization().visit(get(revised, new))
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


def test_no_core_monkeypatch_or_fake_host_in_source():
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute):
                    assert ast.unparse(target) not in {'sealed.launch', 'sealed.sweep', 'sealed.validate_activation',
                                                     'os.uname', 'socket.gethostname', 'launch.submit'}


@pytest.fixture
def local(tmp_path, monkeypatch):
    h = load()
    base = tmp_path / 'scratch'
    base.mkdir()
    lock = base / 'controller.lock'
    lock.write_text('existing scratch inode')
    root = base / 'authority'
    for key, value in {'ROOT': base, 'AUTHORITY_ROOT': root, 'LOCK': lock,
                       'PARENT': base / 'parent', 'RELEASE': base / 'release', 'ARTIFACTS': base / 'old',
                       'ACTUAL_HOST': 'wash.scratch', 'SOURCE': base / 'helper.py',
                       'PYTHON': Path(sys.executable), 'RUNNER': base / 'runner.py'}.items():
        monkeypatch.setattr(h, key, value)
    h.SOURCE.write_text('scratch helper')
    h.RUNNER.write_text('scratch runner')
    monkeypatch.setattr(h.socket, 'gethostname', lambda: 'wash.scratch')
    monkeypatch.setattr(h, 'host_preflight', lambda manifest: {str(h.SOURCE): h.file_sha(h.SOURCE)})
    manifest = write(base / 'manifest.json', {'scratch': True})
    return SimpleNamespace(h=h, base=base, root=root, manifest=manifest, lock=lock)


def prepare_fixture(state):
    h = state.h
    state.root.mkdir()
    (state.root / 'cycles').mkdir()
    data = {'schema': h.SCHEMA, 'root': str(state.root), 'prepared_host': h.ACTUAL_HOST,
            'parent': str(h.PARENT), 'release': str(h.RELEASE), 'artifacts': str(h.ARTIFACTS),
            'lock': str(h.LOCK), 'view_manifest': str(state.manifest),
            'operational_pins': h.host_preflight(state.manifest), 'scientific_pins': {},
            'predecessor': {'status': 'fenced', 'remote_liveness': 'unknown'}, 'recovery_job_id': 123}
    write(state.root / 'plan.json', data)
    write(state.root / 'plan.sha256.json', {'sha256': h.file_sha(state.root / 'plan.json')})
    return data


def test_lock_real_same_process_description_exclusion(local):
    h = local.h
    before = local.lock.read_bytes()
    with h.lifetime_lock() as fd:
        assert h.verify_lock(fd) == h.lock_identity(fd)
        with pytest.raises(BlockingIOError):
            with h.lifetime_lock():
                pytest.fail('second authority acquired same lock')
    assert local.lock.read_bytes() == before


def test_unheld_inherited_fd_is_not_silently_reacquired(local):
    h = local.h
    fd = os.open(local.lock, os.O_RDWR)
    try:
        with pytest.raises(ValueError, match='did not exclude'):
            h.verify_lock(fd)
    finally:
        os.close(fd)


def test_wrong_inode_rejected(local):
    h = local.h
    with h.lifetime_lock() as fd:
        local.lock.rename(local.lock.with_suffix('.preserved'))
        local.lock.write_text('different inode')
        with pytest.raises(ValueError, match='inode changed'):
            h.verify_lock(fd)


def test_symlink_lock_rejected(local):
    local.lock.rename(local.lock.with_suffix('.old'))
    local.lock.symlink_to(local.lock.with_suffix('.old'))
    with pytest.raises(OSError):
        with local.h.lifetime_lock():
            pytest.fail('symlink lock accepted')


def test_missing_predecessor_evidence_refuses(local):
    with pytest.raises(ValueError, match='launch-to-log linkage'):
        local.h.predecessor_evidence()


def test_outside_plan_rechecks_source_fence(local, monkeypatch):
    prepare_fixture(local)
    local.h.verify_plan(local.root, outside=True)
    monkeypatch.setattr(local.h, 'host_preflight', lambda manifest: {'different': 'source'})
    with pytest.raises(ValueError, match='source fence'):
        local.h.verify_plan(local.root, outside=True)


def test_plan_drift_and_false_dead_predecessor_rejected(local):
    plan = prepare_fixture(local)
    (local.root / 'plan.json').write_text('{}')
    with pytest.raises(ValueError, match='plan changed'):
        local.h.verify_plan(local.root, outside=True)
    plan['predecessor']['remote_liveness'] = 'dead'
    write(local.root / 'plan.json', plan)
    write(local.root / 'plan.sha256.json', {'sha256': local.h.file_sha(local.root / 'plan.json')})
    with pytest.raises(ValueError, match='scope changed'):
        local.h.verify_plan(local.root, outside=True)


def test_prepare_has_no_lock_and_no_scientific_mutations(local, monkeypatch):
    h = local.h
    inspected = {'status': 'ready', 'host': h.ACTUAL_HOST, 'scientific_pins': {}, 'recovery_job_id': 123,
                 'predecessor': {'status': 'fenced', 'remote_liveness': 'unknown'}}
    calls = []
    def inspect(command, **kwargs):
        calls.append((command, kwargs))
        assert 'inspect' in command and kwargs['close_fds'] is True and 'pass_fds' not in kwargs
        return SimpleNamespace(returncode=0, stdout=json.dumps(inspected), stderr='')
    monkeypatch.setattr(h.subprocess, 'run', inspect)
    def forbidden():
        pytest.fail('prepare must not acquire controller lock')
    monkeypatch.setattr(h, 'lifetime_lock', forbidden)
    result = h.prepare(local.root, local.manifest)
    assert result['status'] == 'prepared_only' and len(calls) == 1
    assert not (local.root / 'start_intent.json').exists()
    with pytest.raises(ValueError, match='already prepared'):
        h.prepare(local.root, local.manifest)


def test_prepare_failed_readiness_leaves_no_authority(local, monkeypatch):
    monkeypatch.setattr(local.h.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stderr='recovery still running', stdout=''))
    with pytest.raises(ValueError, match='inspection failed'):
        local.h.prepare(local.root, local.manifest)
    assert not local.root.exists()


def test_start_intent_prevents_automatic_restart(local):
    prepare_fixture(local)
    write(local.root / 'start_intent.json', {'previous': 'attempt'})
    with pytest.raises(ValueError, match='already attempted'):
        local.h.supervisor(local.root, cycles=1)


def test_supervisor_continuous_lock_parent_and_child_result(local, monkeypatch):
    h = local.h
    prepare_fixture(local)
    events = []
    class FakeProcess:
        pid = os.getpid()
        def __init__(self, command, **kwargs):
            events.append(command)
            fd, = kwargs['pass_fds']
            assert kwargs['close_fds'] is True
            h.verify_lock(fd)
            with pytest.raises(BlockingIOError):
                with h.lifetime_lock():
                    pytest.fail('parent lock released before child')
        def wait(self):
            write(h.cycle_prefix(local.root, 0).with_suffix('.result.json'),
                  {'authority_started_sha256': h.file_sha(local.root / 'authority_started.json'),
                   'cycle': 0, 'result': {'status': 'needs_new_development_revision'}})
            return 0
    monkeypatch.setattr(h.subprocess, 'Popen', FakeProcess)
    result = h.supervisor(local.root, cycles=1)
    assert result['status'] == 'needs_new_development_revision'
    assert len(events) == 1
    assert (local.root / 'authority_exit.json').exists()
    with pytest.raises(ValueError, match='already attempted'):
        h.supervisor(local.root, cycles=1)
    with h.lifetime_lock():
        pass


def test_child_failure_stops_no_retry(local, monkeypatch):
    h = local.h
    prepare_fixture(local)
    calls = []
    class FailedProcess:
        pid = os.getpid()
        def __init__(self, *args, **kwargs):
            calls.append(args)
        def wait(self):
            return 7
    monkeypatch.setattr(h.subprocess, 'Popen', FailedProcess)
    with pytest.raises(ValueError, match='child failed'):
        h.supervisor(local.root, cycles=1)
    assert len(calls) == 1 and not (local.root / 'authority_exit.json').exists()
    with pytest.raises(ValueError, match='already attempted'):
        h.supervisor(local.root, cycles=1)


def test_prior_ledger_adds_completed_recovery_without_rewriting_core(local, monkeypatch):
    h = local.h
    monkeypatch.setattr(h.sealed, 'prior_submissions', lambda artifacts: [10, 20])
    calls = []
    def complete():
        calls.append(1)
        return {'job_id': 30}
    monkeypatch.setattr(h, 'dispatcher', lambda: SimpleNamespace(completed_recovery=complete))
    assert h.prior_submissions(local.base) == [10, 20, 30]
    assert calls == [1]
    monkeypatch.setattr(h.sealed, 'prior_submissions', lambda artifacts: None)
    assert h.prior_submissions(local.base) is None and calls == [1]


def test_explicit_dispatch_public_hook(local, monkeypatch):
    calls = []
    monkeypatch.setattr(local.h, 'dispatcher', lambda: SimpleNamespace(
        submit=lambda plan, manifest: calls.append((plan, manifest)) or {'status': 'submitted'}))
    assert local.h.dispatch_submit('plan', 'view') == {'status': 'submitted'}
    assert calls == [('plan', 'view')]


@pytest.fixture
def science(campaign, monkeypatch):
    h = load(name='scratch_science_authority')  # imports fixture's explicitly patched science APIs
    state = campaign
    monkeypatch.setattr(h, 'submitted_stage', lambda *args, view_manifest, **kwargs:
                        state.fake_stage(*args, **kwargs))
    state.run_frozen = lambda: h.advance_sweep(state.parent, state.release, state.artifacts,
                                             advance=True, view_manifest='scratch-view')
    return state


def test_missing_original_does_not_advance(science):
    (science.parent / 'level5/results/development/pantry/difficulty_3.json').unlink()
    result = science.run_frozen()
    assert result['status'] == 'waiting_original_development'
    assert not science.release.exists()


def test_missing_source_keeps_early_return(science):
    write(science.artifacts / 'revision_bindings.json',
          {'schema': 'modebench_scale_composite_revision_bindings_v1', 'revisions': {}})
    result = science.run_frozen()
    assert result['status'] == 'needs_source_manifest'
    assert not any(event[0].startswith('freeze') for event in science.events)


def test_all_carried_freezes_before_any_revision_pool(science):
    science.complete_dev = False
    result = science.run_frozen()
    assert result['status'] == 'waiting_revision_development'
    first_pool = next(i for i, event in enumerate(science.events) if event[0] == 'revision_pools')
    frozen = {(event[1], event[2]) for event in science.events[:first_pool] if event[0] == 'freeze_carry'}
    expected = {(level, domain) for level in ('level4', 'level5')
                for domain in ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry')} - science.failed
    assert frozen == expected


def test_failed_revision_stops_before_heldout(science):
    science.revision_failed.add(('level5', 'mathir'))
    result = science.run_frozen()
    assert result['status'] == 'needs_new_development_revision'
    assert not any(event[:2] == ('stage', 'confirmation') for event in science.events)


@pytest.fixture
def nfs_scratch():
    path = Path(tempfile.mkdtemp(prefix='NONPRODUCTION-frozen-authority-tests-', dir=ROOT / 'artifacts'))
    try:
        yield path
    finally:
        for item in path.rglob('*'):
            if item.is_dir():
                item.chmod(0o755)
        path.chmod(0o755)
        shutil.rmtree(path)


def test_real_nfs_cross_process_lock(nfs_scratch, monkeypatch):
    h = load(name='real_scratch_nfs_authority')
    lock = nfs_scratch / 'only-scratch.lock'
    lock.touch()
    monkeypatch.setattr(h, 'LOCK', lock)
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    matches = [row for row in mounts if str(lock).startswith(row[1].rstrip('/') + '/')]
    selected = max(matches, key=lambda row: len(row[1]))
    assert selected[2].startswith('nfs') and 'local_lock=none' in selected[3].split(',')
    code = "import fcntl,sys\nf=open(sys.argv[1],'a')\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(23)\nsys.exit(0)\n"
    with h.lifetime_lock():
        result = subprocess.run([sys.executable, '-B', '-c', code, str(lock)], close_fds=True, timeout=10)
        assert result.returncode == 23
    result = subprocess.run([sys.executable, '-B', '-c', code, str(lock)], close_fds=True, timeout=10)
    assert result.returncode == 0


def test_real_proot_child_retains_inherited_nfs_lock_and_view(nfs_scratch, monkeypatch):
    adapter = load(ROOT / 'artifacts/run_modebench_scale_frozen_view_20260912.py', 'real_scratch_fd_view')
    base = nfs_scratch
    (base / 'artifacts').mkdir()
    canonical = base / 'canonical.py'
    canonical.write_text('NEUTRAL_SCRATCH_ONLY')
    preserved = base / 'preserved.py'
    preserved.write_text('HISTORICAL_SCRATCH_ONLY')
    for key, value in {'ROOT': base, 'CANONICAL': canonical, 'PRESERVED': preserved,
                       'OLD_SHA256': adapter.file_sha(preserved),
                       'NEUTRAL_SHA256': adapter.file_sha(canonical)}.items():
        monkeypatch.setattr(adapter, key, value)
    view = base / 'artifacts/view'
    adapter.prepare(view)
    lock = base / 'inherited.lock'
    lock.touch()
    ready = base / 'child.ready.json'
    script = base / 'child.py'
    script.write_text('''import fcntl,json,os,sys\nfrom pathlib import Path\nfd=int(sys.argv[1]);lock=Path(sys.argv[2]);ready=Path(sys.argv[3]);canonical=Path(sys.argv[4])\nassert os.fstat(fd).st_ino==lock.stat().st_ino\nfcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\nwith lock.open('a') as probe:\n try: fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)\n except BlockingIOError: pass\n else: raise AssertionError('lost inherited exclusion')\nready.write_text(json.dumps({'pid':os.getpid(),'source':canonical.read_text()}))\nassert sys.stdin.readline().strip()=='finish'\n''')
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    executable, argv, environment = adapter.execution(view / 'manifest.json',
        [sys.executable, '-B', str(script), str(fd), str(lock), str(ready), str(canonical)])
    child = subprocess.Popen(argv, executable=executable, env=environment, pass_fds=(fd,),
                             close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.exists(), child.communicate(timeout=2)
        assert json.loads(ready.read_text())['source'] == 'HISTORICAL_SCRATCH_ONLY'
        os.close(fd)
        fd = None
        with lock.open('a') as independent:
            with pytest.raises(BlockingIOError):
                fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output, error = child.communicate('finish\n', timeout=10)
        assert child.returncode == 0, (output, error)
        assert canonical.read_text() == 'NEUTRAL_SCRATCH_ONLY'
        with lock.open('a') as independent:
            fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if fd is not None:
            os.close(fd)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def make_child_records(state, fd, monkeypatch):
    h = state.h
    plan = prepare_fixture(state)
    plan['scientific_pins'] = {str(h.SOURCE): h.file_sha(h.SOURCE)}
    write(state.root / 'plan.json', plan)
    write(state.root / 'plan.sha256.json', {'sha256': h.file_sha(state.root / 'plan.json')})
    owner = h.proc_identity(os.getpid())
    intent = {'host': h.ACTUAL_HOST, 'owner': owner, 'lock': h.lock_identity(fd),
              'plan_sha256': h.file_sha(state.root / 'plan.json')}
    write(state.root / 'start_intent.json', intent)
    started = {**intent, 'intent_sha256': h.file_sha(state.root / 'start_intent.json')}
    write(state.root / 'authority_started.json', started)
    prefix = h.cycle_prefix(state.root, 0)
    command = h.view_command(plan['view_manifest'], 'advance-once', '--root', state.root,
                             '--authority-fd', fd, '--cycle', 0)
    write(Path(str(prefix) + '.intent.json'), {'command': command,
          'authority_started_sha256': h.file_sha(state.root / 'authority_started.json')})
    write(Path(str(prefix) + '.launch.json'), {'process': owner, 'command': command})
    monkeypatch.setattr(h, 'inspect_inputs', lambda manifest: {'recovery_job_id': 123,
                         'scientific_pins': plan['scientific_pins']})
    return prefix


def test_advance_child_validates_identity_and_never_reenters(local, monkeypatch):
    h = local.h
    with h.lifetime_lock() as fd:
        prefix = make_child_records(local, fd, monkeypatch)
        calls = []
        monkeypatch.setattr(h, 'advance_sweep', lambda *args, **kwargs:
                            calls.append((args, kwargs)) or {'status': 'needs_source_manifest'})
        assert h.advance_once(local.root, fd, 0)['status'] == 'needs_source_manifest'
        assert len(calls) == 1
        recorded = h.read(Path(str(prefix) + '.runtime.json'))
        assert recorded['actual_process']['pid'] == os.getpid()
        with pytest.raises(ValueError, match='already entered'):
            h.advance_once(local.root, fd, 0)
        assert len(calls) == 1


@pytest.mark.parametrize('damage', ['host', 'owner_ticks', 'launch_command', 'scientific_pin'])
def test_child_bad_identity_never_advances(local, monkeypatch, damage):
    h = local.h
    with h.lifetime_lock() as fd:
        prefix = make_child_records(local, fd, monkeypatch)
        if damage in ('host', 'owner_ticks'):
            path = local.root / 'authority_started.json'
            value = h.read(path)
            if damage == 'host':
                value['host'] = 'soak.scratch'
            else:
                value['owner']['start_ticks'] = 'wrong'
            write(path, value)
        elif damage == 'launch_command':
            path = Path(str(prefix) + '.launch.json')
            value = h.read(path); value['command'] = ['wrong']
            write(path, value)
        else:
            h.SOURCE.write_text('drift')
        def forbidden(*args, **kwargs):
            pytest.fail('bad authority must not advance science')
        monkeypatch.setattr(h, 'advance_sweep', forbidden)
        with pytest.raises(ValueError):
            h.advance_once(local.root, fd, 0)
        assert not Path(str(prefix) + '.runtime.json').exists()


def test_real_parent_exit_child_keeps_inherited_nfs_authority(nfs_scratch):
    lock = nfs_scratch / 'orphaned-parent.lock'
    ready, finish = nfs_scratch / 'orphan.ready', nfs_scratch / 'orphan.finish'
    child_script = nfs_scratch / 'orphan_child.py'
    child_script.write_text("import fcntl,os,sys,time\nfrom pathlib import Path\nfd=int(sys.argv[1]);ready=Path(sys.argv[2]);finish=Path(sys.argv[3])\nfcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\nready.write_text(str(os.getpid()))\nwhile not finish.exists(): time.sleep(.05)\n")
    parent_script = nfs_scratch / 'exiting_parent.py'
    parent_script.write_text("import fcntl,os,subprocess,sys,time\nfrom pathlib import Path\nlock=Path(sys.argv[1]);fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600)\nfcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\nsubprocess.Popen([sys.executable,'-B',sys.argv[2],str(fd),sys.argv[3],sys.argv[4]],pass_fds=(fd,),close_fds=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nwhile not Path(sys.argv[3]).exists(): time.sleep(.01)\nos._exit(0)\n")
    parent = subprocess.Popen([sys.executable, '-B', str(parent_script), str(lock), str(child_script),
                               str(ready), str(finish)], close_fds=True)
    try:
        assert parent.wait(timeout=10) == 0 and ready.exists()
        child_pid = int(ready.read_text())
        assert Path('/proc', str(child_pid)).exists()
        with lock.open('a') as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finish.touch()
            deadline = time.monotonic() + 10
            while True:
                try:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline
                    time.sleep(.05)
    finally:
        finish.touch()
        if parent.poll() is None:
            parent.kill(); parent.wait(timeout=5)


def test_inspection_missing_predecessor_evidence_has_no_mutation(local, monkeypatch):
    h = local.h
    canonical = local.base / 'historical_scratch_source.py'
    canonical.write_text('scratch historical')
    monkeypatch.setattr(h, 'adapter', lambda: SimpleNamespace(CANONICAL=canonical, OLD_SHA256=h.file_sha(canonical)))
    monkeypatch.setattr(h, 'operational_pins', lambda manifest: {})
    monkeypatch.setattr(h, 'load', lambda *args: SimpleNamespace(authenticate_activation=lambda: {'scratch': 'pin'}))
    def forbidden(*args, **kwargs):
        pytest.fail('missing predecessor evidence must stop before later work')
    monkeypatch.setattr(h, 'dispatcher', forbidden)
    monkeypatch.setattr(h, 'lifetime_lock', forbidden)
    before = {str(p): p.read_bytes() for p in local.base.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='launch-to-log linkage'):
        h.inspect_inputs(local.manifest)
    after = {str(p): p.read_bytes() for p in local.base.rglob('*') if p.is_file()}
    assert after == before
