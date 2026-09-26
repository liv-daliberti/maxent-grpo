"""Exercise heldout submission boundaries with synthetic providers and children."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('heldout_activation_recorder_test', ROOT/'artifacts/run_modebench_scale_level4_first_r4_transport_activation_20260913.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')
    return path


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    state = tmp_path/'release'
    state.mkdir()
    for key, value in {'STATE':state, 'OUT':state/'activation',
        'PROVIDER':write(tmp_path/'provider.py', {'fixture':'provider'}),
        'TRANSPORT':write(tmp_path/'transport.py', {'fixture':'transport'}),
        'REVIEW':tmp_path/'review.json'}.items():
        monkeypatch.setattr(a, key, value)
    pins = {str(p):a.sha(p) for p in (a.SOURCE, a.PROVIDER, a.TRANSPORT)}
    write(a.REVIEW, {'status':'reviewed', 'blocking_findings':[], 'files_sha256':pins})
    lock = write(tmp_path/'lock', {})
    held, calls, entries = [], [], []
    dispatch = {'prepare':0, 'submit':0, 'drift':False}

    @contextmanager
    def fence():
        entries.append('acquired')
        held.append(8)
        try:
            yield 8
        finally:
            held.clear()

    def assert_fence(fd):
        assert held == [fd] == [8]

    provider = SimpleNamespace(live_host_guard=lambda **kw: None,
        process_identity=lambda pid:{'pid':pid, 'command':[str(a.PYTHON), '-B', str(a.SOURCE)]},
        lifetime_fence=fence, assert_fence=assert_fence, LOCK=lock, LOCK_INODE=lock.stat().st_ino,
        RUNNER=tmp_path/'runner.py', MANIFEST=tmp_path/'manifest.json')
    util = SimpleNamespace(spec_from_file_location=lambda *args:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None)),
        module_from_spec=lambda spec:provider)
    monkeypatch.setattr(a, 'importlib', SimpleNamespace(util=util))

    def run(argv, **kwargs):
        assert held == [8] and kwargs['pass_fds'] == (8,)
        assert kwargs['env']['OPENBLAS_NUM_THREADS'] == kwargs['env']['OMP_NUM_THREADS'] == '1'
        action = argv[argv.index('--root')-1]
        calls.append((action, argv))
        assert str(state) == argv[argv.index('--root')+1]
        assert argv[:7] == [str(a.PYTHON), '-B', str(provider.RUNNER), 'exec', '--manifest', str(provider.MANIFEST), '--']
        kwargs['stdout'].write('{"synthetic_child":true}\n')
        (state/'confirmation').mkdir(exist_ok=True)
        if action == 'submit' and dispatch[action] == 0:
            write(state/'confirmation/submission_result.json', {'status':'submitted','returncode':0,'array_job_id':70000666})
        if action == 'prepare' and dispatch['drift']:
            a.TRANSPORT.write_text('changed after preparation')
        return SimpleNamespace(returncode=dispatch[action])

    monkeypatch.setattr(a, 'subprocess', SimpleNamespace(run=run))

    def launch():
        monkeypatch.setattr(a.sys, 'argv', [str(a.SOURCE), '--provider-sha256', a.sha(a.PROVIDER),
            '--transport-sha256', a.sha(a.TRANSPORT), '--review-sha256', a.sha(a.REVIEW)])
        return a.main()

    return SimpleNamespace(state=state, held=held, calls=calls, entries=entries, dispatch=dispatch, run=launch)


def test_one_owner_retains_descriptor_across_prepare_and_submit(fixture):
    f = fixture
    f.run()
    assert f.entries == ['acquired'] and not f.held
    assert [action for action, _ in f.calls] == ['prepare','submit']
    assert json.loads((a.OUT/'result.json').read_text())['array_job_id'] == 70000666
    for action, argv in f.calls:
        intent = json.loads((a.OUT/(action+'_intent.json')).read_text())
        exited = json.loads((a.OUT/(action+'_exit.json')).read_text())
        assert intent['command'] == argv and exited['returncode'] == 0
        assert exited['intent_sha256'] == a.sha(a.OUT/(action+'_intent.json'))
    with pytest.raises(ValueError, match='fresh'):
        f.run()
    assert len(f.calls) == 2


@pytest.mark.parametrize('code', [1,143,-15])
def test_failed_prepare_is_preserved_and_never_submitted(fixture, code):
    f = fixture
    f.dispatch['prepare'] = code
    with pytest.raises(ValueError, match='literal child failure'):
        f.run()
    assert [action for action, _ in f.calls] == ['prepare']
    assert json.loads((a.OUT/'prepare_exit.json').read_text())['returncode'] == code
    assert not (a.OUT/'submit_intent.json').exists() and not f.held


def test_changed_transport_after_prepare_prevents_submission(fixture):
    f = fixture
    f.dispatch['drift'] = True
    with pytest.raises(ValueError, match='source changed'):
        f.run()
    assert [action for action, _ in f.calls] == ['prepare']
    assert not (a.OUT/'submit_intent.json').exists()


def test_unreviewed_recorder_cannot_create_action_or_submit(fixture):
    review = json.loads(a.REVIEW.read_text())
    del review['files_sha256'][str(a.SOURCE)]
    write(a.REVIEW, review)
    with pytest.raises(ValueError, match='bind this exact activation recorder'):
        fixture.run()
    assert not a.OUT.exists() and not fixture.calls and not fixture.entries
