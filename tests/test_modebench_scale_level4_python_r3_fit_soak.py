"""Synthetic portability and actual-audit-history barriers; no real science."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_modebench_scale_level5_ready_fits import fixture as prior_fixture
from test_modebench_scale_level4_python_r3_fit import fixture as python_fixture, write, change

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT/'artifacts/continue_modebench_scale_level4_python_r3_fit_soak_20260913.py'
    spec = importlib.util.spec_from_file_location('scratch_soak_python_fit', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def fixture(python_fixture, monkeypatch):
    x = python_fixture
    a, m = load(), x.m
    monkeypatch.setattr(a, 'prior', x.a)
    monkeypatch.setattr(a, 'engine', m)
    historical_first = m.first
    monkeypatch.setattr(m, 'first', a.live_facade(historical_first))
    monkeypatch.setattr(m.first, 'host_guard', historical_first.host_guard)
    for key in ('SOURCE', 'TESTS'):
        monkeypatch.setattr(m, key, getattr(a, key))
    monkeypatch.setattr(m, 'SEALED', {**m.SEALED, **a.PREDECESSOR_PINS, **a.SOAK_PREVIOUS_PINS})
    monkeypatch.setattr(m, 'reviewed', a.reviewed)
    monkeypatch.setattr(a, 'AUDIT_ROOT', x.f.state.parent/'audit_outer')
    monkeypatch.setattr(a, 'PRIOR_STATE', x.f.state.parent/'old_host_fit_state')
    monkeypatch.setattr(a, 'READONLY_ROOT', x.f.state.parent/'audit_readonly')
    x.f.owner['command'] = [str(m.PYTHON), '-B', str(m.SOURCE), 'fit']
    change(x.certificate, audit_completed_at_utc='2026-09-13T04:00:30+00:00')
    oldreview = m.read(m.REVIEW)
    oldreview['files_sha256'][str(x.certificate)] = m.sha(x.certificate)
    oldreview['execution_certificate_sha256'] = m.sha(x.certificate)
    write(m.REVIEW, oldreview)
    proof = m.read(x.certificate)
    for i, (action, root) in enumerate([('audit', a.AUDIT_ROOT), ('verify', a.READONLY_ROOT)]):
        command = a.audit_command(action)
        identity = {'schema': 'modebench_scale_python_r3_soak_outer_execution_v1',
            'command': command, 'action': action, 'actual_host': a.LIVE_HOST,
            'uid': a.LIVE_UID, 'pid': 8000+i, 'started_at_utc': f'2026-09-13T04:0{i*2}:00+00:00'}
        write(root/'intent.json', identity)
        write(root/'process.json', {'at_utc': f'2026-09-13T04:0{i*2}:01+00:00', 'child_pid': 9000+i,
            **{k: identity[k] for k in ('actual_host', 'uid', 'pid')}})
        (root/'stdout.txt').write_text(json.dumps({key: proof[key] for key in
            ('schema', 'status', 'array_job_id')} | {'files': len(proof['files_sha256'])})+'\n')
        (root/'stderr.txt').write_text('')
        write(root/'exit.json', {**identity, 'returncode': 143 if action == 'audit' else 0,
            'certificate_present': True, 'new_grader_invocations': 24704 if action == 'audit' else 0,
            'intent_sha256': m.sha(root/'intent.json'), 'certificate_sha256': m.sha(x.certificate),
            'finished_at_utc': f'2026-09-13T04:0{i*2+1}:00+00:00',
            'stdout_sha256': m.sha(root/'stdout.txt'), 'stderr_sha256': m.sha(root/'stderr.txt')})
    for name in ('claim.json','registration.json'):
        write(m.RECOVERY/'execution_audit'/name, {'scratch_observed_audit_history': name})
    observed_paths = [*a.AUDIT_ROOT.iterdir(), x.certificate,
                      m.RECOVERY/'execution_audit/claim.json', m.RECOVERY/'execution_audit/registration.json']
    monkeypatch.setattr(a, 'OBSERVED_AUDIT_PINS', {p: m.sha(p) for p in observed_paths})
    value = m.read(m.REVIEW)
    value.update(operational_amendment=a.AMENDMENT, cpu_fit_host=a.LIVE_HOST, cpu_fit_uid=a.LIVE_UID,
                 audit_execution_policy=a.AUDIT_EXECUTION_POLICY, audit_outer_returncode=143, audit_exit_cause='unknown')
    value['files_sha256'].update({str(p): m.sha(p) for p in (m.SOURCE, m.TESTS, *a.PREDECESSOR_PINS, *a.SOAK_PREVIOUS_PINS,
        *a.OBSERVED_AUDIT_PINS, *a.READONLY_ROOT.iterdir())})
    value.update(audit_outer_exit_sha256=m.sha(a.AUDIT_ROOT/'exit.json'),
                 audit_readonly_exit_sha256=m.sha(a.READONLY_ROOT/'exit.json'))
    write(m.REVIEW, value)
    return SimpleNamespace(a=a, x=x, m=m, f=x.f, historical_first=historical_first,
        run=lambda: a.run(x.f.state, m.sha(m.REVIEW)))


def repin(x):
    """Scratch record rehashing exposes semantic checks, never real evidence."""
    for root in (x.a.AUDIT_ROOT, x.a.READONLY_ROOT):
        change(root/'exit.json', stdout_sha256=x.m.sha(root/'stdout.txt'),
               stderr_sha256=x.m.sha(root/'stderr.txt'), intent_sha256=x.m.sha(root/'intent.json'))
    value = x.m.read(x.m.REVIEW)
    value['files_sha256'] = {p: x.m.sha(p) for p in value['files_sha256']}
    value.update(audit_outer_exit_sha256=x.m.sha(x.a.AUDIT_ROOT/'exit.json'),
                 audit_readonly_exit_sha256=x.m.sha(x.a.READONLY_ROOT/'exit.json'))
    write(x.m.REVIEW, value)


def test_actual_soak_records_and_original_one_fit_are_preserved(fixture):
    x = fixture
    result = x.run()
    assert x.f.calls == ['python_factors'] and not x.f.held
    assert x.a.verify_existing(x.f.state) == result
    assert x.f.calls == ['python_factors'] and x.x.verification == [x.m.RECOVERY]
    registration = x.m.read(x.f.state/'registration.json')
    assert registration['host'] == 'soak.cs.princeton.edu'
    assert registration['revisions'][0]['execution']['state'] == 'FAILED'
    assert registration['revisions'][0]['execution']['exit_code'] == '143:0'
    assert x.historical_first.HOST == x.m.BASE.HOST == 'spin.cs.princeton.edu'
    for root in (x.a.AUDIT_ROOT, x.a.READONLY_ROOT):
        assert all(registration['files_sha256'][str(p)] == x.m.sha(p) for p in root.iterdir())
    with pytest.raises(ValueError, match='already attempted'):
        x.run()


def test_negative_fit_is_saved_without_retry(fixture):
    x = fixture
    x.f.failures.add('python_factors')
    result = x.run()
    assert result['failed_domains'] == ['python_factors']
    assert x.a.verify_existing(x.f.state) == result and x.f.calls == ['python_factors']


def test_no_source_or_historical_module_mutation_and_same_fence_functions():
    a = load()
    assert a.HISTORICAL_FIRST.HOST == a.engine.BASE.HOST == 'spin.cs.princeton.edu'
    assert a.engine.first.HOST == 'soak.cs.princeton.edu'
    assert a.engine.first is not a.HISTORICAL_FIRST
    assert a.engine.assert_fence is a.HISTORICAL_FIRST.assert_fence
    assert a.engine.lifetime_fence is a.HISTORICAL_FIRST.lifetime_fence
    assert a.engine.fit_registered is a.prior.fit_registered
    assert a.engine.collect_inputs is a.prior.collect_inputs
    assert all(a.engine.sha(p) == d for p, d in a.PREDECESSOR_PINS.items())


@pytest.mark.parametrize('guest', [False, True])
@pytest.mark.parametrize('host,uid,source', [
    ('soak.cs.princeton.edu', 363432, 'expected'),
    ('spin.cs.princeton.edu', 363432, 'expected'),
    ('soak.cs.princeton.edu', 363433, 'expected'),
    ('soak.cs.princeton.edu', 363432, 'wrong'),
])
def test_real_new_host_same_user_and_source_view_are_required(monkeypatch, guest, host, uid, source):
    a = load()
    monkeypatch.setattr(a, 'os', SimpleNamespace(uname=lambda: SimpleNamespace(nodename=host),
        getuid=lambda: uid, geteuid=lambda: uid))
    monkeypatch.setattr(a, 'socket', SimpleNamespace(gethostname=lambda: host))
    expected = a.HISTORICAL_FIRST.OLD_SHA if guest else a.HISTORICAL_FIRST.NEUTRAL_SHA
    monkeypatch.setattr(a.engine, 'sha', lambda p: expected if source == 'expected' else 'wrong')
    if host == a.LIVE_HOST and uid == a.LIVE_UID and source == 'expected':
        a.live_host_guard(guest=guest)
    else:
        with pytest.raises(ValueError):
            a.live_host_guard(guest=guest)


@pytest.mark.parametrize('action,code', [('audit',0),('audit',1),('audit',-15),('audit',True),
                                          ('verify',1),('verify',143),('verify',-15),('verify',True)])
def test_no_unobserved_nonzero_audit_exit_waiver(fixture, action, code):
    x = fixture
    root = x.a.AUDIT_ROOT if action == 'audit' else x.a.READONLY_ROOT
    change(root/'exit.json', returncode=code)
    repin(x)
    with pytest.raises(ValueError):
        x.run()
    assert not x.f.calls and not x.f.state.exists()


@pytest.mark.parametrize('kind', ['wrong_host', 'wrong_uid', 'missing_certificate', 'wrong_command',
    'wrong_intent', 'false_pid', 'wrong_stdout', 'regraded_readonly', 'readonly_action',
    'naive_timestamp', 'reversed_timestamp', 'readonly_before_audit', 'wrong_review_host',
    'missing_exit_pin', 'missing_stream_pin', 'wrong_exit_review_digest', 'wrong_certificate_digest',
    'intent_identity', 'intent_started', 'wrong_intent_digest', 'process_parent', 'process_child', 'process_time'])
def test_incomplete_or_invented_execution_history_rejected_before_claim(fixture, kind):
    x = fixture
    root = x.a.READONLY_ROOT
    if kind == 'wrong_host': change(root/'exit.json', actual_host='spin.cs.princeton.edu')
    elif kind == 'wrong_uid': change(root/'exit.json', uid=0)
    elif kind == 'missing_certificate': change(root/'exit.json', certificate_present=False)
    elif kind == 'wrong_command': change(root/'exit.json', command=['invented'])
    elif kind == 'wrong_intent': change(root/'intent.json', command=['invented'])
    elif kind == 'false_pid': change(root/'exit.json', pid=True)
    elif kind == 'wrong_stdout': (root/'stdout.txt').write_text('{"status":"invented"}\n')
    elif kind == 'regraded_readonly': change(root/'exit.json', new_grader_invocations=24704)
    elif kind == 'readonly_action': change(root/'exit.json', action='audit')
    elif kind == 'naive_timestamp': change(root/'exit.json', started_at_utc='2026-09-13T04:02:00')
    elif kind == 'reversed_timestamp': change(root/'exit.json', finished_at_utc='2026-09-13T03:00:00+00:00')
    elif kind == 'readonly_before_audit': change(root/'exit.json', started_at_utc='2026-09-13T03:00:00+00:00')
    elif kind == 'wrong_certificate_digest': change(root/'exit.json', certificate_sha256='a'*64)
    elif kind == 'intent_identity': change(root/'intent.json', pid=1)
    elif kind == 'intent_started': change(root/'intent.json', started_at_utc='2026-09-13T03:00:00+00:00')
    elif kind == 'process_parent': change(root/'process.json', pid=1)
    elif kind == 'process_child': change(root/'process.json', child_pid=True)
    elif kind == 'process_time': change(root/'process.json', at_utc='2026-09-13T03:00:00+00:00')
    repin(x)
    if kind == 'wrong_intent_digest':
        change(root/'exit.json', intent_sha256='a'*64)
        value = x.m.read(x.m.REVIEW)
        value['files_sha256'][str(root/'exit.json')] = x.m.sha(root/'exit.json')
        value['audit_readonly_exit_sha256'] = x.m.sha(root/'exit.json')
        write(x.m.REVIEW, value)
    value = x.m.read(x.m.REVIEW)
    if kind == 'wrong_review_host': value['cpu_fit_host'] = 'spin.cs.princeton.edu'
    elif kind == 'missing_exit_pin': value['files_sha256'].pop(str(root/'exit.json'))
    elif kind == 'missing_stream_pin': value['files_sha256'].pop(str(root/'stderr.txt'))
    elif kind == 'wrong_exit_review_digest': value['audit_readonly_exit_sha256'] = 'a'*64
    write(x.m.REVIEW, value)
    with pytest.raises(ValueError):
        x.run()
    assert not x.f.calls and not x.f.state.exists()


def test_missing_final_review_cannot_authorize_action(fixture):
    x = fixture
    x.m.REVIEW.unlink()
    with pytest.raises(FileNotFoundError):
        x.run()
    assert not x.f.calls and not x.f.state.exists()


def test_existing_recipe_refuses_a_second_host_fit(fixture):
    x = fixture
    write(x.m.REVISION_ROOT/'level4/recipes/python_factors.json', {'preserved_prior_attempt': True})
    with pytest.raises(ValueError, match='unfitted'):
        x.run()
    assert not x.f.calls


def test_actual_live_lock_device_still_enforced(fixture):
    x = fixture
    x.f.dispatch['drift'] = {'lock_device': 0}
    with pytest.raises(ValueError, match='owner/view/fence'):
        x.run()
    assert not x.f.calls


def test_static_verification_preserves_recorded_device_without_live_host_comparison(fixture, monkeypatch):
    x = fixture
    result = x.run()
    class ForeignLock:
        def stat(self):
            raise AssertionError('static history cannot inspect current-host device')
    monkeypatch.setattr(x.m.first, 'LOCK', ForeignLock())
    assert x.a.verify_existing(x.f.state) == result
    assert x.f.calls == ['python_factors']


@pytest.mark.parametrize('time', ['2026-09-13T03:59:59+00:00', '2026-09-13T04:01:01+00:00'])
def test_certificate_completion_must_be_within_actual_audit(fixture, time):
    x = fixture
    change(x.x.certificate, audit_completed_at_utc=time)
    value = x.m.read(x.m.REVIEW)
    value['execution_certificate_sha256'] = x.m.sha(x.x.certificate)
    write(x.m.REVIEW, value)
    for root in (x.a.AUDIT_ROOT, x.a.READONLY_ROOT):
        change(root/'exit.json', certificate_sha256=x.m.sha(x.x.certificate))
    repin(x)
    x.a.OBSERVED_AUDIT_PINS = {p: x.m.sha(p) for p in x.a.OBSERVED_AUDIT_PINS}
    with pytest.raises(ValueError, match='completion must fall inside'):
        x.run()
    assert not x.f.calls and not x.f.state.exists()


@pytest.mark.parametrize('kind', ['existing', 'dangling_symlink'])
def test_original_host_attempt_cannot_be_retried_on_new_host(fixture, kind):
    x = fixture
    if kind == 'existing':
        write(x.a.PRIOR_STATE/'fits/python_factors/claim.json', {'preserved_original_claim': True})
    else:
        x.a.PRIOR_STATE.symlink_to(x.a.PRIOR_STATE.parent/'missing_original_attempt')
    with pytest.raises(ValueError, match='original host fit already attempted'):
        x.run()
    assert not x.f.calls and not x.f.state.exists()


@pytest.mark.parametrize('field,value',[('audit_execution_policy','generic_failed_audit'),
    ('audit_outer_returncode',0),('audit_outer_returncode',1),('audit_exit_cause','proot_explained')])
def test_observed_audit_failure_identity_cannot_be_reinterpreted(fixture,field,value):
    x=fixture
    review=x.m.read(x.m.REVIEW);review[field]=value;write(x.m.REVIEW,review)
    with pytest.raises(ValueError,match='exact observed audit143'):
        x.run()
    assert not x.f.calls and not x.f.state.exists()


@pytest.mark.parametrize('name',['intent.json','process.json','exit.json','claim.json','registration.json','certificate'])
def test_exact_observed_audit_history_not_adoptable_after_scratch_review_rehash(fixture,name):
    x=fixture
    if name=='certificate':path=x.x.certificate
    elif name in ('claim.json','registration.json'):path=x.m.RECOVERY/'execution_audit'/name
    else:path=x.a.AUDIT_ROOT/name
    change(path,other_observed_execution=True)
    repin(x)
    if name=='certificate':
        review=x.m.read(x.m.REVIEW);review['execution_certificate_sha256']=x.m.sha(path);write(x.m.REVIEW,review)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls and not x.f.state.exists()
