"""Scratch process recorder tests; subprocesses/audits/graders never run."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r4_audit_recorder_test',ROOT/'artifacts/run_modebench_scale_level4_python_r5_audit_soak_v2_20260913.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return path

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'development';root.mkdir()
    monkeypatch.setattr(a,'DEVELOPMENT',root)
    for name in ('SOURCE','TESTS','AUDITOR','AUDITOR_TESTS','REVIEW','CONTRACT'):
        monkeypatch.setattr(a,name,put(tmp_path/(name+'.json'),{'synthetic':name}))
    monkeypatch.setattr(a,'PINS',{})
    terminal=put(root/'terminal_accounting.json',{'actual_fixture_terminal':True})
    pins={str(getattr(a,k)):a.digest(getattr(a,k)) for k in ('SOURCE','TESTS','AUDITOR','AUDITOR_TESTS','CONTRACT')}
    pins[str(terminal)]=a.digest(terminal)
    put(a.REVIEW,{'schema':'scratch_review','status':'reviewed','terminal_sha256':a.digest(terminal),'files_sha256':pins})
    guards=[];calls=[];dispatch={'code':0,'certificate':True,'mutate':False}
    def guard(source,action,*,guest):
        assert source==a.SOURCE and guest is False
        assert not (root/('actual_audit_soak_v2_20260913' if action=='audit' else 'actual_audit_soak_readonly_v2_20260913')).exists()
        guards.append(action)
    monkeypatch.setattr(a.contract,'outer_command_guard',guard)
    helper=SimpleNamespace(reviewed=lambda digest,guest:dict(pins))
    monkeypatch.setattr(a,'load',lambda *args:helper)
    class Process:
        pid=9876
        def __init__(self,command,**kwargs):
            calls.append(command)
            assert kwargs['env']['OPENBLAS_NUM_THREADS']==kwargs['env']['OMP_NUM_THREADS']=='1'
            if dispatch['certificate']:put(root/'execution_reconciliation.json',{'status':'synthetic_complete'})
            kwargs['stdout'].write(b'{"schema":"synthetic","status":"complete"}\n');kwargs['stdout'].flush()
        def wait(self):
            if dispatch['mutate']:a.SOURCE.write_text('changed after child')
            return dispatch['code']
    monkeypatch.setattr(a.subprocess,'Popen',Process)
    return SimpleNamespace(root=root,pins=pins,guards=guards,calls=calls,dispatch=dispatch,
        run=lambda action='audit':a.run(action,a.digest(a.REVIEW)))

def test_literal_actual_zero_and_full_command_are_recorded_once(fixture):
    f=fixture;assert f.run()==0
    root=f.root/'actual_audit_soak_v2_20260913';intent=a.read(root/'intent.json');exit=a.read(root/'exit.json')
    assert f.guards==['audit'] and len(f.calls)==1 and exit['returncode']==0
    assert intent['command']==exit['command']==f.calls[0]
    assert exit['intent_sha256']==a.digest(root/'intent.json') and exit['process_sha256']==a.digest(root/'process.json')
    assert exit['certificate_sha256']==a.digest(f.root/'execution_reconciliation.json')
    assert intent['grader_invocations_field_is_planned_until_certificate'] is True
    with pytest.raises((ValueError,AssertionError)):f.run()
    assert len(f.calls)==1

@pytest.mark.parametrize('code',[143,1,-15])
def test_actual_nonzero_exit_and_existing_certificate_are_never_reinterpreted(fixture,code):
    f=fixture;f.dispatch['code']=code
    assert f.run()==(code if code>=0 else 128-code)
    record=a.read(f.root/'actual_audit_soak_v2_20260913/exit.json')
    assert record['returncode']==code and record['certificate_present'] is True
    assert len(f.calls)==1

def test_partial_failure_without_certificate_is_preserved(fixture):
    f=fixture;f.dispatch.update(code=1,certificate=False);assert f.run()==1
    record=a.read(f.root/'actual_audit_soak_v2_20260913/exit.json')
    assert record['certificate_present'] is False and record['certificate_sha256'] is None

def test_separate_readonly_action_records_zero_new_graders(fixture):
    f=fixture;put(f.root/'execution_reconciliation.json',{'synthetic_complete':True})
    assert f.run('verify')==0
    record=a.read(f.root/'actual_audit_soak_readonly_v2_20260913/exit.json')
    assert record['new_grader_invocations']==0 and record['action']=='verify'
    assert '--terminal-sha256' not in f.calls[0]

def test_readonly_requires_existing_certificate_without_process(fixture):
    with pytest.raises(ValueError):fixture.run('verify')
    assert not fixture.calls

def test_relative_outer_rejection_precedes_state_creation(fixture,monkeypatch):
    def reject(*args,**kwargs):raise ValueError('full absolute command required')
    monkeypatch.setattr(a.contract,'outer_command_guard',reject)
    with pytest.raises(ValueError,match='absolute'):fixture.run()
    assert not fixture.calls and not (fixture.root/'actual_audit_soak_v2_20260913').exists()

def test_existing_audit_claim_blocks_all_replay(fixture):
    """A real audit record must still block replay outright under v2."""
    d=fixture.root/'execution_audit';d.mkdir()
    (d/'action.lock').write_bytes(b'');(d/'claim.json').write_text('{}')
    with pytest.raises(ValueError,match='never replay any claim|only an empty lock'):fixture.run()
    assert not fixture.calls

def test_an_audit_directory_of_any_other_shape_is_refused(fixture):
    """v2 reconciles exactly one shape: a lone empty action.lock. A bare directory,
    or one holding anything else, is not that shape and must not proceed."""
    (fixture.root/'execution_audit').mkdir()
    with pytest.raises(ValueError,match='only an empty lock'):fixture.run()
    assert not fixture.calls

def test_the_reconcilable_empty_lock_does_not_block_the_retry(fixture):
    d=fixture.root/'execution_audit';d.mkdir();(d/'action.lock').write_bytes(b'')
    fixture.run()
    assert fixture.calls

def test_postexit_input_drift_is_saved_and_blocks_clean_outcome(fixture):
    f=fixture;f.dispatch['mutate']=True
    with pytest.raises(ValueError,match='source changed'):f.run()
    exited=a.read(f.root/'actual_audit_soak_v2_20260913/exit.json')
    assert exited['returncode']==0 and exited['postcheck_error'] is not None



# --- Reconciliation of the refused first attempt ------------------------------
# The first audit refused inside inspect() because the activation review was short
# 404 batch pins. It took no scientific action, but the native engine had already
# created execution_audit/ with an EMPTY action.lock. v2 reconciles exactly that
# and nothing broader.

import importlib.util as _ilu
_V2 = ROOT/'artifacts/run_modebench_scale_level4_python_r5_audit_soak_v2_20260913.py'


def _mod():
    spec = _ilu.spec_from_file_location('_v2_recorder_under_test', _V2)
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_absent_audit_directory_is_the_normal_fresh_case(tmp_path):
    assert _mod().reconcile_audit_directory(tmp_path/'execution_audit') == 'absent'


def test_exactly_the_empty_lock_is_reconciled(tmp_path):
    d = tmp_path/'execution_audit'; d.mkdir(); (d/'action.lock').write_bytes(b'')
    assert _mod().reconcile_audit_directory(d) == 'reconciled_empty_lock'


def test_a_nonempty_lock_is_refused(tmp_path):
    d = tmp_path/'execution_audit'; d.mkdir(); (d/'action.lock').write_text('x')
    with pytest.raises(ValueError, match='exactly the empty file'):
        _mod().reconcile_audit_directory(d)


@pytest.mark.parametrize('name', ['claim.json', 'registration.json', 'failure.json'])
def test_any_real_audit_record_is_refused(tmp_path, name):
    d = tmp_path/'execution_audit'; d.mkdir(); (d/'action.lock').write_bytes(b'')
    (d/name).write_text('{}')
    with pytest.raises(ValueError, match='only an empty lock'):
        _mod().reconcile_audit_directory(d)


def test_a_tiers_directory_is_refused(tmp_path):
    d = tmp_path/'execution_audit'; d.mkdir(); (d/'action.lock').write_bytes(b''); (d/'tiers').mkdir()
    with pytest.raises(ValueError, match='only an empty lock'):
        _mod().reconcile_audit_directory(d)


def test_the_real_refused_directory_on_disk_is_reconcilable_today():
    """Binds to the actual state the refusal left, not a synthetic copy."""
    real = ROOT/'var/artifacts/modebench_scale_level4_python_r5_20260913/development/execution_audit'
    assert _mod().reconcile_audit_directory(real) == 'reconciled_empty_lock'
    assert not (real/'claim.json').exists() and not (real/'tiers').exists()
    assert not (real.parent/'execution_reconciliation.json').exists()
