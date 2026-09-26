import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
import resume_e122_shared_release_20260911 as m


def status(next_id='31158682',reason=None):
    return {'next_job_id':next_id,'blocked_reason':reason,'issues':[],
            'needs_operator_review_job_ids':[]}


def test_shared_denial_prevents_release_candidate(monkeypatch):
    seen=[]
    monkeypatch.setattr(m,'ORIGINAL_STATUS',lambda *a:status())
    storage=SimpleNamespace(storage_report=lambda **kw:seen.append(kw) or {'allowed':False})
    value=m.gated_status({},Path('/unused'),storage)
    assert value['next_job_id'] is None and value['blocked_reason']=='waiting_shared_storage'
    assert seen==[{'include_held_job_ids':['31158682']}]


def test_shared_approval_preserves_original_candidate(monkeypatch):
    monkeypatch.setattr(m,'ORIGINAL_STATUS',lambda *a:status())
    value=m.gated_status({},Path('/unused'),SimpleNamespace(storage_report=lambda **k:{'allowed':True}))
    assert value['next_job_id']=='31158682' and value['blocked_reason'] is None


@pytest.mark.parametrize('reason',['unknown_scheduler_state','ambiguous_or_external_activity','concurrency_cap','needs_operator_review'])
def test_original_no_release_gates_are_never_relaxed(monkeypatch,reason):
    monkeypatch.setattr(m,'ORIGINAL_STATUS',lambda *a:status(None,reason))
    storage=SimpleNamespace(storage_report=lambda **k:pytest.fail('No candidate needs no admission'))
    assert m.gated_status({},Path('/unused'),storage)==status(None,reason)


def test_unknown_reobservation_never_retries_ambiguous_intent():
    value=status(None,'unknown_scheduler_state')
    assert m.retryable_observation(value)
    value['issues']=['ambiguous_release:31158682']
    assert not m.retryable_observation(value)
    value['issues']=[];value['needs_operator_review_job_ids']=['31158682']
    assert not m.retryable_observation(value)


def test_original_controller_called_only_inside_all_admission_locks(monkeypatch):
    inside=[];events=[];original=m.c.status
    @contextmanager
    def locks():
        inside.append(True);events.append('lock')
        try:yield
        finally:inside.pop();events.append('unlock')
    def advance(*a):
        assert inside and m.c.status is not original
        events.append('advance');raise RuntimeError('fixture')
    monkeypatch.setattr(m,'admission_locks',locks)
    monkeypatch.setattr(m.c,'advance_once',advance)
    with pytest.raises(RuntimeError,match='fixture'):m.observe_once(None,None)
    assert events==['lock','advance','unlock'] and m.c.status is original


def test_busy_admission_lock_never_enters_original_controller(monkeypatch):
    @contextmanager
    def busy():
        raise BlockingIOError('peer admission')
        yield
    monkeypatch.setattr(m,'admission_locks',busy)
    monkeypatch.setattr(m.c,'advance_once',lambda *a:pytest.fail('Must preserve busy peer'))
    with pytest.raises(BlockingIOError):m.observe_once(None,None)


def test_initial_refill_is_three_calls_at_most_and_requires_held_cpu(monkeypatch,tmp_path):
    plan={'binding':{'fixture':True},'storage_module':'types'}
    monkeypatch.setattr(m,'ART',tmp_path)
    monkeypatch.setattr(m,'verify_plan',lambda:plan)
    monkeypatch.setattr(m,'read',lambda p:{'job_id':'123','plan_sha256':'fixture'})
    monkeypatch.setattr(m.c,'digest',lambda p:'fixture')
    held=[];calls=[]
    monkeypatch.setattr(m,'cpu_record',lambda jid,held=None:held_checks(jid,held))
    def held_checks(jid,flag):
        assert jid=='123' and flag is True
        held.append(True)
    monkeypatch.setattr(m,'observe_once',lambda *a:calls.append(1) or status())
    results=m.initial_refill(None,{'binding':plan['binding']})
    assert len(results)==len(calls)==3 and len(held)==4
    assert (tmp_path/'initial_refill.json').is_file()


def test_initial_refill_stops_after_a_shared_storage_denial(monkeypatch,tmp_path):
    plan={'binding':{'fixture':True},'storage_module':'types'}
    monkeypatch.setattr(m,'ART',tmp_path)
    monkeypatch.setattr(m,'verify_plan',lambda:plan)
    monkeypatch.setattr(m,'read',lambda p:{'job_id':'123','plan_sha256':'fixture'})
    monkeypatch.setattr(m.c,'digest',lambda p:'fixture')
    monkeypatch.setattr(m,'cpu_record',lambda *a,**k:None)
    calls=[]
    monkeypatch.setattr(m,'observe_once',lambda *a:calls.append(1) or status(None,'waiting_shared_storage'))
    results=m.initial_refill(None,{'binding':plan['binding']})
    assert len(results)==len(calls)==1
