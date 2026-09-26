"""Regression checks for the exact CPU handoff and original watcher gates."""
from contextlib import contextmanager
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import handoff_e124_array_storage_20260911 as a


@contextmanager
def noop(): yield


def test_controller_changes_only_storage_and_outer_lock(monkeypatch):
    events=[]
    @contextmanager
    def shared():
        events.append('shared_enter'); yield; events.append('shared_exit')
    @contextmanager
    def original():
        events.append('original_enter'); yield; events.append('original_exit')
    monkeypatch.setattr(a,'shared_lock',shared)
    monkeypatch.setattr(a.h,'pins',lambda plan:events.append('pins'))
    unchanged=object()
    controller=SimpleNamespace(lock=original,storage_report=object(),watch=unchanged,verify=unchanged)
    result=a.configure_controller({},controller)
    assert result is controller and controller.watch is unchanged and controller.verify is unchanged
    assert controller.storage_report is a.storage.controller_storage_report
    with controller.lock(): events.append('body')
    assert events==['shared_enter','original_enter','pins','body','original_exit','shared_exit']


def test_science_job_state_and_id_are_strict(monkeypatch):
    canonical={'plan_sha256':'proof','cells':[{'cell_id':'science'}]}
    tx={'plan_sha256':'proof','rows':{'science':{'job_id':2,'status':'held'},'systems':{'job_id':1,'status':'released'}}}
    monkeypatch.setattr(a.h,'read',lambda p:canonical if p==a.h.CANONICAL_PLAN else tx)
    calls=[]
    x=SimpleNamespace(systems_row=lambda p:{'cell_id':'systems'}, audit_job=lambda *args,**kwargs:calls.append(kwargs['held']))
    monkeypatch.setattr(a.h,'v2',lambda:x)
    plan={'science_plan_sha256':'proof','gpu_ids':{'science':2,'systems':1}}
    assert a.audit_gpu_rows(plan) is tx and calls==[False,True]
    tx['rows']['science']['status']='released'
    with pytest.raises(RuntimeError,match='transaction advanced'):a.audit_gpu_rows(plan)
    tx['rows']['science']['status']='held';tx['rows']['science']['job_id']=3
    with pytest.raises(RuntimeError,match='identity changed'):a.audit_gpu_rows(plan)


def test_uncertain_cpu_release_is_not_repeated(monkeypatch):
    monkeypatch.setattr(a.h,'load',lambda:({}, {'job_id':99,'release_intent':True}))
    monkeypatch.setattr(a.h,'audit_cpu',lambda *args,**kwargs:{'Priority':'0'})
    monkeypatch.setattr(a.h,'release',lambda:pytest.fail('repeated release'))
    assert a.release()['status']=='uncertain_CPU_release_requires_reconciliation'


@pytest.fixture
def handoff(monkeypatch,tmp_path):
    plan={'sha256':'proof','pins':{str(a.h.V2):'source'}};tx={'job_id':99}
    monkeypatch.setattr(a.h,'GATE',tmp_path/'gate.json')
    monkeypatch.setattr(a.h,'load',lambda:(plan,tx))
    monkeypatch.setattr(a.h,'ready_waiter',lambda *args:None)
    monkeypatch.setattr(a,'shared_lock',noop)
    monkeypatch.setattr(a.h,'original_locks',noop)
    monkeypatch.setattr(a,'old_identity',lambda *args:None)
    monkeypatch.setattr(a,'audit_gpu_rows',lambda *args:None)
    monkeypatch.setattr(a.h,'pins',lambda *args:None)
    monkeypatch.setattr(a.h,'v2',lambda:SimpleNamespace(lock=noop))
    commands=[];events=[]
    monkeypatch.setattr(a.h,'command',lambda cmd:commands.append(cmd))
    monkeypatch.setattr(a.h,'event',lambda tx,label:events.append(label))
    return plan,tx,commands,events


def test_cancel_only_exact_old_cpu_and_never_repeat(handoff,monkeypatch):
    plan,tx,commands,events=handoff
    monkeypatch.setattr(a.h,'inactive_old',lambda:None)
    assert a.handoff()['status']=='waiting_old_CPU_inactive'
    assert commands==[['scancel',str(a.OLD)]] and tx['cancel_intent']
    a.handoff()
    assert commands==[['scancel',str(a.OLD)]]


def test_stale_waiter_cannot_cancel_old_cpu(handoff,monkeypatch):
    _,_,commands,_=handoff
    def stale(*args):raise RuntimeError('stale')
    monkeypatch.setattr(a.h,'ready_waiter',stale)
    with pytest.raises(RuntimeError):a.handoff()
    assert not commands


def test_inactive_old_required_before_promotion(handoff,monkeypatch):
    plan,tx,commands,events=handoff
    monkeypatch.setattr(a.h,'inactive_old',lambda:{'job_id':a.OLD,'state':'CANCELLED'})
    promoted=[]
    monkeypatch.setattr(a.h,'promote_controller',lambda *args:promoted.append(True))
    assert a.handoff()['status']=='gate_open' and promoted==[True]
    assert a.h.read(a.h.GATE)['old_job_id']==a.OLD and not commands


def test_original_promotion_preserves_gpu_rows_and_deadline_retry_state(tmp_path,monkeypatch):
    # Exercise the already audited promotion code, not a duplicate implementation.
    canonical={'plan_sha256':'science','controller':{'job_id':a.OLD,'self_requeues':12},
        'rows':{'systems':{'job_id':1,'status':'released','retries':0}},'status':'systems_queued'}
    source=copy.deepcopy(canonical)
    monkeypatch.setattr(a.h,'CANONICAL_TX',tmp_path/'canonical.json')
    monkeypatch.setattr(a.h,'TX',tmp_path/'local.json')
    a.h.write(a.h.CANONICAL_TX,canonical)
    plan={'science_plan_sha256':'science','command':['exact','cpu']}
    a.h.promote_controller(plan,{'job_id':99})
    after=a.h.read(a.h.CANONICAL_TX)
    assert after['rows']==source['rows'] and after['status']==source['status']
    assert after['controller']['self_requeues']==12 and after['controller']['job_id']==99
    assert after['controller_history']==[source['controller']]
