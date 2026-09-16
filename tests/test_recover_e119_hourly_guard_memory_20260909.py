import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

OPS = Path(__file__).resolve().parents[1] / 'ops/exp_scaling'
sys.path.insert(0, str(OPS))
spec = importlib.util.spec_from_file_location('cpu_memory_recovery', OPS / 'recover_e119_hourly_guard_memory_20260909.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def record(state='RUNNING', memory='512M', restarts=1, reason='None', priority='100'):
    return f'JobState={state} MinMemoryNode={memory} Restarts={restarts} Reason={reason} Priority={priority}'


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'PLAN', tmp_path/'plan.json'); monkeypatch.setattr(m, 'TX', tmp_path/'tx.json')
    plan={'cpu_restarts':1}; m.write(m.PLAN, plan)
    monkeypatch.setattr(m, 'audit_science', lambda p: None)
    monkeypatch.setattr(m, 'cpu_profile', lambda p,r, memory=None: m.require(memory is None or m.field(r,'MinMemoryNode')==memory, 'memory differs'))
    records=[record()]; calls=[]
    monkeypatch.setattr(m, 'show', lambda j, **kwargs: records[0])
    def command(args):
        calls.append(args)
        if args[1]=='requeuehold': records[0]=record('PENDING',restarts=2,reason='job_requeued_in_held_state',priority='0')
        elif args[1]=='update': records[0]=record('PENDING','4G',2,'job_requeued_in_held_state','0')
        elif args[1]=='release': records[0]=record('PENDING','4G',2,'Priority','100')
    monkeypatch.setattr(m, 'command', command)
    return plan, records, calls


def tx(**kwargs):
    value={'plan_sha256':m.digest(m.PLAN),'events':[],**kwargs};m.write(m.TX,value)


def test_only_cpu_id_and_memory_are_mutated(recovery):
    _,_,calls=recovery;m.apply()
    assert calls==[['scontrol','requeuehold','31159699'], ['scontrol','update','JobId=31159699','MinMemoryNode=4096'], ['scontrol','release','31159699']]
    assert m.read(m.TX)['released']
    m.apply();assert len(calls)==3


def test_requeue_timeout_never_repeats_mutation(recovery,monkeypatch):
    _,records,calls=recovery
    def timeout(args):
        calls.append(args)
        raise subprocess.TimeoutExpired(args,45)
    monkeypatch.setattr(m,'command',timeout)
    with pytest.raises(subprocess.TimeoutExpired):m.apply()
    assert m.read(m.TX)['requeue_intent']
    with pytest.raises(RuntimeError,match='owned CPU requeue hold'):m.apply()
    assert len(calls)==1


def test_uncertain_requeue_can_adopt_exact_owned_hold(recovery):
    _,records,calls=recovery;tx(requeue_intent=True)
    records[0]=record('PENDING',restarts=2,reason='job_requeued_in_held_state',priority='0');m.apply()
    assert all(x[1]!='requeuehold' for x in calls)
    assert m.read(m.TX)['released']


@pytest.mark.parametrize('reason,restarts',[('JobHeldUser',2),('JobHeldAdmin',2),('job_requeued_in_held_state',3)])
def test_unowned_hold_or_extra_restart_never_released(recovery,reason,restarts):
    _,records,calls=recovery;tx(requeue_intent=True)
    records[0]=record('PENDING',restarts=restarts,reason=reason,priority='0')
    with pytest.raises(RuntimeError):m.apply()
    assert not calls


def test_release_lost_ack_reconciled_without_repeating(recovery):
    _,records,calls=recovery;tx(requeue_intent=True,memory_intent=True,release_intent=True)
    records[0]=record('RUNNING','4G',2);m.apply()
    assert not calls and m.read(m.TX)['released']


def test_uncertain_memory_update_unchanged_stops(recovery):
    _,records,calls=recovery;tx(requeue_intent=True,memory_intent=True)
    records[0]=record('PENDING',restarts=2,reason='job_requeued_in_held_state',priority='0')
    with pytest.raises(RuntimeError,match='Unacknowledged memory'):m.apply()
    assert not calls


def test_science_audit_failure_precedes_all_mutations(recovery,monkeypatch):
    _,_,calls=recovery
    def fail(plan):raise RuntimeError('Science retry counters or absolute deadlines changed')
    monkeypatch.setattr(m,'audit_science',fail)
    with pytest.raises(RuntimeError,match='Science retry counters'):m.apply()
    assert not calls and not m.TX.exists()


def test_science_snapshot_preserves_absolute_bounds(tmp_path,monkeypatch):
    path=tmp_path/'science.json';monkeypatch.setattr(m.guard,'TX',path)
    data={key:'original' for key in m.SCIENCE_STATE};data.update(status='watching',attempts=[{'released':True,'checkpoint':{'step':1056}}])
    m.write(path,data);assert m.science_snapshot()=={key:data[key] for key in m.SCIENCE_STATE}
    data['attempts'][0]['released']=False;m.write(path,data)
    with pytest.raises(RuntimeError,match='Unfinished scientific mutation'):m.science_snapshot()


def test_completing_requeue_waits_for_exact_hold(recovery, monkeypatch):
    plan, records, calls = recovery
    monkeypatch.setattr(m.time, 'sleep', lambda seconds: None)
    records[0] = record('PENDING',restarts=2,reason='job_requeued_in_held_state',priority='0')
    result = m.await_owned_hold(plan, record('COMPLETING',restarts=2))
    assert result == records[0] and not calls
