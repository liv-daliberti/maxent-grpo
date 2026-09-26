"""Prevent hourly retries from losing progress, duplicating writers, or looping."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
spec=importlib.util.spec_from_file_location('hourly_guard',ROOT/'ops/exp_scaling/guard_mathir_hourly_timeouts_20260909.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def checkpoint(tmp_path, step=816, **changes):
    path=tmp_path/f'debug_job101/checkpoints/step_{step:05d}';path.mkdir(parents=True)
    state=dict(global_steps=step,global_step=step,policy_sgd_step=float(step),prompt_batches_consumed_total=step,online_canonical_bank_state={})
    state.update(changes)
    for name,data in [('mp_rank_00_model_states.pt',state),('bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',{'optimizer_state_dict':{'state':{0:{'step':step}}}})]:
        with zipfile.ZipFile(path/name,'w') as archive:archive.writestr('archive/data.pkl',pickle.dumps(data,protocol=2))
    return path


def test_model_optimizer_and_prompt_counters_validate_without_tensor_load(tmp_path):
    result=guard.checked_checkpoint(checkpoint(tmp_path))
    assert result['step']==816
    assert result['model_counters'][0]['policy_sgd_step']==[816.0]
    assert result['optimizer_counters'][0]['step']==[816]


@pytest.mark.parametrize('field',guard.SCALAR_KEYS)
def test_counter_disagreement_rejects_checkpoint(tmp_path,field):
    with pytest.raises(RuntimeError,match='counters disagree'):
        guard.checked_checkpoint(checkpoint(tmp_path,**{field:768}))


def test_nonfinite_counter_rejects_checkpoint(tmp_path):
    with pytest.raises(RuntimeError,match='Invalid scalar counter'):
        guard.checked_checkpoint(checkpoint(tmp_path,policy_sgd_step=float('nan')))


def test_partial_or_wrong_optimizer_checkpoint_rejected(tmp_path):
    path=checkpoint(tmp_path);optim=path/'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
    optim.write_bytes(b'incomplete zip')
    with pytest.raises(RuntimeError,match='Structurally invalid'):
        guard.checked_checkpoint(path)
    with zipfile.ZipFile(optim,'w') as archive:archive.writestr('archive/data.pkl',pickle.dumps({'step':768},protocol=2))
    with pytest.raises(RuntimeError,match='Optimizer counters disagree'):
        guard.checked_checkpoint(path)


@pytest.mark.parametrize('floor',[768,960])
def test_first_retry_requires_progress_beyond_initial_checkpoint(floor):
    item={'initial_checkpoint_step':floor}
    for step in [floor-48,floor]:
        with pytest.raises(RuntimeError,match='No durable progress'):
            guard.require_progress(item,{}, {'step':step})
    guard.require_progress(item,{}, {'step':floor+48})


def test_later_retry_cannot_repeat_saved_progress():
    item={'initial_checkpoint_step':768}
    with pytest.raises(RuntimeError,match='No durable progress'):
        guard.require_progress(item,{'last_resume_step':864},{'step':864})
    guard.require_progress(item,{'last_resume_step':864},{'step':912})


@pytest.fixture
def harness(monkeypatch,tmp_path):
    item={'job_id':101,'identity':{'run_dir':str(tmp_path),'domain':'mathir','arm':'maxrl','seed':72,'run_stamp':'test'},'initial_checkpoint_step':768,'max_requeues':24,'long_route_command':['sbatch','--time=3-00:00:00']}
    record={'JobState':'TIMEOUT','Reason':'TimeLimit','Restarts':'0','RunTime':'01:00:00'}
    calls=[]
    monkeypatch.setattr(guard,'mapping',lambda *args:{101:item})
    monkeypatch.setattr(guard,'show',lambda _:record)
    monkeypatch.setattr(guard,'stable',lambda *_:None)
    monkeypatch.setattr(guard.base,'field',lambda r,k:r[k])
    monkeypatch.setattr(guard.base,'queue',lambda:{})
    monkeypatch.setattr(guard.base,'command',lambda args,**kw:calls.append(args))
    monkeypatch.setattr(guard.recovery,'complete',lambda _:False)
    monkeypatch.setattr(guard.recovery,'state',lambda _:'TIMEOUT')
    monkeypatch.setattr(guard,'checkpoint_and_writer',lambda _:{'step':768})
    monkeypatch.setattr(guard,'save',lambda *_:None)
    monkeypatch.setattr(guard.capacity,'fallback',lambda job,reason:calls.append(['fallback',job,reason]))
    return item,record,calls


@pytest.mark.parametrize('state',['RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'])
def test_live_or_pending_allocation_never_requeued(harness,state):
    item,record,calls=harness;record['JobState']=state
    result=guard.observe_one({'jobs':{}},item,apply=True)
    assert result['status']=='monitoring' and calls==[]


def test_no_progress_timeout_never_issues_scheduler_command(harness):
    item,record,calls=harness
    result=guard.observe_one({'jobs':{}},item,apply=True)
    assert result['status']=='fallback_complete'
    assert calls==[['fallback',101,'no_progress']]


def test_other_failure_and_retry_cap_never_requeue(harness):
    item,record,calls=harness;record['JobState']='FAILED'
    with pytest.raises(RuntimeError,match='manual inspection'):
        guard.observe_one({'jobs':{}},item,apply=True)
    record['JobState']='TIMEOUT'
    tx={'jobs':{'101':{'status':'monitoring','attempts':[{'released':True}]*24}}}
    result=guard.observe_one(tx,item,apply=True)
    assert result['status']=='fallback_complete'
    assert calls==[['fallback',101,'requeue_cap']]


def test_valid_progress_dry_run_still_does_not_mutate(harness,monkeypatch):
    item,record,calls=harness
    monkeypatch.setattr(guard,'checkpoint_and_writer',lambda _:{'step':816})
    result=guard.observe_one({'jobs':{}},item,apply=False)
    assert result['status']=='would_requeue_same_id' and result['checkpoint']['step']==816
    assert calls==[]


@pytest.mark.parametrize('protocol',[2,4,5])
def test_memoized_repeated_optimizer_keys_cannot_hide_mismatches(tmp_path,protocol):
    path=checkpoint(tmp_path)
    archive=path/'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
    data={'state':{0:{'step':816},1:{'step':768}}}
    raw=pickle.dumps(data,protocol=protocol)
    assert guard.metadata_scalars(raw,('step',))[0]['step']==[816,768]
    with zipfile.ZipFile(archive,'w') as handle:handle.writestr('archive/data.pkl',raw)
    with pytest.raises(RuntimeError,match='Optimizer counters disagree'):
        guard.checked_checkpoint(path)


def test_expired_deadline_keeps_running_final_allocation_untouched(harness):
    item,record,calls=harness;record['JobState']='RUNNING'
    result=guard.observe_one({'jobs':{},'deadline_utc':'2020-01-01T00:00:00+00:00'},item,apply=True)
    assert result['status']=='monitoring_final_allocation' and calls==[]


@pytest.mark.parametrize('state',['TIMEOUT','PENDING'])
def test_expired_deadline_hands_inactive_or_pending_job_to_reviewed_fallback(harness,state):
    item,record,calls=harness;record['JobState']=state
    result=guard.observe_one({'jobs':{},'deadline_utc':'2020-01-01T00:00:00+00:00'},item,apply=True)
    assert result['status']=='fallback_complete' and calls==[['fallback',101,'deadline']]


def test_deadline_is_rechecked_after_checkpoint_work_before_requeue(harness,monkeypatch):
    item,record,calls=harness;count=[0]
    monkeypatch.setattr(guard,'checkpoint_and_writer',lambda _:{'step':816})
    def expired(_):
        count[0]+=1
        return count[0]>=2
    monkeypatch.setattr(guard,'retry_deadline_reached',expired)
    result=guard.observe_one({'jobs':{}},item,apply=True)
    assert result['status']=='fallback_complete' and calls==[['fallback',101,'deadline']]


def test_persisted_absolute_deadline_does_not_reset_between_calls():
    from datetime import datetime,timezone
    tx={'deadline_utc':'2026-09-10T12:00:00+00:00'}
    assert not guard.retry_deadline_reached(tx,datetime(2026,9,10,11,59,tzinfo=timezone.utc))
    assert guard.retry_deadline_reached(tx,datetime(2026,9,10,12,0,tzinfo=timezone.utc))
    assert guard.retry_deadline_reached(tx,datetime(2026,9,11,12,0,tzinfo=timezone.utc))
    assert tx['deadline_utc']=='2026-09-10T12:00:00+00:00'


def test_terminal_receipt_retires_only_its_dormant_fallback(harness,monkeypatch):
    item,record,calls=harness
    monkeypatch.setattr(guard.recovery,'complete',lambda _:True)
    monkeypatch.setattr(guard.capacity,'retire_completed',lambda job:calls.append(['retire_completed',job]),raising=False)
    result=guard.observe_one({'jobs':{}},item,apply=True)
    assert result['status']=='completed' and calls==[['retire_completed',101]]


def test_mapping_of_hourly_peer_survives_other_cells_fallback(tmp_path,monkeypatch):
    items=[];rows=[]
    for old,new,arm,floor in [(31158503,101,'maxrl',768),(31158504,102,'replay_maxrl',960)]:
        ident={'domain':'mathir','arm':arm,'seed':72,'run_dir':str(tmp_path/arm),'run_stamp':arm}
        items.append(dict(ident,old_job_id=old,new_job_id=new,checkpoint={'step':floor},long_route_command=['sbatch']))
        rows.append(dict(ident,job_id=old if new==101 else new))
    items[0]['fallback']={'status':'complete'}
    source=tmp_path/'source.json';aggregate=tmp_path/'aggregate.json';deployment=tmp_path/'deployment.json'
    source.write_text(json.dumps({'runs':rows+[{'job_id':1000+i} for i in range(48)]}))
    aggregate.write_text(json.dumps({'runs':[dict(r,scale='qwen3b') for r in rows]+[{'job_id':2000+i} for i in range(148)]}))
    deployment.write_text(json.dumps({'schema':'campaign-mathir-hourly-backfill-v1','items':items}))
    monkeypatch.setattr(guard.base,'LEDGER',source);monkeypatch.setattr(guard.campaign,'E118_LEDGER',aggregate);monkeypatch.setattr(guard,'DEPLOYMENT',deployment)
    assert guard.mapping(101)[101]['fallback_complete']
    assert guard.mapping(102)[102]['identity']['arm']=='replay_maxrl'


def test_only_exact_audited_dormant_hold_is_exempt_from_writer_check(tmp_path,monkeypatch):
    item={'job_id':101,'dormant_old_job_id':31158503,'identity':{'run_dir':str(tmp_path)}}
    deployment=tmp_path/'deployment.json';deployment.write_text(json.dumps({'items':[{'new_job_id':101,'old_job_id':31158503,'old_fallback_held':True}]}))
    monkeypatch.setattr(guard,'DEPLOYMENT',deployment)
    monkeypatch.setattr(guard.recovery,'complete',lambda _:False)
    monkeypatch.setattr(guard.recovery,'select_latest_checkpoint',lambda _:(tmp_path/'step_00816',{}))
    monkeypatch.setattr(guard,'checked_checkpoint',lambda _:{'step':816})
    audited=[];monkeypatch.setattr(guard.capacity,'old_guard',lambda row,held:audited.append((row['old_job_id'],held)))
    monkeypatch.setattr(guard.recovery,'active_writers',lambda:{str(tmp_path):{101,31158503}})
    assert guard.checkpoint_and_writer(item)['step']==816 and audited==[(31158503,True)]
    monkeypatch.setattr(guard.recovery,'active_writers',lambda:{str(tmp_path):{101,31158503,999}})
    with pytest.raises(RuntimeError,match='Unexpected active or pending writer'):
        guard.checkpoint_and_writer(item)


def test_committed_owned_hold_can_finish_once_just_after_deadline(harness,monkeypatch):
    from datetime import datetime,timedelta,timezone
    item,record,calls=harness
    now=datetime.now(timezone.utc)
    tx={'jobs':{},'deadline_utc':(now-timedelta(seconds=1)).isoformat()}
    action={'hold_intent':True,'requested_at_utc':(now-timedelta(seconds=2)).isoformat(),'before_restarts':0,'checkpoint_before':{'step':816}}
    state={'attempts':[action],'status':'holding'}
    record.update(JobState='PENDING',Reason='job_requeued_in_held_state',Restarts='1')
    monkeypatch.setattr(guard,'checkpoint_and_writer',lambda _:{'step':816})
    def command(args,**kw):
        calls.append(args);record['Reason']='Priority'
    monkeypatch.setattr(guard.base,'command',command)
    guard.reconcile_action(tx,item,action,state)
    assert calls==[['scontrol','release','101']]
    assert action['released'] and state['last_resume_step']==816


def test_postdeadline_intent_cannot_use_committed_transition_exception():
    from datetime import datetime,timedelta,timezone
    now=datetime.now(timezone.utc)
    with pytest.raises(RuntimeError,match='not committed before'):
        guard.require_committed_before_deadline({'deadline_utc':(now-timedelta(seconds=2)).isoformat()},
            {'hold_intent':True,'requested_at_utc':(now-timedelta(seconds=1)).isoformat()})


def test_guard_errors_are_durably_reported_without_secondary_keyerror(tmp_path,monkeypatch):
    plan=tmp_path/'plan.json';tx=tmp_path/'tx.json';lock=tmp_path/'ledger.lock'
    row={'job_id':101,'dormant_old_job_id':31158503,'long_route_command':['sbatch']}
    plan.write_text(json.dumps({'schema':'test','controller_sha256':'same','protocol_sha256':'same','helper_sha256':{},'rows':[row]}))
    monkeypatch.setattr(guard,'PLAN',plan);monkeypatch.setattr(guard,'TX',tx);monkeypatch.setattr(guard,'LEDGER_LOCK',lock)
    monkeypatch.setattr(guard.recovery,'digest',lambda _:'same')
    def fail(*args,**kwargs):raise RuntimeError('injected diagnostic failure')
    monkeypatch.setattr(guard,'observe_one',fail)
    guard.run(watch=False,apply=True)
    state=json.loads(tx.read_text())
    assert state['jobs']['101']['status']=='manual_stop'
    assert state['jobs']['101']['dormant_old_job_id']==31158503
    assert 'injected diagnostic failure' in state['jobs']['101']['error']


def test_pending_deadline_race_continues_observing_active_job(harness,monkeypatch):
    item,record,calls=harness;record['JobState']='PENDING'
    def raced(*_):
        record['JobState']='RUNNING'
        raise RuntimeError('hourly allocation started; defer fallback until inactive')
    monkeypatch.setattr(guard.capacity,'fallback',raced)
    tx={'jobs':{},'deadline_utc':'2020-01-01T00:00:00+00:00'}
    result=guard.observe_one(tx,item,apply=True)
    assert result['status']=='monitoring_final_allocation' and calls==[]
    assert tx['jobs']['101']['status']=='monitoring'
