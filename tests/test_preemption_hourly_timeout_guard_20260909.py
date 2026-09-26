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
spec=importlib.util.spec_from_file_location('preemption_hourly_guard',ROOT/'ops/exp_scaling/guard_preemption_hourly_timeouts_20260909.py')
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


def test_e118_and_e120_mappings_remain_independent_after_one_fallback(tmp_path, monkeypatch):
    items = []
    for new, (old, expected) in enumerate(guard.EXPECTED.items(), start=101):
        cohort, domain, arm, seed, floor, _ = expected
        items.append(dict(cohort=cohort, domain=domain, arm=arm, seed=seed,
                          run_dir=str(tmp_path/str(old)), run_stamp=str(old),
                          old_job_id=old, new_job_id=new, checkpoint={'step':floor},
                          long_route_command=['sbatch']))
    items[0]['fallback'] = {'status':'complete'}
    deployment = tmp_path/'deployment.json'
    deployment.write_text(json.dumps({'schema':'campaign-preemption-hourly-backfill-v1','items':items}))
    monkeypatch.setattr(guard, 'DEPLOYMENT', deployment)
    audited = []
    monkeypatch.setattr(guard.capacity, 'current_identity',
                        lambda item, effective: audited.append((item['cohort'], effective)))
    result = guard.mapping()
    assert audited == [('e118',31158505),('e118',102),('e120',103)]
    assert result[101]['fallback_complete']
    assert result[102]['identity']['arm'] == 'replay_maxrl'
    assert result[103]['initial_checkpoint_step'] == 576
    assert result[103]['memory'] == '128G'
    assert result[102]['memory'] == '116G'
    assert guard.mapping(103)[103]['cohort'] == 'e120'


def test_unregistered_cell_cannot_enter_hourly_mapping(tmp_path, monkeypatch):
    items = []
    for new, (old, expected) in enumerate(guard.EXPECTED.items(), start=101):
        cohort, domain, arm, seed, floor, _ = expected
        items.append(dict(cohort=cohort, domain=domain, arm=arm, seed=seed,
                          run_dir=str(tmp_path/str(old)), run_stamp=str(old),
                          old_job_id=old, new_job_id=new, checkpoint={'step':floor},
                          long_route_command=['sbatch']))
    items[-1]['seed'] = 75
    deployment = tmp_path/'deployment.json'
    deployment.write_text(json.dumps({'schema':'campaign-preemption-hourly-backfill-v1','items':items}))
    monkeypatch.setattr(guard, 'DEPLOYMENT', deployment)
    monkeypatch.setattr(guard.capacity, 'current_identity', lambda *_:None)
    with pytest.raises(RuntimeError, match='Unexpected scientific cell'):
        guard.mapping()


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


def transition_tx():
    from datetime import datetime,timedelta,timezone
    now=datetime.now(timezone.utc)
    return {'jobs':{},'deadline_utc':(now+timedelta(hours=1)).isoformat(),
            'cleanup_deadline_utc':(now+timedelta(hours=2)).isoformat()}


@pytest.mark.parametrize('lost',['requeuehold','release'])
def test_lost_retry_ack_is_reconciled_without_repeating_scheduler_mutation(harness,monkeypatch,lost):
    item,record,calls=harness;item['dormant_old_job_id']=98
    tx=transition_tx()
    monkeypatch.setattr(guard,'checkpoint_and_writer',lambda _:{'step':816})
    def command(argv,**kwargs):
        calls.append(argv)
        if argv[1]=='requeuehold':
            record.update(JobState='PENDING',Reason='job_requeued_in_held_state',Restarts='1')
        elif argv[1]=='release':record['Reason']='Priority'
        if argv[1]==lost:raise RuntimeError('lost '+lost+' acknowledgement')
    monkeypatch.setattr(guard.base,'command',command)
    with pytest.raises(RuntimeError,match='lost') as error:
        guard.observe_one(tx,item,apply=True)
    result=guard.handle_transition_error(tx,item,error.value)
    assert result['status']=='reconciliation_pending'
    assert len(tx['jobs']['101']['attempts'])==1
    guard.observe_one(tx,item,apply=True)
    assert tx['jobs']['101']['attempts'][0]['released']
    assert tx['jobs']['101']['last_resume_step']==816
    assert [c[1] for c in calls]==['requeuehold','release']


@pytest.mark.parametrize('kind',['retry','fallback','retirement'])
def test_only_persisted_transitions_get_bounded_error_reconciliation(harness,kind):
    item,record,calls=harness;item['dormant_old_job_id']=98
    tx=transition_tx();state={'status':'holding','attempts':[]}
    if kind=='retry':state['attempts']=[{'released':False}]
    if kind=='fallback':state['status']='fallback_in_progress'
    if kind=='retirement':state['retirement_in_progress']=True
    tx['jobs']['101']=state;deadline=tx['deadline_utc']
    for _ in range(7):
        assert guard.handle_transition_error(tx,item,RuntimeError('lost ack'))['status']=='reconciliation_pending'
    assert guard.handle_transition_error(tx,item,RuntimeError('lost ack'))['status']=='manual_stop'
    assert tx['deadline_utc']==deadline and calls==[]


def test_reconciliation_never_extends_cleanup_deadline(harness):
    item,record,calls=harness;item['dormant_old_job_id']=98
    tx=transition_tx();tx['cleanup_deadline_utc']='2020-01-01T00:00:00+00:00'
    tx['jobs']['101']={'status':'holding','attempts':[{'released':False}]}
    assert guard.handle_transition_error(tx,item,RuntimeError('lost ack'))['status']=='manual_stop'
    assert calls==[]


def late_hold_fixture(harness,monkeypatch):
    from datetime import datetime,timedelta,timezone
    item,record,calls=harness;item['dormant_old_job_id']=98
    now=datetime.now(timezone.utc)
    tx={'jobs':{},'deadline_utc':(now-timedelta(minutes=6)).isoformat(),
        'cleanup_deadline_utc':(now+timedelta(minutes=59)).isoformat()}
    action={'hold_intent':True,'requested_at_utc':(now-timedelta(minutes=7)).isoformat(),
            'before_restarts':0,'checkpoint_before':{'step':816},'released':False}
    state={'status':'holding','attempts':[action]};tx['jobs']['101']=state
    record.update(JobState='PENDING',Reason='job_requeued_in_held_state',Restarts='1',Priority='0')
    deployed={'new_job_id':101,'old_job_id':98};deployment={'items':[deployed]}
    monkeypatch.setattr(guard,'checkpoint_and_writer',lambda _:{'step':816})
    monkeypatch.setattr(guard.capacity,'load_transaction',lambda:deployment)
    monkeypatch.setattr(guard.capacity,'current_identity',lambda *_:None)
    monkeypatch.setattr(guard.capacity,'old_guard',lambda *args,**kwargs:None)
    monkeypatch.setattr(guard.capacity,'event',lambda *args:None)
    return item,record,calls,tx,action,state,deployed


def test_late_owned_hold_falls_back_without_another_requeue(harness,monkeypatch):
    item,record,calls,tx,action,state,deployed=late_hold_fixture(harness,monkeypatch)
    def command(argv,**kw):calls.append(argv);record['Reason']='JobHeldUser'
    monkeypatch.setattr(guard.base,'command',command)
    guard.reconcile_action(tx,item,action,state)
    assert calls==[['scontrol','hold','101'],['fallback',101,'deadline']]
    assert deployed['fallback']['deadline_hold_requested']
    assert deployed['fallback']['guard_deadline_owned_hold']['expected_restarts']==1
    assert state['status']=='fallback_complete'


def test_lost_hold_normalization_ack_retains_ownership_and_reconciles(harness,monkeypatch):
    item,record,calls,tx,action,state,deployed=late_hold_fixture(harness,monkeypatch)
    def command(argv,**kw):
        calls.append(argv);record['Reason']='JobHeldUser'
        raise RuntimeError('lost normalization ack')
    monkeypatch.setattr(guard.base,'command',command)
    with pytest.raises(RuntimeError,match='normalization') as error:
        guard.reconcile_action(tx,item,action,state)
    assert action['deadline_handoff_requested'] and state['status']=='deadline_hold_handoff'
    assert guard.handle_transition_error(tx,item,error.value)['status']=='reconciliation_pending'
    guard.observe_one(tx,item,apply=True)
    assert calls==[['scontrol','hold','101'],['fallback',101,'deadline']]
    assert state['status']=='fallback_complete'


@pytest.mark.parametrize('reason',['JobHeldUser','JobHeldAdmin'])
def test_late_handoff_cannot_claim_unrelated_hold(harness,monkeypatch,reason):
    item,record,calls,tx,action,state,deployed=late_hold_fixture(harness,monkeypatch)
    record['Reason']=reason
    with pytest.raises(RuntimeError,match='unrelated'):
        guard.deadline_hold_handoff(tx,item,action,state)
    assert calls==[] and 'fallback' not in deployed


def test_late_handoff_rejects_wrong_restart_or_no_progress(harness,monkeypatch):
    item,record,calls,tx,action,state,deployed=late_hold_fixture(harness,monkeypatch)
    record['Restarts']='2'
    with pytest.raises(RuntimeError,match='restart increment'):
        guard.deadline_hold_handoff(tx,item,action,state)
    record['Restarts']='1'
    monkeypatch.setattr(guard,'checkpoint_and_writer',lambda _:{'step':768})
    with pytest.raises(RuntimeError,match='No durable progress'):
        guard.deadline_hold_handoff(tx,item,action,state)
    assert calls==[]
