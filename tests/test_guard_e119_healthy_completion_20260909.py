from __future__ import annotations
import copy
from datetime import datetime, timedelta, timezone
import importlib
from pathlib import Path
import sys
import subprocess
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
x=importlib.import_module('guard_e119_healthy_completion_20260909')
REAL_CHECKPOINT_AND_WRITER = x.checkpoint_and_writer


@pytest.fixture
def context(monkeypatch):
    item={'job_id':999,'old_job_id':111,'identity':{'run_dir':'/run','run_stamp':'stamp'},
          'initial_resume_step':192,'max_requeues':3}
    tx={'jobs':{},'events':[],'deadline_utc':(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()}
    record='JobId=999 JobState=TIMEOUT Reason=TimeLimit Restarts=0 Priority=10 RunTime=1-12:00:00'
    monkeypatch.setattr(x,'mapping',lambda:{999:{'identity':item['identity']}})
    monkeypatch.setattr(x,'stable',lambda *args:None)
    monkeypatch.setattr(x,'show',lambda job:record)
    monkeypatch.setattr(x.recovery,'complete',lambda path:False)
    monkeypatch.setattr(x.base,'queue',lambda:{})
    monkeypatch.setattr(x.recovery,'state',lambda job:'TIMEOUT')
    monkeypatch.setattr(x,'checkpoint_and_writer',lambda item:{'path':'/run/step_00288','step':288})
    monkeypatch.setattr(x,'save',lambda *args:None)
    return tx,item


def test_initial_checkpoint_must_advance(context,monkeypatch):
    tx,item=context
    monkeypatch.setattr(x,'checkpoint_and_writer',lambda item:{'step':192})
    with pytest.raises(RuntimeError,match='No newer valid checkpoint'):
        x.observe_one(tx,item,apply=False)


def test_read_only_timeout_qualified(context):
    tx,item=context
    result=x.observe_one(tx,item,apply=False)
    assert result['status']=='would_requeue_same_id' and result['checkpoint']['step']==288
    assert tx['jobs']=={}


@pytest.mark.parametrize('state',['FAILED','OUT_OF_MEMORY','PREEMPTED','NODE_FAIL','CANCELLED'])
def test_only_timeout_is_retried(context,monkeypatch,state):
    tx,item=context
    monkeypatch.setattr(x,'show',lambda job:f'JobState={state} Reason=None Restarts=0')
    monkeypatch.setattr(x.base,'command',lambda *a,**k:pytest.fail('no mutation'))
    with pytest.raises(RuntimeError,match='manual inspection'):
        x.observe_one(tx,item,apply=True)


def test_running_allocation_is_preserved(context,monkeypatch):
    tx,item=context
    monkeypatch.setattr(x,'show',lambda job:'JobState=RUNNING Reason=None Restarts=0 RunTime=00:05:00')
    monkeypatch.setattr(x.base,'command',lambda *a,**k:pytest.fail('no mutation'))
    assert x.observe_one(tx,item,apply=True)['state']=='RUNNING'


def test_three_retry_cap(context):
    tx,item=context
    tx['jobs']['999']={'status':'monitoring','last_resume_step':192,'attempts':[{'released':True}]*3}
    with pytest.raises(RuntimeError,match='allowance exhausted'):
        x.observe_one(tx,item,apply=True)


def test_uncertain_requeue_never_repeated(context,monkeypatch):
    tx,item=context;calls=[]
    def fail(command):
        assert tx['jobs']['999']['attempts'][0]['hold_intent']
        calls.append(command);raise subprocess.TimeoutExpired(command,10)
    monkeypatch.setattr(x.base,'command',fail)
    with pytest.raises(subprocess.TimeoutExpired):x.observe_one(tx,item,apply=True)
    with pytest.raises(RuntimeError,match='Owned hold is not pending'):
        x.observe_one(tx,item,apply=True)
    assert calls==[['scontrol','requeuehold','999']]


def test_expired_deadline_never_requeues(context,monkeypatch):
    tx,item=context;tx['deadline_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(x.base,'command',lambda *a,**k:pytest.fail('deadline forbids mutation'))
    with pytest.raises(RuntimeError,match='deadline reached'):
        x.observe_one(tx,item,apply=True)


def test_expired_deadline_retains_owned_retry_hold(context,monkeypatch):
    tx,item=context;tx['deadline_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    action={'before_restarts':0,'checkpoint_before':{'step':288},'released':False}
    state={'status':'holding','last_resume_step':192,'attempts':[action]}
    monkeypatch.setattr(x,'show',lambda job:'JobState=PENDING Reason=job_requeued_in_held_state Restarts=1 Priority=0')
    monkeypatch.setattr(x.base,'command',lambda *a,**k:pytest.fail('deadline forbids release'))
    with pytest.raises(RuntimeError,match='deadline reached'):
        x.reconcile_action(tx,item,action,state)


def test_new_partial_checkpoint_blocks(context,monkeypatch):
    tx,item=context
    monkeypatch.setattr(x.checkpoints,'checkpoint',lambda identity:{'step':288,'rejected':{'/run/step_00384':['partial']}})
    with pytest.raises(RuntimeError,match='newer incomplete'):
        REAL_CHECKPOINT_AND_WRITER(item)


def test_dormant_predecessor_is_only_permitted_extra_writer(monkeypatch):
    item={'job_id':999,'old_job_id':111,'identity':{'run_dir':'/run'}}
    monkeypatch.setattr(x,'dormant',lambda item:None)
    monkeypatch.setattr(x.recovery,'active_writers',lambda:{'/run':{999,111}})
    x.writer_check(item)
    monkeypatch.setattr(x.recovery,'active_writers',lambda:{'/run':{999,111,222}})
    with pytest.raises(RuntimeError,match='unexpected same-cell writer'):x.writer_check(item)


def test_no_mutable_display_helper_hash_dependency():
    text=Path(x.__file__).read_text()
    assert 'import campaign_stats' not in text
    assert 'campaign_stats.py' not in text
    assert 'timedelta(days=7)' in text
    assert "'max_requeues': 3" in text


def test_successful_lost_release_ack_reconciles_without_new_mutation(context,monkeypatch):
    tx,item=context
    action={'before_restarts':0,'checkpoint_before':{'step':288},'checkpoint_before_release':{'step':288},
            'release_requested':True,'released':False}
    state={'status':'holding','last_resume_step':192,'attempts':[action]}
    monkeypatch.setattr(x,'show',lambda job:'JobState=RUNNING Reason=None Restarts=1 Priority=10')
    monkeypatch.setattr(x,'writer_check',lambda item:None)
    monkeypatch.setattr(x.base,'command',lambda *a,**k:pytest.fail('must not repeat release'))
    x.reconcile_action(tx,item,action,state)
    assert action['released'] and state['last_resume_step']==288


def test_admin_hold_is_never_claimed_as_requeue_owned(context):
    tx,item=context
    with pytest.raises(RuntimeError,match='Expected requeue-owned hold'):
        x.own_hold(item,'JobState=PENDING Reason=JobHeldAdmin Priority=0 Restarts=1',{'before_restarts':0})


def test_wrong_restart_increment_is_never_released(context):
    tx,item=context
    with pytest.raises(RuntimeError,match='exactly one requeue restart'):
        x.own_hold(item,'JobState=PENDING Reason=job_requeued_in_held_state Priority=0 Restarts=2',{'before_restarts':0})


@pytest.mark.parametrize(('field','wrong'),[('MinMemoryNode','96G'),('ReqNodeList','node208'),
    ('TresPerNode','gres/gpu:a5000:1'),('TimeLimit','01:00:00'),('UserId','other(1)')])
def test_frozen_actual_resource_profile_rejects_drift(tmp_path,monkeypatch,field,wrong):
    import shlex
    script=tmp_path/'train.slurm';script.write_text('#!/bin/bash\n')
    tokens=['sbatch','--parsable',str(script)]
    fields={k:'(null)' for k in x.PRESERVE}
    fields.update(UserId='od2961(363432)',MinMemoryNode='116G',ReqNodeList='node205,node207,node302',
                  TresPerNode='gres/gpu:1',TimeLimit='1-12:00:00',NumNodes='1')
    item={'job_id':999,'submit_tokens':tokens,'launcher_sha256':x.recovery.digest(script),'resources':copy.deepcopy(fields)}
    fields[field]=wrong
    record='JobId=999 '+' '.join(f'{k}={v}' for k,v in fields.items())+' SubmitLine='+shlex.join(tokens)+' WorkDir=/repo'
    with pytest.raises(RuntimeError,match='Guarded resource changed'):
        x.stable(item,record)


def test_scheduler_rpc_wrapper_is_bounded(monkeypatch):
    seen={}
    monkeypatch.setattr(x.subprocess,'run',lambda parts,**kwargs:seen.update(parts=parts,**kwargs))
    x.bounded_command(['scontrol','show','job','999'])
    assert seen['timeout']==120 and seen['check'] is True


def test_transient_errors_stop_after_three_consecutive_observations(context):
    tx,item=context
    error=subprocess.TimeoutExpired(['scontrol','show','job','999'],120)
    assert x.observation_error(tx,item,error)['status']=='transient_query_error'
    assert x.observation_error(tx,item,error)['status']=='transient_query_error'
    assert x.observation_error(tx,item,error)['status']=='manual_stop'
    assert tx['jobs']['999']['consecutive_query_errors']==3


def test_validation_error_stops_immediately(context):
    tx,item=context
    assert x.observation_error(tx,item,RuntimeError('resource changed'))['status']=='manual_stop'


def test_timeout_after_applied_requeue_retains_intent_and_never_repeats(context,monkeypatch):
    tx,item=context;commands=[];shown={'record':'JobState=TIMEOUT Reason=TimeLimit Restarts=0 Priority=10'}
    monkeypatch.setattr(x,'show',lambda job:shown['record'])
    def command(parts,**kwargs):
        commands.append(parts)
        if parts[1]=='requeuehold':
            shown['record']='JobState=PENDING Reason=job_requeued_in_held_state Restarts=1 Priority=0'
            raise subprocess.TimeoutExpired(parts,120)
        assert parts[1]=='release'
        shown['record']='JobState=PENDING Reason=Priority Restarts=1 Priority=10'
        return subprocess.CompletedProcess(parts,0,'','')
    monkeypatch.setattr(x.base,'command',command)
    monkeypatch.setattr(x,'writer_check',lambda item:None)
    try:x.observe_one(tx,item,apply=True)
    except subprocess.TimeoutExpired as error:
        assert x.observation_error(tx,item,error)['status']=='transient_query_error'
    else:pytest.fail('simulated lost acknowledgement did not occur')
    assert tx['jobs']['999']['attempts'][0]['hold_intent']
    x.observe_one(tx,item,apply=True)
    assert commands==[['scontrol','requeuehold','999'],['scontrol','release','999']]
    assert tx['jobs']['999']['attempts'][0]['released']


@pytest.fixture
def cpu_registration(tmp_path,monkeypatch):
    import json,os
    plan=tmp_path/'plan.json';plan.write_text('{"rows": []}')
    script=tmp_path/'supervisor.slurm';script.write_text('#!/bin/bash\n')
    tokens=['sbatch','--hold',str(script)]
    proposal={'command':tokens,'script_sha256':x.recovery.digest(script)}
    (tmp_path/'cpu_submission.json').write_text(json.dumps(proposal))
    monkeypatch.setattr(x,'ART',tmp_path);monkeypatch.setattr(x,'PLAN',plan)
    monkeypatch.setattr(x,'REGISTRATION',tmp_path/'supervisor.json')
    fields={'JobState':'PENDING','Reason':'JobHeldUser','Priority':'0','UserId':f'user({os.getuid()})',
            'ReqTRES':'cpu=1,mem=2G,node=1','Account':'mltheory','Partition':'lowprio',
            'ReqNodeList':'node915,node917','MinMemoryNode':'2G','TimeLimit':'1-01:10:00',
            'Requeue':'1','JobName':'e119-healthy-timeout-guard','Command':str(script),'Comment':'test'}
    monkeypatch.setattr(x.base,'submit_tokens',lambda record:tokens)
    monkeypatch.setattr(x,'show',lambda job:' '.join(f'{k}={v}' for k,v in fields.items()))
    return fields,script


def test_cpu_registration_binds_held_identity_without_scheduler_mutations(cpu_registration,monkeypatch):
    import json
    monkeypatch.setattr(x.base,'command',lambda *a,**k:pytest.fail('registration is scheduler-read-only'))
    x.register(123)
    proof=json.loads(x.REGISTRATION.read_text())
    assert proof['job_id']==123 and proof['plan_sha256']==x.recovery.digest(x.PLAN)
    assert proof['resources']['MinMemoryNode']=='2G'


@pytest.mark.parametrize(('field','value'),[('JobState','RUNNING'),('Reason','JobHeldAdmin'),
    ('UserId','other(-1)'),('ReqTRES','cpu=1,mem=2G,node=1,gres/gpu=1')])
def test_cpu_registration_rejects_unowned_or_non_cpu_allocation(cpu_registration,field,value):
    fields,script=cpu_registration;fields[field]=value
    with pytest.raises(RuntimeError):x.register(123)
    assert not x.REGISTRATION.exists()


def test_absent_historical_comment_is_preserved_explicitly():
    assert x.predecessor_resource('JobId=111 MinMemoryNode=96G', 'Comment') is None
    assert x.predecessor_resource('JobId=111 Comment=reviewed', 'Comment') == 'reviewed'
    with pytest.raises(RuntimeError, match='MinMemoryNode'):
        x.predecessor_resource('JobId=111', 'MinMemoryNode')


def test_predecessor_comment_addition_is_drift(monkeypatch):
    record='JobState=PENDING Reason=JobHeldUser Priority=0 Restarts=0 Comment=changed'
    item={'old_job_id':111,'old_submit_tokens':['sbatch'],'old_restarts':'0','old_resources':{'Comment':None}}
    monkeypatch.setattr(x,'show',lambda job:record)
    monkeypatch.setattr(x.base,'submit_tokens',lambda value:['sbatch'])
    with pytest.raises(RuntimeError,match='predecessor resource changed: Comment'):
        x.dormant(item)
