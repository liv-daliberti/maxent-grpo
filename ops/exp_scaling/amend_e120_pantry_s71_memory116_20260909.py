#!/usr/bin/env python3
"""Prepare or apply exactly one pending-only E120 host-memory amendment."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import hashlib
import json
from pathlib import Path
import campaign_stats as campaign
import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/e120_pantry_s71_memory116_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e120_pantry_s71_memory116_20260909.md'
JOB = 31033711
LEDGERS = (campaign.E120_LEDGER, campaign.E120_CONTINUATIONS)
PRESERVE = ('UserId','Account','Partition','ReqNodeList','ExcNodeList','NumCPUs','NumTasks','CPUs/Task','TresPerNode','Requeue','Nice','QOS','Dependency','Features','TimeLimit','Command','WorkDir','StdOut','StdErr')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    ledger=json.loads(campaign.E120_LEDGER.read_text())
    rows=ledger['runs'];assert len(rows)==45
    row=next(r for r in rows if int(r['job_id'])==JOB)
    assert (row['model_key'],row['domain'],int(row['seed']))==('qwen3b','pantry_plan',71)
    assert campaign.e120_continuation_jobs(campaign.E120_LEDGER).get(JOB,JOB)==JOB
    record=base.show(JOB);tokens=base.submit_tokens(record);env=base.exports(tokens)
    assert base.field(record,'UserId').startswith(getpass.getuser()+'(')
    assert env['SAVE_PATH']==row['run_dir'] and env['RUN_STAMP']==row['run_stamp']
    expected={'Account':'mltheory','Partition':'mltheory','ReqNodeList':'node302','NumCPUs':'16','NumTasks':'1','TresPerNode':'gres/gpu:a100:1','TimeLimit':'3-00:00:00','QOS':'none','Nice':'100','Requeue':'1','Dependency':'(null)'}
    assert all(base.field(record,k)==v for k,v in expected.items())
    assert base.field(record,'NumNodes') in {'1','1-1'}
    assert env['OAT_ZERO_AUTO_RESUME']=='1' and env['OAT_ZERO_VLLM_GPU_RATIO']=='0.25'
    assert env['OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE']=='4' and env['OAT_ZERO_MAX_MODEL_LEN']=='704'
    assert 'pvl' not in record.lower()
    assert not recovery.complete(Path(row['run_dir']))
    recovery.no_other_writer(dict(old_job_id=JOB,new_job_id=JOB,identity=row),recovery.active_writers())
    return record,row,env


def eligible(record):
    return (base.field(record,'JobState')=='PENDING' and int(base.field(record,'Priority'))>0
            and 'hold' not in base.field(record,'Reason').lower()
            and 'held' not in base.field(record,'Reason').lower())


def stable(before,after,memory):
    assert base.field(after,'MinMemoryNode')==memory
    assert base.submit_tokens(after)==base.submit_tokens(before)
    for k in PRESERVE:assert base.field(after,k)==base.field(before,k),k
    assert base.field(after,'NumNodes') in {'1','1-1'}
    assert dict(x.split('=',1) for x in base.field(after,'ReqTRES').split(','))['gres/gpu']=='1'


def own_hold(record):
    assert base.field(record,'JobState')=='PENDING'
    assert base.field(record,'Reason')=='JobHeldUser' and base.field(record,'Priority')=='0'


def fingerprint(env,record):
    paths=[Path(env['OAT_ZERO_OPS_SNAPSHOT_ROOT'])/'run_experiment.sh',Path(env['OAT_ZERO_OPS_SNAPSHOT_ROOT'])/'slurm/train_node302.slurm',Path(base.field(record,'Command'))]
    paths+=sorted(Path(env['OAT_ZERO_SOURCE_ROOT']).rglob('*.py'))
    return {str(p):sha(p) for p in paths}


def prepare():
    assert not PLAN.exists() and not TX.exists(),'Inspect an existing plan instead of overwriting it'
    before,row,env=identity();assert eligible(before)
    assert base.field(before,'MinMemoryNode')=='128G'
    diagnosis=json.loads((ART/'diagnosis.json').read_text())
    assert diagnosis['job_id']==JOB and diagnosis['scheduler_mutations'] is False
    assert {p['job_id'] for p in diagnosis['peers']}=={31033713,31033714}
    for peer in diagnosis['peers']:
        assert len(peer['samples'])==2
        events=[]
        for sample in peer['samples']:
            assert sample['noncache_gib']<90
            assert int(sample['memory.high'])==128*2**30
            assert sample['memory.stat']['inactive_file']>20*2**30
            assert sample['memory.stat']['file_dirty']+sample['memory.stat']['file_writeback']<2**30
            e=dict(x.split() for x in sample['memory.events'].splitlines());events.append(e)
            assert int(e['oom'])==int(e['oom_kill'])==0
        assert events[0]['high']==events[1]['high'],'New pressure events during qualification require review'
    wrapper=Path(base.field(before,'Command')).read_text()
    assert 'MinMemoryNode' not in wrapper and '--mem' not in wrapper
    plan={'schema':'e120-pantry-s71-memory116-pending-only-v1','created_at_utc':base.now(),'job_id':JOB,'before':before,'row':row,
          'authorization':'User requested accelerated completion; parent requested preparation only pending final review.',
          'only_scheduler_change':{'MinMemoryNode':{'before':'128G','after':'116G','scontrol_value':'118784'}},
          'controller_sha256':sha(__file__),'protocol_sha256':sha(PROTOCOL),'diagnosis_sha256':sha(ART/'diagnosis.json'),
          'helper_sha256':{str(Path(m.__file__).resolve()):sha(m.__file__) for m in (campaign,base,recovery)},
          'ledger_sha256':{str(p):sha(p) for p in LEDGERS},'runtime_sha256':fingerprint(env,before),
          'checkpoint':str(recovery.select_latest_checkpoint(Path(row['run_dir']))[0]),
          'rollback_updates':0,'same_id':True,'pending_only':True,'scientific_and_runtime_exports_changed':False,
          'startup_validation_required':True,'scheduler_mutations':False}
    base.atomic(PLAN,plan)
    print(json.dumps({'prepared':True,'job_id':JOB,'memory_before':'128G','memory_after':'116G','scheduler_mutations':False,'plan':str(PLAN)}))


def apply():
    plan=json.loads(PLAN.read_text());assert plan['job_id']==JOB
    assert not TX.exists(),'Never repeat an existing transaction without reconciliation'
    for path,key in [(__file__,'controller_sha256'),(PROTOCOL,'protocol_sha256'),(ART/'diagnosis.json','diagnosis_sha256')]:assert sha(path)==plan[key]
    for group in ['helper_sha256','ledger_sha256','runtime_sha256']:
        assert all(sha(p)==v for p,v in plan[group].items()),group
    before,row,env=identity()
    if not eligible(before):
        base.atomic(TX,{'job_id':JOB,'status':'skipped_not_pending_unheld','before':before,'scheduler_mutations':False});return
    stable(plan['before'],before,'128G')
    tx={'job_id':JOB,'status':'holding','before':before,'plan_sha256':sha(PLAN),'created_at_utc':base.now(),'hold_requested':True,'resized':False,'released':False}
    base.atomic(TX,tx)
    base.command(['scontrol','hold',str(JOB)])
    held=base.show(JOB)
    if base.field(held,'JobState')!='PENDING':
        assert base.field(held,'JobState') in {'RUNNING','CONFIGURING','COMPLETING'}
        assert base.field(held,'Priority')=='0'
        stable(before,held,'128G')
        base.command(['scontrol','release',str(JOB)])
        tx.update(status='skipped_started_during_hold',after=base.show(JOB));base.atomic(TX,tx);return
    own_hold(held);stable(before,held,'128G');identity()
    tx['held_before_update']=held;tx['resize_requested']=True;base.atomic(TX,tx)
    base.command(['scontrol','update',f'JobId={JOB}','MinMemoryNode=118784'])
    resized,_,_=identity();own_hold(resized);stable(before,resized,'116G')
    assert all(sha(p)==v for p,v in plan['ledger_sha256'].items())
    tx.update(status='resized_owned_hold',resized=True,held_after_update=resized,release_requested=True);base.atomic(TX,tx)
    base.command(['scontrol','release',str(JOB)])
    after=base.show(JOB);stable(before,after,'116G')
    assert base.field(after,'JobState') in {'PENDING','RUNNING','CONFIGURING'} and int(base.field(after,'Priority'))>0
    assert all(sha(p)==v for p,v in plan['ledger_sha256'].items())
    tx.update(status='complete',released=True,after=after,completed_at_utc=base.now());base.atomic(TX,tx)
    print(json.dumps({'applied':True,'job_id':JOB,'memory':'116G','state':base.field(after,'JobState'),'no_scientific_changes':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','apply']);args=parser.parse_args()
    ART.mkdir(exist_ok=True)
    with (ROOT/'var/artifacts/e120_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        globals()[args.phase]()
