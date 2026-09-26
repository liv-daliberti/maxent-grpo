#!/usr/bin/env python3
"""Held protected-partition successor for one existing E120 Graph cell."""
from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import re

import amend_e120_graph_s74_owner105_20260910 as route

b, parent = route.b, route.parent
ROOT = route.ROOT
ART = ROOT/'var/artifacts/e120_graph_s74_protected105_20260910'
PLAN, TX = ART/'plan.json', ART/'transaction.json'
PROTOCOL=ROOT/'paper/preregistration/e120_graph_s74_protected105_20260910.md'
OLD=route.JOB
OLD_FIELDS=('UserId', *route.widening.PRESERVE, 'Restarts', 'Features', 'ReqNodeList')
IDENTITY=('domain','model_key','seed','run_dir','run_stamp')
require=route.require


def make_command(tokens):
    changes={'--partition':'mltheory','--nodelist':'node105','--comment':'e120-graph-s74-protected105-20260910-old31158507'}
    command=[x for x in tokens[:-1] if x!='--hold' and x.split('=',1)[0] not in changes]
    command += [f'{k}={v}' for k,v in changes.items()] + ['--hold',tokens[-1]]
    require(b.exports(command)==b.exports(tokens),'science exports differ')
    return command


def old_record(plan, held):
    rec=b.show(OLD)
    require(b.submit_tokens(rec)==plan['old_tokens'],'old submission changed')
    require(all(b.field(rec,k)==v for k,v in plan['old_resources'].items()),'old resources or restart changed')
    require(b.field(rec,'JobState')=='PENDING','old job started; preserve allocation')
    require(b.field(rec,'NumNodes') in ('1','1-1'),'old node count changed')
    if held:
        require(b.field(rec,'Reason')=='JobHeldUser' and b.field(rec,'Priority')=='0','exact old owned hold missing')
    else:
        require(b.field(rec,'Priority')!='0','old job has preexisting hold')
    return rec


def new_record(plan,job,held):
    rec=b.show(job)
    require(b.submit_tokens(rec)==plan['command'],'new submission differs')
    expected={k:v for k,v in plan['old_resources'].items() if k not in ('Partition','ReqTRES','Restarts','Comment','ReqNodeList','StdOut','StdErr')}
    expected.update(Partition='mltheory',ReqNodeList='node105',Comment='e120-graph-s74-protected105-20260910-old31158507')
    require(all(b.field(rec,k)==v for k,v in expected.items()),'new actual resources differ')
    require(b.field(rec,'NumNodes') in ('1','1-1'),'new node count changed')
    require(b.field(rec,'StdOut')==str(ROOT/f'slurm-{job}.out') and b.field(rec,'StdErr')=='','new log routing differs')
    if held:
        require(b.field(rec,'JobState')=='PENDING' and b.field(rec,'Reason')=='JobHeldUser' and b.field(rec,'Priority')=='0' and b.field(rec,'Restarts')=='0','new job is not never-started held allocation')
    return rec


def immutable(plan):
    for path,key in [(Path(__file__),'controller_sha256'),(PROTOCOL,'protocol_sha256')]:
        require(parent.digest(path)==plan[key],'prepared implementation changed')
    require(parent.digest(parent.campaign.E120_LEDGER)==plan['primary_sha256'],'scientific ledger changed')
    require(parent.runtime_fingerprints([{'original_command':plan['old_tokens']}])==plan['runtime_fingerprints'],'runtime snapshot changed')
    require(parent.checkpoint(plan['item']['run_dir'])==plan['item']['checkpoint'],'checkpoint changed')
    require(not parent.recovery.complete(Path(plan['item']['run_dir'])),'cell complete')


def no_competitor(plan, allowed):
    writers=parent.recovery.active_writers().get(str(Path(plan['item']['run_dir']).resolve()),set())
    require(writers <= set(allowed),f'unexpected writer: {writers-set(allowed)}')


def event(tx,message):
    tx['events'].append({'at_utc':b.now(),'event':message});b.atomic(TX,tx)


def prepare():
    require(not PLAN.exists() and not TX.exists(),'existing immutable preparation')
    source=json.loads(route.PLAN.read_text());done=json.loads(route.TX.read_text())
    require(done['status']=='complete','node105 route amendment incomplete')
    rec=b.show(OLD);tokens=b.submit_tokens(rec);require(route.node_set(rec)=={'node105'},'old route differs')
    require(b.field(rec,'Partition')=='lowprio','old partition differs')
    ledger=json.loads(parent.CONTINUATIONS.read_text());row=next(r for r in ledger['continuations'] if r['continuation_job_id']==OLD)
    require(parent.campaign.e120_continuation_jobs(parent.campaign.E120_LEDGER).get(31033709)==OLD,'old mapping differs')
    plan={'schema':'e120-graph-s74-protected105-20260910-v1','created_at_utc':b.now(),
          'old_tokens':tokens,'old_resources':{k:b.field(rec,k) for k in OLD_FIELDS},'old_before':rec,
          'command':make_command(tokens),'row_before':row,'item':source['item'],
          'primary_sha256':parent.digest(parent.campaign.E120_LEDGER),'continuation_before_sha256':parent.digest(parent.CONTINUATIONS),
          'runtime_fingerprints':parent.runtime_fingerprints([{'original_command':tokens}]),
          'controller_sha256':parent.digest(__file__),'protocol_sha256':parent.digest(PROTOCOL),
          'node_record':route.node_health(),'scientific_exports_changed':False,'same_scientific_cells':True}
    old_record(plan,False);immutable(plan);no_competitor(plan,[OLD]);require(not parent.incoming(OLD),'old job acquired dependency')
    test=b.command([plan['command'][0],'--test-only',*[x for x in plan['command'][1:] if x!='--hold']])
    plan['test_only']={'returncode':test.returncode,'stdout':test.stdout,'stderr':test.stderr}
    b.atomic(PLAN,plan);print(json.dumps({'prepared':str(PLAN),'checkpoint':plan['item']['checkpoint']['step'],'scheduler_mutations':False}))


def staged_row(plan,job):
    row=copy.deepcopy(plan['row_before']);row['continuation_job_id']=job
    row.setdefault('intermediate_job_ids',[]).append(OLD)
    row['dormant_fallback_job_id']=OLD
    row['fallback_resource_class']='Exact owned-held node105 lowprio predecessor; activate only after successor inactive and audited mapping restoration'
    row['repair_kind']='protected_owner105_partition_20260910'
    row['protected_owner105_amendment']=str(PROTOCOL);row['protected_owner105_audit']=str(TX)
    row.setdefault('scheduler_placement_amendments',[]).append(str(PROTOCOL))
    row['new_placement']=dict(row['new_placement'],partition='mltheory',node='node105')
    return row


def stage():
    plan=json.loads(PLAN.read_text());immutable(plan)
    require(not TX.exists(),'existing transaction requires reconciliation; never resubmit blindly')
    tx={'schema':plan['schema'],'plan_sha256':parent.digest(PLAN),'status':'staging','events':[]}
    old_record(plan,False);no_competitor(plan,[OLD]);route.node_health()
    require(parent.digest(parent.CONTINUATIONS)==plan['continuation_before_sha256'],'continuation ledger changed')
    tx['old_hold_intent']=True;event(tx,'Persisted hold intent for pending old node105 job')
    b.command(['scontrol','hold',str(OLD)])
    tx['old_held']=old_record(plan,True);immutable(plan);no_competitor(plan,[OLD])
    tx['submission_uncertain']=True;event(tx,'Submitting exactly one held protected node105 successor')
    response=b.command(plan['command']).stdout.strip();require(re.fullmatch(r'\d+(;\S+)?',response),'ambiguous successor submission receipt')
    job=int(response.split(';')[0]);tx['new_job_id']=job;tx['submission_receipt']=response;tx['submission_uncertain']=False
    event(tx,'Recorded exact protected successor ID before further actions')
    tx['new_held']=new_record(plan,job,True);old_record(plan,True);immutable(plan);no_competitor(plan,[OLD,job])
    require(parent.digest(parent.CONTINUATIONS)==plan['continuation_before_sha256'],'continuation ledger changed before promotion')
    ledger=json.loads(parent.CONTINUATIONS.read_text());rows=[r for r in ledger['continuations'] if r['continuation_job_id']==OLD]
    require(len(rows)==1 and rows[0]==plan['row_before'],'old continuation row changed')
    before=copy.deepcopy(ledger);rows[0].clear();rows[0].update(staged_row(plan,job))
    ledger.setdefault('placement_amendments',[]).append(str(PROTOCOL))
    b.atomic(ART/'continuation.before.json',before);b.atomic(ART/'continuation.staged.json',ledger)
    tx['ledger_commit_intent']=True;event(tx,'Promoting exact held successor in one continuation row; old remains dormant held')
    b.atomic(parent.CONTINUATIONS,ledger)
    require(parent.campaign.e120_continuation_jobs(parent.campaign.E120_LEDGER).get(31033709)==job,'new mapping invalid')
    tx['continuation_after_sha256']=parent.digest(parent.CONTINUATIONS);tx['status']='staged_held';event(tx,'Protected successor registered and held for coordinated node105 capacity')
    print(json.dumps({'status':tx['status'],'new_job_id':job,'old_job_id':OLD,'checkpoint':plan['item']['checkpoint']['step']}))


def release():
    plan=json.loads(PLAN.read_text());tx=json.loads(TX.read_text());require(tx['status']=='staged_held','successor not staged or release already attempted')
    require(tx['plan_sha256']==parent.digest(PLAN),'plan changed');immutable(plan)
    job=tx['new_job_id'];old_record(plan,True);new_record(plan,job,True);no_competitor(plan,[OLD,job])
    require(parent.digest(parent.CONTINUATIONS)==tx['continuation_after_sha256'],'continuation ledger changed')
    require(parent.campaign.e120_continuation_jobs(parent.campaign.E120_LEDGER).get(31033709)==job,'successor mapping changed')
    node=route.node_health()
    require(int(b.field(node,'RealMemory'))-int(b.field(node,'AllocMem'))>=128*1024,'insufficient currently unallocated host RAM; retain held successor')
    require(int(b.field(node,'CPUEfctv'))-int(b.field(node,'CPUAlloc'))>=16,'insufficient currently unallocated CPUs')
    allocated=dict(t.split('=',1) for t in b.field(node,'AllocTRES').split(','))
    require(int(allocated.get('gres/gpu',0))<10,'no unallocated A5000')
    tx['capacity_before_release']=node;tx['release_intent']=True;tx['status']='releasing';event(tx,'Releasing protected job after coordinated free-capacity audit')
    b.command(['scontrol','release',str(job)]);rec=new_record(plan,job,False)
    require(b.field(rec,'JobState') in ('PENDING','CONFIGURING','RUNNING') and b.field(rec,'Priority')!='0','new release did not become eligible')
    old_record(plan,True);tx['final']=rec;tx['status']='released';event(tx,'Protected successor eligible; old exact fallback remains held')
    print(json.dumps({'status':'released','job_id':job,'state':b.field(rec,'JobState')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','stage','release']);args=parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for name in ('e118_ledger_promotion.lock','e120_ledger_promotion.lock'):
            f=stack.enter_context((ROOT/'var/artifacts'/name).open('a+'));fcntl.flock(f,fcntl.LOCK_EX)
        globals()[args.phase]()
