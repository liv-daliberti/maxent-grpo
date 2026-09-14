#!/usr/bin/env python3
"""Expand nine existing E119 pending routes through one inherited guard handoff."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import time

import amend_e119_max44_node208_20260910 as h

g, b, recovery = h.guard, h.b, h.recovery
ROOT = h.ROOT
ART = ROOT/'var/artifacts/e119_pending208_completion_20260910'
PLAN, TX = ART/'amendment_plan.json', ART/'amendment_transaction.json'
PROTOCOL = ROOT/'paper/preregistration/e119_pending208_completion_20260910.md'
OLD_ART = ROOT/'var/artifacts/e119_max44_node208_20260910/guard'
OLD_PLAN, OLD_TX = OLD_ART/'plan.json',OLD_ART/'transaction.json'
OLD_LOCK = ROOT/'var/artifacts/e119_healthy_completion_guard_20260909/singleton.lock'
OLD_CPU=31170078
NEW=ART/'guard'
NODES='node205,node207,node208,node302'
TARGETS={31163339,31163560,31163584,31163595,31163624,31163665,31163676,31163683,31163710}
PEERS={31158679,31158680,31158681}
require,atomic=g.require,g.atomic

# Reuse reviewed CPU identity, immutable-plan loading and singleton watch logic
# with process-local paths. The original helper and existing guard files stay intact.
for name in ('ART','PLAN','TX','PROTOCOL','OLD_ART','OLD_PLAN','OLD_TX','OLD_LOCK','OLD_CPU','NEW','NODES'):
    setattr(h,name,globals()[name])
h.__file__=str(Path(__file__).resolve())


def health(path):
    proof=json.loads(Path(path).read_text())
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(proof['at_utc'])).total_seconds()
    require(0<=age<=7200 and proof['node']=='node208','node208 physical proof stale or wrong node')
    gpu=proof['gpu']
    require(gpu['name']=='NVIDIA RTX A6000' and gpu['memory_total_MiB']>=48000 and gpu['temperature_C']<80,'node208 physical GPU health differs')
    node=b.command(['scontrol','show','node','-o','node208']).stdout
    require(not any(x in b.field(node,'State') for x in ('DRAIN','DOWN','FAIL')),'node208 unhealthy')
    require('gpu:a6000:' in b.field(node,'Gres') and 'lowprio' in b.field(node,'Partitions').split(','),'node208 resource pool differs')
    return {'physical_proof':str(path),'sha256':recovery.digest(path),'node_record':node}


def target(item, *, amended,held):
    job=item['job_id'];record=g.show(job);expected=copy.deepcopy(item)
    if amended: expected['resources']['ReqNodeList']=NODES
    g.stable(expected,record)
    require(b.field(record,'JobState')=='PENDING' and b.field(record,'RunTime')=='00:00:00' and b.field(record,'Restarts')==item['amendment_restarts'],'target started/restarted; preserve allocation')
    if held:require(b.field(record,'Reason')=='JobHeldUser' and b.field(record,'Priority')=='0','owned target hold missing')
    else:require(b.field(record,'Priority')!='0','preexisting target hold')
    g.dormant(item)
    return record


def checkpoint(item):
    selected,rejected=g.checkpoints.select_latest_checkpoint(Path(item['identity']['run_dir']))
    if selected is None:
        require(item['initial_resume_step']==0 and not rejected,'missing or rejected nonzero checkpoint')
        return {'path':None,'step':0,'rejected':[]}
    detail=g.checkpoints.checkpoint(item['identity'])
    require(detail['step']>=item['initial_resume_step'],'checkpoint regressed below guard floor')
    require(not any(int(Path(p).name[5:])>=detail['step'] for p in detail['rejected']),'newer incomplete checkpoint')
    return detail


def targets_safe(plan, *, amended,held):
    mapping=g.ledger_mapping();writers=recovery.active_writers()
    for item in plan['items']:
        target(item,amended=amended,held=held)
        current=mapping.get(item['job_id'])
        require(current and current['identity']==item['identity'] and current['original_job_id']==item['original_job_id'],'authoritative E119 mapping changed')
        require(writers.get(str(Path(item['identity']['run_dir']).resolve()),set())<={item['job_id'],item['old_job_id']},'unexpected same-cell writer')
        require(checkpoint(item)==plan['checkpoints'][str(item['job_id'])],'pending checkpoint changed')


def cpu_command():
    return ['sbatch','--parsable','--hold','--job-name=e119-pending208-guard','--account=mltheory','--partition=lowprio',
            '--nodelist=node915,node917','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=2G','--gres=none',
            '--time=1-01:10:00','--requeue','--export=NONE',f'--chdir={ROOT}',f'--output={NEW}/supervisor-%j.out',
            f'--error={NEW}/supervisor-%j.err','--comment=e119-pending208-guard-20260910',str(NEW/'supervisor.slurm')]


def prepare(proof):
    require(not PLAN.exists() and not TX.exists(),'immutable preparation already exists')
    old=json.loads(OLD_PLAN.read_text());old_tx=json.loads(OLD_TX.read_text())
    require(old_tx['plan_sha256']==recovery.digest(OLD_PLAN),'old guard binding differs')
    items=[copy.deepcopy(r) for r in old['rows'] if r['job_id'] in TARGETS]
    require(len(items)==9 and len(old['rows'])==10,'guard scope changed')
    for item in items:
        before=g.show(item['job_id']);item['amendment_restarts']=b.field(before,'Restarts');item['amendment_before']=before
    require(all(i['resources']['ReqNodeList']=='node205,node207,node302' for i in items),'unexpected prior pools')
    cpu=g.show(OLD_CPU);registration=json.loads((OLD_ART/'supervisor.json').read_text())
    require(b.field(cpu,'JobState')=='RUNNING' and registration['job_id']==OLD_CPU and registration['plan_sha256']==recovery.digest(OLD_PLAN),'old CPU registration differs')
    require(b.submit_tokens(cpu)==registration['submit_tokens'] and all(b.field(cpu,k)==v for k,v in registration['resources'].items()),'old CPU identity differs')
    ledger=json.loads(g.CONTINUATIONS.read_text());byjob={r['continuation_job_id']:r for r in ledger['continuations']}
    plan={'schema':'e119-pending208-guard-handoff-v1','created_at_utc':b.now(),'items':items,
          'checkpoints':{str(i['job_id']):checkpoint(i) for i in items},
          'ledger_rows_before':{str(i['job_id']):byjob[i['job_id']] for i in items},
          'health_path':str(Path(proof).resolve()),'health_at_preparation':health(proof),
          'old_cpu_job_id':OLD_CPU,'old_cpu_submit_tokens':b.submit_tokens(cpu),'old_cpu_resources':{k:b.field(cpu,k) for k in h.CPU_FIELDS},
          'old_cpu_restarts':b.field(cpu,'Restarts'),'old_plan_sha256':recovery.digest(OLD_PLAN),
          'controller_sha256':recovery.digest(__file__),'protocol_sha256':recovery.digest(PROTOCOL),
          'cpu_submit_tokens':cpu_command(),'scheduler_mutations':False}
    targets_safe(plan,amended=False,held=False);h.final_old_state(plan)
    NEW.mkdir(parents=True,exist_ok=True)
    (NEW/'supervisor.slurm').write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'+f'cd {ROOT}\nexec /usr/bin/python3 -u {Path(__file__).resolve()} watch\n')
    successor=copy.deepcopy(old)
    for r in successor['rows']:
        if r['job_id'] in TARGETS:r['resources']['ReqNodeList']=NODES
    for p in (Path(__file__),PROTOCOL):successor['helper_sha256'][str(p.resolve())]=recovery.digest(p)
    atomic(NEW/'plan.json',successor)
    plan['new_guard_plan_sha256']=recovery.digest(NEW/'plan.json');plan['cpu_script_sha256']=recovery.digest(NEW/'supervisor.slurm')
    atomic(PLAN,plan);atomic(NEW/'cpu_submission.json',{'command':cpu_command(),'script_sha256':plan['cpu_script_sha256'],'scheduler_mutations':False})
    print(json.dumps({'prepared':str(PLAN),'targets':sorted(TARGETS),'scheduler_mutations':False}))


def submit_cpu():
    plan,tx=h.load();require(not tx.get('cpu_submission_intent'),'CPU submission already attempted; reconcile receipt')
    health(plan['health_path']);targets_safe(plan,amended=False,held=False)
    tx['cpu_submission_intent']=True;h.save(tx,'Persisted exact held CPU submission intent')
    response=b.command(plan['cpu_submit_tokens']).stdout.strip()
    require(re.fullmatch(r'\d+(;\S+)?',response),'uncertain CPU submission receipt; do not repeat')
    job=int(response.split(';')[0]);tx.update(new_cpu_job_id=job,cpu_submission_receipt=response);h.save(tx,'Recorded exact successor CPU ID')
    tx['cpu_held_record']=h.new_cpu(plan,job,held=True);h.save(tx,'Audited held CPU identity and resource profile')
    print(json.dumps({'new_cpu_job_id':job,'state':'held'}))


def route():
    plan,tx=h.load();job=tx['new_cpu_job_id']
    require(not tx.get('route_started'),'route already attempted; reconcile durable transaction rather than repeat')
    h.new_cpu(plan,job,held=True);health(plan['health_path']);targets_safe(plan,amended=False,held=False)
    tx['route_started']=True;h.save(tx,'Beginning one nine-target handoff at ledger boundary')
    for item in plan['items']:
        target(item,amended=False,held=False);tx.setdefault('target_hold_intents',[]).append(item['job_id']);h.save(tx,f"Holding exact pending target {item['job_id']}")
        b.command(['scontrol','hold',str(item['job_id'])]);target(item,amended=False,held=True)
    h.old_cpu(plan,running=True);h.final_old_state(plan)
    tx['old_cpu_stop_requested']=True;h.save(tx,'Stopping only exact registered CPU after all target holds')
    b.command(['scancel',str(OLD_CPU)])
    with OLD_LOCK.open('a+') as oldlock:
        until=time.monotonic()+45
        while True:
            if OLD_CPU not in b.queue():
                try:fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:pass
            require(time.monotonic()<until,'old CPU stop not acknowledged; targets retained held');time.sleep(.5)
        previous=h.final_old_state(plan);tx['old_final_tx_sha256']=recovery.digest(OLD_TX)
        atomic(ART/'old_guard_transaction.final.json',previous);h.save(tx,'Captured final retry states with original singleton held')
        targets_safe(plan,amended=False,held=True)
        for item in plan['items']:
            health(plan['health_path']);tx.setdefault('node_update_intents',[]).append(item['job_id']);h.save(tx,f"Adding208 only to target {item['job_id']}")
            b.command(['scontrol','update',f"JobId={item['job_id']}",f'NodeList={NODES}']);target(item,amended=True,held=True)
        data=json.loads(g.CONTINUATIONS.read_text());require(len(data['continuations'])==75,'continuation cardinality changed');atomic(ART/'continuation.before.json',data)
        for item in plan['items']:
            row=next(r for r in data['continuations'] if r['continuation_job_id']==item['job_id'])
            require(row==plan['ledger_rows_before'][str(item['job_id'])],'target row changed')
            row['actual_requested_nodes']=NODES.split(',');row['actual_scheduler_profile']['ReqNodeList']=NODES
            row['pending208_guard_amendment']=str(TX)
        data.setdefault('repair_history',[]).append({'at':b.now(),'audit':str(TX),'job_ids':sorted(TARGETS),'only_scheduler_change':'add node208 to pending ReqNodeList; same jobs and GPU/resource requests'})
        tx['ledger_commit_requested']=True;h.save(tx,'Persisted nine-row placement provenance intent')
        atomic(g.CONTINUATIONS,data);tx['ledger_updated']=True
        copied=copy.deepcopy(previous);copied['plan_sha256']=plan['new_guard_plan_sha256'];atomic(NEW/'transaction.json',copied)
        tx.update(guard_state_copied=True,status='routed_targets_cpu_held');h.save(tx,'Inherited original deadline, retry counts, resume floors and all ten cells')
    print(json.dumps({'status':tx['status'],'new_cpu_job_id':job}))


def activate():
    plan,tx=h.load();job=tx['new_cpu_job_id'];require(tx.get('guard_state_copied') and tx.get('ledger_updated'),'route incomplete')
    require(not tx.get('cpu_release_requested') and OLD_CPU not in b.queue(),'CPU release already attempted or old CPU active')
    h.new_cpu(plan,job,held=True);targets_safe(plan,amended=True,held=True)
    h.configured_guard().register(job)
    tx['cpu_release_requested']=True;h.save(tx,'Registered successor CPU; releasing only CPU watcher')
    b.command(['scontrol','release',str(job)]);tx['cpu_released']=True;h.save(tx,'CPU released; GPU targets remain held pending fresh heartbeat and E122 starts')


def peer_starts():
    records={}
    for job in sorted(PEERS):
        rec=g.show(job)
        if b.field(rec,'JobState')=='RUNNING' and b.field(rec,'NodeList')=='node208':
            require(b.field(rec,'JobName').startswith('e122_level3_countdown_') and b.field(rec,'MinMemoryNode')=='128G','E122 peer identity/resource differs')
            records[str(job)]=rec
    require(len(records)>=2,'two new E122 node208 allocations not yet RUNNING; retain E119 holds')
    return records


def release():
    plan,tx=h.load();require(not tx.get('target_release_requested'),'GPU release already attempted; reconcile receipts')
    # ready() validates all ten monitoring rows and the shared CPU identity.
    h.TARGET=min(TARGETS);h.ready(plan,tx);health(plan['health_path']);targets_safe(plan,amended=True,held=True)
    tx['e122_running_before_release']=peer_starts();tx['target_release_requested']=True;h.save(tx,'Two new E122 allocations running; releasing nine E119 owned holds for later backfill')
    for item in plan['items']:
        target(item,amended=True,held=True);tx.setdefault('target_release_intents',[]).append(item['job_id']);h.save(tx,f"Releasing owned E119 hold {item['job_id']}")
        b.command(['scontrol','release',str(item['job_id'])]);rec=g.show(item['job_id']);amended=copy.deepcopy(item);amended['resources']['ReqNodeList']=NODES;g.stable(amended,rec)
        require(b.field(rec,'JobState') in ('PENDING','RUNNING','CONFIGURING') and b.field(rec,'Priority')!='0','target release differs')
        tx.setdefault('target_after',{})[str(item['job_id'])]=rec;h.save(tx,f"Verified E119 target {item['job_id']} eligible")
    tx.update(targets_released=True,status='released');h.save(tx,'All nine future node208 backfill routes released; running Max44 unchanged')
    print(json.dumps({'status':'released','targets':sorted(TARGETS),'guard_cpu':tx['new_cpu_job_id']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','submit','route','activate','release','watch']);p.add_argument('--health-proof');a=p.parse_args()
    g.install_bounded_scheduler();ART.mkdir(parents=True,exist_ok=True)
    if a.phase=='watch':h.watch();return
    with (ART/'amendment.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with g.LEDGER_LOCK.open('a+') as ledger:
            fcntl.flock(ledger,fcntl.LOCK_EX)
            if a.phase=='prepare':require(a.health_proof,'physical health proof required');prepare(a.health_proof)
            elif a.phase=='submit':submit_cpu()
            else:globals()[a.phase]()


if __name__=='__main__':main()
