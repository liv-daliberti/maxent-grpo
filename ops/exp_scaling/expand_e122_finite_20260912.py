#!/usr/bin/env python3
"""Finite E122 expansion on admitted data and existing healthy GPU classes.

Preserves the persistent cap4 supervisor and all scientific/source/identity
checks. New releases use its migrated, exactly-once journal under shared locks.
Only pending jobs gain node208, already qualified for E122 on September10.
"""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import migrate_level3_neutral_v5 as migration
import e122_nonpython_burst_20260911 as burst
import modebench_current_data as current_data
c,old=migration.c,migration.old
ART=ROOT/'var/artifacts/e122_finite_expansion_20260912'
PLAN=ART/'plan.json'
MIGRATION_SHA='3b5c13cdab5a4d7cd0ca71437736b1f8189358e945b7103a4791470c0b47b903'
SOURCE=Path(__file__).resolve()
GIB=1024**3
POOL={'node205','node206','node207','node208','node302'}
CANDIDATES=['31254498','31254499','31254500','31254501','31158700','31158701']
ARRAY=('31254520',7,ROOT/'var/artifacts/modebench_scale_composite_v1/revision_development/frozen_transport/worker.slurm',ROOT/'var/artifacts/modebench_scale_composite_v1/revision_development/plan.json')
read,digest,new,require,run=migration.read,migration.digest,migration.new,migration.require,migration.run
INSTALLED=False
PRESERVE=('UserId','JobName','Account','Partition','QOS','NumCPUs','NumTasks','CPUs/Task','MinMemoryNode','TimeLimit','ExcNodeList','Requeue','Nice','Features','Dependency','WorkDir','Command','TresPerNode','ReqTRES')


def node_capacity():
    records=json.loads(run(['scontrol','show','nodes','--json']))['nodes']
    rows=[]
    for n in records:
        if n['name'] not in POOL:continue
        healthy=not set(n['state'])&{'DOWN','DRAIN','FAIL','MAINT','RESERVED','PLANNED'}
        allowed=healthy and 'lowprio' in n['partitions'] and any(s in n['gres'] for s in ['gpu:a6000:','gpu:a100:'])
        total=sum(int(x) for x in re.findall(r'gpu:[^:,(]+:(\d+)',n['gres']))
        used=sum(int(x) for x in re.findall(r'gpu:[^:,(]+:(\d+)',n['gres_used']))
        free_mem=max(0,n['real_memory']-n['alloc_memory'])
        free_cpu=max(0,n['cpus']-n['alloc_cpus'])
        slots=min(max(0,total-used),free_mem//65536,free_cpu//8) if allowed else 0
        rows.append({'node':n['name'],'state':n['state'],'free_64g_slots':slots,'free_memory_mib':free_mem,'free_gpu':max(0,total-used),'record':n})
    require(any(r['node']=='node208' and r['free_64g_slots']>0 for r in rows),'qualified node208 has no schedulable capacity')
    return rows


def select_next(status, candidates):
    if status['blocked_reason'] is not None:return None
    return next((j for j in candidates if j in status['staged_held_job_ids']),None)


def setup(cap,candidates):
    global INSTALLED
    if not INSTALLED:
        migration.install_controller(MIGRATION_SHA)
        INSTALLED=True
    c.MAX_ACTIVE=cap
    burst.CAP=cap
    burst.ARRAYS[ARRAY[0]]=ARRAY[1:]
    def status(campaign,root):
        result=old.ORIGINAL_STATUS(campaign,root)
        result['next_job_id']=select_next(result,candidates)
        if result['next_job_id']:
            budget=burst.storage_budget(campaign,[result['next_job_id']])
            result['shared_storage']=budget
            if not budget['allowed']:result.update(next_job_id=None,blocked_reason='waiting_shared_storage')
        return result
    c.status=status


def prepare():
    require(not PLAN.exists(),'existing expansion plan must be retained')
    setup(4,[])
    campaign=migration.merged_campaign(MIGRATION_SHA)
    with old.admission_locks(),c.locked(c.JOURNAL_ROOT),c.locked(migration.NEW_JOURNAL):
        status=old.ORIGINAL_STATUS(campaign,migration.NEW_JOURNAL)
        require(not status['issues'] and not status['needs_operator_review_job_ids'] and not status['unknown'],'campaign requires reconciliation')
        capacity=node_capacity()
        pending=[]
        for jid in status['reserved_unfinished_job_ids']:
            raw=run(['scontrol','show','job','-dd','-o',jid])
            if c.field(raw,'JobState')=='PENDING':
                mem=c.field(raw,'MinMemoryNode');require(mem.endswith('G'),'unrecognized pending memory')
                pending.append({'job_id':jid,'memory_gib':int(mem[:-1]),'record':raw})
        available=sum(r['free_64g_slots'] for r in capacity)-sum((r['memory_gib']+63)//64 for r in pending)
        candidates=[j for j in CANDIDATES if j in status['staged_held_job_ids']][:max(0,available)]
        require(candidates,'no additional capacity after existing released reservations')
        cap=status['reserved_unfinished_slots']+len(candidates)
        setup(cap,candidates)
        jobs={j['job_id']:j for j in campaign['jobs']}
        held={}
        for jid in candidates:
            raw=old.launcher.audit_held(int(jid),jobs[jid]['cell'])
            require(c.field(raw,'MinMemoryNode')=='64G','candidate memory is not64GiB')
            held[jid]=raw
        dataset,proof=current_data.neutral_python_dataset()
        budget=burst.storage_budget(campaign,candidates)
        require(budget['allowed'],'aggregate storage insufficient')
        files=[SOURCE,ROOT/'tests/test_expand_e122_finite_20260912.py',Path(migration.__file__),Path(burst.__file__),Path(current_data.__file__),ARRAY[2],ARRAY[3]]
        plan={'schema':'e122_finite_capacity_expansion_v1','created_at':c.now(),'authorization':'User requested as many E122 jobs started as possible after neutral Python admission, on September12.',
              'migration_plan_sha256':MIGRATION_SHA,'binding':campaign['binding'],'max_unfinished_slots':cap,'max_additional_releases':len(candidates),'candidate_job_ids':candidates,
              'selection':'Initial neutral Python seed43 four-arm factorial, then graph seed43 remaining MaxRL pair; truncate only to currently observed GPU/CPU/memory capacity.',
              'initial_status':status,'capacity':capacity,'pending_reservations':pending,'held_records':held,'storage':budget,'neutral_dataset':str(dataset),'neutral_proof':proof,
              'allowed_pool':sorted(POOL),'persistent_controller_cap_unchanged':4,'no_automatic_replenishment':True,
              'source_pins':{str(p):digest(p) for p in files}}
        ART.mkdir(parents=True,exist_ok=True);new(PLAN,plan)
    return {'status':'prepared','plan_sha256':digest(PLAN),'additional':len(candidates),'cap':cap,'candidates':candidates,'storage_margin_gib':budget['margin_bytes']/GIB}


def validate(expected):
    require(digest(PLAN)==expected,'expansion plan changed')
    plan=read(PLAN)
    for p,sha in plan['source_pins'].items():require(digest(p)==sha,'source changed: '+p)
    require(plan['max_additional_releases']==len(plan['candidate_job_ids']) and set(plan['candidate_job_ids'])<=set(CANDIDATES),'finite candidate set differs')
    require(plan['max_unfinished_slots']==plan['initial_status']['reserved_unfinished_slots']+len(plan['candidate_job_ids']),'finite cap differs')
    return plan


def route_pending(jid,cell):
    out=ART/'routes'/jid
    if (out/'intent.json').exists():return
    raw=run(['scontrol','show','job','-dd','-o',jid])
    if c.field(raw,'JobState')!='PENDING':return
    require(c.field(raw,'Priority')!='0' and c.field(raw,'Reason') not in ['JobHeldUser','JobHeldAdmin'],'route amendment preserves existing holds')
    nodes=set(run(['scontrol','show','hostnames',c.field(raw,'ReqNodeList')]).split())
    require(nodes<=POOL and nodes,'unexpected current route')
    if nodes==POOL:return
    # Frozen SubmitLine and full exported science are checked independently.
    import prioritize_e118_capacity_20260905 as identity
    env=identity.exports(identity.submit_tokens(raw))
    require(all(env.get(k)==v for k,v in cell['environment'].items()),'runtime/scientific environment changed')
    node_capacity()
    argv=['scontrol','update','JobId='+jid,'ReqNodeList='+','.join(sorted(POOL))]
    new(out/'intent.json',{'created_at':c.now(),'before':raw,'command':argv,'source_sha256':digest(SOURCE)})
    result=subprocess.run(argv,text=True,capture_output=True,timeout=45)
    new(out/'ack.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    after=run(['scontrol','show','job','-dd','-o',jid])
    require(result.returncode==0,'route update failed; inspect receipt without retry')
    require(identity.submit_tokens(raw)==identity.submit_tokens(after),'SubmitLine changed')
    for key in PRESERVE:require(c.field(raw,key)==c.field(after,key),'resource/science field changed: '+key)
    require(set(run(['scontrol','show','hostnames',c.field(after,'ReqNodeList')]).split())==POOL,'route readback differs')
    new(out/'verified.json',{'after':after,'scientific_configuration_preserved':True,'only_pending_route_expanded':True})


def release(expected):
    plan=validate(expected)
    require(not (ART/'claim.json').exists(),'expansion already claimed; inspect once-only journals')
    setup(plan['max_unfinished_slots'],plan['candidate_job_ids'])
    new(ART/'claim.json',{'plan_sha256':expected,'at':c.now()})
    released=[]
    with old.admission_locks(),c.locked(c.JOURNAL_ROOT):
        campaign=migration.merged_campaign(MIGRATION_SHA)
        jobs={j['job_id']:j for j in campaign['jobs']}
        for pending in plan['pending_reservations']:route_pending(pending['job_id'],jobs[pending['job_id']]['cell'])
        for index in range(plan['max_additional_releases']):
            validate(expected)
            result=c.advance_once(old.fresh_args(),old.launcher,root=migration.NEW_JOURNAL)
            new(ART/f'release_{index}.json',result)
            jid=result.get('last_release_job_id')
            if not jid:break
            released.append(jid)
            print(json.dumps({'released':jid,'count':len(released)}),flush=True)
            require(not result['issues'] and not result['needs_operator_review_job_ids'],'release journal requires review')
            route_pending(jid,jobs[jid]['cell'])
        final=c.status(migration.merged_campaign(MIGRATION_SHA),migration.NEW_JOURNAL)
    result={'status':'finite_expansion_complete','released_job_ids':released,'count':len(released),'final_status':final,'persistent_cap':4,'plan_sha256':expected}
    new(ART/'result.json',result)
    return {k:v for k,v in result.items() if k!='final_status'}


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','release']);p.add_argument('--plan-sha256');a=p.parse_args()
    print(json.dumps(prepare() if a.action=='prepare' else release(a.plan_sha256)),flush=True)
if __name__=='__main__':main()
