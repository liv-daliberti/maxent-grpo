#!/usr/bin/env python3
"""User-authorized finite burst: four additional E122 non-Python jobs.

Eight unfinished slots maximum; original resources, snapshots and release
journals are retained. Every pending inference task keeps a full32GiB reserve.
No automatic replenishment and no Python release are authorized by this file.
"""
from pathlib import Path
import argparse
import json
import os
import re
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import migrate_level3_neutral as archive
import e124_storage_admission as base
c,old=archive.c,archive.old
ART=ROOT/'var/artifacts/e122_nonpython_burst_20260911'
PLAN=ART/'plan.json'
SOURCE=Path(__file__).resolve()
CAP=8
MAX_RELEASES=4
GIB=1024**3
ARRAYS={
 '31243495':(10,ROOT/'var/artifacts/modebench_scale_runtime_v3/development/worker.slurm',ROOT/'var/artifacts/modebench_scale_runtime_v3/development/plan.json'),
 '31251914':(5,ROOT/'artifacts/modebench_inference_followups_20260911/pantry/worker.slurm',ROOT/'artifacts/modebench_inference_followups_20260911/pantry/execution_plan.json'),
 '31252444':(45,ROOT/'artifacts/modebench_base_level_grid_20260911/collection_v3/worker.slurm',ROOT/'artifacts/modebench_base_level_grid_20260911/collection_v3/plan.json'),
 '31252445':(14,ROOT/'artifacts/modebench_base_level_grid_20260911/collection_v3/worker.slurm',ROOT/'artifacts/modebench_base_level_grid_20260911/collection_v3/plan.json'),
 '31252690':(4,archive.calibration.SOURCE,archive.calibration.PLAN),
}
read,digest,new,require,run=archive.read,archive.digest,archive.new,archive.require,archive.run


def snapshot():
    # Expanded Slurm array tasks are separate future writers, even while queued.
    return run(['squeue','--array','-h','-u',str(os.getuid()),'-o','%i|%T|%r|%b'])


def storage_budget(campaign,forced=(),queue=None):
    queue=snapshot() if queue is None else queue
    registry=base.canonical_registry()
    own={j['job_id'] for j in campaign['jobs']}
    force=set(map(str,forced));seen=set();reservations=[];own_live=[];array_checks={}
    for line in queue.splitlines():
        jid,state,reason,gres=line.split('|',3)
        if 'gpu' not in gres.lower():continue
        held=state=='PENDING' and reason in base.HELD_REASONS
        if held and jid not in force:continue
        require(jid not in seen,'duplicate GPU writer');seen.add(jid)
        if jid in own:
            own_live.append(jid);continue
        if '_' in jid:
            parent,index=jid.split('_',1)
            require(parent in ARRAYS and index.isdecimal() and int(index)<ARRAYS[parent][0],'unregistered inference task: '+jid)
            count,worker,plan=ARRAYS[parent]
            if parent not in array_checks:
                raw=run(['scontrol','show','job','-dd','-o',jid])
                require(c.field(raw,'UserId').endswith(f'({os.getuid()})') and c.field(raw,'WorkDir')==str(ROOT)
                        and str(worker) in raw,'inference worker identity changed')
                array_checks[parent]={'representative_job':jid,'scheduler_record':raw,'worker_sha256':digest(worker),'plan_sha256':digest(plan)}
            reservations.append({'job_id':jid,'kind':'inference','bytes':32*GIB});continue
        require(jid in registry and not registry[jid].get('model_error'),'unknown training writer: '+jid)
        identity=registry[jid];choice=base.model_choice(identity)
        base.bounded_run_path(identity['run_dir'])
        # Conservative: reserve first plus overlap even if a checkpoint exists.
        reservations.append({'job_id':jid,'kind':'training','model_choice':choice,'run_dir':identity['run_dir'],
                             'bytes':2*base.CHECKPOINT_BYTES[choice]})
    require(force<=seen,'candidate absent from scheduler')
    require(len(own_live)<=CAP,'E122 unfinished live cap exceeded')
    fixed=(64+125+16*CAP)*GIB
    required=fixed+sum(r['bytes'] for r in reservations)
    stat=os.statvfs(ROOT);free=stat.f_bavail*stat.f_frsize
    return {'allowed':free>=required and stat.f_favail>=10000,'free_bytes':free,'required_bytes':required,
            'margin_bytes':free-required,'e122_cap':CAP,'e122_peak_bytes':16*CAP*GIB,
            'e122_terminal_bytes':125*GIB,'shared_headroom_bytes':64*GIB,
            'own_live_job_ids':own_live,'reservations':reservations,'inference_identity_checks':array_checks,
            'queue':queue,'observed_at':c.now()}


def nonpython_next(status,campaign):
    if status['blocked_reason'] is not None:return None
    allowed={j['job_id'] for j in campaign['jobs'] if j['cell']['domain']!='python_factors'}
    return next((jid for jid in status['staged_held_job_ids'] if jid in allowed),None)


def setup():
    c.MAX_ACTIVE=CAP
    c.load_campaign=lambda *_a,**_k:archive.legacy_campaign()
    def scheduler(ids):
        q=run(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%r'])
        a=run(['sacct','-X','-n','-P','-j',','.join(ids),'-o','JobIDRaw,State,ExitCode'])
        return c.parse_scheduler(ids,q,a)
    c.scheduler_snapshot=scheduler
    def status(campaign,root):
        value=old.ORIGINAL_STATUS(campaign,root)
        value['next_job_id']=nonpython_next(value,campaign)
        if value['next_job_id']:
            budget=storage_budget(campaign,[value['next_job_id']]);value['shared_storage']=budget
            if not budget['allowed']:value.update(next_job_id=None,blocked_reason='waiting_shared_storage')
        return value
    c.status=status


def prepare():
    require(not PLAN.exists(),'burst already prepared')
    setup();campaign=archive.legacy_campaign()
    with old.admission_locks(),c.locked(c.JOURNAL_ROOT):
        status=c.status(campaign,c.JOURNAL_ROOT)
        require(not status['issues'] and not status['needs_operator_review_job_ids'] and not status['unknown_job_ids'],'campaign requires reconciliation')
        require(len(status['reserved_unfinished_job_ids'])==4,'expected original four unfinished jobs')
        python=[j for j in campaign['jobs'] if j['cell']['domain']=='python_factors']
        require(len(python)==20 and all(j['job_id'] in status['staged_held_job_ids'] for j in python),'all20 Python jobs must remain held')
        budget=storage_budget(campaign)
        require(budget['allowed'],'insufficient aggregate storage headroom')
    ART.mkdir(parents=True,exist_ok=True)
    sources=[SOURCE,Path(archive.__file__),Path(base.__file__),ROOT/'tests/test_e122_nonpython_burst_20260911.py',
             archive.ORIGINAL_TEMPLATE,*[p for _,w,p in ARRAYS.values()],*[w for _,w,p in ARRAYS.values()]]
    new(PLAN,{'schema':'e122_finite_nonpython_burst_v1','created_at':c.now(),'max_unfinished_slots':CAP,
              'max_additional_releases':MAX_RELEASES,'initial_unfinished_ids':status['reserved_unfinished_job_ids'],
              'protected_python_job_ids':[j['job_id'] for j in python],
              'original_binding':campaign['binding'],'source_pins':{str(p):digest(p) for p in sources},
              'shared_storage_at_prepare':budget,'python_release_allowed':False,
              'authorization':'User requested more E122 releases, explicitly excluding Python Level3.'})
    return {'status':'prepared','plan_sha256':digest(PLAN),'free_gib':budget['free_bytes']/GIB,
            'required_gib':budget['required_bytes']/GIB,'cap':CAP}


def release(expected):
    require(digest(PLAN)==expected,'burst plan hash changed')
    plan=read(PLAN)
    for p,sha in plan['source_pins'].items():require(digest(p)==sha,'burst source changed: '+p)
    require(not (ART/'release_claim.json').exists(),'burst already claimed; inspect its journals')
    new(ART/'release_claim.json',{'plan_sha256':expected,'created_at':c.now()})
    setup();released=[]
    for index in range(MAX_RELEASES):
        with old.admission_locks():
            current=archive.legacy_campaign()
            status=c.status(current,c.JOURNAL_ROOT)
            require(all(j in status['staged_held_job_ids'] for j in plan['protected_python_job_ids']),'Python hold changed')
            if status['next_job_id'] is None:break
            require(status['next_job_id'] not in plan['protected_python_job_ids'],'Python is excluded')
            result=c.advance_once(old.fresh_args(),old.launcher)
            new(ART/f'release_{index}.json',result)
            if result.get('last_release_job_id'):released.append(result['last_release_job_id'])
            else:break
    final=c.status(archive.legacy_campaign(),c.JOURNAL_ROOT)
    require(all(j in final['staged_held_job_ids'] for j in plan['protected_python_job_ids']),'Python hold changed')
    result={'status':'released','released_job_ids':released,'count':len(released),
            'running':final['running'],'eligible_or_running':final['released_nonterminal'],
            'python_held':20,'final_status':final,'burst_plan_sha256':expected}
    new(ART/'result.json',result)
    return {k:v for k,v in result.items() if k!='final_status'}


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','release']);p.add_argument('--plan-sha256');a=p.parse_args()
    print(json.dumps(prepare() if a.action=='prepare' else release(a.plan_sha256)),flush=True)

if __name__=='__main__':main()
