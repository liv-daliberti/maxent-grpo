#!/usr/bin/env python3
"""One authenticated four-cell E122 graph burst; retain the original cap4 watcher."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

import amend_e122_countdown_peers_node208_20260910 as route
import launch_e122_level3_factorial as launcher
import e122_slurm_2511_compat as compat
import control_e122_level3_release as controller

ROOT=launcher.ROOT
ART=ROOT/'var/artifacts/e122_graph43_burst_20260910'
PLAN=ART/'plan.json'
PROTOCOL=ROOT/'paper/preregistration/e122_graph43_burst_20260910.md'
SOURCE=Path(__file__).resolve()
TEST=ROOT/'tests/test_e122_graph43_burst_20260910.py'
JOBS={31158698:'drgrpo',31158699:'replay_drgrpo',31158700:'maxrl',31158701:'replay_maxrl'}
JOURNAL=controller.JOURNAL_ROOT
STDOUT=launcher.HERE/'release_controller.stdout'
GIB=controller.GIB
require=route.require

def authenticate():
    watch=json.loads((launcher.HERE/'root_watch_arm_result.json').read_text())['command']
    src=watch[watch.index('--compat-source-sha256')+1]
    test=watch[watch.index('--compat-test-sha256')+1]
    compat.install_compat(launcher,src,test)
    args=controller.parse_args(watch[watch.index('--')+1:])
    campaign=controller.load_campaign(args,launcher)
    selected=[j for j in campaign['jobs'] if int(j['job_id']) in JOBS]
    require(len(selected)==4,'Exact four graph43 jobs required')
    for job in selected:
        cell=job['cell']
        require((cell['domain'],cell['arm'],cell['seed'])==('graph',JOBS[int(job['job_id'])],43),'Graph cell changed')
        require(cell['environment']['OAT_ZERO_MAX_PROMPT_EPOCHS']=='8','Horizon changed')
    return campaign,selected

def prepare():
    require(not PLAN.exists(),'Inspect existing burst plan; never overwrite')
    campaign,selected=authenticate()
    before=controller.status(campaign,JOURNAL)
    require(before['reserved_unfinished_slots']==4 and before['running']>=1,'Original four released reservations changed')
    require(not before['issues'] and not before['unknown'] and not before['needs_operator_review_job_ids'],'Controller requires review')
    audits={j['job_id']:launcher.audit_held(int(j['job_id']),j['cell']) for j in selected}
    require(all(j['job_id'] in before['staged_held_job_ids'] for j in selected),'Graph43 cell not staged held')
    files=[SOURCE,TEST,PROTOCOL,Path(route.__file__),Path(launcher.__file__),Path(controller.__file__),Path(compat.__file__),compat.TEST,launcher.PLAN,launcher.LEDGER,JOURNAL/'context.json']
    plan={'schema':'e122-graph43-bounded-burst-v1','created_at':controller.now(),'authorization':'User requested accelerating E122 today while finishing E118/E119/E120.','binding':campaign['binding'],'jobs':[j['job_id'] for j in selected],'held_audits':audits,'cap_during_burst':8,'original_controller_cap_retained':4,'storage_headroom_bytes':64*GIB,'terminal_reserve_bytes':125*GIB,'per_active_peak_bytes':16*GIB,'required_at_eight_bytes':317*GIB,'external_incremental_peak_reserve_bytes':0,'files_sha256':{str(p):controller.digest(p) for p in files},'status_before':{k:before[k] for k in ('observed_at','running','reserved_unfinished_slots','issues','storage')},'after_release_nodes':route.POOL,'no_new_submission':True}
    controller.immutable_json(PLAN,plan)
    print(json.dumps({'status':'prepared','jobs':plan['jobs'],'plan':str(PLAN),'plan_sha256':controller.digest(PLAN),'required_at_eight_GiB':317}),flush=True)

def verify_plan():
    plan=json.loads(PLAN.read_text())
    require(all(controller.digest(Path(p))==sha for p,sha in plan['files_sha256'].items()),'Frozen burst or campaign input changed')
    require(plan['jobs']==[str(j) for j in JOBS] and plan['cap_during_burst']==8,'Burst jobs or cap changed')
    require(plan['external_incremental_peak_reserve_bytes']==0,'Budget amendment requires review')
    return plan

def latest_heartbeat():
    with STDOUT.open('rb') as stream:
        stream.seek(0,2); size=stream.tell(); stream.seek(max(0,size-150000))
        lines=stream.read().splitlines()
    data=json.loads(lines[-1])
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(data['observed_at'])).total_seconds()
    require(data['max_released_nonterminal']==4 and not data['issues'] and not data['unknown'],'Original controller health changed')
    return data,age

def route_released(job):
    jid=int(job['job_id']); before=route.base.show(jid)
    if route.base.field(before,'JobState')!='PENDING':
        return {'job_id':jid,'status':'preserved_started','before':before}
    require(route.base.field(before,'Priority')!='0' and route.base.field(before,'Reason') not in {'JobHeldUser','JobHeldAdmin'},'Released job is held')
    require(route.nodes(before)==route.OLD_NODES,'Unexpected graph route')
    route.capacity()
    argv=['scontrol','update',f'JobId={jid}',f'ReqNodeList={route.POOL}']
    intent=ART/f'{jid}.route.intent.json'
    controller.immutable_json(intent,{'job_id':jid,'before':before,'command':argv,'created_at':controller.now()})
    result=route.command(argv,check=False)
    controller.immutable_json(ART/f'{jid}.route.result.json',{'job_id':jid,'intent_sha256':controller.digest(intent),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'recorded_at':controller.now()})
    after=route.base.show(jid)
    require(route.base.submit_tokens(after)==route.base.submit_tokens(before),'SubmitLine changed')
    require(all(route.base.field(after,k)==route.base.field(before,k) for k in route.PRESERVE),'Preserved scheduler field changed')
    require(result.returncode==0 and route.nodes(after)==route.NEW_NODES,'Graph route expansion not acknowledged')
    controller.immutable_json(ART/f'{jid}.route.readback.json',{'job_id':jid,'record':after,'created_at':controller.now()})
    return {'job_id':jid,'status':'route_expanded','state':route.base.field(after,'JobState')}

def apply():
    require(not (ART/'storage_blocked.json').exists(), 'Graph burst is storage-blocked; require a new reviewed aggregate-budget amendment')
    plan=verify_plan()
    require(not (ART/'burst.intent.json').exists(),'Burst already claimed; inspect journals rather than retry')
    campaign,selected=authenticate()
    require(campaign['binding']==plan['binding'],'Campaign binding changed')
    # Heavy admission authentication and each exact held audit stay outside the
    # shared controller lock. The original watcher uses a nonblocking lock.
    for job in selected:
        launcher.audit_held(int(job['job_id']),job['cell'])
    verify_plan()
    # Wait for a freshly published status: the original watcher is entering its
    # 60-second sleep. Critical scheduler operations below have short deadlines.
    wait_end=time.monotonic()+90
    while True:
        heartbeat,age=latest_heartbeat()
        if 0<=age<=3: break
        require(time.monotonic()<wait_end,'No fresh original heartbeat; do not risk controller collision')
        time.sleep(.4)
    require(heartbeat['reserved_unfinished_slots']==4,'Released slots changed before bounded burst')
    old_command=controller.command
    deadline=time.monotonic()+25
    def bounded_command(argv):
        remaining=deadline-time.monotonic()
        require(remaining>1,'Burst lock deadline reached; stop further releases')
        return subprocess.run(argv,capture_output=True,text=True,check=False,timeout=min(3,remaining-0.2))
    controller.command=bounded_command
    controller.MAX_ACTIVE=8
    try:
        with controller.locked(JOURNAL):
            controller.immutable_json(ART/'burst.intent.json',{'created_at':controller.now(),'plan_sha256':controller.digest(PLAN),'heartbeat_at':heartbeat['observed_at'],'job_ids':plan['jobs'],'lock_deadline_seconds':25})
            for job in selected:
                require(deadline-time.monotonic()>10,'Insufficient bounded window for another release')
                jid=job['job_id']
                before=controller.status(campaign,JOURNAL)
                require(not before['issues'] and not before['unknown'] and not before['needs_operator_review_job_ids'],'Fresh status needs review')
                require(before['reserved_unfinished_slots']<8 and before['storage']['can_release'],'Storage or cap blocks graph release')
                require(jid in before['staged_held_job_ids'],'Graph cell no longer staged held')
                # Exact current held audit remains mandatory; the read operation
                # itself uses the bounded scheduler command.
                current=bounded_command(['scontrol','show','job','-dd','-o',jid])
                require(current.returncode==0,'Current graph held audit unavailable')
                held=launcher.audit_held_record(current.stdout,int(jid),job['cell'])
                intent=JOURNAL/'jobs'/f'{jid}.intent.json'
                argv=['scontrol','release',jid]
                controller.immutable_json(intent,{'schema':'e122_level3_release_intent_v1','created_at':controller.now(),'binding':campaign['binding'],'job_id':jid,'cell':job['cell'],'held_scheduler_record':held,'command':argv,'decision':before,'operational_amendment_plan':str(PLAN),'operational_amendment_sha256':controller.digest(PLAN)})
                result={'schema':'e122_level3_release_result_v1','job_id':jid,'intent_sha256':controller.digest(intent),'command':argv}
                try:
                    done=bounded_command(argv)
                    result.update(returncode=done.returncode,stdout=done.stdout,stderr=done.stderr,error=None)
                except Exception as exc:
                    result.update(returncode=None,error=f'{type(exc).__name__}: {exc}')
                result['recorded_at']=controller.now()
                controller.immutable_json(JOURNAL/'jobs'/f'{jid}.result.json',result)
                require(result['returncode']==0 and not result.get('error'),'Uncertain release; consumed intent must be reconciled')
                print(json.dumps({'event':'released','job_id':jid}),flush=True)
            after=controller.status(campaign,JOURNAL)
            controller.immutable_json(ART/'burst.result.json',{'created_at':controller.now(),'status':after,'lock_elapsed_seconds':25-(deadline-time.monotonic())})
    finally:
        controller.command=old_command
        controller.MAX_ACTIVE=4
    # Routing happens only after leaving the original controller lock, and
    # never changes held requests before their strict original release audit.
    for job in selected:
        print(json.dumps(route_released(job)),flush=True)
    print(json.dumps({'event':'complete','released':list(JOBS),'legacy_cap':4}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    with (ART/'singleton.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (apply if args.apply else prepare)()
