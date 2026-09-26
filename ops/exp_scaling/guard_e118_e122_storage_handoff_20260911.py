#!/usr/bin/env python3
"""Reversible pending-only E118 holds with bounded shared-storage-gated releases."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime,timedelta,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

import accelerate_a5000_completion_20260909 as runtime
import e122_shared_storage_admission as storage

b, recovery=runtime.base,runtime.recovery
ROOT=b.ROOT
ART=ROOT/'var/artifacts/e118_e122_storage_handoff_20260911'
PLAN,TX,REG=ART/'plan.json',ART/'transaction.json',ART/'supervisor.json'
PROTOCOL=ROOT/'paper/preregistration/e118_e122_storage_handoff_20260911.md'
STORAGE_LOCK=ROOT/'var/artifacts/shared_storage_admission_20260911.lock'
LEDGER_LOCK=ROOT/'var/artifacts/e118_ledger_promotion.lock'
SINGLETON=ART/'singleton.lock'
E124_CPU=31164037
E124_BASE=ROOT/'var/artifacts/e124_qwen7b_three_level'
E124_WAITER=E124_BASE/'controller_recovery_20260909_1940/cpu_waiter_v2'
E124_EXTRA_BYTES=220*1024**3
HELD_REASONS={'JobHeldUser','job_requeued_in_held_state','job requeued in held state'}
TARGETS=(31151400,31124282,31124283,31124285,31193187,31048143)
SOURCE=b.LEDGER
AGGREGATE=runtime.campaign.E118_LEDGER
GUARD_TX=(ROOT/'var/artifacts/campaign_timeout_guard_20260908/transaction.json',
          ROOT/'var/artifacts/e118_pantry_successor_guard_20260910/transaction.json')
FIELDS=('UserId','JobName','Account','Partition','QOS','ReqNodeList','ExcNodeList','MinMemoryNode',
        'NumCPUs','NumTasks','CPUs/Task','TresPerNode','TimeLimit','Nice','Requeue','Features',
        'WorkDir','Command','StdOut','StdErr','Comment','Restarts')
CPU_FIELDS=('UserId','JobName','Account','Partition','QOS','ReqNodeList','MinMemoryNode','NumCPUs',
            'NumTasks','Requeue','TimeLimit','Command','Comment')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def now():return datetime.now(timezone.utc)
def digest(path):return runtime.digest(path)
def command(parts,*,check=True):return subprocess.run(parts,capture_output=True,text=True,check=check,timeout=45)
b.command=command


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as f:f.write(b.encoded(value));f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def field(record,key):
    try:return b.field(record,key)
    except RuntimeError:
        if key=='Comment':return None
        raise


def event(tx,message):
    tx['updated_at_utc']=now().isoformat();tx.setdefault('events',[]).append({'at_utc':tx['updated_at_utc'],'event':message})
    atomic(TX,tx);print(json.dumps({'at_utc':tx['updated_at_utc'],'event':message}),flush=True)


@contextmanager
def admission_locks():
    # Same order for handoff, release and E122 pair admission; never acquire its journal lock.
    with STORAGE_LOCK.open('a+') as storage_lock:
        fcntl.flock(storage_lock,fcntl.LOCK_EX)
        with LEDGER_LOCK.open('a+') as ledger_lock:
            fcntl.flock(ledger_lock,fcntl.LOCK_EX)
            yield


def mapping(item):
    source=json.loads(SOURCE.read_text())['runs'];aggregate=json.loads(AGGREGATE.read_text())['runs']
    require(len(source)==50 and len(aggregate)==150,'E118 cardinality changed')
    rows=[r for r in source if r['job_id']==item['job_id']]
    matches=[r for r in aggregate if r['job_id']==item['job_id']]
    require(len(rows)==len(matches)==1 and matches[0]==dict(rows[0],scale='qwen3b'),'authoritative source/aggregate mapping differs')
    require({k:rows[0][k] for k in b.IDENTITY}==item['identity'],'scientific identity changed')


def dependency_ids(text):
    if text=='(null)':return set()
    require(all(part.startswith('afterany:') for part in text.split(',')),'unexpected dependency type')
    ids=set(map(int,re.findall(r'(?:afterany:|:)(\d+)',text)))
    require(ids,'malformed afterany dependency');return ids


def dependency_preserved(item,record):
    original=set(item['dependency_parents']);remaining=dependency_ids(b.field(record,'Dependency'))
    require(remaining<=original,'new dependency parent appeared')
    for parent in original-remaining:
        result=command(['scontrol','show','job','-dd','-o',str(parent)],check=False)
        if result.returncode==0 and result.stdout.strip():
            require(b.field(result.stdout,'JobState') not in ('RUNNING','PENDING','CONFIGURING','SUSPENDED','COMPLETING'),'live afterany parent disappeared')


def stable(item,record,*,held=None):
    require(b.field(record,'JobId')==str(item['job_id']),'job ID changed')
    require(b.submit_tokens(record)==item['submit_tokens'],'submitted recipe changed')
    for key,value in item['resources'].items():require(field(record,key)==value,'resource or restart changed: '+key)
    require(b.field(record,'NumNodes') in ('1','1-1'),'node count changed')
    dependency_preserved(item,record)
    if held is not None:
        require(b.field(record,'JobState')=='PENDING' and b.field(record,'RunTime')=='00:00:00','job started; preserve active allocation')
        if held:require(b.field(record,'Reason')=='JobHeldUser' and b.field(record,'Priority')=='0','exact owned pending hold absent')
        else:require(b.field(record,'Priority')!='0' and b.field(record,'Reason') not in ('JobHeldUser','JobHeldAdmin','job_requeued_in_held_state'),'preexisting hold belongs to another operation')


def checkpoint(item):return runtime.checkpoint(item['identity']['run_dir'])


def scientific(item,*,allowed=None):
    mapping(item)
    require(not recovery.complete(Path(item['identity']['run_dir'])),'terminal receipt exists; no future allocation required')
    require(checkpoint(item)==item['checkpoint'],'held checkpoint changed')
    require(digest(item['submit_tokens'][-1])==item['launcher_sha256'],'frozen launcher changed')
    writers=recovery.active_writers().get(str(Path(item['identity']['run_dir']).resolve()),set())
    require(writers<=set(allowed or [item['job_id']]),'duplicate run writer')


def pantry_guard_clear():
    evidence={}
    for path in GUARD_TX:
        if not path.exists():continue
        d=json.loads(path.read_text());state=d.get('jobs',{}).get('31048143',{})
        require(not any(not x.get('released') for x in state.get('attempts',[])),'Pantry guard has unresolved retry')
        require(state.get('status','monitoring') in ('monitoring','completed'),'Pantry guard requires manual review')
        evidence[str(path)]={'status':state.get('status','monitoring'),'retries':len(state.get('attempts',[]))}
    return evidence


@contextmanager
def e124_boundary():
    # The unchanged E124 watcher serializes every GPU mutation on this lock.
    # Nonblocking acquisition never interrupts an in-progress GPU transaction.
    with (E124_BASE/'controller.lock').open('r+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def e124_gpu_holds(expected=None):
    canonical=json.loads((E124_BASE/'transaction.json').read_text())
    require(canonical['controller']['job_id']==E124_CPU,'E124 canonical controller changed')
    rows=canonical['rows'];require(len(rows)==31 and all(r['status']=='held' for r in rows.values()),'E124 GPU mutation/live intent exists')
    identities={k:int(v['job_id']) for k,v in rows.items()}
    if expected is not None:require(identities==expected,'E124 held GPU inventory changed')
    queued=command(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%r']).stdout
    states={int(parts[0]):parts[1:] for line in queued.splitlines() if len(parts:=line.split('|'))==3 and parts[0].isdecimal()}
    require(all(states.get(jid)==['PENDING','JobHeldUser'] for jid in identities.values()),'E124 GPU is no longer an original pending user hold')
    return identities


def e124_item():
    rec=b.show(E124_CPU);tokens=b.submit_tokens(rec)
    require(tokens[-1]==str(E124_WAITER/'waiter.slurm') and '--gres=none' in tokens,'unexpected E124 CPU command')
    keys=[k for k in FIELDS if k not in ('NumCPUs','TresPerNode','Restarts')]+['MinCPUsNode','TresPerTask','Dependency']
    return {'job_id':E124_CPU,'submit_tokens':tokens,'resources':{k:field(rec,k) for k in keys},
            'before':rec,'prepared_restarts':int(b.field(rec,'Restarts')),'gpu_ids':e124_gpu_holds()}


def e124_stable(item,record,*,held=False,expected_restarts=None):
    require(b.field(record,'JobId')==str(E124_CPU) and b.submit_tokens(record)==item['submit_tokens'],'E124 CPU submitted identity changed')
    require('gres/gpu' not in b.field(record,'ReqTRES'),'E124 waiter requests GPU')
    for key,value in item['resources'].items():require(field(record,key)==value,'E124 CPU requested resource changed: '+key)
    require(b.field(record,'NumNodes') in ('1','1-1'),'E124 CPU node count changed')
    require(int(b.field(record,'Restarts'))>=item['prepared_restarts'],'E124 CPU restart history decreased')
    if expected_restarts is not None:require(int(b.field(record,'Restarts'))==expected_restarts,'E124 owned hold restart history changed')
    if held:
        require(b.field(record,'JobState')=='PENDING' and b.field(record,'Reason') in HELD_REASONS and b.field(record,'Priority')=='0','E124 exact inactive CPU owned hold absent')
        require(b.field(record,'RunTime')=='00:00:00' and b.field(record,'NodeList') in ('','(null)'),'E124 CPU allocation not yet inactive')
    return record


def pause_e124(plan,tx):
    item=plan['e124_cpu'];require(not tx.get('e124',{}).get('hold_intent'),'E124 CPU pause already attempted; reconcile receipt')
    with e124_boundary():
        e124_gpu_holds(item['gpu_ids']);rec=e124_stable(item,b.show(E124_CPU))
        require(b.field(rec,'JobState') in ('PENDING','RUNNING'),'E124 CPU is transitioning; observe again before pause')
        require(b.field(rec,'Priority')!='0' and b.field(rec,'Reason') not in HELD_REASONS|{'JobHeldAdmin'},'E124 CPU has a preexisting foreign hold')
        mode='hold' if b.field(rec,'JobState')=='PENDING' else 'requeuehold'
        state=tx['e124']={'hold_intent':True,'mode':mode,'before':rec,'before_restarts':int(b.field(rec,'Restarts'))}
        state['expected_restarts']=state['before_restarts']+(mode=='requeuehold');event(tx,'Persisted exact E124 CPU '+mode+' intent at idle controller boundary')
        command(['scontrol',mode,str(E124_CPU)])
        # Requeuehold may need Slurm's epilog to finish; no repeated mutation.
        for _ in range(15):
            rec=b.show(E124_CPU)
            if b.field(rec,'JobState')=='PENDING':break
            time.sleep(1)
        e124_stable(item,rec,held=True,expected_restarts=state['expected_restarts']);e124_gpu_holds(item['gpu_ids'])
        state.update(status='owned_held',held_record=rec);event(tx,'E124 CPU inactive in exact owned hold; original gate and all 31 GPU holds unchanged')


def reconcile_e124_pause(plan,tx):
    state=tx.get('e124',{})
    if not state.get('hold_intent') or state.get('status') in ('owned_held','released'):return
    rec=e124_stable(plan['e124_cpu'],b.show(E124_CPU),held=True,expected_restarts=state['expected_restarts'])
    e124_gpu_holds(plan['e124_cpu']['gpu_ids']);state.update(status='owned_held',held_record=rec);event(tx,'Reconciled exact E124 inactive CPU pause without repeating command')


def e124_restore(plan,tx,*,apply):
    state=tx.get('e124',{});summary={'job_id':E124_CPU,'status':state.get('status','not_owned'),'extra_bytes':E124_EXTRA_BYTES}
    if not state.get('hold_intent') or state.get('status')=='released':return summary
    # Never wake the other admission controller while any owned GPU hold remains.
    if any(v.get('status') not in ('released','started_race','not_owned') for v in tx['jobs'].values()):
        summary['status']='waiting_all_owned_e118_restored';return summary
    with e124_boundary():
        rec=e124_stable(plan['e124_cpu'],b.show(E124_CPU),expected_restarts=state['expected_restarts'])
        if state.get('release_intent'):
            if b.field(rec,'JobState') in ('PENDING','RUNNING','CONFIGURING','COMPLETING','COMPLETED') and b.field(rec,'Reason') not in HELD_REASONS|{'JobHeldAdmin'}:
                if apply:state.update(status='released',release_record=rec);event(tx,'Reconciled E124 CPU restore acknowledgement')
                summary['status']='released';return summary
            summary['status']='release_requires_reconciliation';return summary
        e124_stable(plan['e124_cpu'],rec,held=True,expected_restarts=state['expected_restarts']);e124_gpu_holds(plan['e124_cpu']['gpu_ids'])
        budget=storage.storage_report();required=(budget.get('required_bytes') or 0)+E124_EXTRA_BYTES
        allowed=budget['allowed'] and budget['free_bytes']>=required
        summary.update(status='waiting_storage',allowed=allowed,required_bytes=required,margin_bytes=(budget.get('free_bytes') or 0)-required,budget=budget)
        if not allowed:return summary
        if not apply:summary['status']='would_release';return summary
        before_deadline(plan);state['release_intent']=True;state['release_budget']=summary;event(tx,'Full shared E122 budget plus additional 220GiB admits final E124 CPU restore')
    # Drop the E124 controller lock before making its CPU eligible: the unchanged
    # waiter tests that lock nonblocking on startup. Shared storage lock stays held.
    command(['scontrol','release',str(E124_CPU)])
    rec=e124_stable(plan['e124_cpu'],b.show(E124_CPU),expected_restarts=state['expected_restarts'])
    require(b.field(rec,'JobState') in ('PENDING','RUNNING','CONFIGURING','COMPLETING','COMPLETED') and b.field(rec,'Reason') not in HELD_REASONS|{'JobHeldAdmin'},'E124 restore acknowledgement uncertain')
    state.update(status='released',release_record=rec);event(tx,'E124 original CPU waiter restored last; original gate and controller unchanged');summary['status']='released';return summary


def cpu_command():
    return ['sbatch','--parsable','--hold','--job-name=e118-e122-storage-guard','--account=mltheory','--partition=lowprio',
            '--nodelist=node915,node917','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=2G','--gres=none',
            '--time=1-01:10:00','--requeue','--export=NONE',f'--chdir={ROOT}',f'--output={ART}/supervisor-%j.out',
            f'--error={ART}/supervisor-%j.err','--comment=e118-e122-storage-handoff-20260911',str(ART/'supervisor.slurm')]


def hypothetical_holds():
    snap=storage.base.scheduler_snapshot();found=set()
    for row in snap['jobs']:
        if int(row['job_id']) in TARGETS:
            require(row['state']=='PENDING' and row['reason'] not in storage.base.HELD_REASONS,'hold candidate started or already held')
            row['reason']='JobHeldUser';found.add(int(row['job_id']))
    require(found==set(TARGETS),'hold candidate missing from scheduler')
    return storage.storage_report(external_snapshot=snap)


def prepare():
    require(not PLAN.exists() and not TX.exists(),'immutable preparation already exists')
    source=json.loads(SOURCE.read_text())['runs'];items=[]
    for job in TARGETS:
        row=next(r for r in source if r['job_id']==job);rec=b.show(job);tokens=b.submit_tokens(rec)
        item={'job_id':job,'identity':{k:row[k] for k in b.IDENTITY},'submit_tokens':tokens,
              'resources':{k:field(rec,k) for k in FIELDS},'dependency_parents':sorted(dependency_ids(b.field(rec,'Dependency'))),
              'before':rec,'launcher_sha256':digest(tokens[-1]),'original_command':tokens}
        item['checkpoint']=checkpoint(item);stable(item,rec,held=False);scientific(item);items.append(item)
    current=storage.storage_report();projected=hypothetical_holds()
    require(not current['errors'] and not projected['errors'],'storage classification unresolved')
    guard_evidence=pantry_guard_clear()
    with e124_boundary():e124=e124_item()
    (ART/'supervisor.slurm').write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'+f'cd {ROOT}\nexec /usr/bin/python3 -u {Path(__file__).resolve()} watch\n')
    paths=[Path(__file__),PROTOCOL,Path(storage.__file__),Path(storage.base.__file__),Path(runtime.__file__),
           Path(b.__file__),Path(recovery.__file__),ROOT/'ops/validate_deepspeed_checkpoint.py',
           E124_WAITER/'waiter.slurm',E124_WAITER/'waiter_control.py',E124_WAITER/'plan.json',E124_WAITER/'gate.json',
           E124_BASE/'controller_v2.py',E124_BASE/'plan.json']
    e124_recovery=json.loads((E124_WAITER/'plan.json').read_text())
    paths.extend(Path(path) for path in e124_recovery['pins'])
    plan={'schema':'e118-e122-storage-handoff-v1','created_at_utc':now().isoformat(),'deadline_utc':(now()+timedelta(days=7)).isoformat(),
          'items':items,'e124_cpu':e124,'e124_restore_extra_bytes':E124_EXTRA_BYTES,'pins':{str(p.resolve()):digest(p) for p in paths},'runtime_fingerprints':runtime.runtime_fingerprints(items),
          'budget_before':current,'budget_after_hypothetical_holds':projected,'pantry_guard_evidence':guard_evidence,
          'cpu_submit_tokens':cpu_command(),'cpu_script_sha256':digest(ART/'supervisor.slurm'),
          'poll_seconds':60,'renew_after_seconds':23*3600,'max_cpu_restarts':8,
          'shared_storage_lock':str(STORAGE_LOCK),'no_ledger_or_checkpoint_writes':True,'scheduler_mutations':False}
    atomic(PLAN,plan);atomic(ART/'cpu_submission.json',{'command':cpu_command(),'scheduler_mutations':False})
    print(json.dumps({'prepared':str(PLAN),'target_ids':TARGETS,'projected_allowed':projected['allowed'],
                      'projected_margin_bytes':projected.get('margin_bytes'),'scheduler_mutations':False}))


def load():
    plan=json.loads(PLAN.read_text());require(all(digest(p)==sha for p,sha in plan['pins'].items()),'prepared implementation changed')
    require(digest(ART/'supervisor.slurm')==plan['cpu_script_sha256'],'CPU script changed')
    tx=json.loads(TX.read_text()) if TX.exists() else {'schema':plan['schema'],'plan_sha256':digest(PLAN),'jobs':{},'events':[],'status':'prepared'}
    require(tx['plan_sha256']==digest(PLAN),'plan binding changed');return plan,tx


def before_deadline(plan):require(now()<datetime.fromisoformat(plan['deadline_utc']),'seven-day release deadline reached')


def cpu_record(plan,job,*,held=False,running=False):
    rec=b.show(job);require(b.submit_tokens(rec)==plan['cpu_submit_tokens'],'CPU submitted identity changed')
    require('gres/gpu' not in b.field(rec,'ReqTRES'),'CPU supervisor requests GPU')
    expected={'UserId':plan['items'][0]['resources']['UserId'],'Account':'mltheory','Partition':'lowprio','MinMemoryNode':plan.get('cpu_memory_override','2G'),
              'NumCPUs':'2','NumTasks':'1','Requeue':'1','TimeLimit':'1-01:10:00','Command':str(ART/'supervisor.slurm'),
              'Comment':'e118-e122-storage-handoff-20260911','JobName':'e118-e122-storage-guard'}
    require(all(b.field(rec,k)==v for k,v in expected.items()),'CPU actual resource profile differs')
    require(set(command(['scontrol','show','hostnames',b.field(rec,'ReqNodeList')]).stdout.split())=={'node915','node917'},'CPU pool changed')
    require(b.field(rec,'NumNodes') in ('1','1-1'),'CPU node count changed')
    if held:require(b.field(rec,'JobState')=='PENDING' and b.field(rec,'Reason')=='JobHeldUser' and b.field(rec,'Priority')=='0' and b.field(rec,'Restarts')=='0','CPU not never-started held')
    if running:require(b.field(rec,'JobState')=='RUNNING','CPU not running')
    require(int(b.field(rec,'Restarts'))<=plan['max_cpu_restarts'],'CPU renewal cap exceeded');return rec


def submit():
    plan,tx=load();before_deadline(plan);require(not tx.get('cpu_submission_intent'),'submission already attempted; reconcile exact receipt')
    tx['cpu_submission_intent']=True;event(tx,'Persisted one held CPU submission intent')
    result=command(plan['cpu_submit_tokens']);response=result.stdout.strip();tx['cpu_raw_receipt']={'stdout':response,'stderr':result.stderr};event(tx,'Recorded raw CPU acknowledgement')
    require(re.fullmatch(r'\d+(;\S+)?',response),'ambiguous CPU receipt; never resubmit blindly')
    job=int(response.split(';')[0]);tx['cpu_job_id']=job;event(tx,'Recorded exact CPU supervisor ID')
    rec=cpu_record(plan,job,held=True);atomic(REG,{'job_id':job,'plan_sha256':digest(PLAN),'submit_tokens':plan['cpu_submit_tokens'],'resources':{k:b.field(rec,k) for k in CPU_FIELDS}})
    tx['cpu_registered']=True;event(tx,'Verified and registered held CPU supervisor');print(json.dumps({'cpu_job_id':job,'held':True}))


def hold():
    plan,tx=load();before_deadline(plan);require(tx.get('cpu_registered'),'register CPU before GPU holds')
    cpu_record(plan,tx['cpu_job_id'],held=True);require(not tx.get('hold_started'),'hold already attempted; reconcile durable intents')
    require(runtime.runtime_fingerprints(plan['items'])==plan['runtime_fingerprints'],'runtime changed')
    with admission_locks():
        projected=hypothetical_holds();require(projected['allowed'],'six pending holds do not currently create full safe headroom')
        pantry_guard_clear()
        for item in plan['items']:stable(item,b.show(item['job_id']),held=False);scientific(item)
        tx['hold_started']=True;tx['projected_before_holds']=projected;event(tx,'Beginning exact pending hold package with E124 CPU paused first')
        pause_e124(plan,tx)
        for item in plan['items']:
            job=item['job_id'];stable(item,b.show(job),held=False);scientific(item)
            state=tx['jobs'].setdefault(str(job),{});state['hold_intent']=True;event(tx,f'{job}: persisted pending hold intent')
            command(['scontrol','hold',str(job)]);rec=b.show(job)
            if b.field(rec,'JobState')!='PENDING':
                stable(item,rec);state['race_release_intent']=True;event(tx,f'{job}: start raced hold; restoring unchanged allocation')
                command(['scontrol','release',str(job)]);state['status']='started_race';event(tx,f'{job}: running allocation preserved')
                raise RuntimeError('start race; partial holds need fresh coordinated admission')
            stable(item,rec,held=True);state.update(status='owned_held',held_record=rec);event(tx,f'{job}: exact owned hold verified')
        tx['hold_scope_complete']=True;tx['status']='held_pending_activation';tx['budget_after_holds']=storage.storage_report();event(tx,'Six pending reservations deferred; all checkpoint files and ledgers preserved')
    print(json.dumps({'status':tx['status'],'safe_budget':tx['budget_after_holds']['allowed'],'target_ids':TARGETS}))


def activate():
    plan,tx=load();before_deadline(plan);require(tx.get('hold_started') and not tx.get('cpu_release_intent'),'hold package not attempted or activation attempted')
    cpu_record(plan,tx['cpu_job_id'],held=True)
    for item in plan['items']:
        if tx['jobs'].get(str(item['job_id']),{}).get('status')=='owned_held':stable(item,b.show(item['job_id']),held=True)
    tx['cpu_release_intent']=True;tx['status']='monitoring';event(tx,'Releasing registered CPU observer; six GPU jobs remain held pending full admission')
    command(['scontrol','release',str(tx['cpu_job_id'])]);tx['cpu_released']=True;event(tx,'CPU observer eligible')


def reconcile_release(item,state,tx):
    rec=b.show(item['job_id']);stable(item,rec)
    if b.field(rec,'JobState') in ('PENDING','RUNNING','CONFIGURING','COMPLETING','COMPLETED') and b.field(rec,'Reason') not in ('JobHeldUser','JobHeldAdmin','job_requeued_in_held_state'):
        state.update(status='released',release_record=rec);event(tx,f"{item['job_id']}: reconciled prior release acknowledgement");return True
    require(b.field(rec,'JobState')=='PENDING' and b.field(rec,'Reason')=='JobHeldUser','uncertain release identity/state changed')
    state['status']='release_uncertain';return False


def one_pass(plan,tx,*,apply):
    report={'at_utc':now().isoformat(),'deadline_utc':plan['deadline_utc'],'read_only':not apply,'jobs':[]}
    if now()>=datetime.fromisoformat(plan['deadline_utc']):
        report.update(status='deadline_reached',owned_hold_ids=[k for k,v in tx['jobs'].items() if v.get('status')!='released']);return report
    with admission_locks():
        if apply:reconcile_e124_pause(plan,tx)
        for item in plan['items']:
            state=tx['jobs'].get(str(item['job_id']),{})
            if state.get('hold_intent') and not state.get('status'):
                rec=b.show(item['job_id']);stable(item,rec)
                if b.field(rec,'JobState')=='PENDING' and b.field(rec,'Reason')=='JobHeldUser':
                    stable(item,rec,held=True)
                    if apply:state.update(status='owned_held',held_record=rec);event(tx,f"{item['job_id']}: reconciled prior pending hold acknowledgement")
                else:
                    require(b.field(rec,'Reason') not in HELD_REASONS|{'JobHeldAdmin'},'foreign hold during uncertain hold reconciliation')
                    if apply:state['status']='not_owned';event(tx,f"{item['job_id']}: hold absent; no repeated hold command")
        if tx.get('e124',{}).get('status')=='owned_held' and not tx.get('e124',{}).get('release_intent'):
            e124_stable(plan['e124_cpu'],b.show(E124_CPU),held=True,expected_restarts=tx['e124']['expected_restarts'])
            e124_gpu_holds(plan['e124_cpu']['gpu_ids'])
        uncertain=[int(k) for k,v in tx['jobs'].items() if v.get('release_intent') and v.get('status')!='released']
        for item in plan['items']:
            state=tx['jobs'].get(str(item['job_id']),{})
            if state.get('release_intent') and state.get('status')!='released' and apply:
                try:reconcile_release(item,state,tx)
                except Exception as error:state.update(status='manual_stop',error=repr(error));event(tx,f"{item['job_id']}: release requires manual reconciliation")
        uncertain=[int(k) for k,v in tx['jobs'].items() if v.get('release_intent') and v.get('status')!='released']
        base_budget=storage.storage_report(include_held_job_ids=uncertain);report['budget']=base_budget
        if uncertain:report['status']='release_requires_reconciliation';return report
        report['status']='monitoring'
        for item in plan['items']:
            key=str(item['job_id']);state=tx['jobs'].get(key,{})
            summary={'job_id':item['job_id'],'status':state.get('status','not_owned')};report['jobs'].append(summary)
            if state.get('status')!='owned_held':continue
            try:
                stable(item,b.show(item['job_id']),held=True);scientific(item)
                budget=storage.storage_report(include_held_job_ids=[item['job_id']]);summary.update(storage_allowed=budget['allowed'],margin_bytes=budget.get('margin_bytes'),blocked_reason=budget.get('blocked_reason'))
                if not budget['allowed']:continue
                if not apply:summary['status']='would_release';continue
                before_deadline(plan);require(runtime.runtime_fingerprints(plan['items'])==plan['runtime_fingerprints'],'frozen runtime changed')
                stable(item,b.show(item['job_id']),held=True);scientific(item)
                # Hash/science audits may take time; recompute the complete live budget last.
                budget=storage.storage_report(include_held_job_ids=[item['job_id']]);require(budget['allowed'],'headroom changed during final validation')
                before_deadline(plan);state['release_intent']=True;state['release_budget']=budget;event(tx,f"{item['job_id']}: full shared storage admits unchanged queued job; releasing owned hold")
                command(['scontrol','release',key]);reconcile_release(item,state,tx);summary['status']=state['status']
                break  # One release per pass; next pass sees all resulting writes.
            except Exception as error:
                summary.update(status='observation_error',error=repr(error))
                if apply:
                    if state.get('release_intent'):state['status']='release_uncertain'
                    elif isinstance(error,(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError,BlockingIOError)):state.update(status='owned_held',last_observation_error=repr(error))
                    else:state.update(status='manual_stop',error=repr(error))
                    event(tx,f"{item['job_id']}: retained receipt/hold for safe reconciliation: {type(error).__name__}")
        report['e124']=e124_restore(plan,tx,apply=apply)
        return report


def watch():
    from bounded_checkpoint_zip_metadata_20260911 import install
    install()
    plan,tx=load();require(tx.get('cpu_release_intent') and REG.exists(),'supervisor not armed')
    registration=json.loads(REG.read_text());own=os.environ.get('SLURM_JOB_ID')
    require(own==str(registration['job_id'])==str(tx['cpu_job_id']) and registration['plan_sha256']==digest(PLAN),'wrong CPU supervisor identity')
    with SINGLETON.open('a+') as singleton:
        fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
        while True:
            try:
                plan,tx=load();rec=cpu_record(plan,int(own),running=True)
                pending=tx.get('cpu_renewal_intent')
                if pending and not pending.get('acknowledged'):
                    require(int(b.field(rec,'Restarts'))==pending['before_restarts']+1,'CPU renewal restart differs')
                    pending['acknowledged']=True;event(tx,'Same-ID CPU renewal acknowledged; original deadline unchanged')
                report=one_pass(plan,tx,apply=True)
                atomic(ART/'status.json',report);atomic(ART/'ready.json',{'at_utc':now().isoformat(),'job_id':int(own),'plan_sha256':digest(PLAN),'singleton':True})
                if report['status']=='deadline_reached':
                    tx['status']='deadline_reached';event(tx,'Absolute deadline reached; remaining owned holds retained for manual review');return
                if all(v.get('status') in ('released','started_race','not_owned') for v in tx['jobs'].values()) and tx.get('e124',{}).get('status')=='released':
                    tx['status']='complete';event(tx,'All owned original GPU holds returned to scheduler and E124 CPU restored last');return
                if time.monotonic()-started>=plan['renew_after_seconds']:
                    before_deadline(plan);require(int(b.field(rec,'Restarts'))<plan['max_cpu_restarts'],'CPU restart cap reached')
                    tx['cpu_renewal_intent']={'before_restarts':int(b.field(rec,'Restarts')),'at_utc':now().isoformat()};event(tx,'Renewing only registered CPU ID; absolute deadline and GPU holds retained')
                    command(['scontrol','requeue',own]);return
            except Exception as error:
                atomic(ART/'status.json',{'at_utc':now().isoformat(),'status':'observation_error','error':repr(error),'no_scheduler_retry':True})
            time.sleep(plan['poll_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','submit','hold','activate','once','watch']);args=parser.parse_args();ART.mkdir(parents=True,exist_ok=True)
    if args.phase=='watch':watch();return
    with (ART/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='prepare':
            with admission_locks():prepare()
        elif args.phase=='once':
            plan,tx=load();print(json.dumps(one_pass(plan,tx,apply=False),indent=2))
        else:globals()[args.phase]()


if __name__=='__main__':main()
