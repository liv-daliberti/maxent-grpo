#!/usr/bin/env python3
"""Audited same-ID128GiB repair for two explicitly bounded E119 Pantry cells."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
import pickletools
from pathlib import Path
import subprocess
import time
import zipfile
import recover_e119_memory_pressure_20260905 as base
import recover_e119_pantry_s43_memory128_20260909 as prior
import recover_terminal_timeouts_20260908 as recovery
import prioritize_e118_capacity_20260905 as cli

ROOT=base.ROOT
ART=ROOT/'var/artifacts/e119_pantry_s43_guarded_memory128_20260909'
PROTOCOL=ROOT/'paper/preregistration/e119_pantry_s43_guarded_memory128_20260909.md'
TARGETS={31037827:('drgrpo',768,ART/'diagnosis/31037827.json'),
         31048179:('replay_maxrl',960,ART/'diagnosis/31048179.json')}
PRESERVE=tuple(dict.fromkeys((*base.PRESERVE,'QOS','Nice','ReqNodeList','Features','UserId')))


def require(ok,message):
    if not ok:raise RuntimeError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def command(args,*,check=True,timeout=45):
    return subprocess.run(args,capture_output=True,text=True,check=check,timeout=timeout)


def call(*args):return command(list(args)).stdout.strip()

def show(job):return call('scontrol','show','job','-dd','-o',str(job))

def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    tmp.replace(path);fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def event(path,tx,label):
    tx.setdefault('events',[]).append({'at':datetime.now(timezone.utc).isoformat(),'event':label});write(path,tx)


def identity(job):
    require(job in TARGETS,'unreviewed target');run=base.identities()[job];arm,_,_=TARGETS[job]
    require((run['domain'],run['arm'],int(run['seed']))==('pantry_plan',arm,43),'scientific identity changed')
    return {k:run[k] for k in ('domain','arm','seed','run_stamp','run_dir','original_job_id','effective_job_id')}


def replay_metadata(job,name,metadata):
    if TARGETS[job][0]=='replay_maxrl' and name.endswith('_model_states.pt'):
        require(any(arg=='online_canonical_bank_state' for _,arg,_ in pickletools.genops(metadata)),
                'Re:Max checkpoint lacks saved replay bank state')


def checkpoint(job,run):
    d=base.timing_and_checkpoint(job,run);step=TARGETS[job][1]
    require(d['checkpoint_step']==step and d['current_step']==step-1 and d['unsaved_steps']==0,'checkpoint boundary changed; preserve current work')
    require(not d['rejected_checkpoints'] and not d['fresh_restart'],'partial or missing checkpoint')
    require(d['saved_counter_validation']['saved_counters']==dict.fromkeys(('global_steps','global_step','prompt_batches_consumed_total'),step),'saved counters disagree')
    files={}
    for p in sorted(Path(d['checkpoint']).glob('*.pt')):
        s=p.stat()
        with zipfile.ZipFile(p) as z:
            info=next(i for i in z.infolist() if i.filename.endswith('/data.pkl'))
            require(info.file_size<4*1024*1024,'unexpectedly large pickle metadata')
            metadata=z.read(info)
        replay_metadata(job,p.name,metadata)
        files[p.name]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'pickle_sha256':hashlib.sha256(metadata).hexdigest()}
    return {'detail':d,'files':files}


def profile(plan,record,memory=None,same_attempt=False):
    require(cli.field(record,'JobId')==str(plan['job_id']),'job ID changed')
    require(base.submitline(record)==base.submitline(plan['before']),'original scientific submission changed')
    env=base.exports(record);run=plan['identity']
    require(env.get('RUN_STAMP')==run['run_stamp'] and Path(env.get('SAVE_PATH','')).resolve()==Path(run['run_dir']).resolve()
            and env.get('OAT_ZERO_AUTO_RESUME')=='1','scheduler run/resume exports do not match scientific ledger')
    for k in PRESERVE:require(cli.field(record,k)==cli.field(plan['before'],k),'resource changed: '+k)
    require(cli.field(record,'NumNodes') in {'1','1-1'},'node count changed')
    if memory:require(cli.field(record,'MinMemoryNode')==memory,'memory differs')
    if same_attempt:
        for k in ('JobState','NodeList','StartTime','Restarts'):require(cli.field(record,k)==cli.field(plan['before'],k),'allocation changed: '+k)


def audit(plan):
    require(identity(plan['job_id'])==plan['identity'],'authoritative scientific mapping changed')
    for p,h in plan['pins'].items():require(digest(p)==h,'frozen input changed: '+p)
    require(prior.runtime_fingerprints(plan['before'])==plan['runtime_fingerprints'],'scientific runtime drift')
    current=checkpoint(plan['job_id'],plan['identity'])
    require(current['files']==plan['checkpoint']['files'],'saved checkpoint shard changed')
    require(current['detail']['checkpoint']==plan['checkpoint']['detail']['checkpoint'],'checkpoint path changed')
    return current


def sole_writer(plan):
    actual=recovery.active_writers().get(str(Path(plan['identity']['run_dir']).resolve()),set())
    require(actual<={plan['job_id']},'duplicate same-cell writer: '+str(actual))


def owned_hold(plan,record):
    profile(plan,record)
    require(cli.field(record,'JobState')=='PENDING' and cli.field(record,'Priority')=='0'
            and cli.field(record,'Reason')=='job_requeued_in_held_state','exact owned requeue hold missing; poll and reconcile without resending')
    require(int(cli.field(record,'Restarts'))==int(cli.field(plan['before'],'Restarts'))+1,'unexpected restart count')


def validated_pressure(job,outer):
    require(outer.get('job_id')==job and outer.get('returncode')==0,'pressure probe target or result differs')
    pressure=json.loads(outer['stdout']);require(pressure.get('job_id')==job,'pressure payload target differs')
    require(pressure['high_events_delta']>0,'no live memory throttling')
    for sample in (pressure['before'],pressure['after']):
        require(sample.get('job_id')==job and sample.get('cgroup_path')==f'/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{job}',
                'pressure cgroup is not the exact target allocation')
        require(sample['memory.high']==96*2**30 and 96<sample['noncache_gib']<116,'pressure or target headroom differs')
        require(sample['events']['oom']==sample['events']['oom_kill']==0,'OOM requires separate diagnosis')
    return pressure


def prepare(job):
    directory=ART/str(job);p=directory/'plan.json';require(not p.exists() and not (directory/'transaction.json').exists(),'existing immutable preparation')
    before=show(job);run=identity(job);detail=checkpoint(job,run)
    require(cli.field(before,'JobState')=='RUNNING' and cli.field(before,'MinMemoryNode')=='96G','expected running96GiB allocation missing')
    require(cli.field(before,'Account')=='allcs' and cli.field(before,'Partition')=='cs','owner route changed')
    require(cli.field(before,'NodeList')=='node205' and cli.field(before,'TimeLimit')=='1-12:00:00','reviewed allocation changed')
    require(cli.field(before,'UserId').endswith(f'({os.getuid()})'),'job ownership differs')
    require(detail['detail']['metric_age_seconds']>=900,'cell is making recent progress; preserve it')
    evidence=Path(TARGETS[job][2]);require(time.time()-evidence.stat().st_mtime<1800,'refresh pressure evidence')
    pressure=validated_pressure(job,read(evidence))
    pins=[Path(__file__).resolve(),PROTOCOL,evidence,Path(base.__file__),Path(prior.__file__),Path(recovery.__file__),Path(cli.__file__),Path(base.checked_checkpoint.__code__.co_filename),Path(base.select_latest_checkpoint.__code__.co_filename),base.campaign.E119_LEDGER]
    plan={'schema':'e119-pantry-s43-checkpoint-memory-v1','job_id':job,'created_at':datetime.now(timezone.utc).isoformat(),'before':before,'identity':run,
          'checkpoint':detail,'target_memory':'128G','pins':{str(x):digest(x) for x in pins},'pressure':pressure,
          'runtime_fingerprints':prior.runtime_fingerprints(before),'node_before':call('scontrol','show','node','-o','node205'),
          'queue_tradeoff':'same account/partition/node pool; scheduling and automatic owner-priority preemption are left to Slurm'}
    profile(plan,before,'96G',same_attempt=True);sole_writer(plan);audit(plan);write(p,plan)
    print(json.dumps({'prepared':job,'plan_sha256':digest(p),'checkpoint':TARGETS[job][1],'scheduler_mutations':False}),flush=True)


def apply(job):
    directory=ART/str(job);p=directory/'plan.json';t=directory/'transaction.json';plan=read(p)
    tx=read(t) if t.exists() else {'plan_sha256':digest(p),'events':[]}
    require(tx['plan_sha256']==digest(p),'plan changed')
    if tx.get('released'):
        profile(plan,show(job),'128G');print(json.dumps({'already_released':job}));return
    current=show(job);profile(plan,current)
    if tx.get('release_intent') and cli.field(current,'Priority')!='0':
        profile(plan,current,'128G');require(cli.field(current,'JobState') in {'RUNNING','PENDING','CONFIGURING'},'ambiguous release state')
        require(int(cli.field(current,'Restarts'))==int(cli.field(plan['before'],'Restarts'))+1,'restart drift after release')
        tx.update(released=True,after=current);event(t,tx,'Reconciled release without repeating mutation');return
    audit(plan)
    if not tx.get('hold_intent'):
        profile(plan,current,'96G',same_attempt=True);sole_writer(plan)
        archive=directory/'before_stop';archive.mkdir(exist_ok=True)
        tx['archives']=base.archive(job,current,plan['checkpoint']['detail'],archive)
        audit(plan);profile(plan,show(job),'96G',same_attempt=True)
        tx['hold_intent']=True;event(t,tx,'Persisted exact same-ID requeuehold intent; never repeat uncertain request')
        command(['scontrol','requeuehold',str(job)])
    current=show(job);owned_hold(plan,current);audit(plan)
    memory=cli.field(current,'MinMemoryNode');require(memory in {'96G','128G'},'unreviewed memory value')
    if memory=='96G':
        require(not tx.get('memory_intent'),'unacknowledged unchanged memory edit; inspect manually')
        tx['memory_intent']=True;event(t,tx,'Persisted owned-held memory increase96GiB to128GiB')
        command(['scontrol','update',f'JobId={job}','MinMemoryNode=131072'])
    current=show(job);owned_hold(plan,current);profile(plan,current,'128G');audit(plan);sole_writer(plan)
    tx['held_after']=current
    if not tx.get('release_intent'):
        tx['release_intent']=True;event(t,tx,'Persisted release intent after immutable checkpoint/science/profile audit')
        command(['scontrol','release',str(job)])
    current=show(job);profile(plan,current,'128G')
    require(cli.field(current,'JobState') in {'RUNNING','PENDING','CONFIGURING'} and cli.field(current,'Priority')!='0','release not acknowledged; reconcile without repeating')
    require(int(cli.field(current,'Restarts'))==int(cli.field(plan['before'],'Restarts'))+1,'restart drift after release')
    tx.update(released=True,after=current);event(t,tx,'Same-ID128GiB recovery released; checkpoint and scientific workload preserved')
    print(json.dumps({'released':job,'state':cli.field(current,'JobState'),'checkpoint':TARGETS[job][1]}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','apply']);p.add_argument('--job-id',type=int,choices=TARGETS,default=31037827);a=p.parse_args()
    cli.command=command;directory=ART/str(a.job_id);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'operation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='prepare':prepare(a.job_id)
        else:apply(a.job_id)


if __name__=='__main__':main()
