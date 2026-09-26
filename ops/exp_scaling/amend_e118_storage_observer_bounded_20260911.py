#!/usr/bin/env python3
"""Audited same-ID CPU-only bounded ZIP/memory amendment; science untouched."""
from __future__ import annotations
import argparse,copy,fcntl,json,os,time
from pathlib import Path
import guard_e118_e122_storage_handoff_20260911 as g
ART=g.ART/'bounded_zip_memory_amendment'
PLAN=ART/'amendment_plan.json';TX=ART/'amendment_transaction.json'
SOURCE=Path(g.__file__).resolve()
HELPER=SOURCE.with_name('bounded_checkpoint_zip_metadata_20260911.py')
PROTOCOL=g.ROOT/'paper/preregistration/e118_storage_bounded_zip_memory_20260911.md'
JOB=31220594


def write_bytes(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def event(tx,label):tx.setdefault('events',[]).append({'at_utc':g.now().isoformat(),'event':label});g.atomic(TX,tx)


def no_intents(tx):
    g.require(tx['cpu_job_id']==JOB and tx['cpu_released'],'wrong CPU owner')
    g.require(not tx.get('cpu_renewal_intent'),'CPU renewal outstanding')
    g.require(all(v['status'] in ('owned_held','released') and (not v.get('release_intent') or v['status']=='released') for v in tx['jobs'].values()),'unresolved GPU release')
    g.require(tx['e124']['status']=='owned_held' and not tx['e124'].get('release_intent'),'E124 pause/restore uncertain')


def preserve_holds(plan,tx):
    no_intents(tx)
    for item in plan['items']:
        if tx['jobs'][str(item['job_id'])]['status']=='owned_held':g.stable(item,g.b.show(item['job_id']),held=True)
    g.e124_stable(plan['e124_cpu'],g.b.show(g.E124_CPU),held=True,expected_restarts=tx['e124']['expected_restarts'])
    g.e124_gpu_holds(plan['e124_cpu']['gpu_ids'])


def prepare():
    g.require(not PLAN.exists(),'amendment already prepared')
    plan,tx=g.load();preserve_holds(plan,tx);rec=g.cpu_record(plan,JOB)
    original={}
    for path in [SOURCE,g.PLAN,g.REG,g.TX,g.ART/'supervisor.slurm']:
        dest=ART/('original_'+path.name);write_bytes(dest,path.read_bytes());original[str(path)]={'archive':str(dest),'sha256':g.digest(dest)}
    text=SOURCE.read_text()
    before="'Partition':'lowprio','MinMemoryNode':'2G',"
    after="'Partition':'lowprio','MinMemoryNode':plan.get('cpu_memory_override','2G'),"
    g.require(text.count(before)==1,'CPU profile edit no longer singular');text=text.replace(before,after)
    before='def watch():\n    plan,tx=load();'
    after='def watch():\n    from bounded_checkpoint_zip_metadata_20260911 import install\n    install()\n    plan,tx=load();'
    g.require(text.count(before)==1,'watch entry edit no longer singular');text=text.replace(before,after)
    staged=ART/'bounded_guard_source.py';write_bytes(staged,text.encode());compile(text,str(SOURCE),'exec')
    amendment={'schema':'e118-storage-bounded-zip-memory-amendment-v1','prepared_at_utc':g.now().isoformat(),'job_id':JOB,'deadline_utc':plan['deadline_utc'],'before_cpu':rec,'original':original,'staged_source':str(staged),'staged_source_sha256':g.digest(staged),'cpu_memory_before':'2G','cpu_memory_after':'8G','cpu_submit_tokens':plan['cpu_submit_tokens'],'pins':{str(p.resolve()):g.digest(p) for p in [Path(__file__),HELPER,PROTOCOL,ART/'tests.json']},'gpu_mutations':False}
    g.atomic(PLAN,amendment)
    nextplan=copy.deepcopy(plan);nextplan['cpu_memory_override']='8G';nextplan['pins'][str(SOURCE)]=g.digest(staged);nextplan['pins'][str(HELPER)]=g.digest(HELPER);nextplan['pins'][str(PROTOCOL)]=g.digest(PROTOCOL)
    nextplan.setdefault('operational_amendments',[]).append({'plan':str(PLAN),'sha256':g.digest(PLAN),'original_guard_archive':original[str(SOURCE)]['archive'],'original_plan_archive':original[str(g.PLAN)]['archive']})
    g.atomic(ART/'updated_observer_plan.json',nextplan)
    print(json.dumps({'prepared':str(PLAN),'sha256':g.digest(PLAN),'scheduler_mutations':False}))


def load():
    p=json.loads(PLAN.read_text());g.require(all(g.digest(x)==h for x,h in p['pins'].items()),'amendment input changed')
    g.require(g.digest(p['staged_source'])==p['staged_source_sha256'],'staged source changed')
    g.require(all(g.digest(proof['archive'])==proof['sha256'] for proof in p['original'].values()),'historical archive changed')
    t=json.loads(TX.read_text()) if TX.exists() else {'plan_sha256':g.digest(PLAN),'events':[]}
    g.require(t['plan_sha256']==g.digest(PLAN),'amendment binding changed');return p,t


def pause():
    p,t=load();g.require(not t.get('pause_intent'),'CPU pause already attempted; reconcile exact receipt')
    original,tx=g.load();preserve_holds(original,tx);rec=g.cpu_record(original,JOB)
    g.require(g.b.field(rec,'JobState') in ('RUNNING','PENDING') and g.b.field(rec,'Priority')!='0','CPU cannot be paused from current state')
    mode='hold' if g.b.field(rec,'JobState')=='PENDING' else 'requeuehold'
    t.update(pause_intent=True,pause_mode=mode,before=rec,expected_restarts=int(g.b.field(rec,'Restarts'))+(mode=='requeuehold'));event(t,'Persisted one exact CPU-only pause intent')
    g.command(['scontrol',mode,str(JOB)]);t['pause_command_acknowledged']=True;event(t,'CPU pause command acknowledged; observe before promotion')


def stopped(p,t):
    rec=g.b.show(JOB);g.require(g.b.submit_tokens(rec)==p['cpu_submit_tokens'],'CPU submitted command changed')
    g.require(g.b.field(rec,'JobState')=='PENDING' and g.b.field(rec,'Reason') in g.HELD_REASONS and g.b.field(rec,'Priority')=='0' and g.b.field(rec,'NodeList') in ('','(null)'),'CPU is not inactive in exact owned hold')
    g.require(int(g.b.field(rec,'Restarts'))==t['expected_restarts'],'CPU restart acknowledgement changed');return rec


def promote():
    p,t=load();g.require(t.get('pause_command_acknowledged') and not t.get('memory_intent'),'pause absent or promotion attempted; reconcile')
    rec=stopped(p,t)
    for path,proof in p['original'].items():g.require(g.digest(path)==proof['sha256'],'original artifact changed before stopped handoff: '+path)
    original,tx=g.load();preserve_holds(original,tx)
    with g.admission_locks():
        stopped(p,t);t['memory_intent']=True;event(t,'Persisted stopped CPU memory2G to8G amendment intent')
        g.command(['scontrol','update',f'JobId={JOB}','MinMemoryNode=8192']);after=stopped(p,t)
        keys=[k for k in g.FIELDS if k not in ('MinMemoryNode','TresPerNode')]+['Dependency','MinCPUsNode','TresPerTask']
        g.require(all(g.field(after,k)==g.field(rec,k) for k in keys),'other CPU resource changed')
        g.require(g.b.field(after,'MinMemoryNode')=='8G' and g.b.field(after,'ReqTRES')=='cpu=2,mem=8G,node=1','CPU actual memory differs')
        t['after_memory']=after;t['promotion_intent']=True;event(t,'Memory8G confirmed; promoting only observer operational pins while CPU held')
        nextplan=copy.deepcopy(original);nextplan['cpu_memory_override']='8G'
        nextplan['pins'][str(SOURCE)]=p['staged_source_sha256'];nextplan['pins'][str(HELPER)]=p['pins'][str(HELPER)];nextplan['pins'][str(PROTOCOL)]=p['pins'][str(PROTOCOL)]
        nextplan.setdefault('operational_amendments',[]).append({'plan':str(PLAN),'sha256':g.digest(PLAN),'original_guard_archive':p['original'][str(SOURCE)]['archive'],'original_plan_archive':p['original'][str(g.PLAN)]['archive']})
        g.require(nextplan==json.loads((ART/'updated_observer_plan.json').read_text()),'staged plan differs from deterministic authorized amendment')
        g.require(nextplan['deadline_utc']==p['deadline_utc'],'original deadline changed')
        write_bytes(SOURCE,Path(p['staged_source']).read_bytes());g.atomic(g.PLAN,nextplan)
        registration=json.loads(g.REG.read_text());registration['plan_sha256']=g.digest(g.PLAN);registration['resources']['MinMemoryNode']='8G';registration['operational_amendment']=str(PLAN);g.atomic(g.REG,registration)
        tx['plan_sha256']=g.digest(g.PLAN);tx.setdefault('operational_amendments',[]).append({'plan':str(PLAN),'sha256':g.digest(PLAN),'at_utc':g.now().isoformat()});g.atomic(g.TX,tx)
        t.update(promoted=True,updated_observer_plan_sha256=g.digest(g.PLAN),updated_source_sha256=g.digest(SOURCE),gpu_states={k:v['status'] for k,v in tx['jobs'].items()});event(t,'Observer source/plan/registration/transaction rebound; original archives preserved')
    print(json.dumps({'promoted':True,'job_id':JOB,'held':True,'memory':'8G','deadline_utc':p['deadline_utc']}))


def release():
    p,t=load();g.require(t.get('promoted') and not t.get('release_intent'),'promotion absent or release attempted; reconcile')
    stopped(p,t);current,tx=g.load();preserve_holds(current,tx);g.cpu_record(current,JOB)
    t['release_intent']=True;event(t,'Persisted same-ID CPU release after bounded-source and8G verification')
    g.command(['scontrol','release',str(JOB)]);rec=g.cpu_record(current,JOB);g.require(g.b.field(rec,'Priority')!='0','CPU release uncertain')
    t.update(released=True,after_release=rec);event(t,'Same CPU eligible with bounded metadata reads; all GPU holds/science preserved')
    print(json.dumps({'released':True,'job_id':JOB,'state':g.b.field(rec,'JobState')}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','pause','promote','release']);args=ap.parse_args();ART.mkdir(parents=True,exist_ok=True)
    with (ART/'controller.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.phase]()


if __name__=='__main__':main()
