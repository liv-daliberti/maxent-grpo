#!/usr/bin/env python3
"""Finite capacity census after CLI recovery; release only native 64GiB non-Python jobs."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import resume_python_level3_cli_storage_20260912 as recovery
import expand_e122_finite_20260912 as capacity
m=recovery.m;c=m.c;old=m.old
ART=ROOT/'var/artifacts/e122_after_cli_capacity_20260912'
PLAN=ART/'plan.json';SOURCE=Path(__file__).resolve()
REG_SHA='24ceaf01e351b4bfe7b7934daa557a4e0305d08115a5fd8721337cdb879b1f2b'
read,sha,new,require,run=m.read,m.sha,m.new,m.require,m.run

def setup(cap,candidates):
    recovery.install(REG_SHA)
    return m.install(recovery.PLAN_SHA,cap,candidates)

def prepare():
    require(not PLAN.exists(),'plan already exists')
    candidates=[];campaign=setup(100,candidates)
    with m.admission_locks(),c.locked(m.JOURNAL):
        status=c.status(campaign,m.JOURNAL)
        require(not status['issues'] and not status['needs_operator_review_job_ids'] and not status['unknown'],'campaign requires reconciliation')
        nodes=capacity.node_capacity();pending=[]
        for jid in status['reserved_unfinished_job_ids']:
            raw=run(['scontrol','show','job','-dd','-o',jid])
            if c.field(raw,'JobState')=='PENDING':
                memory=c.field(raw,'MinMemoryNode');require(memory.endswith('G'),'unknown memory unit')
                pending.append({'job_id':jid,'slots':(int(memory[:-1])+63)//64,'record':raw})
        available=max(0,sum(n['free_64g_slots'] for n in nodes)-sum(r['slots'] for r in pending))
        eligible=[j for j in campaign['jobs'] if j['job_id'] in status['staged_held_job_ids'] and j['cell']['domain']!='python_factors' and '--mem=64G' in j['cell']['command']]
        selected=eligible[:available];require(selected,'no eligible capacity')
        candidates.extend(j['job_id'] for j in selected)
        cap=status['reserved_unfinished_slots']+len(candidates);c.MAX_ACTIVE=cap;m.budget_helper.CAP=cap
        held={j['job_id']:old.launcher.audit_held(int(j['job_id']),j['cell']) for j in selected}
        budget=m.budget_helper.storage_budget(campaign,candidates);require(budget['allowed'],'insufficient shared storage')
        paths=[SOURCE,Path(recovery.__file__),recovery.REG,Path(m.__file__),m.PLAN,m.COMMIT,Path(capacity.__file__)]
        new(PLAN,{'schema':'e122_after_cli_finite_capacity_v1','at':c.now(),'authorization':'User requested as many E122 starts as available GPUs permit; preserve scientific recipes and qualification gates.','binding':campaign['binding'],'initial_status':status,'capacity':nodes,'pending_reservations':pending,'candidate_job_ids':candidates,'max_unfinished_slots':cap,'held_records':held,'storage':budget,'source_pins':{str(p):sha(p) for p in paths},'persistent_cap':4,'no_automatic_replenishment':True})
    return {'plan_sha256':sha(PLAN),'additional':len(candidates),'cap':cap,'candidates':candidates,'storage_margin_gib':budget['margin_bytes']/1024**3}

def validate(expected):
    require(sha(PLAN)==expected,'plan changed');plan=read(PLAN)
    for p,h in plan['source_pins'].items():require(sha(Path(p))==h,'source changed: '+p)
    require(plan['max_unfinished_slots']==plan['initial_status']['reserved_unfinished_slots']+len(plan['candidate_job_ids']),'cap differs')
    return plan

def route(jid,cell):
    raw=run(['scontrol','show','job','-dd','-o',jid])
    if c.field(raw,'JobState')!='PENDING':return
    require(c.field(raw,'Priority')!='0' and c.field(raw,'Reason') not in ['JobHeldUser','JobHeldAdmin'],'preserve existing hold')
    nodes=set(run(['scontrol','show','hostnames',c.field(raw,'ReqNodeList')]).split())
    require(nodes and nodes<=capacity.POOL,'unexpected route')
    if nodes==capacity.POOL:return
    node=run(['scontrol','show','node','node208']);require(not any(x in c.field(node,'State') for x in ['DOWN','DRAIN','FAIL','MAINT']),'node208 unhealthy')
    import prioritize_e118_capacity_20260905 as identity
    require(identity.exports(identity.submit_tokens(raw))==cell['environment'],'science environment differs')
    path=ART/'routes'/jid
    command=['scontrol','update','JobId='+jid,'ReqNodeList='+','.join(sorted(capacity.POOL))]
    new(path/'intent.json',{'before':raw,'command':command,'at':c.now()})
    result=subprocess.run(command,text=True,capture_output=True,timeout=45)
    new(path/'result.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    require(result.returncode==0,'route acknowledgement failed; reconcile before retry')
    after=run(['scontrol','show','job','-dd','-o',jid])
    require(identity.submit_tokens(raw)==identity.submit_tokens(after),'submitted command changed')
    for key in capacity.PRESERVE:require(c.field(raw,key)==c.field(after,key),'resource/science changed: '+key)
    require(set(run(['scontrol','show','hostnames',c.field(after,'ReqNodeList')]).split())==capacity.POOL,'route readback differs')
    new(path/'verified.json',{'after':after,'science_and_resources_unchanged':True})

def release(expected):
    plan=validate(expected);campaign=setup(plan['max_unfinished_slots'],plan['candidate_job_ids'])
    require(not (ART/'claim.json').exists(),'already claimed; reconcile durable journals')
    jobs={j['job_id']:j for j in campaign['jobs']};released=[]
    with m.admission_locks():
        new(ART/'claim.json',{'at':c.now(),'plan_sha256':expected})
        for index in range(len(plan['candidate_job_ids'])):
            validate(expected)
            result=c.advance_once(old.fresh_args(),old.launcher,root=m.JOURNAL)
            new(ART/f'release_{index}.json',result)
            jid=result.get('last_release_job_id')
            if not jid:break
            require(jid in plan['candidate_job_ids'],'unexpected release')
            released.append(jid);print(json.dumps({'released':jid,'count':len(released)}),flush=True)
            require(not result['issues'] and not result['needs_operator_review_job_ids'],'release needs reconciliation')
            route(jid,jobs[jid]['cell'])
    new(ART/'result.json',{'at':c.now(),'released_job_ids':released,'count':len(released),'plan_sha256':expected,'persistent_cap':4})
    return {'released':released,'count':len(released)}

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','release']);p.add_argument('--plan-sha256');a=p.parse_args()
    print(json.dumps(prepare() if a.action=='prepare' else release(a.plan_sha256)),flush=True)
if __name__=='__main__':main()
