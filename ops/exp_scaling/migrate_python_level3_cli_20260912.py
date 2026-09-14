#!/usr/bin/env python3
"""Replace zero-step neutral Python jobs with CLI-correct immutable runtimes."""
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager,ExitStack
import argparse,json,os,re,shlex,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import migrate_level3_neutral_v5 as parent
import e122_nonpython_burst_20260911 as budget_helper
c,old=parent.c,parent.old
ART=ROOT/'var/artifacts/python_level3_cli_recovery_20260912'
PLAN=ART/'migration_plan.json';COMMIT=ART/'committed.json';JOURNAL=ART/'e122_release_controller'
PARENT_SHA='3b5c13cdab5a4d7cd0ca71437736b1f8189358e945b7103a4791470c0b47b903'
FAILED={'31254498','31254499','31254500','31254501'}
SOURCE=Path(__file__).resolve()
read,sha,new,require,run=parent.read,parent.digest,parent.new,parent.require,parent.run

@contextmanager
def admission_locks():
    with ExitStack() as stack:
        for attempt in range(60):
            try:stack.enter_context(old.admission_locks());break
            except BlockingIOError:
                if attempt==59:raise
                time.sleep(.5)
        stack.enter_context(c.locked(c.JOURNAL_ROOT))
        yield

def old_held(row):
    jid=row['job_id'];cell=row['cell'];raw=run(['scontrol','show','job','-dd','-o',jid])
    require(c.field(raw,'JobState')=='PENDING' and c.field(raw,'Reason')=='JobHeldUser' and c.field(raw,'Priority')=='0','old job not held')
    require(c.field(raw,'UserId').endswith(f'({os.getuid()})'),'owner differs')
    require(not Path(cell['run_dir']).exists(),'old run has scientific output; cannot use zero-step recovery')
    if jid in FAILED:
        log=Path(c.field(raw,'StdErr'))
        require("invalid choice 'qwen_level3_python_factors_neutral_v1'" in log.read_text(),'missing exact CLI error')
    else:require(c.field(raw,'Restarts')=='0' and c.field(raw,'StartTime')=='Unknown','unregistered execution history')
    import prioritize_e118_capacity_20260905 as identity
    env=identity.exports(identity.submit_tokens(raw))
    require(env==cell['environment'],'old exported science changed')
    return raw

def make_cell(row):
    campaign=row['campaign'];cell=deepcopy(row['cell']);oldroot=Path(cell['environment']['OAT_ZERO_OPS_SNAPSHOT_ROOT']).parent
    runtime=ART/'runtime_v2'/campaign
    env={k:v.replace(str(oldroot),str(runtime)) for k,v in cell['environment'].items()}
    stamp=cell['run_stamp']+'_cli_v1';directory=cell['run_dir']+'_cli_v1'
    env.update(RUN_STAMP=stamp,SAVE_PATH=directory)
    command=[]
    for arg in cell['command']:
        if arg.startswith('--export='):arg='--export=ALL,'+','.join(k+'='+v for k,v in sorted(env.items()))
        elif arg.startswith('--job-name='):arg='--job-name='+stamp
        elif arg.startswith('--output=') or arg.startswith('--error='):arg=arg.replace(cell['run_stamp'],stamp)
        else:arg=arg.replace(str(oldroot),str(runtime))
        command.append(arg)
    require('--hold' in command,'successor must submit held')
    changes={k for k in env if env[k]!=cell['environment'][k]}
    require(changes=={'RUN_STAMP','SAVE_PATH','OAT_ZERO_OPS_SNAPSHOT_ROOT','OAT_ZERO_SOURCE_ROOT'},'unexpected environment changes')
    require(env['OAT_ZERO_SOURCE_ROOT']==str(runtime/'src'),'successor Python source differs')
    cell.update(environment=env,run_stamp=stamp,run_dir=directory,command=command)
    return cell

def prepare():
    require(not PLAN.exists(),'plan already exists')
    parent.install_controller(PARENT_SHA)
    admission=parent.admission()
    runtimes=read(ART/'runtime_prepared.json')
    require(len(runtimes)==2 and all('/runtime_v2/' in r['runtime'] for r in runtimes),'wrong runtime generation')
    test=ROOT/'paper/audits/paper_refresh_20260912/python_cli_runtime_tests.log'
    require('4 passed' in test.read_text() and 'failed' not in test.read_text(),'native CLI tests must pass')
    pins={str(p):sha(p) for p in [SOURCE,Path(parent.__file__),ROOT/'ops/prepare_python_level3_cli_runtime_20260912.py',ROOT/'tests/test_python_level3_cli_runtime_20260912.py',test,parent.PLAN,parent.COMMIT,parent.calibration.ART/'admission.json']}
    for r in runtimes:
        root=Path(r['runtime']);ident=root/'CLI_AMENDMENT_IDENTITY.json';require(sha(ident)==r['identity_sha256'],'runtime identity changed')
        d=read(ident);pins[str(ident)]=sha(ident)
        for rel,digest in d['inventory_sha256'].items():require(sha(root/rel)==digest,'runtime file changed');pins[str(root/rel)]=digest
    rows=[]
    with admission_locks():
        for row in read(parent.COMMIT)['replacements']:
            raw=old_held(row);cell=make_cell(row)
            require(not Path(cell['run_dir']).exists(),'successor output namespace exists')
            rows.append({'campaign':row['campaign'],'old_job_id':row['job_id'],'old_cell':row['cell'],'cell':cell,'old_held_record':raw,'resume_prior_release':row['job_id'] in FAILED})
    require(len(rows)==22 and sum(r['resume_prior_release'] for r in rows)==4,'replacement cohort differs')
    new(PLAN,{'schema':'neutral_python_cli_migration_v1','at':c.now(),'rows':rows,'source_pins':pins,'parent_migration_sha256':PARENT_SHA,'admission':admission,'scientific_change':'none; admit the already rendered neutral template in argument parsing/validation','persistent_cap':4,'resume_existing_slots_cap':13})
    return {'status':'prepared','plan_sha256':sha(PLAN),'replacements':22}

def validate(expected):
    require(sha(PLAN)==expected,'plan changed')
    plan=read(PLAN)
    for p,digest in plan['source_pins'].items():require(sha(p)==digest,'pinned file changed: '+p)
    parent.validate(PARENT_SHA)
    return plan

def fresh_held(jid,cell,campaign):
    raw=run(['scontrol','show','job','-dd','-o',jid])
    if campaign=='e122':old.launcher.audit_held_record(raw,int(jid),cell)
    else:
        require(c.field(raw,'JobState')=='PENDING' and c.field(raw,'Reason')=='JobHeldUser' and c.field(raw,'Restarts')=='0' and c.field(raw,'RunTime')=='00:00:00','E124 replacement not zero-step held')
        import prioritize_e118_capacity_20260905 as identity
        require(identity.exports(identity.submit_tokens(raw))==cell['environment'],'E124 export mismatch')
        require(c.field(raw,'MinMemoryNode')=='256G' and c.field(raw,'NumCPUs')=='8' and c.field(raw,'Account')=='mltheory','E124 resources changed')
    return raw

def migrate(expected):
    plan=validate(expected);parent.install_controller(PARENT_SHA)
    require(not (ART/'migration_claim.json').exists(),'migration already claimed; reconcile durable records')
    with admission_locks(),c.locked(parent.NEW_JOURNAL):
        for row in plan['rows']:old_held({'job_id':row['old_job_id'],'cell':row['old_cell']})
        new(ART/'migration_claim.json',{'at':c.now(),'plan_sha256':expected})
        replacements=[]
        for index,row in enumerate(plan['rows']):
            path=ART/'submissions'/str(index);new(path/'intent.json',{'row':row,'plan_sha256':expected})
            result=subprocess.run(row['cell']['command'],text=True,capture_output=True,env=old.launcher.clean_submit_environment(),timeout=60)
            new(path/'result.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            require(result.returncode==0 and re.fullmatch(r'\d+(?:;[^\s]+)?\s*',result.stdout),'submission failed/ambiguous; no retry')
            jid=result.stdout.strip().split(';')[0];raw=fresh_held(jid,row['cell'],row['campaign'])
            new(path/'held.json',{'job_id':jid,'scheduler_record':raw})
            replacements.append({**row,'job_id':jid,'held_record':raw})
            print(json.dumps({'submitted_held':jid,'index':index+1}),flush=True)
        for row in replacements:
            old_held({'job_id':row['old_job_id'],'cell':row['old_cell']})
            new(ART/'cancellations'/f"{row['old_job_id']}.intent.json",{'replacement':row['job_id'],'old_job_id':row['old_job_id']})
            result=subprocess.run(['scancel',row['old_job_id']],text=True,capture_output=True,timeout=45)
            new(ART/'cancellations'/f"{row['old_job_id']}.result.json",{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            require(result.returncode==0,'cancellation acknowledgement failed; inspect without retry')
        ledger=deepcopy(read(parent.ART/'e122_jobs.json'));mapping={r['old_job_id']:r for r in replacements if r['campaign']=='e122'}
        for row in ledger['runs']:
            if str(row['job_id']) in mapping:
                r=mapping[str(row['job_id'])];row.update({k:r['cell'][k] for k in c.CELL_FIELDS});row.update(job_id=int(r['job_id']),command=r['cell']['command'])
        new(ART/'e122_jobs.json',ledger)
        tx=deepcopy(read(parent.ART/'e124_transaction.json'))
        for key,row in tx['rows'].items():
            matches=[r for r in replacements if r['campaign']=='e124' and str(row['job_id'])==r['old_job_id']]
            if matches:row.update(job_id=int(matches[0]['job_id']),superseded_job_id=int(matches[0]['old_job_id']),cell=matches[0]['cell'])
        new(ART/'e124_transaction.json',tx)
        new(COMMIT,{'at':c.now(),'plan_sha256':expected,'replacements':replacements,'old_jobs_cancelled':22,'new_jobs_held':22})
    return {'status':'migrated','replacements':22}

def merged(expected):
    validate(expected);commit=read(COMMIT);require(commit['plan_sha256']==expected,'commit mismatch')
    campaign=deepcopy(parent.merged_campaign(PARENT_SHA));mapping={r['old_job_id']:r for r in commit['replacements'] if r['campaign']=='e122'}
    for job in campaign['jobs']:
        if job['job_id'] in mapping:
            r=mapping[job['job_id']];job.update(job_id=r['job_id'],cell=r['cell']);job['row'].update({k:r['cell'][k] for k in c.CELL_FIELDS});job['row']['job_id']=int(r['job_id'])
    campaign['binding'].update(cli_migration_plan_sha256=expected,cli_migration_commit_sha256=sha(COMMIT))
    return campaign

def install(expected,cap=4,candidates=None):
    parent.install_controller(PARENT_SHA)
    prior=parent.merged_campaign(PARENT_SHA);raw_reader=parent.ORIGINAL_READ_JOURNALS
    current=merged(expected);replaced={r['old_job_id'] for r in read(COMMIT)['replacements'] if r['campaign']=='e122'}
    def journals(root,binding,job_ids):
        legacy=parent.legacy_campaign()
        first=raw_reader(c.JOURNAL_ROOT,legacy['binding'],{j['job_id'] for j in legacy['jobs']})
        second=raw_reader(parent.NEW_JOURNAL,prior['binding'],{j['job_id'] for j in prior['jobs']})
        second={k:v for k,v in second.items() if k not in replaced}
        third=raw_reader(root,binding,job_ids)
        require(not set(first)&set(second) and not (set(first)|set(second))&set(third),'duplicate ownership')
        return {**first,**second,**third}
    c.read_journals=journals;c.load_campaign=lambda *_a,**_k:merged(expected);c.MAX_ACTIVE=cap
    base=budget_helper.base;old_registry=base.canonical_registry
    def registry():
        result=old_registry()
        for r in read(COMMIT)['replacements']:
            require(r['job_id'] not in result,'duplicate registered writer')
            result[r['job_id']]={'run_dir':r['cell']['run_dir'],'model_choice':'05b' if r['campaign']=='e122' else '7b','ledger':str(COMMIT)}
        return result
    base.canonical_registry=registry;budget_helper.CAP=cap
    budget_helper.ARRAYS['31254520']=(7,ROOT/'var/artifacts/modebench_scale_composite_v1/revision_development/frozen_transport/worker.slurm',ROOT/'var/artifacts/modebench_scale_composite_v1/revision_development/plan.json')
    def status(campaign,root):
        result=old.ORIGINAL_STATUS(campaign,root)
        if candidates is not None and result['blocked_reason'] is None:result['next_job_id']=next((j for j in candidates if j in result['staged_held_job_ids']),None)
        if result['next_job_id']:
            budget=budget_helper.storage_budget(campaign,[result['next_job_id']]);result['shared_storage']=budget
            if not budget['allowed']:result.update(next_job_id=None,blocked_reason='waiting_shared_storage')
        return result
    c.status=status
    return current

def resume_four(expected):
    ids=[r['job_id'] for r in read(COMMIT)['replacements'] if r['resume_prior_release']]
    install(expected,13,ids)
    for i in range(4):
        with admission_locks():result=c.advance_once(old.fresh_args(),old.launcher,root=JOURNAL)
        new(ART/f'resume_{i}.json',result)
        if not result.get('last_release_job_id'):break
        print(json.dumps({'released':result['last_release_job_id']}),flush=True)
    return {'resumed':[read(p)['last_release_job_id'] for p in sorted(ART.glob('resume_*.json')) if read(p).get('last_release_job_id')]}

def watch(expected,once=False):
    install(expected)
    while True:
        with admission_locks():result=c.status(merged(expected),JOURNAL) if once else c.advance_once(old.fresh_args(),old.launcher,root=JOURNAL)
        old.write(ART/'e122_status.json',result)
        print(json.dumps({k:result.get(k) for k in ['observed_at','blocked_reason','running','issues','needs_operator_review_job_ids','last_release_job_id']}),flush=True)
        if once or result['issues'] or result['needs_operator_review_job_ids']:return result
        time.sleep(60)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','migrate','resume-four','status','watch']);p.add_argument('--plan-sha256');a=p.parse_args()
    result=prepare() if a.action=='prepare' else migrate(a.plan_sha256) if a.action=='migrate' else resume_four(a.plan_sha256) if a.action=='resume-four' else watch(a.plan_sha256,a.action=='status')
    print(json.dumps(result if a.action not in ['watch','status'] else {'status_written':True}),flush=True)
if __name__=='__main__':main()
