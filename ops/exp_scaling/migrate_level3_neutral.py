#!/usr/bin/env python3
"""Migrate only the 22 unstarted Python L3 cells after neutral confirmation.

Original plans, ledgers, runtime snapshots and release intents remain intact.
Historical source hashes resolve against exact original snapshot bytes when a
working-tree template has acquired the new renderer. This is explicit archival
provenance, never a difficulty certificate for the neutral task.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import calibrate_modebench_level3_neutral as calibration
import resume_e122_shared_release_20260911 as old
import e122_storage_with_modebench_array_20260911 as storage
import modebench_array_throttle2_storage_20260911 as throttle
c=old.c
ART=ROOT/'var/artifacts/python_level3_neutral_migration_20260911'
PLAN=ART/'plan.json'
COMMIT=ART/'committed.json'
SOURCE=Path(__file__).resolve()
PREPARED=ROOT/'artifacts/modebench_level3_neutral_default_20260911/successor_plans.json'
TEMPLATE=ROOT/'src/oat_drgrpo/templates.py'
OLD_TEMPLATE_SHA='6af7522ff71e02dbe46867a722288c0681f7136ca8d297b59c3c9a55fd30b123'
ORIGINAL_TEMPLATE=ROOT/'var/artifacts/source_snapshots/e122_level3_6caff9b52d509a86499f/src/oat_drgrpo/templates.py'
E122_PLAN=ROOT/'var/artifacts/e122_level3_factorial_plan.json'
E124_PLAN=ROOT/'var/artifacts/e124_qwen7b_three_level/plan.json'
E124_TX=ROOT/'var/artifacts/e124_qwen7b_three_level/transaction.json'
NEW_JOURNAL=ART/'e122_release_controller'
read,digest,new,require=calibration.read,calibration.digest,calibration.new,calibration.require
ORIGINAL_LOAD=c.load_campaign
ORIGINAL_READ_JOURNALS=c.read_journals
ORIGINAL_VERIFY_PINS=old.launcher.common.verify_pins


def run(argv):
    p=subprocess.run(list(map(str,argv)),text=True,capture_output=True,timeout=60)
    require(p.returncode==0, 'command failed: '+shlex.join(list(map(str,argv)))+' '+p.stderr)
    return p.stdout


def verify_historical_pins(files,trees=None):
    require(digest(ORIGINAL_TEMPLATE)==OLD_TEMPLATE_SHA,'archived historical template bytes changed')
    for path,sha in files.items():
        actual=ORIGINAL_TEMPLATE if Path(path)==TEMPLATE and sha==OLD_TEMPLATE_SHA else Path(path)
        require(digest(actual)==sha,'historical source/evidence changed: '+path)
    for directory,expected in (trees or {}).items():
        actual=sorted(str(p.resolve()) for p in Path(directory).rglob('*') if p.is_file())
        require(actual==expected and all(p in files for p in actual),'historical directory inventory changed')


@contextmanager
def historical_e122():
    """Authenticate the completed immutable proof using its archived sources."""
    proof=deepcopy(read(E122_PLAN)['admission_proof'])
    verify_historical_pins(proof['files_sha256'],proof['directory_files'])
    report=read(proof['path'])
    require(digest(proof['path'])==proof['sha256'] and proof['status']=='matched_fixed_reference'
            and report['all_five_domains_complete'] and not report['errors']
            and all(r['observed_approximate_match'] and all(r['within_tolerance'].values()) for r in report['domains'].values()), 'historical matched report changed')
    prior=old.launcher.admission_proof
    old.launcher.admission_proof=lambda **kwargs:deepcopy(proof)
    old.launcher.common.verify_pins=verify_historical_pins
    try: yield
    finally:
        old.launcher.admission_proof=prior
        old.launcher.common.verify_pins=ORIGINAL_VERIFY_PINS


def legacy_campaign():
    old.compat.install_compat(old.launcher,old.COMPAT_SHA,old.COMPAT_TEST_SHA)
    with historical_e122(): return ORIGINAL_LOAD(old.fresh_args(),old.launcher)


def held_without_training(job_id,cell,campaign):
    raw=run(['scontrol','show','job','-dd','-o',job_id])
    f=lambda k:c.field(raw,k)
    require(f('JobId')==str(job_id) and f('JobState')=='PENDING' and f('Reason')=='JobHeldUser'
            and f('Priority')=='0' and f('RunTime')=='00:00:00' and f('Restarts')=='0'
            and f('StartTime')=='Unknown' and f('UserId').endswith(f'({os.getuid()})'),'job has started or is not exactly held: '+str(job_id))
    require(not Path(cell['run_dir']).exists(),'job run directory already exists')
    if campaign=='e122': old.launcher.audit_held_record(raw,int(job_id),cell)
    else:
        from launch_e124_qwen7b_three_level import job_command
        expected_command=cell.get('command') or job_command(read(E124_PLAN),cell)
        expected_name=next(p.split('=',1)[1] for p in expected_command if p.startswith('--job-name='))
        require(f('JobName')==expected_name and f('Account')=='mltheory'
                and f('Partition')=='lowprio' and f('MinMemoryNode')=='256G'
                and f('NumCPUs')=='8' and f('TimeLimit')=='3-00:00:00'
                and 'gres/gpu:a6000=1' in f('ReqTRES'),'7B resources changed')
    tokens=shlex.split(raw.split('SubmitLine=',1)[1].split(' WorkDir=',1)[0])
    exported=next(t for t in tokens if t.startswith('--export='))[len('--export=ALL,'):]
    require(dict(item.split('=',1) for item in exported.split(','))==cell['environment'],'held job environment mismatch')
    return raw


def make_cell(original,prepared,campaign,admission):
    cell=deepcopy(original)
    env=deepcopy(prepared['environment'])
    stamp=original['run_stamp']+'_neutral_matched_v1'
    directory=str(ROOT/'var/data/modebench_python_level3_neutral_matched_v1'/campaign/original['arm']/('s'+str(original['seed'])))
    env.update(SAVE_PATH=directory,RUN_STAMP=stamp,
               OAT_ZERO_TRAIN_DATA=str(calibration.DATA/'python_factors/train'),
               OAT_ZERO_EVAL_DATA=str(calibration.DATA/'python_factors/eval'))
    command=[]
    for part in prepared['held_submission_command']:
        if part.startswith('--export='): part='--export=ALL,'+','.join(k+'='+v for k,v in sorted(env.items()))
        elif part.startswith('--job-name='): part='--job-name='+stamp
        elif part.startswith('--comment='): part='--comment=python-l3-neutral-matched:'+admission[:16]
        command.append(part)
    cell.update(environment=dict(sorted(env.items())),run_stamp=stamp,run_dir=directory,command=command,
                prompt_condition='python_level3_neutral_v1',dataset_identity_sha256=digest(calibration.DATA/'identity.json'))
    require('--hold' in command and env['OAT_ZERO_PROMPT_TEMPLATE']=='qwen_level3_python_factors_neutral_v1','new neutral held command required')
    return cell


def admission():
    path=calibration.ART/'admission.json'
    r=read(path)
    require(r['admitted'] is True and r['status']=='observed_approximate_match'
            and all(r['gates'].values()) and r['tolerances']==calibration.TOLERANCES,'neutral difficulty has not passed')
    calibration.validate_plan(r['registration_sha256'])
    require(digest(calibration.DATA/'identity.json')==r['dataset_identity_sha256']
            and digest(calibration.RESULTS/'confirmation.json')==r['confirmation_sha256'],'neutral confirmation binding changed')
    calibration.common.verify_pins(read(calibration.DATA/'identity.json')['files_sha256'])
    return r


def prepare():
    require(not PLAN.exists(),'migration already prepared')
    admitted=admission(); parent=legacy_campaign(); staged=read(PREPARED)
    p124=read(E124_PLAN); tx124=read(E124_TX)
    verify_historical_pins(p124['admission']['files_sha256'],p124['admission']['directory_files'])
    ART.mkdir(parents=True,exist_ok=True)
    new(ART/'e124_original_transaction.json',tx124)
    new(ART/'e122_original_binding.json',parent['binding'])
    rows=[]
    for campaign in staged['campaigns']:
        name=campaign['campaign']
        for prepared in campaign['cells']:
            if name=='e122':
                candidates=[j for j in parent['jobs'] if j['cell']['domain']=='python_factors' and j['cell']['arm']==prepared['arm'] and j['cell']['seed']==prepared['seed']]
                require(len(candidates)==1,'ambiguous 0.5B source cell')
                oldjob=candidates[0]; original=oldjob['cell']; old_id=oldjob['job_id']
            else:
                candidates=[r for r in p124['cells'] if r['domain']=='python_factors' and r['level']==3 and r['arm']==prepared['arm'] and r['seed']==prepared['seed']]
                require(len(candidates)==1,'ambiguous 7B source cell')
                original=candidates[0]; old_id=str(tx124['rows'][original['cell_id']]['job_id'])
            raw=held_without_training(old_id,original,name)
            cell=make_cell(original,prepared,name,digest(calibration.ART/'admission.json'))
            require(not Path(cell['run_dir']).exists(),'new namespace already exists')
            rows.append({'campaign':name,'old_job_id':old_id,'old_cell':original,'cell':cell,'old_held_record':raw})
    require(len(rows)==22 and sum(r['campaign']=='e122' for r in rows)==20,'exactly 22 replacements required')
    pins={str(p):digest(p) for p in (SOURCE,ROOT/'tests/test_level3_neutral_migration.py',PREPARED,
        E122_PLAN,E124_PLAN,ROOT/'ops/exp_scaling/control_e124_level3_neutral.py',
        ROOT/'var/artifacts/e124_qwen7b_three_level/controller_v2.py',ART/'e124_original_transaction.json',ART/'e122_original_binding.json',
        calibration.ART/'admission.json',ORIGINAL_TEMPLATE,
        ROOT/'ops/exp_scaling/e122_storage_with_modebench_array_20260911.py',
        ROOT/'ops/exp_scaling/modebench_array_throttle2_storage_20260911.py')}
    for campaign in staged['campaigns']:
        runtime=Path(campaign['cells'][0]['environment']['OAT_ZERO_OPS_SNAPSHOT_ROOT']).parent
        manifest=runtime/'PROMPT_AMENDMENT_IDENTITY.json';pins[str(manifest)]=digest(manifest)
        for rel,sha in read(manifest)['inventory_sha256'].items():pins[str(runtime/rel)]=sha
    new(PLAN,{'schema':'python_level3_neutral_matched_queue_migration_v1','rows':rows,'files_sha256':pins,
              'created_at':calibration.now(),'admission_sha256':digest(calibration.ART/'admission.json'),
              'archival_source_resolution':{str(TEMPLATE):{'path':str(ORIGINAL_TEMPLATE),'sha256':OLD_TEMPLATE_SHA}},
              'e122_cap':4,'e124_cap':1,'e124_block_preserved':tx124.get('status')=='needs_review',
              'e122_deadline':read(old.PLAN)['deadline_utc'],
              'old_controller_ids':['31246501','31246471']})
    return {'status':'prepared','plan_sha256':digest(PLAN),'cells':len(rows)}


def validate(expected):
    require(digest(PLAN)==expected,'migration plan hash mismatch')
    plan=read(PLAN); calibration.common.verify_pins(plan['files_sha256']); admission()
    require(plan['e122_cap']==4 and plan['e124_cap']==1,'release caps changed')
    return plan


def submit(plan,index,row):
    before=ART/f'submission_{index:02d}_intent.json';after=ART/f'submission_{index:02d}_result.json'
    require(not before.exists() and not after.exists(),'existing/ambiguous replacement submission')
    new(before,{'old_job_id':row['old_job_id'],'command':row['cell']['command'],'created_at':calibration.now()})
    p=subprocess.run(row['cell']['command'],text=True,capture_output=True)
    new(after,{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'command':row['cell']['command']})
    require(p.returncode==0 and re.fullmatch(r'[1-9][0-9]*(?:;[^\s]+)?\s*',p.stdout),'submission failed/ambiguous')
    jid=p.stdout.strip().split(';')[0]
    raw=held_without_training(jid,row['cell'],row['campaign'])
    new(ART/f'submission_{index:02d}_audit.json',{'job_id':jid,'held_record':raw,'cell':row['cell']})
    return {**row,'job_id':jid,'held_record':raw}


def migrate(expected):
    plan=validate(expected); require(not COMMIT.exists(),'migration already committed')
    old.compat.install_compat(old.launcher,old.COMPAT_SHA,old.COMPAT_TEST_SHA)
    with old.admission_locks(),c.locked(c.JOURNAL_ROOT):
        # Both prior CPU owners must still be inactive; no competing release.
        queue=run(['squeue','-h','-u',str(os.getuid()),'-o','%i'])
        require(not set(plan['old_controller_ids'])&set(queue.split()),'old controller is active')
        for row in plan['rows']: held_without_training(row['old_job_id'],row['old_cell'],row['campaign'])
        replacements=[submit(plan,i,row) for i,row in enumerate(plan['rows'])]
        for row in replacements:
            held_without_training(row['old_job_id'],row['old_cell'],row['campaign'])
            held_without_training(row['job_id'],row['cell'],row['campaign'])
        new(ART/'cancellation_intent.json',{'old_job_ids':[r['old_job_id'] for r in replacements],
                'new_held_job_ids':[r['job_id'] for r in replacements],'plan_sha256':expected})
        run(['scancel',*[r['old_job_id'] for r in replacements]])
        accounts=run(['sacct','-X','-n','-P','-j',','.join(r['old_job_id'] for r in replacements),'-o','JobIDRaw,State,ElapsedRaw'])
        records={p[0]:p[1:] for line in accounts.splitlines() if (p:=line.split('|')) and p[0] in {r['old_job_id'] for r in replacements}}
        require(len(records)==22 and all(v[0].startswith('CANCELLED') and v[1]=='0' for v in records.values()),'old cancellations require reconciliation')
        new(ART/'cancellation_result.json',{'sacct':accounts,'old_jobs_cancelled_without_runtime':22})
        oldledger=read(ROOT/'var/artifacts/e122_level3_factorial_jobs.json')
        ledger=deepcopy(oldledger); mapping={r['old_job_id']:r for r in replacements if r['campaign']=='e122'}
        for i,row in enumerate(ledger['runs']):
            replacement=mapping.get(str(row['job_id']))
            if replacement:
                ledger['runs'][i]={**row,**{k:replacement['cell'][k] for k in c.CELL_FIELDS},
                                  'job_id':int(replacement['job_id']),'held_scheduler_record':replacement['held_record']}
        ledger.update(schema='e122_neutral_migration_current_jobs_v1',migration_plan_sha256=expected)
        new(ART/'e122_jobs.json',ledger)
        tx=read(ART/'e124_original_transaction.json')
        tx['prompt_dataset_migration_sha256']=expected
        for r in replacements:
            if r['campaign']=='e124':
                cell_id=r['cell']['cell_id']; tx['rows'][cell_id]={**tx['rows'][cell_id],
                    'job_id':int(r['job_id']),'command':r['cell']['command'],'status':'held',
                    'superseded_job_id':int(r['old_job_id'])}
        new(ART/'e124_transaction.json',tx)
        new(COMMIT,{'plan_sha256':expected,'replacements':replacements,'created_at':calibration.now(),
                    'old_jobs_cancelled':22,'new_jobs_held':22,'original_registrations_preserved':True})
    return {'status':'migrated','old_jobs_cancelled':22,'new_jobs_held':22}


def merged_campaign(expected):
    plan=validate(expected); committed=read(COMMIT)
    require(committed['plan_sha256']==expected,'migration commit mismatch')
    parent=legacy_campaign(); result=deepcopy(parent)
    mapping={r['old_job_id']:r for r in committed['replacements'] if r['campaign']=='e122'}
    for job in result['jobs']:
        if job['job_id'] in mapping:
            r=mapping[job['job_id']];job['job_id']=r['job_id'];job['cell']=r['cell']
            job['row']={**job['row'],**{k:r['cell'][k] for k in c.CELL_FIELDS},'job_id':int(r['job_id'])}
    result['binding']={**parent['binding'],'migration_plan_sha256':expected,'migration_commit_sha256':digest(COMMIT)}
    return result


def install_controller(expected):
    plan=validate(expected)
    old.compat.install_compat(old.launcher,old.COMPAT_SHA,old.COMPAT_TEST_SHA)
    c.load_campaign=lambda *_args,**_kwargs:merged_campaign(expected)
    def journals(root,binding,job_ids):
        parent=read(ART/'e122_original_binding.json')
        old_ids={str(r['job_id']) for r in read(ROOT/'var/artifacts/e122_level3_factorial_jobs.json')['runs']}
        inherited=ORIGINAL_READ_JOURNALS(c.JOURNAL_ROOT,parent,old_ids)
        require(set(inherited)<=job_ids,'superseded Python job has release history')
        current=ORIGINAL_READ_JOURNALS(root,binding,job_ids)
        require(not set(current)&set(inherited),'duplicate release ownership')
        return {**inherited,**current}
    c.read_journals=journals
    def snapshot(ids):
        queue=run(['squeue','--noheader','--user',str(os.getuid()),'--format=%i|%T|%r'])
        account=run(['sacct','-n','-X','-j',','.join(ids),'--format=JobIDRaw,State,ExitCode','-P'])
        return c.parse_scheduler(ids,queue,account)
    c.scheduler_snapshot=snapshot
    # Add the exact replacement identities without removing any old reservation.
    base=storage.original.base; original_registry=base.canonical_registry
    def registry():
        result=original_registry()
        for r in read(COMMIT)['replacements']:
            require(r['job_id'] not in result,'replacement ID conflicts with another registry')
            result[r['job_id']]={'run_dir':r['cell']['run_dir'],'model_choice':'05b' if r['campaign']=='e122' else '7b','ledger':str(COMMIT)}
        return result
    base.canonical_registry=registry
    storage.original.E122_LEDGER=ART/'e122_jobs.json'
    def status(campaign,root):
        with throttle.compatible_array_parser(): return old.gated_status(campaign,root,storage)
    c.status=status
    return plan


def watch(expected,once=False):
    plan=install_controller(expected)
    NEW_JOURNAL.mkdir(parents=True,exist_ok=True)
    if not (NEW_JOURNAL/'context.json').exists():
        new(NEW_JOURNAL/'context.json',{'schema':'e122_level3_release_context_v1','binding':merged_campaign(expected)['binding']})
    while True:
        require(datetime.now(timezone.utc)<datetime.fromisoformat(plan['e122_deadline']),'original controller deadline reached')
        try:
            with old.admission_locks(),c.locked(c.JOURNAL_ROOT):
                result=c.status(merged_campaign(expected),NEW_JOURNAL) if once else c.advance_once(old.fresh_args(),old.launcher,root=NEW_JOURNAL)
            old.write(ART/'e122_status.json',result)
            print(json.dumps({k:result.get(k) for k in ('observed_at','blocked_reason','running','next_job_id','last_release_job_id','issues')}),flush=True)
            if result['issues'] or result['needs_operator_review_job_ids']: return 2
        except BlockingIOError: result={'blocked_reason':'waiting_shared_lock'}
        if once:return result
        time.sleep(60)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','migrate','status','watch'])
    p.add_argument('--plan-sha256');a=p.parse_args()
    if a.action=='prepare':result=prepare()
    elif a.action=='migrate':result=migrate(a.plan_sha256)
    else:result=watch(a.plan_sha256,once=a.action=='status')
    print(json.dumps(result,default=str),flush=True)

if __name__=='__main__':main()
