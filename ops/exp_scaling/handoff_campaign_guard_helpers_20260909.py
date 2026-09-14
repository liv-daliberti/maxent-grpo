#!/usr/bin/env python3
"""Transparent CPU-supervisor handoffs for a verified presentation-only change."""
from __future__ import annotations
import argparse
import ast
import copy
import difflib
import fcntl
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import prioritize_e118_capacity_20260905 as base

ROOT=base.ROOT
ART=ROOT/'var/artifacts/supervisor_helper_compatibility_20260909'
PLAN=ART/'handoff_plan.json'
TX=ART/'handoff_transaction.json'
CAMPAIGN=ROOT/'ops/exp_scaling/campaign_stats.py'
BEFORE=ART/'campaign_stats_52e80139e7976dc1fb60b87eb848588c5be63f9892bdd12fb883ab6ed155f380.py'
LEDGER_LOCK=ROOT/'var/artifacts/e118_ledger_promotion.lock'
TARGETS=(
    (31151368,'campaign_timeout_guard_20260908','guard_campaign_timeouts_20260908.py'),
    (31158474,'campaign_timeout_guard_31048146_20260909','guard_pantry_timeout_31048146_20260909.py'),
    (31158536,'campaign_timeout_guard_31048182_20260909','guard_pantry_timeout_31048182_20260909.py'),
    (31158629,'mathir_hourly_timeout_guard_20260909','guard_mathir_hourly_timeouts_20260909.py'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value,message):
    value.setdefault('events',[]).append({'at':base.now(),'message':message})
    base.atomic(TX,value)
    print(json.dumps({'at':base.now(),'event':message}),flush=True)


def presentation_audit(before,after):
    old=ast.parse(before);new=ast.parse(after)
    def normalize(tree):
        tree=copy.deepcopy(tree)
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='render':
                node.body=[ast.Pass()]
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='main':
                for part in ast.walk(node):
                    if isinstance(part,ast.keyword) and part.arg=='help':
                        part.value=ast.Constant(value='presentation help text')
        return ast.dump(tree,include_attributes=False)
    assert normalize(old)==normalize(new),'Change affects logic outside render() or CLI help strings'
    assert before!=after,'No presentation update to migrate'
    return {'outside_render_and_help_ast_equal':True,'unified_diff':''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(BEFORE),tofile=str(CAMPAIGN)))}


def controller_module(path):
    # Read constants without executing a supervisor or changing imports.
    module={}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            module[node.targets[0].id]=node.value
    lock=module['LOCK']
    assert isinstance(lock,ast.BinOp) and isinstance(lock.right,ast.Constant)
    return str(ROOT/lock.right.value)


def prepare():
    assert not PLAN.exists() and not TX.exists(),'Inspect existing handoff rather than overwrite'
    assert sha(BEFORE)=='52e80139e7976dc1fb60b87eb848588c5be63f9892bdd12fb883ab6ed155f380'
    audit=presentation_audit(BEFORE.read_text(),CAMPAIGN.read_text())
    (ART/'presentation_only.diff').write_text(audit.pop('unified_diff'))
    rows=[]
    for old,dirname,name in TARGETS:
        directory=ROOT/'var/artifacts'/dirname;plan=directory/'plan.json';transaction=directory/'transaction.json'
        p=json.loads(plan.read_text());t=json.loads(transaction.read_text());record=base.show(old)
        assert base.field(record,'JobState')=='RUNNING' and base.field(record,'NodeList')=='node915'
        assert 'gres/gpu' not in base.field(record,'ReqTRES')
        assert t['plan_sha256']==sha(plan)
        controller=ROOT/'ops/exp_scaling'/name
        assert sha(controller)==p['controller_sha256']
        for helper,expected in p['helper_sha256'].items():
            if Path(helper).resolve()!=CAMPAIGN.resolve():assert sha(helper)==expected,helper
        rows.append({'old_job_id':old,'new_job_id':None,'directory':str(directory),'guard_plan':str(plan),'guard_tx':str(transaction),
            'controller':str(controller),'controller_sha256':sha(controller),'singleton_lock':controller_module(controller),
            'old_record':record,'old_command':base.field(record,'Command'),'old_plan_sha256':sha(plan),
            'old_campaign_pin':p['helper_sha256'][str(CAMPAIGN)],'deadline_utc':t['deadline_utc'],
            'cleanup_deadline_utc':t.get('cleanup_deadline_utc'),'time_limit':base.field(record,'TimeLimit'),
            'ready_token':str(ART/f'ready_{old}.json')})
    plan={'schema':'supervisor-display-helper-handoff-v1','prepared_at_utc':base.now(),'rows':rows,
          'controller_sha256':sha(__file__),'before_campaign_sha256':sha(BEFORE),'after_campaign_sha256':sha(CAMPAIGN),
          'presentation_audit':audit,'science_jobs_changed':False,'events':[]}
    base.atomic(PLAN,plan)
    print(json.dumps({'prepared':True,'old_guard_ids':[r['old_job_id'] for r in rows],'plan':str(PLAN),'presentation_audit':audit}),flush=True)


def check_cpu(record,item,new=False):
    assert base.field(record,'NodeList') in {'','node915'}
    assert 'gres/gpu' not in base.field(record,'ReqTRES')
    assert base.field(record,'Account')=='mltheory' and base.field(record,'Partition')=='mltheory'
    if not new:assert base.field(record,'Command')==item['old_command']


def stage_waiters():
    plan=json.loads(PLAN.read_text());assert sha(__file__)==plan['controller_sha256'] and sha(CAMPAIGN)==plan['after_campaign_sha256']
    tx=json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    for item in tx['rows']:
        if item['new_job_id'] is not None:continue
        assert not item.get('submission_uncertain'),'Reconcile uncertain CPU submission manually'
        old=item['old_job_id'];script=ART/f'waiter_{old}.slurm'
        script.write_text('#!/bin/bash\nset -euo pipefail\nexport PATH=/usr/bin:/bin\ncd '+str(ROOT)+'\nexec /usr/local/anaconda3/2024.02/bin/python3 '+str(Path(__file__).resolve())+' wait '+str(old)+'\n')
        command=['sbatch','--parsable','--hold',f'--job-name=guard-handoff-{old}','--account=mltheory','--partition=mltheory','--nodelist=node915',
                 '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=256M','--gres=none',f'--time={item["time_limit"]}','--requeue',
                 f'--chdir={ROOT}',f'--output={ART}/guard-{old}-%j.out',f'--error={ART}/guard-{old}-%j.err','--export=NONE',
                 f'--comment=guard-helper-handoff-20260909-{old}',str(script)]
        item['submission_command']=command;item['submission_uncertain']=True;save(tx,f'Intent to create waiting CPU successor for {old}')
        result=base.command(command,check=False);item['submission_result']={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr};save(tx,f'CPU submission returned for {old}')
        assert result.returncode==0,result.stderr
        item['new_job_id']=int(result.stdout.strip().split(';')[0]);item['submission_uncertain']=False;save(tx,f'Recorded CPU successor {item["new_job_id"]}')
        record=base.show(item['new_job_id']);check_cpu(record,item,new=True)
        assert base.field(record,'JobState')=='PENDING' and base.field(record,'Reason')=='JobHeldUser'
        item['new_held_record']=record;item['release_requested']=True;save(tx,f'Releasing CPU waiter {item["new_job_id"]}; old guard remains active')
        base.command(['scontrol','release',str(item['new_job_id'])])
        item['waiter_released']=True;save(tx,f'CPU waiter {item["new_job_id"]} released')
    tx['status']='waiters_staged';save(tx,'All successor CPU waiters submitted; no old guard stopped and no plan modified')


def stage_images(tx,item):
    directory=ART/f'guard_{item["old_job_id"]}';directory.mkdir(exist_ok=True)
    plan_path=Path(item['guard_plan']);tx_path=Path(item['guard_tx'])
    p=json.loads(plan_path.read_text());t=json.loads(tx_path.read_text())
    assert t['plan_sha256']==sha(plan_path)
    for row in t['jobs'].values():
        assert not [a for a in row.get('attempts',[]) if not a.get('released')],'Guard has an unfinished science mutation; do not hand off'
    assert t['deadline_utc']==item['deadline_utc'] and t.get('cleanup_deadline_utc')==item['cleanup_deadline_utc']
    before_p=copy.deepcopy(p);before_t=copy.deepcopy(t)
    (directory/'plan.before.json').write_bytes(plan_path.read_bytes());(directory/'transaction.before.json').write_bytes(tx_path.read_bytes())
    p['helper_sha256'][str(CAMPAIGN)]=tx['after_campaign_sha256']
    after_plan=directory/'plan.after.json';base.atomic(after_plan,p)
    t['plan_sha256']=sha(after_plan);after_tx=directory/'transaction.after.json';base.atomic(after_tx,t)
    expected=copy.deepcopy(before_p);expected['helper_sha256'][str(CAMPAIGN)]=tx['after_campaign_sha256'];assert p==expected
    expected_t=copy.deepcopy(before_t);expected_t['plan_sha256']=sha(after_plan);assert t==expected_t
    item['images']={'plan_before_sha256':sha(plan_path),'tx_before_sha256':sha(tx_path),
        'plan_after':str(after_plan),'plan_after_sha256':sha(after_plan),'tx_after':str(after_tx),'tx_after_sha256':sha(after_tx)}
    item['preserved_retry_state']=before_t['jobs'];save(tx,f'Staged exact hash-only receipt updates for {item["old_job_id"]}')


def apply():
    tx=json.loads(TX.read_text());assert sha(__file__)==tx['controller_sha256'] and sha(CAMPAIGN)==tx['after_campaign_sha256']
    presentation_audit(BEFORE.read_text(),CAMPAIGN.read_text())
    for item in tx['rows']:
        if item.get('handoff_complete'):continue
        successor=base.show(item['new_job_id']);check_cpu(successor,item,new=True)
        assert base.field(successor,'JobState')=='RUNNING','Waiting successor must run before retiring old guard'
        with LEDGER_LOCK.open('a+') as ledger_lock:
            fcntl.flock(ledger_lock,fcntl.LOCK_EX)
            if not item.get('old_stop_requested'):
                old=base.show(item['old_job_id']);check_cpu(old,item)
                assert base.field(old,'JobState')=='RUNNING'
                current=json.loads(Path(item['guard_tx']).read_text())
                assert not any(not a.get('released') for row in current['jobs'].values() for a in row.get('attempts',[]))
                item['old_before_stop']=old;item['old_stop_requested']=True;save(tx,f'Retiring CPU guard {item["old_job_id"]} at a locked inactive-transition boundary')
                base.command(['scancel',str(item['old_job_id'])])
            with Path(item['singleton_lock']).open('a+') as guard_lock:
                until=time.monotonic()+45
                while True:
                    try:fcntl.flock(guard_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:
                        assert time.monotonic()<until,'Old guard has not relinquished singleton lock; do not modify receipts'
                        time.sleep(0.25)
                assert item['old_job_id'] not in base.queue(),'Old CPU allocation still active'
                if not item.get('images'):stage_images(tx,item)
                images=item['images']
                for path,before,after,image in [(item['guard_plan'],images['plan_before_sha256'],images['plan_after_sha256'],images['plan_after']),
                                                (item['guard_tx'],images['tx_before_sha256'],images['tx_after_sha256'],images['tx_after'])]:
                    assert sha(path) in {before,after},'Receipt changed outside the handoff'
                    assert sha(image)==after
                    if sha(path)!=after:base.atomic(Path(path),json.loads(Path(image).read_text()))
                item['receipts_updated']=True;save(tx,f'Hash receipts updated; retry counts and absolute deadlines unchanged for {item["old_job_id"]}')
            base.atomic(Path(item['ready_token']),{'new_job_id':item['new_job_id'],'guard_plan_sha256':images['plan_after_sha256'],'guard_tx_sha256':images['tx_after_sha256'],'after_campaign_sha256':tx['after_campaign_sha256'],'at':base.now()})
            item['handoff_complete']=True;save(tx,f'Successor {item["new_job_id"]} may enter unchanged guard logic')
    tx['status']='handed_off';save(tx,'All four CPU guard handoffs complete; verify fresh monitoring passes next')


def wait(old):
    plan=json.loads(PLAN.read_text());assert sha(__file__)==plan['controller_sha256']
    item=next(r for r in plan['rows'] if r['old_job_id']==old)
    print(json.dumps({'waiting_for_handoff':old,'at':base.now()}),flush=True)
    deadline=time.monotonic()+3600
    token=Path(item['ready_token'])
    while not token.exists():
        assert time.monotonic()<deadline,'No handoff authorization within one hour; old guard remains authoritative'
        time.sleep(1)
    ready=json.loads(token.read_text())
    assert sha(CAMPAIGN)==ready['after_campaign_sha256'] and sha(item['guard_plan'])==ready['guard_plan_sha256']
    assert sha(item['controller'])==item['controller_sha256']
    print(json.dumps({'handoff_accepted':old,'at':base.now(),'preserved_deadline_utc':item['deadline_utc']}),flush=True)
    sys.argv=[item['controller'],'watch','--apply']
    runpy.run_path(item['controller'],run_name='__main__')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','stage','apply','wait'));parser.add_argument('old_job_id',nargs='?',type=int);args=parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    if args.phase=='wait':assert args.old_job_id is not None;wait(args.old_job_id)
    elif args.phase=='prepare':prepare()
    elif args.phase=='stage':stage_waiters()
    else:apply()
