#!/usr/bin/env python3
"""E124 controller overlay for the two neutral, difficulty-matched Python cells.

A pre-existing systems qualification failure remains a release block. The
original transaction is archived, and all new state goes to the migration's
sidecar transaction/ledger. Other cells and systems evidence stay unchanged.
"""
from copy import deepcopy
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import migrate_level3_neutral as migration
import e124_storage_with_modebench_array_20260911 as storage
read,require=migration.read,migration.require
SOURCE=Path(__file__).resolve()
BASE_SOURCE=ROOT/'var/artifacts/e124_qwen7b_three_level/controller_v2.py'


def install(expected):
    migration.validate(expected)
    original=read(migration.E124_PLAN)
    prior=read(migration.ART/'e124_original_transaction.json')
    commit=read(migration.COMMIT)
    amended=deepcopy(original)
    by_id={r['cell']['cell_id']:r for r in commit['replacements'] if r['campaign']=='e124'}
    require(len(by_id)==2,'exactly two E124 replacements required')
    for i,row in enumerate(amended['cells']):
        if row['cell_id'] in by_id:amended['cells'][i]=by_id[row['cell_id']]['cell']
    spec=importlib.util.spec_from_file_location('_e124_neutral_controller',BASE_SOURCE)
    x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)
    base_verify=x.verify; base_digest=x.digest; base_write=x.write
    x.TX=migration.ART/'e124_transaction.json'
    x.LEDGER=migration.ART/'e124_jobs.json'
    def digest(path):
        if Path(path)==migration.TEMPLATE:
            require(base_digest(migration.ORIGINAL_TEMPLATE)==migration.OLD_TEMPLATE_SHA,'historical source changed')
            return migration.OLD_TEMPLATE_SHA
        return base_digest(path)
    x.digest=digest
    def verify(plan,tx,*,full=False):
        migration.validate(expected)
        base_verify(original,prior,full=full)
        require(plan==amended and tx['plan_sha256']==original['plan_sha256']
                and tx['prompt_dataset_migration_sha256']==expected,'E124 amendment binding changed')
        for key,r in by_id.items():
            require(tx['rows'][key]['job_id']==int(r['job_id'])
                    and tx['rows'][key]['command']==r['cell']['command'],'E124 replacement identity changed')
    x.verify=verify
    def write(path,value,**kwargs):
        path=Path(path)
        if path.is_relative_to(x.ART) and path not in (x.TX,x.LEDGER):
            path=migration.ART/'e124_controller'/path.relative_to(x.ART)
        return base_write(path,value,**kwargs)
    x.write=write
    def scheduler_state(job_id):
        queue=x.command(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%r'])
        matches=[s.split('|',2) for s in queue.splitlines() if s.split('|',1)[0]==str(job_id)]
        require(len(matches)<=1,'duplicate live scheduler identity')
        if matches:return {'state':matches[0][1],'reason':matches[0][2],'inactive':False}
        text=x.command(['sacct','-X','-n','-P','-j',str(job_id),'-o','JobIDRaw,State,ExitCode'])
        rows=[s.split('|') for s in text.splitlines() if s.split('|')[0]==str(job_id)]
        require(len(rows)==1 and rows[0][1].split()[0].rstrip('+') in x.TERMINAL,'terminal allocation accounting required')
        return {'state':rows[0][1].split()[0].rstrip('+'),'exit_code':rows[0][2],'inactive':True}
    x.scheduler_state=scheduler_state
    x.storage_report=storage.controller_storage_report
    return x,amended


def observe(expected):
    x,plan=install(expected)
    with migration.old.admission_locks(),migration.c.locked(migration.c.JOURNAL_ROOT):
        tx=read(x.TX);x.verify(plan,tx)
        if tx.get('status')=='needs_review':
            result={'status':'blocked_existing_systems_failure','all_scientific_jobs_remain_held':True,
                    'systems':x.scheduler_state(tx['rows']['systems']['job_id']),
                    'neutral_python_job_ids':[tx['rows'][r['cell_id']]['job_id'] for r in plan['cells'] if r['level']==3 and r['domain']=='python_factors'],
                    'original_error':tx.get('error')}
        else:
            done=x.observe_pass(plan,tx)
            result={'status':tx['status'],'done':done}
        migration.old.write(migration.ART/'e124_status.json',result)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-sha256',required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        result=observe(a.plan_sha256);print(json.dumps(result),flush=True)
        if not a.watch or result['status'].startswith('blocked_') or result.get('done'):return
        time.sleep(60)

if __name__=='__main__':main()
