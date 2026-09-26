#!/usr/bin/env python3
"""Register the new two-task inference writer before resuming existing Python slots."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import migrate_python_level3_cli_20260912 as m
PLAN_SHA='af966347f733140f055ba5f07c43cca445303d980c5a394d3c77e839a01d2854'
ART=m.ART/'storage_successor'
REG=ART/'registration.json'
INFERENCE=ROOT/'var/artifacts/modebench_scale_revised_node_failure_recovery_20260912'
WORKER=INFERENCE/'none_start_transport/worker.slurm'
PLAN=INFERENCE/'none_start_transport/plan.json'
SOURCE=Path(__file__).resolve()

def prepare():
    m.validate(PLAN_SHA)
    m.require(not REG.exists(),'registration exists')
    m.require(not list(m.JOURNAL.glob('intent*')) and not list(m.ART.glob('resume_*.json')),'reconcile prior release before retry')
    m.require({p.name for p in m.JOURNAL.iterdir()} <= {'.lock','context.json'},'unexpected release journal files')
    raw=m.run(['scontrol','show','job','-dd','-o','31258973_0'])
    m.require(m.c.field(raw,'ArrayTaskId')=='0' and m.c.field(raw,'ArrayTaskThrottle')=='1' and str(WORKER) in raw,'inference identity differs')
    intent=m.read(INFERENCE/'none_start_transport/submission_intent.json')
    result=m.read(INFERENCE/'none_start_transport/submission_result.json')
    paths=[SOURCE,Path(m.__file__),m.PLAN,m.COMMIT,m.ART/'post_migration_readback.json',WORKER,PLAN,INFERENCE/'plan.json',INFERENCE/'none_start_transport/submission_intent.json',INFERENCE/'none_start_transport/submission_result.json']
    m.new(REG,{'at':m.c.now(),'migration_plan_sha256':PLAN_SHA,'inference_job_id':'31258973','array_tasks':2,'reserve_gib_per_task':32,'scheduler_record':raw,'submission_intent':intent,'submission_result':result,'pins':{str(p):m.sha(p) for p in paths},'recovery':'Prior resume failed in storage census before any release intent; retain the same four admitted slots and exactly-once journals.'})
    return {'status':'prepared','registration_sha256':m.sha(REG)}

def install(expected):
    m.require(m.sha(REG)==expected,'storage registration changed')
    record=m.read(REG)
    def verify():
        for p,h in record['pins'].items():m.require(m.sha(Path(p))==h,'storage source changed: '+p)
    verify()
    m.budget_helper.ARRAYS['31258973']=(2,WORKER,PLAN)
    original=m.budget_helper.storage_budget
    def budget(*a,**k):verify();return original(*a,**k)
    m.budget_helper.storage_budget=budget

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','resume-four','status','watch']);p.add_argument('--registration-sha256');a=p.parse_args()
    if a.action=='prepare':result=prepare()
    else:
        install(a.registration_sha256)
        result=m.resume_four(PLAN_SHA) if a.action=='resume-four' else m.watch(PLAN_SHA,a.action=='status')
    print(json.dumps(result if a.action in ['prepare','resume-four'] else {'status_written':True}),flush=True)
if __name__=='__main__':main()
