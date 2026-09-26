#!/usr/bin/env python3
"""Clone one pending E119 cell onto node105; site policy forbids account updates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import campaign_stats as campaign
from recover_e119_health_20260905 import checkpoint, atomic
JOB = 31048204
ORIGINAL = 31014487
ART = ROOT / 'var/artifacts/campaign_health_capacity_20260905'
AUDIT = ART / 'e119_node105_backfill.json'


def call(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def field(shown: str, name: str) -> str:
    match = re.search(r'(?:^| )' + re.escape(name) + r'=([^ ]*)', shown)
    assert match, name
    return match.group(1)


def tres(shown: str, name: str) -> dict[str,str]:
    return dict(x.split('=',1) for x in field(shown,name).split(',') if '=' in x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--allow-queue', action='store_true', help='Allow Slurm to queue the unchanged request while node105 is busy')
    args = parser.parse_args()
    if AUDIT.exists():
        raise SystemExit(f'Existing backfill receipt requires inspection: {AUDIT}')
    assert campaign.e119_continuation_jobs(campaign.E119_LEDGER)[ORIGINAL] == JOB
    run = next(r for r in json.loads(campaign.E119_LEDGER.read_text())['runs'] if int(r['job_id']) == ORIGINAL)
    assert (run['domain'], run['arm'], int(run['seed'])) == ('python_factors', 'replay_drgrpo', 45)
    cp = checkpoint(run)
    assert cp['step'] == 1152
    before = call('scontrol', 'show', 'job', '-dd', '-o', str(JOB))
    for key, value in {'JobState':'PENDING', 'Reason':'Priority', 'Partition':'cs', 'Account':'allcs', 'MinMemoryNode':'40G', 'NumCPUs':'8', 'TresPerNode':'gres/gpu:1'}.items():
        assert field(before,key) == value, (key,field(before,key))
    submit = before.split(' SubmitLine=',1)[1].split(' WorkDir=',1)[0]
    command = shlex.split(submit)
    assert command[0]=='sbatch' and '--hold' in command
    replacements={'--partition=':'mltheory','--account=':'mltheory','--nodelist=':'node105'}
    for prefix,value in replacements.items():
        indices=[i for i,x in enumerate(command) if x.startswith(prefix)]
        assert len(indices)==1,(prefix,indices)
        command[indices[0]]=prefix+value
    export=next(x for x in command if x.startswith('--export='))
    assert export in submit and f'RUN_STAMP={run["run_stamp"]}' in export and f'SAVE_PATH={run["run_dir"]}' in export
    node = call('scontrol','show','node','-o','node105')
    assert 'gpu:a5000:10' in field(node,'Gres')
    cfg,alloc=tres(node,'CfgTRES'),tres(node,'AllocTRES')
    free_gpu=int(cfg['gres/gpu'])-int(alloc.get('gres/gpu',0))
    free_mem=int(field(node,'RealMemory'))-int(field(node,'AllocMem'))
    free_cpu=int(field(node,'CPUTot'))-int(field(node,'CPUAlloc'))
    assert not any(word in field(node,'State') for word in ('DOWN','DRAIN','FAIL'))
    assert 'mltheory' in field(node,'Partitions').split(',')
    assert int(cfg['gres/gpu'])>=1 and int(field(node,'RealMemory'))>=40960 and int(field(node,'CPUEfctv'))>=8
    fits_now=free_gpu>=1 and free_mem>=40960 and free_cpu>=8
    assert fits_now or args.allow_queue,(free_gpu,free_mem,free_cpu)
    record = {'schema':'e119-node105-capacity-backfill-v2', 'created_at':datetime.now(timezone.utc).isoformat(),
              'old_job_id':JOB, 'original_job_id':ORIGINAL, 'checkpoint':cp,
              'same_scientific_cell':True, 'same_submitted_exports':True, 'same_run_directory':True,
              'site_constraint':'job_submit.lua rejects modifications to Account/Partition; use audited held replacement',
              'before_scheduler_record':before, 'original_submitline':submit, 'command':command, 'node_before':node,
              'free_capacity_before':{'gpus':free_gpu,'memory_mib':free_mem,'cpus':free_cpu},
              'allow_queue':args.allow_queue,'fits_now':fits_now,
              'changes':{'Partition':'mltheory','Account':'mltheory','ReqNodeList':'node105'},
              'resources_unchanged':{'memory_gib':40,'cpus':8,'gpus':1,'walltime':'1-12:00:00'},
              'joint_capacity_review':{'companion_jobs':'3 E120 Falcon allocations,64GiB each', 'combined_gpus':4,'combined_cpus':32,'combined_memory_gib':232},
              'applied':False,'released':False}
    atomic(ART/'e119_node105_backfill_plan.json',record)
    if not args.apply:
        print(json.dumps({'plan_validated':True,'old_job_id':JOB,'checkpoint':1152,'resources':'1GPU/8CPU/40GiB','free_capacity':record['free_capacity_before']}));return
    ledger_before=campaign.E119_CONTINUATIONS.read_bytes()
    call('scontrol','hold',str(JOB))
    record['held_old']=call('scontrol','show','job','-dd','-o',str(JOB))
    assert field(record['held_old'],'JobState')=='PENDING'
    assert field(record['held_old'],'Reason')=='JobHeldUser'
    atomic(AUDIT,record)
    new_id=None
    committed=False
    old_cancelled=False
    released=False
    try:
        new_id=int(call(*command).split(';',1)[0])
        record['new_job_id']=new_id
        atomic(AUDIT,record)
        held=call('scontrol','show','job','-dd','-o',str(new_id))
        for value in ('JobState=PENDING','Reason=JobHeldUser','Account=mltheory','Partition=mltheory','ReqNodeList=node105',export):assert value in held,value
        for key in ('MinMemoryNode','NumCPUs','NumNodes','TresPerNode','ExcNodeList','TimeLimit'):
            assert field(held,key)==field(before,key),key
        record['held_new']=held
        continuation=json.loads(ledger_before)
        row=next(r for r in continuation['continuations'] if int(r['original_job_id'])==ORIGINAL)
        assert int(row['continuation_job_id'])==JOB
        row['previous_continuation_job_ids']=[*row.get('previous_continuation_job_ids',[]),JOB]
        row['continuation_job_id']=new_id
        row['repair_kind']='20260905_node105_capacity_same_cell_continuation'
        continuation['operational_change']=str(continuation.get('operational_change',''))+'; 2026-09-05 Python Re:Dr s45 pending cell backfilled onto node105/mltheory from1152'
        assert campaign.E119_CONTINUATIONS.read_bytes()==ledger_before,'Concurrent continuation-ledger mutation'
        (ART/'e119_node105_continuation.before.json').write_bytes(ledger_before)
        atomic(campaign.E119_CONTINUATIONS,continuation)
        committed=True
        assert campaign.e119_continuation_jobs(campaign.E119_LEDGER)[ORIGINAL]==new_id
        atomic(AUDIT,record)
        call('scancel',str(JOB))
        old_cancelled=True
        record['old_pending_cancelled']=True
        atomic(AUDIT,record)
        assert not call('squeue','-h','-j',str(JOB),'-o','%i')
        call('scontrol','release',str(new_id))
        released=True
        record.update(applied=True,released=True)
        atomic(AUDIT,record)
        record['after_scheduler_record']=call('scontrol','show','job','-dd','-o',str(new_id))
        atomic(AUDIT,record)
    except Exception as exc:
        record['error']=repr(exc)
        if not old_cancelled and not released:
            if committed:campaign.E119_CONTINUATIONS.write_bytes(ledger_before)
            if new_id is not None:subprocess.run(['scancel',str(new_id)],check=False)
            subprocess.run(['scontrol','release',str(JOB)],check=False)
            record['rolled_back_to_old_pending']=True
        atomic(AUDIT,record)
        raise
    print(json.dumps({'old_job_id':JOB,'new_job_id':new_id,'audit':str(AUDIT),'released':True}))

if __name__=='__main__':main()
