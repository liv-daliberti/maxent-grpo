#!/usr/bin/env python3
"""Submit explicit, ready campaign tasks with an auditable Slurm receipt."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'src')]
import evaluate_modebench_fresh_concentration as collector
from analyze_modebench_fresh_panel import authenticate_task
from restore_modebench_fresh_concentration import atomic_new,digest
BASE=ROOT/'artifacts/modebench_fresh_concentration_20260912'
AMENDMENT_SCHEMA='modebench_fresh_concentration_execution_amendment_v1'

def checked_json(reference):
    path=Path(reference['path'])
    if not path.is_file() or digest(path)!=reference['sha256']:
        raise ValueError('execution amendment evidence hash differs')
    return json.loads(path.read_text())

def load_execution_amendment(base,plan,previous):
    """Admit only the documented zero-output Falcon hardware recovery."""
    path=Path(base)/'execution_amendments/falcon_a100.json'
    if not path.exists():return None
    a=json.loads(path.read_text())
    if (a.get('schema')!=AMENDMENT_SCHEMA or a.get('plan_sha256')!=digest(Path(base)/'plan.json')
            or a.get('collector_sha256')!=digest(Path(collector.__file__))
            or a.get('gres')!='gpu:a100:1' or a.get('no_committed_response_slots') is not True):
        raise ValueError('execution amendment does not bind the frozen campaign')
    affected=validate_indices(a['affected_task_indices'],plan)
    expected=[i for i,t in enumerate(plan['tasks']) if t.get('model_scale')=='falcon1b']
    if affected!=expected or not expected:
        raise ValueError('execution amendment must cover exactly all Falcon tasks')
    probe=checked_json(a['device_probe'])
    if (probe.get('gpu_names')!=a.get('gpu_names') or len(a.get('gpu_names',[]))!=1
            or 'A100' not in a['gpu_names'][0]
            or probe.get('shared_memory_per_block_optin_bytes',0)<163840
            or probe.get('bf16_supported') is not True):
        raise ValueError('A100 device probe does not meet the prefix kernel requirement')
    old={(str(r['job_id']),i) for r in previous for i in r['task_indices']
         if i in affected and not r.get('execution_amendment')}
    cancellation=checked_json(a['cancellation_receipt']) if a.get('cancellation_receipt') else None
    superseded=set()
    for attempt in a['superseded_attempts']:
        key=(str(attempt['job_id']),attempt['task_index'])
        if key in superseded or key not in old or attempt['terminal_state'] not in ('FAILED','CANCELLED'):
            raise ValueError('execution amendment has an unknown, duplicate, or nonterminal old attempt')
        superseded.add(key)
        ref=attempt.get('archive_receipt')
        if ref is None:
            if attempt['terminal_state']!='CANCELLED':
                raise ValueError('failed attempts require archived zero-output evidence')
            proof_ref=attempt.get('state_evidence',{}).get('cancellation_receipt')
            if proof_ref:cancellation=checked_json(proof_ref)
            pending_proof=(cancellation and cancellation.get('cancel_exit')==0 and
                           any(x.get('job_id')==f'{key[0]}_{key[1]}' and x.get('state')=='PENDING'
                               for x in cancellation.get('affected',[])))
            if not pending_proof:
                raise ValueError('pending cancellation lacks successful recorded cancellation evidence')
            continue
        archive=checked_json(ref)
        if (str(archive.get('job_id'))!=key[0] or archive.get('task_index')!=key[1]
                or archive.get('task_id')!=plan['tasks'][key[1]]['task_id']
                or archive.get('no_committed_response_slots') is not True):
            raise ValueError('archived attempt identity or zero-output assertion differs')
        archive_path=Path(archive['archive_path'])
        listed=set()
        for f in archive['files']:
            rel=Path(f['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('unsafe archived attempt path')
            target=archive_path/rel;listed.add(rel.as_posix())
            if not target.is_file() or digest(target)!=f['sha256']:
                raise ValueError('archived attempt file hash differs')
        actual={x.relative_to(archive_path).as_posix() for x in archive_path.rglob('*') if x.is_file()}
        if actual!=listed or any(Path(x).name.startswith(('batch_b','responses'))
                                 or Path(x).name=='result.json' for x in actual):
            raise ValueError('archived attempt contains committed or unlisted output')
    if superseded!=old:
        raise ValueError('execution amendment must reconcile every old Falcon submission')
    ref={'path':str(path),'sha256':digest(path)}
    if any(r.get('execution_amendment') and r['execution_amendment']!=ref for r in previous):
        raise ValueError('submitted execution amendment changed')
    return {**a,'reference':ref,'superseded_keys':superseded}

def effective_indices(receipt,amendment):
    superseded=amendment['superseded_keys'] if amendment else set()
    return {i for i in receipt['task_indices'] if (str(receipt['job_id']),i) not in superseded}

def assert_runtime_hardware(task,amendment,output_root):
    receipt=json.loads((Path(output_root)/task['task_id']/'runtime.json').read_text())
    runtime=receipt['runtime']
    if receipt.get('runtime_sha256')!=collector.runtime_fingerprint(runtime):
        raise ValueError('hardware audit runtime fingerprint differs')
    names=runtime.get('gpu_names',[])
    expected=amendment['gpu_names'] if amendment and task.get('model_scale')=='falcon1b' else ['NVIDIA RTX A5000']
    if names!=expected:
        raise ValueError(f"unexpected runtime hardware for {task['task_id']}: {names}")
    return {'task_id':task['task_id'],'gpu_names':names,'runtime_sha256':receipt['runtime_sha256']}

def authenticate_interfaces(plan_path,plan,rows_by_task,indices,amendment=None):
    for i in indices:
        task=plan['tasks'][i]
        result=Path(plan['output_root'])/task['task_id']/'result.json'
        if not result.exists() or json.loads(result.read_text()).get('status')!='complete':
            raise ValueError('both interface tasks must be complete before broad submission')
        authenticate_task(plan_path,plan,task,rows_by_task[task['task_id']])
        if amendment:assert_runtime_hardware(task,amendment,plan['output_root'])

def validate_indices(indices,plan):
    if not indices or len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<len(plan['tasks']) for i in indices):
        raise ValueError('explicit unique in-range task indices required')
    return sorted(indices)

def assert_ready(task):
    if task['checkpoint_stage']=='terminal':
        receipt=ROOT/'var/cache/modebench_fresh_concentration_20260912/receipts'/f"{task['task_id']}.json"
        if not receipt.is_file():raise ValueError(f"checkpoint is not restored: {task['task_id']}")
        r=json.loads(receipt.read_text())
        if r['model_path']!=task['model_path'] or r['files']!=task['files']:
            raise ValueError('restored checkpoint receipt differs from sealed plan')
    # Weight SHA validation is repeated on the GPU worker before loading; this
    # submission guard checks availability/size without rereading all weights.
    for f in task['files']:
        p=Path(task['model_path'])/f['name']
        if not p.is_file() or p.stat().st_size!=f['bytes']:
            raise ValueError(f'missing checkpoint input: {p}')

def submit(indices,phase,base=BASE):
    base=Path(base);plan_path=base/'plan.json'
    plan=json.loads(plan_path.read_text());rows_by_task=collector.validate_plan(plan)
    indices=validate_indices(indices,plan)
    if phase not in ('interface_validation','falcon_interface_validation','full_panel'):raise ValueError('unknown phase')
    submissions=base/'slurm'/'submissions';submissions.mkdir(parents=True,exist_ok=True)
    with (submissions/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        previous=[json.loads(p.read_text()) for p in submissions.glob('*.json')]
        amendment=load_execution_amendment(base,plan,previous)
        requested=set(indices)
        resolved_intents={s['intent_id'] for s in previous}
        unresolved=[p.stem for p in (base/'slurm'/'intents').glob('*.json') if p.stem not in resolved_intents]
        if unresolved:
            raise ValueError(f'unresolved submission intents require scheduler reconciliation: {unresolved}')
        if any(requested & effective_indices(s,amendment) for s in previous):
            raise ValueError('an index already has a submission receipt; inspect it before retrying')
        falcon={i for i in indices if plan['tasks'][i].get('model_scale')=='falcon1b'}
        if falcon and falcon!=requested:raise ValueError('campaign waves must use a single GPU family')
        if falcon and amendment is None:raise ValueError('Falcon requires an authenticated execution amendment')
        gres=amendment['gres'] if falcon else 'gpu:a5000:1'
        if amendment:
            retry_indices={i for _,i in amendment['superseded_keys']}
            for i in requested & retry_indices:
                if (Path(plan['output_root'])/plan['tasks'][i]['task_id']).exists():
                    raise ValueError('superseded attempt output must be archived before retry')
        if phase=='interface_validation':
            if indices!=[0,1] or previous:raise ValueError('first submission must validate exactly tasks 0 and 1')
            cap=2
        else:
            authenticate_interfaces(plan_path,plan,rows_by_task,(0,1))
            if phase=='falcon_interface_validation':
                if indices!=[50,51] or not falcon:
                    raise ValueError('Falcon interface validation requires exactly tasks 50 and 51')
                cap=2
            else:
                if falcon:authenticate_interfaces(plan_path,plan,rows_by_task,(50,51),amendment)
                cap=8
            # No overlapping running/pending arrays: one campaign wave at a time.
            if previous:
                ids=','.join(str(s['job_id']) for s in previous)
                pending=subprocess.run(['squeue','-h','-j',ids,'-o','%i %T'],capture_output=True,text=True,check=True).stdout.strip()
                if pending:raise ValueError(f'previous campaign jobs remain active: {pending}')
        for i in indices:assert_ready(plan['tasks'][i])
        worker=base/'slurm'/'collect.sh'
        args=['sbatch','--parsable','--job-name=mb-fresh-C','--partition=lowprio','--account=mltheory',
              '--gres='+gres,'--cpus-per-task=6','--mem=40G','--time=02:00:00',
              '--requeue','--array='+','.join(map(str,indices))+f'%{cap}',
              '--output='+str(base/'slurm'/'%A_%a.out'),
              '--error='+str(base/'slurm'/'%A_%a.err'),str(worker)]
        intent={'created_at_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'task_indices':indices,
                'plan':str(plan_path),'plan_sha256':digest(plan_path),'worker_sha256':digest(worker),
                'command':args,'max_concurrent_owned_gpus':cap}
        if falcon:intent['execution_amendment']=amendment['reference']
        intent_id=hashlib.sha256(json.dumps(intent,sort_keys=True).encode()).hexdigest()[:16]
        intents=base/'slurm'/'intents';intents.mkdir(exist_ok=True)
        atomic_new(intents/(intent_id+'.json'),intent)
        run=subprocess.run(args,capture_output=True,text=True,check=True)
        job_id=run.stdout.strip().split(';')[0]
        if not job_id.isdigit():raise ValueError(f'unexpected sbatch response; inspect recorded intent: {run.stdout!r}')
        receipt={**intent,'job_id':job_id,'sbatch_stdout':run.stdout,'sbatch_stderr':run.stderr,'intent_id':intent_id}
        atomic_new(submissions/(job_id+'.json'),receipt)
        show=subprocess.run(['scontrol','show','job',job_id],capture_output=True,text=True,check=True)
        (submissions/(job_id+'.slurm.txt')).write_text(show.stdout)
        return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indices',required=True,help='comma-separated explicit task indices')
    p.add_argument('--phase',required=True,choices=['interface_validation','falcon_interface_validation','full_panel'])
    a=p.parse_args();print(json.dumps(submit([int(x) for x in a.indices.split(',')],a.phase),sort_keys=True))
