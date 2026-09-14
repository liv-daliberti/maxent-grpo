#!/usr/bin/env python3
"""Submit the frozen initial checkpoint while archived weights are staged."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_modebench_prompt_ablation_local_v2 import validate_plan, verify_checkpoint
from evaluate_modebench_level3 import atomic_new, file_sha
ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'artifacts/modebench_prompt_ablation_20260911/local'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('initial','remaining'))
    args=p.parse_args()
    plan_path=LOCAL/'plan_v2.json'
    plan=json.loads(plan_path.read_text())
    validate_plan(plan)
    if (LOCAL/'submission_intent.json').exists():
        raise ValueError('original full array already submitted or uncertain')
    initial=LOCAL/'submission_v2_initial_result.json'
    if args.action=='initial':
        indices=[0]; array='0%1'; dependency=[]
    else:
        first=json.loads(initial.read_text())
        if first['returncode'] or not str(first['job_id']).isdigit():
            raise ValueError('initial submission failed or uncertain')
        indices=list(range(1,25)); array='1-24%2'
        dependency=['--dependency=afterok:'+str(first['job_id'])]
        if not (LOCAL/'staging_complete.json').exists():
            raise ValueError('full isolated model staging incomplete')
    for index in indices:
        verify_checkpoint(plan['checkpoints'][index])
    amendment=LOCAL/'execution_amendment_v2_initial_gate.json'
    if not amendment.exists():
        atomic_new(amendment,{'schema':'modebench-local-execution-amendment-v1',
            'plan_sha256':file_sha(plan_path),'created_at':datetime.now(timezone.utc).isoformat(),
            'reason':'Run already available initial checkpoint while 24 archived checkpoints download.',
            'scientific_changes':False,'initial_array':'0%1','remaining_array':'1-24%2',
            'remaining_dependency':'afterok initialarray','maximum_concurrent_gpus':2,
            'worker_sha256':file_sha(LOCAL/'worker_v2.slurm'),'launcher':str(Path(__file__).resolve()),
            'launcher_sha256':file_sha(Path(__file__))})
    command=['sbatch','--parsable','--job-name=mb-prompt-local-v2','--partition=lowprio','--account=mltheory',
             '--gres=gpu:a5000:1','--cpus-per-task=6','--mem=40G','--time=01:00:00','--array='+array,
             '--chdir='+str(ROOT),'--output='+str(LOCAL/'logs/%A_%a.out'),'--error='+str(LOCAL/'logs/%A_%a.err'),
             *dependency,str(LOCAL/'worker_v2.slurm')]
    atomic_new(LOCAL/f'submission_v2_{args.action}_intent.json',{'command':command,
               'plan_sha256':file_sha(plan_path),'checkpoint_indices':indices})
    result=subprocess.run(command,text=True,capture_output=True)
    job_id=result.stdout.strip().split(';')[0] if result.returncode==0 else None
    atomic_new(LOCAL/f'submission_v2_{args.action}_result.json',{'returncode':result.returncode,
               'stdout':result.stdout,'stderr':result.stderr,'job_id':job_id})
    if result.returncode:raise RuntimeError(result.stderr)
    print(json.dumps({'status':'submitted','action':args.action,'job_id':job_id,'indices':indices}),flush=True)
if __name__=='__main__':main()
