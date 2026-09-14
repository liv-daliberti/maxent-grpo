#!/usr/bin/env python3
"""Submit a disjoint ready checkpoint subset, chained to prior completed work."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_modebench_prompt_ablation_local_v2 import validate_plan, verify_checkpoint
from evaluate_modebench_level3 import atomic_new, file_sha
ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'artifacts/modebench_prompt_ablation_20260911/local'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indices',type=int,nargs='+',required=True)
    p.add_argument('--after-job',required=True)
    a=p.parse_args()
    indices=sorted(a.indices)
    if len(set(indices))!=len(indices) or any(i<1 or i>24 for i in indices) or not a.after_job.isdigit():
        raise ValueError('invalid task indices or dependency')
    plan_path=LOCAL/'plan_v2.json';plan=json.loads(plan_path.read_text());validate_plan(plan)
    if (LOCAL/'submission_intent.json').exists() or (LOCAL/'submission_v2_remaining_intent.json').exists():
        raise ValueError('full trained array already submitted or uncertain')
    for index in indices:verify_checkpoint(plan['checkpoints'][index])
    with (LOCAL/'subset_submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        existing=sorted(LOCAL.glob('submission_v2_subset_*_intent.json'))
        used=set();latest=json.loads((LOCAL/'submission_v2_initial_result.json').read_text())['job_id']
        for path in existing:
            record=json.loads(path.read_text());used.update(record['checkpoint_indices'])
            result_path=Path(str(path).replace('_intent.json','_result.json'))
            if not result_path.exists():raise ValueError('previous subset submission uncertain')
            result=json.loads(result_path.read_text())
            if result['returncode']:raise ValueError('previous subset submission failed')
            latest=result['job_id']
        if used.intersection(indices):raise ValueError('checkpoint already submitted')
        if str(latest)!=a.after_job:raise ValueError('must chain immediately after latest owned array')
        ordinal=len(existing);prefix=LOCAL/f'submission_v2_subset_{ordinal:02d}'
        command=['sbatch','--parsable','--job-name=mb-prompt-local-v2','--partition=lowprio','--account=mltheory',
            '--gres=gpu:a5000:1','--cpus-per-task=6','--mem=40G','--time=01:00:00',
            '--array='+','.join(str(i) for i in indices)+'%2','--dependency=afterok:'+a.after_job,
            '--chdir='+str(ROOT),'--output='+str(LOCAL/'logs/%A_%a.out'),'--error='+str(LOCAL/'logs/%A_%a.err'),
            str(LOCAL/'worker_v2.slurm')]
        atomic_new(Path(str(prefix)+'_intent.json'),{'command':command,'checkpoint_indices':indices,
            'plan_sha256':file_sha(plan_path),'worker_sha256':file_sha(LOCAL/'worker_v2.slurm'),
            'launcher_path':str(Path(__file__).resolve()),'launcher_sha256':file_sha(Path(__file__)),
            'reason':'Use staged exact checkpoints while remaining archives download; disjoint indices and serial subset arrays keep max2GPUs.',
            'created_at':datetime.now(timezone.utc).isoformat()})
        result=subprocess.run(command,text=True,capture_output=True)
        job_id=result.stdout.strip().split(';')[0] if not result.returncode else None
        atomic_new(Path(str(prefix)+'_result.json'),{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'job_id':job_id})
        if result.returncode:raise RuntimeError(result.stderr)
        print(json.dumps({'status':'submitted','job_id':job_id,'indices':indices,'dependency':a.after_job}),flush=True)
if __name__=='__main__':main()
