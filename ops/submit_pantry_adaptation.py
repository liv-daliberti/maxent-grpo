"""Freeze the inference implementation and submit the five preregistered workers."""
from pathlib import Path
import json,shutil,subprocess,sys
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent))
from followup_metrics import atomic_new,file_sha
root=Path(__file__).resolve().parents[1];base=root/'artifacts/modebench_inference_followups_20260911/pantry'
if (base/'submission_intent.json').exists():raise SystemExit('Existing submission intent: inspect receipt instead of resubmitting.')
d=json.loads((base/'inputs.json').read_text());code=base/'code'
files=[root/'ops'/n for n in ('run_pantry_adaptation.py','followup_metrics.py','frontier_modebench_contract.py','evaluate_modebench_level2_viability.py','repo_env.sh')]+list((root/'src/oat_drgrpo').rglob('*.py'))
for p in files:
 dest=code/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert file_sha(dest)==file_sha(p)
 else:shutil.copyfile(p,dest)
plan={'schema':'pantry-adaptation-execution-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'inputs':str(base/'inputs.json'),'inputs_sha256':file_sha(base/'inputs.json'),'code_sha256':{str(code/p.relative_to(root)):file_sha(code/p.relative_to(root)) for p in files},'saved_result_sha256':{c['label']:file_sha(Path(d['saved_results_root'])/c['label']/'result.json') for c in d['checkpoints']},'context_amendment':'8192-token context accommodates full eight-response portfolio and up to seven failed recoveries. Response cap remains 192; no truncation. This changes memory/runtime versus original 2048 context, so no equal-compute claim.','feasibility_metadata_note':'The inherited certified_mode_count field is original-task metadata required by the verifier parser (minimum two). Revised support counts, including singleton/empty sets, are explicitly certified in each perturbation record; never interpret the inherited field as revised-task support.','maximum_new_responses':len(d['checkpoints'])*(16*3*8+32*16+sum(p['feasible'] for t in d['tasks'] if t['split']=='eval' for p in t['perturbations'])*3*8),'seed_rule':'Per-checkpoint SHA256 uid seed, explicit child seed uniqueness guard; deterministic resumes reuse cached outputs.'}
atomic_new(base/'execution_plan.json',plan)
worker=base/'worker.slurm'
worker.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+str(root)+'\nsource '+str(code/'ops/repo_env.sh')+'\nexport HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_ATTENTION_BACKEND=XFORMERS\nexec '+str(root/'var/seed_paper_eval/paper310/bin/python')+' -B '+str(code/'ops/run_pantry_adaptation.py')+' --plan '+str(base/'execution_plan.json')+' --index "${SLURM_ARRAY_TASK_ID:?}"\n')
(base/'logs').mkdir(exist_ok=True)
cmd=['sbatch','--parsable','--job-name=pantry-adaptation','--partition=lowprio','--account=mltheory','--gres=gpu:a5000:1','--cpus-per-task=6','--mem=40G','--time=02:00:00','--array=0-4%2','--chdir='+str(root),'--output='+str(base/'logs/%A_%a.out'),'--error='+str(base/'logs/%A_%a.err'),str(worker)]
atomic_new(base/'submission_intent.json',{'command':cmd,'execution_plan_sha256':file_sha(base/'execution_plan.json'),'worker_sha256':file_sha(worker),'created_at_utc':datetime.now(timezone.utc).isoformat()})
r=subprocess.run(cmd,text=True,capture_output=True)
receipt={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'job_id':r.stdout.strip().split(';')[0] if r.returncode==0 else None}
atomic_new(base/'submission_result.json',receipt);print(json.dumps(receipt));r.check_returncode()
