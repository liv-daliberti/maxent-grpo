#!/usr/bin/env python3
"""Deterministic construction recovery and placement-only four-draw confirmation."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import calibrate_modebench_level3_neutral_v5 as c
import watch_level3_neutral_calibration_v5_r2_20260911 as w
EXPECTED=w.EXPECTED
REC=c.ART/'construction_recovery_20260912'
REG=REC/'registration.json'
PARALLEL=c.ART/'confirmation_parallel_registration.json'
SOURCE=Path(__file__).resolve()


def check():
 c.validate_plan(EXPECTED)
 r=c.read(REG);c.require(r['code_sha256']==c.digest(SOURCE),'recovery code changed')
 c.require(r['recipe_sha256']==c.digest(c.ART/'recipe.json'),'chosen recipe changed')
 c.common.verify_pins(c.read(c.ART/'automatic_continuation_r2_registration.json')['continuation_files_sha256'])
 return r


def construct():
 r=check()
 c.require(not (REC/'result.json').exists(),'construction recovery already attempted')
 c.require(not (c.ART/'confirmation_submission_intent.json').exists(),'confirmation ownership already exists')
 c.require(c.read(c.ART/'recipe.json')['development_fit_pass'] is True,'development did not pass')
 with (c.ART/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  c.require(not (c.DATA/'identity.json').exists(),'dataset already complete')
  actual={str(p.relative_to(c.DATA)):c.digest(p) for p in c.DATA.rglob('*') if p.is_file()}
  c.require(actual==r['partial_files_sha256'],'partial construction changed')
  old=REC/'construction_attempt0';c.require(not old.exists(),'partial dataset already preserved')
  # Preserve the complete failed partial construction, not just its row values.
  c.DATA.rename(old)
  for name in ('automatic_continuation_status.json','automatic-r2-31253881.out','automatic-r2-31253881.err'):
   shutil.copy2(c.ART/name,REC/name)
  import make_python_factor_mode_data as cert
  original=cert.validate_python_factor_function_external
  log=(REC/'certification_calls.jsonl').open('x')
  def logged(program,spec):
   before=time.monotonic();result=original(program,spec)
   log.write(json.dumps({'candidate':program,'spec':spec,'key':None if result is None else result.canonical_key,'elapsed_seconds':time.monotonic()-before})+'\n');log.flush()
   return result
  cert.validate_python_factor_function_external=logged
  try:
   c.warm_verifier()
   result=c.finalize(EXPECTED)
   from datasets import load_from_disk
   oldrows=[dict(row) for rows in load_from_disk(str(old/c.DOMAIN/'dev')).values() for row in rows]
   newrows=c.rows_at(c.DATA/c.DOMAIN/'dev')
   c.require(oldrows==newrows,'development rows changed during construction recovery')
   result.update(status='same_recipe_construction_complete',recipe_sha256=c.digest(c.ART/'recipe.json'),
                 dataset_identity_sha256=c.digest(c.DATA/'identity.json'),unchanged_development_rows=True,
                 verifier_timeout_changed=False,sampling_or_selection_retried=False)
  except Exception as error:
   result={'status':'construction_recovery_failed','error':str(error),'error_type':type(error).__name__}
   c.new(REC/'result.json',result);raise
  finally:log.close()
  c.new(REC/'result.json',result)
 return result


def task(i):
 return {**c.task(),'seeds':[c.CONF_LABELS[i]],'output':str(c.RESULTS/f'confirmation_shard{i}.json')}


def prepare_confirmation():
 check();c.require(c.read(REC/'result.json')['status']=='same_recipe_construction_complete','construction incomplete')
 c.require(not PARALLEL.exists(),'parallel placement already registered')
 identity=c.read(c.DATA/'identity.json');c.common.verify_pins(identity['files_sha256'])
 rows=c.mixture.read_jsonl(c.ART/'eval.jsonl');e=c.neutral.evaluator
 full=e.schedule_record(c.DOMAIN,rows,c.CONF_LABELS)
 paths={}
 for i in range(4):
  single=e.schedule_record(c.DOMAIN,rows,[c.CONF_LABELS[i]])
  c.require(all(single['request_seeds'][j][0]==full['request_seeds'][j][i] and single['child_seeds'][j][0]==full['child_seeds'][j][i] for j in range(len(rows))), 'sharding changed a random stream')
  p=c.ART/f'confirmation_shard{i}_tasks.json';c.new(p,[task(i)]);paths[str(p)]=c.digest(p)
 cmd=c.gpu_command('confirmation',EXPECTED)
 cmd=[x for x in cmd if not x.startswith('--wrap=')]
 cmd+=['--array=0-3%4','--wrap=exec '+shlex.join([str(c.PYTHON),'-u','-B',str(SOURCE),'worker'])]
 plan={'schema':'neutral-v5-placement-only-confirmation-v1','created_at':c.now(),'registration_sha256':EXPECTED,
       'code_sha256':c.digest(SOURCE),'dataset_identity_sha256':c.digest(c.DATA/'identity.json'),
       'recipe_sha256':c.digest(c.ART/'recipe.json'),'tasks_sha256':paths,'command':cmd,
       'draw_labels':c.CONF_LABELS,'responses':4096,'prompts':128,'gpus_at_most':4,
       'effective_request_and_child_seeds_identical':True,'full_schedule_sha256':c.mixture.sha(full),
       'changed':['one independent draw per GPU task','unscored fixed verifier warmup before dataset construction'],
       'unchanged':['dataset recipe','case generation seeds','selected dev rows','fresh confirmation cases','model','prompt','verifier','timeouts','K','draw labels','effective random streams','batch size','engine','token cap','tolerances','admission and held-migration gates']}
 c.new(PARALLEL,plan)
 return {'job_id':c.submit_once('confirmation',cmd),'parallel_registration_sha256':c.digest(PARALLEL)}


def check_parallel():
 check();r=c.read(PARALLEL)
 c.require(r['code_sha256']==c.digest(SOURCE),'parallel coordinator changed')
 c.require(r['dataset_identity_sha256']==c.digest(c.DATA/'identity.json'),'parallel dataset changed')
 c.common.verify_pins(r['tasks_sha256']);c.common.verify_pins(c.read(c.DATA/'identity.json')['files_sha256'])
 return r


def worker():
 check_parallel();i=int(os.environ['SLURM_ARRAY_TASK_ID']);c.require(i in range(4),'invalid shard')
 c.new(c.ART/f'confirmation_shard{i}_execution.json',{'job_id':os.environ['SLURM_JOB_ID'],'array_id':os.environ['SLURM_ARRAY_JOB_ID'],'parallel_registration_sha256':c.digest(PARALLEL),'created_at':c.now()})
 c.warm_verifier()
 plan=c.read(c.PLAN)
 c.neutral.evaluator.main(['--model',plan['model']['path'],'--model-label','3b','--tasks-json',str(c.ART/f'confirmation_shard{i}_tasks.json'),'--confirm-eval'])


def merge():
 check_parallel();plan=c.read(c.PLAN);receipts=[];sources={}
 for i in range(4):
  tsk=task(i);receipt,rows=c.validate_receipt(tsk['output'],tsk,plan)
  c.require(receipt['information_boundary']['confirmation_explicitly_authorized'] is True,'unauthorized confirmation receipt')
  receipts.append(receipt);sources[tsk['output']]=c.digest(tsk['output'])
 target=Path(c.task()['output'])
 if target.exists():
  existing=c.read(target);c.require(existing['aggregation']['source_sha256']==sources,'combined receipt source mismatch');c.validate_receipt(target,c.task(),plan);return
 merged=deepcopy(receipts[0]);e=c.neutral.evaluator
 immutable=lambda d:{k:v for k,v in d.items() if k not in ('seeds','seed_schedule','seed_schedule_sha256')}
 c.require(all(immutable(r['identity'])==immutable(receipts[0]['identity']) for r in receipts),'shards have different scientific identities')
 schedule=e.schedule_record(c.DOMAIN,rows,c.CONF_LABELS)
 merged['identity'].update(seeds=c.CONF_LABELS,seed_schedule=schedule,seed_schedule_sha256=c.mixture.sha(schedule))
 merged['identity_sha256']=c.mixture.sha(merged['identity']);merged['sampling']['seeds']=c.CONF_LABELS
 for j,result in enumerate(merged['prompt_results']):
  draws=[r['prompt_results'][j]['draws'][0] for r in receipts]
  c.require([d['seed'] for d in draws]==c.CONF_LABELS,'missing or duplicated draw')
  result['draws']=draws
  result.update({m:statistics.mean(d[m] for d in draws) for m in ('pass1','pass8','distinct8')})
 merged['metrics']=e.summarize(merged['prompt_results']);merged['generated_at']=c.now()
 merged['aggregation']={'kind':'union_of_four_independent_draw_receipts','source_sha256':sources,'parallel_registration_sha256':c.digest(PARALLEL),'coordinator_sha256':c.digest(SOURCE),'synthetic_model_responses':False}
 e.validate_seed_receipt(merged,rows);c.new(target,merged);c.validate_receipt(target,c.task(),plan)


def audit_and_migrate():
 check_parallel()
 with (c.ART/'automatic_continuation.lock').open('a') as observer, (c.ART/'controller.lock').open('a') as lock:
  fcntl.flock(observer,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  merge()
  # The original exact external regrade, thresholds and held-migration routine
  # consume the combined full receipt; the coordinator does not relax any gate.
  status=w.finish_migration(c,w.advance(c,EXPECTED))
  status.update(updated_at=c.now(),registration_sha256=EXPECTED,parallel_registration_sha256=c.digest(PARALLEL))
  c.new(c.ART/'parallel_continuation_result.json',status)
  path=c.ART/'automatic_continuation_status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(path)
 return status


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['construct','prepare_confirmation','worker','audit_and_migrate']);a=p.parse_args()
 result=globals()[a.action]();print(json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':main()
