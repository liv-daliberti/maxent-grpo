#!/usr/bin/env python3
"""Audit four direct Python v6 jobs and the fixed baseline after completion.

Publication replays all20,480 original-grader attempts. The validation API
reauthenticates the bound bytes and semantics without replaying unchanged code.
No recipe, dataset, scheduler job, or confirmation outcome is modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import signal
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for directory in ('ops','ops/exp_scaling','src'):
    sys.path.insert(0,str(ROOT/directory))
from evaluate_modebench_level3_independent import (sha,file_sha,atomic_new,load_rows,
    validate_seed_receipt,schedule_record,prompt_messages,summarize)
from audit_modebench_level3_recovery_execution import (scheduler_records,completion_status,
    verify_engine_log,grade_completed_receipt)
from audit_modebench_level3_independent_match import validate_receipt

CAMPAIGN = ROOT/'var/artifacts/modebench_level3_v2/python_v6'
SEAL = CAMPAIGN/'implementation_seal.json'
CLAIM = CAMPAIGN/'development_execution_claim.json'
RECIPE = CAMPAIGN/'recipe.json'
REPORT = CAMPAIGN/'development_fit_report.json'
INTENT = CAMPAIGN/'development_fit_intent.json'
CANONICAL = CAMPAIGN/'independent_completed_development_audit.json'
TEST_SOURCE = ROOT/'tests/test_modebench_level3_python_v6_development_audit.py'
BASELINE = ROOT/'var/results/modebench_level3_v2/calibration_05b_python_factors.json'
BASELINE_SHA = 'd72b6c76ea0d3ccf7dafb5bdb0520031e0670780437e823da31f8bf1f385ac7b'
RECEIPTS = [BASELINE,*[ROOT/f'var/results/modebench_level3_v2/calibration_3b_python_v6_d{i}.json' for i in range(4)]]
SCHEMA = 'modebench_level3_python_v6_completed_development_audit_v1'

class ReviewDeadline(BaseException):pass

def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def digest(path):return file_sha(Path(path))
def now():return datetime.now(timezone.utc).isoformat()
def add_pin(pins,path,expected=None):
    path=str(Path(path).resolve());actual=digest(path)
    require(expected is None or actual==expected,'bound source changed: '+path)
    require(path not in pins or pins[path]==actual,'conflicting source binding: '+path)
    pins[path]=actual

def launcher_for(seal_sha256):
    require(isinstance(seal_sha256,str) and re.fullmatch('[0-9a-f]{64}',seal_sha256) is not None
            and digest(SEAL)==seal_sha256,'explicit scientific seal mismatch')
    saved=read(SEAL);path=CAMPAIGN/'launch.py'
    require(saved['files_sha256'].get(str(path))==digest(path),'development launcher source is not sealed')
    spec=importlib.util.spec_from_file_location('authenticated_python_v6_development_launcher',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def verify_account(account,jobid):
    require(account['JobIDRaw']==jobid and account['State']=='COMPLETED' and account['ExitCode']=='0:0',
            'direct Python v6 job did not complete successfully')
    require(account['Partition']=='all' and account['AllocCPUS']=='6'
            and account['ReqMem'] in ('48G','48Gn') and account['Timelimit']=='01:00:00'
            and account['NodeList'] not in ('','Unknown','None assigned')
            and account['Start'] not in ('','Unknown') and account['End'] not in ('','Unknown')
            and 'gres/gpu:rtx_6000=1' in account['ReqTRES']
            and 'gres/gpu:rtx_6000=1' in account['AllocTRES'],'completed direct-job resources differ')

def verify_worker_runtime(job,jobid,seal_sha,claim,runtime,account,log_text,model_path):
    verify_account(account,jobid)
    require(claim['identity']=={'job_id':jobid,'seal_sha256':seal_sha,'task_sha256':digest(job['tasks'])},
            'actual worker ownership differs')
    actual=runtime['identity'];scheduler=runtime['scheduler']
    allowed_gpu_names=(['Quadro RTX 6000'],['Quadro RTX6000'])
    require(actual.get('gpu_names') in allowed_gpu_names,'actual single RTX6000 GPU differs')
    require(actual=={'job_id':jobid,'cell':job['name'],'seal_sha256':seal_sha,
            'gpu_names':actual['gpu_names'],'partition':'all','preemption_mode':'OFF','cpus':6,
            'effective_qos':scheduler['effective_qos']} and scheduler['effective_qos'] in ('normal','none'),
            'actual GPU/queue/runtime identity differs')
    require(scheduler['job_id']==jobid and scheduler['partition']=='all'
            and scheduler['preemption_mode']=='OFF' and scheduler['cpus']==6,'worker scheduler metadata differs')
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=(\S+)',scheduler['scheduler_job']))
    allocated=dict(item.split('=',1) for item in fields.get('AllocTRES','').split(',') if '=' in item)
    require(fields.get('JobId')==jobid and fields.get('JobState')=='RUNNING'
            and fields.get('Partition')=='all' and fields.get('NumCPUs')=='6'
            and fields.get('QOS')==scheduler['effective_qos'] and fields.get('TimeLimit')=='01:00:00'
            and fields.get('NodeList')==account['NodeList']
            and allocated.get('gres/gpu')==allocated.get('gres/gpu:rtx_6000')=='1'
            and allocated.get('mem') in ('48G','49152M'),'worker actual scheduler allocation differs')
    require(re.search(r'(?:^|\s)PreemptMode=OFF(?:\s|$)',scheduler['scheduler_partition']) is not None,
            'worker queue preemption mode differs')
    events=[]
    for line in log_text.splitlines():
        if line.startswith('{'):
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if event.get('event')=='worker_authenticated':events.append(event)
    require(events and all(event.get('job_id')==jobid and event.get('cell')==job['name']
            and event.get('seal_sha256')==seal_sha and event.get('gpu_names') in allowed_gpu_names
            for event in events),'actual worker authentication event differs')
    verify_engine_log(log_text,model_path)

def verify_completed_cache(task,receipt,rows,pins):
    """Require every committed batch exactly once and equality with public draws."""
    folder=Path(task['output']+'.batches');run=folder/'run.json'
    require(read(run)=={'identity_sha256':receipt['identity_sha256'],'identity':receipt['identity']},
            'committed run identity differs from public receipt')
    expected={run};seen=set()
    for label_index,label in enumerate(task['seeds']):
        for start in range(0,len(rows),task['batch_size']):
            end=min(start+task['batch_size'],len(rows))
            path=folder/f'seed-{label}__rows-{start:06d}-{end:06d}.json';expected.add(path)
            batch=read(path)
            require(set(batch)=={'identity_sha256','seed','start','end','draws','draws_sha256'}
                    and batch['identity_sha256']==receipt['identity_sha256'] and type(batch['seed']) is int
                    and batch['seed']==label and type(batch['start']) is type(batch['end']) is int
                    and batch['start']==start and batch['end']==end and len(batch['draws'])==end-start
                    and batch['draws_sha256']==sha(batch['draws']),'committed batch metadata/hash differs')
            for index,draw in enumerate(batch['draws'],start):
                require((index,label_index) not in seen,'duplicate committed coordinate')
                seen.add((index,label_index))
                require(sha(draw)==sha(receipt['prompt_results'][index]['draws'][label_index]),
                        'public receipt differs from committed sample draw')
    actual={path for path in folder.rglob('*') if path.is_file()}
    require(actual==expected,'completed cache file inventory differs')
    require(seen=={(i,j) for i in range(len(rows)) for j in range(4)},'completed draw coverage differs')
    for path in sorted(expected):add_pin(pins,path)
    return {'batches':len(expected)-1,'draws':len(seen),'attempts':len(seen)*8,
            'directory':str(folder),'directory_files':sorted(map(str,expected))}

def verify_fit_owner(recipe,seal_sha,pins):
    owner,started=read(REPORT),read(INTENT)
    receipt_pins={str(path):digest(path) for path in RECEIPTS}
    watcher=ROOT/'ops/exp_scaling/watch_modebench_level3_python_v6.py'
    adapter=ROOT/'ops/exp_scaling/fit_modebench_level3_python_v6_independent.py'
    require(not (CAMPAIGN/'development_fit_failure.json').exists(),'fit owner recorded an execution failure')
    require(started['schema']=='modebench_level3_python_v6_fit_intent_v1'
            and started['domain']=='python_factors' and started['candidate_revision']=='python_v6'
            and started['seal_path']==str(SEAL) and started['seal_sha256']==seal_sha
            and started['watcher_sha256']==digest(watcher) and started['adapter_sha256']==digest(adapter)
            and started['receipt_sha256']==receipt_pins and started['recipe_path']==str(RECIPE)
            and started['result_path']==str(REPORT)
            and started['fit_rule']=='unchanged_full_pool_forecast_then_one_hash_fixed_selected_set'
            and started['confirmation_outcomes_used'] is False,'sole-owner fit intent differs')
    require(owner['schema']=='modebench_level3_python_v6_fit_result_v1'
            and owner['domain']=='python_factors' and owner['candidate_revision']=='python_v6'
            and owner['seal_sha256']==seal_sha and owner['watcher_sha256']==digest(watcher)
            and owner['intent_path']==str(INTENT) and owner['intent_sha256']==digest(INTENT)
            and owner['receipt_sha256']==receipt_pins and owner['recipe_path']==str(RECIPE)
            and owner['recipe_sha256']==digest(RECIPE) and owner['confirmation_outcomes_used'] is False
            and all(owner[key]==recipe[key] for key in ('development_fit_pass','decision','weights','development'))
            and owner['status']==('development_fit_pass' if recipe['development_fit_pass'] else 'needs_calibration_revision'),
            'sole-owner fit report differs')
    for path in (REPORT,INTENT,RECIPE,watcher,adapter):add_pin(pins,path)

def audit_completed_development(*,seal_sha256,publish=False,regrade=True):
    require(not publish or regrade,'publication requires full original-grader replay')
    if publish:require(not CANONICAL.exists(),'immutable completed development audit already exists')
    launcher=launcher_for(seal_sha256);seal=launcher.authenticate_saved_seal(seal_sha256)
    plan=read(launcher.PROTOCOL);launcher.check_static(plan)
    pins=dict(seal['files_sha256'])
    for path in (SEAL,CLAIM,Path(__file__),TEST_SOURCE):add_pin(pins,path)
    claim=read(CLAIM)
    require(claim['protocol']==str(launcher.PROTOCOL) and claim['protocol_sha256']==launcher.PROTOCOL_SHA
            and claim['seal']==str(SEAL) and claim['seal_sha256']==seal_sha256
            and claim['candidate_revision']=='python_v6' and claim['jobs']==4 and claim['dependencies_afterany']==[],
            'direct development campaign claim differs')
    job_ids=[]
    for index,job in enumerate(plan['jobs']):
        intent_path=CAMPAIGN/f'submission_{index:02d}_intent.json'
        result_path=CAMPAIGN/f'submission_{index:02d}_result.json'
        intent,result=read(intent_path),read(result_path)
        command=launcher.command_for(index,job,seal_sha256)
        require(intent['command']==result['command']==command and intent['seal_sha256']==seal_sha256
                and intent['task_sha256']==digest(job['tasks']) and intent['cell']==result['cell']==job['name']
                and result['returncode']==0,'direct submission command/task/result differs')
        job_ids.append(launcher.parse_jobid(result['stdout']))
        add_pin(pins,intent_path);add_pin(pins,result_path)
    require(len(job_ids)==len(set(job_ids))==4,'four unique direct scheduler jobs required')
    accounts=scheduler_records(job_ids);status,failed,waiting=completion_status(accounts)
    require(status=='complete',f'direct Python v6 jobs incomplete: failed={failed}, waiting={waiting}')
    require(all(path.is_file() for path in [*RECEIPTS,RECIPE,REPORT,INTENT]),'complete receipts and sole-owner recipe/report required')
    recipe=read(RECIPE);verify_fit_owner(recipe,seal_sha256,pins)
    import fit_modebench_level3_python_v6_independent as fitter
    require(digest(BASELINE)==BASELINE_SHA==fitter.BASELINE_SHA and fitter.BASELINE==BASELINE,'fixed baseline differs')
    repeated=fitter.fit_recipe(BASELINE,RECEIPTS[1:],'python_factors',output=None)
    require(recipe==json.loads(json.dumps(repeated,allow_nan=False)),'recipe does not reproduce exactly in memory')
    gate_values=[recipe['development']['gates'][stage][metric] for stage in ('expected','selected') for metric in ('pass1','pass8')]
    require(type(recipe['development_fit_pass']) is bool and all(type(value) is bool for value in gate_values)
            and recipe['development_fit_pass'] is all(gate_values),'explicit consistent development decision required')
    prior=launcher.authenticate_prior_development(read(launcher.INHERITED_SEAL),read(launcher.ORIGINAL_SEAL))
    require(len(prior)==16640,'all prior scientific request blocks required')
    sources={source['name']:source for source in seal['sources']}
    from transformers import AutoTokenizer
    tokenizers={label:AutoTokenizer.from_pretrained(model['path'],local_files_only=True) for label,model in seal['models'].items()}
    cells=[];all_blocks=set();new_blocks=set();attempts=0
    old_handler=signal.getsignal(signal.SIGALRM)
    def deadline(signum,frame):raise ReviewDeadline('original-grader replay exceeded300seconds perreceipt')
    signal.signal(signal.SIGALRM,deadline)
    try:
        for index,path in enumerate(RECEIPTS):
            label='05b' if index==0 else '3b';role='baseline' if index==0 else 'candidate'
            job=None if index==0 else plan['jobs'][index-1]
            task=read(plan['baseline']['tasks'] if index==0 else job['tasks'])[0]
            require(Path(task['output'])==path,'canonical receipt/task mapping differs')
            rows,source=load_rows(task);receipt=read(path);add_pin(pins,path)
            require(len(rows)==128 and task['seeds']==[6328000,6328001,6328002,6328003]
                    and task['row_offset']==task['row_limit']==0 and task['batch_size']==8,
                    'all128rows/fourdraws required')
            validate_receipt(receipt,domain='python_factors',role=role,development=True)
            require(receipt['identity']['model']==seal['models'][label] and receipt['identity']['source']==source
                    and type(receipt['identity']['batch_size']) is int and receipt['identity']['batch_size']==8,
                    'receipt model/source differs from scientific seal')
            schedule=validate_seed_receipt(receipt,rows=rows)
            prompts=[tokenizers[label].apply_chat_template(prompt_messages('python_factors',row['problem'],receipt['identity']['interface']['prompt_profile']),tokenize=False,add_generation_prompt=True) for row in rows]
            require(receipt['identity']['rendered_prompts_sha256']==sha(prompts),'native rendered prompts differ')
            require(receipt['answer_mode_histogram']==dict(Counter(str(row['answer_mode_count']) for row in rows))
                    and sha(receipt['metrics'])==sha(summarize(receipt['prompt_results'])),'receipt support or full summary differs')
            blocks={base for row in receipt['identity']['seed_schedule']['request_seeds'] for base in row}
            require(len(blocks)==512 and not blocks&all_blocks,'five completed receipts reuse request blocks')
            all_blocks|=blocks
            cache=verify_completed_cache(task,receipt,rows,pins)
            require(cache['batches']==64 and cache['attempts']==4096,'full64batch/4096attempt cell required')
            runtime_metadata=None
            if job is not None:
                expected=sources[job['name']]
                require(source==expected['identity'] and schedule['seed_schedule_sha256']==expected['seed_schedule_sha256']
                        and expected['distinct_request_blocks']==512 and expected['distinct_child_seeds']==4096,
                        'candidate source/schedule differs from scientific seal')
                require(not blocks&(prior|new_blocks),'new candidate RNG overlaps prior source')
                new_blocks|=blocks
                jobid=job_ids[index-1];worker=CAMPAIGN/f'worker_{index-1:02d}_execution_claim.json'
                runtime=CAMPAIGN/f'worker_{index-1:02d}_runtime.json'
                log=ROOT/f'var/logs/modebench_level3/{jobid}.out';err=log.with_suffix('.err')
                verify_worker_runtime(job,jobid,seal_sha256,read(worker),read(runtime),accounts[jobid],
                                      log.read_text()+'\n'+err.read_text(),plan['models'][label])
                for evidence in (worker,runtime,log,err):add_pin(pins,evidence)
                runtime_metadata={'job_id':jobid,'accounting':accounts[jobid],'worker_claim':str(worker),
                                  'runtime_path':str(runtime),'runtime_identity':read(runtime)['identity']}
            else:require(blocks<=prior,'fixed baseline schedule absent from prior scientific inventory')
            count=0
            if regrade:
                signal.alarm(300)
                try:count=grade_completed_receipt(receipt,rows)
                finally:signal.alarm(0)
                require(count==4096,'original-grader replay coverage differs')
                print(json.dumps({'status':'receipt_regraded','cell':path.name,'attempts':count}),flush=True)
            attempts+=cache['attempts']
            cells.append({'path':str(path),'sha256':digest(path),'identity_sha256':receipt['identity_sha256'],
                          'seed_schedule_sha256':schedule['seed_schedule_sha256'],'source':source,'cache':cache,
                          'metrics':receipt['metrics'],'attempts_regraded':count,'execution':runtime_metadata})
    finally:signal.signal(signal.SIGALRM,old_handler)
    require(attempts==20480 and len(all_blocks)==2560 and len(new_blocks)==2048
            and len(prior|new_blocks)==18688,'complete receipt/scientific RNG coverage differs')
    launcher.verify_seal(seal)
    for path,expected in pins.items():require(digest(path)==expected,'input changed during development audit: '+path)
    payload={'schema':SCHEMA,'status':'passed_development' if recipe['development_fit_pass'] else 'needs_calibration_revision',
             'scientific_seal_path':str(SEAL),'scientific_seal_sha256':seal_sha256,
             'recipe_path':str(RECIPE),'recipe_sha256':digest(RECIPE),'development_gates':recipe['development']['gates'],
             'development_fit_pass':recipe['development_fit_pass'],'canonical_json_exact_recipe_reproduction':True,
             'jobs':4,'attempts':attempts,'receipt_records':cells,'scientific_sources':37,'scientific_request_blocks':18688,
             'receipt_request_blocks':2560,'all_attempts_regraded_with_original_grader':regrade,
             'confirmation_outcomes_used':False,'new_recipe_choices_made':False,'scheduler_mutations_performed':False,
             'files_sha256':pins}
    if publish:atomic_new(CANONICAL,{**payload,'created_at':now()})
    return payload

def validate_completed_development_audit(path=CANONICAL):
    path=Path(path).resolve();require(path==CANONICAL.resolve(),'canonical Python v6 completed audit required')
    require(path.is_file(),'completed Python v6 original-grader audit required before confirmation')
    before=digest(path);saved=read(path)
    gates={'expected':{'pass1':True,'pass8':True},'selected':{'pass1':True,'pass8':True}}
    require(saved.get('schema')==SCHEMA and saved.get('status')=='passed_development'
            and saved.get('development_fit_pass') is True and saved.get('development_gates')==gates
            and all(saved['development_gates'][stage][metric] is True for stage in gates for metric in gates[stage]),
            'Python v6 development must pass all four registered gates')
    require(saved['jobs']==4 and saved['attempts']==20480 and saved['all_attempts_regraded_with_original_grader'] is True
            and len(saved['receipt_records'])==5 and all(cell['attempts_regraded']==4096 for cell in saved['receipt_records']),
            'full original-grader replay evidence required')
    for source,expected in saved['files_sha256'].items():require(digest(source)==expected,'attested development input changed: '+source)
    current=audit_completed_development(seal_sha256=saved['scientific_seal_sha256'],regrade=False)
    expected={key:value for key,value in saved.items() if key!='created_at'}
    expected['all_attempts_regraded_with_original_grader']=False
    expected['receipt_records']=[{**cell,'attempts_regraded':0} for cell in expected['receipt_records']]
    require(sha(current)==sha(expected),'saved completed audit differs from current execution semantics')
    require(digest(path)==before,'completed development audit changed during validation')
    pins=dict(saved['files_sha256']);add_pin(pins,path,before)
    return {'path':str(path),'sha256':before,'files_sha256':pins,
            **{key:saved[key] for key in ('status','jobs','attempts','development_gates','scientific_seal_path',
                                          'scientific_seal_sha256','recipe_path','recipe_sha256')}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal-sha256',required=True)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--watch',action='store_true',help='wait for complete fixed receipts/owner artifacts before publishing')
    args=parser.parse_args()
    require(not args.watch or args.publish,'watch requires explicit publication')
    if args.watch:
        launcher=launcher_for(args.seal_sha256);launcher.authenticate_saved_seal(args.seal_sha256)
        while not all(path.is_file() for path in [*RECEIPTS,RECIPE,REPORT,INTENT]):
            print(json.dumps({'status':'waiting_for_complete_python_v6_development','observed_at':now()}),flush=True)
            time.sleep(30);launcher.authenticate_saved_seal(args.seal_sha256)
        # Public receipts can appear just before SLURM transitions to COMPLETED.
        while True:
            jobids=[launcher.parse_jobid(read(CAMPAIGN/f'submission_{i:02d}_result.json')['stdout']) for i in range(4)]
            state,failed,waiting=completion_status(scheduler_records(jobids))
            require(not failed,'direct development jobs failed: '+repr(failed))
            if state=='complete':break
            print(json.dumps({'status':'waiting_for_scheduler_completion','jobs':waiting}),flush=True);time.sleep(15)
    with (CAMPAIGN/'.completed_development_audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=audit_completed_development(seal_sha256=args.seal_sha256,publish=args.publish)
    print(json.dumps({'status':result['status'],'jobs':result['jobs'],'attempts':result['attempts'],
                      'output':str(CANONICAL) if args.publish else None}),flush=True)

if __name__=='__main__':main()
