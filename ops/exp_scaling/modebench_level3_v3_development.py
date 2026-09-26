#!/usr/bin/env python3
"""Seal and launch the eight prospective V3 development jobs exactly once.

Default use is read-only. Preparation and submission require explicit separate
CLI actions and immutable external hashes. Interrupted ownership records are
never retried automatically. Workers reuse the unchanged independent evaluator.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl,importlib.metadata,json,os,re,shlex,subprocess,sys,tempfile,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
for directory in ('ops','ops/exp_scaling','src'):sys.path.insert(0,str(ROOT/directory))
import modebench_level3_v3_common as common
from evaluate_modebench_level3_independent import load_rows,schedule_record,validate_task
from evaluate_modebench_level3 import model_identity
from fit_modebench_level3 import row_hash,cell_histogram
HERE=common.CAMPAIGN/'development'
PLAN=HERE/'plan.json';SEAL=HERE/'seal.json';CLAIM=HERE/'development_execution_claim.json';WORKER=HERE/'worker.slurm'
SOURCE=Path(__file__).resolve();TEST=ROOT/'tests/test_modebench_level3_v3_development.py'
PYTHON=ROOT/'var/seed_paper_eval/paper310/bin/python';EVALUATOR=ROOT/'ops/evaluate_modebench_level3_independent.py'
TMPDIR=ROOT/'var/tmp/modebench_level3_v3_development_rtx6000'
AMENDMENT=common.CAMPAIGN/'fixed_reference_amendment.json'
AMENDMENT_SHA='2537d45996e304c6f9880cff030332267ce5981c6674ce17f0129ffb1474fdab'
JOB_PREFIX='mb-l3-v3-dev-'
PLAN_SCHEMA='modebench_level3_v3_development_plan_v1';SEAL_SCHEMA='modebench_level3_v3_development_seal_v1'
EXECUTION={'partition':'all','requested_qos':'normal','preemption_mode':'OFF','gpu':'rtx_6000','gpus':1,'cpus':6,'memory':'48G','time_limit':'01:00:00','exclude':'node103','engine':'V0','attention_backend':'XFORMERS'}
require=common.require;read=common.read;digest=common.digest;atomic_new=common.atomic_new

def now():return datetime.now(timezone.utc).isoformat()
def add_pin(pins,path,expected=None):
    path=str(Path(path).resolve());actual=digest(path)
    require(expected is None or actual==expected,'registered file changed: '+path)
    pins.update(common.merge_pins(pins,{path:actual}))

def verify_amendment_binding(registration):
    require(registration['files_sha256'].get(str(AMENDMENT))==AMENDMENT_SHA==digest(AMENDMENT),
            'prospective fixed-reference amendment must be registered unchanged')
    require(read(AMENDMENT)['contract']==common.contract(),'prospective fixed-reference amendment contract differs')

def expected_plan(registration,registration_sha256,models):
    jobs=[]
    for domain,name,rows in (('graph_coloring','graph_v8',142),('python_factors','python_v7',166)):
        revision=registration['candidate_revisions'][domain]
        require(revision['name']==name and Path(revision['pool_root'])==common.POOL_ROOTS[domain],'registered candidate route differs')
        for tier in range(4):
            label=f'{name}_d{tier}';output=str(common.RESULTS/f'calibration_3b_{label}.json')
            require(revision['development_receipts'][str(tier)]==output,'registered receipt path differs')
            jobs.append({'index':len(jobs),'name':label,'domain':domain,'tier':tier,'model_label':'3b','rows':rows,
                'batch_count':4*((rows+7)//8),'tasks':str(HERE/f'{label}_tasks.json'),'output':output,
                'rows_jsonl':str(common.POOL_ROOTS[domain]/'pools'/domain/f'difficulty_{tier}.jsonl')})
    return {'schema':PLAN_SCHEMA,'registration_path':str(common.REGISTRATION),'registration_sha256':registration_sha256,
            'execution':EXECUTION,'development_draw_labels':list(common.DEV_LABELS),'models':{'3b':models['3b']['path']},
            'jobs':jobs,'total_rows':1232,'new_request_blocks':4928,'new_attempts':39424}

def task_for(job):
    return {'domain':job['domain'],'level':'level3','split':'dev','rows_jsonl':job['rows_jsonl'],'output':job['output'],
            'row_offset':0,'row_limit':0,'interface':'level2_qwen_r5_independent_v2','seeds':list(common.DEV_LABELS),'batch_size':8}

def command_for(index,job,seal_sha256):
    require(type(index) is int and index==job['index'] and 0<=index<8,'registered worker index required')
    require(isinstance(seal_sha256,str) and re.fullmatch(r'[a-f0-9]{64}',seal_sha256),'explicit input seal hash required')
    return ['sbatch','--parsable','--partition=all','--qos=normal','--gres=gpu:rtx_6000:1','--cpus-per-task=6',
            '--mem=48G','--time=01:00:00','--exclude=node103',
            f'--export=ALL,TMPDIR={TMPDIR},VLLM_USE_V1=0,VLLM_ATTENTION_BACKEND=XFORMERS',
            '--job-name='+JOB_PREFIX+job['name'],str(WORKER),str(index),seal_sha256]

def worker_text():
    return f'''#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=node103
#SBATCH --output=var/logs/modebench_level3/%j.out
#SBATCH --error=var/logs/modebench_level3/%j.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:?}}"
export TMPDIR="{TMPDIR}"
source ops/repo_env.sh
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS OMP_NUM_THREADS=4
exec {PYTHON} -B {SOURCE} --worker "${{1:?job index required}}" --seal-sha256 "${{2:?sealed input hash required}}"
'''

def atomic_new_text(path,text):
    path=Path(path);fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
    finally:Path(temporary).unlink(missing_ok=True)

def verify_models(models):
    require(set(models)=={'05b','3b'},'both inherited frozen models required')
    for label,expected in models.items():
        actual=model_identity(Path(expected['path']),label);actual['vllm_version']=importlib.metadata.version('vllm')
        require(actual==expected,'frozen model/engine metadata changed: '+label)

def source_inventory(plan,registration,registration_sha256):
    """Authenticate eight exact pools and recompute all 55 aligned RNG sources."""
    old=common.old_rng_inventory();blocks=set(old['blocks']);sources=list(old['manifests'])
    require(len(sources)==47 and len(blocks)==23808,'all historical RNG sources required')
    files=dict(registration['files_sha256']);trees=dict(registration['directory_files']);new_sources=[]
    for job in plan['jobs']:
        task=task_for(job);validate_task(task,False);rows,identity=load_rows(task)
        revision=registration['candidate_revisions'][job['domain']]
        require(len(rows)==job['rows'] and cell_histogram(job['domain'],rows)==common.calibration_histogram(job['domain']),
                'registered union pool rows/support differs')
        require(all(type(row.get('level3_difficulty')) is int and row['level3_difficulty']==job['tier'] for row in rows),'registered pool tier differs')
        path=Path(job['rows_jsonl']);certificate=path.with_suffix('.identity.json');cert=read(certificate)
        require(cert['candidate_revision']==revision['name'] and cert['registration_path']==str(common.REGISTRATION)
                and cert['registration_sha256']==registration_sha256 and cert['rows_sha256']==row_hash(rows)
                and type(cert['rows']) is int and cert['rows']==job['rows'] and type(cert['difficulty']) is int and cert['difficulty']==job['tier']
                and cert['source_sha256']==revision['generator_sha256'] and cert.get('checks')
                and all(value is True for value in cert['checks'].values()),'pool certificate differs from scientific registration')
        for item in (path,certificate):add_pin(files,item)
        directory=path.parent
        trees[str(directory)]=sorted(str(item.resolve()) for item in directory.rglob('*') if item.is_file())
        for item in trees[str(directory)]:add_pin(files,item)
        schedule=schedule_record(job['domain'],rows,task['seeds']);current={seed for group in schedule['request_seeds'] for seed in group}
        require(len(current)==4*job['rows'] and all(type(seed) is int and seed%8==0 for seed in current)
                and not current&blocks,'new development RNG blocks overlap or are not aligned')
        blocks|=current
        source={'phase':'new_development','name':job['name'],'domain':job['domain'],'model_label':'3b','identity':identity,
                'tasks':job['tasks'],'output':job['output'],'seeds':task['seeds'],'seed_schedule_sha256':common.sha(schedule),
                'distinct_request_blocks':len(current),'distinct_child_seeds':8*len(current)}
        new_sources.append(source);sources.append(source)
    require(len(sources)==55 and len(blocks)==28736,'exact current DEV RNG source coverage required')
    common.verify_pins(files,trees)
    return {'sources':sources,'new_sources':new_sources,'distinct_request_blocks':len(blocks),'distinct_child_seeds':len(blocks)*8,
            'sorted_request_blocks_sha256':common.sha(sorted(blocks)),'files_sha256':files,'directory_files':trees}

def verify_inventory_binding(seal,inventory):
    for key in ('sources','new_sources','distinct_request_blocks','distinct_child_seeds','sorted_request_blocks_sha256'):
        require(seal[key]==inventory[key],'saved DEV source/seed inventory differs: '+key)
    require(all(seal['files_sha256'].get(path)==value for path,value in inventory['files_sha256'].items())
            and all(seal['directory_files'].get(path)==value for path,value in inventory['directory_files'].items()),
            'saved DEV seal omits actual pool/certificate files or inventories')

def fresh_output(job):
    require(not Path(job['output']).exists() and not Path(job['output']+'.batches').exists(),'existing output or partial sampling: '+job['name'])

def parse_jobid(stdout):
    match=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*',stdout)
    require(match is not None,'ambiguous or nonpositive scheduler ID; preserve records and review')
    return match.group(1)

def duplicate_preflight(plan):
    require(not CLAIM.exists() and not list(HERE.glob('submission_*_intent.json'))
            and not list(HERE.glob('submission_*_result.json')),'existing submission ownership forbids retry')
    for job in plan['jobs']:fresh_output(job)
    canonical={job['tasks'] for job in plan['jobs']};equivalent=[]
    pool_paths={job['rows_jsonl'] for job in plan['jobs']};outputs={job['output'] for job in plan['jobs']}
    for root in (ROOT/'var/artifacts').glob('modebench_level3*'):
        for path in root.rglob('*tasks*.json'):
            try:tasks=read(path)
            except (OSError,ValueError):continue
            if not isinstance(tasks,list):continue
            matching=[task for task in tasks if isinstance(task,dict) and
                      (task.get('rows_jsonl') in pool_paths or task.get('output') in outputs
                       or bool(set(task.get('seeds',[]))&set(common.DEV_LABELS)))]
            if matching:
                equivalent.append(str(path.resolve()))
                if str(path.resolve()) not in canonical:
                    for task in matching:
                        require(task.get('output') and not Path(task['output']).exists() and not Path(task['output']+'.batches').exists(),
                                'alternate equivalent task has existing work')
    queue=subprocess.run(['squeue','-h','-u',str(os.getuid()),'-o','%i|%j|%o'],check=True,capture_output=True,text=True,timeout=30)
    inspected=[]
    for line in queue.stdout.splitlines():
        jobid,name,command=line.split('|',2)
        if not any(word in name+command for word in ('mb-l3','level3','independent','evaluate_modebench')):continue
        detail=subprocess.run(['scontrol','show','job',jobid,'-o'],check=True,capture_output=True,text=True,timeout=15).stdout
        inspected.append({'job_id':jobid,'name':name})
        text=command+' '+detail
        duplicate=name.startswith(JOB_PREFIX) or str(HERE) in text or any(path in text for path in [*equivalent,*pool_paths])
        require(not duplicate,'live equivalent development job: '+jobid)
    return inspected

def prepare(registration_sha256):
    registration=common.validate_registration(common.REGISTRATION,registration_sha256)
    verify_amendment_binding(registration)
    require(all(registration['files_sha256'].get(str(p))==digest(p) for p in (SOURCE,TEST)),'launcher and tests must be prospectively registered')
    inherited=common.authenticate_inherited();plan=expected_plan(registration,registration_sha256,inherited['models'])
    inventory=source_inventory(plan,registration,registration_sha256);verify_models(inherited['models'])
    HERE.mkdir(parents=True,exist_ok=True)
    with (HERE/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not any(path.exists() for path in (PLAN,SEAL,WORKER,HERE/'prepare_intent.json')) and not list(HERE.glob('*_tasks.json')),
                'existing preparation or ambiguous intent forbids retry')
        inspected=duplicate_preflight(plan)
        atomic_new(HERE/'prepare_intent.json',{'schema':'modebench_level3_v3_development_prepare_intent_v1','created_at':now(),
                   'registration_path':str(common.REGISTRATION),'registration_sha256':registration_sha256,'source_sha256':digest(SOURCE)})
        for job in plan['jobs']:atomic_new(job['tasks'],[task_for(job)])
        atomic_new_text(WORKER,worker_text());atomic_new(PLAN,plan)
        files=dict(inventory['files_sha256'])
        for path in (SOURCE,TEST,common.REGISTRATION,PLAN,WORKER,ROOT/'ops/repo_env.sh',*(Path(job['tasks']) for job in plan['jobs'])):add_pin(files,path)
        common.validate_registration(common.REGISTRATION,registration_sha256)
        common.verify_pins(files,inventory['directory_files'])
        seal={'schema':SEAL_SCHEMA,'created_at':now(),'registration_path':str(common.REGISTRATION),'registration_sha256':registration_sha256,
              'plan':str(PLAN),'plan_sha256':digest(PLAN),'execution':EXECUTION,'models':inherited['models'],
              'development_draw_labels':list(common.DEV_LABELS),'sources':inventory['sources'],'new_sources':inventory['new_sources'],
              'historical_sources':47,'historical_request_blocks':23808,'new_request_blocks':4928,'new_attempts':39424,
              'distinct_request_blocks':28736,'distinct_child_seeds':229888,'sorted_request_blocks_sha256':inventory['sorted_request_blocks_sha256'],
              'files_sha256':files,'directory_files':inventory['directory_files'],'scheduler_jobs_inspected':inspected,
              'new_candidate_outcomes_loaded':False,'treatment_training_started':False}
        atomic_new(SEAL,seal)
    return {'status':'prepared_not_submitted','seal':str(SEAL),'seal_sha256':digest(SEAL),'jobs':8,'sources':55,'request_blocks':28736}

def authenticate_saved_seal(expected_sha256):
    require(isinstance(expected_sha256,str) and re.fullmatch(r'[a-f0-9]{64}',expected_sha256) and digest(SEAL)==expected_sha256,
            'explicit saved DEV seal hash required')
    seal=read(SEAL)
    require(seal['schema']==SEAL_SCHEMA and seal['registration_path']==str(common.REGISTRATION)
            and seal['plan']==str(PLAN) and seal['plan_sha256']==digest(PLAN) and seal['execution']==EXECUTION
            and seal['development_draw_labels']==list(common.DEV_LABELS)
            and seal['new_candidate_outcomes_loaded'] is False and seal['treatment_training_started'] is False,
            'saved DEV scientific contract differs')
    registration=common.validate_registration(common.REGISTRATION,seal['registration_sha256'])
    verify_amendment_binding(registration)
    require(seal['models']==common.authenticate_inherited()['models'],'saved DEV model binding differs from inherited models')
    require(read(PLAN)==expected_plan(registration,seal['registration_sha256'],seal['models']),'saved plan differs from fixed eight-job plan')
    require(all(seal['files_sha256'].get(path)==value for path,value in registration['files_sha256'].items())
            and all(seal['directory_files'].get(path)==value for path,value in registration['directory_files'].items()),
            'saved DEV seal omits registered inherited inputs')
    for path in (SOURCE,TEST,common.REGISTRATION,PLAN,WORKER):require(seal['files_sha256'].get(str(path))==digest(path),'saved DEV source/config is unpinned')
    require(WORKER.read_text()==worker_text(),'worker script differs from fixed command')
    plan=read(PLAN)
    for job in plan['jobs']:
        require(read(job['tasks'])==[task_for(job)] and seal['files_sha256'].get(job['tasks'])==digest(job['tasks']), 'worker task differs')
    common.verify_pins(seal['files_sha256'],seal['directory_files']);verify_models(seal['models'])
    inventory=source_inventory(plan,registration,seal['registration_sha256'])
    verify_inventory_binding(seal,inventory)
    require((seal['historical_sources'],seal['historical_request_blocks'],seal['new_request_blocks'],seal['new_attempts'])==(47,23808,4928,39424),
            'saved DEV counts differ')
    require(digest(SEAL)==expected_sha256,'DEV seal changed during authentication')
    return seal

def authenticate_submissions(expected_sha256):
    seal=authenticate_saved_seal(expected_sha256);plan=read(PLAN);claim=read(CLAIM)
    require(claim['jobs']==8 and claim['seal_sha256']==expected_sha256 and claim['plan_sha256']==digest(PLAN)
            and claim['registration_path']==str(common.REGISTRATION) and claim['registration_sha256']==seal['registration_sha256'], 'global DEV claim differs')
    files={str(CLAIM):digest(CLAIM)};records=[];ids=[]
    for index,job in enumerate(plan['jobs']):
        a=HERE/f'submission_{index:02d}_intent.json';b=HERE/f'submission_{index:02d}_result.json';intent,result=read(a),read(b)
        expected={'cell':job['name'],'command':command_for(index,job,expected_sha256),'registration_sha256':seal['registration_sha256'],
                  'seal_sha256':expected_sha256,'task_sha256':digest(job['tasks'])}
        require(all(intent.get(k)==v and result.get(k)==v for k,v in expected.items()) and result['intent_sha256']==digest(a)
                and result['returncode']==0,'submission command/ownership/exit differs')
        jobid=parse_jobid(result['stdout']);require(jobid not in ids,'duplicate actual scheduler ID');ids.append(jobid)
        records.append({'job_id':jobid,'job':job,'intent':str(a),'result':str(b)})
        add_pin(files,a);add_pin(files,b)
    return {'jobs':records,'job_ids':ids,'seal_sha256':expected_sha256,'registration_sha256':seal['registration_sha256'],'files_sha256':files}

def submit(expected_sha256):
    HERE.mkdir(parents=True,exist_ok=True)
    with (HERE/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        seal=authenticate_saved_seal(expected_sha256);plan=read(PLAN);inspected=duplicate_preflight(plan)
        atomic_new(CLAIM,{'schema':'modebench_level3_v3_development_execution_claim_v1','created_at':now(),'jobs':8,
                  'plan':str(PLAN),'plan_sha256':digest(PLAN),'seal':str(SEAL),'seal_sha256':expected_sha256,
                  'registration_path':str(common.REGISTRATION),'registration_sha256':seal['registration_sha256'],'scheduler_jobs_inspected':inspected})
        ids=[]
        for index,job in enumerate(plan['jobs']):
            authenticate_saved_seal(expected_sha256);fresh_output(job);command=command_for(index,job,expected_sha256)
            payload={'created_at':now(),'cell':job['name'],'command':command,'registration_sha256':seal['registration_sha256'],
                     'seal_sha256':expected_sha256,'task_sha256':digest(job['tasks'])}
            intent=HERE/f'submission_{index:02d}_intent.json';atomic_new(intent,payload)
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            atomic_new(HERE/f'submission_{index:02d}_result.json',{**payload,'created_at':now(),'intent_sha256':digest(intent),
                       'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            require(result.returncode==0,'sbatch failed; preserve all ownership evidence without retry')
            jobid=parse_jobid(result.stdout);require(jobid not in ids,'duplicate scheduler ID');ids.append(jobid)
            print(json.dumps({'event':'development_submitted','cell':job['name'],'job_id':jobid}),flush=True)
        return {'status':'submitted','job_ids':ids,'jobs':8,'seal_sha256':expected_sha256}

def allocated_resources(text):
    fields={}
    for token in text.split(','):
        pair=token.split('=',1)
        require(len(pair)==2 and pair[0] and pair[1] and pair[0] not in fields,'duplicate or malformed allocated resource token')
        fields[pair[0]]=pair[1]
    require({key for key in fields if key.startswith('gres/gpu')}=={'gres/gpu','gres/gpu:rtx_6000'},
            'allocated GPU types differ from exactly one RTX6000')
    return fields

def authenticate_scheduler_runtime(jobid,job):
    job_text=subprocess.run(['scontrol','show','job',jobid,'-o'],check=True,capture_output=True,text=True,timeout=15).stdout
    partition_text=subprocess.run(['scontrol','show','partition','all','-o'],check=True,capture_output=True,text=True,timeout=15).stdout
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=(\S+)',job_text));allocated=allocated_resources(fields.get('AllocTRES',''))
    require(fields.get('JobId')==jobid and fields.get('JobState')=='RUNNING' and fields.get('UserId','').endswith(f'({os.getuid()})')
            and fields.get('JobName')==JOB_PREFIX+job['name'] and fields.get('Command')==str(WORKER)
            and fields.get('Partition')=='all' and fields.get('QOS') in ('normal','none') and fields.get('NumCPUs')=='6'
            and fields.get('MinMemoryNode')=='48G' and fields.get('TimeLimit')=='01:00:00'
            and allocated.get('cpu')=='6' and allocated.get('gres/gpu')==allocated.get('gres/gpu:rtx_6000')=='1' and allocated.get('mem') in ('48G','49152M'),
            'actual scheduler job/resource identity differs')
    require(re.search(r'(?:^|\s)PartitionName=all(?:\s|$)',partition_text) and re.search(r'(?:^|\s)PreemptMode=OFF(?:\s|$)',partition_text),'registered partition must be all with preemption OFF')
    return {'job_id':jobid,'partition':'all','preemption_mode':'OFF','cpus':6,'effective_qos':fields['QOS'],
            'scheduler_job':job_text,'scheduler_partition':partition_text}

def worker_submission_identity(index,job,jobid,seal,expected_sha256):
    intent_path=HERE/f'submission_{index:02d}_intent.json';result_path=HERE/f'submission_{index:02d}_result.json'
    for _ in range(20):
        if result_path.exists():break
        time.sleep(.25)
    intent,result=read(intent_path),read(result_path)
    expected={'cell':job['name'],'command':command_for(index,job,expected_sha256),'registration_sha256':seal['registration_sha256'],
              'seal_sha256':expected_sha256,'task_sha256':digest(job['tasks'])}
    require(all(intent.get(k)==value and result.get(k)==value for k,value in expected.items()) and result['intent_sha256']==digest(intent_path)
            and result['returncode']==0 and parse_jobid(result['stdout'])==jobid,'actual worker differs from immutable submission result')
    return result

def run_worker(index,expected_sha256):
    require(type(index) is int and 0<=index<8,'registered worker index required')
    seal=authenticate_saved_seal(expected_sha256);plan=read(PLAN);job=plan['jobs'][index];claim=read(CLAIM)
    require(claim['jobs']==8 and claim['seal_sha256']==expected_sha256 and claim['plan_sha256']==digest(PLAN)
            and claim['registration_path']==str(common.REGISTRATION) and claim['registration_sha256']==seal['registration_sha256'],'worker global campaign claim differs')
    jobid=os.environ.get('SLURM_JOB_ID','');require(re.fullmatch(r'[1-9][0-9]*',jobid),'worker requires a nonzero scheduler job ID')
    worker_submission_identity(index,job,jobid,seal,expected_sha256)
    require(os.environ.get('TMPDIR')==str(TMPDIR) and os.environ.get('VLLM_USE_V1')=='0'
            and os.environ.get('VLLM_ATTENTION_BACKEND')=='XFORMERS','worker environment differs')
    gpu_names=[name.strip() for name in subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.splitlines()]
    require(gpu_names in (['Quadro RTX 6000'],['Quadro RTX6000']),'exactly one registered RTX6000 required')
    with (HERE/f'.worker_{index:02d}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker_claim=HERE/f'worker_{index:02d}_execution_claim.json';runtime_path=HERE/f'worker_{index:02d}_runtime.json'
        require(not worker_claim.exists() and not runtime_path.exists(),'worker already claimed; no automatic retry or resume')
        fresh_output(job);scheduler=authenticate_scheduler_runtime(jobid,job)
        identity={'job_id':jobid,'cell':job['name'],'seal_sha256':expected_sha256,'task_sha256':digest(job['tasks']),'output':job['output']}
        runtime_identity={'job_id':jobid,'cell':job['name'],'seal_sha256':expected_sha256,'gpu_names':gpu_names,
                          'partition':'all','preemption_mode':'OFF','cpus':6,'effective_qos':scheduler['effective_qos']}
        runtime={'created_at':now(),'identity':runtime_identity,'scheduler':scheduler}
        atomic_new(worker_claim,{'created_at':now(),'identity':identity,'registration_sha256':seal['registration_sha256'],
                                'runtime':runtime,'gpu_names':gpu_names})
        atomic_new(runtime_path,runtime);authenticate_saved_seal(expected_sha256)
        command=[str(PYTHON),'-B',str(EVALUATOR),'--model',plan['models']['3b'],'--model-label','3b','--tasks-json',job['tasks']]
        print(json.dumps({'event':'worker_authenticated','job_id':jobid,'cell':job['name'],'seal_sha256':expected_sha256,
                          'gpu_names':gpu_names,'runtime':runtime,'task_sha256':digest(job['tasks']),'output':job['output']}),flush=True)
        return subprocess.run(command,cwd=ROOT,check=False).returncode

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare',action='store_true');actions.add_argument('--submit',action='store_true');actions.add_argument('--worker',type=int)
    parser.add_argument('--registration-sha256');parser.add_argument('--seal-sha256');args=parser.parse_args(argv)
    if args.prepare:
        require(args.registration_sha256 and not args.seal_sha256,'preparation requires only explicit scientific registration hash')
        result=prepare(args.registration_sha256)
    elif args.submit:
        require(args.seal_sha256 and not args.registration_sha256,'submission requires explicit prepared seal hash')
        result=submit(args.seal_sha256)
    elif args.worker is not None:
        require(args.seal_sha256 and not args.registration_sha256,'worker requires explicit prepared seal hash')
        return run_worker(args.worker,args.seal_sha256)
    else:
        result={'status':'read_only','registration_exists':common.REGISTRATION.is_file(),'prepared':SEAL.is_file(),
                'claim_exists':CLAIM.is_file(),'expected_jobs':8,'sampling_or_scheduler_actions':False}
    print(json.dumps(result,sort_keys=True),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
