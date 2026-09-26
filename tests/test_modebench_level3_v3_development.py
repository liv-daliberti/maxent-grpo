"""Synthetic-only v3 DEV publication, RNG, scheduler and worker ownership guards."""
from collections import Counter
from copy import deepcopy
import importlib.util,json,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
SOURCE=Path(__file__).resolve().parents[1]/'ops/exp_scaling/modebench_level3_v3_development.py'
spec=importlib.util.spec_from_file_location('v3_development_tests',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PIN='a'*64;REG='b'*64
REAL_AUTHENTICATE=m.authenticate_saved_seal

@pytest.fixture
def env(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m.common,'CAMPAIGN',tmp_path/'campaign')
    monkeypatch.setattr(m.common,'RESULTS',tmp_path/'results');monkeypatch.setattr(m.common,'REGISTRATION',tmp_path/'campaign/registration.json')
    monkeypatch.setattr(m.common,'POOL_ROOTS',{'graph_coloring':tmp_path/'graph_v8','python_factors':tmp_path/'python_v7'})
    here=tmp_path/'campaign/development';here.mkdir(parents=True)
    for name,filename in [('HERE',None),('PLAN','plan.json'),('SEAL','seal.json'),('CLAIM','development_execution_claim.json'),('WORKER','worker.slurm')]:monkeypatch.setattr(m,name,here if filename is None else here/filename)
    revisions={}
    for domain,name in [('graph_coloring','graph_v8'),('python_factors','python_v7')]:
        revisions[domain]={'name':name,'pool_root':str(m.common.POOL_ROOTS[domain]),'generator_sha256':'g'*64,
            'development_receipts':{str(t):str(m.common.RESULTS/f'calibration_3b_{name}_d{t}.json') for t in range(4)}}
    registration={'candidate_revisions':revisions,'files_sha256':{},'directory_files':{}}
    models={'3b':{'path':'/fixed/model3b'},'05b':{'path':'/fixed/model05b'}}
    plan=m.expected_plan(registration,REG,models)
    m.atomic_new(m.PLAN,plan)
    for job in plan['jobs']:m.atomic_new(job['tasks'],[m.task_for(job)])
    seal={'registration_sha256':REG,'models':models}
    monkeypatch.setattr(m,'authenticate_saved_seal',lambda pin:deepcopy(seal) if pin==PIN else (_ for _ in ()).throw(ValueError('wrong seal')))
    return SimpleNamespace(root=tmp_path,plan=plan,registration=registration,models=models,seal=seal)

def test_fixed_eight_jobs_rows_labels_and_tail_batches(env):
    jobs=env.plan['jobs'];assert len(jobs)==8
    assert [j['name'] for j in jobs]==[f'{name}_d{t}' for name in ('graph_v8','python_v7') for t in range(4)]
    assert [j['rows'] for j in jobs]==[142]*4+[166]*4
    assert [j['batch_count'] for j in jobs]==[72]*4+[84]*4
    assert all(j['rows']%8==6 for j in jobs)
    assert sum(j['rows']*4 for j in jobs)==4928 and sum(j['rows']*32 for j in jobs)==39424
    for job in jobs:
        task=m.task_for(job);assert task['split']=='dev' and task['seeds']==[6428000,6428001,6428002,6428003]
        assert task['batch_size']==8 and task['row_limit']==task['row_offset']==0
        command=m.command_for(job['index'],job,PIN)
        assert command[-3:]==[str(m.WORKER),str(job['index']),PIN]
        assert '--partition=all' in command and '--gres=gpu:rtx_6000:1' in command

@pytest.mark.parametrize('value',['0','-1','123 extra','123\n124','job123','','123;','123;bad cluster'])
def test_strict_scheduler_response_rejects_ambiguous_or_nonpositive_id(value):
    with pytest.raises(ValueError):m.parse_jobid(value)

@pytest.mark.parametrize('value,expected',[('123','123'),('123\n','123'),('123;cluster-1\n','123')])
def test_valid_scheduler_id_and_optional_cluster(value,expected):assert m.parse_jobid(value)==expected

@pytest.mark.parametrize('kind',['output','cache'])
def test_partial_sampling_prevents_any_fresh_worker(env,kind):
    job=env.plan['jobs'][0];path=Path(job['output']+('.batches' if kind=='cache' else ''));path.parent.mkdir(parents=True)
    path.mkdir() if kind=='cache' else path.write_text('{}')
    with pytest.raises(ValueError,match='existing output or partial'):m.fresh_output(job)

@pytest.mark.parametrize('name',['CLAIM','submission_00_intent.json','submission_00_result.json'])
def test_existing_ownership_stops_before_queue_or_submission(env,monkeypatch,name):
    path=m.CLAIM if name=='CLAIM' else m.HERE/name;path.write_text('{}')
    never=Mock(side_effect=AssertionError('scheduler called'));monkeypatch.setattr(m.subprocess,'run',never)
    with pytest.raises(ValueError,match='ownership forbids retry'):m.duplicate_preflight(env.plan)
    never.assert_not_called()

def test_live_equivalent_job_blocks_submission(env,monkeypatch):
    queue=SimpleNamespace(stdout='100|'+m.JOB_PREFIX+'graph_v8_d0|worker\n')
    monkeypatch.setattr(m.subprocess,'run',Mock(side_effect=[queue,SimpleNamespace(stdout='Command='+str(m.WORKER))]))
    with pytest.raises(ValueError,match='live equivalent'):m.duplicate_preflight(env.plan)

def test_unrelated_old_job_does_not_block_new_namespace(env,monkeypatch):
    queue=SimpleNamespace(stdout='100|mb-l3-old-independent|unrelated\n')
    monkeypatch.setattr(m.subprocess,'run',Mock(side_effect=[queue,SimpleNamespace(stdout='Command=/old/unrelated')]))
    assert m.duplicate_preflight(env.plan)==[{'job_id':'100','name':'mb-l3-old-independent'}]

def test_submission_once_preserves_all_eight_exact_intents_results(env,monkeypatch):
    monkeypatch.setattr(m,'duplicate_preflight',lambda plan:[])
    counter=[]
    def sbatch(command,**kwargs):counter.append(command);return SimpleNamespace(returncode=0,stdout=str(100+len(counter))+'\n',stderr='')
    monkeypatch.setattr(m.subprocess,'run',sbatch)
    result=m.submit(PIN);assert result['job_ids']==list(map(str,range(101,109)))
    proof=m.authenticate_submissions(PIN);assert proof['job_ids']==result['job_ids'] and len(proof['files_sha256'])==17
    for j in env.plan['jobs']:
        a=m.HERE/f'submission_{j["index"]:02d}_intent.json';b=a.with_name(a.name.replace('intent','result'))
        intent,after=m.read(a),m.read(b)
        assert intent['command']==after['command']==m.command_for(j['index'],j,PIN)
        assert after['intent_sha256']==m.digest(a) and intent['registration_sha256']==after['registration_sha256']==REG
    with pytest.raises(ValueError):m.submit(PIN)
    assert len(counter)==8

@pytest.mark.parametrize('failure',['exit','ambiguous','duplicate'])
def test_failed_or_ambiguous_submission_preserves_claim_and_never_retries(env,monkeypatch,failure):
    monkeypatch.setattr(m,'duplicate_preflight',lambda plan:[]);calls=[]
    def sbatch(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if failure=='exit' else 0,stdout='ambiguous' if failure=='ambiguous' else '123',stderr='failure' if failure=='exit' else '')
    monkeypatch.setattr(m.subprocess,'run',sbatch)
    with pytest.raises(ValueError):m.submit(PIN)
    assert m.CLAIM.exists() and (m.HERE/'submission_00_intent.json').exists() and (m.HERE/'submission_00_result.json').exists()
    count=len(calls)
    with pytest.raises(ValueError):m.submit(PIN)
    assert len(calls)==count

def test_source_inventory_all55_sources_and_union_tail_blocks(env,monkeypatch):
    data={}
    for job in env.plan['jobs']:
        rows=[{'problem':job['name']+'/'+str(i),'answer_mode_count':4,'level3_difficulty':job['tier']} for i in range(job['rows'])]
        path=Path(job['rows_jsonl']);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic')
        cert={'candidate_revision':env.registration['candidate_revisions'][job['domain']]['name'],'registration_path':str(m.common.REGISTRATION),
              'registration_sha256':REG,'rows_sha256':m.row_hash(rows),'rows':job['rows'],'difficulty':job['tier'],'source_sha256':'g'*64,'checks':{'verified':True}}
        m.atomic_new(path.with_suffix('.identity.json'),cert);data[str(path)]=(rows,{'rows_sha256':m.common.sha(rows)})
    monkeypatch.setattr(m,'load_rows',lambda task:deepcopy(data[task['rows_jsonl']]))
    monkeypatch.setattr(m.common,'calibration_histogram',lambda domain:Counter({(4,):142 if domain=='graph_coloring' else 166}))
    monkeypatch.setattr(m.common,'old_rng_inventory',lambda:{'blocks':{8*i for i in range(23808)},'manifests':[{'old':i} for i in range(47)]})
    result=m.source_inventory(env.plan,env.registration,REG)
    assert len(result['sources'])==55 and len(result['new_sources'])==8
    assert result['distinct_request_blocks']==28736 and result['distinct_child_seeds']==229888
    assert [s['distinct_request_blocks'] for s in result['new_sources']]==[568]*4+[664]*4
    assert len(result['files_sha256'])==16 and len(result['directory_files'])==2
    monkeypatch.setattr(m,'schedule_record',lambda domain,rows,labels:{'request_seeds':[[8*(i*4+j) for j in range(4)] for i in range(len(rows))]})
    with pytest.raises(ValueError,match='overlap or are not aligned'):m.source_inventory(env.plan,env.registration,REG)


def runtime_text(job):
    fields={'JobId':'123','JobState':'RUNNING','UserId':f'user({os.getuid()})','JobName':m.JOB_PREFIX+job['name'],'Command':str(m.WORKER),
            'Partition':'all','QOS':'normal','NumCPUs':'6','MinMemoryNode':'48G','TimeLimit':'01:00:00',
            'AllocTRES':'cpu=6,mem=48G,gres/gpu=1,gres/gpu:rtx_6000=1'}
    return fields

@pytest.mark.parametrize('field,value',[('JobId','124'),('JobState','PENDING'),('UserId','other(0)'),('JobName','wrong'),('Command','wrong'),
    ('Partition','preempt'),('QOS','scavenger'),('NumCPUs','4'),('MinMemoryNode','24G'),('TimeLimit','02:00:00'),('AllocTRES','cpu=6,mem=48G,gres/gpu=2,gres/gpu:rtx_6000=2'),('AllocTRES','cpu=4,mem=48G,gres/gpu=1,gres/gpu:rtx_6000=1')])
def test_actual_scheduler_runtime_must_match_registered_job(env,monkeypatch,field,value):
    job=env.plan['jobs'][0];fields=runtime_text(job);fields[field]=value
    outputs=[SimpleNamespace(stdout=' '.join(k+'='+v for k,v in fields.items())),SimpleNamespace(stdout='PartitionName=all PreemptMode=OFF')]
    monkeypatch.setattr(m.subprocess,'run',Mock(side_effect=outputs))
    with pytest.raises(ValueError,match='scheduler job/resource'):m.authenticate_scheduler_runtime('123',job)

def test_scheduler_runtime_requires_preemption_off(env,monkeypatch):
    job=env.plan['jobs'][0];fields=runtime_text(job)
    monkeypatch.setattr(m.subprocess,'run',Mock(side_effect=[SimpleNamespace(stdout=' '.join(k+'='+v for k,v in fields.items())),SimpleNamespace(stdout='PreemptMode=REQUEUE')]))
    with pytest.raises(ValueError,match='preemption OFF'):m.authenticate_scheduler_runtime('123',job)

def worker_env(env,monkeypatch):
    job=env.plan['jobs'][0]
    m.atomic_new(m.CLAIM,{'jobs':8,'seal_sha256':PIN,'plan_sha256':m.digest(m.PLAN),'registration_path':str(m.common.REGISTRATION),'registration_sha256':REG})
    payload={'cell':job['name'],'command':m.command_for(0,job,PIN),'registration_sha256':REG,'seal_sha256':PIN,'task_sha256':m.digest(job['tasks'])}
    a=m.HERE/'submission_00_intent.json';m.atomic_new(a,payload)
    m.atomic_new(m.HERE/'submission_00_result.json',{**payload,'intent_sha256':m.digest(a),'returncode':0,'stdout':'123\n','stderr':''})
    for key,value in {'SLURM_JOB_ID':'123','TMPDIR':str(m.TMPDIR),'VLLM_USE_V1':'0','VLLM_ATTENTION_BACKEND':'XFORMERS'}.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(m,'authenticate_scheduler_runtime',lambda jobid,job:{'job_id':jobid,'partition':'all','preemption_mode':'OFF','cpus':6,'effective_qos':'normal','scheduler_job':'synthetic','scheduler_partition':'PreemptMode=OFF'})
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='Quadro RTX 6000\n',returncode=0)
    monkeypatch.setattr(m.subprocess,'run',run)
    return job,calls

def test_worker_exact_ownership_runtime_and_dev_evaluator_command(env,monkeypatch):
    job,calls=worker_env(env,monkeypatch);assert m.run_worker(0,PIN)==0
    claim=m.read(m.HERE/'worker_00_execution_claim.json');runtime=m.read(m.HERE/'worker_00_runtime.json')
    assert claim['identity']=={'job_id':'123','cell':job['name'],'seal_sha256':PIN,'task_sha256':m.digest(job['tasks']),'output':job['output']}
    assert claim['runtime']==runtime and runtime['identity']['gpu_names']==['Quadro RTX 6000']
    command=calls[-1];assert str(m.EVALUATOR) in command and '--confirm-eval' not in command and '--resume' not in command
    assert command[command.index('--tasks-json')+1]==job['tasks']
    with pytest.raises(ValueError,match='already claimed'):m.run_worker(0,PIN)
    assert sum(str(m.EVALUATOR) in c for c in calls)==1

@pytest.mark.parametrize('mutation',['jobid','seal','task','command','gpu','environment','partial'])
def test_worker_rejects_wrong_ownership_or_runtime_before_model_call(env,monkeypatch,mutation):
    job,calls=worker_env(env,monkeypatch)
    if mutation=='jobid':monkeypatch.setenv('SLURM_JOB_ID','124')
    elif mutation=='environment':monkeypatch.setenv('VLLM_USE_V1','1')
    elif mutation=='gpu':monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='NVIDIA A100\n'))
    elif mutation=='partial':
        Path(job['output']).parent.mkdir(parents=True);Path(job['output']).write_text('{}')
    else:
        p=m.HERE/'submission_00_result.json';r=m.read(p)
        r[{'seal':'seal_sha256','task':'task_sha256','command':'command'}[mutation]]='wrong';p.write_text(json.dumps(r))
    with pytest.raises(ValueError):m.run_worker(0,PIN)
    assert not any(str(m.EVALUATOR) in c for c in calls)

def test_default_readonly_does_not_open_outcomes_or_invoke_scheduler(env,monkeypatch):
    never=Mock(side_effect=AssertionError('unexpected action'));monkeypatch.setattr(m,'read',never);monkeypatch.setattr(m.subprocess,'run',never)
    assert m.main([])==0;never.assert_not_called()

@pytest.mark.parametrize('args',[['--prepare'],['--submit'],['--worker','0'],['--prepare','--registration-sha256',REG,'--seal-sha256',PIN]])
def test_cli_explicit_action_hash_guards(env,monkeypatch,args):
    never=Mock(side_effect=AssertionError('unauthorized action'));monkeypatch.setattr(m,'prepare',never);monkeypatch.setattr(m,'submit',never);monkeypatch.setattr(m,'run_worker',never)
    with pytest.raises(ValueError):m.main(args)
    never.assert_not_called()


@pytest.mark.parametrize('missing',['file','tree'])
def test_saved_seal_cannot_omit_actual_pool_certificate_or_tree(missing):
    inventory={'sources':[],'new_sources':[],'distinct_request_blocks':28736,'distinct_child_seeds':229888,
               'sorted_request_blocks_sha256':'s','files_sha256':{'pool':'p','certificate':'c'},'directory_files':{'pool_root':['pool','certificate']}}
    seal=deepcopy(inventory)
    m.verify_inventory_binding(seal,inventory)
    if missing=='file':del seal['files_sha256']['certificate']
    else:del seal['directory_files']['pool_root']
    with pytest.raises(ValueError,match='omits actual pool/certificate'):
        m.verify_inventory_binding(seal,inventory)


def test_unregistered_amendment_is_rejected_before_content_read(monkeypatch):
    never=Mock(side_effect=AssertionError('unregistered content read'));monkeypatch.setattr(m,'read',never)
    with pytest.raises(ValueError,match='amendment must be registered'):
        m.verify_amendment_binding({'files_sha256':{}})
    never.assert_not_called()


@pytest.mark.parametrize('text',[
    'cpu=6,mem=48G,gres/gpu=1,gres/gpu:rtx_6000=1,gres/gpu:rtx_6000=1',
    'cpu=6,mem=48G,gres/gpu=1,gres/gpu:rtx_6000=1,gres/gpu:a100=1',
    'cpu=6,cpu=6,mem=48G,gres/gpu=1,gres/gpu:rtx_6000=1',
    'cpu=6,mem=48G,gres/gpu=1', '', 'cpu=6,malformed'])
def test_duplicate_additional_gpu_or_malformed_resource_tokens_rejected(text):
    with pytest.raises(ValueError):m.allocated_resources(text)


def test_actual_partition_name_must_be_all(env,monkeypatch):
    job=env.plan['jobs'][0];fields=runtime_text(job)
    monkeypatch.setattr(m.subprocess,'run',Mock(side_effect=[SimpleNamespace(stdout=' '.join(k+'='+v for k,v in fields.items())),SimpleNamespace(stdout='PartitionName=other PreemptMode=OFF')]))
    with pytest.raises(ValueError,match='partition must be all'):
        m.authenticate_scheduler_runtime('123',job)


@pytest.mark.parametrize('omission',[None,'certificate','pool_tree'])
def test_actual_saved_seal_authentication_requires_complete_fresh_pool_closure(env,monkeypatch,omission):
    m.common.REGISTRATION.write_text('{}')
    m.WORKER.write_text(m.worker_text())
    registered={str(SOURCE):m.digest(SOURCE),str(m.TEST):m.digest(m.TEST)}
    env.registration['files_sha256']=registered
    files=dict(registered);trees={}
    for path in [m.common.REGISTRATION,m.PLAN,m.WORKER,*[Path(j['tasks']) for j in env.plan['jobs']]]:files[str(path)]=m.digest(path)
    certificates=[]
    for job in env.plan['jobs']:
        pool=Path(job['rows_jsonl']);pool.parent.mkdir(parents=True,exist_ok=True);pool.write_text('synthetic')
        certificate=pool.with_suffix('.identity.json');certificate.write_text('{}');certificates.append(certificate)
        for path in (pool,certificate):files[str(path)]=m.digest(path)
    for job in env.plan['jobs']:
        folder=Path(job['rows_jsonl']).parent
        trees[str(folder)]=sorted(str(p.resolve()) for p in folder.iterdir())
    inventory={'sources':[{'index':i} for i in range(55)],'new_sources':[{'index':i} for i in range(8)],
               'distinct_request_blocks':28736,'distinct_child_seeds':229888,'sorted_request_blocks_sha256':'schedule',
               'files_sha256':files,'directory_files':trees}
    seal={**deepcopy(inventory),'schema':m.SEAL_SCHEMA,'registration_path':str(m.common.REGISTRATION),'registration_sha256':REG,
          'plan':str(m.PLAN),'plan_sha256':m.digest(m.PLAN),'execution':m.EXECUTION,'development_draw_labels':list(m.common.DEV_LABELS),
          'models':env.models,'new_candidate_outcomes_loaded':False,'treatment_training_started':False,
          'historical_sources':47,'historical_request_blocks':23808,'new_request_blocks':4928,'new_attempts':39424}
    if omission=='certificate':
        del seal['files_sha256'][str(certificates[0])]
        del seal['directory_files'][str(certificates[0].parent)]
    elif omission=='pool_tree':del seal['directory_files'][next(iter(trees))]
    m.SEAL.write_text(json.dumps(seal))
    monkeypatch.setattr(m.common,'validate_registration',lambda path,pin:deepcopy(env.registration))
    monkeypatch.setattr(m.common,'authenticate_inherited',lambda:{'models':env.models})
    monkeypatch.setattr(m,'verify_amendment_binding',lambda registration:None)
    monkeypatch.setattr(m,'verify_models',lambda models:None)
    monkeypatch.setattr(m,'source_inventory',lambda *args:deepcopy(inventory))
    if omission:
        with pytest.raises(ValueError,match='omits actual pool/certificate'):
            REAL_AUTHENTICATE(m.digest(m.SEAL))
    else:assert REAL_AUTHENTICATE(m.digest(m.SEAL))==seal
