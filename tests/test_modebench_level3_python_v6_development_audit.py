"""Fail-closed direct-job ownership, committed samples and replay attestation."""
import copy
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'ops/exp_scaling'),str(ROOT/'src')]
import audit_modebench_level3_python_v6_development as audit

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

def account(jobid='123'):
    return {'JobIDRaw':jobid,'State':'COMPLETED','ExitCode':'0:0','Partition':'all','AllocCPUS':'6',
            'ReqMem':'48G','Timelimit':'01:00:00','NodeList':'node007','Start':'2026-09-09T01:00:00',
            'End':'2026-09-09T01:10:00','ReqTRES':'cpu=6,mem=48G,gres/gpu:rtx_6000=1',
            'AllocTRES':'cpu=6,mem=48G,gres/gpu:rtx_6000=1'}

@pytest.mark.parametrize('field,value',[('State','RUNNING'),('State','TIMEOUT'),('ExitCode','1:0'),
    ('Partition','lowprio'),('AllocCPUS','4'),('ReqMem','24G'),('Timelimit','02:00:00'),
    ('NodeList','Unknown'),('Start','Unknown'),('End','Unknown'),('ReqTRES','gres/gpu:a100=1'),('AllocTRES','')])
def test_direct_accounting_rejects_wrong_completion_or_resources(field,value):
    item=account();item[field]=value
    with pytest.raises(ValueError):audit.verify_account(item,'123')


def runtime_fixture(tmp_path):
    task=tmp_path/'task.json';write(task,[{'example':True}]);job={'name':'3b_python_v6_d0','tasks':str(task)}
    seal='a'*64
    claim={'identity':{'job_id':'123','seal_sha256':seal,'task_sha256':audit.digest(task)}}
    fields='JobId=123 JobState=RUNNING Partition=all NumCPUs=6 QOS=none TimeLimit=01:00:00 NodeList=node007 AllocTRES=cpu=6,mem=48G,gres/gpu=1,gres/gpu:rtx_6000=1'
    runtime={'identity':{'job_id':'123','cell':job['name'],'seal_sha256':seal,'gpu_names':['Quadro RTX 6000'],
                        'partition':'all','preemption_mode':'OFF','cpus':6,'effective_qos':'none'},
             'scheduler':{'job_id':'123','partition':'all','preemption_mode':'OFF','cpus':6,'effective_qos':'none',
                          'scheduler_job':fields,'scheduler_partition':'PartitionName=all PreemptMode=OFF'}}
    event={'event':'worker_authenticated','job_id':'123','cell':job['name'],'seal_sha256':seal,'gpu_names':['Quadro RTX 6000']}
    log=json.dumps(event)+"\nInitializing a V0 LLM engine (v0.8.4) model='/model' dtype=torch.float16 tensor_parallel_size=1 quantization=None\nUsing XFormers backend."
    return job,seal,claim,runtime,log

@pytest.mark.parametrize('gpu',['Quadro RTX 6000','Quadro RTX6000'])
def test_actual_runtime_positive_and_wrong_hardware(tmp_path,gpu):
    job,seal,claim,runtime,log=runtime_fixture(tmp_path)
    runtime['identity']['gpu_names']=[gpu];log=log.replace('Quadro RTX 6000',gpu)
    audit.verify_worker_runtime(job,'123',seal,claim,runtime,account(),log,'/model')
    runtime['identity']['gpu_names']=['NVIDIA A100']
    with pytest.raises(ValueError,match='GPU'):audit.verify_worker_runtime(job,'123',seal,claim,runtime,account(),log,'/model')

@pytest.mark.parametrize('mutation',['owner','seal','qos','preemption','allocation','node','event','engine','model'])
def test_runtime_rejects_authenticated_looking_but_different_execution(tmp_path,mutation):
    job,seal,claim,runtime,log=runtime_fixture(tmp_path)
    if mutation=='owner':claim['identity']['job_id']='124'
    elif mutation=='seal':runtime['identity']['seal_sha256']='b'*64
    elif mutation=='qos':runtime['scheduler']['effective_qos']='pvl'
    elif mutation=='preemption':runtime['scheduler']['scheduler_partition']='PreemptMode=REQUEUE'
    elif mutation=='allocation':runtime['scheduler']['scheduler_job']=runtime['scheduler']['scheduler_job'].replace('mem=48G','mem=24G')
    elif mutation=='node':runtime['scheduler']['scheduler_job']=runtime['scheduler']['scheduler_job'].replace('node007','node008')
    elif mutation=='event':log='\n'.join(log.splitlines()[1:])
    elif mutation=='engine':log=log.replace('dtype=torch.float16','dtype=torch.bfloat16')
    elif mutation=='model':log=log.replace("model='/model'","model='/other'")
    with pytest.raises(ValueError):audit.verify_worker_runtime(job,'123',seal,claim,runtime,account(),log,'/model')


def cache_fixture(tmp_path):
    task={'output':str(tmp_path/'receipt.json'),'seeds':[10,11,12,13],'batch_size':2}
    rows=[{},{}];receipt={'identity_sha256':'identity','identity':{'fixed':True},
                       'prompt_results':[{'draws':[{'text':f'{i}:{j}'} for j in range(4)]} for i in range(2)]}
    folder=Path(task['output']+'.batches');write(folder/'run.json',{'identity_sha256':'identity','identity':receipt['identity']})
    for j,seed in enumerate(task['seeds']):
        draws=[r['draws'][j] for r in receipt['prompt_results']]
        write(folder/f'seed-{seed}__rows-000000-000002.json',{'identity_sha256':'identity','seed':seed,'start':0,'end':2,'draws':draws,'draws_sha256':audit.sha(draws)})
    return task,receipt,rows,folder

def test_cache_proves_exact_draw_equality_and_inventory(tmp_path):
    task,receipt,rows,folder=cache_fixture(tmp_path);pins={}
    result=audit.verify_completed_cache(task,receipt,rows,pins)
    assert result['batches']==4 and result['draws']==8 and result['attempts']==64 and len(pins)==5
    receipt['prompt_results'][0]['draws'][0]['text']='altered public result'
    with pytest.raises(ValueError,match='public receipt'):audit.verify_completed_cache(task,receipt,rows,{})

@pytest.mark.parametrize('mutation',['missing','extra','run','hash','identity','seed','bounds','draws'])
def test_cache_rejects_incomplete_or_altered_committed_evidence(tmp_path,mutation):
    task,receipt,rows,folder=cache_fixture(tmp_path);p=folder/'seed-10__rows-000000-000002.json'
    if mutation=='missing':p.unlink()
    elif mutation=='extra':write(folder/'unexpected.json',{})
    elif mutation=='run':write(folder/'run.json',{'identity_sha256':'other','identity':receipt['identity']})
    else:
        v=audit.read(p)
        if mutation=='hash':v['draws_sha256']='other'
        if mutation=='identity':v['identity_sha256']='other'
        if mutation=='seed':v['seed']=11
        if mutation=='bounds':v['start']=1
        if mutation=='draws':v['draws']=[];v['draws_sha256']=audit.sha([])
        write(p,v)
    with pytest.raises((ValueError,FileNotFoundError)):audit.verify_completed_cache(task,receipt,rows,{})


def saved_fixture(tmp_path,monkeypatch):
    path=tmp_path/'audit.json';monkeypatch.setattr(audit,'CANONICAL',path)
    source=tmp_path/'source.txt';source.write_text('fixed')
    saved={'schema':audit.SCHEMA,'status':'passed_development','development_fit_pass':True,
           'development_gates':{'expected':{'pass1':True,'pass8':True},'selected':{'pass1':True,'pass8':True}},
           'jobs':4,'attempts':20480,'all_attempts_regraded_with_original_grader':True,
           'receipt_records':[{'attempts_regraded':4096} for i in range(5)],
           'files_sha256':{str(source):audit.digest(source)},'created_at':'fixed',
           'scientific_seal_path':'seal','scientific_seal_sha256':'a'*64,'recipe_path':'recipe','recipe_sha256':'b'*64}
    write(path,saved)
    current={k:copy.deepcopy(v) for k,v in saved.items() if k!='created_at'}
    current['all_attempts_regraded_with_original_grader']=False
    for cell in current['receipt_records']:cell['attempts_regraded']=0
    calls=[]
    def check(**kwargs):calls.append(kwargs);return current
    monkeypatch.setattr(audit,'audit_completed_development',check)
    return path,source,saved,current,calls

def test_validation_reauthenticates_without_regrading_and_binds_itself(tmp_path,monkeypatch):
    path,source,saved,current,calls=saved_fixture(tmp_path,monkeypatch)
    result=audit.validate_completed_development_audit(path)
    assert result['jobs']==4 and result['attempts']==20480 and result['status']=='passed_development'
    assert result['files_sha256'][str(path)]==audit.digest(path)
    assert calls==[{'seal_sha256':'a'*64,'regrade':False}]

@pytest.mark.parametrize('mutation',['failed','gate','integer_gate','missing_replay','short_replay','count','source','semantics','different_path'])
def test_validator_rejects_failed_or_unproven_completion(tmp_path,monkeypatch,mutation):
    path,source,saved,current,calls=saved_fixture(tmp_path,monkeypatch)
    if mutation=='failed':saved['status']='needs_calibration_revision'
    elif mutation=='gate':saved['development_gates']['expected']['pass1']=False
    elif mutation=='integer_gate':saved['development_gates']['expected']['pass1']=1
    elif mutation=='missing_replay':saved['all_attempts_regraded_with_original_grader']=False
    elif mutation=='short_replay':saved['receipt_records'][0]['attempts_regraded']=4095
    elif mutation=='count':saved['attempts']=16384
    elif mutation=='source':source.write_text('changed')
    elif mutation=='semantics':current['jobs']=3
    elif mutation=='different_path':path=tmp_path/'other.json'
    write(path,saved)
    with pytest.raises(ValueError):audit.validate_completed_development_audit(path)


def test_publication_cannot_skip_original_grader():
    with pytest.raises(ValueError,match='full original-grader'):audit.audit_completed_development(seal_sha256='a'*64,publish=True,regrade=False)


def test_immutable_publication_rejects_existing_proof_before_execution(tmp_path,monkeypatch):
    target=tmp_path/'audit.json';target.write_text('{}');monkeypatch.setattr(audit,'CANONICAL',target)
    with pytest.raises(ValueError,match='already exists'):audit.audit_completed_development(seal_sha256='a'*64,publish=True)
