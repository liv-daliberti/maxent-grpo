"""Scratch-only five-cell transport proofs; no actual Slurm/model/grading calls."""
import copy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_modebench_scale_level4_first_release_r4 import x as release_fixture, fit_fixture, r3_fixture, python_fixture, prior_fixture, base_fixture, observed_base_authorized

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_level4_base_fixture',ROOT/'tests/test_modebench_scale_revised_node_failure_recovery.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
c=previous.c


def load():
    spec=importlib.util.spec_from_file_location('_scratch_level4_transport',ROOT/'artifacts/modebench_scale_level4_first_transport_r4_20260913.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def rewrite(path,value):
    path.chmod(0o644);path.write_text(json.dumps(value))


@pytest.fixture
def t(c,monkeypatch):
    r=load();b=c.r;root=c.base/'level4';root.mkdir()
    monkeypatch.setattr(r,'command_guard',lambda provider,action:{'command':[str(b.PYTHON),'-B','-B',str(r.SOURCE),action],'pid':12345,'uid':r.UID,'start_ticks':'12345'})
    for key,value in {'ROOT':c.base,'SOURCE':c.file('artifacts/transport.py'),'PROVIDER':c.file('artifacts/provider.py'),
        'BASE':b.SOURCE,'LAUNCHER':c.file('artifacts/launcher.py'),'STATE':root,
        'TESTS':c.file('tests/transport.py'),'REVIEW':c.base/'artifacts/transport_review.json',
        'STAGER':c.file('artifacts/stager.py'),'STAGER_TESTS':c.file('tests/stager.py'),
        'SHARED_EVIDENCE':c.file('shared/analysis.json',{'synthetic':'exact diagnostic'}),
        'HISTORICAL_EVIDENCE':c.file('historical/analysis.json',{'synthetic':'exact diagnostic'}),
        'PRESERVATION':c.file('shared/preservation.json',{'synthetic':'exact preservation'}),
        'STAGER_REVIEW':c.file('artifacts/stager_review.json',{'files_sha256':{}})}.items():monkeypatch.setattr(r,key,value)
    proof=c.file('level4/fit_freeze.json',{'synthetic':'fixed fit/freeze source proof'})
    for key,path in [('STAGER_SHA',r.STAGER),('EVIDENCE_SHA',r.SHARED_EVIDENCE),
                     ('PRESERVATION_SHA',r.PRESERVATION),('STAGER_REVIEW_SHA',r.STAGER_REVIEW)]:monkeypatch.setattr(r,key,b.sha(path))
    monkeypatch.setattr(r.socket,'gethostname',lambda:r.HOST)
    monkeypatch.setattr(r,'os',SimpleNamespace(**{**vars(os),'uname':lambda:SimpleNamespace(nodename=r.HOST),
        'getuid':lambda:r.UID,'geteuid':lambda:r.UID}))
    scientific={str(proof):b.sha(proof),str(r.HISTORICAL_EVIDENCE):b.sha(r.HISTORICAL_EVIDENCE)}
    readiness=c.file('level4/confirmation_readiness.json',{'schema':r.READY_SCHEMA,'level':'level4',
        'status':'ready_for_fresh_confirmation','dependency_ids':[901],'files_sha256':scientific})
    cells=[{'id':'level4_'+domain+'_eval','level':'level4','domain':domain,'phase':'eval',
        'source_root':str(c.base/'source'/domain),'source_kind':'campaign_v1' if domain in ('countdown','mathir') else 'domain_revision_v1',
        'tasks':[{'level':'level4','domain':domain,'split':'eval','output':str(c.base/'heldout'/f'{domain}.json'),
            'seeds':[11,12,13,14],'batch_size':8,'row_offset':0,'row_limit':0,'rows_jsonl':str(c.base/'source'/domain/'eval.jsonl')}]} for domain in r.DOMAINS]
    data=SimpleNamespace(r=r,b=b,c=c,root=root,provider_sha=b.sha(r.PROVIDER),calls=[],execs=[],probes=[],stagings=[],staging_error=False,fences=[],fence_lost=False,
                         lose_fence_after_submit=False,probe_timeout=False,probe_parse_error=False,overlap=False,
                         probe_names=['NVIDIA RTX A5000']*2,probe_code=0,error=False,returncode=0,evaluator_code=0,evaluator_error=False,stdout='910;synthetic\n',fresh_checks=[])
    value={'cells':cells,'models':{'7b':'literal/synthetic-model'},'files_sha256':scientific,
        'readiness_path':str(readiness),'readiness_sha256':b.sha(readiness),'view_manifest':str(c.manifest),
        'dependency_ids':[901],'current_disjointness':{d:{'synthetic_current_sources':1} for d in r.DOMAINS}}
    data.value=value
    def fence(fd):
        assert fd==19;data.fences.append(fd)
        if data.fence_lost:raise ValueError('synthetic fence lost')
    provider_review=c.file('artifacts/provider_review.json',{'files_sha256':{str(r.PROVIDER):b.sha(r.PROVIDER)}})
    def confirmation_inputs(root):
        r.require(not data.overlap,'synthetic new cross-source overlap')
        r.require(r.HISTORICAL_EVIDENCE.is_file() and b.sha(r.HISTORICAL_EVIDENCE)==r.EVIDENCE_SHA,'historical evidence missing before static proof')
        return value
    provider=SimpleNamespace(confirmation_inputs=confirmation_inputs,assert_fence=fence,REVIEW=provider_review)
    def inspect(cells,pins,models,*,fresh):
        data.fresh_checks.append(fresh)
        assert cells==value['cells'] and pins==scientific and models=={'7b':{'path':'literal/synthetic-model','label':'7b'}}
        for cell in cells:
            task=cell['tasks'][0]
            if fresh:r.require(not Path(task['output']).exists() and not Path(task['output']+'.batches').exists(),'preexisting calibration outputs')
        return {'synthetic_original_rng_admission':123,'child_seed_schedule':[11,12,13,14]}
    launcher=SimpleNamespace(HARDWARE={'synthetic':'sealed original'},RUNTIME_PROFILE={'thread_environment':{'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}},
        ORIGIN_CHAIN={'synthetic':'sealed'},source_pins=lambda:{str(r.LAUNCHER):b.sha(r.LAUNCHER)},
        checkpoint_identity=lambda path,label:{'path':path,'label':label},inspect_cells=inspect,
        job_command=lambda plan,cell:[plan['python'],'sealed-evaluator','--model',plan['models']['7b']['path'],
            '--tasks-json',cell['tasks'],'--resume','--confirm-eval'])
    monkeypatch.setattr(r,'modules',lambda digest:(b,launcher,provider))
    def contract():return {'schema':'modebench_scale_python_harder_portable_evidence_v1',
        'shared_source':str(r.SHARED_EVIDENCE),'historical_destination':str(r.HISTORICAL_EVIDENCE),
        'sha256':r.EVIDENCE_SHA,'overwrite_permitted':False,'science_or_live_fence_changed':False}
    def stage():
        data.stagings.append(r.socket.gethostname())
        if data.staging_error:raise ValueError('synthetic staging mismatch')
        created=not r.HISTORICAL_EVIDENCE.exists()
        if created:r.HISTORICAL_EVIDENCE.write_bytes(r.SHARED_EVIDENCE.read_bytes())
        r.require(b.sha(r.HISTORICAL_EVIDENCE)==r.EVIDENCE_SHA,'synthetic staging mismatch')
        return {**contract(),'observer_host':r.socket.gethostname(),'created_destination':created}
    def shared_bytes():
        r.require(b.sha(r.SHARED_EVIDENCE)==r.EVIDENCE_SHA,'shared diagnostic mismatch')
        return r.SHARED_EVIDENCE.read_bytes()
    stager=SimpleNamespace(contract=contract,stage=stage,shared_bytes=shared_bytes)
    monkeypatch.setattr(r,'load',lambda path,digest,name:b if path==r.BASE else stager if path==r.STAGER else pytest.fail('unexpected dependency loading'))
    def refresh_review():
        pins={str(path):b.sha(path) for path in (r.SOURCE,r.TESTS,r.PREDECESSOR_COPY,r.PREDECESSOR_TESTS_COPY,r.PREDECESSOR_MANIFEST,r.BASE,r.LAUNCHER,r.PROVIDER,provider.REVIEW,
                Path(value['readiness_path']),r.STAGER,r.STAGER_TESTS,r.SHARED_EVIDENCE,r.PRESERVATION,r.STAGER_REVIEW)}
        pins.update(value['files_sha256'])
        record={'schema':'modebench_scale_level4_first_transport_independent_review_v1','status':'reviewed',
            'provider_sha256':data.provider_sha,'readiness_sha256':value['readiness_sha256'],
            'dependency_ids':value['dependency_ids'],'operational_amendment':r.OPERATIONAL_AMENDMENT,'files_sha256':pins}
        if r.REVIEW.exists():rewrite(r.REVIEW,record)
        else:r.REVIEW.write_text(json.dumps(record))
        data.review_sha=b.sha(r.REVIEW)
    refresh_review();data.refresh_review=refresh_review

    def run(command,**kwargs):
        if '-c' in command:
            assert 'torch.cuda.get_device_name' in command[-1]
            assert kwargs=={'check':False,'text':True,'capture_output':True,'close_fds':True,'timeout':60}
            data.probes.append(command)
            if data.probe_timeout:raise subprocess.TimeoutExpired(command,60,output=b'partial probe')
            return SimpleNamespace(returncode=data.probe_code,stdout='invalid json' if data.probe_parse_error else json.dumps(data.probe_names),stderr='')
        if command[0]!='sbatch':
            assert kwargs=={'check':False,'close_fds':True}
            assert (root/'confirmation/runtime'/(os.environ['SLURM_ARRAY_TASK_ID']+'.json')).exists()
            if data.evaluator_error:raise OSError('synthetic spawn failure')
            data.execs.append((command[0],command))
            return SimpleNamespace(returncode=data.evaluator_code)
        assert command==b.read(root/'confirmation/plan.json')['submit_command']
        assert b.read(root/'confirmation/submission_intent.json')['command']==command
        assert kwargs['close_fds'] is True and kwargs['env']['TZ']=='UTC'
        data.calls.append(command)
        if data.error:raise subprocess.TimeoutExpired(command,60)
        if data.lose_fence_after_submit:data.fence_lost=True
        return SimpleNamespace(returncode=data.returncode,stdout=data.stdout,stderr='')
    monkeypatch.setattr(r,'subprocess',SimpleNamespace(run=run,TimeoutExpired=subprocess.TimeoutExpired))
    return data


def prepare(t):
    t.r.prepare(t.root,t.provider_sha,19,t.review_sha);return t.b.read(t.root/'confirmation/plan.json')


def submit(t):prepare(t);return t.r.submit(t.root,19)


def worker_env(t,monkeypatch,index=0):
    values={**t.r.ENVIRONMENT,'SLURMD_NODENAME':'node203','SLURM_JOB_ACCOUNT':'allcs','SLURM_JOB_PARTITION':'cs',
        'SLURM_ARRAY_JOB_ID':'910','SLURM_ARRAY_TASK_ID':str(index),'SLURM_JOB_ID':str(911+index),'SLURM_CPUS_PER_TASK':'6',
        'SLURM_MEM_PER_NODE':'61440','PYTHONPYCACHEPREFIX':'/tmp/NONPRODUCTION-scale-frozen-pycache-synthetic'}
    for key,value in values.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(t.r.socket,'gethostname',lambda:'node203.synthetic')
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'0.8.4')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2,get_device_name=lambda i:'NVIDIA RTX A5000')))


def test_canonical_five_cell_plan_derives_exact_scientific_commands_and_preserves_provider(t):
    before=Path(t.value['readiness_path']).read_bytes();plan=prepare(t)
    assert t.r.verify(t.root,fresh=True)==plan and Path(t.value['readiness_path']).read_bytes()==before
    assert [cell['domain'] for cell in plan['cells']]==list(t.r.DOMAINS)
    assert all(cell['level']=='level4' and cell['model_label']=='7b' for cell in plan['cells'])
    assert all(cell['command'][-2:]==['--resume','--confirm-eval'] for cell in plan['cells'])
    assert plan['rng_admission']['child_seed_schedule']==[11,12,13,14]
    assert '--account=allcs' in plan['submit_command'] and '--partition=cs' in plan['submit_command']
    assert '--array=0-4%3' in plan['submit_command'] and '--dependency=afterany:901' in plan['submit_command']
    assert not any('31259131' in item or item.startswith('--nodelist') for item in plan['submit_command'])
    assert t.calls==[] and len(t.fences)>=3
    assert plan['input_symlink_targets'][str(t.b.PYTHON)]==str(t.c.base/'python-target')


def test_duplicate_prepare_and_noncanonical_root_refused(t):
    prepare(t)
    with pytest.raises(ValueError,match='already prepared'):prepare(t)
    with pytest.raises(ValueError,match='canonical'):t.r.verify(t.root/'elsewhere')


@pytest.mark.parametrize('change',['level5','missing_domain','order','models','deps','ready_status','ready_sha'])
def test_five_cell_readiness_scope_is_exact(t,change):
    if change=='level5':t.value['cells'][0]['level']='level5'
    elif change=='missing_domain':t.value['cells'].pop()
    elif change=='order':t.value['cells'].reverse()
    elif change=='models':t.value['models']['14b']='model14b'
    elif change=='deps':t.value['dependency_ids'].append(31259131)
    elif change=='ready_status':
        p=Path(t.value['readiness_path']);v=t.b.read(p);v['status']='pending_fit';rewrite(p,v);t.value['readiness_sha256']=t.b.sha(p)
    else:t.value['readiness_sha256']='0'*64
    with pytest.raises(ValueError):prepare(t)
    assert t.calls==[] and not (t.root/'confirmation_preparation_claim.json').exists()


def test_prepare_or_submit_requires_actual_inherited_fence(t):
    t.fence_lost=True
    with pytest.raises(ValueError,match='fence lost'):prepare(t)
    assert not (t.root/'confirmation').exists()
    t.fence_lost=False;prepare(t);t.fence_lost=True
    with pytest.raises(ValueError,match='fence lost'):t.r.submit(t.root,19)
    assert t.calls==[] and not (t.root/'confirmation/submission_intent.json').exists()


@pytest.mark.parametrize('change',['science','provider','worker','task','rng','route','command'])
def test_changed_prepared_inputs_or_plan_refused_before_submit(t,change):
    p=prepare(t)
    if change=='science':Path(next(iter(t.value['files_sha256']))).write_text('different proof')
    elif change=='provider':t.r.PROVIDER.write_text('different provider')
    elif change=='worker':(t.root/'confirmation/worker.slurm').chmod(0o644);(t.root/'confirmation/worker.slurm').write_text('bad')
    elif change=='task':rewrite(Path(p['cells'][0]['tasks']),[{'wrong':True}])
    else:
        if change=='rng':p['rng_admission']={}
        elif change=='route':p['hardware']={**p['hardware'],'account':'cs','partition':'allcs'}
        else:p['cells'][0]['command'].append('--different')
        rewrite(t.root/'confirmation/plan.json',p);rewrite(t.root/'confirmation/plan.sha256.json',{'sha256':t.b.sha(t.root/'confirmation/plan.json')})
    with pytest.raises(ValueError):t.r.submit(t.root,19)
    assert t.calls==[]


def test_fresh_output_required_before_submission(t):
    prepare(t);output=Path(t.value['cells'][0]['tasks'][0]['output']);output.parent.mkdir(parents=True);output.write_text('{}')
    with pytest.raises(ValueError,match='preexisting'):t.r.submit(t.root,19)
    assert t.calls==[]


def test_one_canonical_submit_and_readonly_identity(t):
    result=submit(t);assert t.r.submission_identity(t.root)==result and result['array_job_id']==910
    with pytest.raises(ValueError,match='already attempted'):t.r.submit(t.root,19)
    assert len(t.calls)==1


@pytest.mark.parametrize('failure',['timeout','nonzero','empty','dependency'])
def test_failed_or_ambiguous_submission_cannot_repeat(t,failure):
    prepare(t)
    if failure=='timeout':t.error=True
    elif failure=='nonzero':t.returncode=1
    elif failure=='empty':t.stdout=''
    else:t.stdout='901\n'
    with pytest.raises(ValueError):t.r.submit(t.root,19)
    with pytest.raises(ValueError,match='already attempted'):t.r.submit(t.root,19)
    assert len(t.calls)==1 and (t.root/'confirmation/submission_ambiguous.json').exists()


@pytest.mark.parametrize('index',range(5))
def test_actual_worker_records_exact_level4_cell_then_original_evaluator_exec(t,monkeypatch,index):
    submit(t);worker_env(t,monkeypatch,index);t.r.worker(t.root,index)
    p=t.b.read(t.root/'confirmation/plan.json');v=t.b.read(t.root/f'confirmation/runtime/{index}.json')
    assert v['array_index']==index and v['cell']==p['cells'][index]['id'] and v['level']=='level4'
    assert v['environment']['SLURM_JOB_ACCOUNT']=='allcs' and v['environment']['SLURM_JOB_PARTITION']=='cs'
    assert v['readiness_sha256']==t.value['readiness_sha256']
    assert t.execs==[(p['cells'][index]['command'][0],p['cells'][index]['command'])]
    with pytest.raises(ValueError,match='already claimed'):t.r.worker(t.root,index)
    assert len(t.execs)==1


@pytest.mark.parametrize('key,value',[('SLURM_JOB_ACCOUNT','mltheory'),('SLURM_JOB_ACCOUNT','cs'),('SLURM_JOB_PARTITION','allcs'),
    ('SLURM_ARRAY_JOB_ID','31259131'),('SLURM_CPUS_PER_TASK','12'),('SLURM_MEM_PER_NODE','60000'),('VLLM_USE_V1','1')])
def test_worker_rejects_actual_resource_or_identity_drift(t,monkeypatch,key,value):
    submit(t);worker_env(t,monkeypatch);monkeypatch.setenv(key,value)
    with pytest.raises(ValueError,match='allocation differs'):t.r.worker(t.root,0)
    assert t.execs==[]


def test_worker_rejects_unregistered_target_batches(t,monkeypatch):
    submit(t);worker_env(t,monkeypatch);Path(t.value['cells'][0]['tasks'][0]['output']+'.batches').mkdir(parents=True)
    with pytest.raises(ValueError,match='heldout output'):t.r.worker(t.root,0)
    assert t.execs==[]


def test_other_completed_cell_output_does_not_block_next_independent_worker(t,monkeypatch):
    submit(t);worker_env(t,monkeypatch,0);t.r.worker(t.root,0)
    output=Path(t.value['cells'][0]['tasks'][0]['output']);output.parent.mkdir(parents=True);output.write_text('{}')
    worker_env(t,monkeypatch,1);t.r.worker(t.root,1)
    assert len(t.execs)==2


@pytest.mark.parametrize('returncode',[0,1,143,-15])
def test_actual_evaluator_exit_is_recorded_without_claiming_scheduler_success(t,monkeypatch,returncode):
    submit(t);worker_env(t,monkeypatch);t.evaluator_code=returncode
    assert t.r.worker(t.root,0)==returncode
    report=t.b.read(t.root/'confirmation/runtime/0.evaluator_exit.json')
    assert report['returncode']==returncode and report['scheduler_success_claimed'] is False
    assert report['command']==t.b.read(t.root/'confirmation/plan.json')['cells'][0]['command']
    assert report['runtime_sha256']==t.b.sha(t.root/'confirmation/runtime/0.json')
    assert report['started_at_utc']<=report['finished_at_utc']
    with pytest.raises(ValueError,match='already claimed'):t.r.worker(t.root,0)
    assert len(t.execs)==1


def test_evaluator_spawn_failure_keeps_claim_and_never_retries(t,monkeypatch):
    submit(t);worker_env(t,monkeypatch);t.evaluator_error=True
    with pytest.raises(OSError,match='synthetic spawn'):t.r.worker(t.root,0)
    assert (t.root/'confirmation/runtime/0.evaluator_spawn_failure.json').exists()
    assert not (t.root/'confirmation/runtime/0.evaluator_exit.json').exists()
    with pytest.raises(ValueError,match='already claimed'):t.r.worker(t.root,0)


def test_real_sealed_launcher_argv_and_rng_inspection_compose_with_transport(t,monkeypatch):
    """Keep the real launcher functions; substitute only synthetic scientific data."""
    path=ROOT/'ops/exp_scaling/launch_modebench_scale_domains.py'
    spec=importlib.util.spec_from_file_location('_actual_launcher_for_level4_integration',path)
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    _,_,provider=t.r.modules(t.provider_sha)
    monkeypatch.setattr(t.r,'LAUNCHER',path)
    monkeypatch.setattr(t.r,'modules',lambda digest:(t.b,launcher,provider))
    identity={'synthetic_checkpoint':'7b'};validate_calls=[]
    for cell in t.value['cells']:
        source=Path(cell['source_root']);domain=cell['domain'];base=source/'level4/dataset'/domain
        files={source/'protocol.json':{'models':{'7b':identity}},base/'identity.json':{'synthetic':'frozen128'},
            source/'level4/recipes'/f'{domain}.json':{'synthetic':'fixed passing recipe'},base/'eval.jsonl':'synthetic rows only'}
        for filename,value in files.items():
            filename.parent.mkdir(parents=True,exist_ok=True);filename.write_text(json.dumps(value))
            t.value['files_sha256'][str(filename)]=t.b.sha(filename)
        task=cell['tasks'][0];task['rows_jsonl']=str(base/'eval.jsonl');task['interface']='scratch_original_interface'
    def schedule(domain,rows,seeds):
        prefix=list(t.r.DOMAINS).index(domain)*10000
        return {'request_seeds':[[prefix+i*4+j for j in range(4)] for i in range(len(rows))]}
    evaluator=SimpleNamespace(INTERFACE='scratch_original_interface',model_identity=lambda path,label:identity,
        validate_task=lambda task,confirm:validate_calls.append((task['domain'],confirm)),
        load_rows=lambda task:([{'synthetic':i} for i in range(128)],{'row_offset':0,'row_limit':0}),
        frozen=SimpleNamespace(POLICY='scratch-disjoint-seeds',schedule_record=schedule),sha=lambda obj:json.dumps(obj,sort_keys=True))
    monkeypatch.setattr(launcher,'evaluator',lambda:evaluator)
    monkeypatch.setattr(launcher,'checkpoint_identity',lambda path,label:{'path':path,'label':label})
    monkeypatch.setattr(launcher,'source_pins',lambda:{str(path):t.b.sha(path)})
    readiness=Path(t.value['readiness_path']);ready=t.b.read(readiness);ready['files_sha256']=t.value['files_sha256']
    rewrite(readiness,ready);t.value['readiness_sha256']=t.b.sha(readiness);t.refresh_review()
    plan=prepare(t);assert t.r.verify(t.root,fresh=True)==plan
    assert plan['rng_admission']['distinct_request_blocks']==5*128*4
    assert plan['rng_admission']['distinct_child_seeds']==5*128*32
    assert {domain for domain,confirm in validate_calls if confirm}==set(t.r.DOMAINS)
    for cell in plan['cells']:
        command=cell['command']
        assert command==launcher.job_command({**plan,'concurrency':1},cell)==launcher.job_command({**plan,'concurrency':3},cell)
        assert command[0]==str(t.b.PYTHON) and command[1]==str(ROOT/'ops/evaluate_modebench_scale.py')
        assert '--confirm-eval' in command and '--resume' in command
        assert command[command.index('--tensor-parallel-size')+1]=='2'
        assert command[command.index('--max-model-len')+1]==('1024' if cell['domain']=='graph_coloring' else '2048')
    assert plan['concurrency']==3 and plan['runtime_profile']['concurrency']==1
    assert plan['operational_amendment']['old_scheduler_concurrency_guards_used'] is False
    assert launcher.CONCURRENCY==1 and launcher.runtime.MAX_CONCURRENCY==2
    with pytest.raises(ValueError,match='concurrency one'):launcher.scheduler_command('/unused',plan)
    assert t.calls==[] and t.execs==[]


@pytest.mark.parametrize('names,code',[(['NVIDIA A100']*2,0),(['NVIDIA RTX A5000'],0),(['NVIDIA RTX A5000']*2,1)])
def test_metadata_probe_must_confirm_two_a5000s_and_exit_before_evaluator(t,monkeypatch,names,code):
    submit(t);worker_env(t,monkeypatch);t.probe_names=names;t.probe_code=code
    with pytest.raises(ValueError):t.r.worker(t.root,0)
    assert t.execs==[] and not (t.root/'confirmation/runtime/0.json').exists()


def test_explicit_operational_amendment_keeps_all_per_cell_resources_and_five_outputs(t):
    plan=prepare(t);command=plan['submit_command']
    assert plan['concurrency']==3 and plan['operational_amendment']==t.r.OPERATIONAL_AMENDMENT
    assert plan['operational_amendment']['actual_array_max_gpus']==6
    for flag in ('--array=0-4%3','--gres=gpu:a5000:2','--cpus-per-task=6','--mem=60G','--time=08:00:00',
                 '--nodes=1','--ntasks=1','--no-requeue','--account=allcs','--partition=cs'):
        assert flag in command
    tasks=[t.b.read(cell['tasks'])[0] for cell in plan['cells']]
    assert len({task['output'] for task in tasks})==5
    assert all(task['seeds']==[11,12,13,14] for task in tasks)
    assert plan['rng_admission']['child_seed_schedule']==[11,12,13,14]


@pytest.mark.parametrize('field',['concurrency','operational_amendment'])
def test_final_plan_cannot_reinterpret_registered_operational_amendment(t,field):
    plan=prepare(t)
    plan[field]=1 if field=='concurrency' else {}
    rewrite(t.root/'confirmation/plan.json',plan)
    rewrite(t.root/'confirmation/plan.sha256.json',{'sha256':t.b.sha(t.root/'confirmation/plan.json')})
    with pytest.raises(ValueError,match='contract changed'):t.r.submit(t.root,19)
    assert not t.calls


def test_final_review_must_bind_explicit_operational_amendment(t):
    record=t.b.read(t.r.REVIEW);record.pop('operational_amendment');rewrite(t.r.REVIEW,record)
    t.review_sha=t.b.sha(t.r.REVIEW)
    with pytest.raises(ValueError,match='reviewed exact'):prepare(t)
    assert not t.calls and not (t.root/'confirmation_preparation_claim.json').exists()


@pytest.mark.parametrize('index',range(5))
def test_every_domain_stages_missing_historical_diagnostic_before_static_proof(t,monkeypatch,index):
    submit(t);worker_env(t,monkeypatch,index);t.r.HISTORICAL_EVIDENCE.unlink()
    with pytest.raises(ValueError,match='missing before static'):t.r.inputs(t.root,t.provider_sha)
    assert t.r.worker(t.root,index)==0
    runtime=t.b.read(t.root/f'confirmation/runtime/{index}.json')
    assert runtime['portable_evidence_staging']['created_destination'] is True
    assert runtime['portable_evidence_staging']['observer_host']=='node203.synthetic'
    assert runtime['current_disjointness']==t.value['current_disjointness']
    assert len(t.probes)==len(t.execs)==1


def test_staging_mismatch_fails_before_any_static_worker_or_gpu_action(t,monkeypatch):
    submit(t);worker_env(t,monkeypatch);t.staging_error=True
    with pytest.raises(ValueError,match='staging mismatch'):t.r.worker(t.root,0)
    assert not t.probes and not t.execs


def test_fresh_cross_source_observations_can_grow_but_overlap_blocks_eval(t,monkeypatch):
    submit(t);worker_env(t,monkeypatch)
    plan=t.b.read(t.root/'confirmation/plan.json')
    assert t.b.read(t.root/'confirmation_preparation_claim.json')['current_disjointness_at_preparation']['pantry']=={'synthetic_current_sources':1}
    t.value['current_disjointness']['pantry']={'synthetic_current_sources':2}
    assert t.r.verify(t.root)==plan
    t.r.worker(t.root,0)
    assert t.b.read(t.root/'confirmation/runtime/0.json')['current_disjointness']['pantry']=={'synthetic_current_sources':2}
    worker_env(t,monkeypatch,1);t.overlap=True
    with pytest.raises(ValueError,match='new cross-source overlap'):t.r.worker(t.root,1)
    assert len(t.execs)==1 and not (t.root/'confirmation/runtime/1.json').exists()


@pytest.mark.parametrize('kind',['timeout','parse','returncode','names'])
def test_short_gpu_probe_failures_are_recorded_and_never_retried(t,monkeypatch,kind):
    submit(t);worker_env(t,monkeypatch)
    if kind=='timeout':t.probe_timeout=True
    elif kind=='parse':t.probe_parse_error=True
    elif kind=='returncode':t.probe_code=1
    else:t.probe_names=['NVIDIA A100']*2
    with pytest.raises((ValueError,subprocess.TimeoutExpired)):t.r.worker(t.root,0)
    record=t.b.read(t.root/'confirmation/runtime/0.gpu_probe_failure.json')
    assert record['probe']['timeout_seconds']==60 and record['scheduler_success_claimed'] is False
    if kind=='timeout':assert record['probe']['stdout_bytes_base64']=='cGFydGlhbCBwcm9iZQ=='
    with pytest.raises(ValueError,match='GPU probe failed'):t.r.worker(t.root,0)
    assert not t.execs and len(t.probes)==1


@pytest.mark.parametrize('action',['prepare','submit'])
def test_only_actual_soak_may_prepare_or_submit(t,monkeypatch,action):
    if action=='submit':prepare(t)
    monkeypatch.setattr(t.r.socket,'gethostname',lambda:'node202.synthetic')
    with pytest.raises(ValueError,match='truthful same-user soak'):
        prepare(t) if action=='prepare' else t.r.submit(t.root,19)
    assert not t.calls


@pytest.mark.parametrize('field',['schema','source_sha256','provider_sha256','readiness_sha256','submission_token','review_sha256'])
def test_preparation_claim_fields_are_rebound_beyond_digest_links(t,field):
    plan=prepare(t);path=t.root/'confirmation_preparation_claim.json';claim=t.b.read(path);claim[field]='other';rewrite(path,claim)
    plan['inputs_sha256'][str(path)]=t.b.sha(path);rewrite(t.root/'confirmation/plan.json',plan)
    rewrite(t.root/'confirmation/plan.sha256.json',{'sha256':t.b.sha(t.root/'confirmation/plan.json')})
    with pytest.raises(ValueError,match='claim identity'):t.r.verify(t.root)
    assert not t.calls


def test_fence_loss_after_sbatch_preserves_actual_result_and_refuses_retry(t):
    prepare(t);t.lose_fence_after_submit=True
    with pytest.raises(ValueError,match='fence lost'):t.r.submit(t.root,19)
    result=t.b.read(t.root/'confirmation/submission_result.json')
    assert result['array_job_id']==910 and result['status']=='submitted' and len(t.calls)==1
    t.fence_lost=False
    with pytest.raises(ValueError,match='already attempted'):t.r.submit(t.root,19)
    assert len(t.calls)==1


@pytest.mark.parametrize('kind',['missing_review','placeholder_review','not_actual_readiness','omitted_closure','portable_hash'])
def test_prospective_transport_cannot_activate_without_exact_final_review(t,kind):
    if kind=='missing_review':t.r.REVIEW.unlink()
    elif kind=='placeholder_review':t.review_sha='NOT_YET_REVIEWED'
    elif kind=='portable_hash':t.r.SHARED_EVIDENCE.write_text('different')
    else:
        record=t.b.read(t.r.REVIEW)
        if kind=='not_actual_readiness':record['readiness_sha256']='0'*64
        else:record['files_sha256'].pop(next(iter(t.value['files_sha256'])))
        rewrite(t.r.REVIEW,record);t.review_sha=t.b.sha(t.r.REVIEW)
    with pytest.raises((ValueError,FileNotFoundError)):prepare(t)
    assert not t.calls and not (t.root/'confirmation_preparation_claim.json').exists()


def test_real_release_provider_static_inputs_compose_and_observe_new_disjoint_sources(t,release_fixture,monkeypatch):
    x=release_fixture;x.run();before=list(x.f.f.calls)
    class ForeignLock:
        def stat(self):raise AssertionError('static worker proof must not replay historical NFS device')
    monkeypatch.setattr(x.a,'LOCK',ForeignLock());monkeypatch.setattr(x.f.m.first,'LOCK',ForeignLock())
    x.f.f.canonical.write_text('old')
    monkeypatch.setattr(t.r.socket,'gethostname',lambda:'node203.synthetic')
    monkeypatch.setattr(t.r,'STATE',x.state)
    monkeypatch.setattr(t.r,'modules',lambda digest:(t.b,None,x.a))
    monkeypatch.setattr(t.b,'frozen_view',lambda path:{'synthetic':'source view'})
    first=t.r.inputs(x.state,x.a.sha(x.a.SOURCE))[-1]
    assert [cell['domain'] for cell in first['cells']]==list(t.r.DOMAINS)
    assert first['dependency_ids']==[31261726] and first['models']=={'7b':'synthetic7b'}
    assert first['cells'][2]['tasks'][0]['seeds']==[7204500,7204501,7204502,7204503]
    extra=x.state.parent/'new_external_source.json';extra.write_text('{}')
    x.new_sources[str(extra)]=x.a.sha(extra)
    out=Path(first['cells'][0]['tasks'][0]['output']);out.parent.mkdir(parents=True,exist_ok=True);out.write_text('{}')
    second=t.r.inputs(x.state,x.a.sha(x.a.SOURCE))[-1]
    assert first['files_sha256']==second['files_sha256'] and first['cells']==second['cells']
    assert first['current_disjointness']!=second['current_disjointness'] and x.f.f.calls==before
    x.overlap.add('pantry')
    with pytest.raises(ValueError,match='external overlap'):t.r.inputs(x.state,x.a.sha(x.a.SOURCE))


@pytest.mark.parametrize('action',['prepare','submit'])
@pytest.mark.parametrize('kind',['spin','wash','wrong_uname','wrong_uid','wrong_euid'])
def test_soak_action_identity_rejects_before_proof_or_submission(t,monkeypatch,action,kind):
    if action=='submit':prepare(t)
    if kind in ('spin','wash'):
        monkeypatch.setattr(t.r.socket,'gethostname',lambda:kind+'.cs.princeton.edu')
        monkeypatch.setattr(t.r.os,'uname',lambda:SimpleNamespace(nodename=kind+'.cs.princeton.edu'))
    elif kind=='wrong_uname':monkeypatch.setattr(t.r.os,'uname',lambda:SimpleNamespace(nodename='node202'))
    elif kind=='wrong_uid':monkeypatch.setattr(t.r.os,'getuid',lambda:t.r.UID+1)
    else:monkeypatch.setattr(t.r.os,'geteuid',lambda:t.r.UID+1)
    monkeypatch.setattr(t.r,'inputs',lambda *args:pytest.fail('invalid new action host must precede proof'))
    monkeypatch.setattr(t.r,'verified',lambda *args,**kwargs:pytest.fail('invalid new action host must precede proof'))
    with pytest.raises(ValueError,match='truthful same-user soak'):
        prepare(t) if action=='prepare' else t.r.submit(t.root,19)
    assert not t.calls


def test_new_submitter_host_preserves_original_workers_stager_and_resources():
    r=load()
    assert r.HOST=='soak.cs.princeton.edu' and r.UID==363432
    assert r.NODES==['node202','node203','node204']
    assert r.STAGER==ROOT/'artifacts/stage_modebench_scale_python_harder_evidence_20260912.py'
    assert r.HARDWARE=={'account':'allcs','partition':'cs','gres':'gpu:a5000:2','nodes':1,
        'cpus':6,'memory':'60G','time':'08:00:00','allowed_worker_nodes':r.NODES}
    assert r.CONCURRENCY==3 and r.OPERATIONAL_AMENDMENT['actual_array_max_gpus']==6
    assert r.ENVIRONMENT=={'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','VLLM_USE_V1':'0',
        'VLLM_ATTENTION_BACKEND':'XFORMERS','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'PYTHONDONTWRITEBYTECODE':'1'}


@pytest.mark.parametrize('action',['prepare','submit'])
@pytest.mark.parametrize('kind',['correct','relative_python','relative_source','missing_frozen_flag'])
def test_exact_frozen_command_guard(action,kind):
    r=load();python=ROOT/'var/seed_paper_eval/paper310/bin/python'
    command=[str(python),'-B','-B',str(r.SOURCE),action]
    if kind=='relative_python':command[0]='var/seed_paper_eval/paper310/bin/python'
    elif kind=='relative_source':command[3]='artifacts/'+r.SOURCE.name
    elif kind=='missing_frozen_flag':command.pop(1)
    calls=[]
    provider=SimpleNamespace(PYTHON=python,live_host_guard=lambda **kw:calls.append(kw),
        process_identity=lambda pid:{'command':command,'pid':pid})
    if kind=='correct':assert r.command_guard(provider,action)['command']==command
    else:
        with pytest.raises(ValueError,match='full absolute'):r.command_guard(provider,action)
    assert calls==[{'guest':True}]


@pytest.mark.parametrize('kind',['host','pid','uid','start','command'])
def test_rehashed_preparation_owner_drift_rejected(t,kind):
    plan=prepare(t);path=t.root/'confirmation_preparation_claim.json';claim=t.b.read(path)
    if kind=='host':claim['host']='spin.cs.princeton.edu'
    elif kind=='pid':claim['owner']['pid']=0
    elif kind=='uid':claim['owner']['uid']+=1
    elif kind=='start':claim['owner']['start_ticks']=''
    else:claim['owner']['command'][0]='relative/python'
    rewrite(path,claim);plan['inputs_sha256'][str(path)]=t.b.sha(path)
    rewrite(t.root/'confirmation/plan.json',plan)
    rewrite(t.root/'confirmation/plan.sha256.json',{'sha256':t.b.sha(t.root/'confirmation/plan.json')})
    with pytest.raises(ValueError,match='preparation owner'):t.r.verify(t.root)
