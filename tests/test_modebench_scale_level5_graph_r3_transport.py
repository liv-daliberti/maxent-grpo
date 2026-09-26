"""Scratch one-cell transport integration; no real jobs, GPUs or science."""
import copy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from test_modebench_scale_level5_graph_r3_registration import fixture as prep_fixture, fit_fixture

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_graph_r3_base_fixture',ROOT/'tests/test_modebench_scale_revised_node_failure_recovery.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
c=previous.c


def rewrite(path,value):
    path=Path(path);path.chmod(0o644);path.write_text(json.dumps(value))


@pytest.fixture
def t(c,monkeypatch):
    spec=importlib.util.spec_from_file_location('_scratch_graph_r3_transport',ROOT/'artifacts/modebench_scale_level5_graph_r3_transport_20260912.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    b=c.r;root=c.base/'graph_r3';root.mkdir()
    for key,value in {'ROOT':c.base,'SOURCE':c.file('artifacts/transport.py'),'PROVIDER':c.file('artifacts/provider.py'),
        'BASE':b.SOURCE,'LAUNCHER':c.file('artifacts/launcher.py'),'STATE':root,'REVISION_ROOT':c.base/'source/graph_r3',
        'TESTS':c.file('tests/transport.py'),'REVIEW':c.base/'artifacts/review.json'}.items():monkeypatch.setattr(r,key,value)
    proof=c.file('graph_r3/preparation.json',{'synthetic':'registered Graph r3 native pools'})
    scientific={str(proof):b.sha(proof)}
    readiness=c.file('graph_r3/certificate.json',{'schema':r.READY_SCHEMA,'level':'level5',
        'status':'ready_for_fresh_graph_r3_development','files_sha256':scientific})
    provider_review=c.file('artifacts/provider_review.json',{'files_sha256':{str(r.PROVIDER):b.sha(r.PROVIDER)}})
    cell={'id':'level5_graph_coloring_r3_dev','level':'level5','domain':'graph_coloring','phase':'dev',
        'source_root':str(r.REVISION_ROOT),'source_kind':'domain_revision_v1',
        'tasks':[{'level':'level5','domain':'graph_coloring','split':'dev',
            'output':str(r.REVISION_ROOT/'level5/results/development/graph_coloring'/f'difficulty_{tier}.json'),
            'seeds':r.LABELS,'batch_size':8,'row_offset':0,'row_limit':0,
            'rows_jsonl':str(r.REVISION_ROOT/'level5/pools/graph_coloring'/f'difficulty_{tier}.jsonl')} for tier in range(4)]}
    value={'cell':cell,'models':{'14b':'literal/synthetic-model'},'files_sha256':scientific,
        'readiness_path':str(readiness),'readiness_sha256':b.sha(readiness),'view_manifest':str(c.manifest)}
    monkeypatch.setattr(r.socket,'gethostname',lambda:r.HOST)
    data=SimpleNamespace(r=r,b=b,c=c,root=root,value=value,provider_sha=b.sha(r.PROVIDER),calls=[],execs=[],probes=[],
        fences=[],fence_lost=False,lose_fence_after_submit=False,probe_names=['NVIDIA RTX A5000']*2,probe_code=0,error=False,returncode=0,
        evaluator_code=0,evaluator_error=False,probe_timeout=False,probe_parse_error=False,stdout='910;synthetic\n',fresh_checks=[])
    def fence(fd):
        assert fd==19;data.fences.append(fd)
        if data.fence_lost:raise ValueError('synthetic fence lost')
    provider=SimpleNamespace(development_inputs=lambda root:value,assert_fence=fence,REVIEW=provider_review)
    def inspect(cells,pins,models,*,fresh):
        data.fresh_checks.append(fresh)
        assert cells==[value['cell']] and pins==scientific and models=={'14b':{'path':'literal/synthetic-model','label':'14b'}}
        for task in cells[0]['tasks']:
            if fresh:r.require(not Path(task['output']).exists() and not Path(task['output']+'.batches').exists(),'preexisting development output')
        return {'synthetic_rng_admission':123,'child_seed_schedule':r.LABELS}
    launcher=SimpleNamespace(HARDWARE={'synthetic':'sealed original'},RUNTIME_PROFILE={'thread_environment':{'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}},
        ORIGIN_CHAIN={'synthetic':'sealed'},source_pins=lambda:{str(r.LAUNCHER):b.sha(r.LAUNCHER)},
        checkpoint_identity=lambda path,label:{'path':path,'label':label},inspect_cells=inspect,
        job_command=lambda plan,cell:[plan['python'],'sealed-evaluator','--model',plan['models']['14b']['path'],
            '--tasks-json',cell['tasks'],'--resume'])
    monkeypatch.setattr(r,'modules',lambda digest:(b,launcher,provider))
    monkeypatch.setattr(r,'load',lambda path,digest,name:b if path==r.BASE else pytest.fail('unexpected load'))
    def refresh_review():
        record={'schema':'modebench_scale_level5_graph_r3_transport_independent_review_v1','status':'reviewed',
            'files_sha256':{str(p):b.sha(p) for p in (r.SOURCE,r.TESTS,r.BASE,r.LAUNCHER,r.PROVIDER,provider.REVIEW)}}
        if r.REVIEW.exists():rewrite(r.REVIEW,record)
        else:c.file(str(r.REVIEW.relative_to(c.base)),record)
        data.review_sha=b.sha(r.REVIEW)
    refresh_review();data.refresh_review=refresh_review
    def run(command,**kwargs):
        if '-c' in command:
            assert 'torch.cuda.get_device_name' in command[-1]
            assert kwargs=={'check':False,'text':True,'capture_output':True,'close_fds':True,'timeout':60}
            data.probes.append(command)
            if data.probe_timeout:raise subprocess.TimeoutExpired(command,60,output=b'partial probe bytes')
            return SimpleNamespace(returncode=data.probe_code,stdout='invalid json' if data.probe_parse_error else json.dumps(data.probe_names),stderr='')
        if command[0]!='sbatch':
            assert kwargs=={'check':False,'close_fds':True}
            assert (root/'development/runtime/0.json').exists() and len(data.probes)==1
            if data.evaluator_error:raise OSError('synthetic spawn failure')
            data.execs.append(command)
            return SimpleNamespace(returncode=data.evaluator_code)
        assert command==b.read(root/'development/plan.json')['submit_command']
        assert b.read(root/'development/submission_intent.json')['command']==command
        assert kwargs['close_fds'] and kwargs['env']['TZ']=='UTC'
        data.calls.append(command)
        if data.error:raise subprocess.TimeoutExpired(command,60)
        if data.lose_fence_after_submit:data.fence_lost=True
        return SimpleNamespace(returncode=data.returncode,stdout=data.stdout,stderr='')
    monkeypatch.setattr(r,'subprocess',SimpleNamespace(run=run,TimeoutExpired=subprocess.TimeoutExpired))
    return data


def prepare(t):
    t.r.prepare(t.root,t.provider_sha,19,t.review_sha)
    return t.b.read(t.root/'development/plan.json')


def submit(t):prepare(t);return t.r.submit(t.root,19)


def worker_env(t,monkeypatch):
    values={**t.r.ENVIRONMENT,'SLURMD_NODENAME':'node203','SLURM_JOB_ACCOUNT':'allcs','SLURM_JOB_PARTITION':'cs',
        'SLURM_ARRAY_JOB_ID':'910','SLURM_ARRAY_TASK_ID':'0','SLURM_JOB_ID':'911','SLURM_CPUS_PER_TASK':'6',
        'SLURM_MEM_PER_NODE':'61440','PYTHONPYCACHEPREFIX':'/tmp/NONPRODUCTION-scale-frozen-pycache-synthetic'}
    for key,value in values.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(t.r.socket,'gethostname',lambda:'node203.synthetic')
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'0.8.4')


def test_one_canonical_cell_four_tiers_and_independence_from_live_python(t):
    plan=prepare(t)
    assert t.r.verify(t.root,fresh=True)==plan
    assert len(plan['cells'])==1 and plan['cells'][0]['model_label']=='14b'
    assert len(t.b.read(plan['cells'][0]['tasks']))==4
    assert '--array=0-0%1' in plan['submit_command'] and '--account=allcs' in plan['submit_command']
    assert '--partition=cs' in plan['submit_command'] and '--dependency=afterany:31254520' in plan['submit_command']
    assert not any('31259131' in x or x.startswith('--nodelist') for x in plan['submit_command'])
    assert t.calls==t.execs==t.probes==[] and len(t.fences)>=3
    assert plan['input_symlink_targets'][str(t.b.PYTHON)]==str(t.c.base/'python-target')


@pytest.mark.parametrize('kind',['wrong_root','wrong_level','wrong_phase','wrong_domain','wrong_seed','partial','missing_tier','wrong_model','not_ready'])
def test_exact_prepared_scientific_scope_before_any_plan(t,kind):
    cell=t.value['cell']
    if kind=='wrong_root':cell['source_root']='/other'
    elif kind=='wrong_level':cell['level']='level4'
    elif kind=='wrong_phase':cell['phase']='eval'
    elif kind=='wrong_domain':cell['domain']='pantry'
    elif kind=='wrong_seed':cell['tasks'][0]['seeds']=[1,2,3,4]
    elif kind=='partial':cell['tasks'][0]['row_limit']=128
    elif kind=='missing_tier':cell['tasks'].pop()
    elif kind=='wrong_model':t.value['models']={'7b':'other'}
    else:
        p=Path(t.value['readiness_path']);v=t.b.read(p);v['status']='pending';rewrite(p,v);t.value['readiness_sha256']=t.b.sha(p)
    with pytest.raises(ValueError):prepare(t)
    assert not (t.root/'development').exists() and not t.calls


def test_inherited_original_fence_required_before_prepare_and_submit(t):
    t.fence_lost=True
    with pytest.raises(ValueError,match='fence lost'):prepare(t)
    assert not (t.root/'development').exists()
    t.fence_lost=False;prepare(t);t.fence_lost=True
    with pytest.raises(ValueError,match='fence lost'):t.r.submit(t.root,19)
    assert not t.calls


@pytest.mark.parametrize('kind',['science','provider','worker','tasks','runtime_profile','route','argv','rng'])
def test_pinned_prepared_plan_cannot_drift(t,kind):
    plan=prepare(t)
    if kind=='science':Path(next(iter(t.value['files_sha256']))).write_text('changed')
    elif kind=='provider':t.r.PROVIDER.write_text('changed')
    elif kind=='worker':p=t.root/'development/worker.slurm';p.chmod(0o644);p.write_text('changed')
    elif kind=='tasks':rewrite(plan['cells'][0]['tasks'],[])
    else:
        if kind=='runtime_profile':plan['runtime_profile']={}
        elif kind=='route':plan['hardware']['partition']='mltheory'
        elif kind=='argv':plan['cells'][0]['command'].append('--confirm-eval')
        else:plan['rng_admission']={}
        rewrite(t.root/'development/plan.json',plan)
        rewrite(t.root/'development/plan.sha256.json',{'sha256':t.b.sha(t.root/'development/plan.json')})
    with pytest.raises(ValueError):t.r.submit(t.root,19)
    assert not t.calls


@pytest.mark.parametrize('failure',['timeout','bad_stdout','nonzero'])
def test_submission_ambiguity_never_retries(t,failure):
    prepare(t)
    if failure=='timeout':t.error=True
    elif failure=='bad_stdout':t.stdout='not a job id\n'
    else:t.returncode=1
    with pytest.raises(ValueError):t.r.submit(t.root,19)
    assert len(t.calls)==1 and (t.root/'development/submission_ambiguous.json').exists()
    with pytest.raises(ValueError,match='already attempted'):t.r.submit(t.root,19)
    assert len(t.calls)==1


def test_duplicate_prepare_submit_or_worker_never_repeats(t,monkeypatch):
    submit(t)
    with pytest.raises(ValueError,match='already prepared'):prepare(t)
    with pytest.raises(ValueError,match='already attempted'):t.r.submit(t.root,19)
    worker_env(t,monkeypatch);t.r.worker(t.root,0)
    with pytest.raises(ValueError,match='already claimed'):t.r.worker(t.root,0)
    assert len(t.calls)==len(t.probes)==len(t.execs)==1


@pytest.mark.parametrize('key,value',[('SLURM_JOB_ACCOUNT','mltheory'),('SLURM_JOB_PARTITION','all'),
    ('SLURM_ARRAY_TASK_ID','1'),('SLURM_ARRAY_JOB_ID','999'),('SLURM_CPUS_PER_TASK','4'),
    ('SLURM_MEM_PER_NODE','49152'),('OMP_NUM_THREADS','1'),('VLLM_USE_V1','1')])
def test_actual_allocation_and_scientific_environment_checked(t,monkeypatch,key,value):
    submit(t);worker_env(t,monkeypatch);monkeypatch.setenv(key,value)
    with pytest.raises(ValueError):t.r.worker(t.root,0)
    assert not t.execs and not t.probes


@pytest.mark.parametrize('names,code',[(['NVIDIA A100']*2,0),(['NVIDIA RTX A5000'],0),(['NVIDIA RTX A5000']*2,1)])
def test_short_gpu_probe_finishes_before_evaluator_and_requires_exact_hardware(t,monkeypatch,names,code):
    submit(t);worker_env(t,monkeypatch);t.probe_names=names;t.probe_code=code
    with pytest.raises(ValueError):t.r.worker(t.root,0)
    assert not t.execs and not (t.root/'development/runtime/0.json').exists()


@pytest.mark.parametrize('code',[0,1,143,-15])
def test_exact_evaluator_return_sidecar_preserves_scheduler_uncertainty(t,monkeypatch,code):
    submit(t);worker_env(t,monkeypatch);t.evaluator_code=code
    assert t.r.worker(t.root,0)==code
    side=t.b.read(t.root/'development/runtime/0.evaluator_exit.json')
    assert side['returncode']==code and side['scheduler_success_claimed'] is False
    assert side['runtime_sha256']==t.b.sha(t.root/'development/runtime/0.json')
    assert side['command']==t.execs[0]


def test_process_creation_failure_keeps_runtime_claim_and_refuses_retry(t,monkeypatch):
    submit(t);worker_env(t,monkeypatch);t.evaluator_error=True
    with pytest.raises(OSError):t.r.worker(t.root,0)
    assert (t.root/'development/runtime/0.evaluator_spawn_failure.json').exists()
    with pytest.raises(ValueError,match='already claimed'):t.r.worker(t.root,0)


def test_every_tier_must_be_empty_immediately_before_first_worker_exec(t,monkeypatch):
    submit(t);worker_env(t,monkeypatch)
    output=Path(t.value['cell']['tasks'][-1]['output']);output.parent.mkdir(parents=True);output.write_text('{}')
    with pytest.raises(ValueError,match='output exists'):t.r.worker(t.root,0)
    assert not t.execs


def test_static_proof_survives_its_registered_output_growth(t):
    prepare(t);p=Path(t.value['cell']['tasks'][0]['output']);p.parent.mkdir(parents=True);p.write_text('{}')
    t.r.verify(t.root,fresh=False)
    with pytest.raises(ValueError):t.r.submit(t.root,19)
    assert not t.calls


def test_full_preparation_review_closure_is_required(t):
    _,_,provider=t.r.modules(t.provider_sha)
    extra=t.c.file('artifacts/preparation_dependency.py');v=t.b.read(provider.REVIEW)
    v['files_sha256'][str(extra)]=t.b.sha(extra);rewrite(provider.REVIEW,v);t.refresh_review()
    with pytest.raises(ValueError,match='complete preparation review'):prepare(t)
    assert not t.calls


def test_real_original_launcher_argv_and_full_four_tier_rng_admission(t,monkeypatch):
    path=ROOT/'ops/exp_scaling/launch_modebench_scale_domains.py'
    spec=importlib.util.spec_from_file_location('_actual_graph_r3_launcher',path)
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    _,_,provider=t.r.modules(t.provider_sha)
    monkeypatch.setattr(t.r,'LAUNCHER',path)
    monkeypatch.setattr(t.r,'modules',lambda digest:(t.b,launcher,provider))
    identity={'synthetic_checkpoint':'14b'};validate_calls=[];cell=t.value['cell'];source=Path(cell['source_root'])
    files={source/'protocol.json':{'models':{'14b':identity}},source/'level5/pools/graph_coloring/identity.json':{'synthetic':'four full pools'}}
    for task in cell['tasks']:files[Path(task['rows_jsonl'])]={'synthetic':'146 rows'}
    for filename,value in files.items():
        filename.parent.mkdir(parents=True,exist_ok=True);filename.write_text(json.dumps(value));t.value['files_sha256'][str(filename)]=t.b.sha(filename)
    def load_rows(task):
        tier=int(Path(task['rows_jsonl']).stem[-1]);return ([{'tier':tier,'row':i} for i in range(146)],{'row_offset':0,'row_limit':0})
    def schedule(domain,rows,seeds):
        prefix=rows[0]['tier']*10000
        return {'request_seeds':[[prefix+i*4+j for j in range(4)] for i in range(len(rows))]}
    evaluator=SimpleNamespace(INTERFACE='scratch_original_interface',model_identity=lambda path,label:identity,
        validate_task=lambda task,confirm:validate_calls.append(confirm),load_rows=load_rows,
        frozen=SimpleNamespace(POLICY='scratch-disjoint-seeds',schedule_record=schedule),sha=lambda obj:json.dumps(obj,sort_keys=True))
    monkeypatch.setattr(launcher,'evaluator',lambda:evaluator)
    monkeypatch.setattr(launcher,'checkpoint_identity',lambda path,label:{'path':path,'label':label})
    monkeypatch.setattr(launcher,'source_pins',lambda:{str(path):t.b.sha(path)})
    readiness=Path(t.value['readiness_path']);ready=t.b.read(readiness);ready['files_sha256']=t.value['files_sha256']
    rewrite(readiness,ready);t.value['readiness_sha256']=t.b.sha(readiness);t.refresh_review()
    plan=prepare(t);assert t.r.verify(t.root,fresh=True)==plan
    assert plan['rng_admission']['distinct_request_blocks']==4*146*4
    assert plan['rng_admission']['distinct_child_seeds']==4*146*32
    assert validate_calls and not any(validate_calls)
    command=plan['cells'][0]['command'];assert command==launcher.job_command(plan,plan['cells'][0])
    assert '--resume' in command and '--confirm-eval' not in command
    assert command[command.index('--tensor-parallel-size')+1]=='2'
    assert command[command.index('--max-model-len')+1]=='1024'
    assert not t.calls and not t.execs


@pytest.mark.parametrize('action',['prepare','submit'])
def test_mutations_require_actual_spin_host_but_worker_does_not(t,monkeypatch,action):
    if action=='submit':prepare(t)
    monkeypatch.setattr(t.r.socket,'gethostname',lambda:'wash.cs.princeton.edu')
    with pytest.raises(ValueError,match='truthful current spin'):
        prepare(t) if action=='prepare' else t.r.submit(t.root,19)
    assert not t.calls


@pytest.mark.parametrize('kind',['timeout','parse','returncode','names'])
def test_gpu_probe_failures_are_bounded_recorded_and_never_retried(t,monkeypatch,kind):
    submit(t);worker_env(t,monkeypatch)
    if kind=='timeout':t.probe_timeout=True
    elif kind=='parse':t.probe_parse_error=True
    elif kind=='returncode':t.probe_code=1
    else:t.probe_names=['NVIDIA A100']*2
    with pytest.raises((ValueError,subprocess.TimeoutExpired)):t.r.worker(t.root,0)
    record=t.b.read(t.root/'development/runtime/0.gpu_probe_failure.json')
    assert record['probe']['timeout_seconds']==60 and record['scheduler_success_claimed'] is False
    if kind=='timeout':assert record['probe']['stdout_bytes_base64']=='cGFydGlhbCBwcm9iZSBieXRlcw=='
    else:assert 'stdout' in record['probe'] and 'returncode' in record['probe']
    with pytest.raises(ValueError,match='probe failed'):t.r.worker(t.root,0)
    assert not t.execs and len(t.probes)==1


@pytest.mark.parametrize('field',['schema','source_sha256','provider_sha256','readiness_sha256','submission_token','review_sha256'])
def test_preparation_claim_fields_are_rebound_beyond_digest_links(t,field):
    plan=prepare(t);path=t.root/'development_preparation_claim.json';claim=t.b.read(path);claim[field]='other';rewrite(path,claim)
    plan['inputs_sha256'][str(path)]=t.b.sha(path);rewrite(t.root/'development/plan.json',plan)
    rewrite(t.root/'development/plan.sha256.json',{'sha256':t.b.sha(t.root/'development/plan.json')})
    with pytest.raises(ValueError,match='claim identity'):t.r.verify(t.root)
    assert not t.calls


def test_real_preparer_development_inputs_compose_and_remain_static(t,prep_fixture,monkeypatch):
    x=prep_fixture;x.execute();before=list(x.calls)
    monkeypatch.setattr(t.r,'STATE',x.state);monkeypatch.setattr(t.r,'REVISION_ROOT',x.a.REVISION_ROOT)
    monkeypatch.setattr(t.r,'modules',lambda digest:(t.b,None,x.a))
    monkeypatch.setattr(t.b,'frozen_view',lambda path:{'synthetic':'fixture source view'})
    result=t.r.inputs(x.state,x.a.sha(x.a.SOURCE))[-1]
    assert result['cell']==x.a.read(x.state/'development_cell.json')
    assert result['models']=={'14b':'literal checkpoint'} and len(result['cell']['tasks'])==4
    output=Path(result['cell']['tasks'][-1]['output']);output.parent.mkdir(parents=True,exist_ok=True);output.write_text('{}')
    assert t.r.inputs(x.state,x.a.sha(x.a.SOURCE))[-1]==result and x.calls==before


def test_fence_loss_after_sbatch_preserves_actual_result_and_blocks_retry(t):
    prepare(t);t.lose_fence_after_submit=True
    with pytest.raises(ValueError,match='fence lost'):t.r.submit(t.root,19)
    result=t.b.read(t.root/'development/submission_result.json')
    assert result['array_job_id']==910 and result['status']=='submitted' and len(t.calls)==1
    t.fence_lost=False
    with pytest.raises(ValueError,match='already attempted'):t.r.submit(t.root,19)
    assert len(t.calls)==1
