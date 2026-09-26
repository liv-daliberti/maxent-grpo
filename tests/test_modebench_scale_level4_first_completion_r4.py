"""Synthetic completion proofs and once-only native orchestration; no real science."""
from copy import deepcopy
from datetime import datetime, timedelta
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_text(json.dumps(value,sort_keys=True)+'\n');return path


@pytest.fixture
def x(tmp_path,monkeypatch):
    a=load(ROOT/'artifacts/complete_modebench_scale_level4_first_r4_20260913.py','_scratch_l4_completion')
    state=tmp_path/'release_action';execution=state/'confirmation';root=state/'completion';release=tmp_path/'release';parent=tmp_path/'parent'
    view=put(tmp_path/'view.json',{'scratch':True});lock=put(tmp_path/'controller.lock',{})
    for key,value in {'STATE':root,'TRANSPORT_STATE':state,'EXECUTION':execution,'RELEASE':release,'PARENT':parent,
        'VIEW':view,'REVIEW':tmp_path/'final_review.json'}.items():monkeypatch.setattr(a,key,value)
    core=load(a.CORE,'_scratch_l4_completion_original_core')
    protocols={};cells=[];summaries={};rows_by_domain={};outputs={};sources={};pins={};runtime_schema='synthetic_reviewed_transport'
    target={'pass@1':.25,'pass@8':.5};tolerances={'pass@1':.1,'pass@8':.1}
    for index,domain in enumerate(a.DOMAINS):
        source_root=tmp_path/'sources'/domain;kind='campaign_v1' if domain in ('countdown','mathir') else 'domain_revision_v1'
        protocol={'files_sha256':{},'targets':{domain:{'metrics':target}},'tolerances':tolerances,'revision':3}
        protocols[domain]=protocol;pp=put(source_root/'protocol.json',protocol)
        recipe=put(source_root/'level4/recipes'/f'{domain}.json',{'development_fit_pass':True})
        dataset=source_root/'level4/dataset'/domain
        identity=put(dataset/'identity.json',{'splits':{s:{'rows':n,'rows_sha256':s} for s,n in [('train',384),('dev',128),('eval',128)]}})
        for split in ('train','dev','eval'):(dataset/split).mkdir()
        source={'source_root':str(source_root),'source_kind':kind};sources[domain]=source
        receipt=source_root/'level4/results/confirmation'/f'{domain}.json'
        task={'level':'level4','domain':domain,'split':'eval','seeds':[11,12,13,14],'batch_size':8,'row_offset':0,'row_limit':0,'output':str(receipt)}
        tasks=put(execution/'tasks'/f'{domain}.json',[task])
        cell={'id':f'level4_{domain}_eval','domain':domain,'level':'level4','phase':'eval','model_label':'7b',**source,
            'tasks':str(tasks),'command':['/scratch/python','/scratch/evaluate','--resume','--confirm-eval']}
        cells.append(cell);put(receipt,{'generated_at':'2026-09-12T19:54:00+00:00','metrics':target})
        out=[receipt,put(str(receipt)+'.batches/run.json',{})]
        for seed in task['seeds']:
            for start in range(0,128,8):out.append(put(str(receipt)+f'.batches/seed-{seed}__rows-{start:06d}-{start+8:06d}.json',{}))
        ns=int(datetime.fromisoformat('2026-09-12T19:54:00+00:00').timestamp()*1e9)
        for path in out:os.utime(path,ns=(ns,ns))
        summaries[domain]={'tier':0,'rows':128,'batches':64,'attempts':4096,'receipt':str(receipt),'receipt_sha256':a.file_sha(receipt),
            'source_certificate':str(identity),'metrics':dict(target),'rendered_prompts_sha256':'prompt-'+domain}
        rows_by_domain[domain]=[{'problem':f'SCRATCH_{domain}_{i}'} for i in range(128)];outputs[domain]=out
        pins.update({str(path):a.file_sha(path) for path in [pp,recipe,identity,tasks]})
    manifest={'files_sha256':dict(pins),'sources':sources,'targets':{d:{'metrics':target} for d in a.DOMAINS}}
    manifestpath=put(release/'level4/source_manifest.json',manifest);put(release/'level4/source_manifest.sha256.json',{'sha256':a.file_sha(manifestpath)})
    shared=put(tmp_path/'shared.json',{'exact':'diagnostic'})
    contract={'schema':'modebench_scale_python_harder_portable_evidence_v1','shared_source':str(shared),
        'historical_destination':'/tmp/synthetic_completion/analysis.json','sha256':a.file_sha(shared),
        'overwrite_permitted':False,'science_or_live_fence_changed':False}
    plan={'concurrency':3,'operational_amendment':{'synthetic':'explicit3cells'},'dependency_ids':[901],
        'created_at':'2026-09-12T19:50:00+00:00','cells':cells,'provider_sha256':'provider','readiness_sha256':'ready',
        'view_manifest':str(view),'models':{'7b':{'path':'/scratch/model'}},'python':'/scratch/python','inputs_sha256':pins,'portable_evidence':contract}
    put(execution/'plan.json',plan);put(execution/'plan.sha256.json',{'sha256':a.file_sha(execution/'plan.json')})
    put(execution/'submission_intent.json',{'at_utc':'2026-09-12T19:51:15.100000+00:00'})
    submitted={'array_job_id':910,'at_utc':'2026-09-12T19:51:15.600000+00:00'};put(execution/'submission_result.json',submitted)
    environment={'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','VLLM_USE_V1':'0','VLLM_ATTENTION_BACKEND':'XFORMERS'}
    terminal_rows=[];logs={};runtimes=[];outcomes=[]
    for index,cell in enumerate(cells):
        raw=str(920+index);human=f'910_{index}';node=['node202','node203','node204'][index%3];host=node+'.synthetic'
        env={**environment,'SLURMD_NODENAME':node,'SLURM_JOB_ACCOUNT':'allcs','SLURM_JOB_PARTITION':'cs','SLURM_ARRAY_JOB_ID':'910',
            'SLURM_ARRAY_TASK_ID':str(index),'SLURM_JOB_ID':raw,'SLURM_CPUS_PER_TASK':'6','SLURM_MEM_PER_NODE':'61440',
            'PYTHONPYCACHEPREFIX':'/tmp/NONPRODUCTION-scale-frozen-pycache-scratch'}
        staging={**contract,'observer_host':host,'observer_pid':12345,'observer_uid':1000,'created_destination':True,
            'status':'created_exact_diagnostic_copy','at_utc':'2026-09-12T19:51:18+00:00',
            'destination_stat':{'uid':1000,'device':50,'inode':123,'size_bytes':shared.stat().st_size,'mtime_ns':123,'mode':'0o444'}}
        runtime={'schema':runtime_schema,'status':'validated_before_unchanged_evaluator_subprocess','array_job_id':910,'array_index':index,
            'cell':cell['id'],'level':'level4','plan_sha256':a.file_sha(execution/'plan.json'),'submission_result_sha256':a.file_sha(execution/'submission_result.json'),
            'readiness_sha256':'ready','provider_sha256':'provider','view_manifest':str(view),'view_manifest_sha256':a.file_sha(view),
            'evaluator_command':cell['command'],'environment':env,'hostname':host,'vllm_version':'0.8.4','visible_gpu_names':['NVIDIA RTX A5000']*2,
            'at_utc':'2026-09-12T19:51:40+00:00','portable_evidence_staging':staging,'current_disjointness':{d:{} for d in a.DOMAINS},
            'outputs_before_exec':{t['output']:{'receipt_exists':False,'batch_directory_exists':False} for t in a.read(cell['tasks'])},
            'gpu_metadata_probe':{'command':[plan['python'],'-B','-c','import json,torch;print(json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))'],
                'timeout_seconds':60,'returncode':0,'stdout':json.dumps(['NVIDIA RTX A5000']*2),'stderr':'',
                'started_at_utc':'2026-09-12T19:51:35+00:00','finished_at_utc':'2026-09-12T19:51:39+00:00'}}
        rp=put(execution/'runtime'/f'{index}.json',runtime);runtimes.append(runtime)
        outcome={'schema':runtime_schema,'status':'actual_evaluator_process_return_observed','array_job_id':910,'array_index':index,
            'cell':cell['id'],'command':cell['command'],'runtime_sha256':a.file_sha(rp),'observer_host':host,'observer_pid':12345,
            'scheduler_success_claimed':False,'returncode':0,'started_at_utc':'2026-09-12T19:51:40.100000+00:00','finished_at_utc':'2026-09-12T19:57:19.650000+00:00'}
        put(execution/'runtime'/f'{index}.evaluator_exit.json',outcome);outcomes.append(outcome)
        for suffix in ('','.batch','.extern'):
            terminal_rows.append([raw+suffix,human+suffix,'COMPLETED','0:0','2026-09-12T19:51:17','2026-09-12T19:57:19',node,'6','60G' if not suffix else '',
                'cpu=6,gres/gpu:a5000=2,gres/gpu=2,mem=60G,node=1','2026-09-12T19:51:15' if not suffix else '2026-09-12T19:51:17','allcs','cs' if not suffix else ''])
        out=execution/'logs'/f'{human}.out';out.parent.mkdir(exist_ok=True)
        summary=summaries[cell['domain']];out.write_text(json.dumps({'event':'task_complete','output':summary['receipt'],'metrics':summary['metrics']})+'\n')
        err=execution/'logs'/f'{human}.err';err.write_text('synthetic\n');logs.update({str(p):a.file_sha(p) for p in (out,err)})
    # Three allocations first, then two at the same recorded end/start second.
    # These are synthetic timestamps, not asserted production execution data.
    delta=timedelta(minutes=6,seconds=2)
    def shift(value):return (datetime.fromisoformat(value)+delta).isoformat()
    for index in (3,4):
        runtime=runtimes[index];outcome=outcomes[index]
        runtime['at_utc']=shift(runtime['at_utc'])
        runtime['portable_evidence_staging']['at_utc']=shift(runtime['portable_evidence_staging']['at_utc'])
        for key in ('started_at_utc','finished_at_utc'):runtime['gpu_metadata_probe'][key]=shift(runtime['gpu_metadata_probe'][key])
        put(execution/'runtime'/f'{index}.json',runtime)
        outcome['runtime_sha256']=a.file_sha(execution/'runtime'/f'{index}.json')
        for key in ('started_at_utc','finished_at_utc'):outcome[key]=shift(outcome[key])
        put(execution/'runtime'/f'{index}.evaluator_exit.json',outcome)
        for row in terminal_rows[index*3:index*3+3]:
            row[4]=shift(row[4]);row[5]=shift(row[5])
            if row[1].endswith(('.batch','.extern')):row[10]=shift(row[10])
        domain=a.DOMAINS[index]
        receipt=outputs[domain][0];v=a.read(receipt);v['generated_at']=shift(v['generated_at']);put(receipt,v)
        summaries[domain]['receipt_sha256']=a.file_sha(receipt)
        ns=int(datetime.fromisoformat(v['generated_at']).timestamp()*1e9)
        for path in outputs[domain]:os.utime(path,ns=(ns,ns))
    terminal={'schema':a.TERMINAL_SCHEMA,'array_job_id':910,'command':a.terminal_command(910),'environment':{'TZ':'UTC'},'returncode':0,'stderr':'',
        'stdout':'\n'.join('|'.join(row) for row in terminal_rows)+'\n','observer_host':'soak.cs.princeton.edu','observer_uid':363432,'observer_pid':12345,
        'captured_at_utc':'2026-09-12T20:05:00+00:00','logs_sha256':logs}
    terminalpath=put(execution/'terminal_accounting.json',terminal)
    tx=SimpleNamespace(SCHEMA=runtime_schema,ENVIRONMENT=environment,NODES=['node202','node203','node204'],OPERATIONAL_AMENDMENT=plan['operational_amendment'],
        verified=lambda state:(plan,None,None,None,{}),submission_identity=lambda state:submitted)
    modules=SimpleNamespace(revision=SimpleNamespace(authenticate=lambda path:protocols[Path(path).name]))
    monkeypatch.setattr(a,'load',lambda *args:tx)
    monkeypatch.setattr(a.first,'load_module',lambda path,name:core if path==a.CORE else SimpleNamespace(authenticate=lambda path:None))
    monkeypatch.setattr(a.cell_helper,'scientific_modules',lambda:modules)
    monkeypatch.setattr(core,'source_manifest',lambda *args:manifest)
    monkeypatch.setattr(core.original,'authenticate',lambda path:protocols[Path(path).parent.name])
    def completed(mod,plan,cell,task,tier,protocol,end):
        domain=cell['domain'];assert tier==0 and len(a.read(cell['tasks']))==1
        assert all(path.exists() for path in outputs[domain])
        return summaries[domain],rows_by_domain[domain],outputs[domain]
    monkeypatch.setattr(a.cell_helper,'completed_task',completed)
    prompt_calls=[]
    monkeypatch.setattr(a.cell_helper,'tokenizer_for',lambda context:object())
    def prompts(context,tier,tokenizer):
        prompt_calls.append(context.cell['domain']);return {'rendered_prompts_sha256':context.summaries[0]['rendered_prompts_sha256']}
    monkeypatch.setattr(a.cell_helper,'native_prompts',prompts)
    monkeypatch.setattr(a,'host_guard',lambda **kwargs:None)
    monkeypatch.setattr(a,'check_pins',lambda values,**kwargs:a.cell_helper.pins_check(values))
    actual_context=a.inspect(root,'a'*64,a.file_sha(terminalpath))
    review={'array_job_id':910,'transport_sha256':'a'*64,'terminal_sha256':a.file_sha(terminalpath)}
    put(a.REVIEW,{'synthetic':'final immutable complete barrier'})
    final_pins={**actual_context.pins,str(a.REVIEW):a.file_sha(a.REVIEW)}
    actual_reviewed=a.reviewed
    monkeypatch.setattr(a,'reviewed',lambda digest,**kwargs:(review,final_pins) if digest==a.file_sha(a.REVIEW) else pytest.fail('wrong review'))
    stateful=SimpleNamespace(failed=set(),exception=None,native_calls=[],publish_calls=[],wrapper_code=0,fence_lost=False,held=[],prompt_calls=prompt_calls)
    def native_audit(parent,level,domain,source):
        assert level=='level4' and stateful.held==[19]
        claim=root/'domains'/domain/'claim.json';assert claim.exists() and a.read(claim)['new_original_grader_calls']==4096
        assert all(not (root/'domains'/d/'result.json').exists() for d in a.DOMAINS if d not in stateful.native_calls)
        stateful.native_calls.append(domain)
        if stateful.exception==domain:raise RuntimeError('synthetic native audit exception')
        summary=summaries[domain];protocol=protocols[domain]
        metric=dict(summary['metrics'])
        if domain in stateful.failed:
            # Negative decision is predeclared in the saved input fixture below.
            assert metric['pass@1']==.5
        diff={k:metric[k]-target[k] for k in tolerances};gates={k:abs(diff[k])<=tolerances[k] for k in tolerances}
        result={'schema':'modebench_scale_confirmation_v1' if source['source_kind']=='campaign_v1' else 'modebench_scale_domain_revision_confirmation_v1',
            'level':'level4','domain':domain,'metrics':metric,'target':target,'differences':diff,'gates':gates,'difficulty_matched':all(gates.values()),
            'original_grader_replayed_attempts':4096,'receipt_sha256':summary['receipt_sha256'],
            'recipe_sha256':a.file_sha(Path(source['source_root'])/'level4/recipes'/f'{domain}.json'),
            'dataset_identity_sha256':a.file_sha(Path(source['source_root'])/'level4/dataset'/domain/'identity.json'),
            'candidate_prompt_bootstrap_delta_95':{k:[0,0] for k in tolerances}}
        if source['source_kind']=='domain_revision_v1':result.update(revision=3,cross_source_disjointness={'files_sha256':pins})
        put(Path(source['source_root'])/'level4/confirmation'/f'{domain}.json',result);return result
    monkeypatch.setattr(core,'audit_source',native_audit)
    original_publish=core.publish_level
    def publish(*args,**kwargs):
        stateful.publish_calls.append(kwargs['publish'])
        return original_publish(*args,**kwargs)
    monkeypatch.setattr(core,'publish_level',publish)
    monkeypatch.setattr(core,'frozen_identity',lambda root,level,domain,kind:a.read(Path(root)/level/'dataset'/domain/'identity.json'))
    monkeypatch.setattr(core,'verify_dataset',lambda *args:{'files_sha256':pins})
    monkeypatch.setattr(a.first,'LOCK',lock);monkeypatch.setattr(a.first,'LOCK_INODE',lock.stat().st_ino)
    def fence(fd):
        assert stateful.held==[fd]
        if stateful.fence_lost:raise ValueError('synthetic fence lost')
    monkeypatch.setattr(a.first,'assert_fence',fence)
    @contextmanager
    def lifetime():
        assert not stateful.held;stateful.held.append(19)
        try:yield 19
        finally:stateful.held.clear()
    monkeypatch.setattr(a.first,'lifetime_fence',lifetime)
    active=[];operation=['audit']
    def identity(pid):
        command=[str(a.PYTHON),'-B',str(a.SOURCE),operation[0]]
        if active and pid==os.getpid():
            command=active[0][active[0].index('--')+1:];command.insert(1,'-B')
        return {'pid':pid,'uid':os.getuid(),'start_ticks':'123','state':'R','command':command}
    monkeypatch.setattr(a.first,'process_identity',identity)
    def run(command,**kwargs):
        assert kwargs['pass_fds']==(19,)
        active.append(command)
        try:a.guest(root,operation[0],19,a.file_sha(a.REVIEW))
        finally:active.clear()
        return SimpleNamespace(returncode=stateful.wrapper_code)
    # Make outer and guest identities distinct without any real subprocess.
    real_identity=identity
    def identified(pid):
        if active and pid==os.getpid():return {**real_identity(pid),'pid':123456}
        return real_identity(pid)
    monkeypatch.setattr(a.first,'process_identity',identified)
    monkeypatch.setattr(a,'subprocess',SimpleNamespace(run=run))
    def execute(op):
        operation[0]=op
        # guard asks for the recorded outer pid while guest asks os.getpid.
        def proc(pid):
            if active and pid==123456:return {**real_identity(pid),'pid':123456,'command':[str(a.PYTHON),'-B',str(a.SOURCE),op]}
            if active:return {**real_identity(pid),'pid':222222}
            return {**real_identity(pid),'pid':123456}
        monkeypatch.setattr(a.first,'process_identity',proc)
        return a.run(op,root,a.file_sha(a.REVIEW))
    return SimpleNamespace(a=a,root=root,execution=execution,release=release,plan=plan,terminal=terminal,terminal_rows=terminal_rows,
        runtimes=runtimes,outcomes=outcomes,contexts=actual_context,summary=summaries,outputs=outputs,review=review,pins=final_pins,
        state=stateful,core=core,run=execute,actual_reviewed=actual_reviewed)


def test_structural_barrier_preserves_five_raw_jobs_without_grading(x):
    context=x.a.inspect(x.root,'a'*64,x.review['terminal_sha256'])
    assert [e['domain'] for e in context.entries]==list(x.a.DOMAINS)
    assert [e['execution']['job_id_raw'] for e in context.entries]==[str(i) for i in range(920,925)]
    assert sum(e['summary']['batches'] for e in context.entries)==320
    assert sum(e['summary']['attempts'] for e in context.entries)==20480
    assert context.actual_concurrency['maximum_observed_allocations']==3
    assert context.actual_concurrency['maximum_observed_gpus']==6
    assert context.actual_concurrency['subsecond_overlap_at_shared_endpoints']=='not inferred'
    assert x.state.native_calls==[] and not x.root.exists()


def test_once_only_native_audits_then_native_level4_admission_and_readonly_verify(x):
    result=x.run('audit')
    assert x.state.native_calls==list(x.a.DOMAINS) and result['new_original_grader_calls']==20480
    assert result['native_fixed_recipe_reproduction_performed'] and result['new_fit_publications']==0
    assert x.a.verify_existing(x.root)==result and x.state.native_calls==list(x.a.DOMAINS)
    with pytest.raises(ValueError,match='already attempted'):x.run('audit')
    admitted=x.run('admit')
    assert admitted['status']=='level4_admitted' and x.state.publish_calls==[True]
    assert x.a.verify_existing(x.root)==admitted and x.state.publish_calls==[True,False]
    assert x.state.native_calls==list(x.a.DOMAINS)
    assert not (x.release/'level5').exists() and not (x.release/'admission.json').exists()


def test_native_exception_preserves_partial_claims_and_never_regrades(x):
    x.state.exception='python_factors'
    with pytest.raises(RuntimeError,match='native audit'):x.run('audit')
    assert x.state.native_calls==list(x.a.DOMAINS[:3])
    failure=x.a.read(x.root/'audit_action/failure.json')
    path=Path(x.contexts.entries[0]['native_audit_path'])
    assert failure['files_sha256'][str(path)]==x.a.file_sha(path)
    assert (x.root/'domains/python_factors/claim.json').exists()
    with pytest.raises(ValueError,match='already attempted'):x.run('audit')
    with pytest.raises(ValueError):x.a.verify_existing(x.root)
    assert x.state.native_calls==list(x.a.DOMAINS[:3])


@pytest.mark.parametrize('code',[1,143,-15])
def test_complete_native_outputs_do_not_waive_nonzero_outer_execution(x,code):
    x.state.wrapper_code=code
    with pytest.raises(ValueError,match='reconciliation'):x.run('audit')
    assert x.state.native_calls==list(x.a.DOMAINS)
    assert (x.root/'audit_action/result.json').exists()
    assert x.a.read(x.root/'audit_action/exit.json')['returncode']==code
    with pytest.raises(ValueError,match='reconciliation'):x.a.verify_existing(x.root)
    with pytest.raises(ValueError,match='already attempted'):x.run('audit')
    assert x.state.native_calls==list(x.a.DOMAINS)


def negative_input(x,domain):
    summary=x.summary[domain];summary['metrics']['pass@1']=.5
    receipt=Path(summary['receipt']);value=x.a.read(receipt);value['metrics']=summary['metrics'];put(receipt,value)
    stamp=int(datetime.fromisoformat('2026-09-12T19:54:00+00:00').timestamp()*1e9);os.utime(receipt,ns=(stamp,stamp))
    summary['receipt_sha256']=x.a.file_sha(receipt)
    index=list(x.a.DOMAINS).index(domain);path=x.execution/'logs'/f'910_{index}.out'
    path.write_text(json.dumps({'event':'task_complete','output':str(receipt),'metrics':summary['metrics']})+'\n')
    x.terminal['logs_sha256'][str(path)]=x.a.file_sha(path);put(x.execution/'terminal_accounting.json',x.terminal)
    x.review['terminal_sha256']=x.a.file_sha(x.execution/'terminal_accounting.json')
    context=x.a.inspect(x.root,'a'*64,x.review['terminal_sha256']);x.pins.update(context.pins)
    x.state.failed.add(domain)


def test_negative_native_decision_is_retained_all_five_are_audited_and_admission_is_blocked(x):
    negative_input(x,'python_factors')
    result=x.run('audit')
    assert result['status']=='heldout_confirmation_failed' and result['failed_domains']==['python_factors']
    assert x.state.native_calls==list(x.a.DOMAINS) and result['new_original_grader_calls']==20480
    assert x.a.verify_existing(x.root)==result
    with pytest.raises(ValueError,match='all five native'):x.run('admit')
    assert not x.state.publish_calls and not (x.release/'level4/admission.json').exists()
    assert x.state.native_calls==list(x.a.DOMAINS)


@pytest.mark.parametrize('kind',['missing_receipt','missing_batch','extra_runtime','missing_step','failed_step','bad_sidecar','wrong_counts','source_drift'])
def test_incomplete_or_unexpected_execution_never_claims_any_native_grader(x,kind):
    if kind=='missing_receipt':x.outputs['pantry'][0].unlink()
    elif kind=='missing_batch':x.outputs['pantry'][-1].unlink()
    elif kind=='extra_runtime':put(x.execution/'runtime/5.json',{})
    elif kind in ('missing_step','failed_step'):
        rows=deepcopy(x.terminal_rows)
        if kind=='missing_step':rows.pop()
        else:rows[0][2:4]=['FAILED','1:0']
        x.terminal['stdout']='\n'.join('|'.join(row) for row in rows)+'\n'
        put(x.execution/'terminal_accounting.json',x.terminal);x.review['terminal_sha256']=x.a.file_sha(x.execution/'terminal_accounting.json')
    elif kind=='bad_sidecar':put(x.execution/'runtime/0.evaluator_exit.json',{**x.outcomes[0],'returncode':1})
    elif kind=='wrong_counts':x.summary['pantry']['batches']=63
    else:x.plan['cells'][-1]['source_root']='/other'
    with pytest.raises((ValueError,AssertionError,FileNotFoundError)):x.run('audit')
    assert not x.state.native_calls and not (x.root/'domains').exists()
    assert (x.root/'audit_action/failure.json').exists()


def test_existing_native_audit_is_never_regraded_or_adopted_without_its_claim(x):
    path=Path(x.contexts.entries[2]['native_audit_path']);put(path,{'prior':'unknown attempt'})
    with pytest.raises(ValueError,match='already exists'):x.run('audit')
    assert not x.state.native_calls and not (x.root/'registration.json').exists()
    assert x.a.read(path)=={'prior':'unknown attempt'}


def test_readonly_paths_never_call_native_confirmation_or_grader(x,monkeypatch):
    x.run('audit');x.run('admit');before=list(x.state.native_calls)
    def forbidden(*args,**kwargs):raise AssertionError('readonly must not regrade')
    monkeypatch.setattr(x.core,'audit_source',forbidden)
    monkeypatch.setattr(x.core.revision,'confirm_domain',forbidden)
    monkeypatch.setattr(x.core.previous,'confirmation',forbidden)
    assert x.a.verify_existing(x.root)['status']=='level4_admitted'
    assert x.state.native_calls==before


@pytest.mark.parametrize('kind',['claim','report','audit_result','native_audit','admission','runtime_source','runtime_command','runtime_view'])
def test_readonly_rejects_changed_saved_proof_without_any_grading(x,kind):
    x.run('audit');x.run('admit');before=list(x.state.native_calls)
    if kind=='claim':path=x.root/'domains/countdown/claim.json';value=x.a.read(path);value['new_original_grader_calls']=0
    elif kind=='report':path=x.root/'domains/countdown/result.json';value=x.a.read(path);value['claim_sha256']='wrong'
    elif kind=='audit_result':path=x.root/'audit_action/result.json';value=x.a.read(path);value['new_original_grader_calls']=1
    elif kind=='native_audit':path=Path(x.contexts.entries[0]['native_audit_path']);value=x.a.read(path);value['candidate_prompt_bootstrap_delta_95']={}
    elif kind=='admission':path=x.release/'level4/admission.json';value=x.a.read(path);value['difficulty_matched']=False
    else:
        path=x.root/'audit_action/runtime.json';value=x.a.read(path)
        if kind=='runtime_source':value['source_sha256']='wrong'
        elif kind=='runtime_command':value['command']=['wrong']
        else:value['view_manifest_sha256']='wrong'
    put(path,value)
    with pytest.raises(ValueError):x.a.verify_existing(x.root)
    assert x.state.native_calls==before


def test_static_proof_does_not_compare_historical_device_to_current_host(x,monkeypatch):
    x.run('audit')
    class ForeignLock:
        def stat(self):raise AssertionError('historical NFS device cannot be replayed')
    monkeypatch.setattr(x.a.first,'LOCK',ForeignLock())
    assert x.a.verify_existing(x.root)['status']=='all_five_native_heldout_audits_passed'


def test_live_fence_loss_prevents_native_grading_and_is_durable(x):
    x.state.fence_lost=True
    with pytest.raises(ValueError,match='fence lost'):x.run('audit')
    assert not x.state.native_calls and (x.root/'audit_action/failure.json').exists()
    with pytest.raises(ValueError,match='already attempted'):x.run('audit')


@pytest.fixture
def final_review_contract(x,monkeypatch):
    a=x.a
    prior=put(x.root.parent/'original_review.json',{'files_sha256':{str(a.CELL):a.file_sha(a.CELL)}})
    monkeypatch.setattr(a.first,'REVIEW',prior)
    required=(a.SOURCE,a.TESTS,a.PREDECESSOR_COPY,a.PREDECESSOR_TESTS_COPY,a.PREDECESSOR_MANIFEST,a.FIRST,a.first.TESTS,prior,a.CELL,a.CORE,a.TRANSPORT,x.execution/'terminal_accounting.json',*a.PRESERVED_PINS)
    pins={**x.contexts.pins,**{str(path):a.file_sha(path) for path in required}}
    value={'schema':'modebench_scale_level4_first_completion_independent_review_v1','status':'reviewed','terminal_policy':a.POLICY,
        'host_operational_amendment':a.HOST_AMENDMENT,'cpu_host':a.HOST,'cpu_uid':a.CPU_UID,'terminal_observer_host':a.HOST,
        'array_job_id':910,'transport_sha256':a.file_sha(a.TRANSPORT),'terminal_sha256':a.file_sha(x.execution/'terminal_accounting.json'),
        'files_sha256':pins}
    put(a.REVIEW,value)
    return value


def test_final_review_requires_actual_terminal_and_all_unchanged_source_pins(x,final_review_contract):
    value,pins=x.actual_reviewed(x.a.file_sha(x.a.REVIEW),guest=True)
    assert value==final_review_contract and pins[str(x.a.REVIEW)]==x.a.file_sha(x.a.REVIEW)


@pytest.mark.parametrize('field',['schema','status','policy','terminal','transport','job','missing_source','future_output','prior_closure'])
def test_prospective_or_incomplete_final_review_cannot_activate(x,final_review_contract,field):
    value=final_review_contract
    if field=='schema':value['schema']='wrong'
    elif field=='status':value['status']='pending'
    elif field=='policy':value['terminal_policy']='future_failed_maybe_ok'
    elif field=='terminal':value['terminal_sha256']='0'*64
    elif field=='transport':value['transport_sha256']='0'*64
    elif field=='job':value['array_job_id']=None
    elif field=='missing_source':value['files_sha256'].pop(str(x.a.CELL))
    elif field=='future_output':value['files_sha256'][str(x.root/'audit_action/result.json')]='0'*64
    else:put(x.a.first.REVIEW,{'files_sha256':{str(x.a.CELL):'0'*64}});value['files_sha256'][str(x.a.first.REVIEW)]=x.a.file_sha(x.a.first.REVIEW)
    put(x.a.REVIEW,value)
    with pytest.raises(ValueError):x.actual_reviewed(x.a.file_sha(x.a.REVIEW),guest=True)
    assert not x.state.native_calls


def test_real_original_audit_source_dispatches_one_native_confirmation_per_domain(tmp_path,monkeypatch):
    core=load(ROOT/'ops/exp_scaling/continue_modebench_scale_composite.py','_real_l4_native_dispatch')
    calls=[];parent=tmp_path/'parent'
    marker=put(tmp_path/'marker.json',{})
    def revision(root,*,publish):
        assert publish is True
        domain=Path(root).name;calls.append(('revision',domain))
        report={'cross_source_disjointness':{'files_sha256':{str(marker):core.file_sha(marker)}},'difficulty_matched':False}
        put(Path(root)/'level4/confirmation'/f'{domain}.json',report)
        return report
    def campaign(root,level,domain,protocol):calls.append(('campaign',domain));return {'difficulty_matched':True}
    monkeypatch.setattr(core.revision,'confirm_domain',revision)
    monkeypatch.setattr(core.previous,'confirmation',campaign)
    monkeypatch.setattr(core.original,'authenticate',lambda path:{'synthetic':'original protocol'})
    for domain in core.DOMAINS:
        kind='campaign_v1' if domain in ('countdown','mathir') else 'domain_revision_v1'
        result=core.audit_source(parent,'level4',domain,{'source_kind':kind,'source_root':str(tmp_path/domain)})
        assert result['difficulty_matched'] is (kind=='campaign_v1')
    assert len(calls)==5 and {domain for kind,domain in calls}==set(core.DOMAINS)


@pytest.mark.parametrize('count',[4,5])
def test_four_or_five_overlapping_actual_allocation_intervals_are_rejected(x,count):
    entries=deepcopy(x.contexts.entries)
    for entry in entries[:count]:
        entry['execution']['start_utc']='2026-09-12T19:51:17'
        entry['execution']['end_utc']='2026-09-12T19:57:19'
    with pytest.raises(ValueError,match='exceeds three'):x.a.allocation_concurrency(entries)


def test_equal_second_end_start_uses_explicit_half_open_allocation_convention(x):
    report=x.a.allocation_concurrency(x.contexts.entries)
    assert report['maximum_observed_allocations']==3 and report['per_cell_allocated_gpus']==2
    assert '[Start, End)' in report['interval_convention']
    first=x.contexts.entries[0]['execution'];second=x.contexts.entries[3]['execution']
    assert first['end_utc']==second['start_utc']
    assert report['intervals'][0]['end_utc']==first['end_utc']


@pytest.mark.parametrize('domain',['countdown','python_factors'])
@pytest.mark.parametrize('corrupt',[False,True])
def test_real_sealed_completed_task_campaign_and_revision_heldout_draws(x,domain,corrupt):
    sealed=load(x.a.CELL,'_real_l4_heldout_batch_validator')
    index=list(x.a.DOMAINS).index(domain);cell=x.plan['cells'][index]
    task=x.a.read(cell['tasks'])[0];source_root=Path(cell['source_root']);dataset=source_root/'level4/dataset'/domain
    rows=[{'problem':f'SCRATCH_NATIVE_{domain}_{i}','answer':'{}'} for i in range(128)]
    sourcepath=dataset/'eval.jsonl';sourcepath.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    task['rows_jsonl']=str(sourcepath)
    source={'path':str(sourcepath),'file_sha256':x.a.file_sha(sourcepath)}
    labels=task['seeds'];schedule={'scratch_rows_sha256':x.a.sha(rows),'seeds':labels}
    runtime={'max_model_len':2048,'tensor_parallel_size':2,'gpu_memory_utilization':.82,'swap_space':4.0,'enable_prefix_caching':True}
    interface={'scratch_original_interface':True};model={'path':'/scratch/model','model_label':'7b'}
    identity={'source':source,'seeds':labels,'runtime':runtime,'interface':interface,'model':{**model,'vllm_version':'0.8.4'},
        'seed_schedule_sha256':x.a.sha(schedule),'rendered_prompts_sha256':'scratch-rendered-prompts'}
    prompt_results=[{'draws':[{'attempts':[{'text':f'SCRATCH_{i}_{seed}_{draw}','canonical_key':None} for draw in range(8)]}
                              for seed in labels]} for i in range(128)]
    receipt={'level':'level4','domain':domain,'split':'eval','model_label':'7b','identity':identity,'identity_sha256':x.a.sha(identity),
        'generated_at':'2026-09-12T19:54:00+00:00','prompt_results':prompt_results,'metrics':{'pass@1':.25,'pass@8':.5}}
    put(task['output'],receipt);directory=Path(task['output']+'.batches')
    put(directory/'run.json',{'identity':identity,'identity_sha256':receipt['identity_sha256']})
    for label_index,seed in enumerate(labels):
        for start in range(0,128,8):
            draws=[row['draws'][label_index] for row in prompt_results[start:start+8]]
            put(directory/f'seed-{seed}__rows-{start:06d}-{start+8:06d}.json',
                {'seed':seed,'start':start,'end':start+8,'identity_sha256':receipt['identity_sha256'],
                 'draws':draws,'draws_sha256':x.a.sha(draws)})
    if corrupt:
        path=directory/f'seed-{labels[0]}__rows-000000-000008.json';batch=x.a.read(path)
        batch['draws'][0]['attempts'][0]['text']='CHANGED_BUT_REHASHED'
        batch['draws_sha256']=x.a.sha(batch['draws']);put(path,batch)
    put(dataset/'identity.json',{'protocol_sha256':x.a.file_sha(source_root/'protocol.json'),
        'splits':{'eval':{'rows':128,'rows_sha256':x.a.sha(rows)}}})
    validated=[]
    def validate_task(value,confirm):
        assert value is task and confirm is True;validated.append('native_eval_task')
    def validate_receipt(value,actual_rows):
        assert actual_rows is rows and len(value['prompt_results'])==128;validated.append('native_seed_receipt')
    def runtime_settings(**kwargs):
        assert kwargs==runtime;return runtime
    evaluator=SimpleNamespace(validate_task=validate_task,load_rows=lambda t:(rows,source),validate_seed_receipt=validate_receipt,
        runtime_settings=runtime_settings,frozen_interface=lambda d:interface,model_identity=lambda path,label:model,
        frozen=SimpleNamespace(schedule_record=lambda d,actual_rows,seeds:schedule))
    plan={**x.plan,'rng_admission':{'task_seed_schedule_sha256':{task['output']:x.a.sha(schedule)}}}
    protocol={'draw_labels':{'level4':{'eval':labels}} if cell['source_kind']=='campaign_v1' else {'eval':labels}}
    args=(SimpleNamespace(evaluator=evaluator),plan,cell,task,0,protocol,
        {'worker_started_at_utc':x.outcomes[index]['started_at_utc'],'end_utc':x.outcomes[index]['finished_at_utc']})
    if corrupt:
        with pytest.raises(ValueError,match='saved batch draws'):sealed.completed_task(*args)
    else:
        summary,actual_rows,paths=sealed.completed_task(*args)
        assert (summary['rows'],summary['batches'],summary['attempts'])==(128,64,4096)
        assert actual_rows is rows and len(paths)==66 and validated==['native_eval_task','native_seed_receipt']
    assert not x.state.native_calls


@pytest.mark.parametrize('guest',[False,True])
@pytest.mark.parametrize('case',['valid','wrong_host','wrong_socket','wrong_uid','wrong_euid','wrong_source'])
def test_completion_requires_actual_soak_same_user_and_original_views(monkeypatch,guest,case):
    a=load(ROOT/'artifacts/complete_modebench_scale_level4_first_r4_20260913.py','_scratch_l4_completion_host_guard')
    host='spin.cs.princeton.edu' if case=='wrong_host' else a.HOST
    reported_socket='spin.cs.princeton.edu' if case=='wrong_socket' else a.HOST
    uid=a.CPU_UID+1 if case=='wrong_uid' else a.CPU_UID
    euid=a.CPU_UID+1 if case=='wrong_euid' else a.CPU_UID
    monkeypatch.setattr(a,'os',SimpleNamespace(uname=lambda:SimpleNamespace(nodename=host),
        getuid=lambda:uid,geteuid=lambda:euid))
    monkeypatch.setattr(a,'socket',SimpleNamespace(gethostname=lambda:reported_socket))
    source=a.first.OLD_SHA if guest else a.first.NEUTRAL_SHA
    monkeypatch.setattr(a,'file_sha',lambda path:'wrong' if case=='wrong_source' else source)
    assert a.first.HOST==a.first._base.HOST=='spin.cs.princeton.edu'
    if case=='valid':a.host_guard(guest=guest)
    else:
        with pytest.raises(ValueError):a.host_guard(guest=guest)


@pytest.mark.parametrize('field,value',[
    ('observer_host','spin.cs.princeton.edu'),('observer_uid',0),('observer_uid',True),
    ('observer_pid',0),('observer_pid',True)])
def test_future_terminal_capture_requires_exact_soak_observer_before_grading(x,field,value):
    x.terminal[field]=value
    put(x.execution/'terminal_accounting.json',x.terminal)
    x.review['terminal_sha256']=x.a.file_sha(x.execution/'terminal_accounting.json')
    with pytest.raises(ValueError,match='same-user soak UTC terminal'):
        x.a.inspect(x.root,'a'*64,x.review['terminal_sha256'])
    assert not x.state.native_calls and not (x.root/'domains').exists()


@pytest.mark.parametrize('field,value',[
    ('host_operational_amendment','spin'),('cpu_host','spin.cs.princeton.edu'),
    ('terminal_observer_host','spin.cs.princeton.edu'),('cpu_uid',0),('cpu_uid',True)])
def test_final_completion_review_must_bind_new_same_user_host_amendment(x,final_review_contract,field,value):
    final_review_contract[field]=value
    put(x.a.REVIEW,final_review_contract)
    with pytest.raises(ValueError):x.actual_reviewed(x.a.file_sha(x.a.REVIEW),guest=True)
    assert not x.state.native_calls and not (x.root/'domains').exists()


def test_original_spin_draft_preservation_is_required_in_final_review(x,final_review_contract):
    preserved=x.a.PRESERVED/'complete_modebench_scale_level4_first_20260912.py'
    final_review_contract['files_sha256'].pop(str(preserved))
    put(x.a.REVIEW,final_review_contract)
    with pytest.raises(ValueError):x.actual_reviewed(x.a.file_sha(x.a.REVIEW),guest=True)
    assert x.a.file_sha(preserved)=='bd57206366605b389cecabd01cdfffb43f57a09e3ffc467d53e17cb24b43840b'
    assert x.a.first.HOST=='spin.cs.princeton.edu' and not x.state.native_calls


def test_new_cpu_action_records_actual_soak_without_relabeling_first_module(x):
    result=x.run('audit')
    assert result['new_original_grader_calls']==20480
    for name in ('runtime.json','guest_runtime.json'):
        assert x.a.read(x.root/'audit_action'/name)['host']=='soak.cs.princeton.edu'
    assert x.a.first.HOST==x.a.first._base.HOST=='spin.cs.princeton.edu'
    assert x.a.verify_existing(x.root)==result and x.state.native_calls==list(x.a.DOMAINS)


@pytest.mark.parametrize('operation',['audit','admit'])
@pytest.mark.parametrize('kind',['correct','relative_python','relative_source'])
def test_completion_absolute_prefix_rejected_before_review_and_state(x,monkeypatch,operation,kind):
    a=x.a;command=[str(a.PYTHON),'-B',str(a.SOURCE),operation]
    if kind=='relative_python':command[0]='var/seed_paper_eval/paper310/bin/python'
    elif kind=='relative_source':command[2]='artifacts/'+a.SOURCE.name
    monkeypatch.setattr(a.first,'process_identity',lambda pid:{'command':command,'pid':pid})
    if kind=='correct':assert a.outer_command_guard(operation)['command']==command
    else:
        monkeypatch.setattr(a,'host_guard',lambda **kwargs:None)
        monkeypatch.setattr(a,'reviewed',lambda *args,**kwargs:pytest.fail('must reject before input review'))
        with pytest.raises(ValueError,match='full absolute'):a.run(operation,x.root,'unused')
        assert not (x.root/(operation+'_action')).exists() and not x.state.native_calls
