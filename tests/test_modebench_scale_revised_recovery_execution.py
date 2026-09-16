"""Scratch-only completion/provenance tests; no scheduler, real model or grader."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load():
    path=ROOT/'artifacts/verify_modebench_scale_revised_recovery_execution_20260912.py'
    spec=importlib.util.spec_from_file_location('scratch_revised_completion',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): path.chmod(0o644)
    path.write_text(json.dumps(value) if isinstance(value,(dict,list)) else value)
    return path


@pytest.fixture
def t(tmp_path,monkeypatch):
    u=load(); root=tmp_path/'scratch'; root.mkdir()
    monkeypatch.setattr(u,'ROOT',root); monkeypatch.setattr(u,'RECOVERY',root)
    monkeypatch.setattr(u,'HISTORICAL_RECOVERY',root/'historical')
    monkeypatch.setattr(u,'SOURCE',write(root/'auditor.py','scratch auditor'))
    environment={'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','VLLM_USE_V1':'0','VLLM_ATTENTION_BACKEND':'XFORMERS',
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'}
    helper=SimpleNamespace(SCHEMA='scratch_operational_recovery',ALLOWED_NODES=['node105','node202','node203','node204'],
        ENVIRONMENT=environment,ORIGINAL_ARRAY=31254520,ORIGINAL_PLAN=root/'original/plan.json',
        ORIGINAL_PLAN_SHA='a'*64,FIELDS=','.join(u.FIELDS.split(',')[:11]))
    retirement=write(root/'retirement.json',{'scratch':'unstarted retirement'})
    cs=SimpleNamespace(SCHEMA='scratch_cs',OLD_JOB=31258973,RETIREMENT=retirement,
        HARDWARE={'allowed_worker_nodes':['node202','node203','node204']})
    manifest=write(root/'view.json',{'scratch':True})
    retained=write(root/'retained_original.json',{'scratch':'retained'})
    retained_pins={str(retained):u.file_sha(retained)}
    plan={'prepared_at_utc':'2026-09-12T08:00:00+00:00','view_manifest':str(manifest),
          'preserved_outputs_sha256':retained_pins,'cells':[],'user':'scratch-user','retirement_sha256':u.file_sha(retirement)}
    for i in (0,1):
        tasks=write(root/f'tasks/{i}.json',[{'output':str(root/f'outputs/{i}/{tier}.json')} for tier in range(4)])
        plan['cells'].append({'original_index':i+5,'tasks':str(tasks),'command':['literal-python','evaluator','--resume']})
    write(root/'plan.json',plan)
    submitted={'array_job_id':900,'at_utc':'2026-09-12T08:02:00+00:00'}
    write(root/'submission_identity.json',submitted); write(root/'submission_intent.json',{'at_utc':'2026-09-12T08:01:00+00:00'})
    runtimes=[]; rows=[]
    for i in (0,1):
        node='node202' if i==0 else 'node203'; start=f'2026-09-12T{9+i:02d}:00:00'; end=f'2026-09-12T{10+i:02d}:00:00'
        env={**environment,'SLURM_JOB_ID':str(901+i),'SLURM_ARRAY_JOB_ID':'900','SLURM_ARRAY_TASK_ID':str(i),
             'SLURMD_NODENAME':node,'SLURM_CPUS_PER_TASK':'6','SLURM_MEM_PER_NODE':'61440','SLURM_JOB_ACCOUNT':'allcs',
             'SLURM_JOB_PARTITION':'cs','PYTHONPYCACHEPREFIX':'/tmp/NONPRODUCTION-scale-frozen-pycache-scratch/cache'}
        runtime={'schema':cs.SCHEMA,'status':'validated_before_exact_evaluator_exec','at_utc':start.replace(':00:00',':00:01')+'+00:00',
            'array_job_id':900,'recovery_index':i,'original_index':i+5,'original_array_job_id':31254520,
            'retired_recovery_array_job_id':31258973,'retirement_path':str(retirement),'retirement_sha256':u.file_sha(retirement),
            'original_plan_sha256':helper.ORIGINAL_PLAN_SHA,'plan_sha256':u.file_sha(root/'plan.json'),
            'submission_identity_sha256':u.file_sha(root/'submission_identity.json'),'evaluator_command':plan['cells'][i]['command'],
            'view_manifest':str(manifest),'view_manifest_sha256':u.file_sha(manifest),'environment':env,
            'hostname':node+'.ionic.cs.princeton.edu','vllm_version':'0.8.4','visible_gpu_names':['NVIDIA RTX A5000']*2,
            'outputs_before_exec':{'files_sha256':retained_pins,'batches_by_original_index':{str(i+5):[116,116,116,61] if i==0 else [0]*4},
                'receipts_by_original_index':{str(i+5):[True,True,True,False] if i==0 else [False]*4}}}
        runtimes.append(runtime); write(root/f'runtime/{i}.json',runtime)
        for suffix in ('','.batch','.extern'):
            rows.append([str(901+i)+suffix,f'900_{i}'+suffix,'COMPLETED','0:0',start,end,node,'6',
                '60G' if not suffix else '', 'cpu=6,node=1,mem=60G,gres/gpu=2,gres/gpu:a5000=2',
                '2026-09-12T08:01:01' if not suffix else start,'allcs','cs' if not suffix else ''])
    logpins={}
    for i in (0,1):
        for suffix in ('.out','.err'):
            p=write(root/f'logs/900_{i}{suffix}',''); logpins[str(p)]=u.file_sha(p)
    terminal={'schema':u.TERMINAL_SCHEMA,'array_job_id':900,'command':u.terminal_command(900),'environment':{'TZ':'UTC'},
        'returncode':0,'stdout':'\n'.join('|'.join(r) for r in rows)+'\n','stderr':'','observer_host':'spin.cs.princeton.edu',
        'captured_at_utc':'2026-09-12T12:00:00+00:00','logs_sha256':logpins}
    write(root/'terminal_accounting.json',terminal)
    # Never contact Slurm through an accidentally selected inspection path.
    monkeypatch.setattr(u,'dependencies',lambda:pytest.fail('unexpected full dependency loading in scratch unit test'))
    return SimpleNamespace(u=u,root=root,helper=helper,cs=cs,plan=plan,submitted=submitted,runtimes=runtimes,rows=rows,terminal=terminal)


def terminal(t):
    return t.u.recovery_terminal(t.root,t.helper,t.plan,t.submitted,t.runtimes,t.terminal)


def change_rows(t,edit):
    rows=copy.deepcopy(t.rows); edit(rows); t.terminal['stdout']='\n'.join('|'.join(r) for r in rows)+'\n'


def test_successful_actual_two_cell_accounting_retains_raw_ids_and_nodes(t):
    result=terminal(t)
    assert [v['job_id_raw'] for v in result]==['901','902']
    assert [v['original_index'] for v in result]==[5,6]
    assert [v['node'] for v in result]==['node202','node203']
    assert all(v['scheduler_success'] is True for v in result)


@pytest.mark.parametrize('row_index',[0,1,2,3,4,5])
@pytest.mark.parametrize('state,code',[('FAILED','1:0'),('FAILED','143:0'),('NODE_FAIL','0:0'),('COMPLETED','1:0')])
def test_no_unobserved_terminal_status_is_prewaived(t,row_index,state,code):
    change_rows(t,lambda rows:rows[row_index].__setitem__(slice(2,4),[state,code]))
    with pytest.raises(ValueError,match='separately reviewed amendment'): terminal(t)


@pytest.mark.parametrize('field,value',[(0,'999'),(1,'900_6'),(6,'node200'),(7,'4'),(8,'59G'),
    (9,'cpu=6,node=1,mem=60G,gres/gpu=2,gres/gpu:a100=2'),(11,'other'),(12,'mltheory')])
def test_terminal_requires_actual_original_mapping_and_allocated_resources(t,field,value):
    change_rows(t,lambda rows:rows[0].__setitem__(field,value))
    with pytest.raises(ValueError): terminal(t)


def test_terminal_rejects_missing_step_and_duplicate_resource_keys(t):
    change_rows(t,lambda rows:rows.pop())
    with pytest.raises(ValueError,match='six actual'): terminal(t)
    change_rows(t,lambda rows:rows[0].__setitem__(9,rows[0][9]+',cpu=6'))
    with pytest.raises(ValueError,match='resource record'): terminal(t)


@pytest.mark.parametrize('which',['worker_before_start','receipt_capture_before_end','step_after_allocation','overlap','submit_before_intent'])
def test_execution_chronology_and_concurrency_are_mandatory(t,which):
    if which=='worker_before_start': t.runtimes[0]['at_utc']='2026-09-12T08:59:59+00:00'
    elif which=='receipt_capture_before_end': t.terminal['captured_at_utc']='2026-09-12T10:59:59+00:00'
    elif which=='step_after_allocation': change_rows(t,lambda rows:rows[1].__setitem__(5,'2026-09-12T10:00:01'))
    elif which=='submit_before_intent': change_rows(t,lambda rows:rows[0].__setitem__(10,'2026-09-12T08:00:59'))
    else:
        def overlap(rows):
            for row in rows[3:]: row[4]='2026-09-12T09:59:59'
        change_rows(t,overlap)
    with pytest.raises(ValueError): terminal(t)


@pytest.mark.parametrize('field,value',[('observer_host',''),('returncode',1),('stderr','query failed'),
                                       ('environment',{'TZ':'EST'}),('array_job_id',901)])
def test_terminal_observation_must_be_truthful_successful_utc(t,field,value):
    t.terminal[field]=value
    with pytest.raises(ValueError): terminal(t)


def test_terminal_requires_all_exact_log_hashes(t):
    t.terminal['logs_sha256'].pop(next(iter(t.terminal['logs_sha256'])))
    with pytest.raises(ValueError,match='four recovery log'): terminal(t)


def test_runtime_requires_actual_separate_recovery_identity(t):
    assert t.u.runtime_linkage(t.root,t.helper,t.plan,t.submitted,0,t.cs)['original_index']==5
    value=copy.deepcopy(t.runtimes[0]); value['original_index']=0; write(t.root/'runtime/0.json',value)
    with pytest.raises(ValueError,match='runtime/submission'): t.u.runtime_linkage(t.root,t.helper,t.plan,t.submitted,0,t.cs)


@pytest.mark.parametrize('key,value',[('SLURM_JOB_PARTITION','mltheory'),('SLURM_JOB_ACCOUNT','other'),
    ('SLURM_ARRAY_JOB_ID','31254520'),('SLURM_ARRAY_TASK_ID','5'),('SLURM_CPUS_PER_TASK','4'),
    ('SLURM_MEM_PER_NODE','60416'),('VLLM_USE_V1','1'),('SLURMD_NODENAME','node200')])
def test_runtime_requires_actual_same_hardware_backend_environment(t,key,value):
    t.helper.pins_check=lambda value:None
    runtime=copy.deepcopy(t.runtimes[0]); runtime['environment'][key]=value; write(t.root/'runtime/0.json',runtime)
    with pytest.raises(ValueError): t.u.runtime_linkage(t.root,t.helper,t.plan,t.submitted,0,t.cs)


def test_new_file_chronology_is_durable_and_exact(t):
    p=write(t.root/'new_batch.json',{})
    at=datetime(2026,9,12,9,30,tzinfo=timezone.utc).timestamp(); os.utime(p,(at,at))
    proof=t.u.file_timestamp(p,'2026-09-12T09:00:01+00:00','2026-09-12T10:00:00')
    assert proof=={'mtime_ns':p.stat().st_mtime_ns,'size_bytes':p.stat().st_size}
    os.utime(p,(at-3600,at-3600))
    with pytest.raises(ValueError,match='predates recovery'): t.u.file_timestamp(p,'2026-09-12T09:00:01+00:00','2026-09-12T10:00:00')
    os.utime(p,(at+3600,at+3600))
    with pytest.raises(ValueError): t.u.file_timestamp(p,'2026-09-12T09:00:01+00:00','2026-09-12T10:00:00')


@pytest.fixture
def outputs(t):
    u=t.u; old_plan={'cells':[{} for _ in range(7)]}; summaries={}; delegated=[]; preserved={}
    for index,domain in enumerate(('pantry','python_factors')):
        count=116 if index==0 else 100; row_count=232 if index==0 else 193
        tasks=[]; cellsummaries=[]
        for tier in range(4):
            receipt=t.root/f'outputs/{index}/difficulty_{tier}.json'
            rows=[{'scratch_row':i} for i in range(row_count)]
            task={'output':str(receipt)}; tasks.append(task)
            write(receipt,{'scratch':True})
            paths=[receipt,write(Path(str(receipt)+'.batches')/'run.json',{})]
            for batch in range(count): paths.append(write(Path(str(receipt)+'.batches')/f'seed-1__rows-{batch:06d}.json',{}))
            summary={'tier':tier,'receipt':str(receipt),'receipt_sha256':u.file_sha(receipt),'metrics':{'pass1':.1},
                'batches':count,'attempts':row_count*32,'rendered_prompts_sha256':'scratch-prompts',
                'interface':{},'rows':row_count}
            cellsummaries.append(summary); summaries[str(receipt)]=(summary,rows,paths)
            for p in paths:
                was_old=index==0 and (tier<3 or p.name=='run.json' or (p.name.startswith('seed-') and int(p.stem.rsplit('-',1)[1])<61))
                if was_old: preserved[str(p)]=u.file_sha(p)
                when=datetime(2026,9,12,6,30,tzinfo=timezone.utc) if was_old else datetime(2026,9,12,9+index,30,tzinfo=timezone.utc)
                os.utime(p,(when.timestamp(),when.timestamp()))
        tasks_path=write(t.root/f'tasks/full-{index}.json',tasks)
        cell={'domain':domain,'id':'level5_'+domain+'_r2_dev','tasks':str(tasks_path),'source_root':str(t.root/f'source/{domain}'),
            'level':'level5','command':['literal-python','evaluator','--resume']}
        old_plan['cells'][index+5]=cell
        t.plan['cells'][index]['tasks']=str(tasks_path)
        events=[{'event':'task_complete','output':s['receipt'],'metrics':s['metrics']} for s in cellsummaries]
        write(t.root/f'logs/900_{index}.out','\n'.join(json.dumps(v) for v in (events[3:] if index==0 else events))+'\n')
        if index==0: write(t.helper.ORIGINAL_PLAN.parent/'logs/31254520_5.out','\n'.join(json.dumps(v) for v in events[:3])+'\n')
    t.plan['preserved_outputs_sha256']=preserved
    for runtime in t.runtimes: runtime['outputs_before_exec']['files_sha256']=dict(preserved)
    write(t.helper.ORIGINAL_PLAN.parent/'runtime/5.json',{'at':'2026-09-12T06:00:00+00:00'})
    def completed(modules,plan,cell,task,tier,protocol,term):
        delegated.append((cell['domain'],tier,term))
        return summaries[task['output']]
    helper=SimpleNamespace(completed_task=completed)
    modules=SimpleNamespace(revision=SimpleNamespace(authenticate=lambda path:{'scratch':True}))
    old_execution={'partial_cell':{'end_utc':'2026-09-12T07:00:00'}}
    terminals=[{'array_job_id':900,'worker_started_at_utc':runtime['at_utc'],'end_utc':f'2026-09-12T{10+i:02d}:00:00'}
               for i,runtime in enumerate(t.runtimes)]
    return SimpleNamespace(t=t,old_plan=old_plan,helper=helper,modules=modules,old_execution=old_execution,
        terminals=terminals,summaries=summaries,delegated=delegated)


def complete(o,index):
    t=o.t
    return t.u.completed_cell(t.root,t.helper,o.helper,o.modules,t.plan,o.old_plan,index,t.runtimes[index],
                             o.terminals[index],o.old_execution)


def test_only_three_pinned_receipts_use_original_epoch_all_other_outputs_are_new(outputs):
    o=outputs; _,pantry,_=complete(o,0); _,python,_=complete(o,1)
    assert len(pantry['preserved_batches'])==409 and len(pantry['new_batches'])==55
    assert len(python['new_batches'])==400 and not python['preserved_batches']
    assert len(pantry['new_output_file_times'])==56
    assert len(python['new_output_file_times'])==408
    assert [term['worker_started_at_utc'] for _,_,term in o.delegated[:4]]==['2026-09-12T06:00:00+00:00']*3+['2026-09-12T09:00:01+00:00']


def test_new_batch_cannot_be_claimed_as_present_in_preexec_inventory(outputs):
    o=outputs; t=o.t
    new=next(p for _,_,paths in o.summaries.values() for p in paths if p.name.startswith('seed-') and str(p) not in t.plan['preserved_outputs_sha256'])
    t.runtimes[0]['outputs_before_exec']['files_sha256'][str(new)]=t.u.file_sha(new)
    with pytest.raises(ValueError,match='before actual recovery'): complete(o,0)


def test_missing_new_batch_and_missing_completion_log_are_rejected(outputs):
    o=outputs; t=o.t; path=next(key for key in o.summaries if key.endswith('0/difficulty_3.json'))
    o.summaries[path][2].pop()
    with pytest.raises(ValueError,match='coverage required'): complete(o,0)
    write(t.root/'logs/900_1.out','')
    with pytest.raises(ValueError,match='completion|completed task'): complete(o,1)


def test_retained_batch_tampering_is_rejected(outputs):
    o=outputs; p=Path(next(key for key in o.t.plan['preserved_outputs_sha256'] if '/seed-' in key))
    p.write_text('changed')
    with pytest.raises(ValueError,match='partial outputs changed'): complete(o,0)


@pytest.fixture
def workflow(t,monkeypatch):
    u=t.u; source=write(t.root/'science.json',{'scratch':'fixed'})
    pins={str(source):u.file_sha(source),str(u.SOURCE):u.file_sha(u.SOURCE),str(t.root/'plan.json'):u.file_sha(t.root/'plan.json')}
    graders=[]; natives=[]; state=SimpleNamespace(fail_grade=None,fail_native=None,fail_inspect=False)
    contexts=[]; cellsummaries=[]
    for index,domain in enumerate(('pantry','python_factors')):
        tier_summaries=[]; tasks=[]; rows=[]
        for tier in range(4):
            summary={'tier':tier,'receipt':str(t.root/f'cell{index}/tier{tier}.json'),'metrics':{'pass1':.1},
                'attempts':7424 if index==0 else 6176,'rendered_prompts_sha256':f'prompts-{index}-{tier}'}
            tier_summaries.append(summary); tasks.append({'output':summary['receipt']}); rows.append([{}])
        def grade(path,rows,protocol,phase,*,regrade,index=index):
            assert regrade is True and phase=='dev' and len(natives)==8
            assert (t.root/'execution_audit/claim.json').is_file()
            tier=int(Path(path).stem[-1]); graders.append((index,tier))
            if state.fail_grade==(index,tier): raise ValueError('scratch original grader disagrees')
            return {'metrics':{'pass1':.1}},{'row':{}}
        modules=SimpleNamespace(revision=SimpleNamespace(receipt_scores=grade))
        contexts.append(SimpleNamespace(cell={'domain':domain},summaries=tier_summaries,tasks=tasks,rows=rows,protocol={},modules=modules))
        cellsummaries.append({'recovery_index':index,'summaries':tier_summaries})
    def native(context,tier,tokenizer):
        index=0 if context.cell['domain']=='pantry' else 1; natives.append((index,tier))
        if state.fail_native==(index,tier): raise ValueError('scratch native prompt mismatch')
        return {'rendered_prompts_sha256':context.summaries[tier]['rendered_prompts_sha256']}
    context=SimpleNamespace(root=t.root,submitted={'array_job_id':900},original_execution={'scheduler_success':False},original_terminal_rows_by_index={'scratch':'rows'},
        retired_recovery_execution={'array_job_id':31258973,'no_model_execution':True},
        recovery_executions=[{'state':'COMPLETED','exit_code':'0:0'}]*2,summaries=cellsummaries,pins=pins,contexts=contexts,
        cell_helper=SimpleNamespace(tokenizer_for=lambda context:'scratch tokenizer',native_prompts=native))
    def inspect(root,digest):
        assert root==t.root and digest==u.file_sha(t.root/'terminal_accounting.json')
        if state.fail_inspect: raise ValueError('scratch incomplete final receipt')
        return context
    monkeypatch.setattr(u,'inspect',inspect)
    return SimpleNamespace(t=t,context=context,state=state,graders=graders,natives=natives,
                           digest=u.file_sha(t.root/'terminal_accounting.json'))


def test_all_eight_tasks_are_validated_then_one_durable_grader_replay_and_readonly_verification(workflow):
    w=workflow; u=w.t.u; root=w.t.root
    result=u.audit(root,w.digest)
    assert w.graders==[(i,tier) for i in (0,1) for tier in range(4)]
    assert result['new_grader_invocations']==54400 and result['scheduler_success'] is False and result['recovery_scheduler_success'] is True
    assert u.verify_existing(root)==result
    assert len(w.graders)==8
    with pytest.raises(ValueError,match='already claimed'): u.audit(root,w.digest)
    assert len(w.graders)==8


def test_missing_eighth_receipt_prevents_any_claim_or_grading(workflow):
    w=workflow; w.state.fail_inspect=True
    with pytest.raises(ValueError,match='incomplete'): w.t.u.audit(w.t.root,w.digest)
    assert not w.graders and not (w.t.root/'execution_audit/claim.json').exists()


def test_native_prompt_mismatch_prevents_every_grader_call(workflow):
    w=workflow; w.state.fail_native=(1,3)
    with pytest.raises(ValueError,match='native prompt'): w.t.u.audit(w.t.root,w.digest)
    assert not w.graders and (w.t.root/'execution_audit/claim.json').exists()
    with pytest.raises(ValueError,match='already claimed'): w.t.u.audit(w.t.root,w.digest)


def test_partial_grader_failure_is_durable_and_never_replayed(workflow):
    w=workflow; w.state.fail_grade=(1,1)
    with pytest.raises(ValueError,match='grader disagrees'): w.t.u.audit(w.t.root,w.digest)
    assert w.graders==[(0,0),(0,1),(0,2),(0,3),(1,0),(1,1)]
    assert (w.t.root/'execution_audit/failure.json').exists()
    with pytest.raises(ValueError,match='already claimed'): w.t.u.audit(w.t.root,w.digest)
    assert len(w.graders)==6 and not (w.t.root/'execution_reconciliation.json').exists()


@pytest.mark.parametrize('field,value',[('new_grader_invocations',0),('original_grader_agrees',False),('original_index',0),
                                       ('validation_entrypoint','ungraded'),('tier',9)])
def test_verification_rejects_changed_grader_coverage_without_regrading(workflow,field,value):
    w=workflow; w.t.u.audit(w.t.root,w.digest)
    path=w.t.root/'execution_audit/tiers/1_3.json'; report=w.t.u.read(path); report[field]=value; write(path,report)
    with pytest.raises(ValueError,match='grader evidence'): w.t.u.verify_existing(w.t.root)
    assert len(w.graders)==8


def test_top_level_false_success_or_narrower_coverage_is_rejected(workflow):
    w=workflow; value=w.t.u.audit(w.t.root,w.digest); value['scheduler_success']=True
    write(w.t.root/'execution_reconciliation.json',value)
    with pytest.raises(ValueError,match='differs from authenticated'): w.t.u.verify_existing(w.t.root)
    assert len(w.graders)==8


@pytest.fixture
def history(t):
    u=t.u; base=t.helper; rows=[]
    for i in range(7):
        rows.append([str(800+i) if i<6 else '31254520',f'31254520_{i}',
            'NODE_FAIL' if i==5 else 'FAILED' if i<5 else 'CANCELLED by 363432',
            '143:0' if i==2 else '1:0' if i<6 else '0:0',
            '2026-09-12T06:00:00' if i<6 else 'None', '2026-09-12T07:00:00' if i<6 else '2026-09-12T07:30:00',
            'node105' if i<6 else 'None assigned','6' if i<6 else '0','60G',
            'cpu=6,mem=60G,gres/gpu=2,gres/gpu:a5000=2,node=1' if i<6 else '', '2026-09-12T05:00:00'])
    def observation(command,stdout):
        return {'command':command,'stdout':stdout,'stderr':'','returncode':0,'environment':{'TZ':'UTC'},
                'observer_host':'spin.cs.princeton.edu','at_utc':'2026-09-12T07:31:00+00:00'}
    original=copy.deepcopy(rows); original[6][2]='PENDING'; original[6][4:6]=['Unknown','Unknown']
    incident={'observations':[observation(['sacct','-j','31254520','-n','-P','--format='+base.FIELDS],
        '\n'.join('|'.join(r) for r in original)+'\n')]}
    account=observation(['sacct','-X','--array','-j','31254520','-n','-P','--format='+base.FIELDS],
        '\n'.join('|'.join(r) for r in rows)+'\n')
    queue=observation(['squeue','-u','scratch-user','-r','-h','-o','%i|%T'],'other_job|RUNNING\n')
    path=write(u.HISTORICAL_RECOVERY/'none_start_transport/plan.json',{'scratch':'registered amendment'})
    capture_path=write(t.root/'cancelled_capture.json',{})
    adapter=SimpleNamespace(CAPTURE=capture_path,CAPTURE_SHA=u.file_sha(capture_path),POLICY='exact actual None',
        capture=lambda helper:{'actual_cancelled_row':rows[6],'observer_host':'spin.cs.princeton.edu'})
    value={'status':'original_array_terminal','at_utc':'2026-09-12T07:32:00+00:00','sacct':account,'squeue':queue,
        'rows_by_original_index':{str(i):r for i,r in enumerate(rows)},'none_start_amendment':{
            'transport_plan':str(path),'transport_plan_sha256':u.file_sha(path),'captured_observation':str(capture_path),
            'captured_observation_sha256':adapter.CAPTURE_SHA,'captured_observer_host':'spin.cs.princeton.edu',
            'policy':adapter.POLICY}}
    return SimpleNamespace(t=t,rows=rows,incident=incident,value=value,adapter=adapter)


def old_history(h):
    return h.t.u.original_terminal_snapshot(h.value,h.t.helper,h.incident,h.adapter)


def test_original_failed_and_cancelled_unstarted_history_remains_truthful(history):
    value=old_history(history)
    assert value['scheduler_success'] is False
    assert value['partial_cell']['state']=='NODE_FAIL' and value['partial_cell']['preserved_batches']==409
    assert value['unstarted_cell']['start_utc']=='None' and value['unstarted_cell']['original_runtime_exists'] is False


@pytest.mark.parametrize('field,value',[(2,'COMPLETED'),(4,'Unknown'),(4,'2026-09-12T06:30:00'),
                                       (6,'node105'),(7,'6'),(9,'cpu=6')])
def test_unstarted_cancelled_cell_cannot_be_relabelled_or_assigned_execution(history,field,value):
    h=history; changed=copy.deepcopy(h.rows); changed[6][field]=value
    h.value['sacct']['stdout']='\n'.join('|'.join(r) for r in changed)+'\n'
    h.value['rows_by_original_index']={str(i):r for i,r in enumerate(changed)}
    with pytest.raises(ValueError): old_history(h)


def test_original_array_must_be_absent_from_saved_queue(history):
    history.value['squeue']['stdout']='31254520_6|RUNNING\n'
    with pytest.raises(ValueError,match='still live'): old_history(history)


def test_actual_none_amendment_cannot_be_omitted_or_replaced(history):
    history.value['none_start_amendment']['transport_plan_sha256']='0'*64
    with pytest.raises(ValueError,match='amendment'): old_history(history)


def test_prior_completed_cell_143_status_is_not_rewritten(history):
    changed=copy.deepcopy(history.rows); changed[2][3]='1:0'
    history.value['sacct']['stdout']='\n'.join('|'.join(r) for r in changed)+'\n'
    history.value['rows_by_original_index']={str(i):r for i,r in enumerate(changed)}
    with pytest.raises(ValueError,match='history changed'): old_history(history)


def test_original_partial_worker_is_authenticated_with_exact_raw_id_and_allocation_epoch(t):
    inner={'at':'2026-09-12T06:00:02+00:00'}
    outer={'at_utc':'2026-09-12T06:00:01+00:00','environment':{'SLURM_JOB_ID':'805'}}
    calls=[]
    helper=SimpleNamespace(runtime_linkage=lambda *args:(calls.append(args) or ({},inner,outer)))
    execution={'partial_cell':{'job_id_raw':'805','start_utc':'2026-09-12T06:00:00','end_utc':'2026-09-12T07:00:00'}}
    assert t.u.original_partial_runtime(t.helper,helper,{}, {},execution)==inner
    assert calls==[(5,t.helper.ORIGINAL_PLAN,{}, {})]
    inner['at']='2026-09-12T05:59:59+00:00'
    with pytest.raises(ValueError,match='outside actual'): t.u.original_partial_runtime(t.helper,helper,{}, {},execution)
    inner['at']='2026-09-12T06:00:02+00:00'; outer['environment']['SLURM_JOB_ID']='900'
    with pytest.raises(ValueError,match='outside actual'): t.u.original_partial_runtime(t.helper,helper,{}, {},execution)


def test_inspect_composes_direct_cs_runtime_retired_history_and_mixed_outputs_without_grading(outputs,history,monkeypatch):
    o=outputs; h=history; t=o.t; u=t.u; base=t.helper; cs=t.cs
    source=write(t.root/'frozen_scientific_source.py','scratch sealed dependency')
    pins={str(source):u.file_sha(source)}
    for domain in ('pantry','python_factors'):
        write(t.root/f'source/{domain}/protocol.json',{'files_sha256':pins})
        write(t.root/f'source/{domain}/pool_identity.json',{'scratch':'source certificate'})
    o.modules.revision.authenticate=lambda path:u.read(Path(path)/'protocol.json')
    for receipt,(summary,rows,paths) in o.summaries.items():
        domain='pantry' if '/outputs/0/' in receipt else 'python_factors'
        summary['source_certificate']=str(t.root/f'source/{domain}/pool_identity.json')
    t.plan['inputs_sha256']=pins
    write(t.root/'plan.json',t.plan);write(t.root/'plan.sha256.json',{'sha256':u.file_sha(t.root/'plan.json')})
    write(t.root/'preserved_inventory.json',{'files_sha256':t.plan['preserved_outputs_sha256']})
    write(t.root/'submission_result.json',{'scratch':True})
    retired={'captured_at_utc':'2026-09-12T07:40:00+00:00','retired_cells':[['scratch-cancelled-0'],['scratch-cancelled-1']],
        'absent_runtime_paths':[str(u.HISTORICAL_RECOVERY/f'runtime/{i}.json') for i in (0,1)]}
    cs.retired_rows=lambda record,state:json.loads(record['stdout'])
    cs.queue_gate=lambda record,user,pending:u.require(record['stdout']=='','scratch retired job still live')
    cs.verified=lambda root,require_initial:(t.plan,SimpleNamespace(),retired)
    cs.submission_identity=lambda root:t.submitted
    base.original=lambda:(o.old_plan,{'inputs_sha256':pins})
    base.incident_capture=lambda:h.incident
    o.helper.runtime_linkage=lambda *args:({}, {'at':'2026-09-12T06:00:00+00:00'},
        {'at_utc':'2026-09-12T06:00:00+00:00','environment':{'SLURM_JOB_ID':'805'}})
    def parent_snapshot(stamp,host):
        original=copy.deepcopy(h.value)
        original['observer_host']=host;original['at_utc']=stamp
        for key in ('squeue','sacct'):original[key]['observer_host']=host;original[key]['at_utc']=stamp
        account={'at_utc':stamp,'observer_host':host,'stdout':json.dumps(retired['retired_cells'])}
        queue={'at_utc':stamp,'observer_host':host,'stdout':''}
        return {'at_utc':stamp,'observer_host':host,'original_terminal':original,
            'retired_recovery_terminal':{'terminal_accounting':account,'queue_observation':queue,
                'retired_cells':retired['retired_cells'],'retirement_path':str(cs.RETIREMENT),'retirement_sha256':u.file_sha(cs.RETIREMENT)}}
    write(t.root/'parents_terminal_before_prepare.json',parent_snapshot('2026-09-12T07:59:00+00:00','synthetic-spin'))
    intent=u.read(t.root/'submission_intent.json');intent['parents_terminal']=parent_snapshot(intent['at_utc'],'synthetic-spin')
    write(t.root/'submission_intent.json',intent)
    for i,runtime in enumerate(t.runtimes):
        runtime['plan_sha256']=u.file_sha(t.root/'plan.json')
        runtime['at_utc']=f'2026-09-12T{9+i:02d}:00:04+00:00'
        runtime['parents_terminal']=parent_snapshot(f'2026-09-12T{9+i:02d}:00:03+00:00',runtime['hostname'])
        write(t.root/f'runtime/{i}.json',runtime)
    t.terminal['logs_sha256']={p:u.file_sha(p) for p in t.terminal['logs_sha256']}
    write(t.root/'terminal_accounting.json',t.terminal)
    for key,name in (('HELPER','base.py'),('CELL_HELPER','cell_helper.py'),('ADAPTER','adapter.py'),('CS_HELPER','cs.py')):
        monkeypatch.setattr(u,key,write(t.root/name,'scratch dependency'))
    monkeypatch.setattr(u,'dependencies',lambda:(base,o.helper,o.modules,h.adapter,cs))
    result=u.inspect(t.root,u.file_sha(t.root/'terminal_accounting.json'))
    assert [v['completed_batches'] for v in result.summaries]==[464,400]
    assert result.original_execution['unstarted_cell']['start_utc']=='None'
    assert result.original_terminal_rows_by_index['2'][3]=='143:0'
    assert result.retired_recovery_execution['no_model_execution'] is True
    assert [v['original_index'] for v in result.recovery_executions]==[5,6]
    assert str(t.root/'runtime/0.json') in result.pins
    assert not (u.HISTORICAL_RECOVERY/'runtime/0.json').exists()
    assert len(o.delegated)==8
    extra=u.read(t.root/'runtime/1.json');extra['parents_terminal']['retired_recovery_terminal']['retired_cells']=[['other']]
    write(t.root/'runtime/1.json',extra)
    with pytest.raises(ValueError,match='retired unstarted'):u.inspect(t.root,u.file_sha(t.root/'terminal_accounting.json'))


@pytest.mark.parametrize('change',['wrong_host','before_allocation','after_runtime','retired_host','retired_before_original','parent_host'])
def test_worker_terminal_capture_cannot_escape_its_actual_host_or_execution(change,t):
    value={'hostname':'node202.ionic.cs.princeton.edu','at_utc':'2026-09-12T09:00:04+00:00'}
    snapshot={'observer_host':value['hostname'],'at_utc':'2026-09-12T09:00:03+00:00',
        'squeue':{'observer_host':value['hostname'],'at_utc':'2026-09-12T09:00:01+00:00'},
        'sacct':{'observer_host':value['hostname'],'at_utc':'2026-09-12T09:00:02+00:00'}}
    value['parents_terminal']={'observer_host':value['hostname'],'at_utc':snapshot['at_utc'],'original_terminal':snapshot,
        'retired_recovery_terminal':{key:{'observer_host':value['hostname'],'at_utc':snapshot['at_utc']}
            for key in ('terminal_accounting','queue_observation')}}
    t.u.worker_terminal_chronology(value,'2026-09-12T09:00:00')
    if change=='wrong_host': snapshot['observer_host']='spin.cs.princeton.edu'
    elif change=='before_allocation': snapshot['squeue']['at_utc']='2026-09-12T08:59:59+00:00'
    elif change=='after_runtime': snapshot['at_utc']='2026-09-12T09:00:05+00:00'
    elif change=='retired_host':value['parents_terminal']['retired_recovery_terminal']['terminal_accounting']['observer_host']='synthetic-spin'
    elif change=='retired_before_original':value['parents_terminal']['retired_recovery_terminal']['terminal_accounting']['at_utc']='2026-09-12T09:00:01+00:00'
    else:value['parents_terminal']['observer_host']='synthetic-spin'
    with pytest.raises(ValueError,match='actual host'): t.u.worker_terminal_chronology(value,'2026-09-12T09:00:00')


@pytest.mark.parametrize('field,value',[('original_terminal_rows_by_index',{}),('retired_recovery_execution',{'no_model_execution':False})])
def test_completion_cannot_drop_or_relabel_failed_and_retired_history(workflow,field,value):
    w=workflow; certificate=w.t.u.audit(w.t.root,w.digest)
    certificate[field]=value;write(w.t.root/'execution_reconciliation.json',certificate)
    with pytest.raises(ValueError):w.t.u.verify_existing(w.t.root)
    assert len(w.graders)==8
