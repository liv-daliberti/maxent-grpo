"""Scratch operational fixtures only; no scheduler, model, fit, or real grading."""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


def mutate(path,function):
    path.chmod(0o644);value=json.loads(path.read_text());function(value);write(path,value)


@pytest.fixture
def factory(tmp_path,monkeypatch):
    def create(stage='revision_development',kind='domain_revision_v1'):
        spec=importlib.util.spec_from_file_location('scratch_cell_'+stage+kind,ROOT/'artifacts/reconcile_modebench_scale_completed_stage_cell_20260912.py')
        a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
        base=tmp_path/(stage+kind);base.mkdir()
        def file(name,contents='scratch'):
            path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(contents);return path
        for key,value in {'ROOT':base,'SOURCE':file('helper.py'),'COMPOSITE':base/'composite',
            'TRANSPORT':file('transport.py'),'REVISION':file('revision.py')}.items():monkeypatch.setattr(a,key,value)
        stagepath=a.COMPOSITE/stage;stagepath.mkdir(parents=True)
        path=stagepath/'plan.json';directory=stagepath/'execution_reconciliations/0';directory.mkdir(parents=True)
        phase='dev' if stage=='revision_development' else 'eval'; count=4 if phase=='dev' else 1; row_count=10 if phase=='dev' else 128
        source_root=base/'source';level='level4';domain='graph_coloring';label='7b'
        model=base/'model';model.mkdir();model_identity={'label':label,'path':str(model)}
        labels=[7102000+i for i in range(4)]
        runtime={'dtype':'float16','enable_prefix_caching':True,'gpu_memory_utilization':.82,
                 'swap_space':4.0,'tensor_parallel_size':2,'max_model_len':1024}
        interface={'prompt_profile':'hybrid_solver_v4','max_model_len':1024,'max_tokens':192}
        draws={phase:labels}
        protocol={'files_sha256':{str(a.REVISION):a.file_sha(a.REVISION)},'level':level,'domain':domain,'revision':2,
            'draw_labels':{level:draws} if kind=='campaign_v1' else draws,
            'models':{label:model_identity},'targets':{domain:{'metrics':{'pass1':.5,'pass8':.5}}},
            'tolerances':{'pass1':.04,'pass8':.08}}
        protocolpath=source_root/'protocol.json';write(protocolpath,protocol)
        pool=source_root/level/('pools' if phase=='dev' else 'dataset')/domain
        tasks=[];receipts=[];all_rows=[];sources={};pooltiers={};schedules={}
        def schedule(domain,rows,seeds):return {'rows':[a.sha(row) for row in rows],'seeds':seeds}
        for tier in range(count):
            rows=[{'problem':f'prompt{tier}-{i}','answer':'fixture'} for i in range(row_count)];all_rows.append(rows)
            source=pool/(f'difficulty_{tier}.jsonl' if phase=='dev' else 'eval.jsonl');source.parent.mkdir(parents=True,exist_ok=True)
            source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            source_identity={'path':str(source),'file_sha256':a.file_sha(source),'rows_sha256':a.sha(rows),'row_offset':0,'row_limit':0}
            sources[str(source)]=(rows,source_identity)
            output=source_root/level/'results'/('development' if phase=='dev' else 'confirmation')
            output=output/domain/f'difficulty_{tier}.json' if phase=='dev' else output/(domain+'.json')
            task={'level':level,'domain':domain,'split':phase,'batch_size':8,'row_limit':0,'row_offset':0,
                  'output':str(output),'rows_jsonl':str(source),'seeds':labels,'interface':'fixture'};tasks.append(task)
            identity={'source':source_identity,'seeds':labels,'runtime':runtime,'interface':interface,
                      'model':{**model_identity,'vllm_version':'0.8.4'},'seed_schedule_sha256':a.sha(schedule(domain,rows,labels)),
                      'rendered_prompts_sha256':a.sha([row['problem'] for row in rows])}
            schedules[str(output)]=identity['seed_schedule_sha256']
            results=[{'draws':[{'seed':seed,'attempts':[{'text':f'{i}-{j}','canonical_key':None,'verified':False} for j in range(8)]}
                              for seed in labels]} for i in range(row_count)]
            receipt={'level':level,'domain':domain,'split':phase,'model_label':label,'identity':identity,'identity_sha256':a.sha(identity),'status':'complete',
                     'generated_at':'2026-09-12T02:58:39+00:00','metrics':{'rows':row_count,'pass1':.1,'pass8':.1},
                     'prompt_results':results};write(output,receipt);receipts.append(output)
            batches=Path(str(output)+'.batches');write(batches/'run.json',{'identity':identity,'identity_sha256':a.sha(identity)})
            for draw_index,seed in enumerate(labels):
                for start in range(0,row_count,8):
                    end=min(start+8,row_count);values=[r['draws'][draw_index] for r in results[start:end]]
                    write(batches/f'seed-{seed}__rows-{start:06d}-{end:06d}.json',
                        {'seed':seed,'start':start,'end':end,'identity_sha256':a.sha(identity),'draws':values,'draws_sha256':a.sha(values)})
            pooltiers[str(tier)]={'rows':row_count,'rows_sha256':a.sha(rows)}
        poolidentity={'protocol_sha256':a.file_sha(protocolpath),'tiers':pooltiers,'splits':{'eval':pooltiers['0']}}
        write(pool/'identity.json',poolidentity)
        taskpath=stagepath/'tasks/fixture.json';write(taskpath,tasks)
        cell={'id':'level4_graph_coloring_r2_'+phase,'source_root':str(source_root),'source_kind':kind,'level':level,'domain':domain,
              'model_label':label,'phase':phase,'tasks':str(taskpath),'command':['/literal/python','/literal/evaluator','--resume']}
        pins={str(p):a.file_sha(p) for p in [protocolpath,a.REVISION,pool/'identity.json',taskpath,*map(Path,sources)]}
        plan={'cells':[cell],'models':{label:{'path':str(model)}},'hardware':{'memory':'60G'},'submit_command':['sbatch','fixed'],
              'immutable_inputs_sha256':pins,'rng_admission':{'task_seed_schedule_sha256':schedules}}
        write(path,plan);write(stagepath/'plan.sha256.json',{'sha256':a.file_sha(path)})
        monkeypatch.setattr(a,'DEV_PLAN_SHA',a.file_sha(path))
        write(stagepath/'submission_intent.json',{'command':plan['submit_command'],'plan_sha256':a.file_sha(path)})
        submitted={'status':'submitted','array_job_id':31254520,'returncode':0,'stdout':'31254520\n','stderr':'','at':'2026-09-12T02:52:21.100000+00:00'}
        write(stagepath/'submission_result.json',submitted)
        txpath=stagepath/'frozen_transport/plan.json';view=file('view.json','{}')
        tx={'inputs_sha256':pins,'effective_submit_command':['sbatch','explicit-wrapper'],'view_manifest':str(view)}
        write(txpath,tx);write(txpath.parent/'plan.sha256.json',{'sha256':a.file_sha(txpath)})
        txintent={'canonical_intent_sha256':a.file_sha(stagepath/'submission_intent.json'),'canonical_command':plan['submit_command'],
                  'effective_command':tx['effective_submit_command']};write(txpath.parent/'submission_intent.json',txintent)
        effective={'returncode':0,'stdout':'31254520\n','stderr':'','intent_sha256':a.file_sha(txpath.parent/'submission_intent.json'),
                   'transport_plan_sha256':a.file_sha(txpath)};write(txpath.parent/'submission_result.json',effective)
        inner={'array_job_id':31254520,'cell':cell['id'],'command':cell['command'],'plan_sha256':a.file_sha(path),
               'hostname':'node105.ionic.cs.princeton.edu','at':'2026-09-12T02:53:15+00:00'};write(stagepath/'runtime/0.json',inner)
        env={'SLURM_JOB_ID':'31254521','SLURM_ARRAY_JOB_ID':'31254520','SLURM_ARRAY_TASK_ID':'0','SLURMD_NODENAME':'node105',
             'SLURM_CPUS_PER_TASK':'6','SLURM_MEM_PER_NODE':'61440','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
             'VLLM_USE_V1':'0','VLLM_ATTENTION_BACKEND':'XFORMERS','PYTHONDONTWRITEBYTECODE':'1',
             'PYTHONPYCACHEPREFIX':'/tmp/NONPRODUCTION-scale-frozen-pycache-fixture/pycache'}
        observed={'array_job_id':31254520,'array_index':0,'evaluator_command':cell['command'],'stage_plan_sha256':a.file_sha(path),
            'transport_plan_sha256':a.file_sha(txpath),'canonical_submission_result_sha256':a.file_sha(stagepath/'submission_result.json'),
            'effective_submission_result_sha256':a.file_sha(txpath.parent/'submission_result.json'),'view_manifest_sha256':a.file_sha(view),
            'status':'validated_before_unchanged_scientific_worker_exec','environment':env,'hostname':'node105.ionic.cs.princeton.edu',
            'vllm_version':'0.8.4','visible_gpu_names':['NVIDIA RTX A5000']*2,'at_utc':'2026-09-12T02:53:11+00:00'}
        write(txpath.parent/'runtime/0.json',observed)
        logs=stagepath/'logs';logs.mkdir();out=logs/'31254520_0.out';err=logs/'31254520_0.err'
        out.write_text(''.join(json.dumps({'event':'task_complete','output':str(p),'metrics':a.read(p)['metrics']})+'\n' for p in receipts));err.write_text('scratch shutdown warning')
        rows=[]
        for suffix,state,code in [('', 'FAILED','1:0'),('.batch','FAILED','1:0'),('.extern','COMPLETED','0:0')]:
            rows.append(['31254521'+suffix,'31254520_0'+suffix,state,code,'2026-09-12T02:52:22','2026-09-12T02:58:44',
                'node105','6','60G' if not suffix else '', 'cpu=6,node=1,gres/gpu=2,gres/gpu:a5000=2,mem=60G','2026-09-12T02:52:21'])
        terminal={'schema':'modebench_scale_stage_cell_terminal_accounting_v1','stage':stage,'array_index':0,'array_job_id':31254520,
            'job_id':'31254520_0','command':['sacct','-j','31254520_0','-n','-P','--format='+a.FIELDS],'returncode':0,
            'environment':{'TZ':'UTC'},'observer_host':'wash.cs.princeton.edu','stderr':'','captured_at_utc':'2026-09-12T03:01:00+00:00',
            'stdout':''.join('|'.join(row)+'\n' for row in rows),'logs_sha256':{str(p):a.file_sha(p) for p in [out,err]}}
        terminalpath=directory/'terminal_accounting.json';write(terminalpath,terminal);digest=a.file_sha(terminalpath)
        monkeypatch.setattr(a,'FIRST_TERMINAL_SHA',digest)
        grading=[]
        def receipt_scores(output,rows,p,phase,*,regrade):
            grading.append((output,regrade));return a.read(output),{str(i):{} for i in range(len(rows))}
        def validate(receipt,rows):
            a.require(receipt['status']=='complete' and receipt['identity_sha256']==a.sha(receipt['identity']), 'receipt identity changed')
            a.require(len(receipt['prompt_results'])==len(rows), 'receipt incomplete')
        evaluator=SimpleNamespace(validate_task=lambda task,confirm: None,load_rows=lambda task:sources[task['rows_jsonl']],
            validate_seed_receipt=validate,runtime_settings=lambda **kwargs:{'dtype':'float16',**kwargs},
            model_identity=lambda path,label:model_identity,frozen_interface=lambda domain:interface,
            frozen=SimpleNamespace(schedule_record=schedule,prompt_messages=lambda domain,problem,profile:problem))
        revision=SimpleNamespace(authenticate=lambda root:a.read(Path(root)/'protocol.json'),receipt_scores=receipt_scores)
        def verify_transport(launcher,path):
            a.pins_check(a.read(path)['immutable_inputs_sha256']);return a.read(path),a.read(path.parent/'frozen_transport/plan.json')
        modules=SimpleNamespace(transport=SimpleNamespace(verify_transport=verify_transport),launcher=object(),revision=revision,evaluator=evaluator)
        monkeypatch.setattr(a,'scientific_modules',lambda:modules)
        tokenizer=SimpleNamespace(is_fast=True,apply_chat_template=lambda messages,**kw:messages,encode=lambda prompt,**kw:[1]*10)
        monkeypatch.setattr(a,'tokenizer_for',lambda context:tokenizer)
        if phase=='eval':
            recipe=source_root/level/'recipes'/(domain+'.json');write(recipe,{'development_fit_pass':True})
            confirmation={'schema':'modebench_scale_domain_revision_confirmation_v1' if kind=='domain_revision_v1' else 'modebench_scale_confirmation_v1',
                'level':level,'domain':domain,'revision':2,'metrics':a.read(receipts[0])['metrics'],
                'receipt_sha256':a.file_sha(receipts[0]),'recipe_sha256':a.file_sha(recipe),'dataset_identity_sha256':a.file_sha(pool/'identity.json'),
                'original_grader_replayed_attempts':4096,'difficulty_matched':False,'target':{'pass1':.5,'pass8':.5},
                'differences':{'pass1':-.4,'pass8':-.4},'gates':{'pass1':False,'pass8':False},
                'cross_source_disjointness':{'files_sha256':{str(protocolpath):a.file_sha(protocolpath)}}}
            write(source_root/level/'confirmation'/(domain+'.json'),confirmation)
        return SimpleNamespace(a=a,stage=stage,path=path,directory=directory,digest=digest,cell=cell,tasks=tasks,receipts=receipts,
            modules=modules,grading=grading,rows=all_rows,protocol=protocol,terminal=terminal,observed=observed,source_root=source_root)
    return create


def test_registration_pins_complete_identity_before_grading(factory):
    c=factory();r=c.a.register(c.stage,0,c.digest)
    assert r['terminal']['job_id_raw']=='31254521' and r['terminal']['job_id']=='31254520_0'
    assert r['terminal']['state']=='FAILED' and len(r['summaries'])==4
    assert r['files_sha256'][str(c.directory/'terminal_accounting.json')]==c.digest
    assert c.grading==[] and not (c.directory/'claim.json').exists()
    with pytest.raises(ValueError,match='already exists'):c.a.register(c.stage,0,c.digest)


def test_exact_once_audit_and_verify_never_regrades(factory):
    c=factory();value=c.a.audit(c.stage,0,c.digest)
    assert c.grading==[(str(path),True) for path in c.receipts]
    assert value['scheduler_success'] is False and value['scientific_outputs_complete'] is True and value['exit_cause']=='unknown'
    assert value['attempts_validated']==1280 and value['completed_batches']==32
    assert c.a.verify_reconciliation(c.stage,0)==value
    assert len(c.grading)==4
    with pytest.raises(ValueError,match='already claimed'):c.a.audit(c.stage,0,c.digest)
    assert len(c.grading)==4


@pytest.mark.parametrize('kind',['domain_revision_v1','campaign_v1'])
def test_confirmation_reuses_existing_canonical_failure_without_grading(factory,kind):
    c=factory('confirmation',kind);value=c.a.audit(c.stage,0,c.digest)
    assert c.grading==[] and value['new_grader_invocations']==0
    assert value['canonical_confirmation_audit']['difficulty_matched'] is False
    assert value['scientific_outputs_complete'] is True and 'difficulty_matched' not in value
    c.a.verify_reconciliation(c.stage,0);assert c.grading==[]


def test_missing_confirmation_audit_does_not_claim_or_grade(factory):
    c=factory('confirmation');(c.source_root/'level4/confirmation/graph_coloring.json').unlink()
    with pytest.raises(ValueError,match='wait for unchanged core'):c.a.audit(c.stage,0,c.digest)
    assert not (c.directory/'claim.json').exists() and c.grading==[]


@pytest.mark.parametrize('field,value',[('difficulty_matched',True),('gates',{'pass1':True,'pass8':True}),
    ('differences',{'pass1':0,'pass8':0}),('original_grader_replayed_attempts',1),('receipt_sha256','0'*64)])
def test_confirmation_cannot_relabel_or_weaken_canonical_audit(factory,field,value):
    c=factory('confirmation');mutate(c.source_root/'level4/confirmation/graph_coloring.json',lambda r:r.update({field:value}))
    with pytest.raises(ValueError,match='canonical confirmation'):c.a.register(c.stage,0,c.digest)
    assert c.grading==[]


@pytest.mark.parametrize('column,value',[(0,'31254520'),(1,'31254521'),(2,'COMPLETED'),(3,'0:0'),
    (5,'2026-09-12T02:58:43'),(6,'node007'),(7,'8'),(8,'59G')])
def test_terminal_exact_raw_human_failure_resources_guard(factory,monkeypatch,column,value):
    c=factory()
    def change(r):
        rows=[line.split('|') for line in r['stdout'].splitlines()];rows[0][column]=value;r['stdout']=''.join('|'.join(row)+'\n' for row in rows)
    mutate(c.directory/'terminal_accounting.json',change);digest=c.a.file_sha(c.directory/'terminal_accounting.json')
    monkeypatch.setattr(c.a,'FIRST_TERMINAL_SHA',digest)
    with pytest.raises(ValueError):c.a.register(c.stage,0,digest)
    assert not (c.directory/'registration.json').exists() and c.grading==[]


@pytest.mark.parametrize('field,value',[('SLURM_JOB_ID','31254520'),('SLURM_ARRAY_TASK_ID','1'),('SLURM_ARRAY_JOB_ID','1'),
    ('VLLM_USE_V1','1'),('SLURM_MEM_PER_NODE','60416')])
def test_actual_runtime_identity_and_environment_guard(factory,field,value):
    c=factory();mutate(c.path.parent/'frozen_transport/runtime/0.json',lambda r:r['environment'].update({field:value}))
    with pytest.raises(ValueError):c.a.register(c.stage,0,c.digest)
    assert c.grading==[]


def test_terminal_sha_explicit_and_first_anchor(factory):
    c=factory()
    with pytest.raises(ValueError,match='explicit captured terminal'):c.a.register(c.stage,0,'0'*64)
    assert c.grading==[]


def test_registered_output_mutation_refused_before_claim(factory):
    c=factory();c.a.register(c.stage,0,c.digest)
    mutate(c.receipts[0],lambda r:r['metrics'].update(pass1=.8))
    with pytest.raises(ValueError,match='pinned evidence changed'):c.a.audit(c.stage,0,c.digest)
    assert c.grading==[] and not (c.directory/'claim.json').exists()


def test_partial_final_batch_must_match_receipt(factory):
    c=factory();path=next(Path(str(c.receipts[3])+'.batches').glob('*000008-000010.json'))
    def change(r):r['draws'][0]['attempts'][0]['text']='tampered';r['draws_sha256']=c.a.sha(r['draws'])
    mutate(path,change)
    with pytest.raises(ValueError,match='batch draws differ'):c.a.register(c.stage,0,c.digest)
    assert c.grading==[]


@pytest.mark.parametrize('kind',['missing_receipt','missing_batch','extra_batch','bad_run'])
def test_output_inventory_must_be_complete(factory,kind):
    c=factory();directory=Path(str(c.receipts[3])+'.batches')
    if kind=='missing_receipt':c.receipts[3].unlink()
    if kind=='missing_batch':next(directory.glob('seed-*')).unlink()
    if kind=='extra_batch':write(directory/'extra.json',{})
    if kind=='bad_run':mutate(directory/'run.json',lambda r:r.update(identity_sha256='bad'))
    with pytest.raises((ValueError,FileNotFoundError)):c.a.register(c.stage,0,c.digest)
    assert c.grading==[]


def test_grader_failure_preserves_claim_and_prevents_repeat(factory,monkeypatch):
    c=factory();called=[]
    def fail(*args,**kwargs):called.append(True);raise ValueError('scratch grader disagreed')
    monkeypatch.setattr(c.modules.revision,'receipt_scores',fail)
    with pytest.raises(ValueError,match='grader disagreed'):c.a.audit(c.stage,0,c.digest)
    assert (c.directory/'claim.json').exists() and (c.directory/'failure.json').exists()
    with pytest.raises(ValueError,match='already claimed'):c.a.audit(c.stage,0,c.digest)
    assert called==[True] and not (c.directory/'reconciliation.json').exists()


def test_wrong_native_prompts_prevent_grading(factory,monkeypatch):
    c=factory();monkeypatch.setattr(c.a,'tokenizer_for',lambda ctx:SimpleNamespace(
        apply_chat_template=lambda *args,**kw:'different',encode=lambda *args,**kw:[1]))
    with pytest.raises(ValueError,match='native rendered prompt'):c.a.audit(c.stage,0,c.digest)
    assert c.grading==[] and (c.directory/'failure.json').exists()


def test_action_lock_excludes_duplicate_owner_without_controller_lock(factory):
    c=factory()
    with c.a.action_lock(c.directory):
        with pytest.raises(BlockingIOError):c.a.audit(c.stage,0,c.digest)
    assert not (c.directory/'claim.json').exists()


def test_saved_certificate_cannot_lie_or_omit_pins(factory):
    c=factory();c.a.audit(c.stage,0,c.digest)
    mutate(c.directory/'reconciliation.json',lambda r:r.update(scheduler_success=True))
    with pytest.raises(ValueError,match='differs from authenticated'):c.a.verify_reconciliation(c.stage,0)
    assert len(c.grading)==4


@pytest.mark.parametrize('generated,passes',[('2026-09-12T02:58:44.999999+00:00',True),
    ('2026-09-12T02:58:45.000000+00:00',False),('2026-09-12T02:52:21.999999+00:00',False)])
def test_receipt_chronology_respects_scheduler_second_precision(factory,generated,passes):
    c=factory();mutate(c.receipts[-1],lambda r:r.update(generated_at=generated))
    if passes:c.a.register(c.stage,0,c.digest)
    else:
        with pytest.raises(ValueError,match='outside actual cell execution'):c.a.register(c.stage,0,c.digest)
    assert c.grading==[]


@pytest.mark.parametrize('field,value',[('level','level5'),('domain','pantry'),('split','dev'),('model_label','14b')])
def test_confirmation_receipt_routing_cannot_differ_from_cell(factory,field,value):
    c=factory('confirmation');mutate(c.receipts[0],lambda r:r.update({field:value}))
    with pytest.raises(ValueError,match='receipt routing'):c.a.register(c.stage,0,c.digest)
    assert c.grading==[]


def test_receipt_must_follow_actual_inner_worker_entry(factory):
    c=factory();mutate(c.receipts[0],lambda r:r.update(generated_at='2026-09-12T02:53:14.999999+00:00'))
    with pytest.raises(ValueError,match='outside actual cell execution'):c.a.register(c.stage,0,c.digest)
    assert c.grading==[]


def test_explicit_authenticated_prior_tier_skips_duplicate_grading(factory):
    c=factory();context=c.a.inspect_cell(c.stage,0,c.digest)
    prior=c.directory/'prior_original_grader.json'
    write(prior,{'schema':c.a.TIER_SCHEMA,'status':'passed','summary':context.summaries[0],
        'original_grader_agrees':True,'validation_entrypoint':'revision.receipt_scores(regrade=True)',
        'new_grader_invocations':context.summaries[0]['attempts'],'files_sha256':context.pins})
    value=c.a.audit(c.stage,0,c.digest,[prior])
    assert c.grading==[(str(path),i!=0) for i,path in enumerate(c.receipts)]
    assert value['new_grader_invocations']==960
    assert c.a.verify_reconciliation(c.stage,0)==value and len(c.grading)==4


def test_prior_tier_with_different_outputs_cannot_suppress_grading(factory):
    c=factory();context=c.a.inspect_cell(c.stage,0,c.digest);prior=c.directory/'prior_wrong.json'
    summary=dict(context.summaries[0]);summary['receipt_sha256']='0'*64
    write(prior,{'schema':c.a.TIER_SCHEMA,'status':'passed','summary':summary,'original_grader_agrees':True,
        'validation_entrypoint':'revision.receipt_scores(regrade=True)','new_grader_invocations':summary['attempts'],
        'files_sha256':context.pins})
    with pytest.raises(ValueError,match='prior grader audit'):c.a.audit(c.stage,0,c.digest,[prior])
    assert c.grading==[] and not (c.directory/'claim.json').exists()
