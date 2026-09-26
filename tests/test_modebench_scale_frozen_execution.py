"""Final execution checks use scratch evidence only; no model, grader or scheduler."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def audit():
    path=ROOT/'artifacts/verify_modebench_scale_frozen_execution_20260912.py'
    spec=importlib.util.spec_from_file_location('scratch_final_execution',path)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def accounting(a):
    return {'schema':'modebench_scale_frozen_array_terminal_accounting_v1',
            'job_id':99,'command':['sacct','-j','99','-n','-P','--format='+a.FIELDS],
            'environment':{'TZ':'UTC'},'returncode':0,'observer_host':'wash.cs.princeton.edu',
            'captured_at_utc':'2026-09-12T05:00:00+00:00',
            'stdout':'\n'.join(f'{100+i}|99_{i}|COMPLETED|0:0|2026-09-12T02:00:00|2026-09-12T04:00:00|node105|6|60G|cpu=6,gres/gpu=2,gres/gpu:a5000=2,mem=60G,node=1|2026-09-12T01:00:00' for i in range(2))}


def test_complete_array_with_nonarray_raw_ids(audit):
    rows=audit.accounting_rows(accounting(audit),99,2)
    assert sorted(rows)==[0,1] and rows[1][0]=='101'


@pytest.mark.parametrize('before,after',[
    ('COMPLETED','RUNNING'),('COMPLETED','FAILED'),('0:0','1:0'),
    ('node105','node007'),('|6|60G|','|4|60G|'),('|60G|','|59G|'),
    ('gres/gpu=2','gres/gpu=1'),('gres/gpu:a5000=2','gres/gpu:a6000=2'),
    ('mem=60G','mem=59G'),('cpu=6,','cpu=6,cpu=6,'),
    ('2026-09-12T04:00:00','2026-09-12T06:00:00'),
    ('2026-09-12T02:00:00','2026-09-12T00:30:00')])
def test_wrong_terminal_state_resources_or_chronology_rejected(audit,before,after):
    v=accounting(audit);v['stdout']=v['stdout'].replace(before,after)
    with pytest.raises(ValueError): audit.accounting_rows(v,99,2)


def test_duplicate_or_missing_cell_rejected(audit):
    v=accounting(audit); lines=v['stdout'].splitlines()
    for values in ([lines[0]], [lines[0],lines[0]], [*lines,lines[0]]):
        v['stdout']='\n'.join(values)
        with pytest.raises(ValueError,match='per exact array'): audit.accounting_rows(v,99,2)


def test_batch_steps_do_not_count_as_array_cells(audit):
    v=accounting(audit);v['stdout']+='\n'+v['stdout'].splitlines()[0].replace('|99_0|','|99_0.batch|')
    assert len(audit.accounting_rows(v,99,2))==2


def test_timezone_and_query_authenticated(audit):
    for key,value in [('environment',{}),('returncode',1),('command',['sacct','-j','98']),('observer_host','')]:
        v=accounting(audit);v[key]=value
        with pytest.raises(ValueError): audit.accounting_rows(v,99,2)
    with pytest.raises(ValueError):audit.moment('2026-09-12T01:00:00')


def test_canonical_pin_collision_and_drift(audit,tmp_path):
    p=tmp_path/'file';p.write_text('one'); link=tmp_path/'alias';link.symlink_to(p)
    pins={};audit.pin_files(pins,[p,link]);assert len(pins)==1
    p.write_text('two')
    with pytest.raises(ValueError,match='changed'):audit.add_pins({},pins)
    with pytest.raises(ValueError,match='conflicting'):audit.add_pins(pins,{str(p):audit.sha(p)})


def test_atomic_publication_never_overwrites(audit,tmp_path):
    p=tmp_path/'final.json';audit.atomic_new(p,{'one':1})
    with pytest.raises(FileExistsError):audit.atomic_new(p,{'two':2})
    assert audit.read(p)=={'one':1}


@pytest.fixture
def final_case(audit,tmp_path,monkeypatch):
    a=audit;release=tmp_path/'release';release.mkdir();recovery=tmp_path/'recovery';recovery.mkdir()
    monkeypatch.setattr(a,'RELEASE',release);monkeypatch.setattr(a,'RECOVERY',recovery)
    proof={'status':'verified_recovery_only','files_sha256':{},'recovery_execution':{'job_id':31253118}}
    (recovery/'execution_audit.json').write_text(json.dumps(proof))
    expected={str(p):'reviewed' for p in (a.RECOVERY_AUDITOR,a.DRIVER,a.TRANSPORT)}
    monkeypatch.setattr(a,'requirements',lambda:({'files_sha256':expected},{}))
    legacy=SimpleNamespace(validate_schedule=lambda *args:({'initial_submit_array_spec':'0-9%1','per_cell_memory_overrides':{}},{},{}),
        validate_release=lambda *args:({'level4':{},'level5':{}},{}))
    objects={a.LEGACY:legacy,a.RECOVERY_AUDITOR:SimpleNamespace(audit=lambda:proof),a.DRIVER:object(),a.TRANSPORT:object(),a.LAUNCHER:object()}
    monkeypatch.setattr(a,'module',lambda p,*args:objects[p])
    calls=[]
    monkeypatch.setattr(a,'validate_authority',lambda *args:({'host':'wash'}, {}, {'owner_pid':1}))
    def stage(name,*args):
        calls.append(name);return {'stage':name},{}
    monkeypatch.setattr(a,'validate_stage',stage)
    return SimpleNamespace(a=a,release=release,recovery=recovery,legacy=legacy,calls=calls,proof=proof)


def test_read_only_then_one_final_publication_requires_both_stages(final_case):
    c=final_case;result=c.a.verify()
    assert result['schema']==c.a.SCHEMA and result['status']=='verified'
    assert result['difficulty_matched'] is True and set(result['levels'])=={'level4','level5'}
    assert c.calls==['revision_development','confirmation']
    assert not (c.release/'execution_provenance.json').exists()
    assert c.a.verify(publish=True)==result
    assert c.a.verify(publish=True)==result


def test_missing_scientific_admission_blocks_publication(final_case):
    c=final_case
    def missing(*args):raise ValueError('missing admission')
    c.legacy.validate_release=missing
    with pytest.raises(ValueError,match='missing admission'):c.a.verify(publish=True)
    assert not (c.release/'execution_provenance.json').exists() and not c.calls


def test_partial_recovery_audit_cannot_stand_in_for_final(final_case):
    c=final_case;(c.recovery/'execution_audit.json').write_text('{}')
    with pytest.raises(ValueError,match='reproduce exactly'):c.a.verify(publish=True)
    assert not (c.release/'execution_provenance.json').exists()


def test_bad_authority_or_any_stage_blocks_publication(final_case,monkeypatch):
    c=final_case
    def fail(*args):raise ValueError('missing required execution')
    monkeypatch.setattr(c.a,'validate_stage',fail)
    with pytest.raises(ValueError,match='missing required execution'):c.a.verify(publish=True)
    assert not (c.release/'execution_provenance.json').exists()


def test_legacy_partial_final_artifact_is_never_overwritten(final_case):
    c=final_case;p=c.release/'execution_provenance.json';p.write_text('{"status":"verified"}')
    with pytest.raises(ValueError,match='no overwrite'):c.a.verify(publish=True)
    assert c.a.read(p)=={'status':'verified'}


def test_failed_accounting_capture_creates_no_terminal_record(audit,tmp_path,monkeypatch):
    a=audit;monkeypatch.setattr(a,'EVIDENCE',tmp_path);monkeypatch.setattr(a,'COMPOSITE',tmp_path/'composite')
    stage=a.COMPOSITE/'confirmation';stage.mkdir(parents=True)
    (stage/'plan.json').write_text('{"cells":[{},{}]}')
    (stage/'submission_result.json').write_text('{"status":"submitted","array_job_id":99}')
    monkeypatch.setattr(a,'requirements',lambda:({},{}))
    v=accounting(a);v['stdout']=v['stdout'].replace('COMPLETED','RUNNING')
    monkeypatch.setattr(a.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=v['stdout'],stderr=''))
    with pytest.raises(ValueError):a.capture_accounting('confirmation')
    assert not (tmp_path/'confirmation_terminal_accounting.json').exists()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


def change(path,update):
    value=json.loads(Path(path).read_text());update(value);write(path,value)


@pytest.fixture
def authority_case(audit,tmp_path,monkeypatch):
    import os,pwd
    a=audit;root=tmp_path/'authority';root.mkdir();composite=tmp_path/'composite';composite.mkdir()
    monkeypatch.setattr(a,'AUTHORITY',root);monkeypatch.setattr(a,'COMPOSITE',composite)
    driver_path=tmp_path/'driver.py';driver_path.write_text('reviewed driver')
    monkeypatch.setattr(a,'DRIVER',driver_path)
    python=tmp_path/'literal_venv/python';python.parent.mkdir();python.write_text('runtime')
    transport_path=tmp_path/'transport.py';transport_path.write_text('reviewed transport')
    manifest=tmp_path/'manifest.json';write(manifest,{})
    lock=composite/'controller.lock';lock.write_text('')
    info=lock.stat();uid=os.getuid();host='wash.cs.princeton.edu'
    original=tmp_path/'original/plan.json';cells=[];receipts=[];task_paths=[]
    for i in range(10):
        level='level4' if i<5 else 'level5';domain=['countdown','graph_coloring','python','mathir','pantry'][i%5]
        cell={'id':level+'_'+domain,'level':level,'domain':domain,'model_label':'7b' if i<5 else '14b'}
        tasks=original.parent/f'tasks/{i}.json';values=[]
        for tier in range(4):
            receipt=original.parent/f'results/{i}_{tier}.json'
            write(receipt,{'status':'complete','level':level,'domain':domain,'model_label':cell['model_label'],'split':'dev',
                'generated_at':'2026-09-11T23:00:00+00:00'})
            receipts.append(receipt);values.append({'output':str(receipt)})
        write(tasks,values);task_paths.append(tasks);cells.append({**cell,'tasks':str(tasks)})
    write(original,{'cells':cells})
    def t(minute):return f'2026-09-12T00:{minute:02d}:00+00:00'
    activation={'schema':'modebench_scale_wash_authority_v1','root':str(root),'host':host,'uid':uid,
        'lock_path':str(lock),'lock_inode':info.st_ino,'transport':str(transport_path),
        'transport_sha256':a.sha(transport_path),'recovery_job_id':31253118,'view_manifest':str(manifest),
        'scope':'honest wash scope; historical soak unverified','created_at_utc':t(0),
        'inputs_sha256':{str(p):a.sha(p) for p in [manifest,original,*task_paths,driver_path,transport_path]}}
    write(root/'activation.json',activation);write(root/'activation.sha256.json',{'sha256':a.sha(root/'activation.json')})
    completed={'at_utc':t(2),'commands':[['squeue','-u',pwd.getpwuid(uid).pw_name,'-r','-h','-o','%i|%T'],
        ['sacct','-j','31253118','-X','-n','-P','--format=JobIDRaw,State,ExitCode,End,NodeList']],
        'stdout':['','31253118|COMPLETED|0:0|2026-09-12T00:01:50|node105\n'],
        'receipts_sha256':{str(p):a.sha(p) for p in receipts}}
    write(root/'recovery_completion.json',completed)
    runtime={'schema':'modebench_scale_wash_authority_runtime_v1','at_utc':t(4),'owner_pid':12345,
        'owner_start_ticks':'4567','owner_command':[str(python),'-B',str(driver_path),'watch','--root',str(root),'--interval','60'],
        'uid':uid,'host':host,'authority_path':str(root/'activation.json'),'authority_sha256':a.sha(root/'activation.json'),
        'lock_path':str(lock),'lock_inode':info.st_ino,'lock_device':info.st_dev,'fence_fd':8,
        'recovery_completion_path':str(root/'recovery_completion.json'),
        'recovery_completion_sha256':a.sha(root/'recovery_completion.json')}
    write(root/'outer_runtime.json',runtime)
    candidate={k:runtime[k] for k in ['owner_pid','owner_start_ticks','host','uid','lock_path','lock_inode','lock_device','fence_fd','authority_path','authority_sha256']}
    candidate.update(authority_runtime_path=str(root/'outer_runtime.json'),authority_runtime_sha256=a.sha(root/'outer_runtime.json'),
        view_manifest=str(manifest),view_manifest_sha256=a.sha(manifest))
    driver=SimpleNamespace(HOST=host,NEUTRAL_SHA='neutral',PYTHON=python,ORIGINAL_PLAN=original,
        verify=lambda root,guest: a.read(root/'activation.json'))
    driver.guest_command=lambda root,activation,n,fd:[str(python),'-B','/reviewed/runner.py','exec','--manifest',str(manifest),'--',
        str(python),'-B',str(driver_path),'guest-sweep','--root',str(root),'--sweep',str(n),'--fence-fd',str(fd)]
    def probe(minute):return {'command':[str(python),'-B',str(driver_path),'host-probe','--owner-pid','12345'],
        'returncode':0,'observation':{'at_utc':t(minute),'host':host,'uid':uid,'canonical_sha256':'neutral',
          'old_soak_os_state':'unverified','lock_inode':info.st_ino,'lock_path':str(lock),'lock_device':info.st_dev,
          'local_controllers':[{'pid':12345,'start_ticks':'4567','command':runtime['owner_command']}]}}
    write(root/'start_intent.json',{'at_utc':t(3),'activation_sha256':a.sha(root/'activation.json'),
        'host_probe':probe(1),'scope':activation['scope']})
    sweep=root/'sweeps/000001';sweep.mkdir(parents=True)
    write(sweep/'before.json',probe(5))
    argv=driver.guest_command(root,activation,1,8)
    write(sweep/'intent.json',{'at_utc':t(6),'command':argv,'authority':candidate,'before_sha256':a.sha(sweep/'before.json')})
    write(sweep/'guest_runtime.json',{'at_utc':t(7),'pid':22222,'start_ticks':'5678','uid':uid,'state':'R',
        'command':argv[argv.index('--')+1:],'authority':candidate,'fence_fd':8,'before_sha256':a.sha(sweep/'before.json')})
    result={'status':'admitted','release':'fixture'}
    write(sweep/'result.json',{'at_utc':t(8),'authority':candidate,'guest_runtime_sha256':a.sha(sweep/'guest_runtime.json'),'result':result})
    write(sweep/'exit.json',{'at_utc':t(9),'returncode':0});write(sweep/'after.json',probe(10))
    write(root/'terminal.json',{'at_utc':t(11),'last_sweep':1,'result':result})
    spec=importlib.util.spec_from_file_location('actual_transport_for_final_test',ROOT/'artifacts/modebench_scale_frozen_stage_transport_20260912.py')
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    for key,value in {'SOURCE':transport_path,'ARTIFACTS':composite,'AUTHORITY_ROOT':root,'ORIGINAL_PLAN':original}.items():
        monkeypatch.setattr(transport,key,value)
    return SimpleNamespace(a=a,root=root,sweep=sweep,driver=driver,transport=transport,candidate=candidate,
        runtime=runtime,activation=activation,receipts=receipts,original=original)


def test_complete_authority_ledger_authenticates_actual_transport_closure(authority_case):
    c=authority_case
    summary,pins,record=c.a.validate_authority(c.driver,c.transport)
    assert summary['host']=='wash.cs.princeton.edu' and summary['sweeps']==1
    assert record==c.candidate and set(map(str,c.receipts))<=set(pins)
    assert str(c.root/'recovery_completion.json') in pins


@pytest.mark.parametrize('file,field,value',[
    ('start_intent.json','activation_sha256','wrong'),('start_intent.json','scope','wrong'),
    ('terminal.json','last_sweep',2),('terminal.json','result',{'status':'admitted','release':'wrong'}),
    ('sweeps/000001/intent.json','command',['spoofed']),
    ('sweeps/000001/intent.json','before_sha256','wrong'),
    ('sweeps/000001/guest_runtime.json','command',['spoofed']),
    ('sweeps/000001/guest_runtime.json','uid',999999),('sweeps/000001/guest_runtime.json','fence_fd',9),
    ('sweeps/000001/guest_runtime.json','at_utc','2026-09-12T00:01:00+00:00'),
    ('sweeps/000001/exit.json','returncode',1),
    ('terminal.json','at_utc','2026-09-12T00:00:00+00:00')])
def test_authority_linkage_chronology_or_owner_changes_rejected(authority_case,file,field,value):
    c=authority_case;change(c.root/file,lambda v:v.update({field:value}))
    # Keep the result's byte linkage correct when testing guest semantics.
    if file.endswith('guest_runtime.json'):
        change(c.sweep/'result.json',lambda v:v.update(guest_runtime_sha256=c.a.sha(c.sweep/'guest_runtime.json')))
    with pytest.raises(ValueError):c.a.validate_authority(c.driver,c.transport)


@pytest.mark.parametrize('field,value',[('canonical_sha256','hinted'),('old_soak_os_state','dead'),
    ('host','soak.cs.princeton.edu'),('lock_device',-1),('local_controllers',[{'pid':6789}])])
def test_final_authority_requires_honest_outside_fence_observations(authority_case,field,value):
    c=authority_case;change(c.sweep/'after.json',lambda v:v['observation'].update({field:value}))
    with pytest.raises(ValueError,match='outside'):c.a.validate_authority(c.driver,c.transport)


def test_current_driver_and_transport_share_completion_contract(authority_case,monkeypatch):
    c=authority_case
    spec=importlib.util.spec_from_file_location('actual_driver_completion_integration',ROOT/'artifacts/continue_modebench_scale_composite_wash_20260912.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    monkeypatch.setattr(driver,'ORIGINAL_PLAN',c.original)
    completed=driver.runtime_completion(c.root,c.runtime)
    pins=c.transport.recovery_completion_pins(c.runtime,c.activation)
    assert pins=={str(c.root/'recovery_completion.json'):c.a.sha(c.root/'recovery_completion.json'),**completed['receipts_sha256']}
    assert driver.TRANSPORT_SHA==c.a.sha(ROOT/'artifacts/modebench_scale_frozen_stage_transport_20260912.py')


@pytest.fixture(params=['revision_development','confirmation'])
def stage_case(audit,tmp_path,monkeypatch,request):
    a=audit;composite=tmp_path/'composite';evidence=tmp_path/'evidence';release=tmp_path/'release';evidence.mkdir()
    monkeypatch.setattr(a,'COMPOSITE',composite);monkeypatch.setattr(a,'EVIDENCE',evidence);monkeypatch.setattr(a,'RELEASE',release)
    stage=request.param;phase='dev' if stage=='revision_development' else 'eval'
    path=composite/stage/'plan.json';outer=path.parent/'frozen_transport'
    manifest=tmp_path/'manifest.json';write(manifest,{})
    cell={'id':'level4_graph_coloring','level':'level4','domain':'graph_coloring','phase':phase,'model_label':'7b',
        'source_kind':'domain_revision_v1','source_root':str(tmp_path/'revision'),'command':['/literal/python','evaluator','--resume'],
        'tasks':str(path.parent/'tasks/graph.json')}
    tasks=[];receipt_paths=[]
    for tier in range(4 if phase=='dev' else 1):
        receipt=tmp_path/f'outputs/tier_{tier}.json';receipt_paths.append(receipt)
        task={'output':str(receipt),'split':phase,'seeds':[1,2,3,4],'batch_size':8};tasks.append(task)
        write(receipt,{'status':'complete','level':'level4','domain':'graph_coloring','model_label':'7b','split':phase,
            'generated_at':'2026-09-12T03:00:00+00:00','identity':{'source':{'fixture':'full rows'},'seeds':[1,2,3,4],'batch_size':8,
                'model':{'path':'/fixture/7b','label':'7b','vllm_version':'0.8.4'},
                'runtime':{'max_model_len':1024,'tensor_parallel_size':2,'swap_space':4.0,
                    'gpu_memory_utilization':0.82,'enable_prefix_caching':True}}})
        write(Path(str(receipt)+'.batches')/'0.json',{'fixture':'saved batch'})
    write(cell['tasks'],tasks)
    plan={'phase':phase,'models':{'7b':{'path':'/fixture/7b'}},'concurrency':1,'hardware':{'memory':'60G'},'cells':[cell],
        'submit_command':['sbatch','--parsable','--array=0-0%1','--mem=60G',str(path.parent/'worker.slurm'),str(path)],
        'runtime_profile':{'thread_environment':{'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}},
        'scientific_inputs_sha256':{str(manifest):a.sha(manifest)}}
    write(path,plan);write(path.parent/'plan.sha256.json',{'sha256':a.sha(path)})
    candidate={'owner_pid':12345}
    tx={'authority':candidate,'view_manifest':str(manifest),'prepared_at_utc':'2026-09-12T00:40:00+00:00',
        'canonical_submit_command':plan['submit_command'],'effective_submit_command':plan['submit_command'][:-2]+[str(outer/'worker.slurm'),str(path)],
        'inputs_sha256':{str(manifest):a.sha(manifest)}}
    write(outer/'plan.json',tx);write(outer/'plan.sha256.json',{'sha256':a.sha(outer/'plan.json')})
    write(path.parent/'submission_intent.json',{'at':'2026-09-12T00:50:00+00:00','status':'submission_attempt_started',
        'command':plan['submit_command'],'plan_sha256':a.sha(path)})
    write(outer/'submission_intent.json',{'at_utc':'2026-09-12T00:59:59.123+00:00','status':'effective_submission_attempt_started',
        'transport_plan_sha256':a.sha(outer/'plan.json'),'canonical_intent_sha256':a.sha(path.parent/'submission_intent.json'),
        'effective_command':tx['effective_submit_command'],'canonical_command':plan['submit_command']})
    write(outer/'submission_result.json',{'at_utc':'2026-09-12T01:00:01+00:00','returncode':0,'stdout':'99\n','stderr':'',
        'transport_plan_sha256':a.sha(outer/'plan.json'),'intent_sha256':a.sha(outer/'submission_intent.json')})
    write(path.parent/'submission_result.json',{'at':'2026-09-12T01:00:02+00:00','returncode':0,'stdout':'99\n','stderr':'',
        'status':'submitted','array_job_id':99,'cells':[{'array_index':0,'cell':cell['id']}]})
    raw=accounting(a);raw['stdout']=raw['stdout'].splitlines()[0]
    write(evidence/(stage+'_terminal_accounting.json'),raw)
    for level,sources in [('level4',{'graph_coloring':{'source_kind':cell['source_kind'],'source_root':cell['source_root']}}),('level5',{})]:
        write(release/level/'source_manifest.json',{'sources':sources})
    env={'SLURM_ARRAY_JOB_ID':'99','SLURM_ARRAY_TASK_ID':'0','SLURMD_NODENAME':'node105',
        'SLURM_MEM_PER_NODE':'61440','SLURM_CPUS_PER_TASK':'6','VLLM_USE_V1':'0','VLLM_ATTENTION_BACKEND':'XFORMERS',
        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}
    outer_runtime={'at_utc':'2026-09-12T02:01:00+00:00','status':'validated_before_unchanged_scientific_worker_exec',
        'array_job_id':99,'array_index':0,'stage_plan_sha256':a.sha(path),'transport_plan_sha256':a.sha(outer/'plan.json'),
        'view_manifest_sha256':a.sha(manifest),'effective_submission_result_sha256':a.sha(outer/'submission_result.json'),
        'canonical_submission_result_sha256':a.sha(path.parent/'submission_result.json'),'evaluator_command':cell['command'],
        'hostname':'node105.ionic.cs.princeton.edu','visible_gpu_names':['NVIDIA RTX A5000']*2,'vllm_version':'0.8.4','environment':env}
    write(outer/'runtime/0.json',outer_runtime)
    write(path.parent/'runtime/0.json',{'at':'2026-09-12T02:02:00+00:00','array_job_id':99,'cell':cell['id'],
        'plan_sha256':a.sha(path),'command':cell['command'],'hostname':'node105.ionic.cs.princeton.edu'})
    seed_checks=[]
    evaluator=SimpleNamespace(load_rows=lambda task:([{'full':'row'}],{'fixture':'full rows'}),
        model_identity=lambda path,label:{'path':str(path),'label':label},ENGINE_CONTRACT={'vllm_version':'0.8.4'},
        runtime_settings=lambda **kwargs:kwargs,
        validate_seed_receipt=lambda receipt,rows:seed_checks.append((receipt,rows)))
    launcher=SimpleNamespace(evaluator=lambda:evaluator)
    transport=SimpleNamespace(verify_transport=lambda launcher,path:(a.read(path),a.read(outer/'plan.json')))
    return SimpleNamespace(a=a,path=path,outer=outer,stage=stage,plan=plan,tx=tx,candidate=candidate,launcher=launcher,
        transport=transport,receipts=receipt_paths,seed_checks=seed_checks,release=release,evidence=evidence)


def test_complete_stage_binds_both_ledgers_worker_and_all_four_task_receipts(stage_case):
    c=stage_case;summary,pins=c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate)
    assert summary['array_job_id']==99 and summary['cells']==1
    assert len(c.seed_checks)==(4 if c.stage=='revision_development' else 1) and set(map(str,c.receipts))<=set(pins)


@pytest.mark.parametrize('field,value',[('phase','eval'),('concurrency',2)])
def test_wrong_stage_phase_or_concurrency_rejected(stage_case,field,value):
    c=stage_case;value=('eval' if c.plan['phase']=='dev' else 'dev') if field=='phase' else value
    change(c.path,lambda v:v.update({field:value}))
    with pytest.raises(ValueError):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate)


@pytest.mark.parametrize('field,value',[('command',['wrong']),('plan_sha256','wrong'),('status','wrong')])
def test_final_stage_authenticates_original_intent_semantics_even_if_digest_resealed(stage_case,field,value):
    c=stage_case;p=c.path.parent/'submission_intent.json';change(p,lambda v:v.update({field:value}))
    change(c.outer/'submission_intent.json',lambda v:v.update(canonical_intent_sha256=c.a.sha(p)))
    change(c.outer/'submission_result.json',lambda v:v.update(intent_sha256=c.a.sha(c.outer/'submission_intent.json')))
    with pytest.raises(ValueError,match='submission linkage'):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate)


@pytest.mark.parametrize('field,value',[('vllm_version','0.9.0'),('visible_gpu_names',['A100','A100']),
    ('evaluator_command',['changed']),('at_utc','2026-09-12T05:00:00+00:00')])
def test_final_stage_rejects_actual_runtime_drift(stage_case,field,value):
    c=stage_case;change(c.outer/'runtime/0.json',lambda v:v.update({field:value}))
    with pytest.raises(ValueError):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate)


@pytest.mark.parametrize('field,value',[('split','eval'),('model_label','14b'),
    ('generated_at','2026-09-12T01:00:00+00:00'),('generated_at','2026-09-12T05:00:00+00:00')])
def test_final_stage_rejects_receipt_phase_model_or_execution_time(stage_case,field,value):
    c=stage_case;value=('eval' if c.plan['phase']=='dev' else 'dev') if field=='split' else value
    change(c.receipts[0],lambda v:v.update({field:value}))
    with pytest.raises(ValueError):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate)


def test_every_admitted_revision_source_must_appear_once(stage_case):
    c=stage_case;change(c.release/'level5/source_manifest.json',lambda v:v['sources'].update(
        pantry={'source_kind':'domain_revision_v1','source_root':'/missing/source'}))
    with pytest.raises(ValueError,match='every exact admitted source'):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate)


def test_complete_seed_receipt_validation_is_mandatory(stage_case):
    c=stage_case
    def reject(*args):raise ValueError('seed receipt corrupted')
    c.launcher.evaluator().validate_seed_receipt=reject
    with pytest.raises(ValueError,match='seed receipt corrupted'):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate)


@pytest.mark.parametrize('field,value',[('model',{'label':'7b','path':'/wrong/checkpoint','vllm_version':'0.8.4'}),
    ('runtime',{'max_model_len':2048}),('seeds',[2,3,4,5]),('source',{'fixture':'different rows'})])
def test_final_stage_receipt_must_match_exact_checkpoint_runtime_and_full_task(stage_case,field,value):
    c=stage_case;change(c.receipts[0],lambda v:v['identity'].update({field:value}))
    with pytest.raises(ValueError,match='input/seed identity'):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate)
