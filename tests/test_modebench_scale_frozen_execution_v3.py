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
    path=ROOT/'artifacts/verify_modebench_scale_frozen_execution_v3_20260912.py'
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
    proof={'status':'verified_scientific_outputs_with_failed_execution','files_sha256':{},
        'scheduler_success':False,'scientific_outputs_complete':True,'exit_cause':'unknown',
        'recovery_execution':{'job_id':31253118,'state':'FAILED','exit_code':'1:0'}}
    (recovery/'execution_reconciliation.json').write_text(json.dumps(proof))
    expected={str(p):'reviewed' for p in (a.RECOVERY_AUDITOR,a.DRIVER,a.TRANSPORT,a.CELL_RECONCILER)}
    monkeypatch.setattr(a,'requirements',lambda:({'files_sha256':expected},{}))
    legacy=SimpleNamespace(validate_schedule=lambda *args:({'initial_submit_array_spec':'0-9%1','per_cell_memory_overrides':{}},{},{}),
        validate_release=lambda *args:({'level4':{},'level5':{}},{}))
    objects={a.LEGACY:legacy,a.RECOVERY_AUDITOR:SimpleNamespace(verify_reconciliation=lambda:proof),a.DRIVER:object(),a.TRANSPORT:object(),a.LAUNCHER:object(),a.CELL_RECONCILER:object()}
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
    c=final_case;(c.recovery/'execution_reconciliation.json').write_text('{}')
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
    activation={'schema':'modebench_scale_wash_authority_v2','root':str(root),'host':host,'uid':uid,
        'lock_path':str(lock),'lock_inode':info.st_ino,'transport':str(transport_path),
        'transport_sha256':a.sha(transport_path),'recovery_job_id':31253118,'view_manifest':str(manifest),
        'scope':'honest wash scope; historical soak unverified','created_at_utc':t(0),
        'inputs_sha256':{str(p):a.sha(p) for p in [manifest,original,*task_paths,driver_path,transport_path]}}
    write(root/'activation.json',activation);write(root/'activation.sha256.json',{'sha256':a.sha(root/'activation.json')})
    reconciliation=tmp_path/'recovery/execution_reconciliation.json';monkeypatch.setattr(a,'RECOVERY',reconciliation.parent)
    verifier_path=tmp_path/'recovery_verifier.py';verifier_path.write_text('reviewed verifier')
    proof={'schema':'modebench_scale_frozen_recovery_execution_reconciliation_v2',
        'status':'verified_scientific_outputs_with_failed_execution','scheduler_success':False,
        'scientific_outputs_complete':True,'exit_cause':'unknown','files_sha256':{str(p):a.sha(p) for p in receipts},
        'recovery_execution':{'job_id':31253118,'state':'FAILED','exit_code':'1:0','end_utc':'2026-09-12T00:01:50'}}
    write(reconciliation,proof)
    reconciliation_verification={'command':[str(python),'-B','/reviewed/runner.py','exec','--manifest',str(manifest),'--',
        str(python),'-B',str(verifier_path),'--root',str(reconciliation.parent),'--verify-existing'],
        'returncode':0,'observation':{**{k:proof[k] for k in ['schema','status','scheduler_success','scientific_outputs_complete','exit_cause']},
        'certificate_path':str(reconciliation),'certificate_sha256':a.sha(reconciliation)}}
    write(root/'preparation_reconciliation_verification.json',reconciliation_verification)
    activation['inputs_sha256'].update({str(p):a.sha(p) for p in [reconciliation,verifier_path,root/'preparation_reconciliation_verification.json']})
    write(root/'activation.json',activation);write(root/'activation.sha256.json',{'sha256':a.sha(root/'activation.json')})
    completed={'schema':'modebench_scale_wash_reconciled_completion_v2','scheduler_success':False,'exit_cause':'unknown',
        'reconciliation_path':str(reconciliation),'reconciliation_sha256':a.sha(reconciliation),'at_utc':t(2),'commands':[['squeue','-u',pwd.getpwuid(uid).pw_name,'-r','-h','-o','%i|%T'],
        ['sacct','-j','31253118','-X','-n','-P','--format=JobIDRaw,State,ExitCode,End,NodeList']],
        'stdout':['','31253118|FAILED|1:0|2026-09-12T00:01:50|node105\n'],
        'receipts_sha256':{str(p):a.sha(p) for p in receipts}}
    write(root/'recovery_completion.json',completed)
    runtime={'schema':'modebench_scale_wash_authority_runtime_v2','at_utc':t(4),'owner_pid':12345,
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
    driver.reconciliation_command=lambda manifest:reconciliation_verification['command']
    driver.guest_command=lambda root,activation,n,fd:[str(python),'-B','/reviewed/runner.py','exec','--manifest',str(manifest),'--',
        str(python),'-B',str(driver_path),'guest-sweep','--root',str(root),'--sweep',str(n),'--fence-fd',str(fd)]
    def probe(minute):return {'command':[str(python),'-B',str(driver_path),'host-probe','--owner-pid','12345'],
        'returncode':0,'observation':{'at_utc':t(minute),'host':host,'uid':uid,'canonical_sha256':'neutral',
          'old_soak_os_state':'unverified','lock_inode':info.st_ino,'lock_path':str(lock),'lock_device':info.st_dev,
          'local_controllers':[{'pid':12345,'start_ticks':'4567','command':runtime['owner_command']}]}}
    write(root/'start_intent.json',{'at_utc':t(3),'activation_sha256':a.sha(root/'activation.json'),
        'host_probe':probe(1),'scope':activation['scope'],'reconciliation_verification':reconciliation_verification})
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
    spec=importlib.util.spec_from_file_location('actual_transport_for_final_test',ROOT/'artifacts/modebench_scale_frozen_stage_transport_v2_20260912.py')
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    for key,value in {'SOURCE':transport_path,'ARTIFACTS':composite,'AUTHORITY_ROOT':root,'ORIGINAL_PLAN':original,
        'RECONCILIATION':reconciliation,'RECONCILIATION_VERIFIER':verifier_path,
        'RECONCILIATION_VERIFIER_SHA':a.sha(verifier_path)}.items():
        monkeypatch.setattr(transport,key,value)
    monkeypatch.setattr(transport,'module',lambda *args:SimpleNamespace(verify_reconciliation=lambda:a.read(reconciliation)))
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
    spec=importlib.util.spec_from_file_location('actual_driver_completion_integration',ROOT/'artifacts/continue_modebench_scale_composite_wash_v2_20260912.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    monkeypatch.setattr(driver,'ORIGINAL_PLAN',c.original)
    monkeypatch.setattr(driver,'RECONCILIATION',c.transport.RECONCILIATION)
    monkeypatch.setattr(driver,'RECONCILIATION_VERIFIER',c.transport.RECONCILIATION_VERIFIER)
    monkeypatch.setattr(driver,'RECONCILIATION_VERIFIER_SHA',c.transport.RECONCILIATION_VERIFIER_SHA)
    monkeypatch.setattr(driver,'RECOVERY_END',c.a.read(c.transport.RECONCILIATION)['recovery_execution']['end_utc'])
    completed=driver.runtime_completion(c.root,c.runtime)
    pins=c.transport.recovery_completion_pins(c.runtime,c.activation)
    assert all(pins[path]==digest for path,digest in completed['receipts_sha256'].items())
    assert pins[str(c.root/'recovery_completion.json')]==c.a.sha(c.root/'recovery_completion.json')
    assert driver.TRANSPORT_SHA==c.a.sha(ROOT/'artifacts/modebench_scale_frozen_stage_transport_v2_20260912.py')


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
    monkeypatch.setattr(a,'DEVELOPMENT_ARRAY_JOB_ID',99);monkeypatch.setattr(a,'DEVELOPMENT_PLAN_SHA',a.sha(path))
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


@pytest.mark.parametrize('field,value',[('command',['wrong']),('returncode',1)])
def test_authority_start_verification_must_reproduce_exact_certificate_route(authority_case,field,value):
    c=authority_case
    change(c.root/'start_intent.json',lambda v:v['reconciliation_verification'].update({field:value}))
    with pytest.raises(ValueError,match='read-only reconciliation'):c.a.validate_authority(c.driver,c.transport)


def test_final_certificate_preserves_failed_recovery_without_general_failure_exception(final_case):
    result=final_case.a.verify()
    assert result['recovery_execution']['state']=='FAILED'
    assert result['recovery_execution']['exit_code']=='1:0'
    assert result['recovery_scheduler_success'] is False
    assert result['recovery_scientific_outputs_complete'] is True
    assert result['recovery_exit_cause']=='unknown'
    v=accounting(final_case.a);v['stdout']=v['stdout'].replace('COMPLETED','FAILED')
    with pytest.raises(ValueError):final_case.a.accounting_rows(v,99,2)


def test_v2_retains_identical_successful_future_array_and_receipt_requirements():
    import ast
    paths=[ROOT/'artifacts/verify_modebench_scale_frozen_execution_20260912.py',
           ROOT/'artifacts/verify_modebench_scale_frozen_execution_v3_20260912.py']
    maps=[{node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body
           if isinstance(node,ast.FunctionDef)} for path in paths]
    for name in ['memory_bytes']:
        assert maps[0][name]==maps[1][name],name



def failed_proof(a,row,index=0):
    return {'schema':a.CELL_SCHEMA,'status':a.CELL_STATUS,'scheduler_success':False,
        'scientific_outputs_complete':True,'exit_cause':'unknown','array_job_id':99,'array_index':index,
        'job_id':row[1],'job_id_raw':row[0],'state':'FAILED','exit_code':'1:0','end_utc':row[5],
        'files_sha256':{}}


def test_only_exact_index_failed_proof_changes_default_accounting_gate(audit):
    a=audit;value=accounting(a);lines=value['stdout'].splitlines();lines[0]=lines[0].replace('COMPLETED|0:0','FAILED|1:0')
    value['stdout']='\n'.join(lines);proof=failed_proof(a,lines[0].split('|'))
    with pytest.raises(ValueError,match='requires COMPLETED'):a.accounting_rows(value,99,2)
    rows=a.accounting_rows(value,99,2,{0:proof})
    assert rows[0][2:4]==['FAILED','1:0'] and rows[1][2:4]==['COMPLETED','0:0']
    with pytest.raises(ValueError):a.accounting_rows(value,99,2,{1:proof})


@pytest.mark.parametrize('field,value',[('array_job_id',100),('array_index',1),('job_id','99_1'),
    ('job_id_raw','other'),('end_utc','2026-09-12T03:59:59'),('exit_cause','shutdown_failure'),
    ('scheduler_success',True),('scientific_outputs_complete',False),('state','COMPLETED'),
    ('exit_code','0:0'),('status','verified_recovery_only'),('schema','wrong')])
def test_failed_cell_proof_cannot_authorize_other_identity_or_hide_failure(audit,field,value):
    a=audit;v=accounting(a);v['stdout']=v['stdout'].replace('COMPLETED|0:0','FAILED|1:0',1)
    proof=failed_proof(a,v['stdout'].splitlines()[0].split('|'));proof[field]=value
    with pytest.raises(ValueError):a.accounting_rows(v,99,2,{0:proof})


@pytest.mark.parametrize('old,new',[('|60G|','|59G|'),('node105','node007'),('cpu=6,','cpu=4,'),
    ('2026-09-12T04:00:00','2026-09-12T06:00:00')])
def test_reconciliation_never_overrides_resources_or_terminal_chronology(audit,old,new):
    a=audit;v=accounting(a);v['stdout']=v['stdout'].replace('COMPLETED|0:0','FAILED|1:0',1).replace(old,new)
    proof=failed_proof(a,v['stdout'].splitlines()[0].split('|'))
    with pytest.raises(ValueError):a.accounting_rows(v,99,2,{0:proof})


def prepare_failed_stage(c):
    accounting_path=c.evidence/(c.stage+'_terminal_accounting.json')
    value=c.a.read(accounting_path);value['stdout']=value['stdout'].replace('COMPLETED|0:0','FAILED|1:0')
    write(accounting_path,value);row=value['stdout'].splitlines()[0].split('|')
    base=c.path.parent/'execution_reconciliations/0';registration=base/'registration.json';terminal=base/'terminal_accounting.json'
    write(terminal,{**value,'schema':'modebench_scale_stage_cell_terminal_accounting_v1',
        'command':['sacct','-j','99_0','-n','-P','--format='+c.a.FIELDS]})
    write(registration,{'explicit_cell':'99_0','terminal_sha256':c.a.sha(terminal)})
    proof=failed_proof(c.a,row);proof['stage']=c.stage
    required=[registration,terminal,c.path,c.path.parent/'submission_intent.json',c.path.parent/'submission_result.json',
        c.outer/'plan.json',c.outer/'submission_intent.json',c.outer/'submission_result.json',
        c.path.parent/'runtime/0.json',c.outer/'runtime/0.json',*c.receipts]
    proof['files_sha256']={str(p):c.a.sha(p) for p in required}
    certificate=base/'reconciliation.json';write(certificate,proof);calls=[]
    def verify(stage,index):
        assert stage==c.stage and index==0;calls.append((stage,index));return c.a.read(certificate)
    return SimpleNamespace(helper=SimpleNamespace(verify_reconciliation=verify),certificate=certificate,
        registration=registration,terminal=terminal,proof=proof,calls=calls,accounting=accounting_path)


def test_reconciled_failed_stage_still_verifies_every_original_task_and_keeps_failed_state(stage_case):
    c=stage_case;r=prepare_failed_stage(c)
    summary,pins=c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate,r.helper)
    assert r.calls==[(c.stage,0)] and summary['scheduler_success'] is False
    assert summary['scientific_outputs_complete'] is True
    assert summary['reconciled_failed_cells'][0]['state']=='FAILED'
    assert summary['reconciled_failed_cells'][0]['exit_code']=='1:0'
    assert summary['reconciled_failed_cells'][0]['exit_cause']=='unknown'
    assert 'terminal_state' not in summary and str(r.registration) in pins and str(r.certificate) in pins
    assert len(c.seed_checks)==(4 if c.stage=='revision_development' else 1)


def test_failed_stage_without_explicit_verifier_remains_rejected(stage_case):
    c=stage_case;prepare_failed_stage(c)
    with pytest.raises(ValueError,match='explicitly registered'):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate)


@pytest.mark.parametrize('missing',['registration','terminal','plan','inner_runtime','outer_runtime'])
def test_per_cell_certificate_must_pin_registration_and_actual_execution(stage_case,missing):
    c=stage_case;r=prepare_failed_stage(c)
    p={'registration':r.registration,'terminal':r.terminal,'plan':c.path,
       'inner_runtime':c.path.parent/'runtime/0.json','outer_runtime':c.outer/'runtime/0.json'}[missing]
    change(r.certificate,lambda v:v['files_sha256'].pop(str(p)))
    with pytest.raises(ValueError,match='omits exact registration'):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate,r.helper)


def test_aggregate_accounting_cannot_change_registered_failed_cell_start_even_if_other_checks_pass(stage_case):
    c=stage_case;r=prepare_failed_stage(c)
    change(r.accounting,lambda v:v.update(stdout=v['stdout'].replace('2026-09-12T02:00:00','2026-09-12T01:30:00')))
    with pytest.raises(ValueError,match='aggregate and registered'):c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate,r.helper)


def test_certificate_does_not_override_original_seed_receipt_validation(stage_case):
    c=stage_case;r=prepare_failed_stage(c)
    def reject(*args):raise ValueError('original receipt still invalid')
    c.launcher.evaluator().validate_seed_receipt=reject
    with pytest.raises(ValueError,match='original receipt still invalid'):
        c.a.validate_stage(c.stage,c.launcher,c.transport,c.candidate,r.helper)


def test_unregistered_actual_cell_prevents_accounting_capture(stage_case,monkeypatch):
    c=stage_case;r=prepare_failed_stage(c);r.accounting.unlink()
    def reject(*args):raise ValueError('actual cell has no explicit registration')
    monkeypatch.setattr(c.a,'requirements',lambda:({'files_sha256':{str(c.a.CELL_RECONCILER):'sealed'}},{}))
    monkeypatch.setattr(c.a,'module',lambda *args:SimpleNamespace(verify_reconciliation=reject))
    raw=c.a.read(r.terminal)
    monkeypatch.setattr(c.a,'now',lambda:'2026-09-12T05:00:00+00:00')
    monkeypatch.setattr(c.a.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=raw['stdout'],stderr=''))
    with pytest.raises(ValueError,match='no explicit registration'):c.a.capture_accounting(c.stage)
    assert not r.accounting.exists()


def test_registered_failed_cell_capture_uses_same_verifier_and_writes_actual_failure(stage_case,monkeypatch):
    c=stage_case;r=prepare_failed_stage(c);r.accounting.unlink()
    monkeypatch.setattr(c.a,'requirements',lambda:({'files_sha256':{str(c.a.CELL_RECONCILER):'sealed'}},{}))
    monkeypatch.setattr(c.a,'module',lambda *args:r.helper)
    raw=c.a.read(r.terminal)
    monkeypatch.setattr(c.a,'now',lambda:'2026-09-12T05:00:00+00:00')
    monkeypatch.setattr(c.a.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=raw['stdout'],stderr=''))
    result=c.a.capture_accounting(c.stage)
    assert result['status']=='captured' and r.calls==[(c.stage,0)]
    assert '|FAILED|1:0|' in c.a.read(r.accounting)['stdout']


@pytest.fixture
def requirements_case(audit,tmp_path,monkeypatch):
    a=audit;root=tmp_path/'requirements_case';root.mkdir()
    components=[]
    for key in ['SOURCE','LEGACY','RECOVERY_AUDITOR','DRIVER','TRANSPORT','LAUNCHER','CELL_RECONCILER']:
        path=root/(key+'.py');path.write_text('sealed '+key);components.append(path);monkeypatch.setattr(a,key,path)
    monkeypatch.setattr(a,'LEGACY_SHA',a.sha(a.LEGACY));monkeypatch.setattr(a,'LAUNCHER_SHA',a.sha(a.LAUNCHER))
    monkeypatch.setattr(a,'AUTHORITY',root/'authority');monkeypatch.setattr(a,'RELEASE',root/'release')
    monkeypatch.setattr(a,'COMPOSITE',root/'composite');monkeypatch.setattr(a,'REQUIREMENTS',root/'requirements.json')
    development=a.COMPOSITE/'revision_development';paths=[]
    for directory in (development,development/'frozen_transport'):
        for name in ['plan.json','plan.sha256.json','submission_intent.json','submission_result.json']:
            path=directory/name;write(path,{'array_job_id':99} if name=='submission_result.json' else {'fixture':name});paths.append(path)
    monkeypatch.setattr(a,'DEVELOPMENT_ARRAY_JOB_ID',99);monkeypatch.setattr(a,'DEVELOPMENT_PLAN_SHA',a.sha(development/'plan.json'))
    value={'schema':'modebench_scale_frozen_execution_requirements_v3','status':'registered_before_final_release',
        'destination':str(a.RELEASE/'execution_provenance.json'),'required_schema':a.SCHEMA,'required_status':'verified',
        'authority_root':str(a.AUTHORITY),'development_array_job_id':99,'development_plan_sha256':a.DEVELOPMENT_PLAN_SHA,
        'failed_cell_policy':'individual_explicit_registration_and_complete_original_grader_reconciliation',
        'files_sha256':{str(path):a.sha(path) for path in components+paths}}
    write(a.REQUIREMENTS,value)
    return SimpleNamespace(a=a,components=components,execution_paths=paths,value=value)


def test_v3_requirements_bind_explicit_policy_and_actual_revision_execution(requirements_case):
    c=requirements_case;value,pins=c.a.requirements()
    assert value==c.value and str(c.a.CELL_RECONCILER) in pins
    assert set(map(str,c.execution_paths))<=set(pins)


@pytest.mark.parametrize('kind,index',[('component',6),('execution',0),('execution',3),('execution',4),('execution',7)])
def test_requirements_cannot_omit_per_cell_verifier_or_actual_array_identity(requirements_case,kind,index):
    c=requirements_case;p=(c.components if kind=='component' else c.execution_paths)[index]
    change(c.a.REQUIREMENTS,lambda value:value['files_sha256'].pop(str(p)))
    with pytest.raises(ValueError,match='omit'):c.a.requirements()


@pytest.mark.parametrize('field,value',[('failed_cell_policy','allow_all_failed_jobs'),('development_array_job_id',100),
    ('development_plan_sha256','0'*64),('schema','modebench_scale_frozen_execution_requirements_v2')])
def test_requirements_reject_broader_failure_policy_or_alternative_array(requirements_case,field,value):
    c=requirements_case;change(c.a.REQUIREMENTS,lambda record:record.update({field:value}))
    with pytest.raises(ValueError,match='wrong final'):c.a.requirements()
