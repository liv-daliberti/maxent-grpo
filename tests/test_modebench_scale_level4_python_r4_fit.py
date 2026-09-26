"""Synthetic saved r4 proof and native-fit stub only; no scientific calls."""
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level5_ready_fits import fixture as prior_fixture
from test_modebench_scale_level4_python_r3_fit import fixture as r3_fixture, write, change
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def fixture(r3_fixture,monkeypatch):
    x=r3_fixture;m=x.m;f=x.f
    p=ROOT/'artifacts/continue_modebench_scale_level4_python_r4_fit_20260913.py'
    spec=importlib.util.spec_from_file_location('scratch_r4_fit',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    monkeypatch.setattr(a,'engine',m)
    for key in ('SOURCE','TESTS','SCOPE','SCHEMA','RESULT_SCHEMA'):monkeypatch.setattr(m,key,getattr(a,key))
    original_first=m.first;monkeypatch.setattr(m,'first',a.live_facade(original_first));monkeypatch.setattr(m.first,'host_guard',original_first.host_guard)
    monkeypatch.setattr(a.contract,'negative_predecessor_pins',lambda **kwargs:{})
    monkeypatch.setattr(a.contract,'outer_command_guard',lambda *args,**kwargs:None)
    monkeypatch.setattr(a,'AUDIT_ROOT',f.state.parent/'actual_audit')
    monkeypatch.setattr(a,'READONLY_ROOT',f.state.parent/'actual_readonly')
    for name in ('reviewed','entries_and_proofs','collect_inputs','fit_registered','guest','verify_existing'):
        monkeypatch.setattr(m,name,getattr(a,name))
    f.owner['command']=[str(m.PYTHON),'-B',str(m.SOURCE),'fit']
    for path in (x.prep,x.cell):
        value=m.read(path);value=json.loads(json.dumps(value).replace('python_r3','python_r4').replace('python_factors_r3','python_factors_r4'))
        write(path,value)
    tasks=m.read(x.tasks)
    for task in tasks:task['seeds']=[7204000,7204001,7204002,7204003]
    write(x.tasks,tasks);change(x.cell,tasks=tasks)
    change(m.AUDITOR_REVIEW,schema='modebench_scale_level4_python_r4_execution_independent_review_v1')
    proof=m.read(x.certificate)
    proof.update(schema=a.contract.AUDIT_SCHEMA,status=a.contract.AUDIT_STATUS,array_job_id=70000001,terminal_policy=a.POLICY,
        scheduler_success=True,exit_cause='successful_execution',audit_completed_at_utc='2026-09-13T04:00:30+00:00')
    proof['execution'].update(array_job_id=70000001,job_id_raw='70000001',job_id='70000001_0',state='COMPLETED',exit_code='0:0',scheduler_success=True,exit_cause='successful_execution')
    certificate_batch=f.state.parent/'certificate-only-batch.json'
    write(certificate_batch,{'scratch_saved_batch':True})
    proof['files_sha256']={p:m.sha(p) for p in proof['files_sha256']}
    proof['files_sha256'][str(certificate_batch)]=m.sha(certificate_batch)
    write(x.certificate,proof)
    value=m.read(m.REVIEW);value.update(schema='modebench_scale_level4_python_r4_fit_independent_review_v1',array_job_id=70000001,job_id_raw='70000001',terminal_policy=a.POLICY,
        audit_execution_policy=a.AUDIT_EXECUTION_POLICY,cpu_fit_host=a.HOST,cpu_fit_uid=a.UID)
    value['files_sha256']={p:m.sha(p) for p in value['files_sha256']}
    value['files_sha256'].update(proof['files_sha256'])
    value['files_sha256'].update({str(p):m.sha(p) for p in [m.SOURCE,m.TESTS,a.AUDIT_RECORDER,a.AUDIT_RECORDER_TESTS]})
    value.update({field:m.sha(path) for field,path in m.PROOF_FIELDS.items()});write(m.REVIEW,value)
    for index,(action,root) in enumerate([('audit',a.AUDIT_ROOT),('verify',a.READONLY_ROOT)]):
        identity={'schema':'modebench_scale_python_r4_soak_outer_execution_v1','command':a.audit_command(action),
            'action':action,'actual_host':a.HOST,'uid':a.UID,'pid':8000+index,
            'started_at_utc':f'2026-09-13T04:0{index*2}:00+00:00'}
        write(root/'intent.json',{**identity,'script_sha256':m.sha(a.AUDIT_RECORDER),'auditor_review_sha256':m.sha(m.AUDITOR_REVIEW),
            'terminal_sha256':m.sha(x.terminal)})
        write(root/'process.json',{'at_utc':f'2026-09-13T04:0{index*2}:01+00:00','child_pid':9000+index,
            **{k:identity[k] for k in ('actual_host','uid','pid')}})
        (root/'stdout.txt').write_text(json.dumps({**{k:proof[k] for k in ('schema','status','array_job_id')},'files':len(proof['files_sha256'])})+'\n')
        (root/'stderr.txt').write_text('')
        write(root/'exit.json',{**identity,'returncode':0,'postcheck_error':None,'certificate_present':True,
            'certificate_sha256':m.sha(x.certificate),'new_grader_invocations':0 if action=='verify' else 24704,
            'intent_sha256':m.sha(root/'intent.json'),'process_sha256':m.sha(root/'process.json'),
            'stdout_sha256':m.sha(root/'stdout.txt'),'stderr_sha256':m.sha(root/'stderr.txt'),
            'finished_at_utc':f'2026-09-13T04:0{index*2+1}:00+00:00'})
    value=m.read(m.REVIEW)
    value['files_sha256'].update({str(p):m.sha(p) for root in (a.AUDIT_ROOT,a.READONLY_ROOT) for p in root.iterdir()})
    value.update(audit_outer_exit_sha256=m.sha(a.AUDIT_ROOT/'exit.json'),audit_readonly_exit_sha256=m.sha(a.READONLY_ROOT/'exit.json'))
    write(m.REVIEW,value)
    return SimpleNamespace(a=a,m=m,f=f,x=x,run=lambda:a.run(f.state,m.sha(m.REVIEW)))

def repin(x):
    m=x.m
    for root in (x.a.AUDIT_ROOT,x.a.READONLY_ROOT):
        change(root/'exit.json',intent_sha256=m.sha(root/'intent.json'),process_sha256=m.sha(root/'process.json'),
            stdout_sha256=m.sha(root/'stdout.txt'),stderr_sha256=m.sha(root/'stderr.txt'))
    value=m.read(m.REVIEW);value['files_sha256']={p:m.sha(p) for p in value['files_sha256']}
    value.update({field:m.sha(path) for field,path in m.PROOF_FIELDS.items()})
    value.update(audit_outer_exit_sha256=m.sha(x.a.AUDIT_ROOT/'exit.json'),audit_readonly_exit_sha256=m.sha(x.a.READONLY_ROOT/'exit.json'))
    write(m.REVIEW,value)

def test_one_original_r4_fit_under_inherited_fence_and_readonly_verification(fixture):
    x=fixture;value=x.run()
    assert value['status']=='level4_python_r4_development_gates_passed' and x.f.calls==['python_factors']
    registration=x.m.read(x.f.state/'registration.json')
    assert registration['schema']==x.a.SCHEMA and registration['host']=='soak.cs.princeton.edu'
    assert registration['revisions'][0]['execution']['scheduler_success'] is True
    assert x.a.verify_existing(x.f.state)==value and x.f.calls==['python_factors']
    with pytest.raises(ValueError):x.run()
    assert x.f.calls==['python_factors']

def test_negative_original_fit_is_saved_without_retry(fixture):
    x=fixture;x.f.failures.add('python_factors');value=x.run()
    assert value['status']=='needs_new_development_revision' and value['failed_domains']==['python_factors']
    assert x.a.verify_existing(x.f.state)==value and x.f.calls==['python_factors']

def test_native_exception_preserves_one_claim_and_failure(fixture):
    x=fixture;x.f.exceptions.add('python_factors')
    with pytest.raises(ValueError,match='scratch fitter exception'):x.run()
    assert (x.f.state/'fits/python_factors/claim.json').is_file() and (x.f.state/'action/failure.json').is_file()
    with pytest.raises(ValueError):x.run()
    assert x.f.calls==['python_factors']

@pytest.mark.parametrize('action',['audit','verify'])
@pytest.mark.parametrize('code',[1,143,-15,False])
def test_no_outer_failure_or_boolean_zero_exception_is_accepted(fixture,action,code):
    x=fixture;root=x.a.AUDIT_ROOT if action=='audit' else x.a.READONLY_ROOT
    change(root/'exit.json',returncode=code);repin(x)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls and not x.f.state.exists()

@pytest.mark.parametrize('kind',['certificate','intent_identity','intent_source','process','readonly_grader','chronology','output','postcheck'])
def test_rehashed_outer_evidence_still_needs_exact_semantic_linkage(fixture,kind):
    x=fixture;root=x.a.AUDIT_ROOT
    if kind=='certificate':change(root/'exit.json',certificate_sha256='0'*64)
    elif kind=='intent_identity':change(root/'intent.json',pid=90000)
    elif kind=='intent_source':change(root/'intent.json',script_sha256='0'*64)
    elif kind=='process':change(root/'process.json',pid=90000)
    elif kind=='readonly_grader':change(x.a.READONLY_ROOT/'exit.json',new_grader_invocations=1)
    elif kind=='chronology':change(x.a.READONLY_ROOT/'intent.json',started_at_utc='2026-09-13T03:59:00+00:00');change(x.a.READONLY_ROOT/'exit.json',started_at_utc='2026-09-13T03:59:00+00:00')
    elif kind=='output':(root/'stdout.txt').write_text('{}\n')
    else:change(root/'exit.json',postcheck_error='input drift')
    repin(x)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls

@pytest.mark.parametrize('kind',['draft','missing_pin','future_cycle','wrong_job','failed_scoring'])
def test_final_review_and_fresh_successful_scope_are_mandatory(fixture,kind):
    x=fixture;v=x.m.read(x.m.REVIEW)
    if kind=='draft':v['status']='draft'
    elif kind=='missing_pin':v['files_sha256'].pop(str(x.x.certificate))
    elif kind=='future_cycle':v['files_sha256'][str(x.f.state/'future.json')]='0'*64
    elif kind=='wrong_job':v['array_job_id']=31260226
    else:
        proof=x.m.read(x.x.certificate);proof['execution']['state']='FAILED';write(x.x.certificate,proof)
        v['execution_certificate_sha256']=x.m.sha(x.x.certificate);v['files_sha256'][str(x.x.certificate)]=x.m.sha(x.x.certificate)
    write(x.m.REVIEW,v)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls

def test_scientific_fit_body_only_changes_version_labels():
    old=ROOT/'artifacts/continue_modebench_scale_level4_python_r3_fit_20260912.py'
    new=ROOT/'artifacts/continue_modebench_scale_level4_python_r4_fit_20260913.py'
    def body(path):
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fit_registered')
        for n in ast.walk(node):
            if isinstance(n,ast.Constant) and isinstance(n.value,str):n.value=n.value.replace('python_r3','python_r4').replace('Python r3','Python r4')
        return ast.dump(node,include_attributes=False)
    assert body(old)==body(new)


@pytest.mark.parametrize('kind',['receipt','batch','report'])
@pytest.mark.parametrize('mutation',['omitted','rehashed'])
def test_final_fit_review_must_cover_every_actual_certificate_input(fixture,kind,mutation):
    x=fixture;m=x.m;proof=m.read(x.x.certificate)
    path=(x.x.receipts[0] if kind=='receipt' else
          next(Path(p) for p in proof['files_sha256'] if
               ('certificate-only-batch' in p if kind=='batch' else '/execution_audit/tiers/' in p)))
    if mutation=='rehashed':
        data=m.read(path);data['scratch_changed_after_review']=True;write(path,data)
        proof['files_sha256'][str(path)]=m.sha(path)
        if kind=='receipt':proof['summaries'][0]['receipt_sha256']=m.sha(path)
        if kind=='report':proof['grader_audits'][0]['sha256']=m.sha(path)
        write(x.x.certificate,proof)
        for root in (x.a.AUDIT_ROOT,x.a.READONLY_ROOT):
            change(root/'exit.json',certificate_sha256=m.sha(x.x.certificate))
        repin(x)
    value=m.read(m.REVIEW);value['files_sha256'].pop(str(path));write(m.REVIEW,value)
    with pytest.raises(ValueError,match='pre-pin complete actual audit certificate closure'):
        x.run()
    assert not x.f.calls and not x.f.state.exists()


@pytest.fixture
def authorized_fit(fixture,monkeypatch):
    x=fixture;m=x.m;a=x.a;job=a.contract.OBSERVED_JOB
    authorization=x.f.state.parent/'synthetic-user-permission.json';write(authorization,{'scratch_explicit_permission':True})
    decision=x.f.state.parent/'synthetic-decision.json';write(decision,{'scratch_literal_failed_scheduler':True})
    pins={str(authorization):m.sha(authorization),str(decision):m.sha(decision),str(x.x.terminal):m.sha(x.x.terminal)}
    monkeypatch.setattr(a.contract,'observed_terminal_pins',lambda **kwargs:dict(pins))
    monkeypatch.setattr(a.contract,'DEVELOPMENT',m.RECOVERY)
    classification=a.contract.execution_classification(job)
    proof=m.read(x.x.certificate);proof.update(array_job_id=job,**classification)
    proof['execution'].update(array_job_id=job,job_id_raw=str(job),job_id=str(job)+'_0',**classification)
    proof['files_sha256'].update(pins);write(x.x.certificate,proof)
    review=m.read(m.REVIEW);review.update(array_job_id=job,job_id_raw=str(job),**classification)
    review['files_sha256'].update(pins);write(m.REVIEW,review)
    for root in (a.AUDIT_ROOT,a.READONLY_ROOT):
        (root/'stdout.txt').write_text(json.dumps({**{k:proof[k] for k in ('schema','status','array_job_id')},'files':len(proof['files_sha256'])})+'\n')
        change(root/'exit.json',certificate_sha256=m.sha(x.x.certificate))
    repin(x)
    return x,authorization

def test_user_authorized_failed_scheduler_fit_keeps_original_six_gates_and_one_claim(authorized_fit):
    x,_=authorized_fit;result=x.run()
    assert result['status']=='level4_python_r4_development_gates_passed' and x.f.calls==['python_factors']
    registration=x.m.read(x.f.state/'registration.json');execution=registration['revisions'][0]['execution']
    assert execution['scheduler_success'] is False and execution['state']=='FAILED' and execution['exit_code']=='143:0'
    assert execution['exit_cause']=='unknown' and execution['evaluator_returncode']==0
    assert x.a.verify_existing(x.f.state)==result and x.f.calls==['python_factors']

def test_authorized_negative_fit_stays_negative_and_cannot_retry(authorized_fit):
    x,_=authorized_fit;x.f.failures.add('python_factors');result=x.run()
    assert result['status']=='needs_new_development_revision' and result['failed_domains']==['python_factors']
    with pytest.raises(ValueError):x.run()
    assert x.f.calls==['python_factors']

@pytest.mark.parametrize('kind',['permission_pin','permission_sha','decision_sha','scheduler','cause'])
def test_authorized_final_fit_review_cannot_drop_or_normalize_permission_and_failure(authorized_fit,kind):
    x,authorization=authorized_fit;review=x.m.read(x.m.REVIEW)
    if kind=='permission_pin':review['files_sha256'].pop(str(authorization))
    elif kind=='permission_sha':review['user_authorization_sha256']='0'*64
    elif kind=='decision_sha':review['observed_terminal_decision_sha256']='0'*64
    elif kind=='scheduler':review['scheduler_success']=True
    else:review['exit_cause']='assumed_teardown'
    write(x.m.REVIEW,review)
    with pytest.raises(ValueError,match='exact user-authorized failed scheduler evidence'):x.run()
    assert not x.f.calls and not x.f.state.exists()

@pytest.mark.parametrize('action',['audit','verify'])
def test_authorized_scheduler143_never_waives_future_cpu143(authorized_fit,action):
    x,_=authorized_fit;root=x.a.AUDIT_ROOT if action=='audit' else x.a.READONLY_ROOT
    change(root/'exit.json',returncode=143);repin(x)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls and not x.f.state.exists()
