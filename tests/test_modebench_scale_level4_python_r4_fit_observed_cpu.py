"""Exact observed CPU evidence synthetic tests; no native scientific calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_modebench_scale_level5_ready_fits import fixture as prior_fixture
from test_modebench_scale_level4_python_r3_fit import fixture as r3_fixture, write, change
from test_modebench_scale_level4_python_r4_fit import fixture as base_fixture, authorized_fit, repin as review_repin
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def fixture(authorized_fit,monkeypatch):
    old,_=authorized_fit;m=old.m;f=old.f
    path=ROOT/'artifacts/continue_modebench_scale_level4_python_r4_fit_observed_cpu_20260913.py'
    spec=importlib.util.spec_from_file_location('scratch_observed_cpu_fit',path)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    monkeypatch.setattr(a,'engine',m);monkeypatch.setattr(a,'contract',old.a.contract)
    for key in ('AUDIT_ROOT','READONLY_ROOT'):monkeypatch.setattr(a,key,getattr(old.a,key))
    for key in ('SOURCE','TESTS','SCOPE','SCHEMA','RESULT_SCHEMA'):monkeypatch.setattr(m,key,getattr(a,key))
    m.SEALED={**m.SEALED,a.PRIOR:a.PRIOR_SHA,a.PRIOR_TESTS:m.sha(a.PRIOR_TESTS)}
    for key in ('reviewed','entries_and_proofs','collect_inputs','fit_registered','guest','verify_existing'):
        monkeypatch.setattr(m,key,getattr(a,key))
    f.owner['command']=[str(m.PYTHON),'-B',str(m.SOURCE),'fit']
    native=m.RECOVERY/'execution_audit';write(native/'registration.json',{'array_job_id':31261726})
    write(native/'claim.json',{'at_utc':'2026-09-13T04:00:10+00:00','host':a.HOST,'pid':9002,
        'registration_sha256':m.sha(native/'registration.json'),'script_sha256':m.sha(m.AUDITOR),
        'start_ticks':'123456','status':'original_grader_audit_claimed','uid':a.UID})
    for tier in range(4):
        change(native/'tiers'/f'{tier}.json',tier=tier,status='passed',original_grader_agrees=True,new_grader_invocations=6176,
            registration_sha256=m.sha(native/'registration.json'),claim_sha256=m.sha(native/'claim.json'))
    proof=m.read(old.x.certificate)
    for report in proof['grader_audits']:report['sha256']=m.sha(report['path'])
    proof['files_sha256']={p:m.sha(p) for p in proof['files_sha256']}
    proof['files_sha256'].update({str(p):m.sha(p) for p in [native/'claim.json',native/'registration.json',a.PRIOR,a.PRIOR_TESTS]})
    write(old.x.certificate,proof)
    for action,root,code in [('audit',a.AUDIT_ROOT,143),('verify',a.READONLY_ROOT,0)]:
        (root/'stdout.txt').write_text(json.dumps({**{k:proof[k] for k in ('schema','status','array_job_id')},'files':len(proof['files_sha256'])})+'\n')
        change(root/'exit.json',returncode=code,certificate_sha256=m.sha(old.x.certificate))
    x=SimpleNamespace(a=a,m=m,f=f,x=old.x,run=lambda:a.run(f.state,m.sha(m.REVIEW)))
    review_repin(x)
    decision=f.state.parent/'observed-cpu-decision.json';monkeypatch.setattr(a,'OBSERVED_CPU_DECISION',decision)
    mandatory=[old.x.certificate,native/'claim.json',native/'registration.json',
        *(native/'tiers'/f'{tier}.json' for tier in range(4)),m.AUDITOR,m.AUDITOR_REVIEW,a.AUDIT_RECORDER,
        a.contract.AUTHORIZATION,a.contract.OBSERVED_DECISION,
        *(root/name for root in (a.AUDIT_ROOT,a.READONLY_ROOT)
          for name in ('intent.json','process.json','stdout.txt','stderr.txt','exit.json'))]
    write(decision,{'schema':'modebench_scale_level4_python_r4_observed_cpu_audit_decision_v1',
        'array_job_id':31261726,'audit_outer_returncode':143,'audit_outer_exit_cause':'unknown',
        'readonly_returncode':0,'scheduler_success':False,'scientific_data_checks_passed':True,
        'fixed_fit_and_all_six_gates_unchanged':True,'original_authorization_preserved':True,
        'other_jobs_or_future_cpu_failures_authorized':False,'decision_maker':'/root',
        'fit_calls':0,'actual_native_audit_replays':0,'readonly_new_grader_invocations':0,
        'native_grader_checks':24704,'native_tiers':4,'certificate_files':len(proof['files_sha256']),
        'files_sha256':{str(p):m.sha(p) for p in mandatory}})
    monkeypatch.setattr(a,'OBSERVED_CPU_DECISION_SHA',m.sha(decision))
    review=m.read(m.REVIEW);review['files_sha256'].update(proof['files_sha256'])
    review['files_sha256'].update({str(p):m.sha(p) for p in [m.SOURCE,m.TESTS,a.PRIOR,a.PRIOR_TESTS]})
    review['files_sha256'].update(a.observed_cpu_audit_pins())
    review.update(audit_execution_policy=a.AUDIT_EXECUTION_POLICY,
        observed_cpu_audit_decision_sha256=a.OBSERVED_CPU_DECISION_SHA,
        audit_outer_returncode=143,audit_outer_exit_cause='unknown',audit_readonly_returncode=0)
    write(m.REVIEW,review);review_repin(x)
    return x

# The older authorized fixture expects its dependency name "fixture". Bind an
# independent alias so this module's final fixture does not recursively replace it.
@pytest.fixture(name='authorized_fit')
def observed_base_authorized(base_fixture,monkeypatch):
    from test_modebench_scale_level4_python_r4_fit import authorized_fit as original
    return original.__wrapped__(base_fixture,monkeypatch)


def repin(x):
    """Reseal synthetic evidence only after a composing fixture changes its inputs."""
    review_repin(x)
    decision=x.m.read(x.a.OBSERVED_CPU_DECISION)
    decision['files_sha256']={p:x.m.sha(p) for p in decision['files_sha256']}
    decision['certificate_files']=len(x.m.read(x.x.certificate)['files_sha256'])
    write(x.a.OBSERVED_CPU_DECISION,decision)
    x.a.OBSERVED_CPU_DECISION_SHA=x.m.sha(x.a.OBSERVED_CPU_DECISION)
    value=x.m.read(x.m.REVIEW)
    value['files_sha256'].update(x.a.observed_cpu_audit_pins())
    value['observed_cpu_audit_decision_sha256']=x.a.OBSERVED_CPU_DECISION_SHA
    write(x.m.REVIEW,value)
    review_repin(x)


def test_exact_cpu143_preserves_original_one_fit_and_literal_registration(fixture):
    x=fixture;result=x.run()
    assert result['status']=='level4_python_r4_development_gates_passed' and x.f.calls==['python_factors']
    registration=x.m.read(x.f.state/'registration.json')
    assert registration['revisions'][0]['cpu_audit_execution']=={
        'audit_outer_returncode':143,'audit_outer_exit_cause':'unknown','audit_readonly_returncode':0,
        'observed_cpu_audit_decision_sha256':x.a.OBSERVED_CPU_DECISION_SHA}
    assert registration['revisions'][0]['execution']['scheduler_success'] is False
    assert x.a.verify_existing(x.f.state)==result
    with pytest.raises(ValueError):x.run()
    assert x.f.calls==['python_factors']


def test_fixed_negative_fit_remains_negative_without_retry(fixture):
    x=fixture;x.f.failures.add('python_factors');result=x.run()
    assert result['status']=='needs_new_development_revision'
    assert x.a.verify_existing(x.f.state)==result
    with pytest.raises(ValueError):x.run()
    assert x.f.calls==['python_factors']

@pytest.mark.parametrize('action',['audit','verify'])
@pytest.mark.parametrize('name',['intent.json','process.json','stdout.txt','stderr.txt','exit.json'])
def test_every_outer_byte_is_fixed_even_after_review_repin(fixture,action,name):
    x=fixture;root=x.a.AUDIT_ROOT if action=='audit' else x.a.READONLY_ROOT
    path=root/name
    if name.endswith('.json'):change(path,scratch_changed=True)
    else:path.write_bytes(path.read_bytes()+b' ')
    review_repin(x)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls and not x.f.state.exists()

@pytest.mark.parametrize('name',['claim.json','registration.json','tiers/0.json','tiers/1.json','tiers/2.json','tiers/3.json'])
def test_every_native_report_and_claim_is_fixed(fixture,name):
    x=fixture;path=x.m.RECOVERY/'execution_audit'/name
    change(path,scratch_change=True);review_repin(x)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls and not x.f.state.exists()

@pytest.mark.parametrize('key,value',[
    ('array_job_id',31261727),('audit_outer_returncode',0),('audit_outer_returncode',True),
    ('readonly_returncode',143),('audit_outer_exit_cause','assumed_teardown'),
    ('scheduler_success',True),('scientific_data_checks_passed',False),
    ('native_grader_checks',24703),('original_authorization_preserved',False),
    ('other_jobs_or_future_cpu_failures_authorized',True)])
def test_rehashed_decision_cannot_broaden_or_normalize_observation(fixture,monkeypatch,key,value):
    x=fixture;change(x.a.OBSERVED_CPU_DECISION,**{key:value})
    monkeypatch.setattr(x.a,'OBSERVED_CPU_DECISION_SHA',x.m.sha(x.a.OBSERVED_CPU_DECISION));review_repin(x)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls and not x.f.state.exists()

@pytest.mark.parametrize('field,value',[
    ('observed_cpu_audit_decision_sha256','0'*64),('audit_outer_returncode',0),
    ('audit_outer_returncode',True),('audit_readonly_returncode',143),('audit_outer_exit_cause','successful')])
def test_final_review_cannot_omit_or_normalize_cpu_failure(fixture,field,value):
    x=fixture;change(x.m.REVIEW,**{field:value})
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls and not x.f.state.exists()

@pytest.mark.parametrize('kind',['decision','outer','report','certificate_input','preserved'])
def test_final_review_must_pin_complete_observed_and_native_closure(fixture,kind):
    x=fixture;m=x.m;v=m.read(m.REVIEW)
    if kind=='decision':path=x.a.OBSERVED_CPU_DECISION
    elif kind=='outer':path=x.a.AUDIT_ROOT/'intent.json'
    elif kind=='report':path=m.RECOVERY/'execution_audit/tiers/0.json'
    elif kind=='certificate_input':path=next(Path(p) for p in m.read(x.x.certificate)['files_sha256'] if 'certificate-only-batch' in p)
    else:path=next(iter(m.read(x.a.CPU_PRESERVATION)['files_sha256']))
    v['files_sha256'].pop(str(path));write(m.REVIEW,v)
    with pytest.raises(ValueError):x.run()
    assert not x.f.calls and not x.f.state.exists()

@pytest.mark.parametrize('action',['audit','verify'])
@pytest.mark.parametrize('code',[0,1,143,-15,False])
def test_rehashed_other_cpu_outcomes_are_rejected(fixture,monkeypatch,action,code):
    if (action,code)==('audit',143) or (action,code)==('verify',0) and type(code) is int:return
    x=fixture;root=x.a.AUDIT_ROOT if action=='audit' else x.a.READONLY_ROOT
    change(root/'exit.json',returncode=code);review_repin(x)
    d=x.m.read(x.a.OBSERVED_CPU_DECISION)
    d['files_sha256']={p:x.m.sha(p) for p in d['files_sha256']};write(x.a.OBSERVED_CPU_DECISION,d)
    monkeypatch.setattr(x.a,'OBSERVED_CPU_DECISION_SHA',x.m.sha(x.a.OBSERVED_CPU_DECISION));review_repin(x)
    with pytest.raises(ValueError):x.a.observed_cpu_audit_pins()
    assert not x.f.calls and not x.f.state.exists()


def test_original_native_fit_guest_verification_and_fence_code_is_unchanged():
    old=ROOT/'artifacts/continue_modebench_scale_level4_python_r4_fit_20260913.py'
    new=ROOT/'artifacts/continue_modebench_scale_level4_python_r4_fit_observed_cpu_20260913.py'
    def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
    before,after=functions(old),functions(new)
    for name in ['fit_registered','guest','collect_inputs','verify_existing','outer_launch_guard','run','live_host_guard','live_facade']:
        assert before[name]==after[name],name
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='fe06460188022666eb2f038986abebf2193e18422e10a6aa9a1af338f5585ed7'
    assert hashlib.sha256((ROOT/'tests/test_modebench_scale_level4_python_r4_fit.py').read_bytes()).hexdigest()=='439238e405ad9e9b4e8ef95a840d217e57ea7591e8d01c8653b664f72d4301cb'


def test_claim_fixture_matches_actual_unchanged_native_record_shape(fixture):
    actual=ROOT/'var/artifacts/modebench_scale_level4_python_r4_20260913/development/execution_audit/claim.json'
    saved=json.loads(actual.read_text());scratch=fixture.m.read(fixture.m.RECOVERY/'execution_audit/claim.json')
    assert set(saved)==set(scratch)
    assert saved['status']==scratch['status']=='original_grader_audit_claimed'
    assert 'source_sha256' not in saved and 'script_sha256' in saved

@pytest.mark.parametrize('field,value',[('script_sha256','0'*64),('script_sha256',None),
                                        ('status','claimed'),('status',None)])
def test_repinned_native_claim_must_match_original_key_and_status(fixture,monkeypatch,field,value):
    x=fixture;change(x.m.RECOVERY/'execution_audit/claim.json',**{field:value})
    decision=x.m.read(x.a.OBSERVED_CPU_DECISION)
    decision['files_sha256']={p:x.m.sha(p) for p in decision['files_sha256']}
    write(x.a.OBSERVED_CPU_DECISION,decision)
    monkeypatch.setattr(x.a,'OBSERVED_CPU_DECISION_SHA',x.m.sha(x.a.OBSERVED_CPU_DECISION))
    with pytest.raises(ValueError,match='native certificate/registration/claim'):x.a.observed_cpu_audit_pins()
    assert not x.f.calls and not x.f.state.exists()
