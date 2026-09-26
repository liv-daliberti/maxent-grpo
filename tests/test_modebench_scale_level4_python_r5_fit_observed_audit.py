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
    path=ROOT/'artifacts/continue_modebench_scale_level4_python_r5_fit_observed_audit_20260913.py'
    spec=importlib.util.spec_from_file_location('scratch_observed_cpu_fit',path)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    monkeypatch.setattr(a,'engine',m)
    # The base fixture stamped its evidence from revision 4's contract (job 31261726,
    # exit cause 'unknown'). Keep revision 5's own contract bound -- binding r4's here is
    # the very layer leak the adapter closes -- and restamp the synthetic evidence with
    # revision 5's job and classification so the fixture exercises r5 policy.
    monkeypatch.setattr(a.contract,'DEVELOPMENT',m.RECOVERY)
    monkeypatch.setattr(a.contract,'negative_predecessor_pins',lambda **kwargs:{})
    monkeypatch.setattr(a.contract,'outer_command_guard',lambda *args,**kwargs:None)
    terminal=old.a.contract.observed_terminal_pins(guest=True)
    monkeypatch.setattr(a.contract,'observed_terminal_pins',lambda **kwargs:dict(terminal))
    job=a.contract.OBSERVED_JOB;classification=a.contract.execution_classification(job)
    for target,extra in ((old.x.certificate,True),(m.REVIEW,False)):
        value=m.read(target);value.update(array_job_id=job,job_id_raw=str(job),terminal_policy=a.POLICY,**classification)
        value['schema']=(a.contract.AUDIT_SCHEMA if extra else 'modebench_scale_level4_python_r5_fit_independent_review_v1')
        if extra:value['status']=a.contract.AUDIT_STATUS
        if extra:value['execution'].update(array_job_id=job,job_id_raw=str(job),job_id=str(job)+'_0',**classification)
        value['files_sha256'].update(terminal);write(target,value)
    for key in ('AUDIT_ROOT','READONLY_ROOT'):monkeypatch.setattr(a,key,getattr(old.a,key))
    for key in ('SOURCE','TESTS','SCOPE','SCHEMA','RESULT_SCHEMA'):monkeypatch.setattr(m,key,getattr(a,key))
    m.SEALED={**m.SEALED,a.PRIOR:a.PRIOR_SHA,a.PRIOR_TESTS:m.sha(a.PRIOR_TESTS)}
    for key in ('reviewed','entries_and_proofs','collect_inputs','fit_registered','guest','verify_existing'):
        monkeypatch.setattr(m,key,getattr(a,key))
    f.owner['command']=[str(m.PYTHON),'-B',str(m.SOURCE),'fit']
    native=m.RECOVERY/'execution_audit';write(native/'registration.json',{'array_job_id':31267007})
    write(native/'claim.json',{'at_utc':'2026-09-13T04:00:10+00:00','host':a.HOST,'pid':9002,
        'registration_sha256':m.sha(native/'registration.json'),'script_sha256':m.sha(m.AUDITOR),
        'start_ticks':'123456','status':'original_grader_audit_claimed','uid':a.UID})
    for tier in range(4):
        change(native/'tiers'/f'{tier}.json',tier=tier,status='passed',original_grader_agrees=True,new_grader_invocations=6176,
            registration_sha256=m.sha(native/'registration.json'),claim_sha256=m.sha(native/'claim.json'))
    protocol_path=m.REVISION_ROOT/'protocol.json';protocol=m.read(protocol_path)
    protocol['draw_labels']={'dev':[7205000,7205001,7205002,7205003],
        'eval':[7205500,7205501,7205502,7205503]}
    write(protocol_path,protocol)
    tasks_path=m.PREPARATION/'development_tasks.json';tasks=m.read(tasks_path)
    for task in tasks:task['seeds']=list(protocol['draw_labels']['dev'])
    write(tasks_path,tasks)
    # The receipts record the seeds the scoring worker actually drew. The adapter now binds
    # those to the planned labels, so the fixture must carry them too -- otherwise the
    # plan-to-result half of the check is exercised by nothing.
    for task in tasks:
        receipt_path=Path(task['output']);receipt=m.read(receipt_path)
        receipt.setdefault('identity',{})['seeds']=list(protocol['draw_labels']['dev'])
        receipt.setdefault('sampling',{})['seeds']=list(protocol['draw_labels']['dev'])
        write(receipt_path,receipt)
    cell_value=m.read(m.PREPARATION/'development_cell.json');cell_value['tasks']=tasks
    write(m.PREPARATION/'development_cell.json',cell_value)
    prep=m.read(m.PREPARATION/'certificate.json')
    prep.update(schema='modebench_scale_level4_python_r5_registration_v1',
        status='ready_for_fresh_python_r5_development')
    write(m.PREPARATION/'certificate.json',prep)
    cell=m.read(m.PREPARATION/'development_cell.json');cell['id']='level4_python_factors_r5_dev'
    write(m.PREPARATION/'development_cell.json',cell)
    proof_path=m.PROOF_FIELDS['execution_certificate_sha256']
    if Path(proof_path).exists():
        value=m.read(proof_path);value['files_sha256'][str(protocol_path)]=m.sha(protocol_path)
        write(proof_path,value)
    auditor_review=m.read(m.AUDITOR_REVIEW)
    auditor_review['schema']='modebench_scale_level4_python_r5_execution_independent_review_v1'
    write(m.AUDITOR_REVIEW,auditor_review)
    proof=m.read(old.x.certificate)
    for summary in proof['summaries']:
        summary['receipt_sha256']=m.sha(summary['receipt'])
    for report in proof['grader_audits']:report['sha256']=m.sha(report['path'])
    proof['files_sha256']={p:m.sha(p) for p in proof['files_sha256']}
    proof['files_sha256'].update({str(p):m.sha(p) for p in [native/'claim.json',native/'registration.json',a.PRIOR,a.PRIOR_TESTS]})
    write(old.x.certificate,proof)
    # The base fixture stamped its outer records for revision 4 on soak. Restamp them for
    # revision 5's schema, command and audit host, or the fixture would prove only that r4
    # evidence still passes.
    for action,root,code in [('audit',a.AUDIT_ROOT,143),('verify',a.READONLY_ROOT,0)]:
        (root/'stdout.txt').write_text(json.dumps({**{k:proof[k] for k in ('schema','status','array_job_id')},'files':len(proof['files_sha256'])})+'\n')
        identity={'schema':'modebench_scale_python_r5_soak_outer_execution_v1',
            'command':a.audit_command(action),'actual_host':a.HOST,'uid':a.UID}
        change(root/'intent.json',script_sha256=m.sha(a.AUDIT_RECORDER),
            auditor_review_sha256=m.sha(m.AUDITOR_REVIEW),**identity)
        change(root/'process.json',**{k:identity[k] for k in ('actual_host','uid')})
        change(root/'exit.json',returncode=code,certificate_sha256=m.sha(old.x.certificate),**identity)
    x=SimpleNamespace(a=a,m=m,f=f,x=old.x,run=lambda:a.run(f.state,m.sha(m.REVIEW)))
    review_repin(x)
    decision=f.state.parent/'observed-cpu-decision.json';monkeypatch.setattr(a,'OBSERVED_CPU_DECISION',decision)
    mandatory=[old.x.certificate,native/'claim.json',native/'registration.json',
        *(native/'tiers'/f'{tier}.json' for tier in range(4)),m.AUDITOR,m.AUDITOR_REVIEW,a.AUDIT_RECORDER,
        a.contract.AUTHORIZATION,a.contract.OBSERVED_DECISION,
        *(root/name for root in (a.AUDIT_ROOT,a.READONLY_ROOT)
          for name in ('intent.json','process.json','stdout.txt','stderr.txt','exit.json'))]
    write(decision,{'schema':'modebench_scale_level4_python_r5_observed_audit_decision_v1',
        'array_job_id':31267007,'audit_outer_returncode':143,'audit_outer_exit_cause':'post_completion_teardown',
        'readonly_returncode':0,'scheduler_success':False,'scientific_data_checks_passed':True,
        'fixed_fit_and_all_six_gates_unchanged':True,'original_authorization_preserved':True,
        'other_jobs_or_future_failures_authorized':False,'decision_owner':'/root',
        'fit_calls':0,'actual_native_audit_replays':0,'readonly_new_grader_invocations':0,
        'native_grader_checks':24704,'native_tiers':4,'certificate_files':len(proof['files_sha256']),
        'files_sha256':{str(p):m.sha(p) for p in mandatory}})
    monkeypatch.setattr(a,'OBSERVED_CPU_DECISION_SHA',m.sha(decision))
    review=m.read(m.REVIEW);review['files_sha256'].update(proof['files_sha256'])
    review['files_sha256'].update({str(p):m.sha(p) for p in
        [m.SOURCE,m.TESTS,a.PRIOR,a.PRIOR_TESTS,a.AUDIT_RECORDER,a.AUDIT_RECORDER_TESTS,
         a.AUDIT_RECORDER_POSTRUN_TESTS]})
    review['files_sha256'].update(a.observed_cpu_audit_pins())
    review.update(audit_execution_policy=a.AUDIT_EXECUTION_POLICY,
        observed_cpu_audit_decision_sha256=a.OBSERVED_CPU_DECISION_SHA,
        audit_outer_returncode=143,audit_outer_exit_cause='post_completion_teardown',audit_readonly_returncode=0,
        cpu_fit_host=a.HOST,cpu_fit_uid=a.UID)
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
    assert result['status']=='level4_python_r5_development_gates_passed' and x.f.calls==['python_factors']
    registration=x.m.read(x.f.state/'registration.json')
    assert registration['revisions'][0]['cpu_audit_execution']=={
        'audit_outer_returncode':143,'audit_outer_exit_cause':'post_completion_teardown','audit_readonly_returncode':0,
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
    old=ROOT/'artifacts/continue_modebench_scale_level4_python_r4_fit_observed_cpu_20260913.py'
    new=ROOT/'artifacts/continue_modebench_scale_level4_python_r5_fit_observed_audit_20260913.py'
    def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
    before,after=functions(old),functions(new)
    names=['fit_registered','guest','collect_inputs','verify_existing','outer_launch_guard','run','live_host_guard','live_facade']
    # Two complementary checks. First: identical once the revision token is advanced, so a
    # rename is allowed but nothing else is. Second: identical after blanking every string
    # constant, so no rename can smuggle in a structural change. Neither alone is enough.
    renames=[('r4','r5'),
        ("actual same-user soak CPU fit required","actual same-user fit on the live audit host required")]
    for name in names:
        moved=before[name]
        for was,now in renames:moved=moved.replace(was,now)
        assert moved==after[name],name
    def skeleton(p):
        tree=ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Constant) and isinstance(n.value,str):n.value='<str>'
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    lean,rich=skeleton(old),skeleton(new)
    for name in names:
        assert lean[name]==rich[name],name
    # fit_registered carries the revision token in exactly two literals -- the single-fit
    # guard message and the pass status. Everything else in it, including the one-revision
    # guard and the zero-new-grader accounting, must be byte-identical.
    assert before['fit_registered'].replace('Python r4 fit','Python r5 fit').replace(
        'level4_python_r4_development_gates_passed','level4_python_r5_development_gates_passed')==after['fit_registered']
    # The predecessor here is the observed-CPU adapter; 'fe064601...' is the r4 *fit*
    # adapter and pinned the wrong file. Bind the literal to the adapter's own PRIOR_SHA
    # so the two can never drift apart again.
    spec=importlib.util.spec_from_file_location('_r5_prior_sha_probe',new)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='168de7a208b7cfa415a065555ffd9b2f365875d70a9bea307316511b1ce6b8dc'==module.PRIOR_SHA
    assert module.engine.SEALED[module.PRIOR_TESTS]==hashlib.sha256(module.PRIOR_TESTS.read_bytes()).hexdigest()
    assert str(module.contract.OBSERVED_JOB) in module.POLICY and module.contract.OBSERVED_JOB==31267007
    assert hashlib.sha256((ROOT/'tests/test_modebench_scale_level4_python_r4_fit.py').read_bytes()).hexdigest()=='439238e405ad9e9b4e8ef95a840d217e57ea7591e8d01c8653b664f72d4301cb'


def test_claim_fixture_matches_actual_unchanged_native_record_shape(fixture):
    actual=ROOT/'var/artifacts/modebench_scale_level4_python_r5_20260913/development/execution_audit/claim.json'
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


# --- Real-record bindings -----------------------------------------------------
# Every synthetic test above stamps its evidence from the same constants it then
# checks, so a constant advanced everywhere except one place passes the whole suite.
# Four defects in this adapter had exactly that shape: the r4 exit cause 'unknown',
# the soak host, the r4 outer-record schema, and AUDIT_ROOT still naming the REFUSED
# first attempt instead of the v2 directories the audit actually wrote. These tests
# read the sealed records off disk instead.

def real_adapter():
    path=ROOT/'artifacts/continue_modebench_scale_level4_python_r5_fit_observed_audit_20260913.py'
    spec=importlib.util.spec_from_file_location('_r5_fit_real_record_binding',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def sha_of(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_real_sealed_decision_binds_every_adapter_constant_it_reconciles():
    a=real_adapter();decision=json.loads(Path(a.OBSERVED_CPU_DECISION).read_text())
    assert sha_of(a.OBSERVED_CPU_DECISION)==a.OBSERVED_CPU_DECISION_SHA
    assert decision['schema']=='modebench_scale_level4_python_r5_observed_audit_decision_v1'
    assert decision['array_job_id']==a.OBSERVED_JOB==31267007
    assert decision['audit_outer_returncode']==143 and decision['readonly_returncode']==0
    assert decision['audit_outer_exit_cause']==a.AUDIT_OUTER_EXIT_CAUSE=='post_completion_teardown'
    assert decision['scheduler_success'] is False and decision['readonly_new_grader_invocations']==0
    assert decision['other_jobs_or_future_failures_authorized'] is False


def test_real_certificate_carries_the_contract_classification_not_the_r4_one():
    """The r4 adapter classified its 143 as 'unknown'. Hardcoding that here would have
    refused the real r5 certificate, which the contract classifies as a teardown."""
    a=real_adapter()
    proof=json.loads((ROOT/'var/artifacts/modebench_scale_level4_python_r5_20260913'
        '/development/execution_reconciliation.json').read_text())
    expected=a.contract.execution_classification(a.OBSERVED_JOB)
    assert a.OBSERVED_EXIT_CAUSE==expected['exit_cause']=='post_completion_teardown'!='unknown'
    assert proof['exit_cause']==a.OBSERVED_EXIT_CAUSE and proof['array_job_id']==a.OBSERVED_JOB
    assert proof['schema']==a.contract.AUDIT_SCHEMA and proof['status']==a.contract.AUDIT_STATUS
    assert proof['scheduler_success'] is False


@pytest.mark.parametrize('action',['audit','verify'])
def test_real_outer_roots_are_the_v2_records_the_decision_pins(action):
    a=real_adapter();root=a.AUDIT_ROOT if action=='audit' else a.READONLY_ROOT
    decision=json.loads(Path(a.OBSERVED_CPU_DECISION).read_text())['files_sha256']
    assert root.is_dir() and root.name.endswith('_v2_20260913')
    for name in ('intent.json','process.json','stdout.txt','stderr.txt','exit.json'):
        assert decision.get(str(root/name))==sha_of(root/name),name
    exited=json.loads((root/'exit.json').read_text())
    assert exited['schema']=='modebench_scale_python_r5_soak_outer_execution_v1'
    assert exited['action']==action and exited['returncode']==(143 if action=='audit' else 0)
    assert exited['actual_host']==a.HOST=='wash.cs.princeton.edu' and exited['uid']==a.UID


def test_the_refused_first_attempt_is_never_read_as_the_audit():
    a=real_adapter();refused=a.engine.RECOVERY/'actual_audit_soak_20260913'
    assert refused.is_dir() and refused not in (a.AUDIT_ROOT,a.READONLY_ROOT)
    assert json.loads((refused/'exit.json').read_text())['returncode']==1
    preserved=json.loads(Path(a.CPU_PRESERVATION).read_text())
    assert sha_of(a.CPU_PRESERVATION)==a.CPU_PRESERVATION_SHA
    assert {Path(e['original_path']).parent for e in preserved['files']}=={refused}


def test_adapter_identity_matches_the_contract_and_its_inputs_all_exist():
    a=real_adapter()
    assert sha_of(a.PRIOR)==a.PRIOR_SHA and sha_of(a.CONTRACT)==a.CONTRACT_SHA
    assert str(a.OBSERVED_JOB) in a.POLICY and a.OBSERVED_JOB==a.contract.OBSERVED_JOB
    assert (a.HOST,a.UID)==(a.contract.HOST,a.contract.UID)
    for path in (a.SOURCE,a.TESTS,a.AUDIT_RECORDER,a.AUDIT_RECORDER_TESTS,
                 a.CONTRACT_TESTS,a.PRIOR_TESTS,a.engine.AUDITOR,a.engine.AUDITOR_REVIEW):
        assert Path(path).is_file(),path


@pytest.mark.parametrize('action',['audit','verify'])
def test_real_intents_bind_the_exact_auditor_review_and_recorder_the_run_used(action):
    """A v6/v7 auditor-review mix-up, or a stale recorder path, is invisible to every
    synthetic test here because the fixture rehashes whatever the adapter names."""
    a=real_adapter();m=a.engine;root=a.AUDIT_ROOT if action=='audit' else a.READONLY_ROOT
    intent=json.loads((root/'intent.json').read_text())
    assert intent['auditor_review_sha256']==sha_of(m.AUDITOR_REVIEW)
    assert m.AUDITOR_REVIEW.name.endswith('_v7_20260913.json')
    assert intent['script_sha256']==sha_of(a.AUDIT_RECORDER)
    assert intent['terminal_sha256']==sha_of(m.PROOF_FIELDS['execution_terminal_sha256'])
    # audit_command() needs the not-yet-written fit review only for the trailing
    # --terminal-sha256, so rebuild the prefix and check the tail against the record.
    prefix=[str(m.PYTHON),'-B',str(m.RUNNER),'exec','--manifest',str(m.VIEW),'--',str(m.PYTHON),'-B',
        str(m.AUDITOR),action,'--root',str(m.RECOVERY),'--review-sha256',sha_of(m.AUDITOR_REVIEW)]
    tail=['--terminal-sha256',intent['terminal_sha256']] if action=='audit' else []
    assert intent['command']==prefix+tail


def test_every_sealed_digest_the_adapter_adds_matches_the_file_on_disk():
    """The fixture rebuilds SEALED by rehashing whatever path the adapter names, so a
    digest bound to the WRONG path passes the whole synthetic suite. Two did: PRIOR_TESTS
    carried the r4 *fit* tests' digest and CONTRACT_TESTS carried the r4 contract tests'."""
    a=real_adapter()
    frozen=ROOT/'src/oat_drgrpo/templates.py'   # two-view file; differs outside the frozen view
    mismatched={str(p):(d,sha_of(p)) for p,d in a.engine.SEALED.items()
                if Path(p)!=frozen and sha_of(p)!=d}
    assert not mismatched,mismatched
    assert a.engine.SEALED[a.PRIOR]==a.PRIOR_SHA and a.engine.SEALED[a.CONTRACT]==a.CONTRACT_SHA
    assert len(a.engine.SEALED)>=8 and a.PRIOR_TESTS in a.engine.SEALED and a.CONTRACT_TESTS in a.engine.SEALED


def test_real_cell_tasks_match_what_the_adapter_requires_including_the_draw_seeds():
    """The blocking defect lived here: the adapter required r4's dev draw labels
    [7204000..] while the sealed r5 cell carries [7205000..]. Every fixture inherits its
    tasks from the r4 base fixture, so nothing synthetic could see it."""
    a=real_adapter();m=a.engine;root=m.REVISION_ROOT
    tasks=json.loads((m.PREPARATION/'development_tasks.json').read_text())
    cell=json.loads((m.PREPARATION/'development_cell.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    seeds=protocol['draw_labels']['dev']
    assert seeds==[7205000,7205001,7205002,7205003] and seeds!=[7204000,7204001,7204002,7204003]
    assert len(tasks)==4 and cell['tasks']==tasks and cell['id']=='level4_python_factors_r5_dev'
    for tier,task in enumerate(tasks):
        assert task=={'output':str(root/'level4/results/development/python_factors'/f'difficulty_{tier}.json'),
            'rows_jsonl':str(root/'level4/pools/python_factors'/f'difficulty_{tier}.jsonl'),
            'level':'level4','domain':'python_factors','split':'dev','batch_size':8,
            'row_offset':0,'row_limit':0,'interface':'modebench_qwen_scale_independent_v1','seeds':seeds},tier
    # The certificate must pin the protocol the seeds were read from, or they are unbound.
    proof=json.loads(m.PROOF_FIELDS['execution_certificate_sha256'].read_text())
    assert proof['files_sha256'][str(root/'protocol.json')]==sha_of(root/'protocol.json')
    # Plan-to-result: the receipts record what the scoring worker actually drew. Without
    # this the chain only proves the plan is self-consistent -- the execution auditor never
    # looks at seeds at all, so a run against a stale task file would pass every other check.
    for tier,task in enumerate(tasks):
        receipt=json.loads(Path(task['output']).read_text())
        assert receipt['identity']['seeds']==seeds,tier
        assert receipt['sampling']['seeds']==seeds,tier


def test_the_registrar_and_the_protocol_agree_on_the_revision_five_labels():
    """The draw labels are revision-indexed, so the registrar is the second witness."""
    a=real_adapter()
    source=(ROOT/'artifacts/register_modebench_scale_level4_python_r5_20260913.py').read_text()
    assert "'dev':[7205000,7205001,7205002,7205003]" in source.replace(' ','')
    protocol=json.loads((a.engine.REVISION_ROOT/'protocol.json').read_text())
    assert protocol['draw_labels']['dev']==[7205000,7205001,7205002,7205003]
    assert protocol['draw_labels']['eval']==[7205500,7205501,7205502,7205503]


def test_every_reachable_adapter_layer_binds_revision_fives_contract():
    """The rebind loop used to walk engine.engine, an axis that does not exist here, so
    the r4 fit adapter one layer down kept revision 4's contract and its failed-job waiver."""
    a=real_adapter();bound=[];seen=[]
    for start in (a.prior,a.engine):          # the adapter's own traversal, not a narrower one
        layer=start
        while layer is not None and layer not in seen:
            seen.append(layer)
            contract=getattr(layer,'contract',None)
            if contract is not None:bound.append(contract)
            layer=getattr(layer,'prior',None) or getattr(layer,'engine',None)
    assert bound and all(c is a.contract for c in bound)
    assert len(bound)>=2,'the chain must actually have more than one contract-bearing layer'
    assert a.contract.OBSERVED_JOB==31267007


def test_the_authorized_failure_branch_is_the_one_the_real_evidence_takes():
    """reviewed() gates the authorized-failure clauses on array_job_id == OBSERVED_JOB.
    Confirm the real certificate takes that branch rather than skipping the clauses."""
    a=real_adapter()
    proof=json.loads(a.engine.PROOF_FIELDS['execution_certificate_sha256'].read_text())
    assert proof['array_job_id']==a.contract.OBSERVED_JOB
    assert a.contract.execution_classification(proof['array_job_id'])['scheduler_success'] is False
    for other in a.contract.HISTORICAL_JOBS:
        with pytest.raises(ValueError,match='revision-4 arrays are refused'):
            a.contract.execution_classification(other)


@pytest.mark.parametrize('field',['identity','sampling'])
@pytest.mark.parametrize('value',[[7204000,7204001,7204002,7204003],[7205000,7205001,7205002],None])
def test_a_receipt_drawn_with_other_seeds_is_refused(fixture,field,value):
    """Neutering the receipt-seed clause must break something. The certificate's own
    receipt digest is re-pinned first, so the refusal comes from the seeds and not
    from a stale hash."""
    x=fixture;m=x.m
    tasks=m.read(m.PREPARATION/'development_tasks.json')
    receipt_path=Path(tasks[0]['output']);receipt=m.read(receipt_path)
    if value is None:receipt[field].pop('seeds')
    else:receipt[field]['seeds']=value
    write(receipt_path,receipt)
    proof=m.read(x.x.certificate)
    for summary in proof['summaries']:summary['receipt_sha256']=m.sha(summary['receipt'])
    proof['files_sha256'][str(receipt_path)]=m.sha(receipt_path)
    write(x.x.certificate,proof)
    # The certificate digest moved, so the outer records that pin it must be resealed too;
    # otherwise the refusal would come from a stale hash rather than from the seeds.
    for root in (x.a.AUDIT_ROOT,x.a.READONLY_ROOT):
        (root/'stdout.txt').write_text(json.dumps({**{k:proof[k] for k in ('schema','status','array_job_id')},
            'files':len(proof['files_sha256'])})+'\n')
        change(root/'exit.json',certificate_sha256=m.sha(x.x.certificate))
    repin(x)
    x.a.observed_cpu_audit_pins(guest=True)   # the resealing is complete; only seeds remain wrong
    with pytest.raises(ValueError,match='all four complete audited native Python tiers are required'):
        x.a.entries_and_proofs(guest=True)
