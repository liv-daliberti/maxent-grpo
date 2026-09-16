"""Synthetic publication gates only; actual auditor/fitter imports never run."""
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


@pytest.fixture
def builder():
    spec=importlib.util.spec_from_file_location('scratch_r5_fit_review_builder',ROOT/'artifacts/build_modebench_scale_python_r5_fit_observed_audit_final_review_20260913.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def records(builder,tmp_path,monkeypatch):
    b=builder;m=SimpleNamespace(PYTHON=tmp_path/'python',RUNNER=tmp_path/'runner.py',VIEW=tmp_path/'manifest.json',
        AUDITOR=tmp_path/'auditor.py',AUDITOR_REVIEW=tmp_path/'auditor_review.json',RECOVERY=tmp_path/'development')
    for path in (m.PYTHON,m.RUNNER,m.VIEW,m.AUDITOR,m.AUDITOR_REVIEW):path.write_text('{}')
    a=SimpleNamespace(engine=m,AUDIT_ROOT=tmp_path/'audit',READONLY_ROOT=tmp_path/'readonly',AUDIT_RECORDER=tmp_path/'recorder.py',
        HOST='wash.cs.princeton.edu',UID=363432,contract=SimpleNamespace(moment=datetime.fromisoformat))
    a.AUDIT_RECORDER.write_text('no native actions')
    proof={'schema':'actual_r5_audit','status':'complete_existing_audit','array_job_id':31267007,
        'files_sha256':{str(m.AUDITOR):b.sha(m.AUDITOR)},'audit_completed_at_utc':'2026-09-13T16:00:02+00:00'}
    certificate_sha='c'*64;terminal_sha='d'*64;cpu_sha='e'*64
    monkeypatch.setattr(b,'CPU_DECISION',tmp_path/'cpu_decision.json')
    m.PROOF_FIELDS={'execution_certificate_sha256':tmp_path/'certificate.json'}
    for index,(action,root) in enumerate([('audit',a.AUDIT_ROOT),('verify',a.READONLY_ROOT)]):
        start=f'2026-09-13T16:00:0{index*3}+00:00';finish=f'2026-09-13T16:00:0{index*3+2}+00:00'
        command=b.audit_command(a,action,terminal_sha)
        intent={'schema':b.OUTER_SCHEMA,'action':action,'actual_host':a.HOST,'uid':a.UID,'pid':100+index,
            'started_at_utc':start,'command':command,'script_sha256':b.sha(a.AUDIT_RECORDER),
            'auditor_review_sha256':b.sha(m.AUDITOR_REVIEW),'terminal_sha256':terminal_sha}
        write(root/'intent.json',intent)
        write(root/'process.json',{'actual_host':a.HOST,'uid':a.UID,'pid':100+index,'child_pid':200+index,'at_utc':start})
        summary={k:proof[k] for k in ('schema','status','array_job_id')};summary['files']=len(proof['files_sha256'])
        (root/'stdout.txt').write_text(json.dumps(summary)+'\n');(root/'stderr.txt').write_text('')
        exited={**intent,'finished_at_utc':finish,'returncode':143 if action=='audit' else 0,'postcheck_error':None,'certificate_present':True,
            'certificate_sha256':certificate_sha,'new_grader_invocations':24704 if action=='audit' else 0,
            **{field:b.sha(root/name) for field,name in [('intent_sha256','intent.json'),('process_sha256','process.json'),('stdout_sha256','stdout.txt'),('stderr_sha256','stderr.txt')]}}
        write(root/'exit.json',exited)
    expected={str(b.CPU_DECISION):cpu_sha,str(m.PROOF_FIELDS['execution_certificate_sha256']):certificate_sha,
        str(a.AUDIT_ROOT/'exit.json'):b.sha(a.AUDIT_ROOT/'exit.json'),str(a.READONLY_ROOT/'exit.json'):b.sha(a.READONLY_ROOT/'exit.json')}
    a.observed_cpu_audit_pins=lambda *,guest:dict(expected)
    def run():return b.outer_records(a,proof,certificate_sha,terminal_sha,b.sha(a.AUDIT_ROOT/'exit.json'),b.sha(a.READONLY_ROOT/'exit.json'),cpu_sha)
    return SimpleNamespace(b=b,a=a,proof=proof,run=run,expected=expected)


def test_real_record_contract_requires_exact_observed_audit143_and_separate_readonly_zero(records):
    x=records;pins=x.run();assert len(pins)==12 and all(x.b.sha(p)==d for p,d in pins.items() if p not in (str(x.b.CPU_DECISION),str(x.a.engine.PROOF_FIELDS['execution_certificate_sha256'])))


@pytest.mark.parametrize('action,code',[('audit',0),('audit',-15),('audit',1),('verify',143),('verify',1),('verify',True)])
def test_cpu_nonzero_and_noninteger_exits_are_never_normalized(records,action,code):
    x=records;root=x.a.AUDIT_ROOT if action=='audit' else x.a.READONLY_ROOT
    value=x.b.read(root/'exit.json');value['returncode']=code;write(root/'exit.json',value)
    with pytest.raises(ValueError,match='audit143'):x.run()


@pytest.mark.parametrize('kind',['certificate_hash','missing_certificate','wrong_host','wrong_uid','changed_argv','terminal_hash','recorder_hash','review_hash','postcheck','readonly_grader'])
def test_rehashed_outer_records_cannot_relink_wrong_observation(records,kind):
    x=records;root=x.a.READONLY_ROOT;intent=x.b.read(root/'intent.json');exited=x.b.read(root/'exit.json')
    if kind=='certificate_hash':exited['certificate_sha256']='e'*64
    elif kind=='missing_certificate':exited['certificate_present']=False
    elif kind=='wrong_host':intent['actual_host']=exited['actual_host']='spin.cs.princeton.edu'
    elif kind=='wrong_uid':intent['uid']=exited['uid']=0
    elif kind=='changed_argv':intent['command']=exited['command']=['relative','verify']
    elif kind=='terminal_hash':intent['terminal_sha256']='e'*64
    elif kind=='recorder_hash':intent['script_sha256']='e'*64
    elif kind=='review_hash':intent['auditor_review_sha256']='e'*64
    elif kind=='postcheck':exited['postcheck_error']='failed'
    else:exited['new_grader_invocations']=1
    write(root/'intent.json',intent);exited['intent_sha256']=x.b.sha(root/'intent.json');write(root/'exit.json',exited)
    with pytest.raises(ValueError):x.run()


@pytest.mark.parametrize('kind',['during_readonly','before_audit','missing_record','symlink_record','wrong_stdout'])
def test_chronology_and_actual_record_completeness_are_required(records,kind):
    x=records
    if kind=='during_readonly':x.proof['audit_completed_at_utc']='2026-09-13T16:00:04+00:00'
    elif kind=='before_audit':x.proof['audit_completed_at_utc']='2026-09-13T15:59:59+00:00'
    elif kind=='missing_record':(x.a.READONLY_ROOT/'process.json').unlink()
    elif kind=='symlink_record':
        path=x.a.READONLY_ROOT/'process.json';copy=path.parent/'alias.json';copy.write_bytes(path.read_bytes());path.unlink();path.symlink_to(copy)
    else:
        path=x.a.READONLY_ROOT/'stdout.txt';path.write_text(json.dumps({'schema':'wrong','status':'complete','array_job_id':31267007,'files':1}))
        value=x.b.read(x.a.READONLY_ROOT/'exit.json');value['stdout_sha256']=x.b.sha(path);write(x.a.READONLY_ROOT/'exit.json',value)
    with pytest.raises(ValueError):x.run()


@pytest.mark.parametrize('kind',['review','state','recipe','dataset','confirmation'])
def test_future_review_fit_and_heldout_outputs_are_not_accepted_as_evidence(builder,tmp_path,kind):
    b=builder;m=SimpleNamespace(REVIEW=tmp_path/'review.json',STATE=tmp_path/'fit',REVISION_ROOT=tmp_path/'r5')
    path={'review':m.REVIEW,'state':m.STATE/'runtime.json','recipe':m.REVISION_ROOT/'level4/recipes/python_factors.json',
        'dataset':m.REVISION_ROOT/'level4/dataset/python_factors/train/data.arrow',
        'confirmation':m.REVISION_ROOT/'level4/results/confirmation/python_factors.json.batches/0.json'}[kind]
    with pytest.raises(ValueError,match='future'):b.no_future_pins(m,{str(path):'a'*64})


def test_conflicting_hashes_and_nonabsolute_paths_are_not_silently_normalized(builder,tmp_path):
    p={str(tmp_path/'input'):'a'*64}
    with pytest.raises(ValueError,match='conflicting'):builder.merge(p,{str(tmp_path/'input'):'b'*64})
    with pytest.raises(ValueError,match='absolute'):builder.merge({}, {'relative':'a'*64})


def test_builder_import_and_assembly_contain_no_native_execution_entrypoints(builder):
    tree=ast.parse(builder.SOURCE.read_text())
    forbidden={'fit_domain','fit_registered','collect_inputs','verify_existing','audit','reconcile','register','materialize_pools','launch_inputs','freeze_dataset'}
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&forbidden
    assert 'atomic_new' in calls and 'reviewed' in calls


@pytest.mark.parametrize('key',['decision','certificate','audit','readonly'])
def test_only_exact_pinned_observation_allows_literal_143(records,key):
    x=records
    path={'decision':x.b.CPU_DECISION,'certificate':x.a.engine.PROOF_FIELDS['execution_certificate_sha256'],
        'audit':x.a.AUDIT_ROOT/'exit.json','readonly':x.a.READONLY_ROOT/'exit.json'}[key]
    x.expected[str(path)]='f'*64
    with pytest.raises(ValueError,match='exact pinned observed'):x.run()


def test_observed_root_decision_predicate_failure_cannot_be_bypassed(records):
    x=records
    def rejected(*,guest):raise ValueError('root decision or fixed evidence changed')
    x.a.observed_cpu_audit_pins=rejected
    with pytest.raises(ValueError,match='root decision'):x.run()


def test_original_native_certificate_fit_and_builder_paths_stay_literal(builder):
    """Both earlier fit adapters must stay byte-identical, and this builder must target
    neither of them. A builder pointed at a predecessor would review the wrong source."""
    expected={
        ROOT/'artifacts/continue_modebench_scale_level4_python_r4_fit_20260913.py':'fe06460188022666eb2f038986abebf2193e18422e10a6aa9a1af338f5585ed7',
        ROOT/'tests/test_modebench_scale_level4_python_r4_fit.py':'439238e405ad9e9b4e8ef95a840d217e57ea7591e8d01c8653b664f72d4301cb',
        ROOT/'artifacts/continue_modebench_scale_level4_python_r4_fit_observed_cpu_20260913.py':'168de7a208b7cfa415a065555ffd9b2f365875d70a9bea307316511b1ce6b8dc',
        ROOT/'artifacts/build_modebench_scale_python_r4_fit_final_review_20260913.py':'d9196fbc6822e202b5bd21ce945879b8d8d40b2925b695b05dc94ed092bdbcfb',
        # The files this builder was actually transformed from. Without them the
        # not-in-expected assertion below cannot catch the likeliest mis-target.
        ROOT/'artifacts/build_modebench_scale_python_r4_fit_observed_cpu_final_review_20260913.py':'5f6d6fd29266399f68141d3aea3bb3eb26ce50f90ca05268c71be08bbc21da2c',
        ROOT/'tests/test_modebench_scale_python_r4_fit_observed_cpu_final_review_builder.py':'512c8621564d8658833bdb856ce1ccc09992029814a815802bb46d8429b0a9b6',
        ROOT/'tests/test_modebench_scale_python_r4_fit_final_review_builder.py':'bc53e22afa1f79e5cbbe70d405dfce0778c66566a499d623d4ca834b60ade7a6'}
    assert all(builder.sha(path)==digest for path,digest in expected.items()),'a sealed predecessor changed'
    assert builder.FIT==ROOT/'artifacts/continue_modebench_scale_level4_python_r5_fit_observed_audit_20260913.py'
    assert builder.FIT not in expected and builder.SOURCE not in expected


@pytest.mark.parametrize('kind',['exact','stale_source','stale_tests','missing_source_pin','unreviewed','blocker'])
def test_current_component_review_binds_exact_fit_source_and_tests(builder,tmp_path,monkeypatch,kind):
    b=builder;p=tmp_path/'component.json';monkeypatch.setattr(b,'FIT_COMPONENT_REVIEW',p)
    value={'schema':'modebench_scale_level4_python_r5_fit_observed_audit_component_independent_review_v1',
        'status':'reviewed','blocking_findings':[],'source_sha256':'a'*64,'tests_sha256':'b'*64,
        'files_sha256':{str(b.FIT):'a'*64,str(b.FIT_TESTS):'b'*64}}
    if kind=='stale_source':value['source_sha256']='c'*64
    elif kind=='stale_tests':value['tests_sha256']='c'*64
    elif kind=='missing_source_pin':del value['files_sha256'][str(b.FIT)]
    elif kind=='unreviewed':value['status']='draft'
    elif kind=='blocker':value['blocking_findings']=['unresolved']
    write(p,value);args=SimpleNamespace(fit_component_review_sha256=b.sha(p),fit_source_sha256='a'*64,fit_tests_sha256='b'*64)
    if kind=='exact':assert b.current_fit_component(args)[str(p)]==b.sha(p)
    else:
        with pytest.raises(ValueError,match='completed independent component'):b.current_fit_component(args)


# --- Real-record bindings -----------------------------------------------------
# The synthetic tests above drive the builder with a SimpleNamespace stand-in for the
# fit adapter, so every constant is checked against itself. These read the sealed
# records and the real adapter instead. They are the countermeasure for the defect
# class that produced five blockers in the adapter this builder reviews: a constant
# advanced from r4 to r5 everywhere except one place.

def real_fit_adapter():
    spec=importlib.util.spec_from_file_location('_r5_builder_real_binding',
        ROOT/'artifacts/continue_modebench_scale_level4_python_r5_fit_observed_audit_20260913.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_builder_job_and_causes_equal_the_adapter_and_the_contract(builder):
    a=real_fit_adapter()
    assert builder.JOB==a.OBSERVED_JOB==a.contract.OBSERVED_JOB==31267007
    assert a.OBSERVED_EXIT_CAUSE==a.contract.execution_classification(builder.JOB)['exit_cause']
    assert a.OBSERVED_EXIT_CAUSE=='post_completion_teardown' and a.AUDIT_OUTER_EXIT_CAUSE=='post_completion_teardown'


def test_builder_outer_schema_is_the_literal_the_real_records_carry(builder):
    a=real_fit_adapter()
    for root in (a.AUDIT_ROOT,a.READONLY_ROOT):
        assert json.loads((root/'exit.json').read_text())['schema']==builder.OUTER_SCHEMA
        assert json.loads((root/'intent.json').read_text())['schema']==builder.OUTER_SCHEMA


def test_builder_review_schema_is_the_one_the_adapter_will_demand(builder):
    """reviewed() refuses any other schema, so a drifted literal here publishes a review
    the fit then rejects -- after the review is sealed and unrewritable."""
    source=(ROOT/'artifacts/continue_modebench_scale_level4_python_r5_fit_observed_audit_20260913.py').read_text()
    assert "'"+builder.SCHEMA+"'" in source
    assert builder.SCHEMA=='modebench_scale_level4_python_r5_fit_independent_review_v1'


def test_builder_evidence_paths_exist_and_name_revision_five(builder):
    for path in (builder.SOURCE,builder.TESTS,builder.FIT,builder.FIT_TESTS,
                 builder.AUTHORIZATION,builder.DECISION,builder.CPU_DECISION):
        assert Path(path).is_file(),path
        assert '_r5_' in Path(path).name or '_python_r5' in Path(path).name,path
    # The component review is the one input that must NOT exist yet.
    assert not Path(builder.FIT_COMPONENT_REVIEW).exists()


def test_real_authorization_and_decision_name_this_job_and_this_cause(builder):
    authorization=json.loads(Path(builder.AUTHORIZATION).read_text())
    decision=json.loads(Path(builder.DECISION).read_text())
    assert authorization['schema']=='modebench_scale_level4_python_r5_user_authorization_v1'
    assert authorization['authorized_job_id']==builder.JOB
    assert decision['schema']=='modebench_scale_level4_python_r5_observed_terminal_decision_v1'
    assert decision['actual_job_id']==builder.JOB and decision['scheduler_success'] is False
    assert decision['evaluator_returncode']==0 and decision['exit_cause']=='post_completion_teardown'
    # r5 inverts r4's direction: the authorization predates the decision, so the decision
    # pins the authorization, not the reverse. Asserting r4's direction refuses real records.
    assert decision['user_authorization_sha256']==builder.sha(builder.AUTHORIZATION)
    assert authorization['files_sha256']=={}   # present but empty: it predates the decision
    assert authorization['recorded_at_utc']<decision['at_utc']
