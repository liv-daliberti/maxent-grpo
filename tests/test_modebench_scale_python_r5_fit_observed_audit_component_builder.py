"""Publication gates for the r5 component builder; no fit, audit or grader ever runs."""
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
BUILDER=ROOT/'artifacts/build_modebench_scale_python_r5_fit_observed_audit_component_20260913.py'


@pytest.fixture
def b():
    spec=importlib.util.spec_from_file_location('scratch_r5_component_builder',BUILDER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True));return path


def test_atomic_new_seals_and_never_overwrites(b,tmp_path):
    path=tmp_path/'record.json';b.atomic_new(path,{'a':1})
    assert json.loads(path.read_text())=={'a':1} and oct(path.stat().st_mode)[-3:]=='444'
    with pytest.raises(ValueError,match='cannot be overwritten'):b.atomic_new(path,{'a':2})
    assert json.loads(path.read_text())=={'a':1}
    assert not (tmp_path/'record.json.partial').exists()


def test_a_symlinked_target_is_refused(b,tmp_path):
    real=write(tmp_path/'real.json',{'a':1});link=tmp_path/'link.json';link.symlink_to(real)
    with pytest.raises(ValueError,match='cannot be overwritten'):b.atomic_new(link,{'a':2})


def test_a_blocking_finding_can_never_be_published_as_reviewed(b,tmp_path,monkeypatch):
    findings=write(tmp_path/'f.json',{'reviewer':'/root/x','blocking':['unresolved defect'],
        'non_blocking':[],'reviewed_properties':['p']})
    monkeypatch.setattr(b,'COMPONENT',write(tmp_path/'c.json',{'schema':b.COMPONENT_SCHEMA,
        'source_sha256':b.sha(b.FIT),'tests_sha256':b.sha(b.FIT_TESTS),'validation':{}}))
    a=SimpleNamespace(engine=SimpleNamespace(REVIEW=tmp_path/'nope.json',STATE=tmp_path/'nostate'))
    with pytest.raises(ValueError,match='blocking finding cannot be published'):
        b.independent_review(a,SimpleNamespace(findings=findings))


@pytest.mark.parametrize('field',['blocking','non_blocking','reviewed_properties','reviewer'])
def test_incomplete_reviewer_findings_are_refused(b,tmp_path,monkeypatch,field):
    value={'reviewer':'/root/x','blocking':[],'non_blocking':[],'reviewed_properties':['p']}
    del value[field]
    findings=write(tmp_path/'f.json',value)
    monkeypatch.setattr(b,'COMPONENT',write(tmp_path/'c.json',{'schema':b.COMPONENT_SCHEMA,
        'source_sha256':b.sha(b.FIT),'tests_sha256':b.sha(b.FIT_TESTS),'validation':{}}))
    a=SimpleNamespace(engine=SimpleNamespace(REVIEW=tmp_path/'nope.json',STATE=tmp_path/'nostate'))
    with pytest.raises(ValueError,match='explicit reviewer findings required'):
        b.independent_review(a,SimpleNamespace(findings=findings))


@pytest.mark.parametrize('field',['source_sha256','tests_sha256'])
def test_a_component_describing_stale_bytes_is_refused(b,tmp_path,monkeypatch,field):
    value={'schema':b.COMPONENT_SCHEMA,'source_sha256':b.sha(b.FIT),
        'tests_sha256':b.sha(b.FIT_TESTS),'validation':{}}
    value[field]='0'*64
    monkeypatch.setattr(b,'COMPONENT',write(tmp_path/'c.json',value))
    findings=write(tmp_path/'f.json',{'reviewer':'/root/x','blocking':[],'non_blocking':[],'reviewed_properties':['p']})
    a=SimpleNamespace(engine=SimpleNamespace(REVIEW=tmp_path/'nope.json',STATE=tmp_path/'nostate'))
    with pytest.raises(ValueError,match='current fit bytes'):
        b.independent_review(a,SimpleNamespace(findings=findings))


def test_the_review_cannot_follow_an_existing_fit_review_or_fit_state(b,tmp_path,monkeypatch):
    monkeypatch.setattr(b,'COMPONENT',write(tmp_path/'c.json',{'schema':b.COMPONENT_SCHEMA,
        'source_sha256':b.sha(b.FIT),'tests_sha256':b.sha(b.FIT_TESTS),'validation':{}}))
    findings=write(tmp_path/'f.json',{'reviewer':'/root/x','blocking':[],'non_blocking':[],'reviewed_properties':['p']})
    a=SimpleNamespace(engine=SimpleNamespace(REVIEW=write(tmp_path/'review.json',{}),STATE=tmp_path/'nostate'))
    with pytest.raises(ValueError,match='must precede the fit review'):
        b.independent_review(a,SimpleNamespace(findings=findings))


def test_builder_contains_no_native_execution_entrypoints(b):
    tree=ast.parse(BUILDER.read_text())
    forbidden={'fit_domain','fit_registered','collect_inputs','verify_existing','audit',
               'reconcile','register','materialize_pools','launch_inputs','freeze_dataset','run'}
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&forbidden


# --- Real-record bindings -----------------------------------------------------

def test_facts_are_read_from_the_sealed_records_and_must_agree(b):
    """facts() re-derives the job, the cause and the draw labels from three independent
    records and refuses if they disagree, so it cannot restate a stale constant."""
    a=b.adapter();value=b.facts(a)
    proof=json.loads(a.engine.PROOF_FIELDS['execution_certificate_sha256'].read_text())
    decision=json.loads(Path(a.OBSERVED_CPU_DECISION).read_text())
    protocol=json.loads((a.engine.REVISION_ROOT/'protocol.json').read_text())
    assert value['array_job_id']==proof['array_job_id']==a.contract.OBSERVED_JOB==31267007
    assert value['exit_cause']==proof['exit_cause']==decision['audit_outer_exit_cause']=='post_completion_teardown'
    assert value['dev_draw_labels']==protocol['draw_labels']['dev']==[7205000,7205001,7205002,7205003]
    assert value['native_grader_checks']==24704 and value['native_tiers']==4
    assert value['certificate_files']==len(proof['files_sha256'])


def test_every_evidence_pin_is_a_real_file_at_its_actual_digest(b):
    a=b.adapter();pins=b.evidence(a)
    assert len(pins)>=30
    for path,digest in pins.items():
        assert Path(path).is_file() and b.sha(path)==digest,path
    for required in (b.FIT,b.FIT_TESTS,b.BUILDER,b.BUILDER_TESTS,b.POSTRUN_TESTS,b.SOURCE):
        assert str(required) in pins,required


def test_the_component_records_do_not_exist_yet(b):
    assert not Path(b.COMPONENT).exists() and not Path(b.REVIEW).exists()
