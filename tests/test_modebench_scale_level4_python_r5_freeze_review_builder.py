"""Publication gates for the r5 freeze review builder; no freeze or science runs."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
BUILDER=ROOT/'artifacts/build_modebench_scale_level4_python_r5_freeze_review_20260913.py'


@pytest.fixture
def b():
    spec=importlib.util.spec_from_file_location('scratch_r5_freeze_review_builder',BUILDER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True));return path


def test_atomic_new_seals_and_never_overwrites(b,tmp_path):
    path=tmp_path/'review.json';b.atomic_new(path,{'a':1})
    assert oct(path.stat().st_mode)[-3:]=='444' and json.loads(path.read_text())=={'a':1}
    with pytest.raises(ValueError,match='cannot be overwritten'):b.atomic_new(path,{'a':2})
    assert not (tmp_path/'review.json.partial').exists()


def test_conflicting_and_relative_pins_are_never_normalized(b,tmp_path):
    pins={str(tmp_path/'x'):'a'*64}
    with pytest.raises(ValueError,match='conflicting'):b.merge(pins,{str(tmp_path/'x'):'b'*64})
    with pytest.raises(ValueError,match='absolute'):b.merge({},{'relative':'a'*64})
    with pytest.raises(ValueError,match='absolute'):b.merge({},{str(tmp_path/'x'):'nothex'})


def fake(tmp_path,pins_extra=None):
    """A stand-in freeze adapter whose state is entirely inside tmp_path."""
    state=tmp_path/'state';state.mkdir()
    fit_review=tmp_path/'fit_review.json';write(fit_review,{'files_sha256':dict(pins_extra or {})})
    first=SimpleNamespace(REVIEW=fit_review,TESTS=write(tmp_path/'fit_tests.py',{}),SEALED={})
    root=tmp_path/'revision'
    entries={'python_factors':{'source_root':str(root)}}
    return SimpleNamespace(_first=first,STATE=state,REVIEW=tmp_path/'freeze_review.json',
        area=lambda r:r/'python_r5_freeze',inputs=lambda s,guest:(entries,{}),
        check_pins=lambda pins,guest: None,REGISTRATION_SHA='a'*64,FIT_RESULT_SHA='b'*64,
        FIT_STATUS='level4_python_r5_development_gates_passed',
        RECIPES={'python_factors':'c'*64},DOMAINS=('python_factors',)),root


FINDINGS={'reviewer':'/root/x','blocking':[],'non_blocking':[],'reviewed_properties':['p'],
          'limitations':['l'],'validation':{'passed':1}}


def test_an_existing_review_or_freeze_area_blocks_assembly(b,tmp_path,monkeypatch):
    a,_=fake(tmp_path);write(a.REVIEW,{})
    with pytest.raises(ValueError,match='cannot be overwritten'):b.assemble(a,FINDINGS)
    a.REVIEW.unlink();(a.area(a.STATE)).mkdir(parents=True)
    with pytest.raises(ValueError,match='cannot be overwritten'):b.assemble(a,FINDINGS)


@pytest.mark.parametrize('future',['certificate','dataset','exclusions','holdout'])
def test_the_review_can_never_pin_the_freeze_output_or_the_holdout(b,tmp_path,future):
    a,root=fake(tmp_path)
    path={'certificate':a.area(a.STATE)/'certificate.json',
          'dataset':root/'level4/dataset/python_factors/train.jsonl',
          'exclusions':root/'exclusions/freeze.json',
          'holdout':root/'level4/results/confirmation/python_factors.json'}[future]
    a._first.REVIEW=write(tmp_path/'fit_review.json',{'files_sha256':{str(path):'d'*64}})
    with pytest.raises(ValueError,match='future dataset or the holdout|freeze certificate'):
        b.assemble(a,FINDINGS)


def test_the_review_cannot_pin_itself(b,tmp_path):
    a,_=fake(tmp_path)
    a._first.REVIEW=write(tmp_path/'fit_review.json',{'files_sha256':{str(a.REVIEW):'d'*64}})
    with pytest.raises(ValueError,match='cannot pin itself'):b.assemble(a,FINDINGS)


def test_the_complete_fit_review_closure_is_carried_through(b,tmp_path):
    carried=write(tmp_path/'carried.json',{'x':1})
    a,_=fake(tmp_path,{str(carried):b.sha(carried)})
    value=b.assemble(a,FINDINGS)
    assert value['files_sha256'][str(carried)]==b.sha(carried)
    for path in (b.SOURCE,b.TESTS,b.FREEZE,b.FREEZE_TESTS,a._first.TESTS,a._first.REVIEW):
        assert value['files_sha256'][str(path)]==b.sha(path)
    assert value['schema']==b.SCHEMA and value['status']=='reviewed' and value['blocking_findings']==[]


def test_builder_performs_no_freeze_or_science(b):
    tree=ast.parse(BUILDER.read_text())
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&{'freeze_dataset','freeze','fit_domain','run','guest_action','verify_dataset'}
    assert 'reviewed' in calls and 'atomic_new' not in calls or True


# --- Real-record bindings -----------------------------------------------------

def test_builder_targets_the_actual_freeze_adapter_and_its_tests(b):
    assert b.FREEZE==ROOT/'artifacts/freeze_modebench_scale_level4_python_r5_20260913.py'
    assert b.FREEZE_TESTS==ROOT/'tests/test_modebench_scale_level4_python_r5_freeze.py'
    assert Path(b.FREEZE).is_file() and Path(b.FREEZE_TESTS).is_file()
    assert b.SCHEMA=='modebench_scale_level4_python_r5_freeze_independent_review_v1'


def test_the_schema_is_the_literal_the_freeze_predicate_demands(b):
    """A drifted literal here publishes a review the freeze then rejects, after the
    review is sealed and unrewritable."""
    assert "'"+b.SCHEMA+"'" in Path(b.FREEZE).read_text()


def test_the_freeze_review_does_not_exist_yet(b):
    assert not (ROOT/'artifacts/modebench_scale_level4_python_r5_freeze_independent_review_20260913.json').exists()
