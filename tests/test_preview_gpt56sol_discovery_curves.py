import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'ops/preview_gpt56sol_discovery_curves.py'
spec = importlib.util.spec_from_file_location('sol_preview_tests', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def registry():
    return {'runs': [{'model_id': model, 'arm': arm, 'run_dir': '/unused/'+model+'/'+arm}
                     for model in m.REGISTERED for arm in m.ARMS]}


def full_status():
    return {'gpt56sol/'+arm: {'complete': True, 'saved_responses': 6144,
                            'expected_responses': 6144, 'grading_audit_present': True}
            for arm in m.ARMS}


def test_selection_requires_all_registered_models_and_both_fresh_arms():
    r = registry()
    assert set(m.selected_entries(r)) == set(m.ARMS)
    assert all(x['model_id']=='gpt56sol' for x in m.selected_entries(r).values())
    for bad in ({'runs':r['runs'][:-1]}, {'runs':r['runs']+[r['runs'][0]]}):
        with pytest.raises(ValueError):m.selected_entries(bad)


@pytest.mark.parametrize('arm', m.ARMS)
@pytest.mark.parametrize('field,bad', [('complete',False),('saved_responses',6143),
                                     ('expected_responses',6143),('grading_audit_present',False)])
def test_never_preview_partial_pools_or_ungraded_cohort(arm,field,bad):
    s=full_status();assert m.ready(s)
    s['gpt56sol/'+arm][field]=bad
    assert not m.ready(s)


def test_missing_arm_is_not_ready():
    assert not m.ready({})
    s=full_status();del s['gpt56sol/neutral'];assert not m.ready(s)


def test_indices_exactly_match_registered_full_report_order_and_seed():
    mod=SimpleNamespace(SEED=20260911,PROMPTS_PER_CELL=16,LEVELS=(2,3),
                        DOMAINS=('python_factors','mathir','pantry_plan'))
    actual=m.bootstrap_indices(mod);rng=np.random.default_rng(20260911)
    for level in mod.LEVELS:
        for domain in mod.DOMAINS:
            expected=rng.integers(0,16,(20000,16))
            np.testing.assert_array_equal(actual[level,domain],expected)
    assert len(actual)==6


@pytest.mark.parametrize('name', ['analysis_complete', 'analysis_local_complete', 'analysis_complete/subdir'])
def test_preview_cannot_create_or_modify_official_output(tmp_path,name):
    with pytest.raises(ValueError):m.validate_destination(tmp_path,tmp_path/name)
    assert not (tmp_path/'analysis_complete').exists()


def test_existing_preview_is_preserved(tmp_path):
    out=tmp_path/'gpt56sol_complete_model_preview';out.mkdir();(out/'receipt').write_text('unchanged')
    with pytest.raises(ValueError):m.validate_destination(tmp_path,out)
    assert (out/'receipt').read_text()=='unchanged'


def test_orchestration_uses_frozen_authentication_and_all_six_cells():
    calls=[]
    mod=SimpleNamespace(SEED=20260911,PROMPTS_PER_CELL=16,LEVELS=(2,3),
                        DOMAINS=('python_factors','mathir','pantry_plan'),GRADINGS=('strict','normalized_secondary'))
    cells={f'level{l}/{d}':{} for l in mod.LEVELS for d in mod.DOMAINS}
    expected={'model_id':'gpt56sol','family':'frontier','analyses':{
        grading:{'cells':dict(cells),'prompts':{arm:[{'responses':64} for _ in range(96)] for arm in m.ARMS}}
        for grading in mod.GRADINGS}}
    mod.authenticate_hosted=lambda design,e:(calls.append(('authenticate',e['model_id'],e['arm'])) or e)
    mod.validate_payload_pair=lambda a,b:calls.append(('validate_pair',a['arm'],b['arm']))
    def analyze(design,a,b,indices):
        assert all(x.shape==(20000,16) for x in indices.values())
        calls.append(('analyze',));return expected
    mod.analyze_pair=analyze
    assert m.analyze_completed_model(mod,{},registry()) is expected
    assert calls==[('authenticate','gpt56sol','original'),('authenticate','gpt56sol','neutral'),
                   ('validate_pair','original','neutral'),('analyze',)]
    expected['analyses']['strict']['prompts']['neutral'][0]['responses']=63
    with pytest.raises(ValueError,match='complete 64-draw'):m.analyze_completed_model(mod,{},registry())


def test_orchestration_does_not_swallow_frozen_authentication_failure():
    mod=SimpleNamespace()
    def fail(*args):raise ValueError('native receipt mismatch')
    mod.authenticate_hosted=fail
    with pytest.raises(ValueError,match='native receipt mismatch'):
        m.analyze_completed_model(mod,{},registry())
