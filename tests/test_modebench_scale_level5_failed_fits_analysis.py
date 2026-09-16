"""Arithmetic and real-record tests for the Level5 failed-fit diagnosis."""
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/analyze_modebench_scale_level5_failed_fits_20260914.py'


@pytest.fixture
def a():
    spec=importlib.util.spec_from_file_location('scratch_level5_diagnosis',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_homogeneous_ceiling_is_the_standard_identity(a):
    assert a.homogeneous_pass8(0.0)==0.0
    assert a.homogeneous_pass8(1.0)==1.0
    assert abs(a.homogeneous_pass8(0.5)-(1-0.5**8))<1e-12
    # A pool with no dead rows cannot fall below its homogeneous ceiling by much;
    # a bimodal pool falls far below it. That gap is the diagnostic.
    assert a.homogeneous_pass8(0.2078)>0.8


def test_the_weight_grid_is_the_fitters_own_grid(a):
    weights=list(a.weight_grid())
    assert len(weights)==1771
    assert all(sum(w)==20 and len(w)==4 for w in weights)
    assert len(set(weights))==1771


def test_a_synthetic_reachable_target_is_found_feasible(a,monkeypatch,tmp_path):
    """Guards against a hull search that reports zero for everything."""
    p1=[0.40,0.30,0.20,0.10];p8=[0.80,0.60,0.40,0.20]
    target={'pass1':0.25,'pass8':0.50};tol={'pass1':0.04,'pass8':0.08}
    feasible=0
    for w in a.weight_grid():
        ws=[x/20 for x in w]
        x=sum(u*v for u,v in zip(ws,p1));y=sum(u*v for u,v in zip(ws,p8))
        if max(abs(x-target['pass1'])/tol['pass1'],abs(y-target['pass8'])/tol['pass8'])<=1.0:feasible+=1
    assert feasible>0


def test_an_unreachable_target_is_found_infeasible(a):
    p1=[0.40,0.38,0.36,0.34];p8=[0.70,0.68,0.66,0.64]
    target={'pass1':0.05,'pass8':0.10};tol={'pass1':0.04,'pass8':0.08}
    feasible=0
    for w in a.weight_grid():
        ws=[x/20 for x in w]
        x=sum(u*v for u,v in zip(ws,p1));y=sum(u*v for u,v in zip(ws,p8))
        if max(abs(x-target['pass1'])/tol['pass1'],abs(y-target['pass8'])/tol['pass8'])<=1.0:feasible+=1
    assert feasible==0


# --- Real-record bindings -----------------------------------------------------

def test_every_analysed_receipt_is_pinned_at_its_actual_digest(a):
    for revision,domain,level in a.CASES:
        case=a.analyse(revision,domain,level)
        assert len(case['files_sha256'])==5,'protocol plus four tier receipts'
        for path,digest in case['files_sha256'].items():
            assert Path(path).is_file() and a.sha(path)==digest,path
        assert len(case['tiers'])==4
        assert all(t['rows']>0 for t in case['tiers'])


def test_all_three_cases_are_actually_infeasible_today(a):
    """If any became feasible, the diagnosis is stale and must be rebuilt."""
    for revision,domain,level in a.CASES:
        case=a.analyse(revision,domain,level)
        assert case['joint_feasible_grid_mixtures']==0,revision
        assert case['target_inside_reachable_hull'] is False,revision
        assert case['best_reachable']['normalised_error']>1.0,revision


def test_the_three_cases_fail_for_different_reasons(a):
    """The prescriptions differ, so the diagnosis must separate them."""
    cases={c[0]:a.analyse(*c) for c in a.CASES}
    r2=cases['level5_graph_coloring_r2']
    # Graph r2: both metrics marginally bracketed, yet jointly unreachable.
    assert r2['marginal_bracketing']['pass1']['brackets'] is True
    assert r2['marginal_bracketing']['pass8']['brackets'] is True
    assert r2['joint_feasible_grid_mixtures']==0
    r3=cases['level5_graph_coloring_r3']
    # Graph r3: the ladder lost its gradient -- at least one tier is not harder.
    assert r3['difficulty_inversions'],'r3 must show a measured difficulty inversion'
    assert r3['monotone_pass8'] is False
    assert r3['marginal_bracketing']['pass1']['brackets'] is False
    pantry=cases['level5_pantry_r2']
    # Pantry: every tier is easier than the target on both metrics.
    assert pantry['marginal_bracketing']['pass1']['low']>pantry['target']['pass1']
    assert pantry['marginal_bracketing']['pass8']['low']>pantry['target']['pass8']


def test_graph_r2_is_the_better_branch_point_than_r3(a):
    """r3 was a regression on r2; the record must make that checkable."""
    r2=a.analyse('level5_graph_coloring_r2','graph_coloring','level5')
    r3=a.analyse('level5_graph_coloring_r3','graph_coloring','level5')
    assert r2['best_reachable']['normalised_error']<r3['best_reachable']['normalised_error']
    assert not r2['difficulty_inversions'] or len(r2['difficulty_inversions'])<len(r3['difficulty_inversions'])


def test_the_analysis_performs_no_scientific_action(a):
    import ast
    tree=ast.parse(SOURCE.read_text())
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&{'fit_domain','confirm_domain','freeze_dataset','generate','run','Popen'}
    assert a.build()['actions']=={'fit_calls':0,'new_grader_invocations':0,'model_sampling_calls':0,
                                  'generation_calls':0,'scheduler_actions':0}
