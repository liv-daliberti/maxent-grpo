from pathlib import Path
import sys,json
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import modebench_level3_python_neutral_v5 as g
import modebench_level3_v3_common as common
from materialize_modebench_level3 import modes,identity_set
from collections import Counter

@pytest.mark.parametrize('tier',range(4))
def test_full_reference_support_capacity(tier):
    hist=sum(common.reference_histograms('python_factors').values(),Counter())
    for (support,),count in hist.items(): assert g.available_capacity(support,tier,set())>=4*count

@pytest.mark.parametrize('tier',range(4))
def test_certified_numeric_laws_and_no_solution_hints(tier):
    target=Counter({16:1,80:1,200:1,2240:1})
    rows=g.build_pool('python_factors',target,set(),1345700,'test',tier,1)
    assert modes(rows)==target
    assert len(identity_set('python_factors',rows))==len(rows)
    for r in rows:
        cases=json.loads(r['answer'])['cases']
        assert g.eligible_cases(cases,tier)
        assert all(n%(2 if tier==3 else 3)==0 for n in cases)
        assert 'proper factor' in r['problem'] and 'Only the four listed inputs' in r['problem']
        assert 'n % 2' not in r['problem'] and 'n // 2' not in r['problem'] and 'n=='+str(cases[0]) not in r['problem']
        assert json.loads(r['answer'])['num_externally_certified_modes']==2
