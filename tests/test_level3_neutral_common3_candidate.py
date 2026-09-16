import json
from pathlib import Path
import sys
from itertools import islice
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import modebench_level3_python_neutral_v3 as candidate
import modebench_level3_v3_common as common
from collections import Counter

@pytest.mark.parametrize('tier',range(4))
def test_all_split_support_categories_have_capacity(tier):
    hist=sum((common.reference_histograms('python_factors')[s] for s in common.SPLITS),Counter())
    for (support,),n in hist.items():
        assert candidate.available_capacity(support,tier,set()) >=4*n

@pytest.mark.parametrize('tier',range(4))
def test_real_certified_rows_preserve_support_and_exclusions(tier):
    quota=Counter({56:2,200:2,2240:2})
    rows=candidate.build_pool('python_factors',quota,set(),9031000,'common3_test',tier,1)
    assert Counter(r['answer_mode_count'] for r in rows)==quota
    excluded=set()
    for row in rows:
        spec=json.loads(row['answer']);cases=tuple(spec['cases'])
        assert len(cases)==len(set(cases))==4
        assert all(n%3==0 for n in cases)
        assert candidate.eligible_cases(cases,tier)
        assert spec['num_externally_certified_modes']==2
        assert row['level3_odd_cases']==sum(n%2 for n in cases)
        excluded.add(('python_factors',cases))
        assert 'common factor' not in row['problem'] and 'n % 3' not in row['problem']
    fresh=candidate.build_pool('python_factors',quota,excluded,9031000,'common3_test',tier,1)
    assert not excluded&{('python_factors',tuple(json.loads(r['answer'])['cases'])) for r in fresh}

def test_sampler_is_prefix_stable():
    short=list(islice(candidate._sampler.case_stream(56,set(),991231,0),2))
    long=list(islice(candidate._sampler.case_stream(56,set(),991231,0),5))
    assert short==long[:2]
