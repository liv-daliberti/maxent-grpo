from copy import deepcopy
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import confirm_level3_neutral_fixed_20260912 as c

def shards():
    values=[0,.125,.25,.5]
    return [{'identity':{'seeds':[label]},'information_boundary':{'confirmation_explicitly_authorized':True},
             'prompt_results':[{'row_sha256':'same-row','draws':[{'seed':label,'pass1':v,'pass8':float(v>0),'distinct8':int(v>0)}]}]}
            for label,v in zip(c.LABELS,values)]

def test_independent_draws_are_equal_weighted():
    result=c.combine(shards())[0]
    assert result['pass1']==.21875 and result['pass8']==.75 and result['distinct8']==.75

def test_repeated_random_stream_cannot_count_as_fresh_confirmation():
    rows=shards();rows[3]=deepcopy(rows[0])
    with pytest.raises(ValueError,match='labels'):c.combine(rows)

def test_different_problem_population_cannot_be_pooled():
    rows=shards();rows[2]['prompt_results'][0]['row_sha256']='different-row'
    with pytest.raises(ValueError,match='row mismatch'):c.combine(rows)

def test_development_receipt_cannot_count_as_confirmation():
    rows=shards();rows[2]['information_boundary']['confirmation_explicitly_authorized']=False
    with pytest.raises(ValueError,match='unconfirmed'):c.combine(rows)
