import itertools,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops'))
from followup_metrics import *


def test_portfolio_matches_exhaustive_selection():
    a=['a','a','b',None];b=['a','c',None,None]
    pairs=list(itertools.product(itertools.combinations(a,2),itertools.combinations(b,2)))
    obs=[len({x for x in aa+bb if x is not None}) for aa,bb in pairs]
    r=portfolio([a,b],[2,2])
    assert r['distinct8']==pytest.approx(sum(obs)/len(obs))
    assert r['pass8']==pytest.approx(sum(x>0 for x in obs)/len(obs))


def test_identical_disjoint_and_failed():
    assert cross_metrics(['a']*8,['a']*8)['squared_distance']==0
    assert cross_metrics(['a']*8,['b']*8)['squared_distance']==2
    assert cross_metrics([None]*8,['a']*8)['cross_collision'] is None
    assert portfolio([[None]*8,[None]*8],[4,4])['distinct8']==0
    assert portfolio([['a']*8,['b']*8],[4,4])['distinct8']==2


def test_negative_distance_estimate_is_retained():
    assert cross_metrics(['a','b'],['a','b'])['squared_distance']==-1


def test_graph_permutation_preserves_anchors():
    s={'verifier':'graph_coloring','n':3,'partial_colors':[1,None,None]}
    assert coarse_key('graph_coloring:123',s)==coarse_key('graph_coloring:132',s)
    s['partial_colors']=[1,2,None]
    assert coarse_key('graph_coloring:123',s)!=coarse_key('graph_coloring:132',s)


def test_python_pairs_not_algorithms():
    s={'verifier':'python_factor_function','cases':[12,15]}
    assert coarse_key('python_factor:3,3',s)==coarse_key('python_factor:4,5',s)
    assert coarse_key('python_factor:2,3',s)!=coarse_key('python_factor:3,3',s)
    with pytest.raises(ValueError):coarse_key('python_factor:5,3',s)


def test_only_associative_countdown_operations_merge():
    s={'verifier':'countdown'}
    assert coarse_key('countdown:add(1,add(2,3))',s)==coarse_key('countdown:add(add(3,1),2)',s)
    assert coarse_key('countdown:sub(1,sub(2,3))',s)!=coarse_key('countdown:sub(sub(1,2),3)',s)
    with pytest.raises(ValueError):coarse_key('countdown:__import__(1)',s)


def test_coarsening_monotonicity_same_successes():
    s={'verifier':'python_factor_function','cases':[12]}
    fine=['python_factor:3','python_factor:4','python_factor:2',None]
    coarse=[coarse_key(k,s) for k in fine]
    assert collision(coarse)>=collision(fine)
    assert portfolio([coarse],[4])['distinct8']<=portfolio([fine],[4])['distinct8']
    assert portfolio([coarse],[4])['pass8']==portfolio([fine],[4])['pass8']
