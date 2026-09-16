"""Graph v8 finite support, balanced prefixes, exclusion and publication guards."""
from collections import Counter,defaultdict
from itertools import islice,permutations
import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
import modebench_level3_graph_v8 as graph
import materialize_modebench_level3_graph_v8 as audit
import materialize_modebench_level3_graph_v7 as old

@pytest.mark.parametrize('tier',range(4))
def test_all_supports_keep_original_three_missing_digit_grader(tier):
    rows=graph.build_pool('graph_coloring',Counter({s:2 for s in graph.SUPPORTS}),set(),734105,'test',tier,multiplier=1)
    assert audit.verify_witnesses(rows)==2*sum(graph.SUPPORTS)
    for row in rows:
        spec=json.loads(row['answer'])
        assert spec['n']==5 and sum(c is None for c in spec['partial_colors'])==3
        assert graph.row_identity('graph_coloring',row) in graph.catalogue(tier,row['answer_mode_count'])

@pytest.mark.parametrize('tier',range(3))
def test_first_three_catalogues_retain_exact_old_structural_laws(tier):
    for support in graph.SUPPORTS:assert graph.catalogue(tier,support)==old.catalogue(tier,support)

def test_broader_forest_catalogue_counts_and_actual_acyclic_hidden_structure():
    expected={4:4140,5:720,6:3420,8:3000,9:540,12:1980,18:540}
    for support,count in expected.items():
        catalog=graph.catalogue(3,support);assert len(catalog)==count
        for _,n,edges,text in catalog:
            hidden={i+1 for i,c in enumerate(text) if c=='?'};visible={1,2,3,4,5}-hidden
            assert len([1 for u,v in edges if u in hidden and v in hidden])<=2
            assert tuple(sorted(visible)) in edges
            assert len({text[v-1] for v in visible})==2
            partial=[None if c=='?' else int(c) for c in text]
            assert graph.graph_completion_count(n,edges,partial)==support

@pytest.mark.parametrize('tier',range(4))
def test_all_catalogues_closed_under_global_color_permutations(tier):
    for support in graph.SUPPORTS:
        catalog=graph.catalogue(tier,support)
        for item in catalog:
            domain,n,edges,text=item
            for labels in ((2,3,1),(2,1,3)):
                mapped=''.join('?' if c=='?' else str(labels[int(c)-1]) for c in text)
                assert (domain,n,edges,mapped) in catalog

@pytest.mark.parametrize('tier',range(4))
@pytest.mark.parametrize('support',[4,8,9,12])
def test_color_rounds_and_independent_position_rounds_balance_prefixes(tier,support):
    catalog=graph.catalogue(tier,support);groupkeys={graph.group_key(item) for item in catalog}
    colors={c for c,h in groupkeys};positions={c:{h for c2,h in groupkeys if c2==c} for c in colors}
    color_counts=Counter();position_counts=defaultdict(Counter)
    for item in islice(graph.identity_stream(support,set(),734106,tier),120):
        color,hidden=graph.group_key(item);color_counts[color]+=1;position_counts[color][hidden]+=1
        assert max(color_counts[c] for c in colors)-min(color_counts[c] for c in colors)<=1
        assert max(position_counts[color][h] for h in positions[color])-min(position_counts[color][h] for h in positions[color])<=1

@pytest.mark.parametrize('tier',range(4))
def test_cell_prefix_ignores_quotas_other_cells_and_hashseed(tier):
    small=graph.build_pool('graph_coloring',Counter({4:9,9:3}),set(),734107,'test',tier,multiplier=1)
    large=graph.build_pool('graph_coloring',Counter({4:12,9:7,18:2}),set(),734107,'test',tier,multiplier=1)
    for support in (4,9):
        first=sorted((r for r in small if r['answer_mode_count']==support),key=lambda r:r['level3_cell_index'])
        full=sorted((r for r in large if r['answer_mode_count']==support),key=lambda r:r['level3_cell_index'])
        assert first==full[:len(first)]

@pytest.mark.parametrize('tier',range(4))
def test_fixed_exclusions_and_full_deterministic_depletion(tier):
    catalogue=graph.catalogue(tier,9);blocked=set(sorted(catalogue)[::2])
    first=list(graph.identity_stream(9,blocked,734108,tier))
    assert len(first)==len(set(first)) and set(first)==catalogue-blocked
    assert first==list(graph.identity_stream(9,set(reversed(sorted(blocked))),734108,tier))
    with pytest.raises(RuntimeError,match='no topology/size fallback'):
        graph.build_pool('graph_coloring',Counter({9:len(first)+1}),blocked,734108,'test',tier,multiplier=1)

def test_unequal_group_depletion_removes_only_empty_groups(monkeypatch):
    catalogue=frozenset({('graph_coloring',5,((1,4),),'???12'),('graph_coloring',5,((2,4),),'???12'),('graph_coloring',5,((1,3),),'??1?2'),('graph_coloring',5,((1,4),),'???21')})
    monkeypatch.setattr(graph,'catalogue',lambda tier,support:catalogue)
    actual=list(graph.identity_stream(4,set(),734109,0))
    assert len(actual)==4 and set(actual)==catalogue

@pytest.mark.parametrize('target',[Counter({4:-1}),Counter({7:1}),Counter({4:1.5}),Counter({True:1})])
def test_invalid_quotas_or_supports_rejected(target):
    with pytest.raises(ValueError):graph.build_pool('graph_coloring',target,set(),1,'test',0,multiplier=1)

def test_union_calibration_includes_eval_only_support_and_exact142rows():
    assert audit.calibration_target()==Counter({4:62,5:1,6:28,8:24,9:6,12:21})
    assert sum(audit.calibration_target().values())==142

def test_publication_requires_explicit_registration_pin_before_work(monkeypatch):
    never=Mock(side_effect=AssertionError('unauthorized construction'))
    monkeypatch.setattr(audit,'publish',never)
    with pytest.raises(ValueError,match='explicit registration pin'):
        audit.main(['--materialize-development'])
    never.assert_not_called()

def test_changed_historical_snapshot_rejected_before_generation(monkeypatch):
    never=Mock(side_effect=ValueError('immutable file changed'))
    monkeypatch.setattr(audit.common,'verify_pins',never)
    with pytest.raises(ValueError,match='immutable file changed'):
        audit.verify_generation_exclusions_unchanged({'files_sha256':{},'directory_files':{}})
    never.assert_called_once()

def test_changed_prior_pool_inventory_rejected(monkeypatch):
    monkeypatch.setattr(audit,'verify_snapshot',lambda s:None)
    monkeypatch.setattr(audit,'pool_paths',lambda:[Path('/new/unregistered.jsonl')])
    with pytest.raises(ValueError,match='pool inventory changed'):
        audit.verify_generation_exclusions_unchanged({'candidate_pool_paths':[]})

def test_sampler_explicitly_disclaims_global_uniformity():
    assert graph.SAMPLER_LAW['uniform_over_all_remaining_identities'] is False
    assert graph.SAMPLER_LAW['quota_independent_support_prefix'] is True


def test_certificate_hidden_count_is_computed_from_actual_rows(monkeypatch):
    malformed={'answer':json.dumps({'n':5,'partial_colors':[None,None,1,2,3]})}
    monkeypatch.setattr(graph,'build_pool',lambda *args,**kwargs:[malformed]*142)
    monkeypatch.setattr(audit.materializer,'verify_rows',lambda *args,**kwargs:{'base':True})
    monkeypatch.setattr(audit,'verify_prefix',lambda *args,**kwargs:None)
    monkeypatch.setattr(audit,'verify_witnesses',lambda rows:0)
    with pytest.raises(ValueError,match='candidate checks did not pass'):
        audit.construct(set())
