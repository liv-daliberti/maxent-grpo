"""Exact support and original executable certification for fresh bridge laws."""
from collections import Counter
import copy
import json
from math import gcd
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_scale_bridge_candidates as bridge
from materialize_modebench_scale import deserialize_cells, union_histogram


@pytest.fixture(scope='module')
def complete_pools():
    protocol_path = ROOT / 'var/data/modebench_scale_v1/protocol.json'
    if not protocol_path.is_file():
        pytest.skip('registered local mode histograms unavailable')
    protocol = json.loads(protocol_path.read_text())
    result = {}
    for domain in bridge.DOMAINS:
        histograms = {split: deserialize_cells(hist) for split, hist in protocol['histograms'][domain].items()}
        target = Counter({cell[0]: count for cell, count in union_histogram(histograms).items()})
        excluded = set()
        for tier in range(4):
            rows = bridge.build_pool(domain, target, excluded, 962101, 'bridge-full-quota-test', tier)
            ids = {bridge.identity(domain, row) for row in rows}
            assert len(ids) == len(rows) and not ids & excluded
            excluded |= ids
            result[(domain, tier)] = (target, rows)
    return result


@pytest.mark.parametrize('domain', bridge.DOMAINS)
@pytest.mark.parametrize('tier', range(4))
def test_full_registered_union_quotas_original_prompt_and_all_witnesses(complete_pools, domain, tier):
    target, rows = complete_pools[(domain, tier)]
    assert Counter(row['answer_mode_count'] for row in rows) == target
    assert sum(target.values()) == {'graph_coloring': 146, 'python_factors': 193}[domain]
    audit = bridge.verify_rows(domain, rows)
    assert audit['rows_verified'] == len(rows)
    assert audit['witnesses_verified'] >= len(rows) * 2
    assert audit['exact_canonical_support'] and audit['unchanged_original_prompt']
    for row in rows:
        assert row['scale_candidate_generator'] == bridge.SCHEMA
        assert json.loads(row['scale_candidate_profile']) == bridge.PROFILES[domain][tier]
        if domain == 'python_factors':
            cases = json.loads(row['answer'])['cases']
            assert gcd(*cases) == 1 and len(cases) == len(set(cases)) == 4


@pytest.mark.parametrize('domain,support', [('graph_coloring', 9), ('python_factors', 180)])
@pytest.mark.parametrize('tier', range(4))
def test_deterministic_cell_prefix_quota_independence_and_exclusion(domain, support, tier):
    args = dict(domain=domain, excluded=set(), seed=962203, tag='prefix', tier=tier)
    one = bridge.build_pool(target=Counter({support: 1}), **args)
    two = bridge.build_pool(target=Counter({support: 2}), **args)
    assert one == sorted(two, key=lambda row: row['scale_cell_index'])[:1]
    assert one == bridge.build_pool(target=Counter({support: 1}), **args)
    wider = bridge.build_pool(target=Counter({support: 2, 4 if domain == 'graph_coloring' else 16: 1}), **args)
    assert [row for row in wider if row['answer_mode_count'] == support] == two
    args['excluded'] = {bridge.identity(domain, row) for row in two}
    fresh = bridge.build_pool(target=Counter({support: 2}), **args)
    assert not args['excluded'] & {bridge.identity(domain, row) for row in fresh}


def test_multiplier_does_not_change_proposal_law():
    args = dict(domain='python_factors', excluded=set(), seed=962210, tag='multiply', tier=2)
    assert bridge.build_pool(target=Counter({32: 1}), multiplier=2, **args) == bridge.build_pool(
        target=Counter({32: 2}), **args)


@pytest.mark.parametrize('field,value', [('answer_mode_count', 8), ('problem', 'changed'),
    ('scale_candidate_profile', '{}'), ('scale_bridge_law_version', 'wrong')])
def test_certifier_rejects_support_prompt_or_law_drift(field, value):
    rows = bridge.build_pool('graph_coloring', Counter({5: 1}), set(), 962301, 'tamper', 0)
    changed = copy.deepcopy(rows)
    changed[0][field] = value
    with pytest.raises(RuntimeError, match='audit failed'):
        bridge.verify_rows('graph_coloring', changed)


def test_python_certifier_rejects_wrong_exceptional_branch_profile():
    rows = bridge.build_pool('python_factors', Counter({32: 1}), set(), 962302, 'tamper', 3)
    changed = copy.deepcopy(rows)
    changed[0]['scale_candidate_tier'] = 0
    changed[0]['scale_candidate_profile'] = json.dumps(bridge.PROFILES['python_factors'][0],sort_keys=True,separators=(',', ':'))
    with pytest.raises(RuntimeError, match='structural law'):
        bridge.verify_rows('python_factors', changed)


@pytest.mark.parametrize('domain', bridge.DOMAINS)
def test_arrow_roundtrip_of_all_four_profiles(domain, tmp_path):
    from datasets import Dataset, DatasetDict, load_from_disk
    support = 5 if domain == 'graph_coloring' else 32
    rows = [row for tier in range(4) for row in bridge.build_pool(domain, Counter({support: 1}), set(), 962311, 'arrow', tier)]
    path = tmp_path / domain
    DatasetDict({'multi_answer': Dataset.from_list(rows)}).save_to_disk(str(path))
    assert [dict(row) for row in load_from_disk(str(path))['multi_answer']] == rows


@pytest.mark.parametrize('changes', [{'tier': True}, {'seed': -1}, {'seed': 1.2}, {'multiplier': False},
    {'target': Counter({4: True})}, {'target': Counter({4.0: 1})}, {'target': Counter({4: -1})},
    {'joint_target': Counter({(4, 'unused'): 1})}, {'domain': 'pantry'}])
def test_invalid_inputs_rejected(changes):
    args = dict(domain='graph_coloring',target=Counter({4: 1}),excluded=set(),seed=1,tag='invalid',tier=0)
    args.update(changes)
    with pytest.raises(ValueError):
        bridge.build_pool(**args)


def test_source_inventory_includes_frozen_certificate_dependencies():
    paths = bridge.source_paths()
    assert Path(bridge.__file__).resolve() in paths
    assert Path(bridge.original.__file__).resolve() in paths
    assert len(paths) == len(set(paths)) and all(path.is_file() for path in paths)
