from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_scale_candidates as candidates


@pytest.mark.parametrize('domain', candidates.DOMAINS)
@pytest.mark.parametrize('tier', range(4))
def test_exact_support_original_witnesses_and_cell_prefix(domain, tier):
    support = {'countdown': 5, 'graph_coloring': 6, 'python_factors': 32,
               'mathir': 5, 'pantry': 12}[domain]
    family = 'breakfast_formulation'
    extra_one = {'joint_target': Counter({(support, family): 1})} if domain == 'pantry' else {}
    extra_two = {'joint_target': Counter({(support, family): 2})} if domain == 'pantry' else {}
    arguments = dict(domain=domain, excluded=set(), seed=961100 + tier, tag='test', tier=tier)
    one = candidates.build_pool(target=Counter({support: 1}), **arguments, **extra_one)
    two = candidates.build_pool(target=Counter({support: 2}), **arguments, **extra_two)
    ordered = sorted(two, key=lambda row: row['scale_cell_index'])
    assert one == ordered[:1]
    assert candidates.build_pool(target=Counter({support: 1}), **arguments, **extra_one) == one
    assert Counter(row['answer_mode_count'] for row in two) == Counter({support: 2})
    assert all(not any(key.startswith('level3_') for key in row) for row in two)
    assert all(row['scale_candidate_generator'] == candidates.SCHEMA for row in two)
    audit = candidates.verify_rows(domain, one)
    assert audit['rows_verified'] == 1 and audit['witnesses_verified'] >= 2
    excluded = {candidates.identity(domain, row) for row in one}
    arguments['excluded'] = excluded
    fresh = candidates.build_pool(target=Counter({support: 1}), **arguments, **extra_one)
    assert not excluded & {candidates.identity(domain, row) for row in fresh}


@pytest.mark.parametrize('domain,support,extra_support', [
    ('countdown', 5, 2), ('graph_coloring', 6, 4), ('python_factors', 32, 16),
])
def test_other_support_cells_do_not_change_stream(domain, support, extra_support):
    args = dict(domain=domain, excluded=set(), seed=961231, tag='test', tier=2)
    one = candidates.build_pool(target=Counter({support: 2}), **args)
    other = candidates.build_pool(target=Counter({support: 2, extra_support: 3}), **args)
    assert [row for row in other if row['answer_mode_count'] == support] == one


def test_pantry_other_joint_cells_do_not_change_stream():
    args = dict(domain='pantry', excluded=set(), seed=961231, tag='test', tier=3)
    first_joint = Counter({(12, 'breakfast_formulation'): 2})
    first = candidates.build_pool(target=Counter({12: 2}), joint_target=first_joint, **args)
    second_joint = first_joint + Counter({(12, 'high_fiber_snack'): 1, (8, 'breakfast_formulation'): 1})
    second = candidates.build_pool(target=Counter({12: 3, 8: 1}), joint_target=second_joint, **args)
    assert [row for row in second if (row['answer_mode_count'], row['answer_mode_family'])
            == (12, 'breakfast_formulation')] == first
    assert Counter((row['answer_mode_count'], row['answer_mode_family']) for row in second) == second_joint


def test_multiplier_matches_direct_quota():
    args = dict(domain='pantry', excluded=set(), seed=961291, tag='test', tier=1)
    joint = Counter({(12, 'breakfast_formulation'): 1})
    scaled = candidates.build_pool(target=Counter({12: 1}), joint_target=joint, multiplier=2, **args)
    direct = candidates.build_pool(target=Counter({12: 2}),
                                  joint_target=Counter({(12, 'breakfast_formulation'): 2}), **args)
    assert scaled == direct


@pytest.mark.parametrize('tier', range(4))
def test_graph_every_reference_support_and_registered_structure(tier):
    target = Counter({support: 1 for support in (4, 5, 6, 8, 9, 12, 18)})
    rows = candidates.build_pool('graph_coloring', target, set(), 961333, 'test', tier)
    n, hidden = candidates.GRAPH_PRESETS[tier]
    assert candidates.verify_rows('graph_coloring', rows)['rows_verified'] == 7
    for row in rows:
        spec = json.loads(row['answer'])
        assert spec['n'] == n
        assert spec['partial_colors'].count(None) == hidden


def test_hard_countdown_all_reference_supports():
    target = Counter({support: 1 for support in range(2, 9)})
    rows = candidates.build_pool('countdown', target, set(), 961332, 'test', 3)
    assert candidates.verify_rows('countdown', rows)['rows_verified'] == 7
    for row in rows:
        numbers = json.loads(row['answer'])['numbers']
        assert len(numbers) == len(set(numbers)) == 4 and 16 <= min(numbers) <= max(numbers) <= 192


def test_python_all_frozen_split_supports_have_capacity():
    path = ROOT / 'var/data/modebench_harder_v2_matched_r5/identity.json'
    if not path.exists():
        pytest.skip('local frozen histogram not available')
    data = json.loads(path.read_text())['domains']['python_factors']
    demand = Counter()
    for record in data.values():
        demand.update({int(k): v for k, v in record['answer_mode_count_histogram'].items()})
    for module, origin_tier in candidates.PYTHON_ROUTES:
        for support, count in demand.items():
            assert module.available_capacity(support, origin_tier, set()) >= count


@pytest.mark.parametrize('changes', [
    {'tier': True}, {'seed': -1}, {'seed': 1.5}, {'multiplier': False},
    {'target': Counter({5: True})}, {'target': Counter({5.0: 1})},
    {'target': Counter({5: -1})}, {'joint_target': Counter({(5, 'unused'): 1})},
])
def test_reject_malformed_requests(changes):
    kwargs = dict(domain='mathir', target=Counter({5: 1}), excluded=set(), seed=1, tag='test', tier=0)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        candidates.build_pool(**kwargs)


def test_pantry_requires_exact_joint_target():
    args = dict(domain='pantry', target=Counter({8: 1}), excluded=set(), seed=1, tag='test', tier=0)
    with pytest.raises(ValueError, match='explicit'):
        candidates.build_pool(**args)
    with pytest.raises(ValueError, match='marginal'):
        candidates.build_pool(**args, joint_target=Counter({(9, 'breakfast_formulation'): 1}))
    with pytest.raises(ValueError, match='integer'):
        candidates.build_pool(**args, joint_target=Counter({(8, 'breakfast_formulation'): True}))


def test_exact_support_audit_rejects_tampering():
    rows = candidates.build_pool('graph_coloring', Counter({6: 1}), set(), 98122, 'test', 2)
    corrupted = copy.deepcopy(rows)
    corrupted[0]['answer_mode_count'] = 8
    with pytest.raises(RuntimeError, match='audit failed'):
        candidates.verify_rows('graph_coloring', corrupted)
    corrupted = copy.deepcopy(rows)
    corrupted[0]['problem'] += ' Extra instruction.'
    with pytest.raises(RuntimeError, match='audit failed'):
        candidates.verify_rows('graph_coloring', corrupted)


def test_source_dependency_inventory():
    paths = candidates.source_paths()
    assert Path(candidates.__file__).resolve() in paths
    assert all(path.is_file() for path in paths)
    assert len(paths) == len(set(paths))


@pytest.mark.parametrize('domain', candidates.DOMAINS)
def test_mixed_tier_arrow_roundtrip_preserves_every_row(domain, tmp_path):
    from datasets import Dataset, DatasetDict, load_from_disk
    support = {'countdown': 5, 'graph_coloring': 6, 'python_factors': 32,
               'mathir': 5, 'pantry': 12}[domain]
    kwargs = {'joint_target': Counter({(support, 'breakfast_formulation'): 1})} if domain == 'pantry' else {}
    rows = []
    for tier in range(4):
        rows.extend(candidates.build_pool(domain, Counter({support: 1}), set(),
                                         969121, 'roundtrip', tier, **kwargs))
    expected = json.dumps(rows, sort_keys=True, separators=(',', ':'))
    assert all(isinstance(row['scale_origin_metadata'], str)
               and isinstance(row['scale_candidate_profile'], str) for row in rows)
    path = tmp_path / domain
    DatasetDict({'multi_answer': Dataset.from_list(rows)}).save_to_disk(str(path))
    actual = [dict(row) for row in load_from_disk(str(path))['multi_answer']]
    assert json.dumps(actual, sort_keys=True, separators=(',', ':')) == expected
