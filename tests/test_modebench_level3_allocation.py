"""Controlled rounding preserves support/family cells and the global mixture."""
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_allocation as allocation

DOMAINS = ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry')
REPRESENTATIVE_WEIGHTS = (
    (20, 0, 0, 0), (0, 20, 0, 0), (0, 0, 20, 0), (0, 0, 0, 20),
    (0, 5, 5, 10), (7, 2, 4, 7), (5, 5, 5, 5), (1, 1, 1, 17),
    (1, 2, 8, 9), (0, 1, 9, 10), (3, 7, 3, 7), (10, 0, 10, 0),
)


def assert_margins(target, weights, result, seed):
    assert set(result) == set(target)
    assert all(sum(result[key]) == count for key, count in target.items())
    assert tuple(sum(row[tier] for row in result.values()) for tier in range(4)) == allocation.hamilton_totals(sum(target.values()), weights, seed)
    for key, count in target.items():
        for tier, weight in enumerate(weights):
            assert result[key][tier] in (count * weight // 20, (count * weight + 19) // 20)
            assert type(result[key][tier]) is int


@pytest.mark.parametrize('weights', REPRESENTATIVE_WEIGHTS)
def test_rare_cells_preserve_both_margins(weights):
    target = Counter({(index, 'family'): count for index, count in enumerate((0, 1, 1, 1, 2, 2, 3, 7, 19, 20))})
    result = allocation.allocate_cells(target, weights, 6318000)
    assert_margins(target, weights, result, 6318000)
    assert result[(0, 'family')] == (0, 0, 0, 0)


@pytest.mark.parametrize('domain', DOMAINS)
@pytest.mark.parametrize('split', ('train', 'dev', 'eval'))
def test_actual_reference_cells_preserve_exact_global_mixture(domain, split):
    from datasets import load_from_disk
    path = ROOT / 'var/data/modebench_harder_v2_matched_r5' / domain / split
    if not path.is_dir():
        pytest.skip('local Level 2 reference dataset is not available')
    subset = 'train' if split == 'train' else 'multi_answer'
    rows = [dict(row) for row in load_from_disk(str(path))[subset]]
    target = Counter((int(row['answer_mode_count']), row['answer_mode_family']) if domain == 'pantry'
                     else (int(row['answer_mode_count']),) for row in rows)
    assert sum(target.values()) == (384 if split == 'train' else 128)
    for weights in REPRESENTATIVE_WEIGHTS:
        result = allocation.allocate_cells(target, weights, 6391701)
        assert_margins(target, weights, result, 6391701)


def test_all_singleton_cells_follow_global_tiers():
    target = Counter({(index,): 1 for index in range(128)})
    result = allocation.allocate_cells(target, (0, 5, 5, 10), 6391701)
    assert_margins(target, (0, 5, 5, 10), result, 6391701)
    assert tuple(sum(row[tier] for row in result.values()) for tier in range(4)) == (0, 32, 32, 64)


def test_deterministic_with_input_order_and_cache_mutation_isolation():
    target = Counter({(5, 'b'): 3, (5, 'a'): 7, (10, 'a'): 1, (12, 'b'): 19})
    weights, seed = (7, 2, 4, 7), 77
    allocation.clear_allocation_cache()
    first = allocation.allocate_cells(target, weights, seed)
    allocation.clear_allocation_cache()
    reordered = Counter(dict(reversed(list(target.items()))))
    assert allocation.allocate_cells(reordered, weights, seed) == first
    assert allocation.allocation_cache_info().hits == 0
    repeated = allocation.allocate_cells(target, weights, seed)
    assert repeated == first
    assert allocation.allocation_cache_info().hits == 1
    repeated[(5, 'b')] = (100, 0, 0, 0)
    assert allocation.allocate_cells(target, weights, seed) == first


def test_empty_and_zero_targets():
    assert allocation.allocate_cells(Counter(), (5, 5, 5, 5), 1) == {}
    assert allocation.allocate_cells(Counter({(5,): 0}), (5, 5, 5, 5), 1) == {(5,): (0, 0, 0, 0)}


@pytest.mark.parametrize('target,weights,seed', [
    (Counter({(5,): -1}), (5, 5, 5, 5), 1),
    (Counter({(5,): 1.5}), (5, 5, 5, 5), 1),
    (Counter({(5,): True}), (5, 5, 5, 5), 1),
    (Counter({'wrong': 1}), (5, 5, 5, 5), 1),
    (Counter({(5,): 1}), (5, 5, 5), 1),
    (Counter({(5,): 1}), (5, 5, 5, 6), 1),
    (Counter({(5,): 1}), (-1, 7, 7, 7), 1),
    (Counter({(5,): 1}), (True, 6, 6, 7), 1),
    (Counter({(5,): 1}), (1., 6, 6, 7), 1),
    (Counter({(5,): 1}), (5, 5, 5, 5), -1),
])
def test_invalid_inputs_rejected(target, weights, seed):
    with pytest.raises(ValueError):
        allocation.allocate_cells(target, weights, seed)


def test_solver_failure_never_changes_requested_weights_or_bounds(monkeypatch):
    allocation.clear_allocation_cache()
    monkeypatch.setattr(allocation, 'milp', lambda **kwargs: SimpleNamespace(success=False, x=None, status=2, message='infeasible'))
    with pytest.raises(RuntimeError, match='prescribed global Hamilton margins'):
        allocation.allocate_cells(Counter({(3,): 3, (4,): 7}), (1, 3, 7, 9), 191)
