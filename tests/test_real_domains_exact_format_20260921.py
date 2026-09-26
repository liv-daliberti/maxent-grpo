import math
import random

import pytest

from summarize_real_domains_exact_format_20260921 import task_metrics, pcmd_bounds


def fixtures(probabilities):
    return [({'accepted': i < 3, 'canonical_key': str(i)}, {'sum_logprob': math.log(p)})
            for i, p in enumerate(probabilities)]


def test_complete_format_distribution_matches_exact_diversity():
    probabilities = [.1, .2, .3, .4]
    metrics = task_metrics(fixtures(probabilities))
    assert metrics['success_lower_bound'] == pytest.approx(.6)
    assert metrics['success_upper_bound'] == pytest.approx(.6)
    for k in (8, 32):
        expected = sum(1-(1-p)**k for p in probabilities[:3])
        assert metrics[f'ed{k}_lower_bound'] == pytest.approx(expected)
        assert metrics[f'ed{k}_upper_bound'] == pytest.approx(expected)


def test_unenumerated_outputs_are_bounded_for_arbitrary_mode_allocations():
    rng = random.Random(12091)
    for _ in range(100):
        weights = [rng.random() for _ in range(4)]
        total = sum(weights)
        enumerated_mass = rng.uniform(.01, .99)
        probabilities = [w/total*enumerated_mass for w in weights]
        residual = 1-enumerated_mass
        allocation = [rng.random() for _ in range(4)]
        allocation = [v/sum(allocation)*residual for v in allocation]
        full = [a+b for a, b in zip(probabilities, allocation)]
        metrics = task_metrics(fixtures(probabilities))
        assert metrics['success_lower_bound'] <= sum(full[:3]) <= metrics['success_upper_bound']
        for k in (8, 32):
            true = sum(1-(1-p)**k for p in full[:3])
            assert metrics[f'ed{k}_lower_bound']-1e-12 <= true <= metrics[f'ed{k}_upper_bound']+1e-12


def test_invalid_probability_partition_rejected():
    with pytest.raises(ValueError, match='exceed one'):
        task_metrics(fixtures([.4, .4, .4, .4]))


def test_conditional_diversity_extrema_and_tiny_mass():
    assert pcmd_bounds([.25, .25], 0) == pytest.approx((.5, .5))
    assert pcmd_bounds([.5, 0], .5) == pytest.approx((0, .5))
    assert pcmd_bounds([1e-300, 1e-300], 0) == pytest.approx((.5, .5))
    assert pcmd_bounds([0, 0], .5) == (None, None)
    assert pcmd_bounds([], 0) == (None, None)
    assert pcmd_bounds([.1], .8) == pytest.approx((0, 0))


def test_conditional_diversity_bounds_include_partial_correct_residual():
    rng = random.Random(910228)
    for count in (2, 3, 5):
        for _ in range(200):
            raw = [rng.random() for _ in range(count)]
            mass = rng.uniform(.001, .99)
            p = [v/sum(raw)*mass for v in raw]
            residual = rng.uniform(0, 1-mass)
            lower, upper = pcmd_bounds(p, residual)
            added_correct = rng.uniform(0, residual)
            allocation = [rng.random() for _ in range(count)]
            q = [v + a/sum(allocation)*added_correct for v, a in zip(p, allocation)]
            actual = 1-sum((v/sum(q))**2 for v in q)
            assert lower-1e-12 <= actual <= upper+1e-12
