"""Check sampling statistics and provenance rejection, independent of plot pixels."""
from copy import deepcopy
from itertools import combinations

import pytest

from ops import plot_paper_gpt56_sampling_budget as plot


def test_rarefaction_matches_exhaustive_subsets_with_failures():
    # Two canonical modes plus unsuccessful responses; failed draws stay in k.
    pool = ['a', 'a', 'b', None, None]
    for k in range(len(pool) + 1):
        subsets = list(combinations(pool, k))
        distinct = [len(set(subset) - {None}) for subset in subsets]
        actual = plot.rarefaction([2, 1], len(pool), k)
        assert actual['distinct'] == pytest.approx(sum(distinct) / len(subsets))
        assert actual['pass'] == pytest.approx(sum(v > 0 for v in distinct) / len(subsets))


def test_finite_pool_endpoints_and_zero_success_are_not_extrapolated():
    assert plot.rarefaction([], 64, 64) == {'distinct': 0, 'pass': 0}
    assert plot.rarefaction([31, 2, 1], 64, 64) == {'distinct': 3, 'pass': 1}
    assert plot.rarefaction([31, 2, 1], 64, 1)['distinct'] == 34 / 64
    with pytest.raises(ValueError, match='budget'):
        plot.rarefaction([64], 64, 128)


@pytest.mark.parametrize('counts,n,k', [([0], 64, 1), ([-1], 64, 1), ([65], 64, 1),
                                      ([2.5], 64, 1), ([True], 64, 1), ([1], 0, 0), ([1], 64, -1)])
def test_invalid_count_inventory_is_rejected(counts, n, k):
    with pytest.raises(ValueError):
        plot.rarefaction(counts, n, k)


def prompt_fixture(index=0):
    # Each hypothetical problem has one success and 63 failures. This gives
    # exactly k/64 expected distinct and pass, independent of the implementation.
    return {'level': 2, 'domain': 'python_factors', 'row_index': index,
            'responses': 64, 'mode_counts': [1], 'correct_draws': 1, 'failed_draws': 63,
            'support_reference': {'support_kind': 'certified_lower_bound', 'support_count': 2,
                                  'pair_id': f'problem-{index}'},
            'rarefaction': {str(k): {'distinct': k / 64, 'pass': k / 64} for k in plot.K_GRID}}


def cell_fixture():
    prompts = [prompt_fixture(index) for index in range(16)]
    metrics = {f'rarefaction/{metric}/k{k}': {'estimate': k / 64, 'ci95': [k / 64, k / 64],
                                           'defined_bootstrap_replicates': 20000}
               for k in plot.K_GRID for metric in ('distinct', 'pass')}
    cell = {'counts': {'original': {'prompts': 16, 'responses': 1024, 'correct_draws': 16,
                                   'failed_draws': 1008, 'support_kinds': {'certified_lower_bound': 16}}},
            'original': metrics}
    return prompts, cell


def test_full_cohort_summary_retains_failures_support_and_tail_gain():
    prompts, cell = cell_fixture()
    summary = plot.summarize_cell(prompts, cell, 'original')
    assert summary['points'][0]['distinct']['estimate'] == 1 / 64
    assert summary['tail_gain_32_to_64'] == 0.5
    assert summary['endpoint_pass64']['estimate'] == 1
    assert summary['support_lower_bound']['mean'] == 2
    assert summary['support_lower_bound']['kind'] == 'certified_lower_bound'
    assert len(summary['prompt_ids']) == 16


@pytest.mark.parametrize('mutation', ['counts', 'prompt_metric', 'cell_metric', 'support', 'cohort', 'duplicate'])
def test_statistical_tampering_or_cohort_selection_is_rejected(mutation):
    prompts, cell = cell_fixture()
    if mutation == 'counts':
        prompts[0]['mode_counts'] = [2]
    elif mutation == 'prompt_metric':
        prompts[0]['rarefaction']['32']['distinct'] += 0.1
    elif mutation == 'cell_metric':
        cell['original']['rarefaction/distinct/k32']['estimate'] += 0.1
    elif mutation == 'support':
        prompts[0]['support_reference']['support_kind'] = 'exact'
    elif mutation == 'cohort':
        prompts.pop()
    elif mutation == 'duplicate':
        prompts[-1] = deepcopy(prompts[0])
    with pytest.raises(ValueError):
        plot.summarize_cell(prompts, cell, 'original')


def test_changed_source_hash_is_rejected(tmp_path):
    source = tmp_path / 'raw.json'
    source.write_text('{"verified":true}\n')
    record = plot.binding(source)
    assert plot.bound_file(record) == source
    source.write_text('{"verified":false}\n')
    with pytest.raises(ValueError, match='Changed source binding'):
        plot.bound_file(record)
