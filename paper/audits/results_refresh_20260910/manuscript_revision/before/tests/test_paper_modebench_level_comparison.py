"""Interim level comparisons must pair progress and preserve domain weights."""
from pathlib import Path
import importlib.util
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'ops/exp_scaling/build_paper_modebench_level_comparison.py'
SPEC = importlib.util.spec_from_file_location('modebench_level_comparison_test', SOURCE)
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def cell_rows(domain, seed=43, step=192, value=0.1):
    return [
        dict(level=level, domain=domain, method=method, seed=seed,
             step=step, draw_index=draw, sample_count=8,
             evaluation_kind='fixed_seed_sampled_k_neutral',
             metrics={'any_correct_at_k': value, 'distinct_correct_modes_at_k': value * 2})
        for level in comparison.LEVELS for method in comparison.METHODS for draw in range(4)
    ]


def five_domains():
    return [row for domain in comparison.DOMAINS for row in cell_rows(domain)]


def test_unequal_seed_counts_keep_equal_domain_weight():
    rows = five_domains() + cell_rows('graph_coloring', seed=44, value=0.9)
    result = comparison.build_interim_comparison(rows)
    # Graph mean=.5; four other domain means=.1. Cell pooling would give .2333.
    assert result['means']['level1']['drgrpo']['pass8'] == pytest.approx(.18)
    assert result['coverage_by_domain']['graph_coloring']['n'] == 2
    assert result['eligible_domain_seed_cells'] == 6


def test_latest_checkpoint_requires_all_eight_series_and_four_draws():
    rows = five_domains()
    latest = cell_rows('graph_coloring', step=384, value=0.9)
    latest = [row for row in latest if not (
        row['level'] == 'level1' and row['method'] == 'replay_maxrl' and row['draw_index'] == 3)]
    result = comparison.build_interim_comparison(rows + latest)
    graph = next(cell for cell in result['selected_cells'] if cell['domain'] == 'graph_coloring')
    assert graph['step'] == 192
    assert {row['step'] for level in graph['series'].values() for series in level.values()
            for row in series['draws']} == {192}


def test_selection_uses_latest_availability_even_when_earlier_outcome_is_higher():
    rows = five_domains() + cell_rows('graph_coloring', step=384, value=.01)
    result = comparison.build_interim_comparison(rows)
    graph = next(cell for cell in result['selected_cells'] if cell['domain'] == 'graph_coloring')
    assert graph['step'] == 384
    assert graph['series']['level2']['replay_drgrpo']['means']['pass8'] == .01


def test_missing_domain_cannot_emit_a_five_domain_average():
    rows = [row for row in five_domains() if row['domain'] != 'pantry_plan']
    with pytest.raises(RuntimeError, match='five-domain comparison.*pantry_plan'):
        comparison.build_interim_comparison(rows)


def test_wrong_sampling_contract_cannot_complete_a_checkpoint():
    rows = five_domains()
    row = next(row for row in rows if row['domain'] == 'pantry_plan')
    row['sample_count'] = 1
    row['evaluation_kind'] = 'deterministic_greedy_trace_neutral'
    with pytest.raises(RuntimeError, match='five-domain comparison.*pantry_plan'):
        comparison.build_interim_comparison(rows)


def test_observed_initial_checkpoint_is_labeled_and_never_imputed():
    rows = [dict(row, step=0) if row['domain'] == 'pantry_plan' else row for row in five_domains()]
    result = comparison.build_interim_comparison(rows)
    assert result['initial_only_domains'] == ['pantry_plan']
    assert result['coverage_by_domain']['pantry_plan']['initial_checkpoint_seeds'] == [43]
    assert result['means']['level2']['maxrl']['pass8'] == pytest.approx(.1)
    missing = [row for row in rows if not (row['domain'] == 'pantry_plan' and
               row['level'] == 'level2' and row['method'] == 'maxrl' and row['draw_index'] == 3)]
    with pytest.raises(RuntimeError, match='no common observed checkpoint'):
        comparison.build_interim_comparison(missing)


def test_conflicting_initial_checkpoint_is_not_an_available_fallback():
    rows = [dict(row, step=0) for row in five_domains()]
    duplicate = dict(rows[0], metrics={'any_correct_at_k': .9, 'distinct_correct_modes_at_k': 1.8})
    with pytest.raises(RuntimeError, match='conflicting duplicate'):
        comparison.build_interim_comparison(rows + [duplicate])


def test_conflicting_retries_are_never_selected():
    rows = five_domains()
    duplicate = dict(rows[0], metrics={'any_correct_at_k': .9, 'distinct_correct_modes_at_k': 1.8})
    with pytest.raises(RuntimeError, match='conflicting duplicate'):
        comparison.build_interim_comparison(rows + [duplicate])


def test_identical_retry_is_deduplicated_without_extra_weight():
    rows = five_domains()
    result = comparison.build_interim_comparison(rows + [dict(rows[0])])
    assert result['means']['level2']['maxrl']['pass8'] == pytest.approx(.1)


def test_compact_selected_draw_snapshot_is_checked_against_full_availability():
    rows = five_domains()
    available = [dict(level=level, domain=domain, method=method, seed=seed,
                      complete_steps=[192] if seed == 43 else [])
                 for level in comparison.LEVELS for domain in comparison.DOMAINS
                 for method in comparison.METHODS for seed in comparison.SEEDS]
    result = comparison.build_interim_comparison(rows, available)
    assert result['eligible_domain_seed_cells'] == 5
    for cell in available:
        if cell['domain'] == 'graph_coloring' and cell['seed'] == 43:
            cell['complete_steps'].append(384)
    with pytest.raises(RuntimeError, match='latest common frozen availability'):
        comparison.build_interim_comparison(rows, available)


def test_terminal_progress_ignores_initial_and_partial_training_steps():
    rows = five_domains() + cell_rows('graph_coloring', step=3072, value=.3)
    result = comparison.build_terminal_progress(rows)
    assert result['graph_coloring']['four_arm_matched_seeds'] == [43]
    assert result['graph_coloring']['complete_block'] is False
    assert result['pantry_plan']['terminal_seeds_by_arm']['drgrpo'] == []
    assert result['countdown']['four_arm_n'] == 0
