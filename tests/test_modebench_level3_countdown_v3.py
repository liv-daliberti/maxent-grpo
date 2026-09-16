"""Three-operand anchors retain exact support and a fixed per-cell sampling law."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
from modebench_level3_countdown_v3 import (
    PRESETS, SUPPORT7_EXCEPTION, build_pool, countdown_modes, preset_for_support,
    proposal_weight, row_identity, target_choices, three_target_statistics,
)
from modebench_level3_countdown_v2 import proposal_weight as old_weight
from make_exact_countdown_mode_data import _canonical_expression_keys
from make_modebench_data import _countdown_expression_map
from oat_drgrpo.math_grader import _canonical_countdown_expression_key


def identities(rows):
    return {row_identity('countdown', row) for row in rows}


def per_cell_prefix(rows, support):
    rows = sorted((row for row in rows if row['answer_mode_count'] == support),
                  key=lambda row: row['level3_support_sampling_index'])
    return [row_identity('countdown', row) for row in rows]


@pytest.mark.parametrize('difficulty', range(4))
def test_every_required_cell_matches_independent_enumeration_and_witnesses(difficulty):
    rows = build_pool('countdown', Counter({support: 1 for support in range(2, 9)}), set(),
                      884000 + difficulty, 'exact', difficulty, 1)
    for row in rows:
        spec = json.loads(row['answer'])
        count, lower, upper, cap, _ = preset_for_support(row['answer_mode_count'], difficulty)
        numbers, target = spec['numbers'], spec['target']
        assert len(numbers) == count and len(set(numbers)) == count
        assert lower <= min(numbers) <= max(numbers) <= upper
        assert target > 0 and target not in numbers and (cap is None or target <= cap)
        exact = _canonical_expression_keys(numbers, target, _countdown_expression_map(numbers)[target])
        modes = countdown_modes(tuple(numbers))[target]
        assert len(exact) == len(modes) == row['answer_mode_count']
        assert exact == {'countdown:' + key for key in modes}
        for key, expression in modes.items():
            assert _canonical_countdown_expression_key(expression, spec) == 'countdown:' + key
        assert row['level3_countdown_support7_exception'] is (difficulty in (0, 1) and row['answer_mode_count'] == 7)


@pytest.mark.parametrize('difficulty', range(4))
def test_requested_counts_and_other_cells_do_not_change_prefixes(difficulty):
    small = build_pool('countdown', Counter({2: 2, 5: 2, 7: 2}), set(), 997200, 'small', difficulty, 1)
    larger = build_pool('countdown', Counter({2: 4, 4: 1, 5: 5, 7: 4, 8: 1}), set(), 997200, 'large', difficulty, 1)
    for support in (2, 5, 7):
        assert per_cell_prefix(small, support) == per_cell_prefix(larger, support)[:2]
    excluded = identities(small)
    fresh = build_pool('countdown', Counter({2: 2, 5: 2, 7: 2}), excluded, 997200, 'fresh', difficulty, 1)
    assert not identities(fresh) & excluded


def test_target_weights_are_uniform_for_three_operands_and_match_frozen_four_operand_law():
    for difficulty, numbers, support in ((0, (2, 3, 5), 5), (1, (8, 12, 17), 5)):
        choices = target_choices(numbers, support, difficulty)
        assert choices and {proposal_weight(numbers, item, difficulty) for item in choices} == {1.}
    for difficulty, support in ((0, 7), (1, 7), (2, 5), (3, 5)):
        numbers = (2, 3, 5, 7) if difficulty != 3 else (5, 8, 11, 17)
        choices = target_choices(numbers, support, difficulty)
        assert choices
        inherited = 1 if difficulty != 3 else 3
        assert [proposal_weight(numbers, item, difficulty) for item in choices] == [old_weight(numbers, item, inherited) for item in choices]


def test_support7_exception_is_declared_from_exact_inventory_not_requested_count():
    inventory = Counter()
    for numbers in combinations(range(2, 25), 3):
        for _target, support, *_ in three_target_statistics(numbers):
            inventory[support] += 1
    assert inventory[7] == SUPPORT7_EXCEPTION['tier0_three_operand_total_identities'] == 23
    assert SUPPORT7_EXCEPTION['tier0_fresh_after_historical_and_prior_pool_exclusions'] == 8
    assert SUPPORT7_EXCEPTION['tier1_three_operand_total_identities'] == 1
    assert SUPPORT7_EXCEPTION['quota_or_depletion_fallback'] is False
    assert preset_for_support(7, 0) == preset_for_support(7, 1) == (4, 2, 18, 72, 1)
    assert preset_for_support(4, 0) == PRESETS[0]
    assert preset_for_support(8, 1) == PRESETS[1]


def test_full_384_128_128_splits_all_tiers_exclude_history_and_all_prior_pools():
    from materialize_modebench_level3 import historical_ids, identity_set, reference_rows
    reference_root = ROOT / 'var/data/modebench_harder_v2_matched_r5/countdown'
    if not reference_root.exists():
        pytest.skip('local historical/reference datasets are unavailable')
    blocked = historical_ids('countdown')
    for path in sorted((ROOT / 'var/data').glob('modebench_level3_calibration*/pools/countdown/*.jsonl')):
        blocked |= identity_set('countdown', [json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    seen_supports = set()
    for difficulty in range(4):
        for split, expected in (('train', 384), ('dev', 128), ('eval', 128)):
            target = Counter(int(row['answer_mode_count']) for row in reference_rows('countdown', split))
            rows = build_pool('countdown', target, blocked, 887100 + difficulty * 1000 + (0 if split == 'train' else 100 if split == 'dev' else 200),
                              f'audit_{difficulty}_{split}', difficulty, 1)
            assert len(rows) == expected
            assert Counter(row['answer_mode_count'] for row in rows) == target
            assert len(identities(rows)) == len(rows) and not identities(rows) & blocked
            for row in rows:
                spec = json.loads(row['answer'])
                modes = countdown_modes(tuple(spec['numbers']))[spec['target']]
                assert len(modes) == row['answer_mode_count']
                for key, witness in modes.items():
                    assert _canonical_countdown_expression_key(witness, spec) == 'countdown:' + key
            blocked |= identities(rows)
            seen_supports |= set(target)
    assert seen_supports == set(range(2, 9))


@pytest.mark.parametrize('domain,target,difficulty,multiplier', [
    ('mathir', Counter({5: 1}), 0, 1),
    ('countdown', Counter({1: 1}), 0, 1),
    ('countdown', Counter({5: -1}), 0, 1),
    ('countdown', Counter({5: 1}), 4, 1),
    ('countdown', Counter({5: 1}), 0, 0),
    ('countdown', Counter({5: 1}), 0, True),
])
def test_invalid_inputs_rejected(domain, target, difficulty, multiplier):
    with pytest.raises(ValueError):
        build_pool(domain, target, set(), 1, 'invalid', difficulty, multiplier)
