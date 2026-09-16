"""Independent canonical-count and quota-invariant sampling checks."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
from modebench_level3_countdown_v2 import PRESETS, build_pool, row_identity
from make_exact_countdown_mode_data import _canonical_expression_keys
from make_modebench_data import _countdown_expression_map


def _ordered_cell_ids(rows, support):
    chosen = sorted((row for row in rows if row['answer_mode_count'] == support),
                    key=lambda row: row['level3_support_sampling_index'])
    return [row_identity('countdown', row) for row in chosen]


def test_sampler_prefixes_do_not_depend_on_requested_cell_or_pool_sizes():
    small = build_pool('countdown', Counter({2: 2, 5: 2}), set(), 991201, 'small', 0, multiplier=1)
    larger = build_pool('countdown', Counter({2: 4, 5: 5, 7: 1}), set(), 991201, 'large', 0, multiplier=1)
    for support in (2, 5):
        assert _ordered_cell_ids(small, support) == _ordered_cell_ids(larger, support)[:2]


def test_all_presets_and_rare_supports_match_original_canonical_counter():
    blocked = set()
    for difficulty, (lower, upper, cap) in enumerate(PRESETS):
        rows = build_pool('countdown', Counter({support: 1 for support in range(2, 9)}),
                          blocked, 991250 + difficulty, 'exact_test', difficulty, multiplier=1)
        for row in rows:
            spec = json.loads(row['answer'])
            numbers, target = spec['numbers'], spec['target']
            assert len(numbers) == 4 and lower <= min(numbers) <= max(numbers) <= upper
            assert cap is None or target <= cap
            expressions = _countdown_expression_map(numbers)[target]
            assert len(_canonical_expression_keys(numbers, target, expressions)) == row['answer_mode_count']
        ids = {row_identity('countdown', row) for row in rows}
        assert len(ids) == len(rows) and not ids & blocked
        blocked |= ids
