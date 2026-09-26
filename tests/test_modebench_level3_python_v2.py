from collections import Counter
from itertools import islice
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
from modebench_level3_python_v2 import (
    RESERVOIR_SIZE, TOP_COUNTS, UPPER_BOUNDS, available_capacity,
    build_pool, case_stream, support_profiles,
)
from make_python_factor_mode_data import _certified_programs
from oat_drgrpo.python_modebench import python_factor_mode_count
from oat_drgrpo.python_modebench_process import validate_python_factor_function_external


def cases(row):
    return tuple(json.loads(row['answer'])['cases'])


@pytest.mark.parametrize('difficulty', range(4))
def test_requested_quota_cannot_change_first_generated_rows(difficulty):
    one = build_pool('python_factors', Counter({32: 1}), set(), 7771, 'quota', difficulty, 1)
    eight = build_pool('python_factors', Counter({32: 8}), set(), 7771, 'quota', difficulty, 1)
    prefix = sorted(eight, key=lambda row: row['level3_cell_index'])[:1]
    assert prefix == one
    assert list(islice(case_stream(32, set(), 7771, difficulty), 8)) == [
        cases(row) for row in sorted(eight, key=lambda row: row['level3_cell_index'])]
    # Adding a preceding support cell cannot shift another cell's RNG stream.
    multiple = build_pool('python_factors', Counter({16: 7, 32: 8}), set(), 7771,
                         'quota', difficulty, 1)
    assert sorted((row for row in multiple if row['answer_mode_count'] == 32),
                  key=lambda row: row['level3_cell_index']) == sorted(eight, key=lambda row: row['level3_cell_index'])
    assert {row['level3_reservoir_size'] for row in eight} == {RESERVOIR_SIZE}
    assert {row['level3_selection_top_count'] for row in eight} == {TOP_COUNTS[difficulty]}


@pytest.mark.parametrize('difficulty', range(4))
def test_support_bounds_exclusions_and_external_witnesses(difficulty):
    target = Counter({16: 1, 32: 1, 144: 1, 420: 1, 1200: 1, 3600: 1})
    rows = build_pool('python_factors', target, set(), 22577, 'first', difficulty, 1)
    assert Counter(row['answer_mode_count'] for row in rows) == target
    excluded = {('python_factors', cases(row)) for row in rows}
    second = build_pool('python_factors', target, excluded, 22577, 'second', difficulty, 1)
    assert not excluded & {('python_factors', cases(row)) for row in second}
    for row in rows:
        spec = json.loads(row['answer'])
        assert len(spec['cases']) == 4 and max(spec['cases']) <= UPPER_BOUNDS[difficulty]
        assert python_factor_mode_count(spec['cases']) == row['answer_mode_count']
        validations = [validate_python_factor_function_external(program, spec)
                       for program in _certified_programs(cases(row))]
        assert all(validations)
        assert len({result.canonical_key for result in validations}) == 2


def test_tier_zero_capacity_counts_real_case_combinations():
    # Support16 means all four inputs each have exactly two proper divisors.
    profiles, capacities = support_profiles(0, 16)
    assert profiles == (((2, 4),),)
    capacity = available_capacity(16, 0, set())
    first = next(case_stream(16, set(), 123, 0))
    assert available_capacity(16, 0, {('python_factors', first)}) == capacity - 1
    assert capacity == sum(capacities)
