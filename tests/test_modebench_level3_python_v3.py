from collections import Counter
from itertools import islice
import json
from math import comb
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
from modebench_level3_python_v3 import (
    CASE_WINDOWS, available_capacity, build_pool, case_stream, catalog,
    support_profiles,
)
from make_python_factor_mode_data import _certified_programs
from oat_drgrpo.python_modebench import parse_python_factor_spec, proper_divisors, python_factor_mode_count
from oat_drgrpo.python_modebench_process import validate_python_factor_function_external


def cases(row):
    return tuple(json.loads(row['answer'])['cases'])


@pytest.mark.parametrize('difficulty', range(4))
def test_quotas_do_not_change_stream_prefix_or_other_cells(difficulty):
    one = build_pool('python_factors', Counter({32: 1}), set(), 677771, 'quota', difficulty, 1)
    eight = build_pool('python_factors', Counter({32: 8}), set(), 677771, 'quota', difficulty, 1)
    ordered = sorted(eight, key=lambda row: row['level3_cell_index'])
    assert one == ordered[:1]
    assert list(islice(case_stream(32, set(), 677771, difficulty), 8)) == list(map(cases, ordered))
    multiple = build_pool('python_factors', Counter({16: 7, 32: 8}), set(), 677771,
                          'quota', difficulty, 1)
    assert sorted((row for row in multiple if row['answer_mode_count'] == 32),
                  key=lambda row: row['level3_cell_index']) == ordered


@pytest.mark.parametrize('difficulty', range(4))
def test_exact_support_coverage_exclusions_and_original_witnesses(difficulty):
    target = Counter({16: 1, 32: 1, 144: 1, 420: 1, 1200: 1, 3600: 1})
    first = build_pool('python_factors', target, set(), 672577, 'first', difficulty, 1)
    excluded = {('python_factors', cases(row)) for row in first}
    second = build_pool('python_factors', target, excluded, 672577, 'second', difficulty, 1)
    assert Counter(row['answer_mode_count'] for row in first) == target
    assert not excluded & {('python_factors', cases(row)) for row in second}
    lower, upper = CASE_WINDOWS[difficulty]
    for row in first:
        spec = json.loads(row['answer'])
        assert parse_python_factor_spec(spec) == cases(row)
        assert len(cases(row)) == 4 and len(set(cases(row))) == 4
        assert all(lower <= n <= upper <= 1000 and proper_divisors(n)[0] <= 5 for n in cases(row))
        assert python_factor_mode_count(cases(row)) == row['answer_mode_count']
        validations = [validate_python_factor_function_external(program, spec)
                       for program in _certified_programs(cases(row))]
        assert all(validations)
        assert len({result.canonical_key for result in validations}) == 2


@pytest.mark.parametrize('difficulty', range(4))
def test_capacity_is_exact_and_exclusion_removes_one_case_set(difficulty):
    _, groups = catalog(difficulty)
    profiles, capacities = support_profiles(difficulty, 16)
    assert profiles == (((2, 4),),)
    assert capacities == (comb(len(groups[2]), 4),)
    first = next(case_stream(16, set(), 72321, difficulty))
    capacity = available_capacity(16, difficulty, set())
    assert available_capacity(16, difficulty, {('python_factors', first)}) == capacity - 1


def test_all_reference_split_cells_have_fresh_capacity_after_real_history():
    from materialize_modebench_level3 import historical_ids, identity_set, modes, reference_rows
    reference = ROOT / 'var/data/modebench_harder_v2_matched_r5/python_factors'
    if not reference.exists():
        pytest.skip('local reference datasets are unavailable')
    blocked = historical_ids('python_factors')
    for root in (ROOT / 'var/data').glob('modebench_level3_calibration*'):
        for path in (root / 'pools/python_factors').glob('*.jsonl'):
            blocked |= identity_set('python_factors', [json.loads(line) for line in path.read_text().splitlines()])
    demand = Counter()
    for split in ('train', 'dev', 'eval'):
        demand.update(modes(reference_rows('python_factors', split)))
    # Four 128-row pilot pools plus fresh384 train/fresh128 eval: even assigning
    # all1024 rows to any one numeric window leaves ample cellwise capacity.
    demand.update({support: 3 * count for support, count in
                   modes(reference_rows('python_factors', 'dev')).items()})
    assert sum(demand.values()) == 1024
    for difficulty in range(4):
        for support, count in demand.items():
            assert available_capacity(support, difficulty, blocked) >= count
            fresh = list(islice(case_stream(support, blocked, 674002, difficulty), count))
            assert len(fresh) == count and len(set(fresh)) == count
            assert not {('python_factors', row) for row in fresh} & blocked
            assert all(python_factor_mode_count(row) == support for row in fresh)
