"""Exact proposal probabilities, certified modes, and frozen stream contracts."""
from collections import Counter
from fractions import Fraction
from itertools import combinations, islice, product
import json
from math import comb, prod
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_python_v3 as previous
import modebench_level3_python_v4 as candidate
from make_python_factor_mode_data import _certified_programs, _prompt
from oat_drgrpo.python_modebench import PYTHON_FACTOR_VERSION, parse_python_factor_spec, python_factor_mode_count
from oat_drgrpo.python_modebench_process import validate_python_factor_function_external


def cases(row):
    return tuple(json.loads(row['answer'])['cases'])


@pytest.mark.parametrize('difficulty', range(4))
def test_frozen_catalogs(difficulty):
    divisors, groups = candidate.catalog(difficulty)
    if difficulty < 2:
        lower, upper = candidate.CASE_WINDOWS[difficulty]
        assert set(divisors) == set(range(lower, upper + 1, 2)) | set(candidate.CORE_VALUES[difficulty])
        assert groups[1] == (4,)
        assert all(value % 2 == 0 and 2 in ds for value, ds in divisors.items())
    else:
        assert (divisors, groups) == previous.catalog(0 if difficulty == 2 else 3)
    assert all(4 <= value <= 1000 for value in divisors)


@pytest.mark.parametrize('difficulty', range(4))
def test_quota_multiplier_and_other_cells_preserve_prefix_with_exclusions(difficulty):
    seed = 695443
    excluded = {('python_factors', next(candidate.case_stream(32, set(), seed, difficulty)))}
    small = candidate.build_pool('python_factors', Counter({32: 1}), excluded, seed, 'prefix', difficulty, 1)
    large = candidate.build_pool('python_factors', Counter({16: 2, 32: 1}), excluded, seed, 'prefix', difficulty, 4)
    cell = sorted((row for row in large if row['answer_mode_count'] == 32), key=lambda row: row['level3_cell_index'])
    assert small == cell[:1]
    assert list(islice(candidate.case_stream(32, excluded, seed, difficulty), 4)) == list(map(cases, cell))


@pytest.mark.parametrize('difficulty', range(4))
def test_exact_support_original_prompt_and_external_witnesses(difficulty):
    target = Counter({16: 1, 24: 1, 72: 1, 140: 1, 180: 1, 420: 1, 1200: 1, 3600: 1})
    rows = candidate.build_pool('python_factors', target, set(), 697771, 'certify', difficulty, 1)
    assert Counter(row['answer_mode_count'] for row in rows) == target
    for row in rows:
        spec = json.loads(row['answer'])
        assert parse_python_factor_spec(spec) == cases(row)
        assert len(cases(row)) == len(set(cases(row))) == 4
        assert row['problem'] == _prompt(cases(row))
        assert python_factor_mode_count(cases(row)) == spec['num_modes'] == row['answer_mode_count']
        validations = [validate_python_factor_function_external(program, spec)
                       for program in _certified_programs(cases(row))]
        assert all(validations) and len({value.canonical_key for value in validations}) == 2
        if difficulty < 2:
            assert validate_python_factor_function_external('lambda n: 2', spec) is not None


@pytest.mark.parametrize('difficulty', range(4))
def test_exact_capacity_and_identity_exclusions(difficulty):
    divisors, groups = candidate.catalog(difficulty)
    expected = Counter()
    for profile in __import__('itertools').combinations_with_replacement(groups, 4):
        expected[prod(profile)] += prod(comb(len(groups[count]), repeats)
                                        for count, repeats in Counter(profile).items())
    assert sum(expected.values()) == comb(len(divisors), 4)
    for support in (16, 24, 180, 420, 3600):
        assert candidate.available_capacity(support, difficulty, set()) == expected[support]
        first = next(candidate.case_stream(support, set(), 69700, difficulty))
        assert candidate.available_capacity(support, difficulty, {('python_factors', first)}) == expected[support] - 1


def test_uniform_integer_tickets_give_each_case_set_exactly_equal_probability():
    groups = {1: (4,), 2: (6, 10, 14, 22), 4: (12, 18)}
    profiles = (((2, 4),), ((1, 1), (2, 2), (4, 1)))
    capacities = (1, 12)
    probabilities = Counter()
    for ticket in range(sum(capacities)):
        profile = profiles[0 if ticket == 0 else 1]
        options = [list(combinations(groups[count], repeats)) for count, repeats in profile]
        for selections in product(*options):
            class ExactRNG:
                def __init__(self):
                    self.remaining = iter(selections)

                def randrange(self, stop):
                    assert stop == 13  # Integer tickets, never floating-point profile weights.
                    return ticket

                def sample(self, population, count):
                    selected = next(self.remaining)
                    assert len(selected) == count and set(selected) <= set(population)
                    return selected
            output = candidate._proposal(ExactRNG(), groups, profiles, capacities)
            probabilities[output] += Fraction(1, 13 * prod(map(len, options)))
    assert len(probabilities) == 13 and set(probabilities.values()) == {Fraction(1, 13)}


def test_count_one_case_participates_without_changing_product_support():
    selected = (4, 16, 48, 60)
    assert python_factor_mode_count(selected) == 1 * 3 * 8 * 10 == 240
    spec = {'verifier': 'python_factor_function', 'python_version': PYTHON_FACTOR_VERSION, 'cases': list(selected)}
    assert validate_python_factor_function_external('lambda n: 2', spec) is not None
    assert all(value in candidate.catalog(0)[0] for value in selected)


def test_exhaustion_and_rejection_budget_fail_closed(monkeypatch):
    groups = {2: (6, 10, 14, 22)}
    monkeypatch.setattr(candidate, 'catalog', lambda difficulty: ({value: (2, value // 2) for value in groups[2]}, groups))
    monkeypatch.setattr(candidate, 'support_profiles', lambda difficulty, support: ((((2, 4),),), (1,)))
    all_cases = (6, 10, 14, 22)
    assert list(candidate.case_stream(16, set(), 1, 0)) == [all_cases]
    assert list(candidate.case_stream(16, {('python_factors', all_cases)}, 1, 0)) == []
    with pytest.raises(RuntimeError, match='requested 2'):
        candidate.build_pool('python_factors', Counter({16: 2}), set(), 1, 'fail', 0, 1)
    monkeypatch.setattr(candidate, 'available_capacity', lambda *args: 1)
    monkeypatch.setattr(candidate, '_proposal', lambda *args: all_cases)
    monkeypatch.setattr(candidate, 'MAX_PROPOSALS_PER_ROW', 2)
    with pytest.raises(RuntimeError, match='sampling exhausted'):
        next(candidate.case_stream(16, {('python_factors', all_cases)}, 1, 0))
