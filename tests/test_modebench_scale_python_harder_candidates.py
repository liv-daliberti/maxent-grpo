"""Scratch structural tests: native certifiers/execution/models never run."""
import ast
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from itertools import combinations, product
import json
from math import comb, gcd, prod
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ops/exp_scaling'))
import modebench_scale_python_harder_candidates as candidate
from make_python_factor_mode_data import _certified_programs, _prompt
from oat_drgrpo.python_modebench import parse_python_factor_candidate, parse_python_factor_spec

EVIDENCE = json.loads((ROOT/'artifacts/modebench_scale_python_five_six_case_law_capacity_20260912.json').read_text())
HIST = {split: Counter({int(s): n for s, n in h.items()}) for split, h in EVIDENCE['native_histograms'].items()}
POOL = Counter({int(s): n for s, n in EVIDENCE['native_pool_histogram'].items()})


def divisors(n):
    return tuple(d for d in range(2, n) if n % d == 0)


def scratch_row(*, cases, split_tag, seed, index):
    """Native shape/prompt, explicitly without external certification fields."""
    count = prod(len(divisors(n)) for n in cases)
    return {'problem': _prompt(cases), 'answer_mode_count': count,
            'modebench_task': 'python_factor_function', 'answer_mode_split': split_tag,
            'answer': json.dumps({'verifier': 'python_factor_function', 'python_version': 'factor-v1',
                'cases': list(cases), 'num_modes': count, 'source': 'SCRATCH_NOT_CERTIFIED',
                'instance_id': f'{split_tag}-{seed}-{index}'}, sort_keys=True, separators=(',', ':'))}


@pytest.fixture(autouse=True)
def no_native_certification(monkeypatch):
    monkeypatch.setattr(candidate, 'certified_python_row', scratch_row)
    def forbidden(*args, **kwargs):
        raise AssertionError('native execution/qualification forbidden in structural tests')
    monkeypatch.setattr(candidate.original, 'verify_rows', forbidden)


def test_catalog_is_complete_and_exact_under_independent_divisor_enumeration():
    expected = {n: divisors(n) for n in range(48, 1001)
                if len(divisors(n)) >= 2 and divisors(n)[0] <= 5}
    actual, groups = candidate.catalog()
    assert actual == expected
    assert {n for values in groups.values() for n in values} == set(expected)
    for tier, p in enumerate((13, 17, 19, 23)):
        expected_hard = tuple(n for n in range(4, 1001) if len(divisors(n)) == 2
                              and divisors(n)[0] == p and p < divisors(n)[1] <= 3*p)
        assert candidate.semiprimes(tier) == expected_hard
        assert all(prod(divisors(n)) == n for n in expected_hard)
        assert not set(candidate.EXTRAS[tier]) & (set(expected) | set(expected_hard))
        assert all(len(divisors(n)) == 1 for n in candidate.EXTRAS[tier])


def test_all_60_supports_have_exact_profiles_and_worst_case_final_split_capacity():
    assert len(POOL) == 60 and sum(POOL.values()) == 193
    ordinary, groups = candidate.catalog()
    for support in POOL:
        options, capacities = candidate.proposal_profiles(support)
        assert options and all(prod(count**repeat for count, repeat in profile)*2 == support for profile in options)
        assert capacities == tuple(prod(comb(len(groups[count]), repeat) for count, repeat in profile)
                                   for profile in options)
        for tier in range(4):
            cell = EVIDENCE['tiers'][tier]['cells'][str(support)]
            assert candidate.capacity(tier, support) == cell['raw_unique_bases']
            assert cell['available_bases'] >= POOL[support] + HIST['train'][support] + HIST['eval'][support]


@pytest.mark.parametrize('tier', range(4))
def test_exact_capacity_matches_literal_set_enumeration_at_support3600(tier):
    _, groups = candidate.catalog()
    options, _ = candidate.proposal_profiles(3600)
    seen = set()
    for profile in options:
        for parts in product(*(tuple(combinations(groups[count], repeat)) for count, repeat in profile)):
            ordinary = tuple(n for part in parts for n in part)
            for hard in candidate.semiprimes(tier):
                base = tuple(sorted(ordinary + (hard,)))
                if gcd(*base) == 1:
                    seen.add(base)
    assert len(seen) == candidate.capacity(tier, 3600)
    assert seen
    one = next(iter(seen))
    assert candidate.capacity(tier, 3600, {('python_factors', one)}) == len(seen) - 1


def test_raw_profile_tickets_are_uniform_over_unordered_ordinary_sets(monkeypatch):
    # Toy raw groups produce profiles (2,2,8) and (2,4,4), both for full support64.
    groups = {2: (58, 62, 74), 4: (50, 52, 54), 8: (80, 81)}
    options = (((2, 2), (8, 1)), ((2, 1), (4, 2)))
    weights = tuple(prod(comb(len(groups[c]), r) for c, r in profile) for profile in options)
    monkeypatch.setattr(candidate, 'catalog', lambda: ({}, groups))
    monkeypatch.setattr(candidate, 'semiprimes', lambda tier: (221, 247))
    monkeypatch.setattr(candidate, 'proposal_profiles', lambda support: (options, weights))
    class Ticket:
        def __init__(self, ticket, hard):
            self.ticket, self.hard = ticket, hard
        def choice(self, values):
            assert values == (221, 247)
            return self.hard
        def randrange(self, total):
            assert total == sum(weights)
            return self.ticket
        def sample(self, values, count):
            return list(values[:count])
    observed = Counter()
    for hard in (221, 247):
        for ticket in range(sum(weights)):
            base = candidate._proposal(Ticket(ticket, hard), 0, 64)
            profile = tuple(sorted(Counter(next(c for c, values in groups.items() if n in values)
                                             for n in base if n != hard).items()))
            observed[(hard, profile)] += 1
    assert all(observed[(hard, profile)] == weight
               for hard in (221, 247) for profile, weight in zip(options, weights))
    probabilities = {Fraction(weight, sum(weights))*Fraction(1, weight)*Fraction(1, 2)
                     for weight in weights}
    assert probabilities == {Fraction(1, 2*sum(weights))}


@pytest.fixture
def scratch_pools():
    excluded, pools = set(), {}
    for tier in range(4):
        rows = candidate.build_pool('python_factors', POOL, excluded, 193301, 'SCRATCH_DEV', tier)
        pools[tier] = rows
        excluded.update(candidate.identity('python_factors', row) for row in rows)
    return pools, excluded


@pytest.mark.parametrize('tier', range(4))
def test_full_pool_histogram_original_prompt_and_native_static_witness_bounds(scratch_pools, tier):
    rows = scratch_pools[0][tier]
    assert len(rows) == 193 and Counter(r['answer_mode_count'] for r in rows) == POOL
    assert candidate.verify_structure(rows) == {'python_harder_structural_profile': True, 'rows': 193}
    for row in rows:
        spec = json.loads(row['answer'])
        cases = tuple(spec['cases'])
        assert len(cases) == (5 if tier < 2 else 6)
        assert row['problem'] == _prompt(cases)
        assert parse_python_factor_spec(spec) == cases
        assert row['answer_mode_count'] == prod(len(divisors(n)) for n in cases)
        programs = _certified_programs(cases)  # Construction only; no validator or execution.
        assert programs[0] != programs[1]
        for program in programs:
            parsed = parse_python_factor_candidate(program)
            assert len(list(ast.walk(parsed))) == (33 if tier < 2 else 40)
            assert len(program) <= (92 if tier < 2 else 112)


def test_four_pools_then_arbitrary_worst_case_full_final_splits_keep_fresh_bases(scratch_pools):
    pools, initial = scratch_pools
    for tier in range(4):
        # Each alternative allocates all384 train and128 eval rows to this tier.
        # They are scratch alternatives, not selected or published mixtures.
        excluded = set(initial)
        for split in ('train', 'eval'):
            prior_projections = candidate.blocked_projections(excluded)
            rows = candidate.build_pool('python_factors', HIST[split], excluded,
                                        193302 if split == 'train' else 193303, 'SCRATCH_'+split, tier)
            assert Counter(r['answer_mode_count'] for r in rows) == HIST[split]
            ids = {candidate.identity('python_factors', row) for row in rows}
            bases = {candidate.base_from_cases(key[1], tier) for key in ids}
            assert len(bases) == len(rows) and not bases & prior_projections and not ids & excluded
            excluded.update(ids)


@pytest.mark.parametrize('extra', [(49,), (49, 121), (49, 121, 169, 289)])
def test_historical_five_six_eight_case_projections_block_base_not_only_full_identity(extra, monkeypatch):
    base = (50, 58, 62, 323)
    historical = ('python_factors', tuple(sorted(base + extra)))
    assert base in candidate.blocked_projections({historical})
    assert candidate.capacity(1, 32, {historical}) == candidate.capacity(1, 32) - 1
    monkeypatch.setattr(candidate, '_proposal', lambda *args: base)
    monkeypatch.setattr(candidate, 'MAX_PROPOSALS_PER_ROW', 2)
    with pytest.raises(RuntimeError, match='fixed proposal budget exhausted'):
        candidate.build_pool('python_factors', {32: 1}, {historical}, 193304, 'SCRATCH_OLD_PROJECTION', 1)


def test_same_base_different_appended_extras_and_duplicate_history_count_once():
    base = (50, 58, 62, 323)
    keys = {('python_factors', base), ('python_factors', base + (841,)),
            ('python_factors', base + (841, 961))}
    assert candidate.capacity(1, 32, keys) == candidate.capacity(1, 32) - 1
    assert ('python_factors', base + (841,)) != ('python_factors', base + (841, 961))


def test_new_row_blocks_its_base_before_next_row_and_before_next_split(monkeypatch):
    first, second = (50, 58, 62, 323), (50, 58, 74, 323)
    proposals = iter((first, first, second))
    monkeypatch.setattr(candidate, '_proposal', lambda *args: next(proposals))
    rows = candidate.build_pool('python_factors', {32: 2}, set(), 193305, 'SCRATCH_DYNAMIC', 1)
    assert {candidate.base_from_cases(tuple(json.loads(r['answer'])['cases']), 1)
            for r in rows} == {first, second}
    blocked = {candidate.identity('python_factors', row) for row in rows}
    monkeypatch.setattr(candidate, '_proposal', lambda *args: first)
    monkeypatch.setattr(candidate, 'MAX_PROPOSALS_PER_ROW', 2)
    with pytest.raises(RuntimeError, match='fixed proposal budget exhausted'):
        candidate.build_pool('python_factors', {32: 1}, blocked, 193306, 'SCRATCH_NEXT_SPLIT', 1)


def test_base_gcd_rejected_even_when_appended_square_would_make_full_gcd_one(monkeypatch):
    base = (52, 78, 130, 221)
    assert gcd(*base) == 13 and gcd(*base, 841) == 1
    support = prod(len(divisors(n)) for n in base)
    assert not candidate.valid_base(base, 0, support)
    monkeypatch.setattr(candidate, '_proposal', lambda *args: base)
    monkeypatch.setattr(candidate, 'MAX_PROPOSALS_PER_ROW', 2)
    with pytest.raises(RuntimeError, match='fixed proposal budget exhausted'):
        candidate.build_pool('python_factors', {support: 1}, set(), 193307, 'SCRATCH_BASE_GCD', 0)


@pytest.mark.parametrize('tier', range(4))
def test_reproducible_support_streams_and_prefixes(tier):
    args = dict(domain='python_factors', excluded=set(), seed=193308, tag='SCRATCH_STREAM', tier=tier)
    one = candidate.build_pool(target={32: 1}, **args)
    two = candidate.build_pool(target={32: 2}, **args)
    wider = candidate.build_pool(target={32: 2, 3600: 2}, **args)
    assert two == candidate.build_pool(target={32: 2}, **args)
    assert one == sorted(two, key=lambda r: r['scale_cell_index'])[:1]
    assert two == [r for r in wider if r['answer_mode_count'] == 32]
    assert two == candidate.build_pool(target={32: 1}, multiplier=2, **args)


@pytest.mark.parametrize('field,value', [('problem', 'changed'), ('answer_mode_count', 33),
    ('scale_candidate_profile', '{}'), ('scale_python_harder_law_version', 'changed'),
    ('scale_origin_metadata', 'changed'), ('scale_candidate_tier', 3),
    ('modebench_task', 'changed'), ('answer_mode_count', 32.0)])
def test_structural_checks_reject_prompt_support_tier_and_profile_changes(field, value):
    row = candidate.build_pool('python_factors', {32: 1}, set(), 193309, 'SCRATCH_DRIFT', 0)[0]
    row[field] = value
    with pytest.raises((RuntimeError, ValueError)):
        candidate.verify_structure([row])


@pytest.mark.parametrize('change', ['no_square', 'extra_case', 'duplicate', 'boolean', 'wrong_hard_band', 'prime_square_base'])
def test_base_and_case_shape_validation_rejects_structural_drift(change):
    cases = [50, 58, 62, 221, 841]
    if change == 'no_square': cases[-1] = 961
    elif change == 'extra_case': cases.append(961)
    elif change == 'duplicate': cases[1] = 50
    elif change == 'boolean': cases[0] = True
    elif change == 'wrong_hard_band': cases[3] = 323
    else: cases[3] = 169
    with pytest.raises(ValueError):
        candidate.base_from_cases(tuple(sorted(cases)), 0, 32)


@pytest.mark.parametrize('kwargs', [{'tier': True}, {'tier': 4}, {'seed': -1}, {'seed': False},
    {'multiplier': 0}, {'target': {7: 1}}, {'target': {32: True}}, {'target': {32: -1}},
    {'joint_target': {}}, {'domain': 'graph_coloring'}])
def test_invalid_generation_arguments_fail(kwargs):
    values = dict(domain='python_factors', target={32: 1}, excluded=set(), seed=193310, tag='SCRATCH_BAD', tier=0)
    values.update(kwargs)
    with pytest.raises(ValueError): candidate.build_pool(**values)


@pytest.mark.parametrize('cases', [(50, 50, 58, 221), (True, 50, 58, 221), (2, 50, 58, 221), tuple(range(4, 13))])
def test_malformed_native_exclusion_identity_fails_closed(cases):
    with pytest.raises(ValueError): candidate.blocked_projections({('python_factors', cases)})


def test_native_qualification_stays_explicit_and_delegated(monkeypatch):
    rows = candidate.build_pool('python_factors', {32: 1}, set(), 193311, 'SCRATCH_DELEGATION', 0)
    calls = []
    def sentinel(domain, actual):
        calls.append((domain, actual))
        return {'native_gate_delegated_only': True}
    monkeypatch.setattr(candidate.original, 'verify_rows', sentinel)
    result = candidate.verify_rows('python_factors', rows)
    assert result['native_gate_delegated_only'] and result['python_harder_structural_profile']
    assert calls == [('python_factors', rows)]


def test_public_cached_catalog_apis_do_not_accept_boolean_tiers_or_float_support():
    candidate.capacity(1, 32)
    candidate.semiprimes(1)
    with pytest.raises(ValueError): candidate.capacity(True, 32)
    with pytest.raises(ValueError): candidate.semiprimes(True)
    with pytest.raises(ValueError): candidate.proposal_profiles(32.0)
