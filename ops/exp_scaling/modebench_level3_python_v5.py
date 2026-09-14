"""Fixed parity and one-larger-factor PythonFactors development proposals.

Two presets fix exactly two odd and two even small-factor cases. One further
preset adds exactly one factor-7 case to defeat bounded 2/3/5 factor chains. All four distinct inputs use
original syntax, prompt, external verifier, and exact product support. Integer
profile tickets give each eligible unordered case set equal proposal mass.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations_with_replacement
import json
from math import comb, prod
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
from make_python_factor_mode_data import _row as certified_row
from oat_drgrpo.python_modebench import proper_divisors

PROFILE = 'python_parity_and_factor7_v5'
GENERATOR = 'modebench_level3_python_candidate_v5'
PRESETS = {
    0: 'small_factor_48_192_anchor',
    1: 'exactly_two_odd_two_even_small_factor_48_192',
    2: 'exactly_two_odd_two_even_small_factor_48_384',
    3: 'one_smallest_factor_7_three_small_factor_48_384',
}
SOFT_WINDOWS = ((48, 192), (48, 192), (48, 384), (48, 384))
HARD_FACTORS = ((), (), (), (7,))
HARD_CASE_COUNTS = (0, 0, 0, 1)
MARKED_CASE_COUNTS = (0, 2, 2, 1)
MIN_DIVISORS = 2
MAX_VALUE = 1000
MAX_PROPOSALS_PER_ROW = 100_000


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest(), 'big')


@lru_cache(maxsize=4)
def catalog(difficulty):
    if type(difficulty) is not int or difficulty not in range(4):
        raise ValueError('difficulty must be an exact integer in 0..3')
    lower, upper = SOFT_WINDOWS[difficulty]
    divisors, by_class = {}, defaultdict(list)
    for value in range(4, MAX_VALUE + 1):
        ds = proper_divisors(value)
        if len(ds) < MIN_DIVISORS:
            continue
        hard = ds[0] in HARD_FACTORS[difficulty]
        soft = lower <= value <= upper and ds[0] <= 5
        if hard or soft:
            divisors[value] = ds
            marked = value % 2 if difficulty in (1, 2) else int(hard)
            by_class[(len(ds), marked)].append(value)
    return divisors, {key: tuple(values) for key, values in sorted(by_class.items())}


@lru_cache(maxsize=512)
def support_profiles(difficulty, support):
    _, groups = catalog(difficulty)
    profiles, capacities = [], []
    for profile in combinations_with_replacement(sorted(groups), 4):
        if prod(key[0] for key in profile) != support or sum(key[1] for key in profile) != MARKED_CASE_COUNTS[difficulty]:
            continue
        repeats = Counter(profile)
        capacity = prod(comb(len(groups[key]), count) for key, count in repeats.items())
        if capacity:
            profiles.append(tuple(repeats.items()))
            capacities.append(capacity)
    if not profiles:
        raise ValueError(f'fixed preset {difficulty} cannot realize support {support}')
    return tuple(profiles), tuple(capacities)


def _blocked_cases(excluded):
    return {tuple(sorted(int(value) for value in identity[1]))
            for identity in excluded if isinstance(identity, tuple) and len(identity) == 2
            and identity[0] == 'python_factors'}


def eligible_cases(cases, difficulty):
    divisors, _ = catalog(difficulty)
    return (len(cases) == len(set(cases)) == 4 and all(value in divisors for value in cases)
            and sum((value % 2 if difficulty in (1, 2) else divisors[value][0] in HARD_FACTORS[difficulty])
                    for value in cases) == MARKED_CASE_COUNTS[difficulty])


def available_capacity(support, difficulty, excluded):
    divisors, _ = catalog(difficulty)
    _, capacities = support_profiles(difficulty, support)
    excluded_count = sum(eligible_cases(cases, difficulty)
                         and prod(len(divisors[value]) for value in cases) == support
                         for cases in _blocked_cases(excluded))
    return sum(capacities) - excluded_count


def _proposal(rng, groups, profiles, capacities):
    ticket = rng.randrange(sum(capacities))
    for profile, capacity in zip(profiles, capacities):
        if ticket < capacity:
            break
        ticket -= capacity
    cases = []
    for key, count in profile:
        cases.extend(rng.sample(groups[key], count))
    return tuple(sorted(cases))


def case_stream(support, excluded, seed, difficulty):
    _, groups = catalog(difficulty)
    profiles, capacities = support_profiles(difficulty, support)
    rng = random.Random(_seed(PROFILE, seed, difficulty, support))
    blocked = _blocked_cases(excluded)
    remaining = available_capacity(support, difficulty, excluded)
    while remaining:
        for _ in range(MAX_PROPOSALS_PER_ROW):
            cases = _proposal(rng, groups, profiles, capacities)
            if cases not in blocked:
                break
        else:
            raise RuntimeError(f'fresh case sampling exhausted for support {support}, difficulty {difficulty}')
        blocked.add(cases)
        remaining -= 1
        yield cases


def build_pool(domain, target, excluded, seed, tag, difficulty, multiplier=4):
    if domain != 'python_factors':
        raise ValueError('this generator supports python_factors only')
    catalog(difficulty)
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError('multiplier must be a positive integer')
    if any(type(support) is not int or support < 2 or type(count) is not int or count < 0
           for support, count in target.items()):
        raise ValueError('support and quotas must be exact nonnegative integers with support >=2')
    required = Counter({support: count * multiplier for support, count in target.items() if count})
    rows = []
    for support, count in sorted(required.items()):
        capacity = available_capacity(support, difficulty, excluded)
        if capacity < count:
            raise RuntimeError(f'support {support} has {capacity} fresh cases at difficulty {difficulty}; requested {count}')
        stream = case_stream(support, excluded, seed, difficulty)
        for index in range(count):
            cases = next(stream)
            row = certified_row(cases=cases, split_tag=f'{tag}-support-{support}', seed=seed, index=index)
            row.update(level3_difficulty=difficulty, level3_generator=GENERATOR,
                       level3_generation_profile=PROFILE, level3_cell_index=index,
                       level3_python_preset=PRESETS[difficulty],
                       level3_soft_case_window=list(SOFT_WINDOWS[difficulty]),
                       level3_hard_smallest_factors=list(HARD_FACTORS[difficulty]),
                       level3_hard_case_count=HARD_CASE_COUNTS[difficulty],
                       level3_exact_odd_count=2 if difficulty in (1, 2) else None,
                       level3_case_max=MAX_VALUE)
            rows.append(row)
    rows.sort(key=lambda row: _seed(PROFILE, seed, 'output_order', json.loads(row['answer'])['cases']))
    identities = {('python_factors', tuple(json.loads(row['answer'])['cases'])) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError('generated identities overlap exclusions or each other')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('generated support histogram differs from requested histogram')
    return rows
