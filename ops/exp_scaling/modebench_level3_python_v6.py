"""Fixed minimum-case-band PythonFactors development proposals.

Every preset draws four distinct small-factor cases with its minimum in one
registered numeric band. Integer joint-class profile tickets make every
eligible unordered case set equally likely conditional on exact support.
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

PROFILE = 'python_minimum_case_bands_v6'
GENERATOR = 'modebench_level3_python_candidate_v6'
PRESETS = {
    0: 'minimum_60_69_small_factor_60_192',
    1: 'minimum_60_69_small_factor_60_384',
    2: 'minimum_60_79_small_factor_60_192',
    3: 'minimum_60_89_small_factor_60_384',
}
MINIMUM_BANDS = ((60, 69), (60, 69), (60, 79), (60, 89))
CASE_WINDOWS = ((60, 192), (60, 384), (60, 192), (60, 384))
MIN_DIVISORS = 2
MAX_SMALLEST_FACTOR = 5
MAX_VALUE = 1000
MAX_PROPOSALS_PER_ROW = 100_000


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest(), 'big')


@lru_cache(maxsize=4)
def catalog(difficulty):
    if type(difficulty) is not int or difficulty not in range(4):
        raise ValueError('difficulty must be an exact integer in 0..3')
    lower, upper = CASE_WINDOWS[difficulty]
    band_lower, band_upper = MINIMUM_BANDS[difficulty]
    divisors, by_class = {}, defaultdict(list)
    for value in range(lower, upper + 1):
        ds = proper_divisors(value)
        if len(ds) >= MIN_DIVISORS and ds[0] <= MAX_SMALLEST_FACTOR:
            divisors[value] = ds
            by_class[(len(ds), int(band_lower <= value <= band_upper))].append(value)
    return divisors, {key: tuple(values) for key, values in sorted(by_class.items())}


@lru_cache(maxsize=512)
def support_profiles(difficulty, support):
    _, groups = catalog(difficulty)
    profiles, capacities = [], []
    for profile in combinations_with_replacement(sorted(groups), 4):
        if prod(key[0] for key in profile) != support or not any(key[1] for key in profile):
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
    lower, upper = MINIMUM_BANDS[difficulty]
    return (len(cases) == len(set(cases)) == 4 and all(value in divisors for value in cases)
            and lower <= min(cases) <= upper)


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
                       level3_case_window=list(CASE_WINDOWS[difficulty]),
                       level3_minimum_case_band=list(MINIMUM_BANDS[difficulty]),
                       level3_maximum_smallest_factor=MAX_SMALLEST_FACTOR,
                       level3_case_max=MAX_VALUE)
            rows.append(row)
    rows.sort(key=lambda row: _seed(PROFILE, seed, 'output_order', json.loads(row['answer'])['cases']))
    identities = {('python_factors', tuple(json.loads(row['answer'])['cases'])) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError('generated identities overlap exclusions or each other')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('generated support histogram differs from requested histogram')
    return rows
