"""Support-conditioned PythonFactors sampling with fixed small-factor coverage.

Every four-case set is uniform within its support cell and numeric window,
conditional on every value having a proper divisor in {2, 3, 5}. Difficulty
changes only the frozen window, never proposal counts, ranks, or quotas.

Development hypothesis, not a fitting claim: the 11 covered d0 prompts whose
minimum case was at least 48 had pass1=.2898 and pass8=.9773. This exploratory
small group motivates independent fresh pools; no individual outcomes select
rows. The final window stops at 1000 because the unchanged verifier requires it.
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
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'ops', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_python_factor_mode_data import _row as certified_row
from oat_drgrpo.python_modebench import proper_divisors

PROFILE = 'python_small_factor_windows_v3'
CASE_WINDOWS = ((48, 192), (48, 384), (256, 1000), (512, 1000))
MAX_SMALLEST_FACTOR = 5
MAX_PROPOSALS_PER_ROW = 100_000


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest(), 'big')


@lru_cache(maxsize=4)
def catalog(difficulty):
    if difficulty not in range(4):
        raise ValueError('difficulty must be 0..3')
    lower, upper = CASE_WINDOWS[difficulty]
    divisors, by_count = {}, defaultdict(list)
    for value in range(lower, upper + 1):
        ds = proper_divisors(value)
        if len(ds) >= 2 and ds[0] <= MAX_SMALLEST_FACTOR:
            divisors[value] = ds
            by_count[len(ds)].append(value)
    return divisors, {count: tuple(values) for count, values in by_count.items()}


@lru_cache(maxsize=512)
def support_profiles(difficulty, support):
    _, by_count = catalog(difficulty)
    profiles, capacities = [], []
    for profile in combinations_with_replacement(sorted(by_count), 4):
        if prod(profile) != support:
            continue
        repeats = Counter(profile)
        capacity = prod(comb(len(by_count[count]), repeat) for count, repeat in repeats.items())
        if capacity:
            profiles.append(tuple(repeats.items()))
            capacities.append(capacity)
    if not profiles:
        raise ValueError(f'window {CASE_WINDOWS[difficulty]} cannot realize support {support}')
    return tuple(profiles), tuple(capacities)


def _blocked_cases(excluded):
    return {tuple(sorted(int(value) for value in identity[1]))
            for identity in excluded
            if isinstance(identity, tuple) and len(identity) == 2
            and identity[0] == 'python_factors'}


def available_capacity(support, difficulty, excluded):
    divisors, _ = catalog(difficulty)
    _, capacities = support_profiles(difficulty, support)
    relevant = sum(len(cases) == 4 and len(set(cases)) == 4
                   and all(value in divisors for value in cases)
                   and prod(len(divisors[value]) for value in cases) == support
                   for cases in _blocked_cases(excluded))
    return sum(capacities) - relevant


def _proposal(rng, by_count, profiles, capacities):
    # Exact integer tickets make the profile probability proportional to its
    # number of case sets. Uniform within-profile samples then cancel this
    # capacity, giving every eligible four-case set the same proposal mass.
    ticket = rng.randrange(sum(capacities))
    for profile, capacity in zip(profiles, capacities):
        if ticket < capacity:
            break
        ticket -= capacity
    cases = []
    for count, repeats in profile:
        cases.extend(rng.sample(by_count[count], repeats))
    return tuple(sorted(cases))


def case_stream(support: int, excluded: set, seed: int,
                difficulty: int) -> Iterator[tuple[int, ...]]:
    """Uniform fresh case sets from a per-support stream unaffected by quotas."""
    _, by_count = catalog(difficulty)
    profiles, capacities = support_profiles(difficulty, support)
    rng = random.Random(_seed(PROFILE, seed, difficulty, support))
    blocked = _blocked_cases(excluded)
    remaining = available_capacity(support, difficulty, excluded)
    while remaining:
        for _ in range(MAX_PROPOSALS_PER_ROW):
            selected = _proposal(rng, by_count, profiles, capacities)
            if selected not in blocked:
                break
        else:
            raise RuntimeError(f'fresh case sampling exhausted for support {support}, difficulty {difficulty}')
        blocked.add(selected)
        remaining -= 1
        yield selected


def build_pool(domain, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[dict]:
    if domain != 'python_factors':
        raise ValueError('this generator supports python_factors only')
    catalog(difficulty)
    if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier < 1:
        raise ValueError('multiplier must be a positive integer')
    if any(not isinstance(support, int) or isinstance(support, bool) or support < 2
           or not isinstance(count, int) or isinstance(count, bool) or count < 0
           for support, count in target.items()):
        raise ValueError('support and quotas must be nonnegative integers with support >=2')
    required = Counter({support: count * multiplier for support, count in target.items() if count})
    rows = []
    for support, count in sorted(required.items()):
        capacity = available_capacity(support, difficulty, excluded)
        if capacity < count:
            raise RuntimeError(f'support {support} has {capacity} fresh cases at difficulty {difficulty}; requested {count}')
        stream = case_stream(support, excluded, seed, difficulty)
        for index in range(count):
            cases = next(stream)
            row = certified_row(cases=cases, split_tag=f'{tag}-support-{support}',
                                seed=seed, index=index)
            row.update(level3_difficulty=difficulty,
                       level3_generation_profile=PROFILE,
                       level3_cell_index=index,
                       level3_case_min=CASE_WINDOWS[difficulty][0],
                       level3_case_max=CASE_WINDOWS[difficulty][1],
                       level3_max_smallest_factor=MAX_SMALLEST_FACTOR)
            rows.append(row)
    rows.sort(key=lambda row: _seed(PROFILE, seed, 'output_order',
                                   json.loads(row['answer'])['cases']))
    identities = {('python_factors', tuple(json.loads(row['answer'])['cases'])) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError('generated identities overlap exclusions or each other')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('generated support histogram differs from requested histogram')
    return rows
