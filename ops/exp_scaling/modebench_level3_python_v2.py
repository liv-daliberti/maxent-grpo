"""PythonFactors candidates whose sampling law is independent of cell quotas.

Each output row uses the same fixed-size proposal reservoir. Quotas stop an
otherwise unchanged per-support random stream; they never alter the reservoir,
ranking rule, factorization profile probabilities, or value bounds.
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
from oat_drgrpo.python_modebench import proper_divisors, python_factor_mode_count

PROFILE = 'python_quota_invariant_v2'
UPPER_BOUNDS = (96, 384, 640, 1000)
RESERVOIR_SIZE = 24
TOP_COUNTS = (24, 12, 6, 3)
MAGNITUDE_POWERS = (0.0, 0.75, 1.5, 2.25)
FACTOR_POWERS = (0.0, 0.25, 0.5, 0.75)


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest(), 'big')


@lru_cache(maxsize=4)
def catalog(difficulty):
    if difficulty not in range(4):
        raise ValueError('difficulty must be 0..3')
    upper = UPPER_BOUNDS[difficulty]
    divisors, by_count = {}, defaultdict(list)
    for value in range(6, upper + 1):
        ds = proper_divisors(value)
        if len(ds) >= 2:
            divisors[value] = ds
            by_count[len(ds)].append(value)
    weights = {
        count: tuple((value / upper + 0.15) ** MAGNITUDE_POWERS[difficulty]
                     * divisors[value][0] ** FACTOR_POWERS[difficulty]
                     for value in values)
        for count, values in by_count.items()
    }
    return divisors, dict(by_count), weights


@lru_cache(maxsize=512)
def support_profiles(difficulty, support):
    _, by_count, _ = catalog(difficulty)
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
        raise ValueError(f'bound {UPPER_BOUNDS[difficulty]} cannot realize support {support}')
    return tuple(profiles), tuple(capacities)


def _blocked_cases(excluded):
    result = set()
    for identity in excluded:
        if isinstance(identity, tuple) and len(identity) == 2 and identity[0] == 'python_factors':
            result.add(tuple(int(value) for value in identity[1]))
    return result


def available_capacity(support, difficulty, excluded):
    divisors, _, _ = catalog(difficulty)
    _, capacities = support_profiles(difficulty, support)
    blocked = _blocked_cases(excluded)
    relevant = sum(len(cases) == 4 and len(set(cases)) == 4
                   and all(value in divisors for value in cases)
                   and prod(len(divisors[value]) for value in cases) == support
                   for cases in blocked)
    return sum(capacities) - relevant


def _proposal(rng, difficulty, profiles, capacities):
    _, by_count, weights = catalog(difficulty)
    profile = rng.choices(profiles, weights=capacities, k=1)[0]
    cases = []
    for count, repeats in profile:
        values = by_count[count]
        if difficulty == 0:
            cases.extend(rng.sample(values, repeats))
        else:
            available, available_weights = list(values), list(weights[count])
            for _ in range(repeats):
                index = rng.choices(range(len(available)), weights=available_weights, k=1)[0]
                cases.append(available.pop(index))
                available_weights.pop(index)
    return tuple(sorted(cases))


def _complexity(cases, divisors):
    common = set(divisors[cases[0]])
    for value in cases[1:]:
        common.intersection_update(divisors[value])
    smallest = [divisors[value][0] for value in cases]
    return (not common, min(smallest), sum(smallest), sum(cases))


def case_stream(support: int, excluded: set, seed: int, difficulty: int) -> Iterator[tuple[int, ...]]:
    """Yield a quota-free stream of unique cases for one exact support cell.

    At tier zero, profiles are weighted by their exact combination counts, then
    values are sampled uniformly within profiles. Thus proposals are uniform
    over four-case sets conditional on support, matching Level 1's conditional
    sampling distribution. Harder tiers use frozen value weights and rankings.
    A reservoir contains 24 independently drawn eligible proposals, with repeats
    allowed inside that reservoir; already emitted rows are always excluded.
    """
    divisors, _, _ = catalog(difficulty)
    profiles, capacities = support_profiles(difficulty, support)
    rng = random.Random(_seed(PROFILE, seed, difficulty, support))
    blocked = _blocked_cases(excluded)
    remaining = available_capacity(support, difficulty, excluded)
    while remaining:
        reservoir = []
        for _ in range(RESERVOIR_SIZE):
            for attempt in range(100_000):
                cases = _proposal(rng, difficulty, profiles, capacities)
                if cases not in blocked:
                    reservoir.append(cases)
                    break
            else:
                raise RuntimeError(f'eligible proposal sampling exhausted for support {support}, difficulty {difficulty}')
        if difficulty:
            # Stable sorting leaves random proposal order as the tie breaker.
            reservoir.sort(key=lambda cases: _complexity(cases, divisors), reverse=True)
        selected = reservoir[rng.randrange(TOP_COUNTS[difficulty])]
        blocked.add(selected)
        remaining -= 1
        yield selected


def build_pool(domain, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[dict]:
    """Produce exact support quotas without using quota to tune row difficulty."""
    if domain != 'python_factors':
        raise ValueError('this replacement generator supports python_factors only')
    catalog(difficulty)
    if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier < 1:
        raise ValueError('multiplier must be a positive integer')
    required = Counter({int(support): int(count) * multiplier for support, count in target.items() if count})
    if any(support < 2 or count < 0 for support, count in required.items()):
        raise ValueError('support must be >=2 and counts must be nonnegative')
    rows = []
    for support, count in sorted(required.items()):
        capacity = available_capacity(support, difficulty, excluded)
        if capacity < count:
            raise RuntimeError(f'support {support} has {capacity} fresh cases at difficulty {difficulty}; requested {count}')
        stream = case_stream(support, excluded, seed, difficulty)
        for index in range(count):
            cases = next(stream)
            # Each row receives both original witnesses through the isolated
            # worker, preserving the original verifier and canonical keys.
            row = certified_row(cases=cases, split_tag=f'{tag}-support-{support}',
                                seed=seed, index=index)
            row.update(level3_difficulty=difficulty,
                       level3_generation_profile=PROFILE,
                       level3_cell_index=index,
                       level3_reservoir_size=RESERVOIR_SIZE,
                       level3_selection_top_count=TOP_COUNTS[difficulty])
            rows.append(row)
    # This final display order never consumes a support cell's random stream.
    rows.sort(key=lambda row: _seed(PROFILE, seed, 'output_order',
                                   json.loads(row['answer'])['cases']))
    ids = {('python_factors', tuple(json.loads(row['answer'])['cases'])) for row in rows}
    if len(ids) != len(rows) or ids & excluded:
        raise RuntimeError('generated identities overlap exclusions or each other')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('generated support histogram differs from requested histogram')
    return rows
