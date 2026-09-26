"""Draft five/six-input Python laws; no measured difficulty or target choice.

Every fresh four-case base has one balanced nonsquare semiprime and three
ordinary cases. One/two fixed prime squares add checks while multiplying exact
support by one. Uniform raw proposals restart on base-gcd/history rejection.
Native prompts, witness certification and final row qualification are unchanged.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations, combinations_with_replacement
import json
from math import comb, gcd, prod
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT/directory) not in sys.path:
        sys.path.insert(0, str(ROOT/directory))
import modebench_scale_candidates as original
from make_python_factor_mode_data import _row as certified_python_row, _prompt as native_prompt
from oat_drgrpo.python_modebench import proper_divisors

SCHEMA = 'modebench_scale_python_five_six_case_laws_v1'
DOMAINS = ('python_factors',)
PRIMES = (13, 17, 19, 23)
EXTRAS = ((841,), (841,), (841, 961), (841, 961))
ORDINARY_WINDOW = (48, 1000)
MAX_PROPOSALS_PER_ROW = 100_000
PROFILES = {'python_factors': [
    {'cases': 4 + len(EXTRAS[tier]), 'base_cases': 4, 'ordinary_cases': 3,
     'ordinary_minimum': 48, 'ordinary_maximum': 1000,
     'ordinary_maximum_smallest_factor': 5, 'ordinary_minimum_proper_divisors': 2,
     'semiprime_cases': 1, 'semiprime_least_prime': prime,
     'semiprime_distinct_primes': True, 'semiprime_larger_prime_ratio_maximum': 3,
     'semiprime_maximum': 1000, 'base_gcd': 1,
     'appended_prime_squares': list(EXTRAS[tier]), 'appended_support_multiplier': 1,
     'sampling': 'uniform_four_case_bases_conditioned_on_exact_support_base_gcd_and_projection_exclusions'}
    for tier, prime in enumerate(PRIMES)]}
identity = original.identity


def source_paths():
    return sorted({Path(__file__).resolve(), *original.source_paths(),
                   Path(sys.modules['oat_drgrpo.python_modebench'].__file__).resolve(),
                   Path(sys.modules['make_python_factor_mode_data'].__file__).resolve()})


def _seed(*parts):
    payload = json.dumps([SCHEMA, *parts], sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), 'big')


def _tier(tier):
    if type(tier) is not int or tier not in range(4):
        raise ValueError('fixed harder Python tier0..3 required')
    return PRIMES[tier]


@lru_cache(maxsize=1)
def catalog():
    divisors, groups = {}, defaultdict(list)
    for value in range(ORDINARY_WINDOW[0], ORDINARY_WINDOW[1] + 1):
        ds = proper_divisors(value)
        if len(ds) >= 2 and ds[0] <= 5:
            divisors[value] = tuple(ds)
            groups[len(ds)].append(value)
    return divisors, {count: tuple(values) for count, values in groups.items()}


@lru_cache(maxsize=4, typed=True)
def semiprimes(tier):
    p = _tier(tier)
    return tuple(n for n in range(p*p + 1, 1001)
                 if len(ds := proper_divisors(n)) == 2 and ds[0] == p and p < ds[1] <= 3*p)


@lru_cache(maxsize=256, typed=True)
def proposal_profiles(support):
    if type(support) is not int or support < 2 or support % 2:
        raise ValueError('even exact support required for one nonsquare semiprime')
    _, groups = catalog()
    options, capacities = [], []
    for counts in combinations_with_replacement(sorted(groups), 3):
        if prod(counts) != support//2:
            continue
        repeats = tuple(sorted(Counter(counts).items()))
        capacity = prod(comb(len(groups[count]), repeat) for count, repeat in repeats)
        if capacity:
            options.append(repeats)
            capacities.append(capacity)
    if not options:
        raise ValueError('ordinary triple cannot realize exact support ' + str(support))
    return tuple(options), tuple(capacities)


def blocked_projections(excluded):
    """Expand native historical/current full identities without using scores."""
    result = set()
    for key in excluded:
        if not isinstance(key, (tuple, list)) or len(key) != 2 or key[0] != 'python_factors':
            continue
        cases = key[1]
        if (not isinstance(cases, (tuple, list)) or not 2 <= len(cases) <= 8
                or any(type(n) is not int or not 4 <= n <= 1000 for n in cases)
                or len(set(cases)) != len(cases)):
            raise ValueError('malformed native Python exclusion identity')
        result.update(combinations(sorted(cases), 4))
    return result


def valid_base(base, tier, support=None):
    _tier(tier)
    ordinary, _ = catalog()
    hard = semiprimes(tier)
    return (isinstance(base, tuple) and len(base) == 4
            and all(type(n) is int for n in base) and tuple(sorted(set(base))) == base
            and gcd(*base) == 1 and sum(n in hard for n in base) == 1
            and all(n in ordinary or n in hard for n in base)
            and (support is None or 2*prod(len(ordinary[n]) for n in base if n in ordinary) == support))


def base_from_cases(cases, tier, support=None):
    _tier(tier)
    if (not isinstance(cases, tuple) or len(cases) != 4 + len(EXTRAS[tier])
            or any(type(n) is not int for n in cases)
            or tuple(sorted(set(cases))) != cases or not set(EXTRAS[tier]) <= set(cases)):
        raise ValueError('fixed five/six-case shape changed')
    base = tuple(n for n in cases if n not in EXTRAS[tier])
    if not valid_base(base, tier, support):
        raise ValueError('balanced semiprime/ordinary base law changed')
    return base


@lru_cache(maxsize=256, typed=True)
def _raw_capacity(tier, support):
    _tier(tier)
    _, groups = catalog()
    options, _ = proposal_profiles(support)
    total = 0
    for h in semiprimes(tier):
        p, q = proper_divisors(h)
        for repeats in options:
            for divisor, sign in ((1, 1), (p, -1), (q, -1), (p*q, 1)):
                total += sign * prod(comb(sum(n % divisor == 0 for n in groups[count]), repeat)
                                     for count, repeat in repeats)
    return total


def _capacity(tier, support, projections):
    return _raw_capacity(tier, support) - sum(valid_base(base, tier, support) for base in projections)


def capacity(tier, support, excluded=()):
    _tier(tier)
    proposal_profiles(support)
    return _capacity(tier, support, blocked_projections(excluded))


def _proposal(rng, tier, support):
    """Uniform RAW base; caller rejects and restarts the entire proposal."""
    _, groups = catalog()
    hard = rng.choice(semiprimes(tier))
    options, capacities = proposal_profiles(support)
    ticket = rng.randrange(sum(capacities))
    for repeats, capacity in zip(options, capacities):
        if ticket < capacity:
            break
        ticket -= capacity
    base = [hard]
    for count, repeat in repeats:
        base.extend(rng.sample(groups[count], repeat))
    return tuple(sorted(base))


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    if domain != 'python_factors' or joint_target is not None:
        raise ValueError('Python-only exact marginal support law required')
    _tier(tier)
    if type(seed) is not int or seed < 0 or type(multiplier) is not int or multiplier < 1:
        raise ValueError('nonnegative fixed seed and positive integer multiplier required')
    if any(type(s) is not int or s < 2 or type(n) is not int or n < 0 for s, n in target.items()):
        raise ValueError('exact support quotas required')
    required = Counter({s: n*multiplier for s, n in target.items() if n})
    blocked = set(excluded)
    projections = blocked_projections(blocked)
    rows = []
    for support, count in sorted(required.items()):
        if _capacity(tier, support, projections) < count:
            raise ValueError('insufficient unexcluded harder Python base capacity')
        rng = random.Random(_seed(seed, tier, support, 'python_factors'))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                base = _proposal(rng, tier, support)
                if gcd(*base) != 1 or base in projections:
                    continue
                cases = tuple(sorted(base + EXTRAS[tier]))
                key = ('python_factors', cases)
                if key in blocked:
                    continue
                blocked.add(key)
                projections.update(combinations(cases, 4))
                row = certified_python_row(cases=cases, split_tag=f'{tag}-t{tier}-m{support}',
                                           seed=seed, index=index)
                row.update(scale_candidate_generator=SCHEMA, scale_candidate_tier=tier,
                           scale_candidate_profile=json.dumps(PROFILES[domain][tier], sort_keys=True, separators=(',', ':')),
                           scale_origin_metadata='{}', scale_cell_index=index,
                           scale_python_harder_law_version=SCHEMA)
                rows.append(row)
                break
            else:
                raise RuntimeError(f'harder Python fixed proposal budget exhausted: tier{tier}/support{support}')
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    ids = {identity(domain, row) for row in rows}
    if len(ids) != len(rows) or ids & set(excluded):
        raise RuntimeError('harder Python semantic overlap')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('harder Python exact support histogram drift')
    return rows


def verify_structure(rows):
    for row in rows:
        tier = row.get('scale_candidate_tier')
        _tier(tier)
        spec = json.loads(row['answer'])
        cases = tuple(spec['cases'])
        if type(row.get('answer_mode_count')) is not int or type(spec.get('num_modes')) is not int:
            raise RuntimeError('harder Python exact integer support changed')
        base_from_cases(cases, tier, row['answer_mode_count'])
        if (row.get('modebench_task') != 'python_factor_function'
                or spec.get('verifier') != 'python_factor_function' or spec.get('python_version') != 'factor-v1'
                or spec.get('num_modes') != row['answer_mode_count']
                or row['problem'] != native_prompt(cases)
                or row.get('scale_candidate_generator') != SCHEMA
                or row.get('scale_python_harder_law_version') != SCHEMA
                or row.get('scale_origin_metadata') != '{}'
                or row.get('scale_candidate_profile') != json.dumps(PROFILES['python_factors'][tier], sort_keys=True, separators=(',', ':'))):
            raise RuntimeError('harder Python structural profile/prompt/support changed')
    return {'python_harder_structural_profile': True, 'rows': len(rows)}


def verify_rows(domain, rows):
    if domain != 'python_factors':
        raise ValueError('Python-only harder row audit required')
    structural = verify_structure(rows)
    return {**original.verify_rows(domain, rows), **structural}
