"""Fresh Level-4 graph/Python bridge laws after the original development failed.

The four profiles are prospective structural hypotheses selected from Level-4
DEVELOPMENT evidence only. Every support cell has its own quota-independent
stream. Original prompts, answer syntax, graders and exact canonical support
are retained. These laws do not claim a measured difficulty match.
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
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

import modebench_scale_candidates as original
from make_python_factor_mode_data import _row as certified_python_row
from oat_drgrpo.python_modebench import proper_divisors

SCHEMA = 'modebench_scale_l4_bridge_candidate_laws_v1'
DOMAINS = ('graph_coloring', 'python_factors')
GRAPH_PRESETS = ((5, 3, 'known_first'), (5, 3, 'random'),
                 (6, 3, 'random'), (6, 4, 'random'))
CASE_WINDOW = (48, 1000)
HARD_PRIMES = (None, 7, 11, 13)
MAX_PROPOSALS_PER_ROW = 100_000
PROFILES = {
    'graph_coloring': [
        {'vertices': n, 'hidden_vertices': hidden, 'vertex_order': ordering,
         'planted_colors': 3, 'edge_probability_interval': [.32, .65],
         'isolated_hidden_vertices': True, 'minimum_edges': 1}
        for n, hidden, ordering in GRAPH_PRESETS],
    'python_factors': [
        {'cases': 4, 'case_minimum': CASE_WINDOW[0], 'case_limit': CASE_WINDOW[1],
         'ordinary_maximum_smallest_factor': 5, 'exceptional_smallest_factor': prime,
         'exceptional_cases': 0 if prime is None else 1, 'case_gcd': 1,
         'sampling': 'uniform_four_case_sets_conditioned_on_exact_support_and_profile'}
        for prime in HARD_PRIMES],
}
identity = original.identity


def source_paths():
    return sorted({Path(__file__).resolve(), *original.source_paths(),
                   Path(sys.modules['oat_drgrpo.python_modebench'].__file__).resolve()})


def _seed(*parts):
    payload = json.dumps([SCHEMA, *parts], sort_keys=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest(), 'big')


def _annotate(row, domain, tier, index):
    row = dict(row)
    row.update(scale_candidate_generator=SCHEMA, scale_candidate_tier=tier,
               scale_candidate_profile=json.dumps(PROFILES[domain][tier], sort_keys=True, separators=(',', ':')),
               scale_origin_metadata='{}', scale_cell_index=index,
               scale_bridge_law_version=SCHEMA)
    return row


def _graph_rows(target, excluded, seed, tag, tier):
    n, hidden_count, ordering = GRAPH_PRESETS[tier]
    blocked, rows = set(excluded), []
    for support, count in sorted(target.items()):
        if support not in {4, 5, 6, 8, 9, 12, 18}:
            raise ValueError('graph bridge requires registered exact support')
        rng = random.Random(_seed(seed, tier, support, 'graph_coloring'))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                planted = [rng.randint(1, 3) for _ in range(n)]
                hidden = (set(range(n - hidden_count, n)) if ordering == 'known_first'
                          else set(rng.sample(range(n), hidden_count)))
                partial = [None if v in hidden else color for v, color in enumerate(planted)]
                probability = rng.uniform(.32, .65)
                edges = [[u + 1, v + 1] for u, v in combinations(range(n), 2)
                         if planted[u] != planted[v] and rng.random() < probability]
                # Three hidden vertices need isolates to realize supports9/18.
                if not edges or original.graph_completion_count(n, edges, partial, cap=support) != support:
                    continue
                spec = {'verifier': 'graph_coloring', 'n': n, 'edges': edges,
                        'partial_colors': partial, 'source': SCHEMA,
                        'instance_id': f'{tag}-{seed}-t{tier}-m{support}-{index}',
                        'num_completions': support,
                        'num_solutions': original.graph_completion_count(n, edges, [None] * n)}
                row = {'problem': original._graph_prompt(n, edges, partial),
                       'answer': json.dumps(spec, sort_keys=True), 'modebench_task': 'graph_coloring',
                       'answer_mode_count': support, 'answer_mode_split': tag}
                key = identity('graph_coloring', row)
                if key in blocked:
                    continue
                blocked.add(key)
                rows.append(_annotate(row, 'graph_coloring', tier, index))
                break
            else:
                raise RuntimeError(f'graph bridge exhausted fixed proposal budget: tier{tier}/support{support}')
    return rows


@lru_cache(maxsize=1)
def python_catalog():
    divisors, ordinary, exceptional = {}, defaultdict(list), {p: defaultdict(list) for p in HARD_PRIMES[1:]}
    for value in range(CASE_WINDOW[0], CASE_WINDOW[1] + 1):
        ds = proper_divisors(value)
        if len(ds) < 2:
            continue
        divisors[value] = ds
        if ds[0] <= 5:
            ordinary[len(ds)].append(value)
        if ds[0] in exceptional:
            exceptional[ds[0]][len(ds)].append(value)
    return divisors, {k: tuple(v) for k, v in ordinary.items()}, {
        p: {k: tuple(v) for k, v in groups.items()} for p, groups in exceptional.items()}


@lru_cache(maxsize=512)
def python_proposal_profiles(tier, support):
    _, ordinary, exceptional = python_catalog()
    prime = HARD_PRIMES[tier]
    width = 4 if prime is None else 3
    options, capacities = [], []
    hard_counts = (None,) if prime is None else sorted(exceptional[prime])
    for hard_count in hard_counts:
        for counts in combinations_with_replacement(sorted(ordinary), width):
            if prod(counts) * (hard_count or 1) != support:
                continue
            repeats = tuple(sorted(Counter(counts).items()))
            capacity = prod(comb(len(ordinary[c]), repeat) for c, repeat in repeats)
            if hard_count is not None:
                capacity *= len(exceptional[prime][hard_count])
            if capacity:
                options.append((hard_count, repeats))
                capacities.append(capacity)
    if not options:
        raise ValueError(f'Python bridge tier{tier} cannot realize support{support}')
    return tuple(options), tuple(capacities)


def _python_proposal(rng, tier, support):
    _, ordinary, exceptional = python_catalog()
    options, capacities = python_proposal_profiles(tier, support)
    # Integer tickets weight each profile by its exact number of case sets.
    # Uniform choices within a profile cancel that capacity. GCD/exclusion
    # rejection then conditions this fixed uniform law without ranking rows.
    ticket = rng.randrange(sum(capacities))
    for option, capacity in zip(options, capacities):
        if ticket < capacity:
            break
        ticket -= capacity
    hard_count, repeats = option
    cases = []
    for count, repeat in repeats:
        cases.extend(rng.sample(ordinary[count], repeat))
    if hard_count is not None:
        cases.append(rng.choice(exceptional[HARD_PRIMES[tier]][hard_count]))
    return tuple(sorted(cases))


def _python_rows(target, excluded, seed, tag, tier):
    blocked, rows = set(excluded), []
    for support, count in sorted(target.items()):
        python_proposal_profiles(tier, support)
        rng = random.Random(_seed(seed, tier, support, 'python_factors'))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                cases = _python_proposal(rng, tier, support)
                key = ('python_factors', cases)
                if gcd(*cases) != 1 or key in blocked:
                    continue
                blocked.add(key)
                row = certified_python_row(cases=cases, split_tag=f'{tag}-t{tier}-m{support}',
                                           seed=seed, index=index)
                rows.append(_annotate(row, 'python_factors', tier, index))
                break
            else:
                raise RuntimeError(f'Python bridge exhausted fixed proposal budget: tier{tier}/support{support}')
    return rows


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    if domain not in DOMAINS or type(tier) is not int or tier not in range(4):
        raise ValueError('expected bridge graph/Python domain and integer tier0..3')
    if type(seed) is not int or seed < 0 or type(multiplier) is not int or multiplier < 1:
        raise ValueError('nonnegative integer seed and positive integer multiplier required')
    if joint_target is not None:
        raise ValueError('bridge graph/Python laws do not accept joint histograms')
    if any(type(support) is not int or support < 2 or type(count) is not int or count < 0
           for support, count in target.items()):
        raise ValueError('exact supports and quotas must be nonnegative integers with supports>=2')
    required = Counter({support: count * multiplier for support, count in target.items() if count})
    maker = _graph_rows if domain == 'graph_coloring' else _python_rows
    rows = maker(required, excluded, seed, tag, tier)
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    ids = {identity(domain, row) for row in rows}
    if len(ids) != len(rows) or ids & set(excluded):
        raise RuntimeError('bridge candidate semantic identity overlap')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('bridge exact support histogram drift')
    return rows


def verify_rows(domain, rows):
    if domain not in DOMAINS:
        raise ValueError(domain)
    for row in rows:
        tier = row.get('scale_candidate_tier')
        if type(tier) is not int or tier not in range(4):
            raise RuntimeError('bridge profile audit failed: invalid tier')
        expected = json.dumps(PROFILES[domain][tier], sort_keys=True, separators=(',', ':'))
        if (row.get('scale_candidate_generator') != SCHEMA or row.get('scale_bridge_law_version') != SCHEMA
                or row.get('scale_candidate_profile') != expected):
            raise RuntimeError('bridge profile audit failed: metadata changed')
        spec = json.loads(row['answer'])
        if domain == 'graph_coloring':
            n, hidden, ordering = GRAPH_PRESETS[tier]
            good = spec['n'] == n and spec['partial_colors'].count(None) == hidden and bool(spec['edges'])
            if ordering == 'known_first':
                good = good and all(c is not None for c in spec['partial_colors'][:n-hidden]) and all(
                    c is None for c in spec['partial_colors'][n-hidden:])
        else:
            cases = spec['cases']
            ds = [proper_divisors(n) for n in cases]
            good = (len(cases) == len(set(cases)) == 4 and cases == sorted(cases)
                    and all(CASE_WINDOW[0] <= n <= CASE_WINDOW[1] for n in cases)
                    and all(len(d) >= 2 for d in ds) and gcd(*cases) == 1)
            if good:
                least = [d[0] for d in ds]
                prime = HARD_PRIMES[tier]
                good = (all(p <= 5 for p in least) if prime is None else
                        least.count(prime) == 1 and sum(p <= 5 for p in least) == 3)
        if not good:
            raise RuntimeError('bridge profile audit failed: structural law changed')
    return {**original.verify_rows(domain, rows), 'bridge_structural_profile': True}
