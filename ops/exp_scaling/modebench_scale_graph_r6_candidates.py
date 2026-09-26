"""Graph r6 topology laws: r5's construction, graded finely enough to land a rung.

r5 settled the structural question for this domain. Its four tiers carry
heterogeneity gaps of 0.311, 0.285, 0.293 and 0.286 against a target of 0.282,
where every r2-derived design sits at 0.40 or worse, and its dead fractions fall
to 0.20 where r2-derived designs run 0.38 to 0.44. A single rung at the target
pass@1 carrying an r5 gap satisfies the pass@8 gate outright; carrying an r2 gap
it fails by a factor of twenty. What r5 lacked was the level. Its two knobs were
a fourth hidden vertex, which moved pass@1 from 0.38 to 0.10 in one step and
straight past the 0.208 target, and shown-shown distractor edges, which were
inert at +0.003.

Mixing cannot close that, because mixing does not preserve gaps: blending an easy
population with a hard one adds between-tier heterogeneity, and the best r5
mixture inflates the gap by about 0.10 over its component mean. The target needs
a SINGLE rung near pass@1 0.21.

r6 is therefore a level-finding law, not a redesign. It holds four hidden
vertices and grades difficulty by two structural statistics of the skeleton,
both conditioned on support so the registered histogram is untouched:

  forced hidden vertices  a hidden vertex whose colour domain is a single colour
                          contributes a factor of exactly 1 to the support, so it
                          costs the model a digit without changing the answer
                          count. Fewer free vertices is easier.
  hidden-edge band        the number of hidden-hidden edges, which sets how much
                          constraint propagation the model must do.

Not every (support, forced, band) cell is populated: forced=2 exists only for
supports 4, 6 and 9, and the high-edge band is empty at supports 9, 12 and 18.
RESOLUTION below is the declared fallback, applied per cell and recorded in each
row's profile, so a diluted rung is visible in the data rather than silent.

Tiers 1 and 2 are the two candidates for the 0.21 level. Tier 3 repeats r5's
four-free-vertex configuration as a control against a measured 0.0993 / 0.2734.
No ordering is claimed; the pilot measures it. Prompt renderer, answer syntax,
canonicalizer and verifier are unchanged.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import modebench_scale_candidates as original

SCHEMA = 'modebench_scale_graph_r6_structural_grading_laws_v1'
DOMAINS = ('graph_coloring',)
SUPPORTS = (4, 5, 6, 8, 9, 12, 18)
SHOWN = 3
HIDDEN = 4
SHOWN_PAIRS = tuple(combinations(range(SHOWN), 2))
HIDDEN_PAIRS = tuple(combinations(range(HIDDEN), 2))
COLORINGS = tuple(product((1, 2, 3), repeat=HIDDEN))
MAX_PROPOSALS_PER_ROW = 100_000
BANDS = {'low': (0, 2), 'high': (3, len(HIDDEN_PAIRS)), 'any': (0, len(HIDDEN_PAIRS))}
# (forced_hidden_vertices, hidden_edge_band)
TIERS = ((2, 'any'), (1, 'low'), (1, 'high'), (0, 'any'))


def forced_count(domains):
    """Hidden vertices pinned to a single colour; each contributes a factor of 1."""
    return sum(1 for value in domains if value.bit_count() == 1)


def completion_count(edge_mask, domains):
    """Pure finite constraint count, not a model-answer grader."""
    return sum(all(domains[i] & (1 << (colors[i] - 1)) for i in range(HIDDEN))
               and all(colors[a] != colors[b] for bit, (a, b) in enumerate(HIDDEN_PAIRS)
                       if edge_mask & (1 << bit)) for colors in COLORINGS)


@lru_cache(maxsize=1)
def catalog():
    groups = {support: [] for support in SUPPORTS}
    for edge_mask in range(1 << len(HIDDEN_PAIRS)):
        for domains in product(range(1, 8), repeat=HIDDEN):
            support = completion_count(edge_mask, domains)
            if support in groups:
                groups[support].append((edge_mask, domains))
    missing = [support for support, values in groups.items() if not values]
    if missing:
        raise RuntimeError(f'Graph r6 cannot realize exact registered supports {missing}')
    return {support: tuple(values) for support, values in groups.items()}


def _select(support, forced, band):
    low, high = BANDS[band]
    return tuple((mask, domains) for mask, domains in catalog()[support]
                 if forced_count(domains) == forced and low <= mask.bit_count() <= high)


@lru_cache(maxsize=64)
def resolve(tier, support):
    """Declared fallback: drop the band first, then move to the nearest populated
    forced count, preferring the lower one on a tie. Returns the options actually
    used plus a record of what was relaxed."""
    if type(tier) is not int or tier not in range(4):
        raise ValueError('Graph r6 requires integer tier0..3')
    if type(support) is not int or support not in SUPPORTS:
        raise ValueError('Graph r6 requires a registered exact support')
    forced, band = TIERS[tier]
    options = _select(support, forced, band)
    if options:
        return options, {'forced': forced, 'band': band, 'relaxed': None}
    options = _select(support, forced, 'any')
    if options:
        return options, {'forced': forced, 'band': 'any', 'relaxed': 'band'}
    for distance in range(1, HIDDEN + 1):
        for candidate in (forced - distance, forced + distance):
            if 0 <= candidate <= HIDDEN:
                options = _select(support, candidate, 'any')
                if options:
                    return options, {'forced': candidate, 'band': 'any', 'relaxed': 'forced_and_band'}
    raise RuntimeError(f'Graph r6 has no skeleton for tier{tier}/support{support}')


def _profile(tier, support):
    forced, band = TIERS[tier]
    _, used = resolve(tier, support)
    return {'answer_mode_count': support, 'vertices': SHOWN + HIDDEN, 'hidden_vertices': HIDDEN,
            'shown_colors': [1, 2, 3], 'known_known_edges': 0,
            'requested_forced_hidden_vertices': forced, 'requested_hidden_edge_band': band,
            'realized_forced_hidden_vertices': used['forced'],
            'realized_hidden_edge_band': used['band'], 'fallback_applied': used['relaxed'],
            'hidden_domains': 'all_nonempty_subsets_of_colors_1_2_3',
            'anchor_edges': 'one_edge_for_each_forbidden_hidden_color',
            'graded': 'per_support_structural_statistics_of_the_skeleton',
            'vertex_order': f'uniform_permutation_of_all_{SHOWN + HIDDEN}_vertices',
            'sampling': 'uniform_over_skeletons_conditioned_on_exact_support_forced_count_and_edge_band',
            'difficulty_ordering': 'prospective_level_finding_no_monotonicity_claim'}


PROFILES = {'graph_coloring': [
    {str(support): _profile(tier, support) for support in SUPPORTS} for tier in range(4)]}
identity = original.identity


def source_paths():
    return sorted({Path(__file__).resolve(), *original.source_paths()})


def _seed(*parts):
    value = json.dumps([SCHEMA, *parts], sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(value).digest(), 'big')


def instantiate(edge_mask, domains, permutation):
    """Relabel one abstract constraint graph; no dataset or verifier calls."""
    size = SHOWN + HIDDEN
    if (type(edge_mask) is not int or edge_mask not in range(1 << len(HIDDEN_PAIRS))
            or len(domains) != HIDDEN or any(type(d) is not int or d not in range(1, 8) for d in domains)
            or len(permutation) != size or any(type(v) is not int for v in permutation)
            or set(permutation) != set(range(1, size + 1))):
        raise ValueError('invalid Graph r6 skeleton or vertex permutation')
    edges = [(SHOWN + a, SHOWN + b) for bit, (a, b) in enumerate(HIDDEN_PAIRS) if edge_mask & (1 << bit)]
    edges += [(color - 1, SHOWN + i) for i, allowed in enumerate(domains) for color in (1, 2, 3)
              if not allowed & (1 << (color - 1))]
    edges = sorted(sorted((permutation[a], permutation[b])) for a, b in edges)
    partial = [None] * size
    for base, color in enumerate((1, 2, 3)):
        partial[permutation[base] - 1] = color
    return {'n': size, 'edges': edges, 'partial_colors': partial}


def structural_signature(spec):
    """Recover the exact abstract law from a labeled graph, or reject it."""
    size = SHOWN + HIDDEN
    if type(spec.get('n')) is not int or spec['n'] != size:
        raise ValueError(f'Graph r6 requires {size} vertices')
    partial, edges = spec.get('partial_colors'), spec.get('edges')
    if (not isinstance(partial, list) or len(partial) != size
            or any(c is not None and (type(c) is not int or c not in (1, 2, 3)) for c in partial)
            or sorted(c for c in partial if c is not None) != [1, 2, 3]
            or not isinstance(edges, list)
            or any(not isinstance(e, list) or len(e) != 2 or any(type(v) is not int for v in e)
                   or not 1 <= e[0] < e[1] <= size for e in edges)
            or edges != sorted(edges) or len({tuple(e) for e in edges}) != len(edges)):
        raise ValueError('Graph r6 requires exact shown anchors and a canonical simple graph')
    hidden_ids = [i + 1 for i, color in enumerate(partial) if color is None]
    if len(hidden_ids) != HIDDEN:
        raise ValueError('Graph r6 hidden count changed')
    domains, edge_mask = [7] * HIDDEN, 0
    for a, b in edges:
        if a in hidden_ids and b in hidden_ids:
            edge_mask |= 1 << HIDDEN_PAIRS.index((hidden_ids.index(a), hidden_ids.index(b)))
        elif a not in hidden_ids and b not in hidden_ids:
            raise ValueError('Graph r6 has no known-known edges')
        else:
            h, other = (a, b) if a in hidden_ids else (b, a)
            domains[hidden_ids.index(h)] &= ~(1 << (partial[other - 1] - 1))
    if not all(domains):
        raise ValueError('Graph r6 hidden domains must be nonempty')
    support = completion_count(edge_mask, tuple(domains))
    if support not in SUPPORTS:
        raise ValueError('Graph r6 support outside the registered histogram')
    return edge_mask, tuple(domains), support


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    if type(tier) is not int or tier not in range(4):
        raise ValueError('Graph r6 requires integer tier0..3')
    if (domain != 'graph_coloring' or type(seed) is not int or seed < 0
            or type(multiplier) is not int or multiplier < 1 or joint_target is not None
            or any(type(s) is not int or s not in SUPPORTS or type(n) is not int or n < 0
                   for s, n in target.items())):
        raise ValueError('Graph r6 requires exact support quotas, a fixed seed, and no joint target')
    size = SHOWN + HIDDEN
    required = Counter({s: n * multiplier for s, n in target.items() if n})
    blocked, rows = set(excluded), []
    for support, count in sorted(required.items()):
        options, _ = resolve(tier, support)
        rng = random.Random(_seed(seed, tier, support, domain))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                edge_mask, domains = options[rng.randrange(len(options))]
                permutation = list(range(1, size + 1))
                rng.shuffle(permutation)
                spec = instantiate(edge_mask, domains, permutation)
                spec.update(verifier=domain, source=SCHEMA,
                            instance_id=f'{tag}-{seed}-t{tier}-m{support}-{index}',
                            num_completions=support,
                            num_solutions=original.graph_completion_count(size, spec['edges'], [None] * size))
                row = {'problem': original._graph_prompt(size, spec['edges'], spec['partial_colors']),
                       'answer': json.dumps(spec, sort_keys=True), 'modebench_task': domain,
                       'answer_mode_count': support, 'answer_mode_split': tag,
                       'scale_candidate_generator': SCHEMA, 'scale_candidate_tier': tier,
                       'scale_candidate_profile': json.dumps(_profile(tier, support), sort_keys=True,
                                                             separators=(',', ':')),
                       'scale_origin_metadata': '{}', 'scale_cell_index': index,
                       'scale_graph_topology_law_version': SCHEMA}
                key = identity(domain, row)
                if key in blocked:
                    continue
                blocked.add(key)
                rows.append(row)
                break
            else:
                raise RuntimeError(f'Graph r6 exhausted fixed proposal budget: tier{tier}/support{support}')
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('Graph r6 exact support histogram drift')
    return rows


def verify_structure(rows):
    """Audit intrinsic law/native prompt/count only; never grade an answer."""
    size = SHOWN + HIDDEN
    for row in rows:
        tier = row.get('scale_candidate_tier')
        if type(tier) is not int or tier not in range(4):
            raise RuntimeError('Graph r6 profile audit failed: invalid tier')
        support = row.get('answer_mode_count')
        if support not in SUPPORTS:
            raise RuntimeError('Graph r6 profile audit failed: unregistered support')
        spec = json.loads(row['answer'])
        edge_mask, domains, recovered = structural_signature(spec)
        _, used = resolve(tier, support)
        expected = json.dumps(_profile(tier, support), sort_keys=True, separators=(',', ':'))
        low, high = BANDS[used['band']]
        if (row.get('scale_candidate_generator') != SCHEMA
                or row.get('scale_graph_topology_law_version') != SCHEMA
                or row.get('scale_candidate_profile') != expected or row.get('scale_origin_metadata') != '{}'
                or row.get('modebench_task') != 'graph_coloring' or spec.get('verifier') != 'graph_coloring'
                or spec.get('source') != SCHEMA or recovered != support
                or spec.get('num_completions') != support
                or forced_count(domains) != used['forced']
                or not low <= edge_mask.bit_count() <= high
                or spec.get('num_solutions') != original.graph_completion_count(size, spec['edges'], [None] * size)
                or row.get('problem') != original._graph_prompt(size, spec['edges'], spec['partial_colors'])):
            raise RuntimeError('Graph r6 profile, native prompt, or exact support changed')
    return {'graph_r6_structural_profile': True, 'rows': len(rows)}


def verify_rows(domain, rows):
    if domain != 'graph_coloring':
        raise ValueError('Graph r6 only supports graph_coloring')
    structure = verify_structure(rows)
    return {**original.verify_rows(domain, rows), **structure}
