"""Graph r5 topology laws: the r3 catalog given a real difficulty knob.

r3 was rejected on 2026-09-14 for having no usable difficulty gradient, and the
screening pilot was branched from r2 instead. Measured against the quantity that
actually gates this domain, that was the wrong call. r3 has the lowest
heterogeneity gaps ever recorded for Level5 graph -- 0.329, 0.333, 0.289 and
0.260, falling with its path multiplier -- and its tier 3 is BELOW the 0.282 the
target requires. Every r2-derived design, including the r4 per-cell law, sits at
0.40 or worse. r3 was never too bimodal; it was uniformly too easy, with all four
tiers between pass@1 0.32 and 0.38 and no downward range.

The reason r3 is uniform is structural. Its problems are drawn from a fully
enumerated catalog of abstract skeletons: three shown vertices carrying colors
1, 2 and 3, and three hidden vertices whose colour domains are encoded by anchor
edges to those shown vertices, with hidden-hidden edges supplying coupling.
Every prompt is structurally alike, so per-prompt success is far less dispersed
than in the rejection-sampled random graphs r2 and r4 use. Its only knob, a
weight multiplier on induced hidden paths, changes which skeletons are drawn but
not how hard they are.

r5 keeps that catalog construction exactly and adds two knobs that raise
difficulty without breaking uniformity:

  hidden_vertices 3 -> 4   the answer grows from three digits to four and the
                          hidden search from 27 to 81 assignments, while every
                          prompt keeps the same encoding and stays reachable.
  known_known_edges 0 -> 3 edges among the shown vertices. The shown vertices
                          carry distinct colours, so such an edge is satisfied by
                          construction and cannot change the support or the
                          answer. It is a pure parsing distractor.

The four tiers are the 2x2 factorial of those knobs at the fixed path multiplier
64, r3's lowest-gap setting. Tier 0 reproduces r3 tier 3, a measured control at
pass@1 0.362 / pass@8 0.712. No difficulty ordering is claimed: the factorial is
a hypothesis and the pilot measures it. Prompt renderer, answer syntax,
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

SCHEMA = 'modebench_scale_graph_r5_hidden_topology_laws_v1'
DOMAINS = ('graph_coloring',)
SUPPORTS = (4, 5, 6, 8, 9, 12, 18)
SHOWN = 3
SHOWN_PAIRS = tuple(combinations(range(SHOWN), 2))
MAX_PROPOSALS_PER_ROW = 100_000
# (hidden_vertices, known_known_edges, induced_hidden_path_multiplier)
TIERS = ((3, 0, 64), (3, 3, 64), (4, 0, 64), (4, 3, 64))


def _hidden_pairs(hidden):
    return tuple(combinations(range(hidden), 2))


def _colorings(hidden):
    return tuple(product((1, 2, 3), repeat=hidden))


PROFILES = {'graph_coloring': [
    {'vertices': SHOWN + hidden, 'hidden_vertices': hidden, 'shown_colors': [1, 2, 3],
     'known_known_edges': known, 'hidden_domains': 'all_nonempty_subsets_of_colors_1_2_3',
     'anchor_edges': 'one_edge_for_each_forbidden_hidden_color',
     'hidden_graphs': f'all_simple_labeled_graphs_on_{hidden}_vertices',
     'induced_hidden_path_multiplier': multiplier, 'other_skeleton_weight': 1,
     'vertex_order': f'uniform_permutation_of_all_{SHOWN + hidden}_vertices',
     'sampling': 'weighted_skeletons_conditioned_on_exact_support_then_uniform_relabeling_and_exclusion_rejection',
     'known_known_edge_law': 'complete_prefix_of_the_shown_triangle_satisfied_by_construction',
     'difficulty_ordering': 'prospective_2x2_factorial_no_monotonicity_claim'}
    for hidden, known, multiplier in TIERS]}
identity = original.identity


def source_paths():
    return sorted({Path(__file__).resolve(), *original.source_paths()})


def _seed(*parts):
    value = json.dumps([SCHEMA, *parts], sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(value).digest(), 'big')


def _tier(tier):
    if type(tier) is not int or tier not in range(4):
        raise ValueError('Graph r5 requires integer tier0..3')
    return TIERS[tier]


def completion_count(hidden, edge_mask, domains):
    """Pure finite constraint count, not a model-answer grader."""
    pairs = _hidden_pairs(hidden)
    return sum(all(domains[i] & (1 << (colors[i] - 1)) for i in range(hidden))
               and all(colors[a] != colors[b] for bit, (a, b) in enumerate(pairs)
                       if edge_mask & (1 << bit)) for colors in _colorings(hidden))


@lru_cache(maxsize=4)
def catalog(hidden):
    groups = {support: [] for support in SUPPORTS}
    for edge_mask in range(1 << len(_hidden_pairs(hidden))):
        for domains in product(range(1, 8), repeat=hidden):
            support = completion_count(hidden, edge_mask, domains)
            if support in groups:
                groups[support].append((edge_mask, domains))
    missing = [support for support, values in groups.items() if not values]
    if missing:
        raise RuntimeError(f'Graph r5 cannot realize exact registered supports {missing} '
                           f'with {hidden} hidden vertices')
    return {support: tuple(values) for support, values in groups.items()}


def skeleton_weight(tier, edge_mask):
    _, _, multiplier = _tier(tier)
    return multiplier if edge_mask.bit_count() == 2 else 1


@lru_cache(maxsize=28)
def proposal_profiles(tier, support):
    hidden, _, _ = _tier(tier)
    if type(support) is not int or support not in SUPPORTS:
        raise ValueError('Graph r5 requires a registered exact support')
    options = catalog(hidden)[support]
    return options, tuple(skeleton_weight(tier, mask) for mask, domains in options)


def instantiate(hidden, known, edge_mask, domains, permutation):
    """Relabel one abstract constraint graph; no dataset or verifier calls."""
    size = SHOWN + hidden
    pairs = _hidden_pairs(hidden)
    if (type(edge_mask) is not int or edge_mask not in range(1 << len(pairs))
            or len(domains) != hidden or any(type(d) is not int or d not in range(1, 8) for d in domains)
            or type(known) is not int or known not in range(len(SHOWN_PAIRS) + 1)
            or len(permutation) != size or any(type(v) is not int for v in permutation)
            or set(permutation) != set(range(1, size + 1))):
        raise ValueError('invalid Graph r5 skeleton or vertex permutation')
    edges = [(SHOWN + a, SHOWN + b) for bit, (a, b) in enumerate(pairs) if edge_mask & (1 << bit)]
    edges += [(color - 1, SHOWN + i) for i, allowed in enumerate(domains) for color in (1, 2, 3)
              if not allowed & (1 << (color - 1))]
    # Shown vertices carry distinct colours, so these edges are satisfied by
    # construction: they change neither the support nor the answer.
    edges += list(SHOWN_PAIRS[:known])
    edges = sorted(sorted((permutation[a], permutation[b])) for a, b in edges)
    partial = [None] * size
    for base, color in enumerate((1, 2, 3)):
        partial[permutation[base] - 1] = color
    return {'n': size, 'edges': edges, 'partial_colors': partial}


def structural_signature(hidden, known, spec):
    """Recover the exact abstract law from a labeled graph, or reject it."""
    size = SHOWN + hidden
    if type(spec.get('n')) is not int or spec['n'] != size:
        raise ValueError(f'Graph r5 tier requires {size} vertices')
    partial, edges = spec.get('partial_colors'), spec.get('edges')
    if (not isinstance(partial, list) or len(partial) != size
            or any(c is not None and (type(c) is not int or c not in (1, 2, 3)) for c in partial)
            or sorted(c for c in partial if c is not None) != [1, 2, 3]
            or not isinstance(edges, list)
            or any(not isinstance(e, list) or len(e) != 2 or any(type(v) is not int for v in e)
                   or not 1 <= e[0] < e[1] <= size for e in edges)
            or edges != sorted(edges) or len({tuple(e) for e in edges}) != len(edges)):
        raise ValueError('Graph r5 requires exact shown anchors and a canonical simple graph')
    hidden_ids = [i + 1 for i, color in enumerate(partial) if color is None]
    if len(hidden_ids) != hidden:
        raise ValueError('Graph r5 hidden count changed')
    pairs = _hidden_pairs(hidden)
    domains, edge_mask, known_seen = [7] * hidden, 0, 0
    for a, b in edges:
        if a in hidden_ids and b in hidden_ids:
            edge_mask |= 1 << pairs.index((hidden_ids.index(a), hidden_ids.index(b)))
        elif a not in hidden_ids and b not in hidden_ids:
            known_seen += 1
        else:
            h, other = (a, b) if a in hidden_ids else (b, a)
            domains[hidden_ids.index(h)] &= ~(1 << (partial[other - 1] - 1))
    if known_seen != known:
        raise ValueError('Graph r5 known-known edge count changed')
    if not all(domains):
        raise ValueError('Graph r5 hidden domains must be nonempty')
    support = completion_count(hidden, edge_mask, tuple(domains))
    if support not in SUPPORTS:
        raise ValueError('Graph r5 support outside the registered histogram')
    return edge_mask, tuple(domains), support


def _proposal(tier, support, rng):
    hidden, known, _ = _tier(tier)
    options, weights = proposal_profiles(tier, support)
    ticket = rng.randrange(sum(weights))
    for (edge_mask, domains), weight in zip(options, weights):
        if ticket < weight:
            break
        ticket -= weight
    permutation = list(range(1, SHOWN + hidden + 1))
    rng.shuffle(permutation)
    return instantiate(hidden, known, edge_mask, domains, permutation)


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    hidden, known, _ = _tier(tier)
    if (domain != 'graph_coloring' or type(seed) is not int or seed < 0
            or type(multiplier) is not int or multiplier < 1 or joint_target is not None
            or any(type(s) is not int or s not in SUPPORTS or type(n) is not int or n < 0
                   for s, n in target.items())):
        raise ValueError('Graph r5 requires exact support quotas, a fixed seed, and no joint target')
    size = SHOWN + hidden
    required = Counter({s: n * multiplier for s, n in target.items() if n})
    blocked, rows = set(excluded), []
    for support, count in sorted(required.items()):
        rng = random.Random(_seed(seed, tier, support, domain))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                spec = _proposal(tier, support, rng)
                spec.update(verifier=domain, source=SCHEMA,
                            instance_id=f'{tag}-{seed}-t{tier}-m{support}-{index}',
                            num_completions=support,
                            num_solutions=original.graph_completion_count(size, spec['edges'], [None] * size))
                row = {'problem': original._graph_prompt(size, spec['edges'], spec['partial_colors']),
                       'answer': json.dumps(spec, sort_keys=True), 'modebench_task': domain,
                       'answer_mode_count': support, 'answer_mode_split': tag,
                       'scale_candidate_generator': SCHEMA, 'scale_candidate_tier': tier,
                       'scale_candidate_profile': json.dumps(PROFILES[domain][tier], sort_keys=True,
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
                raise RuntimeError(f'Graph r5 exhausted fixed proposal budget: tier{tier}/support{support}')
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('Graph r5 exact support histogram drift')
    return rows


def verify_structure(rows):
    """Audit intrinsic law/native prompt/count only; never grade an answer."""
    for row in rows:
        tier = row.get('scale_candidate_tier')
        hidden, known, _ = _tier(tier)
        size = SHOWN + hidden
        spec = json.loads(row['answer'])
        _, _, support = structural_signature(hidden, known, spec)
        expected = json.dumps(PROFILES['graph_coloring'][tier], sort_keys=True, separators=(',', ':'))
        if (row.get('scale_candidate_generator') != SCHEMA
                or row.get('scale_graph_topology_law_version') != SCHEMA
                or row.get('scale_candidate_profile') != expected or row.get('scale_origin_metadata') != '{}'
                or row.get('modebench_task') != 'graph_coloring' or spec.get('verifier') != 'graph_coloring'
                or spec.get('source') != SCHEMA or type(row.get('answer_mode_count')) is not int
                or row['answer_mode_count'] != support or spec.get('num_completions') != support
                or spec.get('num_solutions') != original.graph_completion_count(size, spec['edges'], [None] * size)
                or row.get('problem') != original._graph_prompt(size, spec['edges'], spec['partial_colors'])):
            raise RuntimeError('Graph r5 profile, native prompt, or exact support changed')
    return {'graph_r5_structural_profile': True, 'rows': len(rows)}


def verify_rows(domain, rows):
    if domain != 'graph_coloring':
        raise ValueError('Graph r5 only supports graph_coloring')
    structure = verify_structure(rows)
    return {**original.verify_rows(domain, rows), **structure}
