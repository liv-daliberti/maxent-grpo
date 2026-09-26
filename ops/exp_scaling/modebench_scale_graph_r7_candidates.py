"""Graph r7 topology laws: per-cell hidden count chosen by answer density.

Four knobs have now been measured inert for this domain: the induced-path
multiplier (r3), shown-shown distractor edges (r5), forced hidden vertices (r6)
and the hidden-edge band (r6). Twelve configurations spanning three laws collapse
onto two levels, pass@1 0.364 with a standard deviation of 0.019 at three hidden
vertices and 0.105 with 0.006 at four. The reason is that none of those knobs
changes what actually governs difficulty here:

    answer density = exact support / 3 ** hidden_vertices

Density is pinned by the registered support histogram, which is fixed by the
Level 3 release, and by the hidden count being an integer. That quantizes the
reachable level at roughly 0.36 and 0.11 and strands the 0.208 target between
them. No knob applied uniformly across a pool can reach it.

r7 therefore assigns the hidden count per answer-mode-count cell. Each tier
declares a target density, and every cell takes whichever hidden count puts its
own density closest to that target. The support histogram is untouched; only
which cells are rendered on six vertices and which on seven changes. Two things
follow. The weighted mean density becomes tunable between 0.08 and 0.24 rather
than quantized, and cell densities become more equal than they are at any fixed
hidden count, which should reduce the heterogeneity gap rather than inflate it.

That is the difference from the r4 per-cell law, which searched 32,768
assignments against noisy measured cell means, overfitted, and left the gap
unchanged at 0.42. Here the per-cell choice is exact arithmetic and the quantity
being equalised is the one the measurements identify as governing.

Predicted pass@1 at the pooled ratio of 1.388, with the range over the two
per-level ratios of 1.29 and 1.49:

    tier 0  density 0.1892  ->  0.263  (0.244 to 0.282)
    tier 1  density 0.1522  ->  0.211  (0.196 to 0.227)
    tier 2  density 0.1233  ->  0.171  (0.159 to 0.184)
    tier 3  density 0.0816  ->  0.113  (0.105 to 0.122)

Tier 1 is the candidate rung and tier 3 reproduces the all-four-hidden control
measured at 0.0993 and 0.0947. No ordering is claimed; the pilot measures it.
Prompt renderer, answer syntax, canonicalizer and verifier are unchanged.
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

SCHEMA = 'modebench_scale_graph_r7_density_graded_laws_v1'
DOMAINS = ('graph_coloring',)
SUPPORTS = (4, 5, 6, 8, 9, 12, 18)
SHOWN = 3
HIDDEN_CHOICES = (3, 4)
COLORS = 3
MAX_PROPOSALS_PER_ROW = 100_000
# Declared target density per tier. Tier 3 is the all-four-hidden control, whose
# density is exactly the fixed-h=4 weighted mean. Tier 0 sits just above the
# support-8 crossover at 0.1976 so that cell takes three hidden vertices.
TARGET_DENSITY = (0.21, 0.152, 0.123, 0.0816)


def cell_density(support, hidden):
    return support / COLORS ** hidden


def assign_hidden(tier, support):
    """Nearest density to the tier's declared target; ties go to the harder choice."""
    if type(tier) is not int or tier not in range(4):
        raise ValueError('Graph r7 requires integer tier0..3')
    if type(support) is not int or support not in SUPPORTS:
        raise ValueError('Graph r7 requires a registered exact support')
    target = TARGET_DENSITY[tier]
    return min(HIDDEN_CHOICES, key=lambda h: (abs(cell_density(support, h) - target), -h))


def _hidden_pairs(hidden):
    return tuple(combinations(range(hidden), 2))


def _colorings(hidden):
    return tuple(product(range(1, COLORS + 1), repeat=hidden))


def completion_count(hidden, edge_mask, domains):
    """Pure finite constraint count, not a model-answer grader."""
    pairs = _hidden_pairs(hidden)
    return sum(all(domains[i] & (1 << (colors[i] - 1)) for i in range(hidden))
               and all(colors[a] != colors[b] for bit, (a, b) in enumerate(pairs)
                       if edge_mask & (1 << bit)) for colors in _colorings(hidden))


@lru_cache(maxsize=4)
def catalog(hidden):
    if hidden not in HIDDEN_CHOICES:
        raise ValueError('Graph r7 renders six or seven vertices only')
    groups = {support: [] for support in SUPPORTS}
    for edge_mask in range(1 << len(_hidden_pairs(hidden))):
        for domains in product(range(1, 1 << COLORS), repeat=hidden):
            support = completion_count(hidden, edge_mask, domains)
            if support in groups:
                groups[support].append((edge_mask, domains))
    missing = [support for support, values in groups.items() if not values]
    if missing:
        raise RuntimeError(f'Graph r7 cannot realize supports {missing} at {hidden} hidden vertices')
    return {support: tuple(values) for support, values in groups.items()}


def weighted_density(tier, histogram):
    total = sum(histogram.values())
    return sum(count * cell_density(support, assign_hidden(tier, support))
               for support, count in histogram.items()) / total


def _profile(tier, support):
    hidden = assign_hidden(tier, support)
    return {'answer_mode_count': support, 'vertices': SHOWN + hidden, 'hidden_vertices': hidden,
            'shown_colors': list(range(1, COLORS + 1)), 'known_known_edges': 0,
            'target_answer_density': TARGET_DENSITY[tier],
            'cell_answer_density': cell_density(support, hidden),
            'graded': 'per_cell_hidden_count_by_nearest_answer_density',
            'hidden_domains': 'all_nonempty_subsets_of_colors_1_2_3',
            'anchor_edges': 'one_edge_for_each_forbidden_hidden_color',
            'hidden_graphs': f'all_simple_labeled_graphs_on_{hidden}_vertices',
            'vertex_order': f'uniform_permutation_of_all_{SHOWN + hidden}_vertices',
            'sampling': 'uniform_over_skeletons_conditioned_on_exact_support',
            'difficulty_ordering': 'prospective_density_prediction_no_monotonicity_claim'}


PROFILES = {'graph_coloring': [
    {str(support): _profile(tier, support) for support in SUPPORTS} for tier in range(4)]}
identity = original.identity


def source_paths():
    return sorted({Path(__file__).resolve(), *original.source_paths()})


def _seed(*parts):
    value = json.dumps([SCHEMA, *parts], sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(value).digest(), 'big')


def instantiate(hidden, edge_mask, domains, permutation):
    """Relabel one abstract constraint graph; no dataset or verifier calls."""
    size = SHOWN + hidden
    pairs = _hidden_pairs(hidden)
    if (type(edge_mask) is not int or edge_mask not in range(1 << len(pairs))
            or len(domains) != hidden
            or any(type(d) is not int or d not in range(1, 1 << COLORS) for d in domains)
            or len(permutation) != size or any(type(v) is not int for v in permutation)
            or set(permutation) != set(range(1, size + 1))):
        raise ValueError('invalid Graph r7 skeleton or vertex permutation')
    edges = [(SHOWN + a, SHOWN + b) for bit, (a, b) in enumerate(pairs) if edge_mask & (1 << bit)]
    edges += [(color - 1, SHOWN + i) for i, allowed in enumerate(domains)
              for color in range(1, COLORS + 1) if not allowed & (1 << (color - 1))]
    edges = sorted(sorted((permutation[a], permutation[b])) for a, b in edges)
    partial = [None] * size
    for base, color in enumerate(range(1, COLORS + 1)):
        partial[permutation[base] - 1] = color
    return {'n': size, 'edges': edges, 'partial_colors': partial}


def structural_signature(hidden, spec):
    """Recover the exact abstract law from a labeled graph, or reject it."""
    size = SHOWN + hidden
    if type(spec.get('n')) is not int or spec['n'] != size:
        raise ValueError(f'Graph r7 cell requires {size} vertices')
    partial, edges = spec.get('partial_colors'), spec.get('edges')
    if (not isinstance(partial, list) or len(partial) != size
            or any(c is not None and (type(c) is not int or c not in range(1, COLORS + 1)) for c in partial)
            or sorted(c for c in partial if c is not None) != list(range(1, COLORS + 1))
            or not isinstance(edges, list)
            or any(not isinstance(e, list) or len(e) != 2 or any(type(v) is not int for v in e)
                   or not 1 <= e[0] < e[1] <= size for e in edges)
            or edges != sorted(edges) or len({tuple(e) for e in edges}) != len(edges)):
        raise ValueError('Graph r7 requires exact shown anchors and a canonical simple graph')
    hidden_ids = [i + 1 for i, color in enumerate(partial) if color is None]
    if len(hidden_ids) != hidden:
        raise ValueError('Graph r7 hidden count does not match the cell assignment')
    pairs = _hidden_pairs(hidden)
    domains, edge_mask = [(1 << COLORS) - 1] * hidden, 0
    for a, b in edges:
        if a in hidden_ids and b in hidden_ids:
            edge_mask |= 1 << pairs.index((hidden_ids.index(a), hidden_ids.index(b)))
        elif a not in hidden_ids and b not in hidden_ids:
            raise ValueError('Graph r7 has no known-known edges')
        else:
            h, other = (a, b) if a in hidden_ids else (b, a)
            domains[hidden_ids.index(h)] &= ~(1 << (partial[other - 1] - 1))
    if not all(domains):
        raise ValueError('Graph r7 hidden domains must be nonempty')
    support = completion_count(hidden, edge_mask, tuple(domains))
    if support not in SUPPORTS:
        raise ValueError('Graph r7 support outside the registered histogram')
    return edge_mask, tuple(domains), support


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    if type(tier) is not int or tier not in range(4):
        raise ValueError('Graph r7 requires integer tier0..3')
    if (domain != 'graph_coloring' or type(seed) is not int or seed < 0
            or type(multiplier) is not int or multiplier < 1 or joint_target is not None
            or any(type(s) is not int or s not in SUPPORTS or type(n) is not int or n < 0
                   for s, n in target.items())):
        raise ValueError('Graph r7 requires exact support quotas, a fixed seed, and no joint target')
    required = Counter({s: n * multiplier for s, n in target.items() if n})
    blocked, rows = set(excluded), []
    for support, count in sorted(required.items()):
        hidden = assign_hidden(tier, support)
        size = SHOWN + hidden
        options = catalog(hidden)[support]
        rng = random.Random(_seed(seed, tier, support, domain))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                edge_mask, domains = options[rng.randrange(len(options))]
                permutation = list(range(1, size + 1))
                rng.shuffle(permutation)
                spec = instantiate(hidden, edge_mask, domains, permutation)
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
                raise RuntimeError(f'Graph r7 exhausted fixed proposal budget: tier{tier}/support{support}')
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('Graph r7 exact support histogram drift')
    return rows


def verify_structure(rows):
    """Audit intrinsic law/native prompt/count only; never grade an answer."""
    for row in rows:
        tier = row.get('scale_candidate_tier')
        if type(tier) is not int or tier not in range(4):
            raise RuntimeError('Graph r7 profile audit failed: invalid tier')
        support = row.get('answer_mode_count')
        if support not in SUPPORTS:
            raise RuntimeError('Graph r7 profile audit failed: unregistered support')
        hidden = assign_hidden(tier, support)
        size = SHOWN + hidden
        spec = json.loads(row['answer'])
        _, _, recovered = structural_signature(hidden, spec)
        expected = json.dumps(_profile(tier, support), sort_keys=True, separators=(',', ':'))
        if (row.get('scale_candidate_generator') != SCHEMA
                or row.get('scale_graph_topology_law_version') != SCHEMA
                or row.get('scale_candidate_profile') != expected or row.get('scale_origin_metadata') != '{}'
                or row.get('modebench_task') != 'graph_coloring' or spec.get('verifier') != 'graph_coloring'
                or spec.get('source') != SCHEMA or recovered != support
                or spec.get('num_completions') != support
                or spec.get('num_solutions') != original.graph_completion_count(size, spec['edges'], [None] * size)
                or row.get('problem') != original._graph_prompt(size, spec['edges'], spec['partial_colors'])):
            raise RuntimeError('Graph r7 profile, native prompt, or exact support changed')
    return {'graph_r7_structural_profile': True, 'rows': len(rows)}


def verify_rows(domain, rows):
    if domain != 'graph_coloring':
        raise ValueError('Graph r7 only supports graph_coloring')
    structure = verify_structure(rows)
    return {**original.verify_rows(domain, rows), **structure}
