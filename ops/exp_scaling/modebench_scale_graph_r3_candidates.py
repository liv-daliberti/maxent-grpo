"""Prospective support-conditioned Graph r3 topology laws.

Six vertices retain three missing colors and the unchanged native interface.
The three shown vertices carry distinct colors1/2/3. Edges to these anchors
encode the allowed colors of each hidden vertex; the hidden graph supplies
coupling. Four fixed path multipliers alter intrinsic constraint structure.
No receipts or scores are read, and no performance ordering is claimed.

The finite catalog exhausts all8 hidden edge masks and7^3 nonempty domain
triples. Conditional on exact support, skeletons have weight1 except that
induced hidden paths have the tier multiplier. Uniform vertex relabeling and
historical-identity rejection complete the law. Supports5,9,18 have identical
within-support distributions across multipliers:5 is path-only,9/18 have no
paths. Independent tier streams still produce separate rows.
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
    if str(ROOT/directory) not in sys.path:
        sys.path.insert(0, str(ROOT/directory))
import modebench_scale_candidates as original

SCHEMA = 'modebench_scale_graph_r3_hidden_topology_laws_v1'
DOMAINS = ('graph_coloring',)
SUPPORTS = (4, 5, 6, 8, 9, 12, 18)
PATH_MULTIPLIERS = (1, 4, 16, 64)
HIDDEN_PAIRS = tuple(combinations(range(3), 2))
COLORINGS = tuple(product((1, 2, 3), repeat=3))
MAX_PROPOSALS_PER_ROW = 100_000
PROFILES = {'graph_coloring': [
    {'vertices': 6, 'hidden_vertices': 3, 'shown_colors': [1, 2, 3],
     'known_known_edges': 0, 'hidden_domains': 'all_nonempty_subsets_of_colors_1_2_3',
     'anchor_edges': 'one_edge_for_each_forbidden_hidden_color',
     'hidden_graphs': 'all_eight_simple_labeled_graphs_on_three_vertices',
     'induced_hidden_path_multiplier': multiplier, 'other_skeleton_weight': 1,
     'vertex_order': 'uniform_permutation_of_all_six_vertices',
     'sampling': 'weighted_skeletons_conditioned_on_exact_support_then_uniform_relabeling_and_exclusion_rejection',
     'tier_invariant_support_distributions': [5, 9, 18],
     'difficulty_ordering': 'prospective_hypotheses_no_monotonicity_claim'}
    for multiplier in PATH_MULTIPLIERS]}
identity = original.identity


def source_paths():
    return sorted({Path(__file__).resolve(), *original.source_paths()})


def _seed(*parts):
    value = json.dumps([SCHEMA, *parts], sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(value).digest(), 'big')


def _tier(tier):
    if type(tier) is not int or tier not in range(4):
        raise ValueError('Graph r3 requires integer tier0..3')
    return PATH_MULTIPLIERS[tier]


def completion_count(edge_mask, domains):
    """Pure finite constraint count, not a model-answer grader."""
    return sum(all(domains[i] & (1 << (colors[i]-1)) for i in range(3))
               and all(colors[a] != colors[b] for bit, (a, b) in enumerate(HIDDEN_PAIRS)
                       if edge_mask & (1 << bit)) for colors in COLORINGS)


@lru_cache(maxsize=1)
def catalog():
    groups = {support: [] for support in SUPPORTS}
    for edge_mask in range(8):
        for domains in product(range(1, 8), repeat=3):
            support = completion_count(edge_mask, domains)
            if support in groups:
                groups[support].append((edge_mask, domains))
    if any(not values for values in groups.values()):
        raise RuntimeError('Graph r3 cannot realize every exact registered support')
    return {support: tuple(values) for support, values in groups.items()}


def skeleton_weight(tier, edge_mask):
    multiplier = _tier(tier)
    return multiplier if edge_mask.bit_count() == 2 else 1


@lru_cache(maxsize=28)
def proposal_profiles(tier, support):
    _tier(tier)
    if type(support) is not int or support not in SUPPORTS:
        raise ValueError('Graph r3 requires a registered exact support')
    options = catalog()[support]
    return options, tuple(skeleton_weight(tier, mask) for mask, domains in options)


def instantiate(edge_mask, domains, permutation):
    """Relabel one abstract constraint graph; no dataset or verifier calls."""
    if (type(edge_mask) is not int or edge_mask not in range(8)
            or len(domains) != 3 or any(type(d) is not int or d not in range(1, 8) for d in domains)
            or len(permutation) != 6 or any(type(v) is not int for v in permutation)
            or set(permutation) != set(range(1, 7))):
        raise ValueError('invalid Graph r3 skeleton or vertex permutation')
    edges = [(3+a, 3+b) for bit, (a, b) in enumerate(HIDDEN_PAIRS) if edge_mask & (1 << bit)]
    edges += [(color-1, 3+i) for i, allowed in enumerate(domains) for color in (1, 2, 3)
              if not allowed & (1 << (color-1))]
    edges = sorted(sorted((permutation[a], permutation[b])) for a, b in edges)
    partial = [None]*6
    for base, color in enumerate((1, 2, 3)):
        partial[permutation[base]-1] = color
    return {'n': 6, 'edges': edges, 'partial_colors': partial}


def structural_signature(spec):
    """Recover the exact abstract law from a labeled graph, or reject it."""
    if type(spec.get('n')) is not int or spec['n'] != 6:
        raise ValueError('Graph r3 requires six vertices')
    partial, edges = spec.get('partial_colors'), spec.get('edges')
    if (not isinstance(partial, list) or len(partial) != 6
            or any(c is not None and (type(c) is not int or c not in (1, 2, 3)) for c in partial)
            or sorted(c for c in partial if c is not None) != [1, 2, 3]
            or not isinstance(edges, list)
            or any(not isinstance(e, list) or len(e) != 2 or any(type(v) is not int for v in e)
                   or not 1 <= e[0] < e[1] <= 6 for e in edges)
            or edges != sorted(edges) or len({tuple(e) for e in edges}) != len(edges)):
        raise ValueError('Graph r3 requires exact shown anchors and a canonical simple graph')
    hidden = [i+1 for i, color in enumerate(partial) if color is None]
    domains, edge_mask = [7]*3, 0
    for a, b in edges:
        if a in hidden and b in hidden:
            edge_mask |= 1 << HIDDEN_PAIRS.index((hidden.index(a), hidden.index(b)))
        elif a not in hidden and b not in hidden:
            raise ValueError('Graph r3 has no known-known edges')
        else:
            h, known = (a, b) if a in hidden else (b, a)
            domains[hidden.index(h)] &= ~(1 << (partial[known-1]-1))
    if not all(domains):
        raise ValueError('Graph r3 hidden domains must be nonempty')
    support = completion_count(edge_mask, domains)
    if support not in SUPPORTS:
        raise ValueError('Graph r3 support outside the registered histogram')
    return edge_mask, tuple(domains), support


def _proposal(tier, support, rng):
    options, weights = proposal_profiles(tier, support)
    ticket = rng.randrange(sum(weights))
    for (edge_mask, domains), weight in zip(options, weights):
        if ticket < weight:
            break
        ticket -= weight
    permutation = list(range(1, 7))
    rng.shuffle(permutation)
    return instantiate(edge_mask, domains, permutation)


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    _tier(tier)
    if (domain != 'graph_coloring' or type(seed) is not int or seed < 0
            or type(multiplier) is not int or multiplier < 1 or joint_target is not None
            or any(type(s) is not int or s not in SUPPORTS or type(n) is not int or n < 0
                   for s, n in target.items())):
        raise ValueError('Graph r3 requires exact support quotas, a fixed seed, and no joint target')
    required = Counter({s: n*multiplier for s, n in target.items() if n})
    blocked, rows = set(excluded), []
    for support, count in sorted(required.items()):
        rng = random.Random(_seed(seed, tier, support, domain))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                spec = _proposal(tier, support, rng)
                spec.update(verifier=domain, source=SCHEMA,
                    instance_id=f'{tag}-{seed}-t{tier}-m{support}-{index}', num_completions=support,
                    num_solutions=original.graph_completion_count(spec['n'], spec['edges'], [None]*6))
                row = {'problem': original._graph_prompt(spec['n'], spec['edges'], spec['partial_colors']),
                    'answer': json.dumps(spec, sort_keys=True), 'modebench_task': domain,
                    'answer_mode_count': support, 'answer_mode_split': tag,
                    'scale_candidate_generator': SCHEMA, 'scale_candidate_tier': tier,
                    'scale_candidate_profile': json.dumps(PROFILES[domain][tier], sort_keys=True, separators=(',', ':')),
                    'scale_origin_metadata': '{}', 'scale_cell_index': index, 'scale_graph_topology_law_version': SCHEMA}
                key = identity(domain, row)
                if key in blocked:
                    continue
                blocked.add(key)
                rows.append(row)
                break
            else:
                raise RuntimeError(f'Graph r3 exhausted fixed proposal budget: tier{tier}/support{support}')
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('Graph r3 exact support histogram drift')
    return rows


def verify_structure(rows):
    """Audit intrinsic law/native prompt/count only; never grade an answer."""
    for row in rows:
        tier = row.get('scale_candidate_tier')
        _tier(tier)
        spec = json.loads(row['answer'])
        _, _, support = structural_signature(spec)
        expected = json.dumps(PROFILES['graph_coloring'][tier], sort_keys=True, separators=(',', ':'))
        if (row.get('scale_candidate_generator') != SCHEMA or row.get('scale_graph_topology_law_version') != SCHEMA
                or row.get('scale_candidate_profile') != expected or row.get('scale_origin_metadata') != '{}'
                or row.get('modebench_task') != 'graph_coloring' or spec.get('verifier') != 'graph_coloring'
                or spec.get('source') != SCHEMA or type(row.get('answer_mode_count')) is not int
                or row['answer_mode_count'] != support or spec.get('num_completions') != support
                or spec.get('num_solutions') != original.graph_completion_count(6, spec['edges'], [None]*6)
                or row.get('problem') != original._graph_prompt(6, spec['edges'], spec['partial_colors'])):
            raise RuntimeError('Graph r3 profile, native prompt, or exact support changed')
    return {'graph_r3_structural_profile': True, 'rows': len(rows)}


def verify_rows(domain, rows):
    if domain != 'graph_coloring':
        raise ValueError('Graph r3 only supports graph_coloring')
    structure = verify_structure(rows)
    # This existing verifier remains the eventual production qualification gate.
    # Development-only structural tests do not invoke it.
    return {**original.verify_rows(domain, rows), **structure}
