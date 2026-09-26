"""Unregistered direct-anchor Graph law; fresh bases and one to four forced vertices.

The common base is the development-informed r3 tier1 (path multiplier4), which
was selected by the failed r3 fit. Every profile changes the actual task; none
reuses a scored base or repeats the unextended law. No accuracy ordering is
claimed. Native prompts, identity, support counting and eventual grader stay
unchanged. Structural scratch qualification must be refreshed after new sources
are fixed before any future registration.
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
import modebench_scale_graph_r3_candidates as base
original = base.original
identity = original.identity
SCHEMA = 'modebench_scale_graph_direct_anchor_laws_v1'
DOMAINS = ('graph_coloring',)
SUPPORTS = base.SUPPORTS
BASE_TIER = 1
EXTRA_VERTICES = (1, 2, 3, 4)
MAX_PROPOSALS_PER_ROW = 2_000
PROFILES = {'graph_coloring': [
    {'base_law': base.SCHEMA, 'base_tier': BASE_TIER,
     'base_path_multiplier': 4, 'base_choice': 'development_informed_previously_selected_failed_r3_tier1',
     'vertices': 6+extra, 'hidden_vertices': 3+extra,
     'extra_vertices': extra,
     'attachment': 'each_extra_independently_uniform_pair_of_two_distinct_shown_anchor_colors',
     'extra_to_original_hidden_edges': 0, 'extra_to_extra_edges': 0,
     'extra_edges': 2*extra, 'vertex_order': 'uniform_permutation_of_all_vertices',
     'sampling': 'weighted_base_skeleton_then_independent_uniform_anchor_pairs_and_relabeling_then_full_proposal_rejection',
     'freshness': 'full_native_identity_and_all_relevant_compacted_induced_six_vertex_base_projections',
     'difficulty_ordering': 'unmeasured_no_monotonicity_claim'} for extra in EXTRA_VERTICES]}


def source_paths():
    return sorted({Path(__file__).resolve(), *base.source_paths()})


def _tier(tier):
    if type(tier) is not int or tier not in range(4):
        raise ValueError('direct-anchor Graph requires integer tier0..3')
    return EXTRA_VERTICES[tier]


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([SCHEMA, *parts], sort_keys=True,
        separators=(',', ':')).encode()).digest(), 'big')


def graph_key(spec):
    return ('graph_coloring', spec['n'], tuple(map(tuple, spec['edges'])),
        ''.join('?' if c is None else str(c) for c in spec['partial_colors']))


def spec_from_key(key):
    if (not isinstance(key, tuple) or len(key) != 4 or key[0] != 'graph_coloring'
            or type(key[1]) is not int or key[1] < 1 or not isinstance(key[3], str)
            or len(key[3]) != key[1] or any(c not in '?123' for c in key[3])):
        raise ValueError('invalid native Graph exclusion identity')
    edges = key[2]
    if (not isinstance(edges, tuple) or any(not isinstance(e, tuple) or len(e) != 2
            or any(type(v) is not int for v in e) or not 1 <= e[0] < e[1] <= key[1] for e in edges)
            or tuple(sorted(set(edges))) != edges):
        raise ValueError('invalid native Graph exclusion edges')
    return {'n': key[1], 'edges': list(map(list, edges)),
        'partial_colors': [None if c == '?' else int(c) for c in key[3]]}


def induced(spec, vertices):
    vertices = sorted(vertices)
    labels = {v: i+1 for i, v in enumerate(vertices)}
    return {'n': len(vertices), 'edges': [[labels[a], labels[b]] for a, b in spec['edges']
        if a in labels and b in labels], 'partial_colors': [spec['partial_colors'][v-1] for v in vertices]}


@lru_cache(maxsize=4096)
def _abstract_support(mask, domains):
    return base.completion_count(mask, domains)


def projection_support(spec):
    """Return a registered r3 support for this induced graph, else None."""
    known = {i+1: c for i, c in enumerate(spec['partial_colors']) if c is not None}
    hidden = [i+1 for i, c in enumerate(spec['partial_colors']) if c is None]
    if len(known) != 3 or sorted(known.values()) != [1, 2, 3] or len(hidden) != 3:
        return None
    domains, mask = [7]*3, 0
    position = {v: i for i, v in enumerate(hidden)}
    for a, b in spec['edges']:
        if a in known and b in known:
            return None
        if a in position and b in position:
            mask |= 1 << base.HIDDEN_PAIRS.index((position[a], position[b]))
        else:
            h, k = (a, b) if a in position else (b, a)
            domains[position[h]] &= ~(1 << (known[k]-1))
    if not all(domains):
        return None
    count = _abstract_support(mask, tuple(domains))
    return count if count in SUPPORTS else None


def base_projections(key):
    """All relevant labeled induced bases; no provenance metadata is trusted.

    Select one shown anchor of each color and three hidden vertices, retaining
    induced edges and compacting labels in ascending order. This is the native
    labeled identity convention, not an isomorphism quotient. Longer historical
    graphs and generated extensions use the same rule.
    """
    spec = spec_from_key(key)
    anchors = [[i+1 for i, c in enumerate(spec['partial_colors']) if c == color]
        for color in (1, 2, 3)]
    hidden = [i+1 for i, c in enumerate(spec['partial_colors']) if c is None]
    result = set()
    for known in product(*anchors):
        for missing in combinations(hidden, 3):
            sub = induced(spec, (*known, *missing))
            if projection_support(sub) is not None:
                result.add(graph_key(sub))
    return frozenset(result)


def blocked_projections(excluded):
    result = set()
    for key in excluded:
        if isinstance(key, tuple) and key and key[0] == 'graph_coloring':
            result.update(base_projections(key))
    return result


def extend(mask, domains, extra, choices, permutation):
    """Each extra has two shown neighbors, fixing its unique third color."""
    if (type(extra) is not int or extra not in EXTRA_VERTICES or len(choices) != extra
            or len(permutation) != 6+extra or any(type(v) is not int for v in permutation)
            or set(permutation) != set(range(1, 7+extra))):
        raise ValueError('invalid direct-anchor extension size or permutation')
    spec = base.instantiate(mask, domains, list(range(1, 7)))
    for index, pair in enumerate(choices):
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(type(c) is not int for c in pair) or not 1 <= pair[0] < pair[1] <= 3):
            raise ValueError('each attachment needs two distinct sorted shown-anchor colors')
        spec['edges'].extend([[pair[0], 7+index], [pair[1], 7+index]])
    partial = [None]*(6+extra)
    for color in (1, 2, 3):
        partial[permutation[color-1]-1] = color
    spec = {'n': 6+extra,
        'edges': sorted(sorted((permutation[a-1], permutation[b-1])) for a, b in spec['edges']),
        'partial_colors': partial}
    witness = {'base_edge_mask': mask, 'base_domains': list(domains),
        'attachment_anchor_colors': [list(pair) for pair in choices],
        'slot_to_vertex': list(permutation)}
    return spec, witness


def _proposal(tier, support, rng):
    extra = _tier(tier)
    options, weights = base.proposal_profiles(BASE_TIER, support)
    ticket = rng.randrange(sum(weights))
    for (mask, domains), weight in zip(options, weights):
        if ticket < weight:
            break
        ticket -= weight
    pairs = tuple(combinations((1, 2, 3), 2))
    choices = [pairs[rng.randrange(3)] for _ in range(extra)]
    permutation = list(range(1, 7+extra))
    rng.shuffle(permutation)
    return extend(mask, domains, extra, choices, permutation)


def _row(spec, witness, support, index, seed, tag, tier):
    spec.update(verifier='graph_coloring', source=SCHEMA,
        instance_id=f'{tag}-{seed}-t{tier}-m{support}-{index}', num_completions=support,
        num_solutions=original.graph_completion_count(spec['n'], spec['edges'], [None]*spec['n']))
    row = {'problem': original._graph_prompt(spec['n'], spec['edges'], spec['partial_colors']),
        'answer': json.dumps(spec, sort_keys=True), 'modebench_task': 'graph_coloring',
        'answer_mode_count': support, 'answer_mode_split': tag,
        'scale_candidate_generator': SCHEMA, 'scale_candidate_tier': tier,
        'scale_candidate_profile': json.dumps(PROFILES['graph_coloring'][tier], sort_keys=True, separators=(',', ':')),
        'scale_origin_metadata': json.dumps(witness, sort_keys=True, separators=(',', ':')),
        'scale_cell_index': index, 'scale_graph_direct_anchor_law_version': SCHEMA}
    return row


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    _tier(tier)
    if (domain != 'graph_coloring' or type(seed) is not int or seed < 0
            or type(multiplier) is not int or multiplier < 1 or joint_target is not None
            or any(type(s) is not int or s not in SUPPORTS or type(n) is not int or n < 0
                for s, n in target.items())):
        raise ValueError('direct-anchor Graph requires exact support quotas and fixed native inputs')
    required = Counter({s: n*multiplier for s, n in target.items() if n})
    full_blocked, rows = set(excluded), []
    projections = blocked_projections(full_blocked)
    for support, count in sorted(required.items()):
        rng = random.Random(_seed(seed, tier, support, domain))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                spec, witness = _proposal(tier, support, rng)
                key = graph_key(spec)
                keys = base_projections(key)
                if key in full_blocked or keys & projections:
                    continue
                row = _row(spec, witness, support, index, seed, tag, tier)
                full_blocked.add(key)
                projections.update(keys)
                rows.append(row)
                break
            else:
                raise RuntimeError(f'direct-anchor Graph exhausted fixed proposal budget: tier{tier}/support{support}')
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('direct-anchor Graph exact support histogram drift')
    return rows


def verify_structure(rows):
    """Verify a direct-anchor unique-extension witness and original prompt, never grade answers."""
    for row in rows:
        tier = row.get('scale_candidate_tier')
        extra = _tier(tier)
        spec, witness = json.loads(row['answer']), json.loads(row['scale_origin_metadata'])
        expected_spec, expected_witness = extend(witness['base_edge_mask'], witness['base_domains'], extra,
            witness['attachment_anchor_colors'], witness['slot_to_vertex'])
        support = base.completion_count(witness['base_edge_mask'], witness['base_domains'])
        expected_profile = json.dumps(PROFILES['graph_coloring'][tier], sort_keys=True, separators=(',', ':'))
        if (witness != expected_witness or support not in SUPPORTS
                or any(spec.get(k) != v for k, v in expected_spec.items())
                or row.get('scale_candidate_generator') != SCHEMA
                or row.get('scale_graph_direct_anchor_law_version') != SCHEMA
                or row.get('scale_candidate_profile') != expected_profile
                or row.get('modebench_task') != 'graph_coloring' or spec.get('verifier') != 'graph_coloring'
                or spec.get('source') != SCHEMA or type(row.get('answer_mode_count')) is not int
                or row['answer_mode_count'] != support or spec.get('num_completions') != support
                or spec.get('num_solutions') != original.graph_completion_count(spec['n'], spec['edges'], [None]*spec['n'])
                or row.get('problem') != original._graph_prompt(spec['n'], spec['edges'], spec['partial_colors'])):
            raise RuntimeError('direct-anchor Graph witness, native prompt or exact support changed')
    return {'graph_direct_anchor_structural_profile': True, 'rows': len(rows)}


def verify_rows(domain, rows):
    if domain != 'graph_coloring':
        raise ValueError('direct-anchor Graph only supports graph_coloring')
    structure = verify_structure(rows)
    # Unchanged native qualification is available only for a later authorized run.
    return {**original.verify_rows(domain, rows), **structure}
