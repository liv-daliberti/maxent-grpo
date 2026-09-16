"""Graph r4 candidate laws: difficulty graded per answer-mode-count cell.

Every earlier Level5 graph revision graded difficulty by tier alone, applying one
(vertices, hidden, ordering) preset to every cell in the pool. Measured that way,
r2 reached a best normalised error of 1.182, r3 destroyed the gradient at 2.902,
and the 2026-09-14 screening pilot that raised vertices at hidden_vertices=3
reached 1.077 without even being monotone.

The diagnosis was that between-cell difficulty spread, not the tier knob,
dominates. Within a single tier, mean pass@1 per cell spans 0.03 to 0.71, and the
target is LESS bimodal than every tier measured: it needs a heterogeneity gap of
0.282 against measured gaps of 0.35 to 0.50. Tier mixing cannot repair that,
because a mixture that includes dead rows inherits their ceiling.

This law therefore fixes a preset per (tier, answer-mode-count cell). The answer
histogram is fixed by the Level 3 release and is not touched; only which
structural parameters are used inside each cell is free. The existing fitter
already forecasts per (tier, cell), so it needs no change.

The tier table below was chosen against saved r2 development receipts and the
screening pilot, requiring a real ladder that brackets the target rather than the
degenerate optimum of four identical rungs. It is a HYPOTHESIS: the measured
ordering is not claimed in advance and must come from a pilot. Prompt renderer,
answer syntax, canonicalizer and verifier are unchanged.

Provenance note: the screening pilot mislabelled every row because it patched a
module global while the recorded profile was computed from that global at import
time. Here each row's profile is built from the preset actually used to generate
it, and verify_rows recomputes it from the same table.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import modebench_scale_candidates as original

SCHEMA = 'modebench_scale_l5_graph_r4_per_cell_candidate_laws_v1'
DOMAINS = ('graph_coloring',)
MAX_PROPOSALS_PER_ROW = 100_000
PLANTED_COLORS = 3
EDGE_PROBABILITY_INTERVAL = (.32, .65)
MINIMUM_EDGES = 1
# The registered exact supports. 5 and 18 appear in the eval and train histograms
# but not in dev; they are carried so development pools can cover every cell.
SUPPORTS = (4, 5, 6, 8, 9, 12, 18)
# Cells with no measurement of their own borrow their nearest measured neighbour:
# support 5 follows support 6, support 18 follows support 12. Both are under one
# percent of the dev weight.
BORROWED = {5: 6, 18: 12}

_C = {
    'small_known': (5, 3, 'known_first'),
    'small_random': (5, 3, 'random'),
    'medium': (6, 3, 'random'),
    'medium_wide': (6, 4, 'random'),
    'large': (8, 3, 'random'),
    'tall_wide': (7, 4, 'random'),
}
# Per-tier, per-cell presets. Read each row as one tier's assignment across cells.
_TABLE = (
    {4: 'small_known', 6: 'medium', 8: 'tall_wide', 9: 'medium', 12: 'medium_wide'},
    {4: 'small_known', 6: 'large', 8: 'medium_wide', 9: 'medium_wide', 12: 'medium_wide'},
    {4: 'large', 6: 'small_random', 8: 'medium_wide', 9: 'large', 12: 'medium'},
    {4: 'large', 6: 'medium_wide', 8: 'medium_wide', 9: 'small_random', 12: 'medium_wide'},
)


def _expand(row):
    filled = dict(row)
    for cell, source in BORROWED.items():
        filled[cell] = row[source]
    return {support: _C[filled[support]] for support in SUPPORTS}


CELL_PRESETS = tuple(_expand(row) for row in _TABLE)


def _profile(tier, support):
    vertices, hidden, ordering = CELL_PRESETS[tier][support]
    return {'answer_mode_count': support, 'vertices': vertices, 'hidden_vertices': hidden,
            'vertex_order': ordering, 'planted_colors': PLANTED_COLORS,
            'edge_probability_interval': list(EDGE_PROBABILITY_INTERVAL),
            'isolated_hidden_vertices': True, 'minimum_edges': MINIMUM_EDGES,
            'graded': 'per_answer_mode_count_cell',
            'borrowed_from_cell': BORROWED.get(support),
            'sampling': 'rejection_conditioned_on_exact_support_per_cell_preset'}


PROFILES = {'graph_coloring': [
    {str(support): _profile(tier, support) for support in SUPPORTS} for tier in range(4)]}
identity = original.identity


def source_paths():
    return sorted({Path(__file__).resolve(), *original.source_paths()})


def _seed(*parts):
    payload = json.dumps([SCHEMA, *parts], sort_keys=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest(), 'big')


def _annotate(row, tier, support, index):
    row = dict(row)
    # Built from the preset this row actually used, never from a module-level cache.
    row.update(scale_candidate_generator=SCHEMA, scale_candidate_tier=tier,
               scale_candidate_profile=json.dumps(_profile(tier, support),
                                                  sort_keys=True, separators=(',', ':')),
               scale_origin_metadata='{}', scale_cell_index=index,
               scale_bridge_law_version=SCHEMA)
    return row


def _graph_rows(target, excluded, seed, tag, tier):
    blocked, rows = set(excluded), []
    for support, count in sorted(target.items()):
        if support not in SUPPORTS:
            raise ValueError('graph r4 requires a registered exact support')
        n, hidden_count, ordering = CELL_PRESETS[tier][support]
        rng = random.Random(_seed(seed, tier, support, 'graph_coloring'))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                planted = [rng.randint(1, PLANTED_COLORS) for _ in range(n)]
                hidden = (set(range(n - hidden_count, n)) if ordering == 'known_first'
                          else set(rng.sample(range(n), hidden_count)))
                partial = [None if v in hidden else color for v, color in enumerate(planted)]
                probability = rng.uniform(*EDGE_PROBABILITY_INTERVAL)
                edges = [[u + 1, v + 1] for u, v in combinations(range(n), 2)
                         if planted[u] != planted[v] and rng.random() < probability]
                if len(edges) < MINIMUM_EDGES:
                    continue
                if original.graph_completion_count(n, edges, partial, cap=support) != support:
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
                rows.append(_annotate(row, tier, support, index))
                break
            else:
                raise RuntimeError(f'graph r4 exhausted fixed proposal budget: tier{tier}/support{support}')
    return rows


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    if domain not in DOMAINS or type(tier) is not int or tier not in range(4):
        raise ValueError('expected graph_coloring domain and integer tier0..3')
    if type(seed) is not int or seed < 0 or type(multiplier) is not int or multiplier < 1:
        raise ValueError('nonnegative integer seed and positive integer multiplier required')
    if joint_target is not None:
        raise ValueError('graph r4 law does not accept joint histograms')
    if any(type(support) is not int or support < 2 or type(count) is not int or count < 0
           for support, count in target.items()):
        raise ValueError('exact supports and quotas must be nonnegative integers with supports>=2')
    required = Counter({support: count * multiplier for support, count in target.items() if count})
    rows = _graph_rows(required, excluded, seed, tag, tier)
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    ids = {identity(domain, row) for row in rows}
    if len(ids) != len(rows) or ids & set(excluded):
        raise RuntimeError('graph r4 candidate semantic identity overlap')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('graph r4 exact support histogram drift')
    return rows


def verify_rows(domain, rows):
    if domain not in DOMAINS:
        raise ValueError(domain)
    for row in rows:
        tier = row.get('scale_candidate_tier')
        if type(tier) is not int or tier not in range(4):
            raise RuntimeError('graph r4 profile audit failed: invalid tier')
        support = row.get('answer_mode_count')
        if support not in SUPPORTS:
            raise RuntimeError('graph r4 profile audit failed: unregistered support')
        expected = json.dumps(_profile(tier, support), sort_keys=True, separators=(',', ':'))
        if (row.get('scale_candidate_generator') != SCHEMA
                or row.get('scale_bridge_law_version') != SCHEMA
                or row.get('scale_candidate_profile') != expected):
            raise RuntimeError('graph r4 profile audit failed: metadata changed')
        spec = json.loads(row['answer'])
        n, hidden, ordering = CELL_PRESETS[tier][support]
        good = (spec['n'] == n and spec['partial_colors'].count(None) == hidden
                and len(spec['edges']) >= MINIMUM_EDGES and spec['num_completions'] == support)
        if ordering == 'known_first':
            good = good and all(c is not None for c in spec['partial_colors'][:n - hidden]) and all(
                c is None for c in spec['partial_colors'][n - hidden:])
        if not good:
            raise RuntimeError('graph r4 profile audit failed: structural law changed')
    return {**original.verify_rows(domain, rows), 'per_cell_structural_profile': True}
