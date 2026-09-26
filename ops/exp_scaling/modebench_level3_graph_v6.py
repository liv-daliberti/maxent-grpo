"""Graph proposals combining shared-given-color anchors with coupled tasks.

Structural hypothesis only: one uniformly chosen visible color makes repeated
local constraints easier and may raise pass@1 relative to pass@8. The shared
color is uniform over 1, 2, 3; color 1 is never preferred. Support 5 retains the
v5 coupled proposal because independent hidden choices cannot multiply to 5.
Support 9 retains v5 independent random visible colors because the monochrome
law would require an empty graph and severely limit semantic identity capacity.
Larger coupled presets supply harder mixture components. Vertex labels remain
random, and the original prompt, verifier, and canonical support are unchanged.
No quota-dependent ordering, row-outcome selection, or fallback is permitted.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from modebench_level3_discrete import _graph_prompt, graph_completion_count, row_identity
from make_modebench_data import _valid_graph_colorings

SCHEMA = 'modebench_level3_graph_candidate_v6'
SUPPORTS = frozenset({4, 5, 6, 8, 9, 12, 18})
PRESETS = {
    0: 'n5_shared_visible_color_except_support_five_and_nine',
    1: 'n6_shared_visible_color_except_support_five_and_nine',
    2: 'n6_three_hidden_original_coupled_proposal',
    3: 'n7_four_hidden_original_coupled_proposal',
}
MAX_ROW_ATTEMPTS = 100_000


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest(), 'big')


def structure(support, difficulty):
    """Fixed support-conditioned structure, independent of quotas/exclusions."""
    if difficulty not in PRESETS or support not in SUPPORTS:
        raise ValueError('expected difficulty 0..3 and a frozen graph support cell')
    if difficulty < 2:
        return 5 + difficulty, 2 if support in (4, 6, 9) else 3, support != 5
    return (6, 3, False) if difficulty == 2 else (7, 4, False)


def proposal_law(support, difficulty):
    """Explicit support-conditioned exceptions; never depend on row outcomes."""
    structure(support, difficulty)  # Validate the frozen preset/support domain.
    if difficulty >= 2 or support == 5:
        return 'v5_coupled'
    return 'v5_independent_random_visible_colors' if support == 9 else 'shared_visible_color'


@lru_cache(maxsize=2048)
def _colorings(n, edges):
    return tuple(tuple(colors) for colors in _valid_graph_colorings(n, [list(edge) for edge in edges]))


def _candidate(n, hidden_count, independent, rng, monochrome=False):
    hidden = set(rng.sample(range(n), hidden_count))
    if monochrome:
        if not independent:
            raise ValueError('shared visible color requires independent hidden vertices')
        visible_color = rng.randint(1, 3)
        partial = [None if index in hidden else visible_color for index in range(n)]
        edges = [[u + 1, v + 1] for u, v in combinations(range(n), 2)
                 if (u in hidden) != (v in hidden) and rng.random() < .5]
        return edges, partial
    if independent:
        partial = [None if index in hidden else rng.randint(1, 3) for index in range(n)]
        edges = []
        for u, v in combinations(range(n), 2):
            if u in hidden and v in hidden:
                continue
            if partial[u] is not None and partial[u] == partial[v]:
                continue
            if rng.random() < .5:
                edges.append([u + 1, v + 1])
        return edges, partial
    possible = list(combinations(range(1, n + 1), 2))
    edge_count = rng.randint(n - 2, min(n + 2, len(possible)))
    edges = tuple(sorted(rng.sample(possible, edge_count)))
    colorings = _colorings(n, edges)
    if not colorings:
        return None
    coloring = rng.choice(colorings)
    partial = [None if index in hidden else color for index, color in enumerate(coloring)]
    return [list(edge) for edge in edges], partial


def graph_stream(support, excluded, seed, difficulty):
    """A deterministic per-cell stream, extended unchanged by larger quotas."""
    n, hidden_count, independent = structure(support, difficulty)
    blocked = set(excluded)
    index = 0
    while True:
        rng = random.Random(_seed(SCHEMA, seed, difficulty, support, index, 'topology'))
        for _ in range(MAX_ROW_ATTEMPTS):
            candidate = _candidate(n, hidden_count, independent, rng,
                                   monochrome=proposal_law(support, difficulty) == 'shared_visible_color')
            if candidate is None:
                continue
            edges, partial = candidate
            if graph_completion_count(n, edges, partial, cap=support) != support:
                continue
            identity = ('graph_coloring', n, tuple(tuple(edge) for edge in edges),
                        ''.join('?' if color is None else str(color) for color in partial))
            if identity in blocked:
                continue
            blocked.add(identity)
            yield n, edges, partial
            break
        else:
            raise RuntimeError(f'graph v6 preset {difficulty}, support {support}, cell index {index} '
                               'exhausted the fixed rejection budget; no size or topology fallback is permitted')
        index += 1


def build_pool(domain, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[dict]:
    """Return exact multiplied support counts with no model-outcome selection."""
    if domain != 'graph_coloring':
        raise ValueError('graph v6 only supports graph_coloring')
    if difficulty not in PRESETS or type(multiplier) is not int or multiplier < 1:
        raise ValueError('difficulty must be 0..3 and multiplier a positive integer')
    required = Counter({int(support): int(count) * multiplier for support, count in target.items() if count})
    if any(support not in SUPPORTS or count < 0 for support, count in required.items()):
        raise ValueError('expected nonnegative counts in frozen graph support cells 4,5,6,8,9,12,18')
    rows = []
    for support, count in sorted(required.items()):
        stream = graph_stream(support, excluded, seed, difficulty)
        for index in range(count):
            n, edges, partial = next(stream)
            spec = {
                'verifier': domain, 'n': n, 'edges': edges, 'partial_colors': partial,
                'source': SCHEMA, 'instance_id': f'{tag}-{seed}-support-{support}-{index}',
                'num_completions': support,
                'num_solutions': graph_completion_count(n, edges, [None] * n),
            }
            rows.append({
                'problem': _graph_prompt(n, edges, partial), 'answer': json.dumps(spec, sort_keys=True),
                'modebench_task': domain, 'answer_mode_count': support,
                'answer_mode_split': tag, 'level3_difficulty': difficulty,
                'level3_generator': SCHEMA, 'level3_graph_preset': PRESETS[difficulty],
                'level3_cell_index': index,
                'level3_graph_proposal_law': proposal_law(support, difficulty),
            })
    rows.sort(key=lambda row: _seed(SCHEMA, seed, 'output_order', row_identity(domain, row)))
    identities = {row_identity(domain, row) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError('graph v6 semantic identity disjointness failed')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('graph v6 support histogram differs from target')
    return rows
