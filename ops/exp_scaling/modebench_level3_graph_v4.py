"""Small-graph presets with quota-independent per-row size/topology sampling.

Sizes are drawn once per output row from frozen support-conditioned probabilities.
Rejection sampling stays at that size and never falls back after finite identity
exhaustion. Every support cell has its own deterministic random streams.

Before any V4 model outcomes, the n=4 probability was fixed at 10% rather
than V3's 20% to reserve the finite small-graph inventory for all four pilot
pools plus fresh final train/eval. The pre-generation inventory audit found
72 fresh n=4 support-4 identities and 39 fresh n=4 support-6 identities.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from modebench_level3_discrete import _graph_prompt, graph_completion_count, row_identity
from make_modebench_data import _valid_graph_colorings

SCHEMA = 'modebench_level3_graph_candidate_v4'
SUPPORTS = frozenset({4, 5, 6, 8, 9, 12, 18})
PRESETS = {
    0: 'five_vertices_independent_hidden_except_prime_five_support',
    1: 'per_row_ten_percent_four_otherwise_five_when_feasible',
    2: 'five_vertices_original_coupled_topology_distribution',
    3: 'per_row_ten_percent_four_thirty_percent_five_sixty_percent_six_when_feasible',
}
MAX_ROW_ATTEMPTS = 100_000


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest(), 'big')


def size_weights(support: int, difficulty: int) -> tuple[tuple[int, int], ...]:
    """Fixed weights after conditioning on the support's structural feasibility."""
    if difficulty not in PRESETS or support not in SUPPORTS:
        raise ValueError('expected difficulty 0..3 and a frozen graph support cell')
    if difficulty == 0 or support == 5:
        return ((5, 1),)
    if support == 18:
        return ((6, 1),)
    if difficulty == 2:
        return ((5, 1),)
    weights = ((4, 1), (5, 9)) if difficulty == 1 else ((4, 1), (5, 3), (6, 6))
    if support == 9:
        weights = tuple((n, weight) for n, weight in weights if n >= 5)
    return weights


def vertex_count(support: int, difficulty: int, seed: int, cell_index: int) -> int:
    weights = size_weights(support, difficulty)
    rng = random.Random(_seed(SCHEMA, seed, difficulty, support, cell_index, 'size'))
    return rng.choices([n for n, _ in weights], weights=[weight for _, weight in weights], k=1)[0]


@lru_cache(maxsize=8192)
def _colorings(n, edges):
    return tuple(tuple(colors) for colors in _valid_graph_colorings(n, [list(edge) for edge in edges]))


def _candidate(n, rng, local):
    """The V3 topology proposal, with cached full-coloring enumeration only."""
    hidden = set(rng.sample(range(n), 3))
    if local:
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
    edge_count = rng.randint(n - 2, min(8, len(possible)))
    edges = tuple(sorted(rng.sample(possible, edge_count)))
    colorings = _colorings(n, edges)
    if not colorings:
        return None
    coloring = rng.choice(colorings)
    partial = [None if index in hidden else color for index, color in enumerate(coloring)]
    return [list(edge) for edge in edges], partial


def _identity(n, edges, partial):
    return ('graph_coloring', n, tuple(tuple(edge) for edge in edges),
            ''.join('?' if color is None else str(color) for color in partial))


@lru_cache(maxsize=4)
def n4_identities(support):
    """Complete finite inventory for the coupled four-vertex proposal law."""
    result = set()
    possible = list(combinations(range(1, 5), 2))
    for count in range(2, 7):
        for edge_tuples in combinations(possible, count):
            edges = [list(edge) for edge in edge_tuples]
            for shown in range(4):
                for color in (1, 2, 3):
                    partial = [None] * 4
                    partial[shown] = color
                    if graph_completion_count(4, edges, partial, cap=support) == support:
                        result.add(_identity(4, edges, partial))
    return frozenset(result)


def graph_stream(support: int, excluded: set, seed: int, difficulty: int) -> Iterator[tuple]:
    """Yield an unchanged per-cell stream regardless of requested row quotas."""
    weights = size_weights(support, difficulty)
    blocked = set(excluded)
    available_n4 = set(n4_identities(support)) - blocked if any(n == 4 for n, _ in weights) else set()
    local = difficulty == 0 and support != 5
    index = 0
    while True:
        n = vertex_count(support, difficulty, seed, index)
        if n == 4 and not available_n4:
            raise RuntimeError(f'graph v4 support {support} exhausted all fresh n=4 identities at cell index {index}; no size fallback is permitted')
        # Topology rejection cannot consume the next row's size randomness.
        rng = random.Random(_seed(SCHEMA, seed, difficulty, support, index, 'topology'))
        for attempt in range(MAX_ROW_ATTEMPTS):
            candidate = _candidate(n, rng, local)
            if candidate is None:
                continue
            edges, partial = candidate
            if graph_completion_count(n, edges, partial, cap=support) != support:
                continue
            identity = _identity(n, edges, partial)
            if identity in blocked:
                continue
            blocked.add(identity)
            if n == 4:
                available_n4.remove(identity)
            yield n, edges, partial
            break
        else:
            raise RuntimeError(f'graph v4 difficulty {difficulty}, support {support}, chosen n={n} exhausted fixed rejection budget at cell index {index}; no size fallback is permitted')
        index += 1


def build_pool(domain, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[dict]:
    if domain != 'graph_coloring':
        raise ValueError('graph v4 only supports graph_coloring')
    if difficulty not in PRESETS or not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier < 1:
        raise ValueError('difficulty must be 0..3 and multiplier must be a positive integer')
    required = Counter({int(support): int(count) * multiplier for support, count in target.items() if count})
    if any(support not in SUPPORTS or count < 0 for support, count in required.items()):
        raise ValueError('expected nonnegative counts in frozen graph support cells 4,5,6,8,9,12,18')
    rows = []
    for support, count in sorted(required.items()):
        stream = graph_stream(support, excluded, seed, difficulty)
        for index in range(count):
            n, edges, partial = next(stream)
            spec = {
                'verifier': 'graph_coloring', 'n': n, 'edges': edges,
                'partial_colors': partial, 'source': SCHEMA,
                'instance_id': f'{tag}-{seed}-support-{support}-{index}',
                'num_completions': support,
                'num_solutions': graph_completion_count(n, edges, [None] * n),
            }
            rows.append({
                'problem': _graph_prompt(n, edges, partial),
                'answer': json.dumps(spec, sort_keys=True),
                'modebench_task': domain, 'answer_mode_count': support,
                'answer_mode_split': tag, 'level3_difficulty': difficulty,
                'level3_generator': SCHEMA, 'level3_graph_preset': PRESETS[difficulty],
                'level3_cell_index': index,
            })
    rows.sort(key=lambda row: _seed(SCHEMA, seed, 'output_order', row_identity(domain, row)))
    identities = {row_identity(domain, row) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError('graph v4 semantic identity disjointness failed')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('graph v4 support histogram differs from target')
    return rows
