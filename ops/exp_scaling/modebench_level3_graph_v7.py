"""Three-hidden graph laws with sparse visible anchors and simple hidden forests.

All presets keep the original graph prompt and verifier. Presets 0..2 use five
vertices and exactly three missing digits; preset 3 uses six vertices with three
missing digits as a coupled harder bracket. Colors and vertex roles have no
preferred label. Every support cell has its own deterministic stream, with no
quota-dependent proposals, outcome selection, or topology/size fallback.
"""
from __future__ import annotations
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from modebench_level3_discrete import _graph_prompt, graph_completion_count, row_identity

SCHEMA = 'modebench_level3_graph_candidate_v7'
SUPPORTS = frozenset({4, 5, 6, 8, 9, 12, 18})
PRESETS = {
    0: 'n5_three_hidden_minimal_independent_visible_anchors',
    1: 'n5_three_hidden_simple_forest_optional_visible_edge',
    2: 'n5_three_hidden_minimal_anchors_with_legal_visible_edge',
    3: 'n6_three_hidden_coupled_graph',
}
AVAILABILITIES = {4: (1, 2, 2), 6: (1, 2, 3), 8: (2, 2, 2),
                  9: (1, 3, 3), 12: (2, 2, 3), 18: (2, 3, 3)}
MAX_ROW_ATTEMPTS = 100_000


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest(), 'big')


def structure(support, difficulty):
    if difficulty not in PRESETS or support not in SUPPORTS:
        raise ValueError('expected difficulty 0..3 and graph support 4,5,6,8,9,12,18')
    return (6 if difficulty == 3 else 5), 3


def _anchors(visible, partial, forbidden_colors, rng):
    options = [vertices for vertices in combinations(visible, forbidden_colors)
               if len({partial[vertex] for vertex in vertices}) == forbidden_colors]
    return rng.choice(options) if options else None


def _candidate(support, difficulty, rng):
    n, _ = structure(support, difficulty)
    hidden = rng.sample(range(n), 3)  # Ordered roles, uniformly permuted labels.
    visible = [vertex for vertex in range(n) if vertex not in hidden]
    partial = [None if vertex in hidden else rng.randint(1, 3) for vertex in range(n)]
    edges = set()
    def edge(u, v):
        edges.add(tuple(sorted((u + 1, v + 1))))
    def anchor(vertex, count):
        chosen = _anchors(visible, partial, count, rng)
        if chosen is None:
            return False
        for other in chosen:
            edge(vertex, other)
        return True
    if difficulty == 3:
        possible = list(combinations(range(1, n + 1), 2))
        selected = rng.sample(possible, rng.randint(n - 2, n + 2))
        return [list(pair) for pair in sorted(selected)], partial
    if support == 5:
        # Hidden path endpoints forbid distinct visible colors: center choices
        # contribute 2 + 2 + 1 completions. The center has no visible anchor.
        choices = [(u, v) for u in visible for v in visible if partial[u] != partial[v]]
        if not choices:
            return None
        left, right = rng.choice(choices)
        edge(hidden[0], hidden[1]); edge(hidden[1], hidden[2])
        edge(hidden[0], left); edge(hidden[2], right)
    elif difficulty == 1:
        # Rooted forests make the support an explicit product along the tree.
        root_forbidden = {4: 2, 6: 2, 8: 1, 9: 2, 12: 0, 18: 0}[support]
        if not anchor(hidden[0], root_forbidden):
            return None
        if support in (4, 6, 8, 12, 18):
            edge(hidden[0], hidden[1])
        if support in (4, 8, 12):
            edge(hidden[1], hidden[2])
    else:
        for vertex, available in zip(hidden, AVAILABILITIES[support]):
            if not anchor(vertex, 3 - available):
                return None
    # This edge is already satisfied by visible colors and never changes the
    # hidden constraints. Its optional law in preset1 is fixed before sampling.
    legal_visible = [(u, v) for u, v in combinations(visible, 2) if partial[u] != partial[v]]
    if difficulty == 2:
        if not legal_visible:
            return None
        edge(*rng.choice(legal_visible))
    elif difficulty == 1 and legal_visible and rng.random() < .5:
        edge(*rng.choice(legal_visible))
    return [list(pair) for pair in sorted(edges)], partial


def graph_stream(support, excluded, seed, difficulty):
    n, _ = structure(support, difficulty)
    blocked = set(excluded)
    index = 0
    while True:
        rng = random.Random(_seed(SCHEMA, seed, difficulty, support, index, 'proposal'))
        for _ in range(MAX_ROW_ATTEMPTS):
            candidate = _candidate(support, difficulty, rng)
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
            raise RuntimeError(f'graph v7 preset {difficulty}, support {support}, index {index} exhausted '
                               'fixed rejection budget; no size/topology fallback permitted')
        index += 1


def build_pool(domain, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[dict]:
    if domain != 'graph_coloring' or difficulty not in PRESETS:
        raise ValueError('graph v7 requires graph_coloring and difficulty 0..3')
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError('multiplier must be a positive integer')
    if any(type(count) is not int or count < 0 or int(support) not in SUPPORTS for support, count in target.items()):
        raise ValueError('expected nonnegative integer quotas in registered support cells')
    required = Counter({int(support): count * multiplier for support, count in target.items() if count})
    rows = []
    for support, count in sorted(required.items()):
        stream = graph_stream(support, excluded, seed, difficulty)
        for index in range(count):
            n, edges, partial = next(stream)
            spec = {'verifier': domain, 'n': n, 'edges': edges, 'partial_colors': partial,
                    'source': SCHEMA, 'instance_id': f'{tag}-{seed}-support-{support}-{index}',
                    'num_completions': support, 'num_solutions': graph_completion_count(n, edges, [None] * n)}
            rows.append({'problem': _graph_prompt(n, edges, partial), 'answer': json.dumps(spec, sort_keys=True),
                         'modebench_task': domain, 'answer_mode_count': support, 'answer_mode_split': tag,
                         'level3_difficulty': difficulty, 'level3_generator': SCHEMA,
                         'level3_graph_preset': PRESETS[difficulty], 'level3_cell_index': index})
    rows.sort(key=lambda row: _seed(SCHEMA, seed, 'output_order', row_identity(domain, row)))
    identities = {row_identity(domain, row) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError('graph v7 semantic identity disjointness failed')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('graph v7 support histogram drift')
    return rows
