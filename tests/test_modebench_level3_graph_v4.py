from collections import Counter
import itertools
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
from modebench_level3_graph_v4 import (
    build_pool, graph_stream, n4_identities, row_identity, size_weights, vertex_count,
)
from oat_drgrpo.math_grader import _verify_graph_coloring_answer


def ordered(rows, support):
    return sorted((row for row in rows if row['answer_mode_count'] == support),
                  key=lambda row: row['level3_cell_index'])


@pytest.mark.parametrize('difficulty', range(4))
def test_quota_and_other_cells_cannot_change_generation_prefix(difficulty):
    one = build_pool('graph_coloring', Counter({8: 1}), set(), 824100,
                     'quota', difficulty, 1)
    eight = build_pool('graph_coloring', Counter({8: 8}), set(), 824100,
                       'quota', difficulty, 1)
    assert ordered(eight, 8)[:1] == one
    other_cells = build_pool('graph_coloring', Counter({4: 9, 8: 8, 12: 3}), set(),
                             824100, 'quota', difficulty, 1)
    assert ordered(other_cells, 8) == ordered(eight, 8)
    for row in eight:
        assert json.loads(row['answer'])['n'] == vertex_count(8, difficulty, 824100,
                                                             row['level3_cell_index'])


@pytest.mark.parametrize('difficulty', range(4))
def test_verifier_histogram_and_exclusions(difficulty):
    target = Counter({4: 3, 5: 2, 6: 3, 8: 3, 9: 3, 12: 3, 18: 1})
    rows = build_pool('graph_coloring', target, set(), 765000 + difficulty,
                      'verify', difficulty, 1)
    assert Counter(row['answer_mode_count'] for row in rows) == target
    excluded = {row_identity('graph_coloring', row) for row in rows}
    fresh = build_pool('graph_coloring', target, excluded, 765000 + difficulty,
                       'fresh', difficulty, 1)
    assert not excluded & {row_identity('graph_coloring', row) for row in fresh}
    for row in rows:
        spec = json.loads(row['answer'])
        hidden = {index + 1 for index, color in enumerate(spec['partial_colors']) if color is None}
        assert len(hidden) == 3
        if difficulty == 0 and row['answer_mode_count'] != 5:
            assert all(not (u in hidden and v in hidden) for u, v in spec['edges'])
        actual = sum(_verify_graph_coloring_answer(''.join(map(str, colors)), spec)
                     for colors in itertools.product((1, 2, 3), repeat=3))
        assert actual == row['answer_mode_count']


def test_size_weights_are_fixed_conditioned_probabilities():
    assert size_weights(4, 1) == ((4, 1), (5, 9))
    assert size_weights(4, 3) == ((4, 1), (5, 3), (6, 6))
    assert size_weights(9, 3) == ((5, 3), (6, 6))
    assert size_weights(9, 1) == ((5, 9),)
    for difficulty in range(4):
        assert size_weights(5, difficulty) == ((5, 1),)
        assert size_weights(18, difficulty) == (((5, 1),) if difficulty == 0 else ((6, 1),))


def test_depleted_chosen_size_fails_without_fallback():
    seed = next(seed for seed in range(100) if vertex_count(4, 1, seed, 0) == 4)
    with pytest.raises(RuntimeError, match='no size fallback'):
        next(graph_stream(4, set(n4_identities(4)), seed, 1))
