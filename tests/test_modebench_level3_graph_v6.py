"""Shared-color proposal law, original-verifier, and fixed-stream guards."""
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_graph_v5 as previous
import modebench_level3_graph_v6 as graph
from oat_drgrpo.math_grader import _verify_graph_coloring_answer


def cell(rows, support):
    return sorted((row for row in rows if row['answer_mode_count'] == support),
                  key=lambda row: row['level3_cell_index'])


@pytest.mark.parametrize('difficulty', range(4))
def test_per_cell_sampling_prefix_is_quota_independent_with_exclusions(difficulty):
    target = Counter({support: 2 for support in graph.SUPPORTS})
    excluded_rows = graph.build_pool('graph_coloring', target, set(), 949001,
                                     'excluded', difficulty, 1)
    excluded = {graph.row_identity('graph_coloring', row) for row in excluded_rows}
    short = graph.build_pool('graph_coloring', target, excluded, 949001, 'prefix', difficulty, 1)
    long = graph.build_pool('graph_coloring', target, excluded, 949001, 'prefix', difficulty, 3)
    for support in graph.SUPPORTS:
        assert cell(long, support)[:2] == cell(short, support)


@pytest.mark.parametrize('difficulty', range(4))
def test_exact_modes_original_prompt_verifier_witnesses_and_exclusions(difficulty):
    target = Counter({support: 3 for support in graph.SUPPORTS})
    rows = graph.build_pool('graph_coloring', target, set(), 959002 + difficulty,
                            'verifier', difficulty, 1)
    assert Counter(row['answer_mode_count'] for row in rows) == target
    for row in rows:
        spec = json.loads(row['answer'])
        support = row['answer_mode_count']
        n, hidden_count, independent = graph.structure(support, difficulty)
        assert spec['n'] == n >= 5
        assert row['problem'] == graph._graph_prompt(n, spec['edges'], spec['partial_colors'])
        hidden = {i + 1 for i, color in enumerate(spec['partial_colors']) if color is None}
        assert len(hidden) == hidden_count
        assert row['level3_graph_proposal_law'] == graph.proposal_law(support, difficulty)
        if independent:
            assert all(not (u in hidden and v in hidden) for u, v in spec['edges'])
        if graph.proposal_law(support, difficulty) == 'shared_visible_color':
            assert len({color for color in spec['partial_colors'] if color is not None}) == 1
            assert all((u in hidden) != (v in hidden) for u, v in spec['edges'])
        witnesses = [''.join(map(str, fill))
                     for fill in itertools.product((1, 2, 3), repeat=hidden_count)
                     if _verify_graph_coloring_answer(''.join(map(str, fill)), spec)]
        assert len(witnesses) == support
        assert spec['num_solutions'] == graph.graph_completion_count(n, spec['edges'], [None] * n)
    excluded = {graph.row_identity('graph_coloring', row) for row in rows}
    fresh = graph.build_pool('graph_coloring', target, excluded, 959002 + difficulty,
                             'fresh', difficulty, 1)
    assert not excluded & {graph.row_identity('graph_coloring', row) for row in fresh}


@pytest.mark.parametrize('difficulty', range(4))
@pytest.mark.parametrize('support', sorted(graph.SUPPORTS))
def test_frozen_structures_and_mathematical_exceptions(difficulty, support):
    assert graph.structure(support, difficulty) == previous.structure(support, difficulty)
    law = graph.proposal_law(support, difficulty)
    if difficulty >= 2 or support == 5:
        assert law == 'v5_coupled'
    elif support == 9:
        assert law == 'v5_independent_random_visible_colors'
    else:
        assert law == 'shared_visible_color'
    if law != 'shared_visible_color':
        n, hidden_count, independent = graph.structure(support, difficulty)
        for seed in range(12):
            assert graph._candidate(n, hidden_count, independent, random.Random(seed)) == \
                   previous._candidate(n, hidden_count, independent, random.Random(seed))


@pytest.mark.parametrize('n,hidden_count', [(5, 2), (5, 3), (6, 2), (6, 3)])
def test_shared_color_is_one_uniform_draw_and_each_cross_edge_is_bernoulli_half(n, hidden_count):
    class ScriptedRNG:
        def __init__(self, color):
            self.color = color
            self.color_calls = 0
            self.edge_calls = 0

        def sample(self, population, count):
            assert list(population) == list(range(n)) and count == hidden_count
            return list(range(0, n, 2))[:count]

        def randint(self, low, high):
            assert (low, high) == (1, 3)
            self.color_calls += 1
            return self.color

        def random(self):
            self.edge_calls += 1
            return (0.0, 0.499, 0.5, 0.999)[(self.edge_calls - 1) % 4]

    proposals = []
    hidden = set(range(0, n, 2))
    hidden = set(sorted(hidden)[:hidden_count])
    cross_edges = [(u + 1, v + 1) for u, v in itertools.combinations(range(n), 2)
                   if (u in hidden) != (v in hidden)]
    for color in (1, 2, 3):
        rng = ScriptedRNG(color)
        edges, partial = graph._candidate(n, hidden_count, True, rng, monochrome=True)
        assert rng.color_calls == 1
        assert rng.edge_calls == hidden_count * (n - hidden_count)
        assert edges == [list(edge) for index, edge in enumerate(cross_edges) if index % 4 < 2]
        assert partial == [None if i in hidden else color for i in range(n)]
        proposals.append((edges, graph.graph_completion_count(n, edges, partial)))
    assert proposals[0] == proposals[1] == proposals[2]


def test_failed_rejection_never_falls_back_to_another_structure(monkeypatch):
    monkeypatch.setattr(graph, 'MAX_ROW_ATTEMPTS', 2)
    monkeypatch.setattr(graph, '_candidate', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='no size or topology fallback'):
        next(graph.graph_stream(4, set(), 1, 0))
