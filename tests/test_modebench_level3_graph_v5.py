"""Exact verifier, mathematical-feasibility, and fixed-proposal guards."""
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_graph_v5 as graph
from oat_drgrpo.math_grader import _verify_graph_coloring_answer


def cell(rows, support):
    return sorted((row for row in rows if row['answer_mode_count'] == support),
                  key=lambda row: row['level3_cell_index'])


@pytest.mark.parametrize('difficulty', range(4))
def test_per_cell_sampling_prefix_does_not_depend_on_requested_quotas(difficulty):
    short = graph.build_pool('graph_coloring', Counter({4: 1}), set(), 919001, 'prefix', difficulty, 1)
    long = graph.build_pool('graph_coloring', Counter({4: 4, 8: 3, 18: 1}), set(), 919001, 'prefix', difficulty, 1)
    assert cell(long, 4)[:1] == short


@pytest.mark.parametrize('difficulty', range(4))
def test_exact_modes_original_verifier_witnesses_and_exclusions(difficulty):
    target = Counter({support: 2 for support in graph.SUPPORTS})
    rows = graph.build_pool('graph_coloring', target, set(), 929002 + difficulty, 'verifier', difficulty, 1)
    assert Counter(row['answer_mode_count'] for row in rows) == target
    for row in rows:
        spec = json.loads(row['answer'])
        n, hidden_count, independent = graph.structure(row['answer_mode_count'], difficulty)
        assert spec['n'] == n >= 5
        hidden = {i + 1 for i, color in enumerate(spec['partial_colors']) if color is None}
        assert len(hidden) == hidden_count
        if independent:
            assert all(not (u in hidden and v in hidden) for u, v in spec['edges'])
        witnesses = [''.join(map(str, fill)) for fill in itertools.product((1, 2, 3), repeat=hidden_count)
                     if _verify_graph_coloring_answer(''.join(map(str, fill)), spec)]
        assert len(witnesses) == row['answer_mode_count']
    excluded = {graph.row_identity('graph_coloring', row) for row in rows}
    fresh = graph.build_pool('graph_coloring', target, excluded, 929002 + difficulty, 'fresh', difficulty, 1)
    assert not excluded & {graph.row_identity('graph_coloring', row) for row in fresh}


def test_failed_rejection_never_falls_back_to_another_structure(monkeypatch):
    monkeypatch.setattr(graph, 'MAX_ROW_ATTEMPTS', 2)
    monkeypatch.setattr(graph, '_candidate', lambda *args: None)
    with pytest.raises(RuntimeError, match='no size or topology fallback'):
        next(graph.graph_stream(4, set(), 1, 0))
