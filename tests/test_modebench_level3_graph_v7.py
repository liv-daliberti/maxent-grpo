"""Graph-v7 exact support, original grading, finite capacity and proposal invariants."""
from collections import Counter
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_graph_v7 as graph
import materialize_modebench_level3_graph_v7 as audit


@pytest.mark.parametrize('difficulty', range(4))
def test_all_support_cells_preserve_original_three_digit_verifier(difficulty):
    rows = graph.build_pool('graph_coloring', Counter({support: 3 for support in graph.SUPPORTS}),
                            set(), 729101, 'test', difficulty, multiplier=1)
    assert audit.verify_witnesses(rows) == 3 * sum(graph.SUPPORTS)
    for row in rows:
        spec = json.loads(row['answer'])
        assert spec['n'] == (6 if difficulty == 3 else 5)
        assert sum(color is None for color in spec['partial_colors']) == 3
        if difficulty < 3:
            assert graph.row_identity('graph_coloring', row) in audit.catalogue(difficulty, row['answer_mode_count'])


@pytest.mark.parametrize('difficulty', range(4))
def test_each_support_stream_extends_without_quota_or_other_cell_dependence(difficulty):
    small = graph.build_pool('graph_coloring', Counter({4: 5, 9: 2}), set(), 729102, 'test', difficulty, multiplier=1)
    large = graph.build_pool('graph_coloring', Counter({4: 8, 9: 4, 18: 2}), set(), 729102, 'test', difficulty, multiplier=1)
    for support in (4, 9):
        prefix = sorted((row for row in small if row['answer_mode_count'] == support), key=lambda row: row['level3_cell_index'])
        full = sorted((row for row in large if row['answer_mode_count'] == support), key=lambda row: row['level3_cell_index'])
        assert prefix == full[:len(prefix)]


@pytest.mark.parametrize('difficulty', range(4))
def test_exclusions_protect_identity_without_changing_fixed_structure(difficulty):
    first = graph.build_pool('graph_coloring', Counter({4: 8, 9: 3}), set(), 729103, 'test', difficulty, multiplier=1)
    blocked = {graph.row_identity('graph_coloring', row) for row in first}
    later = graph.build_pool('graph_coloring', Counter({4: 8, 9: 3}), blocked, 729103, 'test', difficulty, multiplier=1)
    assert not blocked & {graph.row_identity('graph_coloring', row) for row in later}
    assert audit.verify_witnesses(later) == 8 * 4 + 3 * 9


class ColorRelabeledRNG:
    """Same structural randomness, with a global permutation of color draws."""
    def __init__(self, seed, colors):
        self.base = random.Random(seed)
        self.colors = colors
    def randint(self, low, high):
        value = self.base.randint(low, high)
        return self.colors[value - 1] if (low, high) == (1, 3) else value
    def __getattr__(self, name):
        return getattr(self.base, name)


@pytest.mark.parametrize('difficulty', range(4))
def test_actual_proposal_is_equivariant_under_all_six_color_permutations(difficulty):
    for support in graph.SUPPORTS:
        for seed in range(8):
            original = graph._candidate(support, difficulty, ColorRelabeledRNG(seed, (1, 2, 3)))
            for labels in permutations((1, 2, 3)):
                mapped = graph._candidate(support, difficulty, ColorRelabeledRNG(seed, labels))
                if original is None:
                    assert mapped is None
                    continue
                edges, partial = original
                assert mapped == (edges, [None if value is None else labels[value - 1] for value in partial])


def test_exact_sparse_anchor_capacity_includes_support_nine():
    expected = {4: 720, 5: 360, 6: 720, 8: 720, 9: 180, 12: 1080, 18: 540}
    assert {support: len(audit.catalogue(0, support)) for support in graph.SUPPORTS} == expected
    assert len(audit.catalogue(1, 9) | audit.catalogue(0, 9) | audit.catalogue(2, 9)) == 360


@pytest.mark.parametrize('difficulty', [0, 1, 2])
def test_complete_finite_catalogue_has_exact_support_and_global_color_symmetry(difficulty):
    for support in graph.SUPPORTS:
        catalog = audit.catalogue(difficulty, support)
        assert catalog
        for _, n, edges, text in catalog:
            partial = [None if value == '?' else int(value) for value in text]
            assert graph.graph_completion_count(n, list(edges), partial) == support
            for labels in ((2, 3, 1), (2, 1, 3)):
                renamed = ''.join('?' if value == '?' else str(labels[int(value) - 1]) for value in text)
                assert ('graph_coloring', n, edges, renamed) in catalog


def test_exhausted_cell_raises_without_size_or_topology_fallback(monkeypatch):
    monkeypatch.setattr(graph, 'MAX_ROW_ATTEMPTS', 20)
    with pytest.raises(RuntimeError, match='no size/topology fallback'):
        next(graph.graph_stream(9, set(audit.catalogue(0, 9)), 729104, 0))


@pytest.mark.parametrize('target', [Counter({4: -1}), Counter({7: 1}), Counter({4: 1.5})])
def test_invalid_support_or_quotas_rejected(target):
    with pytest.raises(ValueError, match='nonnegative integer quotas'):
        graph.build_pool('graph_coloring', target, set(), 1, 'test', 0, multiplier=1)
