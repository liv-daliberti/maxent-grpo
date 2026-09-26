"""Verifier-backed structural checks for the refined graph candidate presets."""
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
from modebench_level3_graph_v2 import PRESETS, build_pool, row_identity
from oat_drgrpo.math_grader import _verify_graph_coloring_answer


def test_refined_presets_have_exact_support_and_disjoint_semantic_identities():
    target = Counter({4: 2, 6: 2, 8: 2, 9: 2, 12: 2})
    blocked = set()
    for difficulty, (n, hidden_count) in enumerate(PRESETS):
        rows = build_pool('graph_coloring', target, blocked, 6818000 + difficulty,
                          'refined_test', difficulty, multiplier=2)
        assert Counter(row['answer_mode_count'] for row in rows) == Counter({k: v * 2 for k, v in target.items()})
        identities = {row_identity('graph_coloring', row) for row in rows}
        assert len(identities) == len(rows)
        assert not identities & blocked
        for row in rows:
            spec = json.loads(row['answer'])
            assert spec['n'] == n
            hidden = [i for i, color in enumerate(spec['partial_colors']) if color is None]
            assert len(hidden) == hidden_count
            actual = 0
            for fill in itertools.product((1, 2, 3), repeat=hidden_count):
                actual += _verify_graph_coloring_answer(''.join(map(str, fill)), spec)
            assert actual == row['answer_mode_count']
        blocked |= identities
        repeat = build_pool('graph_coloring', target, blocked, 6818000 + difficulty,
                            'refined_test_repeat', difficulty, multiplier=1)
        assert not blocked & {row_identity('graph_coloring', row) for row in repeat}
        blocked |= {row_identity('graph_coloring', row) for row in repeat}
