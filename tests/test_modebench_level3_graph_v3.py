"""Independent verifier checks for small-graph calibration presets."""
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
from modebench_level3_graph_v3 import build_pool, row_identity, size_targets
from oat_drgrpo.math_grader import _verify_graph_coloring_answer


def test_small_graph_presets_match_verifier_size_quotas_and_prior_exclusions():
    target = Counter({4: 5, 5: 2, 6: 5, 8: 5, 9: 5, 12: 5, 18: 1})
    blocked = set()
    for difficulty in range(4):
        rows = build_pool('graph_coloring', target, blocked, 7918100 + difficulty,
                          'small_test', difficulty, multiplier=1)
        assert Counter(r['answer_mode_count'] for r in rows) == target
        sizes = {}
        for row in rows:
            spec = json.loads(row['answer'])
            hidden = {i + 1 for i, c in enumerate(spec['partial_colors']) if c is None}
            assert len(hidden) == 3
            if difficulty == 0 and row['answer_mode_count'] != 5:
                assert all(not (u in hidden and v in hidden) for u, v in spec['edges'])
            assert spec['n'] >= 5 or row['answer_mode_count'] != 9
            sizes.setdefault(spec['n'], Counter())[row['answer_mode_count']] += 1
            actual = sum(_verify_graph_coloring_answer(''.join(map(str, fill)), spec)
                         for fill in itertools.product((1, 2, 3), repeat=3))
            assert actual == row['answer_mode_count']
        assert sizes == size_targets(target, difficulty)
        ids = {row_identity('graph_coloring', row) for row in rows}
        assert len(ids) == len(rows) and not ids & blocked
        blocked |= ids
        repeat = build_pool('graph_coloring', target, blocked, 7918100 + difficulty,
                            'small_test_repeat', difficulty, multiplier=1)
        assert not {row_identity('graph_coloring', row) for row in repeat} & blocked
        blocked |= {row_identity('graph_coloring', row) for row in repeat}
