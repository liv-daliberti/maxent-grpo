from __future__ import annotations

import json

from datasets import load_from_disk

from oat_drgrpo.point_maze_grid import adapt_point_maze_spec
from ops.evaluate_point_grid_viability_dev import _shortest_steps
from ops.evaluate_point_grid_viability_dev_v2 import (
    _goal_choice_count,
    _simple_path_choices,
)


def test_legal_mask_contains_solutions_and_more_unlabeled_nonsolutions():
    rows = load_from_disk("var/data/point_maze_modebench_v1/dev")[
        "multi_answer"
    ]
    expected_totals = (250, 8798, 5796, 4719)
    expected_goals = (32, 994, 366, 553)
    for row, total, goals in zip(rows, expected_totals, expected_goals):
        spec = adapt_point_maze_spec(json.loads(row["answer"]))
        minimum = _shortest_steps(spec)
        choices = _simple_path_choices(spec, minimum, minimum + 4)
        assert len(choices) == total
        assert _goal_choice_count(spec, choices) == goals
        assert goals < total
