from __future__ import annotations

import json
import re

from datasets import load_from_disk

from oat_drgrpo.point_maze_grid import adapt_point_maze_spec
from ops.evaluate_point_grid_viability_dev import _prompt, _regex, _shortest_steps


def test_prompt_and_regex_expose_only_grid_syntax_and_length():
    row = load_from_disk("var/data/point_maze_modebench_v1/dev")[
        "multi_answer"
    ][0]
    spec = adapt_point_maze_spec(json.loads(row["answer"]))
    minimum = _shortest_steps(spec)
    maximum = minimum + 4
    prompt = _prompt(spec, minimum, maximum)
    pattern = re.compile(_regex(minimum, maximum))
    assert "Each token moves exactly one grid cell" in prompt
    assert "route_a" not in prompt and "route_b" not in prompt
    assert pattern.fullmatch("N E E E E S")
    assert not pattern.fullmatch("N NE E")
