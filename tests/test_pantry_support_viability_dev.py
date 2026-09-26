from __future__ import annotations

import json
from itertools import combinations

from datasets import load_from_disk

from ops.evaluate_pantry_support_viability_dev import (
    _support_choices,
    _support_prompt,
)
from oat_drgrpo.pantry_plan import parse_pantry_plan_spec


def _first_dev_row():
    return load_from_disk("var/data/pantry_plan_modebench_v2/dev")[
        "multi_answer"
    ][0]


def test_action_space_contains_every_prompt_local_support():
    row = _first_dev_row()
    spec = json.loads(row["answer"])
    parsed = parse_pantry_plan_spec(spec)
    ingredient_ids = sorted(item.ingredient_id for item in parsed.ingredients)
    expected = [
        " ".join(support)
        for width in range(parsed.min_ingredients, parsed.max_ingredients + 1)
        for support in combinations(ingredient_ids, width)
    ]
    assert _support_choices(spec) == expected
    assert len(expected) > int(spec["certified_mode_count"])


def test_support_prompt_removes_quantity_answer_contract():
    row = _first_dev_row()
    prompt = _support_prompt(str(row["problem"]))
    assert "Return only ingredient_id=grams pairs" not in prompt
    assert "trusted environment will find quantities" in prompt
    assert "Output only those IDs" in prompt
    assert "<|im_start|>assistant\n" in prompt
