"""Regression checks for overlapping vLLM n=8 RNG streams."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "ops/modebench_independent_seeds.py"
SPEC = importlib.util.spec_from_file_location("independent_seeds", PATH)
seeds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seeds)


def test_adjacent_draw_labels_have_disjoint_child_streams():
    labels = [6318000, 6318001, 6318002, 6318003]
    problems = [f"Color graph {index}" for index in range(128)]
    schedule = seeds.seed_schedule("graph_coloring", problems, labels)
    children = [base + sample for row in schedule for base in row for sample in range(8)]
    assert len(children) == 4096
    assert len(set(children)) == len(children)
    assert min(children) >= 0 and max(children) < 2**63


def test_seed_is_stable_across_order_and_sharding():
    problems = ["first", "second", "third"]
    labels = [6318000, 6318001]
    all_rows = seeds.seed_schedule("mathir", problems, labels)
    assert seeds.seed_schedule("mathir", problems[::-1], labels) == all_rows[::-1]
    assert seeds.seed_schedule("mathir", problems[1:], labels) == all_rows[1:]
    assert seeds.request_seed("mathir", "first", labels[0]) != seeds.request_seed("pantry", "first", labels[0])


def test_collision_fails_before_sampling(monkeypatch):
    monkeypatch.setattr(seeds, "request_seed", lambda *args: 8)
    with pytest.raises(ValueError, match="collision"):
        seeds.seed_schedule("countdown", ["first", "second"], [6318000])


@pytest.mark.parametrize("problems,labels,message", [
    (["same", "same"], [0], "duplicate prompt"),
    (["one"], [0, 0], "duplicate draw"),
    ([], [0], "nonempty"),
    (["one"], [], "nonempty"),
    (["one"], [True], "nonnegative integer"),
    (["one"], [-1], "nonnegative integer"),
])
def test_invalid_schedule_is_rejected(problems, labels, message):
    with pytest.raises(ValueError, match=message):
        seeds.seed_schedule("countdown", problems, labels)
