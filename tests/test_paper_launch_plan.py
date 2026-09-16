"""Contracts for partitioning unfinished paper cells into execution waves."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def load(name: str, path: Path):
    sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_every_method_has_a_missing_cell_wave():
    matrix = load("paper_matrix_for_launch_test", EXP / "paper_matrix.py")
    launch = load("paper_launch_plan_under_test", EXP / "paper_launch_plan.py")

    assert set(launch.MISSING_WAVE_BY_METHOD) == {
        method.key for method in matrix.METHODS
    }
    assert len(launch.WAVE_BY_KEY) == len(launch.WAVES)


def test_empty_registry_still_partitions_all_750_promised_cells():
    launch = load("paper_launch_plan_empty_test", EXP / "paper_launch_plan.py")
    plan = launch.build_plan({})
    by_key = {wave["key"]: wave for wave in plan["waves"]}

    assert plan["terminal_cells"] == 0
    assert plan["unfinished_cells"] == 750
    assert by_key["finish_released"]["cells"] == 0
    assert by_key["direct_comparators"]["cells"] == 225
    assert by_key["core_scale_extensions"]["cells"] == 150
    assert by_key["adaptive_replay"]["cells"] == 75
    assert by_key["fixed_semantic"]["cells"] == 150
    assert by_key["adaptive_semantic_without_replay"]["cells"] == 75
    assert by_key["adaptive_semantic_with_replay"]["cells"] == 75
