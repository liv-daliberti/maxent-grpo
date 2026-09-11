from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    # Plotting scripts run under PLOT_PYTHON, not the pinned paper310 training
    # environment, so skip rather than fail when the suite runs without it.
    pytest.importorskip("matplotlib")
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_metric_inventory_and_readable_page_widths():
    module = _load(
        "ops/exp_scaling/"
        "plot_e70_current_complete_square_pages_v7_20260730.py",
        "complete_square_v7",
    )
    metrics = [
        metric
        for _slug, _title, page_metrics, _footer in module.PAGES
        for metric, _metric_title, _integer in page_metrics
    ]
    expected = {
        "greedy",
        "mean8",
        "pass8",
        "distinct8",
        module.OPEN_SET_ENTROPY,
        module.OPEN_SET_COEFFICIENT,
        module.MASS_COEFFICIENT,
        module.BALANCE_COEFFICIENT,
        "canonical_replay_balance_loss",
        "canonical_replay_available_modes",
        "online_canonical_new_outcome_row_fraction",
        "online_canonical_mean_support_per_prompt",
        "online_canonical_tracked_outcomes",
    }
    assert set(metrics).issuperset(expected)
    assert [len(page[2]) for page in module.PAGES] == [4, 4, 3, 3]
    assert len(metrics) == len(set(metrics))


def test_support_retention_is_the_only_displayed_pantry_cohort():
    module = _load(
        "ops/exp_scaling/"
        "plot_e70_current_complete_square_pages_v7_20260730.py",
        "complete_square_v7_pantry_selection",
    )
    pantry_rows = [
        row for row in module._current_rows() if "pantry" in row[2]
    ]
    assert len(pantry_rows) == 1
    label, _note, domain, prefix, steps, seeds = pantry_rows[0]
    assert label == "PantryPlan"
    assert domain == "pantry_support_retention_repair"
    assert (prefix, steps, seeds) == (
        "pprepair_support_retention_final_v1",
        32,
        module.PANTRY_REPAIR_SEEDS,
    )


def test_balanced_v6_is_the_selected_pointmaze_repair():
    module = _load(
        "ops/exp_scaling/"
        "plot_e70_current_complete_square_pages_v7_20260730.py",
        "complete_square_v7_point_v6_selection",
    )
    selected = [
        row
        for row in module._current_rows()
        if row[2] == "point_maze_algorithm_repair"
    ]
    assert len(selected) == 1
    label, note, _domain, _prefix, _steps, seeds = selected[0]
    assert label == "PointMaze balanced v6"
    assert "52/256" in note
    assert seeds == module.POINT_V6_SEEDS
    assert len(module._load_point_v6_points()) >= 0


def test_builder_has_four_current_pages_and_legacy_reference():
    module = _load(
        "ops/exp_scaling/"
        "build_e70_historical_multipage_v7_20260730.py",
        "build_complete_v7",
    )
    assert len(module.CURRENT_PAGES) == 4
    assert len(module.ROLES) == 5
    assert module.ROLES[-1].endswith("historical_reference")
    assert module.OUTPUT.name.endswith("historical_20260730.pdf")
