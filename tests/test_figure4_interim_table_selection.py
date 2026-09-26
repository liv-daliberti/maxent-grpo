from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "ops/exp_scaling/build_figure4_interim_table.py"


def load():
    spec = importlib.util.spec_from_file_location("figure4_table", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(passes: dict[str, list[int]]) -> dict:
    return {
        "paired_summary_by_pass": {
            key: {"paired_seeds": seeds} for key, seeds in passes.items()
        }
    }


def test_pass_zero_only_panels_are_not_reportable():
    builder = load()
    # At pass 0 both arms are the same untrained model, so every metric matches
    # exactly. Reporting it would print a row of exact ties that reads as "the
    # arms are indistinguishable" when neither has taken a step.
    assert builder._latest_summary(_record({"0.0": [55, 56, 57]})) is None


def test_panels_enter_the_table_once_training_is_paired():
    builder = load()
    latest = builder._latest_summary(_record({"0.0": [55, 56, 57], "0.5": [55]}))
    assert latest is not None
    assert latest[0] == "0.5"
    assert latest[1]["paired_seeds"] == [55]


def test_the_deepest_trained_checkpoint_wins():
    builder = load()
    latest = builder._latest_summary(
        _record({"0.0": [43], "2.0": [43, 44], "8.0": [43, 44, 45]})
    )
    assert latest is not None
    assert latest[0] == "8.0"
    assert latest[1]["paired_seeds"] == [43, 44, 45]


def test_a_panel_with_no_paired_checkpoint_stays_absent():
    builder = load()
    assert builder._latest_summary(_record({})) is None
