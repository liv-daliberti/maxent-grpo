from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops/exp_scaling/build_paper_e113r4_dapo_progress.py"


def _module():
    spec = importlib.util.spec_from_file_location("paper_e113r4_dapo_progress", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_dapo_progress_is_exact_and_not_relabelled_as_pass8() -> None:
    payload = _module().build()
    assert payload["registered_science_cells"] == 50
    assert payload["terminal_science_cells"] >= 4
    assert (
        payload["terminal_science_cells"]
        + payload["running_science_cells"]
        + payload["pending_science_cells"]
        + payload["failed_science_cells"]
        + payload["other_nonterminal_science_cells"]
        == 50
    )
    by_cell = {
        (row["family"], row["domain"], row["seed"]): row
        for row in payload["records"]
    }
    expected = {
        ("qwen05b", "graph_coloring", 43): 0.2734375,
        ("qwen05b", "graph_coloring", 44): 0.265625,
        ("qwen05b", "graph_coloring", 45): 0.2734375,
        ("qwen05b", "graph_coloring", 46): 0.265625,
    }
    for cell, final_acc in expected.items():
        row = by_cell[cell]
        assert row["accepted_training_steps"] == 24
        assert row["final_upstream_validation_acc_at_1"] == final_acc
        assert row["standardized_pass_at_8_available"] is False
        assert row["standardized_breadth_at_8_available"] is False
        assert row["evidence_class"] == "terminal_upstream_acc_at_1_diagnostic"
    assert "mean" not in payload
    assert "interval" not in payload
