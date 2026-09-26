from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLOTTER = ROOT / "ops/exp_scaling/plot_e112r1_private_interim.py"
FREEZE = ROOT / "var/artifacts/e112r1_private_interim_unblinding_freeze.json"
PAYLOAD = ROOT / "var/artifacts/private_interim/e112r1_subset_endpoint_effects.json"


def _load():
    spec = importlib.util.spec_from_file_location("e112_private_plot_test", PLOTTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_e112_payload_is_exactly_the_preoutcome_freeze() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    frozen = {
        (cell["scale"], cell["domain"], int(cell["seed"]), int(cell["job_id"]))
        for cell in freeze["cells"]
    }
    plotted = {
        (cell["scale"], cell["domain"], int(row["seed"]), int(row["treatment_job_id"]))
        for cell in payload["cells"]
        for row in cell["per_seed"]
    }
    assert plotted == frozen
    assert len(plotted) == freeze["terminal_cells"] == 14
    assert payload["label"] == "PRIVATE EXPLORATORY INTERIM — NOT FOR PAPER OR SELECTION"
    assert payload["confirmatory"] is False
    assert payload["campaign_mutation_allowed"] is False
    assert payload["paper_efficacy_output_allowed"] is False
    assert payload["pointmaze"] == "excluded"
    assert all(cell["summaries"] is None for cell in payload["cells"])


def test_private_e112_plot_refuses_every_output_outside_private_tree(tmp_path: Path) -> None:
    plotter = _load()
    plotter._assert_private_output(plotter.DEFAULT_OUTPUT)
    with pytest.raises(RuntimeError, match="must stay under"):
        plotter._assert_private_output(ROOT / "paper/figures/e112_interim")
    with pytest.raises(RuntimeError, match="must stay under"):
        plotter._assert_private_output(tmp_path / "e112_interim")
