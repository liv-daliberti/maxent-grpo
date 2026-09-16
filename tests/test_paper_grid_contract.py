"""Paper graphs always draw both vertical and horizontal grid guides."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "ops/paper_style.py"
GRAPH_GENERATORS = tuple(
    path for path in (ROOT / "ops").rglob("*.py")
    if path.name.startswith("plot_") or path.name.startswith("summarize_")
)


def load_style():
    spec = importlib.util.spec_from_file_location("paper_style_grid_test", STYLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_style_enforces_two_axis_grids():
    style = load_style()
    source = STYLE.read_text(encoding="utf-8")

    assert style.PAPER_GRID_AXIS == "both"
    assert "axis=PAPER_GRID_AXIS" in source


def test_graph_generators_do_not_request_one_axis_grids():
    for path in GRAPH_GENERATORS:
        source = path.read_text(encoding="utf-8")
        assert '.grid(axis="y"' not in source, path
        assert '.grid(axis="x"' not in source, path
        assert "grid='y'" not in source, path
        assert "grid='x'" not in source, path
