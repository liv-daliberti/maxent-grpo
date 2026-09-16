"""Check omissions and page layout that can silently misrepresent a trajectory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "primary_training_curve_plotter", ROOT / "ops/exp_scaling/plot_paper_training_curves.py")
assert SPEC is not None and SPEC.loader is not None
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def _record(seeds=(43, 44), *, missing_steps=()):
    points = []
    for step in range(0, 3073, 192):
        values = {str(seed): {"pass8": .2 + index * .1 + step / 10000,
                              "distinct8": .5 + index * .2 + step / 4000}
                  for index, seed in enumerate(seeds)}
        if step in missing_steps:
            values.pop(str(seeds[-1]))
        complete = len(values) == len(seeds)
        points.append({"step": step, "training_pass": step / 384, "complete": complete,
                       "per_seed": values, "mean": {
                           metric: sum(row[metric] for row in values.values()) / len(seeds)
                           if complete else None for metric in plotter.METRICS}})
    return {"cohort_n": len(seeds), "cohort_seeds": list(seeds),
            "cohort_policy": "paired terminal seeds within objective", "points": points}


@pytest.fixture
def payload():
    panels = [
        {"level": level, "scale": scale, "domain": domain,
         "methods": {method: _record(missing_steps=(384,) if method == "maxrl" else ())
                     for method in plotter.METHODS}}
        for level, scales in (("level1", plotter.SCALES), ("level2", ("qwen05b",)))
        for scale in scales for domain in plotter.DOMAINS
    ]
    # One unfinished history on each MaxRL arm must not be mistaken for
    # the complete Dr.GRPO pair in the adjacent cell.
    pantry = panels[-1]
    for method, seed in (("maxrl", 43), ("replay_maxrl", 46)):
        pantry["methods"].pop(method)
        record = _record((seed,), missing_steps=(384, 2880, 3072))
        record["cohort_policy"] = "partial/unpaired histories; no paired effect"
        pantry.setdefault("supplementary_methods", {})[method] = record
    return {"schema": plotter.SNAPSHOT_SCHEMA, "registered_steps": list(range(0, 3073, 192)),
            "panels": panels}


@pytest.mark.parametrize("metric", list(plotter.METRICS))
def test_a_missing_seed_breaks_both_mean_line_and_seed_range(metric):
    """A mean may not connect across a checkpoint missing one fixed-cohort seed."""
    figure, axis = plotter.plt.subplots()
    try:
        audit = plotter.draw_method(axis, "maxrl", _record(missing_steps=(384,)), False,
                                    metric, list(range(0, 3073, 192)))
        assert audit["segments"] == [[0, 192], list(range(576, 3073, 192))]
        assert len(axis.lines) == 2
        assert len(axis.collections) == 2
        for line in axis.lines:
            assert not (min(line.get_xdata()) < 1 < max(line.get_xdata()))
        for band in axis.collections:
            for path in band.get_paths():
                xs = path.vertices[:, 0]
                assert not (min(xs) < 1 < max(xs))
    finally:
        plotter.plt.close(figure)


def test_partial_histories_have_no_band_or_fixed_cohort_mean(payload):
    figure, audit = plotter.build_figure(payload, level="level2")
    try:
        for axis in (figure.axes[4], figure.axes[9]):
            max_lines = [line for line in axis.lines if line.get_gid().startswith(("maxrl:", "replay_maxrl:"))]
            assert max_lines
            assert all(":partial:" in line.get_gid() for line in max_lines)
            assert not any(band.get_gid().startswith(("maxrl:", "replay_maxrl:"))
                           for band in axis.collections)
            assert any("Max histories\nn=1/1†" in text.get_text() for text in axis.texts)
            assert any("Independent partial histories" in text.get_text() for text in figure.texts)
        for panel in audit["panels"]:
            if panel["domain"] == "pantry_plan":
                assert panel["methods"]["maxrl"]["partial_unpaired"] is True
    finally:
        plotter.plt.close(figure)


def test_wrong_cohort_mean_is_rejected():
    record = _record()
    record["points"][0]["mean"]["pass8"] = .99
    with pytest.raises(ValueError, match="fixed cohort"):
        plotter.primary_points(record, "pass8")


@pytest.mark.parametrize("level,metric", [("level1", "pass8"), ("level1", "distinct8"), ("level2", None)])
def test_composed_figure_preserves_all_cells_and_separates_labels(payload, level, metric):
    """Measure the complete grid: row labels and count notes must remain legible."""
    figure, audit = plotter.build_figure(payload, level=level, metric=metric)
    try:
        assert len(figure.axes) == (15 if level == "level1" else 10)
        assert all(set(panel["methods"]) == set(plotter.METHODS) for panel in audit["panels"])
        if level == "level1":
            assert audit["models"] == list(plotter.SCALES)
            for column in range(5):
                assert len({figure.axes[row * 5 + column].get_ylim() for row in range(3)}) == 1
        assert all(axis.get_xlim() == (0, 8) for axis in figure.axes)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        # Count notes, titles, row labels, legend and global labels must remain
        # separate on the actual authoring canvas, including partial histories.
        labels = [*figure.texts, *figure.legends]
        labels.extend(axis.title for axis in figure.axes if axis.get_title())
        labels.extend(axis.yaxis.label for axis in figure.axes if axis.get_ylabel())
        labels.extend(text for axis in figure.axes for text in axis.texts if text.get_gid() == "cohort-note")
        boxes = [label.get_window_extent(renderer) for label in labels]
        canvas = figure.get_window_extent(renderer)
        for index, box in enumerate(boxes):
            assert box.x0 >= canvas.x0 and box.x1 <= canvas.x1
            assert box.y0 >= canvas.y0 and box.y1 <= canvas.y1
            for other in boxes[index + 1:]:
                assert not box.overlaps(other), "figure labels overlap at the actual saved canvas size"
    finally:
        plotter.plt.close(figure)


def test_partner_gap_is_shared_without_mutating_raw_snapshot(payload):
    panel = payload["panels"][0]
    assert panel["methods"]["replay_maxrl"]["points"][2]["complete"] is True
    selected = plotter.selected_methods(panel)
    assert selected["replay_maxrl"][0]["points"][2]["complete"] is False
    assert 384 not in selected["replay_maxrl"][0]["paired_checkpoint_steps"]
    assert panel["methods"]["replay_maxrl"]["points"][2]["complete"] is True
    for method in ("maxrl", "replay_maxrl"):
        assert [point["step"] for point in plotter.primary_points(selected[method][0], "pass8")] == [
            step for step in payload["registered_steps"] if step != 384]
