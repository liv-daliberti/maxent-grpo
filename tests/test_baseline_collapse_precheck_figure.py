"""Contract tests for the problem-precheck builder and its figure.

The figure makes a claim about *every* registered cell, so the things worth
pinning are the ones that would quietly weaken it: a scale silently dropping to
four seeds, a retained-breadth badge printed against a denominator at the
measurement floor, and --- because this grid packs sixty bars and thirty badges
onto one 7.35in canvas --- a badge that collides with the bar it annotates.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "ops/exp_scaling/build_paper_baseline_collapse_precheck.py"
PLOTTER = ROOT / "ops/exp_scaling/plot_paper_baseline_collapse_precheck.py"
FROZEN = ROOT / "paper/results/baseline_collapse_precheck.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def payload() -> dict:
    if not FROZEN.is_file():
        pytest.skip(f"{FROZEN} has not been built")
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def test_every_registered_cell_has_both_endpoints(payload):
    """All 150 domain/seed cells resolve at pass 0 and pass 8, or the figure lies."""

    builder = _load(BUILDER, "precheck_builder")
    for arm in ("drgrpo", "grpo"):
        for scale in builder.SCALES:
            block = payload["arms"][arm]["scales"][scale]
            assert sorted(block["domains"]) == sorted(builder.DOMAINS), (
                f"{arm}/{scale} is missing a domain"
            )
            for domain, record in block["domains"].items():
                assert len(record["per_seed"]) == 5, (
                    f"{arm}/{scale}/{domain} has {len(record['per_seed'])} seeds"
                )
                for seed, endpoints in record["per_seed"].items():
                    for endpoint in ("pass0", "pass8"):
                        metrics = endpoints[endpoint]
                        assert metrics["distinct8"] >= metrics["pass8"], (
                            f"{arm}/{scale}/{domain}/{seed}: distinct@8 below pass@8"
                        )
                        assert metrics["extra_modes"] == pytest.approx(
                            metrics["distinct8"] - metrics["pass8"]
                        )
            assert block["macro"]["complete"] is True


def test_retained_breadth_is_withheld_below_the_floor(payload):
    """A retention percentage is reported only where there was breadth to lose."""

    plotter = _load(PLOTTER, "precheck_plotter")
    floor = payload["breadth_floor"]
    reported = set()
    for arm in ("drgrpo", "grpo"):
        for scale in plotter.SCALES:
            for domain, record in (
                payload["arms"][arm]["scales"][scale]["domains"].items()
            ):
                share = plotter.retained(payload, record)
                origin = record["endpoints"]["pass0"]["extra_modes"]["mean"]
                assert (share is None) == (origin <= floor)
                if share is not None:
                    reported.add(domain)
    # Only the two domains the manuscript describes as starting with
    # substantial breadth may carry a badge; if a third ever qualifies, the
    # accompanying prose has to be rewritten rather than the test relaxed.
    assert reported == {"graph_coloring", "pantry_plan"}


def test_pass0_agreement_between_arms_is_recorded_not_assumed(payload):
    """Where the two arms do not share pass 0, the exception is in the record."""

    agreement = payload["pass0_arm_agreement"]
    mismatched = {
        (scale, domain)
        for scale, domains in agreement.items()
        for domain, entry in domains.items()
        if entry.get("comparable") and not entry["identical"]
    }
    assert mismatched == {
        ("qwen3b", "mathir"), ("qwen3b", "pantry_plan"),
    }, "pass-0 arm agreement changed; the figure caption states which cells share it"


def _rendered(plotter, payload, tmp_path):
    """Render through the real entry point and hand back the composed figure.

    The first version of this test built a lone panel at an invented figure
    size and measured that. It passed while the actual grid was unreadable ---
    three rotated row labels overprinting each other, a subtitle struck through
    the panel titles, the legend sitting on the arm names. None of that exists
    in a single panel, so only the composed figure is worth asserting on.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    real_save, real_close = plotter.style.save, plt.close
    plotter.style.save = lambda figure, path, **kw: captured.update(
        figure=figure, renderer=figure.canvas.get_renderer()
    ) or real_save(figure, path, **kw)
    plt.close = lambda *a, **k: None
    try:
        plotter.render(payload, tmp_path / "precheck")
    finally:
        plotter.style.save, plt.close = real_save, real_close
    figure = captured["figure"]
    figure.canvas.draw()
    return figure, figure.canvas.get_renderer()


def _boxes(figure, renderer):
    """Every text and mark that has to stay out of everything else's way."""

    axes = figure.get_axes()
    return {
        "suptitle": [figure._suptitle.get_window_extent(renderer)],
        "figure_text": [t.get_window_extent(renderer) for t in figure.texts],
        "legend": [figure.legends[0].get_window_extent(renderer)],
        "panel_titles": [
            a.title.get_window_extent(renderer) for a in axes if a.get_title()
        ],
        "row_labels": [
            a.yaxis.label.get_window_extent(renderer) for a in axes
            if a.get_ylabel()
        ],
        "arm_labels": [
            l.get_window_extent(renderer) for a in axes
            for l in a.get_xticklabels(minor=True) if l.get_text()
        ],
        "pass_labels": [
            l.get_window_extent(renderer) for a in axes
            for l in a.get_xticklabels(minor=False) if l.get_text()
        ],
        "badges": [
            t.get_window_extent(renderer) for a in axes for t in a.texts
        ],
    }


def test_composed_figure_has_no_overlapping_labels(payload, tmp_path):
    """Nothing in the assembled grid may print on top of anything else."""

    plotter = _load(PLOTTER, "precheck_plotter_layout")
    figure, renderer = _rendered(plotter, payload, tmp_path)
    boxes = _boxes(figure, renderer)

    # Row labels are rotated and stacked in one narrow column; they collided
    # with each other before the metric name was hoisted out of them.
    for group in ("row_labels", "arm_labels", "badges", "panel_titles"):
        items = boxes[group]
        for i, one in enumerate(items):
            for other in items[i + 1:]:
                assert not one.overlaps(other), f"two {group} overlap"

    for left, right in (
        ("suptitle", "panel_titles"), ("suptitle", "badges"),
        ("figure_text", "panel_titles"), ("figure_text", "row_labels"),
        ("figure_text", "badges"), ("legend", "arm_labels"),
        ("legend", "pass_labels"), ("arm_labels", "pass_labels"),
        ("row_labels", "pass_labels"), ("badges", "panel_titles"),
    ):
        for one in boxes[left]:
            for other in boxes[right]:
                assert not one.overlaps(other), f"{left} overlaps {right}"


def test_composed_figure_marks_stay_clear_of_the_badges(payload, tmp_path):
    """Each retained-breadth badge clears every bar and whisker in its panel."""

    plotter = _load(PLOTTER, "precheck_plotter_marks")
    figure, renderer = _rendered(plotter, payload, tmp_path)
    for axis in figure.get_axes():
        badges = [t.get_window_extent(renderer) for t in axis.texts]
        if not badges:
            continue
        for mark in list(axis.patches) + list(axis.lines):
            extent = mark.get_window_extent(renderer)
            if extent.width <= 0 or extent.height <= 0:
                continue
            for badge in badges:
                assert not badge.overlaps(extent), (
                    "a retained-breadth badge overlaps a bar or whisker"
                )


def test_saved_figure_is_not_wider_than_its_canvas(payload, tmp_path):
    r"""Tight bounds must not exceed the authoring width.

    ``style.save`` writes with ``bbox_inches="tight"``. Any text that runs off
    the canvas silently widens the saved page, and \includegraphics then scales
    the whole grid down to \linewidth --- so an overlong caption line shrinks
    every number in the figure. This caught a subtitle that ran 70pt off both
    ends.
    """

    plotter = _load(PLOTTER, "precheck_plotter_bbox")
    figure, renderer = _rendered(plotter, payload, tmp_path)
    width = figure.get_window_extent(renderer).x1
    for text in [figure._suptitle, *figure.texts]:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.x1 <= width + 1, (
            f"{text.get_text()[:40]!r} runs off the canvas"
        )
