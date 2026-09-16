"""E118 keeps valid MaxRL pairs when a historical Dr.GRPO pair is excluded."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e118_integrity_plotter",
    ROOT / "ops/exp_scaling/plot_paper_e118_all_scale_progress.py",
)
assert SPEC is not None and SPEC.loader is not None
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def matched_fixture(excluded_seed=59):
    base = {"models": {}}
    trajectory = {"cells": {}}
    record = {"cells": {}}
    for scale, model, seeds in plotter.MODELS[:2]:
        domains = base["models"].setdefault(model, {"domains": {}})["domains"]
        record["cells"][scale] = {}
        for index, domain in enumerate(plotter.DOMAINS):
            sources = domains.setdefault(domain, {"methods": {}})["methods"]
            for arm, offset in (("control", 0.1), ("replay", 0.4)):
                sources[arm] = {"per_seed": {
                    str(seed): {
                        metric: offset + index * 0.01 + position * 0.05
                        for metric in ("pass8", "distinct8")
                    }
                    for position, seed in enumerate(seeds)
                }}
            if scale == "falcon1b" and domain == "countdown":
                del sources["replay"]["per_seed"][str(excluded_seed)]
            trajectory["cells"][f"{scale}/{domain}"] = {"methods": {
                "drgrpo": {"summaries": {"0": {
                    metric: {"per_seed": {
                        str(seed): 0.2 + index * 0.01 + position * 0.05
                        for position, seed in enumerate(seeds)
                    }}
                    for metric in ("pass8", "distinct8")
                }}}
            }}
            cell = {"matched_seeds": list(seeds), "methods": {
                method: {metric: [value] * 5 for metric in ("pass8", "distinct8")}
                for method, value in (("maxrl", 0.5), ("replay_maxrl", 0.8))
            }}
            plotter.attach_reference_methods(
                cell, base=base, trajectory=trajectory,
                scale=scale, model=model, domain=domain,
            )
            record["cells"][scale][domain] = cell
    return base, record


@pytest.mark.parametrize("excluded_seed", [56, 59])
def test_missing_dr_pair_does_not_remove_valid_maxrl_seed(excluded_seed):
    _, record = matched_fixture(excluded_seed)
    cell = record["cells"]["falcon1b"]["countdown"]
    retained = [seed for seed in (55, 56, 57, 58, 59) if seed != excluded_seed]
    assert cell["matched_seeds"] == [55, 56, 57, 58, 59]
    for method in ("maxrl", "replay_maxrl", "before_training"):
        assert cell["method_seeds"][method] == [55, 56, 57, 58, 59]
        assert len(cell["methods"][method]["pass8"]) == 5
    for method in ("drgrpo", "replay_drgrpo", "before_training_drgrpo"):
        assert cell["method_seeds"][method] == retained
        assert len(cell["methods"][method]["pass8"]) == 4


@pytest.mark.parametrize("excluded_seed", [56, 59])
def test_macro_averages_identical_seed_subset_in_every_domain(excluded_seed):
    base, record = matched_fixture(excluded_seed)
    averages = plotter.absolute_cross_domain_averages(record)
    retained = [seed for seed in (55, 56, 57, 58, 59) if seed != excluded_seed]
    source = base["models"]["Falcon3-1B"]["domains"]
    for method, arm in (("drgrpo", "control"), ("replay_drgrpo", "replay")):
        actual = averages["falcon1b"]["pass8"][method]
        expected = {
            str(seed): statistics.fmean(
                source[domain]["methods"][arm]["per_seed"][str(seed)]["pass8"]
                for domain in plotter.DOMAINS
            )
            for seed in retained
        }
        assert actual["per_seed"] == expected
        assert actual["mean"] == statistics.fmean(expected.values())
        assert actual["seeds"] == retained
        assert actual["n"] == 4
        assert "student_t_95" not in actual
    assert averages["falcon1b"]["pass8"]["before_training_drgrpo"]["seeds"] == retained
    for scale in ("qwen05b", "falcon1b"):
        assert averages[scale]["pass8"]["maxrl"]["n"] == 5
        assert averages[scale]["pass8"]["replay_maxrl"]["n"] == 5
    assert averages["qwen05b"]["pass8"]["drgrpo"]["n"] == 5


def test_main_and_appendix_label_partial_track_and_match_initial_reference():
    _, record = matched_fixture()
    averages = plotter.absolute_cross_domain_averages(record)
    figure, axes = plotter.plt.subplots(1, 2)
    try:
        plotter.draw_cross_domain_panel(
            axes[0], absolute_averages=averages, metric="pass8", title="main",
            xlim=(0, 1),
        )
        assert [text.get_text() for text in axes[0].texts].count("n=4") == 1
        initial_points = [
            float(line.get_xdata()[0]) for line in axes[0].lines
            if line.get_marker() == "D"
        ]
        for method in ("before_training", "before_training_drgrpo"):
            assert averages["falcon1b"]["pass8"][method]["mean"] in initial_points
        assert (
            averages["falcon1b"]["pass8"]["before_training"]["mean"]
            != averages["falcon1b"]["pass8"]["before_training_drgrpo"]["mean"]
        )
        plotter.draw_pair_panel(
            axes[1], record=record, absolute_averages=averages,
            scale="falcon1b", metric="pass8", title="appendix",
            rows=tuple(zip(plotter.DOMAINS, plotter.LABELS)), xlim=(0, 1),
            show_untrained=True,
        )
        assert [text.get_text() for text in axes[1].texts].count("n=4") == 1
        figure.canvas.draw()
    finally:
        plotter.plt.close(figure)
