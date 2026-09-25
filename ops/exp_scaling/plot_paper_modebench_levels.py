#!/usr/bin/env python3
"""Render replay terminal results; retain historical admission only as provenance."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style
from exp_scaling.build_paper_modebench_level_comparison import (
    DOMAINS, build_interim_comparison, build_terminal_progress,
)
from exp_scaling.build_paper_modebench_three_level_comparison import (
    LEVELS, PAIRS, build_baseline_points, build_pmd_comparison, build_replay_gains,
    build_terminal_comparison,
)

SNAPSHOT = ROOT / "paper/results/modebench_level_comparison_snapshot.json"
LEVEL3_SNAPSHOT = ROOT / "paper/results/modebench_level3_comparison_snapshot.json"
BASELINE_SNAPSHOT = ROOT / "paper/results/modebench_level_baseline_snapshot.json"
MODE_DIVERSITY = ROOT / "paper/results/mode_diversity_training.json"
OUT = ROOT / "paper/figures/modebench_level_admission"
LEVEL1_COLOR = "#475569"
LEVEL2_COLOR = "#0F766E"
LEVEL3_COLOR = "#7C2D12"
LEVEL_COLORS = {"level1": LEVEL1_COLOR, "level2": LEVEL2_COLOR, "level3": LEVEL3_COLOR}
# Levels read as a mark shape rather than a third colour family: the colour slot
# is already spent naming the four methods, and a fill-only cue asked the reader
# to judge how much of a 6pt dot was painted.
LEVEL_MARKERS = {"level1": "o", "level2": "s", "level3": "^"}
LEVEL_NAMES = {"level1": "Level 1", "level2": "Level 2", "level3": "Level 3"}
PARTIAL_ARM_POLICY = "Partial-domain markers use each arm's available terminal seeds; differences between these arm means are not paired effects."
METHOD_COLOURS = {
    "drgrpo": style.CONTROL,
    "replay_drgrpo": style.ADAPTIVE,
    "maxrl": style.COMPARATOR,
    "replay_maxrl": style.ABLATION,
}
# "(ours)" marks the two arms this paper introduces.
METHOD_LEGEND = {
    "drgrpo": "Dr.GRPO",
    "replay_drgrpo": "Re:Dr (ours)",
    "maxrl": "MaxRL",
    "replay_maxrl": "Re:Max (ours)",
}
DOMAIN_NAMES = dict(zip(
    DOMAINS, ("Graph", "Countdown", "Python", "MathIR", "PantryPlan")))
METHODS = {
    "drgrpo": "Dr.GRPO",
    "replay_drgrpo": "Re:Dr",
    "maxrl": "MaxRL",
    "replay_maxrl": "Re:Max",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pmd_comparison(level3: dict, path: Path = MODE_DIVERSITY) -> dict:
    """Terminal PCMD per method and level, on domains matched across all three."""
    payload = json.loads(path.read_text())
    comparison = build_pmd_comparison(
        payload["arms"], level3["pmd_cells"], payload["definition"]["min_defined_prompts"])
    comparison["source"] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
    return comparison


def build_record(snapshot: dict, snapshot_path: Path, level3: dict, level3_path: Path,
                 baseline: dict, baseline_path: Path) -> dict:
    if snapshot.get("schema") != "modebench-level-comparison-frozen-snapshot-v1":
        raise RuntimeError("wrong frozen Level1/Level2 comparison snapshot")
    if level3.get("schema") != "modebench-level3-comparison-frozen-snapshot-v1":
        raise RuntimeError("wrong frozen Level3 comparison snapshot")
    if baseline.get("schema") != "modebench-level-baseline-frozen-snapshot-v1":
        raise RuntimeError("wrong frozen untrained-checkpoint snapshot")
    reference = snapshot["reference_figure"]
    record = {
        "schema": "modebench-level-admission-and-terminal-v5",
        "target_step": 3072,
        "model": "Qwen2.5-0.5B-Instruct",
        "levels": list(LEVELS),
        "sources": {
            **reference["sources"],
            **snapshot["terminal_sources"],
            str(snapshot_path.resolve()): digest(snapshot_path),
            str(level3_path.resolve()): digest(level3_path),
            str(baseline_path.resolve()): digest(baseline_path),
        },
        "snapshot_collected_at_utc": snapshot["collected_at_utc"],
        "level3_collected_at_utc": level3["collected_at_utc"],
        "admission_rows": copy.deepcopy(reference["admission_rows"]),
        "admission_rows_role": "Historical development admission retained for provenance; construction is described in Methods and is not plotted here.",
        # Preserve the complete Graph example as separate terminal evidence.
        "partial_treatment": copy.deepcopy(reference["partial_treatment"]),
        "terminal_graph_role": "Historical separate example retained for provenance; the main figure uses all complete terminal domain blocks.",
        "terminal_comparison": build_terminal_comparison(
            snapshot["terminal_evaluations"] + level3["terminal_evaluations"],
            snapshot["terminal_admission"] + level3["terminal_admission"]),
        "level3_admission_rule": level3["admission_rule"],
        # Cells the third level has not yet produced an admissible endpoint for.
        # They are named here rather than averaged over, and they are why a
        # domain can be partial at Level 3 while complete at the other two.
        "level3_unadmitted_cells": copy.deepcopy(level3["unadmitted_cells"]),
        "interim_comparison_role": "Historical checkpoint-matched diagnostic retained in the sidecar; not plotted.",
        "interim_comparison": build_interim_comparison(snapshot["evaluations"], snapshot["availability"]),
        "terminal_progress_by_domain": build_terminal_progress(snapshot["level2_terminal_evaluations"]),
        "pmd_comparison": pmd_comparison(level3),
    }
    record["baseline_points"] = build_baseline_points(
        baseline["baseline_admission"] + level3["baseline_admission"],
        baseline["baseline_pmd_cells"] + level3["baseline_pmd_cells"],
        record["terminal_comparison"]["complete_domains"],
        record["pmd_comparison"]["matched_domains"])
    record["baseline_admission_rule"] = baseline["admission_rule"]
    record["baseline_points_role"] = (
        "Where each level's training starts, retained beside the gains it is not plotted "
        "with; the figure plots differences, and an absolute position is not one.")
    record["replay_gains"] = build_replay_gains(
        record["terminal_comparison"], record["pmd_comparison"])
    return record


def _axis(axis):
    axis.set_facecolor(style.PANEL)
    axis.grid(color=style.GRID, lw=.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=6.6)
    axis.tick_params(axis="x", labelsize=6.6)


def _level_marks(axis, values, y, *, size=6):
    """One row, one mark per level: open, half, filled as difficulty rises."""
    xs = [values[level] for level in LEVELS if values.get(level) is not None]
    axis.plot(xs, [y] * len(xs), color=style.GRID, lw=2, zorder=1)
    for level in LEVELS:
        if values.get(level) is None:
            continue
        axis.plot(values[level], y, marker=LEVEL_MARKERS[level], markersize=size,
                  color=LEVEL_COLORS[level], markeredgecolor=LEVEL_COLORS[level],
                  markeredgewidth=.6, linestyle="none", zorder=3)


def render(record: dict, output: Path = OUT) -> None:
    """Two panels, one question: what does replay add, level by level.

    Earlier drafts plotted the four arms as positions and asked the reader to
    subtract one from another by eye, twice per level, on two axes at once. The
    quantity the section claims is the difference, so the difference is what is
    drawn: each bar is a replay arm minus its own fresh objective at the same
    level, on the prompts, seeds and domains they share. Breadth is on top
    because it is the axis the paper is about; correctness is under it because
    the claim is that breadth is not bought with it.
    """
    figure = plt.figure(figsize=(2.85, 2.16))
    axes = [figure.add_axes([.205, .575, .775, .350]),
            figure.add_axes([.205, .140, .775, .350])]
    gains = record["replay_gains"]["levels"]
    positions = {level: index for index, level in enumerate(LEVELS)}
    width = .34

    for axis, (metric, label, ceiling) in zip(axes, (("pmd", "PCMD", .38),
                                                     ("pass8", "pass@8", .44))):
        axis.set_facecolor(style.PANEL)
        axis.grid(axis="y", color=style.GRID, lw=.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            axis.spines[side].set_color(style.MUTED)
            axis.spines[side].set_linewidth(.6)
        axis.tick_params(length=2, width=.6, labelsize=6.8)
        for offset, (control, replay) in zip((-width / 2, width / 2), PAIRS):
            axis.bar([positions[level] + offset for level in LEVELS],
                     [gains[level][replay][metric] for level in LEVELS],
                     width=width, color=METHOD_COLOURS[replay],
                     edgecolor=METHOD_COLOURS[replay], linewidth=0,
                     label=METHOD_LEGEND[replay], zorder=3)
        axis.set_xticks(list(positions.values()),
                        [LEVEL_NAMES[level].replace("Level ", "L") for level in LEVELS],
                        fontsize=7.0)
        axis.set_xlim(-.62, len(LEVELS) - .38)
        axis.set_ylim(0, ceiling)
        axis.set_ylabel(label + " gain", fontsize=7.6)

    axes[0].legend(ncol=1, loc="upper right", frameon=False, fontsize=6.2,
                   handlelength=.8, handletextpad=.3, labelspacing=.16,
                   borderpad=.1, borderaxespad=.25)
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def render_domains(record: dict, output: Path) -> None:
    comparison = record["terminal_comparison"]
    figure, axes = plt.subplots(2, 5, figsize=(style.WIDTH, 3.65))
    figure.subplots_adjust(left=.14, right=.985, bottom=.17, top=.80,
                           wspace=.20, hspace=.42)
    domain_names = ("Graph coloring", "Countdown", "Python factors", "MathIR", "PantryPlan")
    for column, (domain, name) in enumerate(zip(DOMAINS, domain_names)):
        result = comparison["domain_results"][domain]
        for row, (metric, label) in enumerate((("pass8", "pass@8"), ("distinct8", "distinct@8"))):
            axis = axes[row, column]
            _axis(axis)
            for y, method in enumerate(METHODS):
                series = {level: result["series"][level][method] for level in LEVELS}
                if result["complete_block"]:
                    _level_marks(axis, {level: series[level]["means"][metric]
                                        for level in LEVELS}, y, size=5.4)
                else:
                    # A partial block is not a paired row, so its levels are
                    # offset rather than joined, and each carries its own n.
                    for offset, level in zip((-.16, 0, .16), LEVELS):
                        one = series[level]
                        if not one["n"]:
                            continue
                        axis.plot(one["means"][metric], y + offset,
                                  marker=LEVEL_MARKERS[level], markersize=4.4,
                                  color=LEVEL_COLORS[level],
                                  markeredgecolor=LEVEL_COLORS[level],
                                  markeredgewidth=.6, linestyle="none", zorder=3)
                    axis.text(.99, y, "/".join(str(series[level]["n"]) for level in LEVELS),
                              transform=axis.get_yaxis_transform(), ha="right", va="center",
                              fontsize=5.4, color=style.MUTED,
                              bbox=dict(facecolor="white", edgecolor="none", pad=.3))
            axis.set_yticks(range(4), list(METHODS.values()) if column == 0 else [""] * 4,
                            fontsize=6.3)
            axis.set_ylim(3.5, -.5)
            axis.set_xlabel(label, fontsize=6.6)
            if metric == "pass8":
                # The partial panel carries its per-level n at the right edge,
                # so it needs room the complete panels do not.
                axis.set_xlim(-.035, 1.30 if not result["complete_block"] else 1.035)
                axis.set_xticks((0, .5, 1), ("0", ".5", "1"))
            else:
                axis.set_xlim(-.07, 3.25)
                axis.set_xticks((0, 1, 2, 3))
            if row == 0:
                subtitle = ("5 matched seeds" if result["complete_block"]
                            else "partial; n = L1/L2/L3")
                axis.set_title(name + "\n" + subtitle, fontsize=6.9, pad=6)
    figure.text(.14, .975, "Terminal results by domain \u00b7 Qwen2.5-0.5B \u00b7 training pass 8",
                ha="left", va="top", fontsize=8.0)
    figure.legend(handles=[
        Line2D([0], [0], marker=LEVEL_MARKERS[level], linestyle="none",
               color=LEVEL_COLORS[level], markeredgecolor=LEVEL_COLORS[level],
               label=LEVEL_NAMES[level]) for level in LEVELS],
        ncol=3, loc="upper center", bbox_to_anchor=(.55, .93), frameon=False, fontsize=6.8)
    figure.text(.55, .022, "Partial markers use each arm's available seeds; differences are not paired effects. Four K=8 draws/seed.",
                ha="center", va="bottom", fontsize=6.2, color=style.MUTED)
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def build_domain_record(record: dict) -> dict:
    return {
        "schema": "modebench-level-terminal-by-domain-v2",
        "sources": record["sources"],
        "terminal_comparison": record["terminal_comparison"],
        "partial_arm_policy": PARTIAL_ARM_POLICY,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--level3-snapshot", type=Path, default=LEVEL3_SNAPSHOT)
    parser.add_argument("--baseline-snapshot", type=Path, default=BASELINE_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--domain-output", type=Path, default=OUT.with_name("modebench_level_terminal_by_domain"))
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    level3 = json.loads(args.level3_snapshot.read_text())
    baseline = json.loads(args.baseline_snapshot.read_text())
    record = build_record(snapshot, args.snapshot, level3, args.level3_snapshot,
                          baseline, args.baseline_snapshot)
    render(record, args.output)
    render_domains(record, args.domain_output)
    args.domain_output.with_suffix(".json").write_text(
        json.dumps(build_domain_record(record), indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
