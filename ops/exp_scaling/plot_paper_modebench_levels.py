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
    DOMAINS, build_interim_comparison, build_terminal_comparison, build_terminal_progress,
)

SNAPSHOT = ROOT / "paper/results/modebench_level_comparison_snapshot.json"
OUT = ROOT / "paper/figures/modebench_level_admission"
LEVEL1_COLOR = "#475569"
LEVEL2_COLOR = "#0F766E"
PARTIAL_ARM_POLICY = "Partial-domain markers use each arm's available terminal seeds; differences between these arm means are not paired effects."
METHODS = {
    "drgrpo": "Dr.GRPO",
    "replay_drgrpo": "Re:Dr.GRPO",
    "maxrl": "MaxRL",
    "replay_maxrl": "Re:MaxRL",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_record(snapshot: dict, snapshot_path: Path) -> dict:
    if snapshot.get("schema") != "modebench-level-comparison-frozen-snapshot-v1":
        raise RuntimeError("wrong frozen Level1/Level2 comparison snapshot")
    reference = snapshot["reference_figure"]
    record = {
        "schema": "modebench-level-admission-and-terminal-v4",
        "target_step": 3072,
        "model": "Qwen2.5-0.5B-Instruct",
        "sources": {
            **reference["sources"],
            **snapshot["terminal_sources"],
            str(snapshot_path.resolve()): digest(snapshot_path),
        },
        "snapshot_collected_at_utc": snapshot["collected_at_utc"],
        "admission_rows": copy.deepcopy(reference["admission_rows"]),
        "admission_rows_role": "Historical development admission retained for provenance; construction is described in Methods and is not plotted here.",
        # Preserve the complete Graph example as separate terminal evidence.
        "partial_treatment": copy.deepcopy(reference["partial_treatment"]),
        "terminal_graph_role": "Historical separate example retained for provenance; the main figure uses all complete terminal domain blocks.",
        "terminal_comparison": build_terminal_comparison(
            snapshot["terminal_evaluations"], snapshot["terminal_admission"]),
        "interim_comparison_role": "Historical checkpoint-matched diagnostic retained in the sidecar; not plotted.",
        "interim_comparison": build_interim_comparison(snapshot["evaluations"], snapshot["availability"]),
        "terminal_progress_by_domain": build_terminal_progress(snapshot["level2_terminal_evaluations"]),
    }
    return record


def _axis(axis):
    axis.set_facecolor(style.PANEL)
    axis.grid(color=style.GRID, lw=.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=6.6)
    axis.tick_params(axis="x", labelsize=6.6)


def _level_pair(axis, level1, level2, y):
    axis.plot([level1, level2], [y, y], color=style.GRID, lw=2, zorder=1)
    axis.scatter(level1, y, s=30, facecolors="none", edgecolors=LEVEL1_COLOR, lw=1.2, zorder=3)
    axis.scatter(level2, y, s=30, color=LEVEL2_COLOR, zorder=4)


def render(record: dict, output: Path = OUT) -> None:
    # Half the previous height: the in-figure title is caption material, and the
    # legend rides on the same line as the panel titles instead of above them.
    figure = plt.figure(figsize=(style.WIDTH, 1.02))
    pass_axis = figure.add_axes([.20, .30, .33, .50])
    distinct_axis = figure.add_axes([.64, .30, .33, .50])
    for axis in (pass_axis, distinct_axis):
        _axis(axis)

    comparison = record["terminal_comparison"]
    for axis, metric, label in ((pass_axis, "pass8", "pass@8"),
                                (distinct_axis, "distinct8", "distinct@8")):
        for y, method in enumerate(METHODS):
            _level_pair(axis, comparison["means"]["level1"][method][metric],
                        comparison["means"]["level2"][method][metric], y)
        axis.set_ylim(3.5, -.5)
        axis.set_title(label, fontsize=7.1, pad=3)
        if metric == "pass8":
            axis.set_yticks(range(4), list(METHODS.values()), fontsize=6.5)
            axis.set_xlim(0, 1.0)
            axis.set_xticks((0, .5, 1), ("0", ".5", "1"))
            axis.set_xlabel("probability", fontsize=6.6)
        else:
            axis.set_yticks(range(4), [""] * 4)
            axis.set_xlim(0, 2.0)
            axis.set_xticks((0, 1, 2))
            axis.set_xlabel("verified modes", fontsize=6.6)

    domain_count = len(comparison["complete_domains"])
    partial_notes = []
    for domain in comparison["partial_domains"]:
        counts = [comparison["domain_results"][domain]["series"]["level2"][method]["n"] for method in METHODS]
        name = dict(zip(DOMAINS, ("Graph", "Countdown", "Python", "MathIR", "PantryPlan")))[domain]
        partial_notes.append(f"{name} Level 2 terminal n (method order above): " + ", ".join(map(str, counts)))
    if partial_notes:
        figure.text(.5, .005, "; ".join(partial_notes) + "; excluded from mean.",
                    ha="center", va="bottom", fontsize=5.6, color=style.MUTED)
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=LEVEL1_COLOR, label="Level 1"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL2_COLOR, markeredgecolor=LEVEL2_COLOR, label="Level 2"),
        ],
        frameon=True, framealpha=1.0, facecolor="white", edgecolor=style.GRID,
        fancybox=False, borderpad=.25, ncol=2, loc="upper center",
        bbox_to_anchor=(.245, 1.16), fontsize=6.4, handletextpad=.3, columnspacing=.9,
    )
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
                first = result["series"]["level1"][method]
                second = result["series"]["level2"][method]
                if result["complete_block"]:
                    _level_pair(axis, first["means"][metric], second["means"][metric], y)
                else:
                    for offset, series, color, fill in ((-.10, first, LEVEL1_COLOR, "none"),
                                                        (.10, second, LEVEL2_COLOR, LEVEL2_COLOR)):
                        if series["n"]:
                            axis.scatter(series["means"][metric], y + offset, s=24,
                                         facecolors=fill, edgecolors=color, lw=1.1, zorder=3)
                    axis.text(.97, y, f"{first['n']}/{second['n']}",
                              transform=axis.get_yaxis_transform(), ha="right", va="center",
                              fontsize=5.7, color=style.MUTED,
                              bbox=dict(facecolor="white", edgecolor="none", pad=.3))
            axis.set_yticks(range(4), list(METHODS.values()) if column == 0 else [""] * 4,
                            fontsize=6.3)
            axis.set_ylim(3.5, -.5)
            axis.set_xlabel(label, fontsize=6.6)
            if metric == "pass8":
                axis.set_xlim(-.035, 1.08 if not result["complete_block"] else 1.035)
                axis.set_xticks((0, .5, 1), ("0", ".5", "1"))
            else:
                axis.set_xlim(-.07, 3.25)
                axis.set_xticks((0, 1, 2, 3))
            if row == 0:
                subtitle = "5 matched seeds" if result["complete_block"] else "partial; n = L1/L2"
                axis.set_title(name + "\n" + subtitle, fontsize=6.9, pad=6)
    figure.text(.14, .975, "Terminal results by domain · Qwen2.5-0.5B · training pass 8",
                ha="left", va="top", fontsize=8.0)
    figure.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=LEVEL1_COLOR, label="Level 1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL2_COLOR, markeredgecolor=LEVEL2_COLOR, label="Level 2"),
    ], ncol=2, loc="upper center", bbox_to_anchor=(.55, .93), frameon=False, fontsize=6.8)
    figure.text(.55, .022, "Partial markers use each arm's available seeds; differences are not paired effects. Four K=8 draws/seed.",
                ha="center", va="bottom", fontsize=6.2, color=style.MUTED)
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def build_domain_record(record: dict) -> dict:
    return {
        "schema": "modebench-level-terminal-by-domain-v1",
        "sources": record["sources"],
        "terminal_comparison": record["terminal_comparison"],
        "partial_arm_policy": PARTIAL_ARM_POLICY,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--domain-output", type=Path, default=OUT.with_name("modebench_level_terminal_by_domain"))
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    record = build_record(snapshot, args.snapshot)
    render(record, args.output)
    render_domains(record, args.domain_output)
    args.domain_output.with_suffix(".json").write_text(
        json.dumps(build_domain_record(record), indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
