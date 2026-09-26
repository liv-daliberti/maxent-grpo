#!/usr/bin/env python3
"""Render the readable current-outcome/mechanism companion page."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_e70_primary_square_repairs_v3_20260730 as corrected


primary = corrected.primary
ROOT = primary.ROOT
OUTPUT = (
    ROOT
    / "paper/figures/"
    "e68_e58_vs_grpo_05b_12ep_current_diagnostics_20260730.png"
)
PDF_OUTPUT = OUTPUT.with_suffix(".pdf")
SIDECAR = (
    ROOT
    / "var/artifacts/"
    "e68_e58_vs_grpo_05b_12ep_current_diagnostics_"
    "20260730_provenance.json"
)
METRICS = (
    (
        "online_canonical_tracked_outcomes",
        "Cumulative verified\ndiscoveries",
        True,
    ),
    (
        "canonical_replay_available_modes",
        "Replay-available\nmodes",
        True,
    ),
    (
        "online_canonical_mean_support_per_prompt",
        "Mean modes per\ntracked prompt",
        False,
    ),
    (
        "online_canonical_support_at_least_two_prompt_fraction",
        "Tracked prompts\nwith ≥2 modes",
        False,
    ),
)


def _points(domain, prefix, steps):
    rows = [
        dict(row)
        for row in primary._points(domain, prefix, steps)
    ]
    for row in rows:
        if (
            "online_canonical_support_at_least_two_prompt_fraction"
            not in row
            and "canonical_support_at_least_two_prompt_fraction" in row
        ):
            row[
                "online_canonical_support_at_least_two_prompt_fraction"
            ] = row["canonical_support_at_least_two_prompt_fraction"]
    return rows


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def render() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#64748B",
            "ytick.color": "#64748B",
        }
    )
    figure = plt.figure(figsize=(16.0, 24.0), facecolor="#F8FAFC")
    grid = figure.add_gridspec(
        len(primary.ROWS),
        len(METRICS) + 1,
        width_ratios=[1.55, 3, 3, 3, 3],
        left=0.035,
        right=0.99,
        top=0.915,
        bottom=0.045,
        wspace=0.20,
        hspace=0.34,
    )
    generated = datetime.now(timezone.utc)
    repairs = primary.repair._repair_summary()
    figure.text(
        0.035,
        0.982,
        "Clean 0.5B campaign — mechanism diagnostics",
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
        va="top",
    )
    figure.text(
        0.035,
        0.961,
        "Companion mechanism page · the four frozen outcome metrics are "
        "kept together on page 1",
        fontsize=11.5,
        color="#475569",
        va="top",
    )
    figure.text(
        0.035,
        0.943,
        f"Original estimator: {primary.repair._count_original_terminal()}/80 "
        "audited terminal · repairs excluded · "
        f"Pantry {repairs['pantry_model_jobs_launched']}/10 · "
        f"Point v5 online {repairs['point_model_jobs_launched']}/10 · "
        f"Ant v18 {repairs['ant_controller_gate_status']} · "
        f"{generated.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=9.5,
        color="#64748B",
        va="top",
    )
    for column, (_metric, title, _integer) in enumerate(
        METRICS, start=1
    ):
        position = grid[0, column].get_position(figure)
        figure.text(
            (position.x0 + position.x1) / 2,
            0.922,
            title,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1E293B",
        )
    for row_index, (
        label,
        note,
        domain,
        prefix,
        steps,
        seeds,
    ) in enumerate(primary.ROWS):
        label_axis = figure.add_subplot(grid[row_index, 0])
        label_axis.axis("off")
        label_axis.text(
            0.02,
            0.62,
            label,
            fontsize=12,
            fontweight="bold",
            color="#0F172A",
            va="center",
            wrap=True,
        )
        label_axis.text(
            0.02,
            0.32,
            note,
            fontsize=8.3,
            color="#64748B",
            va="center",
            wrap=True,
        )
        points = _points(domain, prefix, steps)
        if not points and domain in {
            "pantry_support_retention_repair",
            "point_maze_algorithm_repair",
            "ant_maze_harder_repair",
        }:
            status_axis = figure.add_subplot(grid[row_index, 1:])
            headline, detail, color = primary._status(domain)
            status_axis.set_facecolor("#FFFFFF")
            status_axis.set_xticks([])
            status_axis.set_yticks([])
            for spine in status_axis.spines.values():
                spine.set_color("#CBD5E1")
                spine.set_linewidth(0.9)
            status_axis.text(
                0.5,
                0.60,
                headline,
                transform=status_axis.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )
            status_axis.text(
                0.5,
                0.39,
                detail,
                transform=status_axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#64748B",
                wrap=True,
            )
            continue
        for column, (metric, _title, integer) in enumerate(
            METRICS, start=1
        ):
            axis = figure.add_subplot(grid[row_index, column])
            axis.set_facecolor("#FFFFFF")
            primary._plot(axis, points, seeds, metric, integer)
            if row_index == len(primary.ROWS) - 1:
                axis.set_xlabel("training passes", fontsize=8.5)
    figure.legend(
        handles=[
            Line2D(
                [], [], color=primary.BLUE, lw=2.5, marker="o",
                label="compute-matched Dr.GRPO",
            ),
            Line2D(
                [], [], color=primary.ORANGE, lw=2.5, marker="D",
                markerfacecolor="white",
                label="online verified MaxEnt",
            ),
            Line2D(
                [], [], color="#475569", lw=2.3,
                label="thick = available-seed mean; band = range",
            ),
        ],
        loc="upper right",
        ncol=1,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.985, 0.987),
    )
    figure.text(
        0.5,
        0.014,
        "Mechanism panels expose discovery/support/replay state; they are "
        "descriptive telemetry, not additional efficacy claims.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_png = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    figure.savefig(
        temporary_png,
        dpi=220,
        format="png",
        facecolor=figure.get_facecolor(),
    )
    temporary_png.replace(OUTPUT)
    temporary_pdf = PDF_OUTPUT.with_name(f".{PDF_OUTPUT.name}.tmp")
    figure.savefig(
        temporary_pdf,
        format="pdf",
        facecolor=figure.get_facecolor(),
    )
    temporary_pdf.replace(PDF_OUTPUT)
    plt.close(figure)
    _atomic_json(
        SIDECAR,
        {
            "schema": "e70-current-diagnostics-square-v1",
            "generated_at": generated.isoformat(),
            "output": str(OUTPUT.relative_to(ROOT)),
            "rows": [row[2] for row in primary.ROWS],
            "panels": [metric for metric, _title, _integer in METRICS],
            "axes_box_aspect": 1,
            "repair_excluded_from_original_estimator": True,
        },
    )
    print(f"[e70-diagnostics-square] wrote {OUTPUT}")
    print(f"[e70-diagnostics-square] wrote {PDF_OUTPUT}")


if __name__ == "__main__":
    render()
