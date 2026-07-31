#!/usr/bin/env python3
"""Render a readable square-panel primary E70 + repair progress surface."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import plot_e70_clean_05b_wide_live as base
import plot_e70_clean_05b_wide_repairs_20260730 as repair


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "paper/figures/"
    "e68_e58_vs_grpo_05b_12ep_terminal_provenance_historical_20260730.png"
)
PDF_OUTPUT = OUTPUT.with_suffix(".pdf")
SIDECAR = (
    ROOT
    / "var/artifacts/"
    "e68_e58_vs_grpo_05b_12ep_terminal_provenance_"
    "historical_20260730_provenance.json"
)
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
BLUE = "#0057A8"
ORANGE = "#D55E00"
ORIGINAL_SEEDS = (43, 44, 45, 46, 47)
POINT_REPAIR_SEEDS = (76521,)
SEED_STYLES = {
    43: "-",
    44: (0, (5, 2)),
    45: (0, (1.5, 1.5)),
    46: "-.",
    47: (0, (7, 2, 1.5, 2)),
    76521: (0, (3, 1, 1, 1)),
}
METRICS = (
    ("greedy", "Neutral\npass@1", False),
    ("mean8", "Neutral mean\ncorrect@8", False),
    ("pass8", "Neutral\npass@8", False),
    ("distinct8", "Mean distinct\ncorrect@8", True),
)
ROWS = (
    (
        "Graph coloring",
        "Original cohort · audited",
        "graph_coloring",
        "gce70_clean_stage_a_05b_12pass",
        192,
        ORIGINAL_SEEDS,
    ),
    (
        "Countdown",
        "Original cohort · audited",
        "countdown",
        "cde70_clean_stage_a_05b_12pass",
        384,
        ORIGINAL_SEEDS,
    ),
    (
        "Python factors",
        "Original cohort · audited",
        "python_factor",
        "pye70_clean_stage_a_05b_12pass",
        384,
        ORIGINAL_SEEDS,
    ),
    (
        "MathIR action menu",
        "Original cohort · 9 jobs still active",
        "mathir",
        "mie70_clean_stage_a_05b_12pass",
        384,
        ORIGINAL_SEEDS,
    ),
    (
        "PointMaze geometry shift",
        "Original replacement · 10/10 audit passed",
        "point_maze_geometry_shift",
        None,
        None,
        ORIGINAL_SEEDS,
    ),
    (
        "PantryPlan support retention",
        "Secondary repair · five fresh paired seeds",
        "pantry_support_retention_repair",
        None,
        None,
        (),
    ),
    (
        "PointMaze algorithm repair",
        "Secondary balanced-warmstart v5 qualification",
        "point_maze_algorithm_repair",
        None,
        None,
        POINT_REPAIR_SEEDS,
    ),
    (
        "AntMaze harder task",
        "Secondary stable-handoff v18 controller/admission repair",
        "ant_maze_harder_repair",
        None,
        None,
        (),
    ),
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _points(
    domain: str, prefix: str | None, steps: int | None
) -> list[dict[str, Any]]:
    if domain == "point_maze_geometry_shift":
        return base._load_maze_stage_b_points(domain)
    if domain == "point_maze_algorithm_repair":
        return repair._load_point_repair()
    if prefix is not None and steps is not None:
        return base._load_points(prefix, steps)
    return []


def _series(
    points: Iterable[Mapping[str, Any]],
    arm: str,
    seed: int,
    metric: str,
) -> list[tuple[float, float]]:
    merged: dict[float, float] = {}
    for row in points:
        if (
            row.get("arm") == arm
            and row.get("seed") == seed
            and _finite(row.get("passes"))
            and _finite(row.get(metric))
        ):
            merged[round(float(row["passes"]), 9)] = float(row[metric])
    return sorted(merged.items())


def _mean(
    points: Iterable[Mapping[str, Any]],
    arm: str,
    seeds: tuple[int, ...],
    metric: str,
) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for seed in seeds:
        for pass_index, value in _series(points, arm, seed, metric):
            grouped[pass_index].append(value)
    xs = sorted(grouped)
    return (
        xs,
        [statistics.fmean(grouped[x]) for x in xs],
        [min(grouped[x]) for x in xs],
        [max(grouped[x]) for x in xs],
        [len(grouped[x]) for x in xs],
    )


def _plot(
    axis: plt.Axes,
    points: list[dict[str, Any]],
    seeds: tuple[int, ...],
    metric: str,
    integer: bool,
) -> None:
    values: list[float] = []
    for arm, color, marker in (
        (CONTROL, BLUE, "o"),
        (TREATMENT, ORANGE, "D"),
    ):
        for seed in seeds:
            rows = _series(points, arm, seed, metric)
            if not rows:
                continue
            axis.plot(
                [row[0] for row in rows],
                [row[1] for row in rows],
                color=color,
                linestyle=SEED_STYLES[seed],
                linewidth=1.0,
                marker=marker,
                markersize=2.5,
                markerfacecolor=color if arm == CONTROL else "white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                alpha=0.55,
                zorder=2,
            )
        xs, means, lows, highs, counts = _mean(
            points, arm, seeds, metric
        )
        if not xs:
            continue
        values.extend(lows)
        values.extend(highs)
        axis.fill_between(
            xs, lows, highs, color=color, alpha=0.10, linewidth=0, zorder=1
        )
        axis.plot(
            xs,
            means,
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=4.2,
            markerfacecolor=color if arm == CONTROL else "white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            zorder=4,
        )
        axis.annotate(
            f"n={counts[-1]}",
            (xs[-1], means[-1]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.3,
            color=color,
            clip_on=True,
        )
    axis.set_xlim(0, 12)
    if metric in {"greedy", "pass8"}:
        axis.set_ylim(0, 1.02)
    elif values:
        high = max(values)
        axis.set_ylim(0, high * 1.08 if high > 0 else 1)
    else:
        axis.set_ylim(0, 1)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.7, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7.5, length=2.5, width=0.7)
    axis.xaxis.set_major_locator(MaxNLocator(7))
    axis.yaxis.set_major_locator(MaxNLocator(5, integer=integer))
    axis.set_box_aspect(1)


def _status(domain: str) -> tuple[str, str, str]:
    if domain == "pantry_support_retention_repair":
        return repair._repair_status(domain)
    if domain == "ant_maze_harder_repair":
        return repair._repair_status(domain)
    raise KeyError(domain)


def render() -> None:
    repair.REPAIR_INPUTS["pantry_audit"] = (
        ROOT
        / "var/artifacts/"
        "pantry_support_retention_final_v1_audit_amendment_v3.json"
    )
    repair.REPAIR_INPUTS["point_protocol"] = (
        ROOT
        / "paper/preregistration/"
        "point_maze_balanced_warmstart_v5_20260730.md"
    )
    repair.REPAIR_INPUTS["point_identity"] = (
        ROOT
        / "var/artifacts/"
        "point_maze_balanced_warmstart_v5_identity.json"
    )
    repair.REPAIR_INPUTS["point_audit"] = (
        ROOT
        / "var/artifacts/"
        "point_maze_balanced_warmstart_v5_qualification.json"
    )
    # Bind the displayed Ant state to the currently active v18 gate.
    repair.REPAIR_INPUTS["ant_protocol"] = (
        ROOT
        / "paper/preregistration/"
        "ant_stable_handoff_controller_v18_20260730.md"
    )
    repair.REPAIR_INPUTS["ant_identity"] = (
        ROOT
        / "var/artifacts/"
        "ant_stable_handoff_controller_v18_identity.json"
    )
    repair.REPAIR_INPUTS["ant_controller_gate"] = (
        ROOT
        / "var/maze_runtime/controllers/"
        "ant_stable_handoff_v18.evaluation.json"
    )
    repair.REPAIR_INPUTS["ant_v17_failed_gate"] = (
        ROOT
        / "var/maze_runtime/controllers/"
        "ant_sequential_waypoint_v17.evaluation.json"
    )

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
        len(ROWS),
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
    repairs = repair._repair_summary()
    figure.text(
        0.035,
        0.982,
        "Clean 0.5B campaign + repaired environment cohorts",
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
        va="top",
    )
    figure.text(
        0.035,
        0.961,
        "Online verified MaxEnt vs compute-matched Dr.GRPO · "
        "12 passes · square primary-result panels",
        fontsize=11.5,
        color="#475569",
        va="top",
    )
    figure.text(
        0.035,
        0.943,
        f"Original estimator: {repair._count_original_terminal()}/80 audited "
        "terminal · Repairs excluded from that count · "
        f"Pantry {repairs['pantry_model_jobs_launched']}/10 launched · "
        f"Point v5 online {repairs['point_model_jobs_launched']}/10 · "
        f"Ant v18 job {repairs['ant_controller_job_id']} "
        f"({repairs['ant_controller_gate_status']}) · "
        f"{generated.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=9.5,
        color="#64748B",
        va="top",
    )

    for column, (_metric, title, _integer) in enumerate(METRICS, start=1):
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
    ) in enumerate(ROWS):
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
            headline, detail, color = _status(domain)
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
            _plot(axis, points, seeds, metric, integer)
            if row_index == len(ROWS) - 1:
                axis.set_xlabel("training passes", fontsize=8.5)

    handles = [
        Line2D(
            [], [], color=BLUE, lw=2.5, marker="o",
            label="compute-matched Dr.GRPO",
        ),
        Line2D(
            [], [], color=ORANGE, lw=2.5, marker="D",
            markerfacecolor="white",
            label="online verified MaxEnt",
        ),
        Line2D(
            [], [], color="#475569", lw=2.3,
            label="thick = available-seed mean; band = range",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper right",
        ncol=1,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.985, 0.987),
    )
    figure.text(
        0.5,
        0.014,
        "Original rows retain identity-bound trajectories. Repair rows are "
        "prospective secondary cohorts and remain fail-closed until their "
        "own audits pass. No historical values are copied or carried forward.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )

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

    provenance = {
        "schema": "e70-primary-square-with-secondary-repairs-v1",
        "generated_at": generated.isoformat(),
        "output": str(OUTPUT.relative_to(ROOT)),
        "rows": [row[2] for row in ROWS],
        "panels": [metric for metric, _title, _integer in METRICS],
        "layout": {
            "rows": 8,
            "primary_metric_columns": 4,
            "figsize_inches": [16.0, 24.0],
            "axes_box_aspect": 1,
            "status_rows_span_all_metric_columns": True,
        },
        "original_campaign_terminal_cells": repair._count_original_terminal(),
        "original_campaign_expected_cells": 80,
        "repair_band": repairs,
        "repair_excluded_from_original_estimator": True,
        "historical_values_imported": False,
        "carry_forward": False,
        "hashes": {
            "plot_source": _sha256(Path(__file__).resolve()),
            "figure": _sha256(OUTPUT),
            "stage_a_audit": _sha256(base.AUDIT),
            **{
                key: _sha256(path)
                for key, path in repair.REPAIR_INPUTS.items()
            },
        },
    }
    temporary_sidecar = SIDECAR.with_suffix(SIDECAR.suffix + ".tmp")
    temporary_sidecar.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_sidecar.replace(SIDECAR)
    print(
        "[e70-primary-square] "
        f"output={OUTPUT.relative_to(ROOT)} rows=8 panels=4"
    )


if __name__ == "__main__":
    render()
