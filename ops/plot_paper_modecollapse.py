#!/usr/bin/env python3
"""Render the paper's validator-checked ModeBench evidence figures.

Training panels read the audited five-seed E70 and original Pantry Stage-B
curves in the historical E68 trajectory style. Every row ends at the common
pass-12 checkpoint; incomplete cohorts fail closed instead of being averaged.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
EXP_SCALING = ROOT / "ops/exp_scaling"
if str(EXP_SCALING) not in sys.path:
    sys.path.insert(0, str(EXP_SCALING))
import plot_e61r1_e58_vs_grpo_12pass as e68_plot  # noqa: E402

OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))
import paper_style as style  # noqa: E402

OUT_STORY = ROOT / "paper/figures/modecollapse_story"
OUT_TRAINING = ROOT / "paper/figures/modecollapse_training"

INK = style.INK
MUTED = style.MUTED
GRID = style.GRID
PANEL = style.PANEL
WHITE = style.WHITE
CONTROL = style.CONTROL
METHOD = style.METHOD
ACCENT = CONTROL
MODE_COLORS = list(style.MODE_RAMP)

# The arm colours the trajectory panels draw with. These deliberately shadow
# ``e68_plot.BLUE``/``ORANGE``: that module paints matched Dr.GRPO blue and
# xGRPO orange, which is the opposite of the orange/teal the E72 figures use
# for the same two arms. Colour follows the entity, so the shared definition
# wins and the historical one is not imported for drawing.
ARM_COLOR = {e68_plot.CONTROL: CONTROL, e68_plot.TREATMENT: METHOD}

DOMAINS = [
    "Graph coloring",
    "Countdown",
    "Python factors",
    "MathIR action menu",
    "PantryPlan",
]
PANTRY_DOMAIN = "PantryPlan"

style.apply_rcparams()


def rounded_box(ax, x, y, w, h, *, face=WHITE, edge=GRID, radius=0.018, lw=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, *, size=8.2, weight="normal", color=INK, **kwargs):
    return ax.text(
        x,
        y,
        value,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=color,
        **kwargs,
    )


def arrow(ax, start, end, *, color=MUTED, lw=1.1):
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        color=color,
        shrinkA=1,
        shrinkB=1,
        clip_on=False,
    )
    ax.add_patch(patch)


def distribution(ax, x, y, w, values, *, label_prefix="m"):
    max_value = max(values)
    for index, (value, color) in enumerate(zip(values, MODE_COLORS), start=1):
        yy = y - (index - 1) * 0.105
        text(
            ax,
            x,
            yy,
            f"{label_prefix}{index}",
            size=7.0,
            weight="bold",
            color=MUTED,
            ha="left",
            va="center",
        )
        rounded_box(
            ax,
            x + 0.037,
            yy - 0.022,
            w,
            0.044,
            face="#E8EEF3",
            edge="#E8EEF3",
            radius=0.009,
            lw=0,
        )
        rounded_box(
            ax,
            x + 0.037,
            yy - 0.022,
            max(0.012, w * value / max_value),
            0.044,
            face=color,
            edge=color,
            radius=0.009,
            lw=0,
        )


def render_story() -> None:
    fig = plt.figure(figsize=(7.35, 2.18))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    rounded_box(ax, 0.010, 0.055, 0.305, 0.885, face=PANEL, radius=0.024, lw=1.15)
    rounded_box(ax, 0.338, 0.055, 0.285, 0.885, face="#FCF7F4", radius=0.024, lw=1.15)
    rounded_box(ax, 0.646, 0.055, 0.344, 0.885, face="#F1FAF8", radius=0.024, lw=1.15)

    text(ax, 0.032, 0.842, "A", size=11.5, weight="bold", color=MODE_COLORS[0], va="center")
    text(ax, 0.065, 0.842, "Executable modes", size=11.0, weight="bold", va="center")
    text(ax, 0.034, 0.717, "response", size=7.4, weight="bold", color=MUTED, va="center")
    text(ax, 0.154, 0.717, "execute + verify", size=7.4, weight="bold", color=MUTED, va="center")
    text(ax, 0.261, 0.717, "key", size=7.4, weight="bold", color=MUTED, va="center")

    examples = [
        ("1×2×3×4", "AST-1", MODE_COLORS[0]),
        ("(1+3)×(2+4)", "AST-2", MODE_COLORS[1]),
        ("4×(3+2+1)", "AST-3", MODE_COLORS[2]),
        ("4×4+…", "invalid", MUTED),
    ]
    for row, (response, key, color) in enumerate(examples):
        yy = 0.600 - row * 0.128
        rounded_box(ax, 0.034, yy - 0.037, 0.092, 0.074, face=WHITE, edge=GRID, radius=0.010)
        text(ax, 0.080, yy, response, size=7.2, weight="bold", ha="center", va="center")
        arrow(ax, (0.132, yy), (0.208, yy), color=GRID, lw=1.0)
        rounded_box(
            ax,
            0.214,
            yy - 0.037,
            0.077,
            0.074,
            face=WHITE if key != "invalid" else "#EEF1F4",
            edge=color,
            radius=0.010,
        )
        text(ax, 0.2525, yy, key, size=7.0, weight="bold", color=color, ha="center", va="center")
    text(
        ax,
        0.163,
        0.090,
        "format aliases merge; executed alternatives remain distinct",
        size=6.8,
        color=MUTED,
        ha="center",
        va="center",
    )

    text(ax, 0.360, 0.842, "B", size=11.5, weight="bold", color=ACCENT, va="center")
    text(ax, 0.393, 0.842, "Dr.GRPO collapse", size=10.2, weight="bold", va="center")
    text(ax, 0.363, 0.725, "all correct modes receive reward 1", size=7.4, color=MUTED, va="center")
    distribution(ax, 0.366, 0.602, 0.202, [0.90, 0.14, 0.07, 0.04])
    text(ax, 0.482, 0.130, "frequency-weighted updates", size=7.3, weight="bold", color=ACCENT, ha="center")
    text(ax, 0.482, 0.072, "no force restores a missed mode", size=6.8, color=MUTED, ha="center")

    text(ax, 0.668, 0.842, "C", size=11.5, weight="bold", color=METHOD, va="center")
    text(ax, 0.701, 0.842, "xGRPO", size=10.5, weight="bold", va="center")
    text(ax, 0.701, 0.755, "online verified MaxEnt", size=7.0, color=MUTED, va="center")
    rounded_box(ax, 0.676, 0.630, 0.083, 0.086, face=WHITE, edge=METHOD, radius=0.012)
    rounded_box(ax, 0.780, 0.630, 0.083, 0.086, face=WHITE, edge=METHOD, radius=0.012)
    rounded_box(ax, 0.884, 0.630, 0.083, 0.086, face=WHITE, edge=METHOD, radius=0.012)
    text(ax, 0.7175, 0.673, "verified\nbank", size=7.2, weight="bold", color=METHOD, ha="center", va="center")
    text(ax, 0.8215, 0.673, "rare + new\nadvantage", size=7.0, weight="bold", color=METHOD, ha="center", va="center")
    text(ax, 0.9255, 0.673, "global\nreplay", size=7.2, weight="bold", color=METHOD, ha="center", va="center")
    arrow(ax, (0.761, 0.673), (0.777, 0.673), color=METHOD)
    arrow(ax, (0.865, 0.673), (0.881, 0.673), color=METHOD)
    distribution(ax, 0.722, 0.500, 0.200, [0.95, 0.82, 0.77, 0.70])
    text(ax, 0.842, 0.090, "pressure acts only on validator-positive execution modes", size=6.8, color=MUTED, ha="center")

    text(ax, 0.986, 0.965, "schematic", size=6.4, color=MUTED, ha="right", va="top")
    OUT_STORY.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STORY.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_STORY.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)




E68_TRAIN_METRICS = {
    "greedy": {
        "title": "neutral pass@1",
        "suffix": "pass1",
        "label": "Neutral pass@1",
        "integer": False,
    },
    "pass8": {
        "title": "neutral pass@8",
        "suffix": "pass8",
        "label": "Neutral pass@8",
        "integer": False,
    },
    "distinct8": {
        "title": "mean # distinct correct@8",
        "suffix": "distinct8",
        "label": "Mean # distinct correct@8",
        "integer": True,
    },
    "online_canonical_tracked_outcomes": {
        "title": "cumulative verified discoveries",
        "suffix": "discoveries",
        "label": "Cumulative verified discoveries",
        "integer": True,
    },
}
E70_TRAIN_DOMAIN_SPECS = {
    "Graph coloring": ("gce71_scale384_05b_12pass", 384),
    "Countdown": ("cde70_clean_stage_a_05b_12pass", 384),
    "Python factors": ("pye70_clean_stage_a_05b_12pass", 384),
    "MathIR action menu": ("mie70_clean_stage_a_05b_12pass", 384),
    "PantryPlan": ("ppe71_scale384_05b_12pass", 384),
}
PAPER_SEEDS = (43, 44, 45, 46, 47)
PLOT_PASSES = (0, 3, 6, 9, 12)
PAPER_SEED_STYLES = {
    43: "-",
    44: (0, (5, 2)),
    45: (0, (1.5, 1.5)),
    46: "-.",
    47: (0, (7, 2, 1.5, 2)),
}
E70_AUDIT = ROOT / "var/artifacts/e70_clean_stage_a_05b_audit_latest.json"
PANTRY_CURVE = ROOT / "var/artifacts/ppe71_scale384_05b_12pass_scaling_curve.json"
PANTRY_AUDIT = ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json"


def _load_training_points(prefix: str, steps_per_pass: int) -> list[dict]:
    """Load one identity-bound E70 curve without carrying checkpoints forward."""

    path = ROOT / f"var/artifacts/{prefix}_scaling_curve.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    merged: dict[tuple[str, int, int], dict] = {}
    for raw in payload:
        arm = raw.get("arm")
        seed = raw.get("seed")
        step = raw.get("step")
        if (
            arm not in {e68_plot.CONTROL, e68_plot.TREATMENT}
            or seed not in PAPER_SEEDS
            or not e68_plot._finite(step)
            or raw.get("split") != "multi_answer"
        ):
            continue
        key = (str(arm), int(seed), int(step))
        point = merged.setdefault(
            key,
            {
                "arm": str(arm),
                "seed": int(seed),
                "step": int(step),
                "passes": float(step) / float(steps_per_pass),
            },
        )
        if int(step) == 0:
            point["online_canonical_tracked_outcomes"] = 0.0
        for metric, value in raw.items():
            if e68_plot._finite(value):
                point[str(metric)] = float(value)
    return [merged[key] for key in sorted(merged)]


def _require_terminal_audits() -> None:
    core = json.loads(E70_AUDIT.read_text(encoding="utf-8"))
    if core.get("status") != "pass" or core.get("summary", {}).get("terminal_runs") != 40:
        raise ValueError("E70 Stage A is not a complete audited 40-run cohort")
    pantry = json.loads(PANTRY_AUDIT.read_text(encoding="utf-8"))
    if pantry.get("status") != "pass" or pantry.get("decision") != "pantry_terminal_eligible":
        raise ValueError("original PantryPlan Stage B audit is not passing")


def _e68_training_points(
    _document: dict,
    domain: str,
) -> tuple[list[dict], int, tuple[int, ...]]:
    """Load five seeds at five displayed checkpoints through audited pass 12."""

    _require_terminal_audits()
    prefix, steps_per_pass = E70_TRAIN_DOMAIN_SPECS[domain]
    points = _load_training_points(prefix, steps_per_pass)
    frozen_pass = 12
    seeds = PAPER_SEEDS
    points = [
        point
        for point in points
        if point["arm"] in {e68_plot.CONTROL, e68_plot.TREATMENT}
        and float(point["passes"]) in PLOT_PASSES
    ]
    for metric in E68_TRAIN_METRICS:
        for arm in (e68_plot.CONTROL, e68_plot.TREATMENT):
            for seed in seeds:
                endpoint = [
                    point
                    for point in points
                    if point["arm"] == arm
                    and point["seed"] == seed
                    and float(point["passes"]) == frozen_pass
                    and e68_plot._finite(point.get(metric))
                ]
                if len(endpoint) != 1:
                    raise ValueError(
                        f"{domain}: expected one {metric} endpoint for "
                        f"{arm}/seed {seed} at pass 12, got {len(endpoint)}"
                    )
    return points, frozen_pass, seeds


def _complete_mean(points, arm, metric, seeds):
    pass_sets = []
    for seed in seeds:
        pass_sets.append(
            {
                float(point["passes"])
                for point in points
                if point["arm"] == arm
                and point["seed"] == seed
                and e68_plot._finite(point.get(metric))
            }
        )
    shared = sorted(set.intersection(*pass_sets)) if pass_sets else []
    means, lows, highs = [], [], []
    for training_pass in shared:
        values = [
            next(
                float(point[metric])
                for point in points
                if point["arm"] == arm
                and point["seed"] == seed
                and float(point["passes"]) == training_pass
                and e68_plot._finite(point.get(metric))
            )
            for seed in seeds
        ]
        means.append(float(np.mean(values)))
        lows.append(float(np.min(values)))
        highs.append(float(np.max(values)))
    return shared, means, lows, highs


def _seed_style(seed):
    return PAPER_SEED_STYLES[seed]


def _plot_e68_training_axis(
    axis,
    points: list[dict],
    metric: str,
    frozen_pass: int,
    seeds: tuple[int, ...],
) -> None:
    """Plot one panel with the historical E68 seed/mean/range grammar."""

    plotted_values: list[float] = []
    for arm, zorder in ((e68_plot.CONTROL, 2), (e68_plot.TREATMENT, 3)):
        color = ARM_COLOR[arm]
        dash = style.ARM_DASH[color]
        # Seeds stay in the panel --- the caption promises them --- but they
        # carry no markers and sit at low alpha, so they read as the texture of
        # the spread rather than competing with the mean.
        for seed in seeds:
            xs, ys = e68_plot._series(points, arm, seed, metric)
            plotted_values.extend(ys)
            axis.plot(
                xs,
                ys,
                color=color,
                lw=style.SEED_LW,
                alpha=0.38,
                zorder=zorder,
            )
        xs, means, lows, highs = _complete_mean(points, arm, metric, seeds)
        if xs:
            plotted_values.extend(lows)
            plotted_values.extend(highs)
            axis.fill_between(
                xs,
                lows,
                highs,
                color=color,
                alpha=style.BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                xs,
                means,
                color=color,
                lw=style.MEAN_LW,
                linestyle=dash,
                zorder=5,
            )

    e68_plot._set_y_limits(axis, metric, plotted_values)
    axis.set_xlim(0, frozen_pass + max(0.08, frozen_pass * 0.06))
    axis.set_xticks(PLOT_PASSES)
    style.style_axis(axis)
    if E68_TRAIN_METRICS[metric]["integer"]:
        axis.yaxis.set_major_locator(MaxNLocator(5, integer=True))


def _e68_training_legend() -> list[Line2D]:
    return [
        Line2D(
            [], [], color=CONTROL, lw=style.MEAN_LW,
            linestyle=style.ARM_DASH[CONTROL],
            label="matched Dr.GRPO",
        ),
        Line2D(
            [], [], color=METHOD, lw=style.MEAN_LW,
            linestyle=style.ARM_DASH[METHOD],
            label="xGRPO",
        ),
        Line2D(
            [], [], color=MUTED, lw=style.SEED_LW, alpha=0.38,
            label="individual seeds",
        ),
        Line2D(
            [], [], color=MUTED, lw=style.MEAN_LW,
            label="mean + seed range",
        ),
    ]


def render_e68_training() -> None:
    """Render the manuscript's E68-style five-domain by four-metric grid."""

    document: dict = {}
    metrics = tuple(E68_TRAIN_METRICS)
    # Authored at the shared canvas width rather than 11.5in: at \linewidth the
    # old canvas was scaled by .48, which put this grid's nominal 8.2pt type on
    # the page at about 3.9pt.
    fig, axes = plt.subplots(
        len(DOMAINS),
        len(metrics),
        figsize=(style.WIDTH, style.panel_height(len(DOMAINS))),
        sharex="row",
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, domain in enumerate(DOMAINS):
        points, frozen_pass, seeds = _e68_training_points(document, domain)
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index, column_index]
            _plot_e68_training_axis(axis, points, metric, frozen_pass, seeds)
            # Every row carries its metric titles, not just the top one: the
            # rows are domains on independent y scales, and the paper's figure
            # contract requires all five to be labelled.
            axis.set_title(
                E68_TRAIN_METRICS[metric]["title"],
                fontsize=style.TITLE_FONT,
                color=INK,
                pad=3,
            )
            if column_index == 0:
                axis.set_ylabel(
                    domain, fontsize=style.LABEL_FONT, fontweight="bold"
                )
                axis.text(
                    0.985,
                    0.05,
                    f"paired through pass {frozen_pass}",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=style.SMALL_FONT - 0.8,
                    color=MUTED,
                )
            if row_index == len(DOMAINS) - 1:
                axis.set_xlabel("training passes", fontsize=style.LABEL_FONT)

    style.bottom_legend(
        fig,
        _e68_training_legend(),
        [handle.get_label() for handle in _e68_training_legend()],
        y=-0.022,
    )
    fig.text(
        0.5,
        -0.045,
        "Thin lines are seeds 43–47; thick lines are five-seed means and bands "
        "are seed ranges. Each row shows checkpoints 0, 3, 6, 9, and 12; all endpoints are audited.",
        ha="center",
        fontsize=style.SMALL_FONT,
        color=MUTED,
    )
    style.save(fig, OUT_TRAINING)
    plt.close(fig)


MODES_METRIC = "distinct8"
# Frozen pass-12 five-seed means transcribed into `tab:main-results`, keyed by
# domain as (matched Dr.GRPO, xGRPO). Rendering aborts rather than publish a
# curve whose endpoint no longer reproduces the manuscript table.
FROZEN_MODE_ENDPOINTS = {
    "Graph coloring": (0.325, 2.406),
    "Countdown": (0.627, 1.893),
    "Python factors": (0.172, 1.594),
    "MathIR action menu": (0.666, 0.926),
    "PantryPlan": (0.634, 2.186),
}


def render_modes_by_epoch() -> Path:
    """Render the main-body average-modes trajectory: one panel per domain.

    Restricting the grid to ``distinct8`` puts all five domains on one shared
    vertical scale, so the average number of distinct verified modes an
    eight-sample draw returns is comparable across domains and readable as a
    trajectory rather than as an endpoint.
    """

    document: dict = {}
    out = OUT_TRAINING.with_name("modes_per_epoch")
    fig, axes = plt.subplots(
        1,
        len(DOMAINS),
        figsize=(style.WIDTH, 2.15),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    ceiling = 0.0
    for panel_index, (axis, domain) in enumerate(zip(axes.flat, DOMAINS)):
        points, frozen_pass, seeds = _e68_training_points(document, domain)
        for arm_index, arm in enumerate(
            (e68_plot.CONTROL, e68_plot.TREATMENT)
        ):
            _, means, _, _ = _complete_mean(points, arm, MODES_METRIC, seeds)
            frozen = FROZEN_MODE_ENDPOINTS[domain][arm_index]
            if not means or abs(means[-1] - frozen) > 5e-4:
                raise ValueError(
                    f"{domain}/{arm}: pass-{frozen_pass} mean # distinct "
                    f"correct@8 is {means[-1] if means else None}, but the "
                    f"manuscript reports {frozen}"
                )
        _plot_e68_training_axis(axis, points, MODES_METRIC, frozen_pass, seeds)
        ceiling = max(ceiling, axis.get_ylim()[1])
        axis.axhline(
            1.0,
            color=MUTED,
            lw=0.75,
            ls=(0, (3, 2)),
            zorder=0,
        )
        axis.set_title(domain, fontsize=style.TITLE_FONT, color=INK, pad=3)
        # The reporting grid is 0, 3, 6, 9, 12; the shared styler relabels the
        # axis on an even locator, so restore the evaluated checkpoints.
        axis.set_xticks(PLOT_PASSES)
        if panel_index == 0:
            # Same name the results table uses for `distinct@8`. Rendered by
            # matplotlib's own text engine, so the `#` needs no LaTeX escape.
            axis.set_ylabel("#modes", fontsize=style.LABEL_FONT, labelpad=2)

    axes.flat[0].set_ylim(0.0, ceiling)
    axes.flat[0].text(
        6.0,
        1.06,
        "one mode",
        fontsize=style.SMALL_FONT,
        color=MUTED,
        ha="center",
        va="bottom",
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.6},
    )

    fig.supxlabel("training epoch", fontsize=style.FONT)
    # One row is only 2.15in tall, so the reference offset would drop the
    # legend on top of the shared x label; clear it explicitly.
    style.bottom_legend(
        fig,
        _e68_training_legend(),
        [handle.get_label() for handle in _e68_training_legend()],
        y=-0.14,
    )
    style.save(fig, out)
    plt.close(fig)
    return out


COMPACT_METRICS = ("pass8", "distinct8")


def render_e68_training_compact() -> Path:
    """Render the main-body two-metric by five-domain trajectory grid.

    Identical data and grammar to :func:`render_e68_training`, restricted to
    the two headline metrics so the panels stay legible at ``\\linewidth``.
    """

    document: dict = {}
    out = OUT_TRAINING.with_name("modecollapse_training_compact")
    fig, axes = plt.subplots(
        len(COMPACT_METRICS),
        len(DOMAINS),
        figsize=(7.35, 3.55),
        squeeze=False,
    )
    for column_index, domain in enumerate(DOMAINS):
        points, frozen_pass, seeds = _e68_training_points(document, domain)
        for row_index, metric in enumerate(COMPACT_METRICS):
            axis = axes[row_index, column_index]
            _plot_e68_training_axis(axis, points, metric, frozen_pass, seeds)
            axis.tick_params(labelsize=6.2)
            if row_index == 0:
                axis.set_title(domain, fontsize=8.4, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(
                    E68_TRAIN_METRICS[metric]["title"],
                    fontsize=7.6,
                )
            if row_index == len(COMPACT_METRICS) - 1:
                axis.set_xlabel(
                    f"training passes (paired to {frozen_pass})",
                    fontsize=6.6,
                )

    fig.legend(
        handles=_e68_training_legend(),
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=7.4,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), h_pad=0.9, w_pad=0.9)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        out.with_suffix(".png"),
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)
    return out


def render_pantry_training_strip(document: dict) -> Path:
    """Render PantryPlan's four registered headline metrics in one row."""

    out = OUT_TRAINING.with_name("modecollapse_training_pantry")
    metrics = tuple(E68_TRAIN_METRICS)
    points, frozen_pass, seeds = _e68_training_points(document, PANTRY_DOMAIN)
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.35, 1.72), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        _plot_e68_training_axis(axis, points, metric, frozen_pass, seeds)
        axis.set_title(E68_TRAIN_METRICS[metric]["title"], fontsize=7.1)
        axis.set_xlabel("training passes", fontsize=6.5)
        axis.tick_params(labelsize=6.1)
    fig.legend(
        handles=_e68_training_legend(), loc="upper center", ncol=4,
        frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84), w_pad=0.75)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        out.with_suffix(".png"), dpi=240,
        bbox_inches="tight", pad_inches=0.03,
    )
    plt.close(fig)
    return out


def render_e68_training_metric(document: dict, metric: str) -> Path:
    """Render a standalone E68-style 3x2 domain figure for one metric."""

    metadata = E68_TRAIN_METRICS[metric]
    out = OUT_TRAINING.with_name(
        f"modecollapse_training_{metadata['suffix']}"
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.35, 6.75), squeeze=False)
    for panel_index, (axis, domain) in enumerate(zip(axes.flat, DOMAINS)):
        # The sixth cell is hidden after the five domains are drawn.
        points, frozen_pass, seeds = _e68_training_points(document, domain)
        _plot_e68_training_axis(axis, points, metric, frozen_pass, seeds)
        axis.set_title(
            f"{chr(65 + panel_index)}   {domain}",
            loc="left",
            fontsize=9.0,
            fontweight="bold",
            pad=6,
        )
        axis.text(
            0.99,
            1.025,
            f"paired through pass {frozen_pass}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.4,
            color="#666666",
        )
        axis.set_xlabel("training passes", fontsize=7.2)

    axes.flat[-1].set_axis_off()
    fig.suptitle(
        metadata["label"],
        fontsize=11.0,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles=_e68_training_legend(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=7.2,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.88,
        bottom=0.17,
        hspace=0.44,
        wspace=0.22,
    )
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(
        out.with_suffix(".png"),
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)
    return out


def main() -> None:
    render_e68_training()
    modes = render_modes_by_epoch()
    compact = render_e68_training_compact()
    document: dict = {}
    pantry_strip = render_pantry_training_strip(document)
    standalone = [
        render_e68_training_metric(document, metric)
        for metric in E68_TRAIN_METRICS
    ]
    outputs = [OUT_TRAINING, modes, compact, pantry_strip, *standalone]
    print("wrote " + ", ".join(f"{output}.{{pdf,png}}" for output in outputs))


if __name__ == "__main__":
    main()
