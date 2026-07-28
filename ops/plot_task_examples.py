#!/usr/bin/env python3
"""Render the paper's first-page motivating-example figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "task_examples.pdf"

INK = "#17324D"
MUTED = "#64798C"
LINE = "#C9D7E2"
PANEL = "#F7FAFC"
WHITE = "#FFFFFF"
PURPLE = "#6C5CE7"
ORANGE = "#E76F51"
BLUE = "#3A7CA5"
GREEN = "#2A9D8F"
COLORS = {1: ORANGE, 2: BLUE, 3: GREEN}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


def box(ax, x, y, width, height, *, face=WHITE, edge=LINE, radius=0.018, lw=1.1):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.007,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, value, *, size=12, weight="normal", color=INK, **kwargs):
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


def color_node(ax, x, y, value, *, radius=0.017, label_size=8.5):
    ax.add_patch(
        Circle(
            (x, y),
            radius,
            transform=ax.transAxes,
            facecolor=COLORS[value],
            edgecolor=WHITE,
            linewidth=1.3,
            zorder=3,
        )
    )
    label(ax, x, y, str(value), size=label_size, weight="bold", color=WHITE,
          ha="center", va="center", zorder=4)


def unknown_node(ax, x, y, *, radius=0.022):
    ax.add_patch(
        Circle(
            (x, y),
            radius,
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=MUTED,
            linewidth=1.5,
            zorder=3,
        )
    )
    label(ax, x, y, "?", size=9.5, weight="bold", color=MUTED,
          ha="center", va="center", zorder=4)


def edge(ax, left, right, y, *, lw=1.5):
    ax.plot(
        [left, right],
        [y, y],
        transform=ax.transAxes,
        color="#90A5B7",
        linewidth=lw,
        solid_capstyle="round",
        zorder=1,
    )


def filled_path(ax, center_x, center_y, values):
    xs = [center_x - 0.045, center_x, center_x + 0.045]
    edge(ax, xs[0], xs[1], center_y, lw=1.5)
    edge(ax, xs[1], xs[2], center_y, lw=1.5)
    for x, value in zip(xs, values):
        color_node(ax, x, center_y, value, radius=0.026, label_size=10.5)


def compact_path(ax, center_x, center_y, values):
    xs = [center_x - 0.025, center_x, center_x + 0.025]
    edge(ax, xs[0], xs[1], center_y, lw=1.0)
    edge(ax, xs[1], xs[2], center_y, lw=1.0)
    for x, value in zip(xs, values):
        color_node(ax, x, center_y, value, radius=0.014, label_size=6.4)


def mode_bar(ax, x, y, width, value, color, mode):
    label(ax, x, y, mode, size=7.5, weight="bold", color=MUTED,
          ha="left", va="center")
    box(ax, x + 0.037, y - 0.023, width, 0.046, face="#E7EEF3",
        edge="#E7EEF3", radius=0.010, lw=0.0)
    box(ax, x + 0.037, y - 0.023, max(0.012, width * value), 0.046,
        face=color, edge=color, radius=0.010, lw=0.0)


def main():
    fig = plt.figure(figsize=(7.5, 2.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    box(ax, 0.010, 0.060, 0.385, 0.885, face=PANEL, radius=0.025, lw=1.2)
    box(ax, 0.415, 0.060, 0.270, 0.885, face=PANEL, radius=0.025, lw=1.2)
    box(ax, 0.705, 0.060, 0.285, 0.885, face="#F2FAF7", radius=0.025, lw=1.2)

    # A: two multi-answer domains plus the single-answer MATH transfer test.
    label(ax, 0.035, 0.840, "A", size=12.0, weight="bold", color=PURPLE,
          ha="left", va="center")
    label(ax, 0.069, 0.840, "Reasoning domains", size=14.0, weight="bold",
          ha="left", va="center")
    label(ax, 0.040, 0.690, "Countdown", size=9.0, weight="bold", color=MUTED,
          ha="left", va="center")
    label(ax, 0.175, 0.690, "2, 3, 4  →  target 10", size=9.0, weight="bold",
          ha="left", va="center")
    box(ax, 0.040, 0.535, 0.145, 0.105, face="#EEEAFE", edge="#D5CEFA",
        radius=0.010, lw=0.8)
    box(ax, 0.205, 0.535, 0.155, 0.105, face="#FFF0EA", edge="#F4C8B8",
        radius=0.010, lw=0.8)
    label(ax, 0.1125, 0.588, "2 × 3 + 4", size=10.3, weight="bold",
          ha="center", va="center")
    label(ax, 0.2825, 0.588, "4 × 3 − 2", size=10.3, weight="bold",
          ha="center", va="center")
    label(ax, 0.040, 0.425, "Graph coloring", size=9.0, weight="bold", color=MUTED,
          ha="left", va="center")
    label(ax, 0.205, 0.425, "four valid fills", size=9.0, weight="bold",
          ha="left", va="center")
    completions = [(1, 2, 1), (1, 2, 3), (3, 2, 1), (3, 2, 3)]
    for values, cx in zip(completions, [0.075, 0.165, 0.255, 0.345]):
        compact_path(ax, cx, 0.305, values)
    label(ax, 0.040, 0.145, "MATH", size=9.0, weight="bold", color=MUTED,
          ha="left", va="center")
    label(
        ax,
        0.118,
        0.145,
        "MATH12K-384 train  →  MATH-500 eval",
        size=7.7,
        weight="bold",
        ha="left",
        va="center",
    )
    label(
        ax,
        0.118,
        0.080,
        "single-answer semantic-entropy stress test",
        size=7.1,
        color=ORANGE,
        ha="left",
        va="center",
    )

    # B: verifier reward does not distinguish broad from concentrated success.
    label(ax, 0.440, 0.840, "B", size=12.0, weight="bold", color=ORANGE,
          ha="left", va="center")
    label(ax, 0.474, 0.840, "Reward only", size=14.0, weight="bold",
          ha="left", va="center")
    label(ax, 0.440, 0.695, "every valid mode earns 1", size=8.7, color=MUTED,
          ha="left", va="center")
    for mode, y, value, color in zip(
        ["M1", "M2", "M3", "M4"],
        [0.565, 0.445, 0.325, 0.205],
        [1.00, 0.16, 0.10, 0.07],
        [PURPLE, ORANGE, BLUE, GREEN],
    ):
        mode_bar(ax, 0.442, y, 0.195, value, color, mode)
    label(ax, 0.550, 0.105, "valid but narrow", size=9.5, weight="bold",
          color=ORANGE, ha="center", va="center")

    # C: the free-form conditional-token regularizer aims to retain alternatives.
    label(ax, 0.730, 0.840, "C", size=12.0, weight="bold", color=GREEN,
          ha="left", va="center")
    label(ax, 0.764, 0.840, "Token-policy MaxEnt", size=13.2, weight="bold",
          ha="left", va="center")
    label(ax, 0.730, 0.695, "reward + conditional token entropy", size=8.0, color=MUTED,
          ha="left", va="center")
    for mode, y, value, color in zip(
        ["M1", "M2", "M3", "M4"],
        [0.565, 0.445, 0.325, 0.205],
        [0.88, 0.76, 0.82, 0.72],
        [PURPLE, ORANGE, BLUE, GREEN],
    ):
        mode_bar(ax, 0.732, y, 0.205, value, color, mode)
    label(ax, 0.848, 0.105, "broader • free-form", size=9.5, weight="bold",
          color=GREEN, ha="center", va="center")

    label(ax, 0.985, 0.965, "schematic", size=7.0, color=MUTED,
          ha="right", va="top")
    assert 2 * 3 + 4 == 10 and 4 * 3 - 2 == 10
    assert all(left != middle and middle != right for left, middle, right in completions)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {OUT} (+ .png preview)")


if __name__ == "__main__":
    main()
