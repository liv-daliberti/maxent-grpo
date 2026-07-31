#!/usr/bin/env python3
"""Render xGRPO as one clear discover-retain-rebalance loop."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/xdr_mechanism"

INK = "#19324A"
MUTED = "#607487"
GRID = "#D8E2EA"
PANEL = "#F6F9FB"
WHITE = "#FFFFFF"
PURPLE = "#6C5CE7"
ORANGE = "#D95F45"
BLUE = "#3A7CA5"
GREEN = "#2A9D8F"
RED = "#C76A3A"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


def box(ax, x, y, w, h, *, face=WHITE, edge=GRID, radius=0.012, lw=0.9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, **kwargs):
    defaults = {
        "transform": ax.transAxes,
        "fontsize": 7.2,
        "color": INK,
        "ha": "center",
        "va": "center",
    }
    defaults.update(kwargs)
    return ax.text(x, y, value, **defaults)


def arrow(ax, start, end, *, color=MUTED, lw=1.0, scale=8, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=scale,
        color=color,
        linewidth=lw,
        shrinkA=1,
        shrinkB=1,
        connectionstyle=connectionstyle,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def card(ax, x, letter, step, title, accent):
    box(ax, x, 0.305, 0.302, 0.630, face=PANEL, edge=GRID, radius=0.018, lw=1.0)
    text(ax, x + 0.020, 0.886, letter, fontsize=10.5, fontweight="bold", color=accent, ha="left")
    text(ax, x + 0.052, 0.887, step, fontsize=6.0, fontweight="bold", color=accent, ha="left")
    text(ax, x + 0.052, 0.846, title, fontsize=9.4, fontweight="bold", ha="left")


def draw_discover(ax, x):
    card(ax, x, "A", "STEP 1", "Discover verified modes", PURPLE)
    text(ax, x + 0.151, 0.790, "same task validator supplies reward + identity", fontsize=6.3, color=MUTED)
    rows = (
        ("y₁", "c₁ · seen", BLUE, WHITE),
        ("y₂", "c₂ · NEW +ν", ORANGE, "#FFF4EE"),
        ("y₃", "⊥ · invalid (0)", RED, "#FFF5F1"),
    )
    for index, (rollout, result, color, face) in enumerate(rows):
        yy = 0.690 - index * 0.120
        box(ax, x + 0.024, yy - 0.032, 0.050, 0.064, edge=GRID, radius=0.008)
        text(ax, x + 0.049, yy, rollout, fontsize=7.1, fontweight="bold")
        arrow(ax, (x + 0.078, yy), (x + 0.105, yy), color=PURPLE, scale=6)
        box(ax, x + 0.110, yy - 0.032, 0.044, 0.064, edge=PURPLE, radius=0.008)
        text(ax, x + 0.132, yy, "V", fontsize=8.0, fontweight="bold", color=PURPLE)
        arrow(ax, (x + 0.158, yy), (x + 0.183, yy), color=color, scale=6)
        box(ax, x + 0.188, yy - 0.032, 0.090, 0.064, face=face, edge=color, radius=0.008)
        text(ax, x + 0.233, yy, result, fontsize=6.3, fontweight="bold", color=color)
    text(ax, x + 0.151, 0.355, "rare / first-seen bonus only after V accepts", fontsize=6.5, fontweight="bold", color=PURPLE)


def draw_retain(ax, x):
    card(ax, x, "B", "STEP 2", "Retain every discovery", BLUE)
    text(ax, x + 0.065, 0.748, "pre-group bank", fontsize=5.8, color=MUTED)
    box(ax, x + 0.026, 0.650, 0.078, 0.075, face=WHITE, edge=BLUE, radius=0.010)
    text(ax, x + 0.065, 0.688, "{c₁}", fontsize=8.0, fontweight="bold", color=BLUE)
    arrow(ax, (x + 0.109, 0.688), (x + 0.187, 0.688), color=ORANGE, scale=8)
    text(ax, x + 0.148, 0.718, "+ c₂", fontsize=6.0, fontweight="bold", color=ORANGE)
    text(ax, x + 0.234, 0.748, "persistent bank", fontsize=5.8, color=MUTED)
    box(ax, x + 0.192, 0.640, 0.084, 0.095, face="#EEF7F4", edge=GREEN, radius=0.010)
    text(ax, x + 0.234, 0.688, "{c₁,c₂}", fontsize=8.0, fontweight="bold", color=GREEN)

    arrow(ax, (x + 0.234, 0.642), (x + 0.234, 0.565), color=BLUE, scale=7)
    box(ax, x + 0.050, 0.465, 0.202, 0.092, face=WHITE, edge=BLUE, radius=0.012, lw=1.0)
    text(ax, x + 0.151, 0.522, "GLOBAL ROUND-ROBIN REPLAY", fontsize=6.3, fontweight="bold", color=BLUE)
    text(ax, x + 0.151, 0.489, "one verified bank each optimizer update", fontsize=6.2, color=MUTED)
    text(ax, x + 0.151, 0.355, "the key acts again before its prompt returns", fontsize=6.5, fontweight="bold", color=BLUE)


def vertical_bar(ax, x, y, height, color, label):
    box(ax, x, y, 0.031, height, face=color, edge=color, radius=0.004, lw=0)
    text(ax, x + 0.0155, y - 0.024, label, fontsize=5.8, color=MUTED)


def draw_rebalance(ax, x):
    card(ax, x, "C", "STEP 3", "Preserve mass + rebalance", GREEN)
    text(ax, x + 0.080, 0.785, "before replay", fontsize=6.1, color=MUTED)
    vertical_bar(ax, x + 0.040, 0.555, 0.170, PURPLE, "c₁")
    vertical_bar(ax, x + 0.086, 0.555, 0.055, ORANGE, "c₂")
    arrow(ax, (x + 0.130, 0.640), (x + 0.178, 0.640), color=GREEN, scale=8)
    text(ax, x + 0.224, 0.785, "after replay", fontsize=6.1, color=MUTED)
    vertical_bar(ax, x + 0.194, 0.555, 0.120, PURPLE, "c₁")
    vertical_bar(ax, x + 0.240, 0.555, 0.120, ORANGE, "c₂")

    box(ax, x + 0.030, 0.445, 0.115, 0.065, face=WHITE, edge=BLUE, radius=0.009)
    text(ax, x + 0.0875, 0.478, "verified mass", fontsize=6.2, fontweight="bold", color=BLUE)
    box(ax, x + 0.158, 0.445, 0.115, 0.065, face=WHITE, edge=GREEN, radius=0.009)
    text(ax, x + 0.2155, 0.478, "known-key KL", fontsize=6.2, fontweight="bold", color=GREEN)
    text(ax, x + 0.151, 0.355, "keep P⁺ high without a one-key monopoly", fontsize=6.5, fontweight="bold", color=GREEN)


def controller(ax):
    box(ax, 0.018, 0.040, 0.964, 0.205, face=WHITE, edge=GRID, radius=0.015, lw=1.0)
    text(ax, 0.035, 0.207, "SELF-REFERENCED CONTROL", fontsize=6.2, fontweight="bold", color=INK, ha="left")
    sensors = (
        (0.035, "mode entropy ↓", "β rare/new ↑", PURPLE),
        (0.220, "exemplar surprisal ↑", "μ mass ↑", BLUE),
        (0.405, "bank entropy ↓", "α balance ↑", GREEN),
    )
    for x, sensor, pressure, color in sensors:
        box(ax, x, 0.082, 0.168, 0.082, face=PANEL, edge=color, radius=0.009, lw=0.8)
        text(ax, x + 0.084, 0.137, sensor, fontsize=6.1, color=MUTED)
        text(ax, x + 0.084, 0.103, pressure, fontsize=6.3, fontweight="bold", color=color)

    ax.plot([0.595, 0.595], [0.065, 0.220], transform=ax.transAxes, color=GRID, lw=0.9)
    text(ax, 0.615, 0.207, "SINGLETON ESCAPE", fontsize=6.2, fontweight="bold", color=ORANGE, ha="left")
    box(ax, 0.620, 0.100, 0.070, 0.060, edge=PURPLE, radius=0.008)
    text(ax, 0.655, 0.130, "{c₁}", fontsize=7.0, fontweight="bold", color=PURPLE)
    arrow(ax, (0.695, 0.130), (0.750, 0.130), color=ORANGE, scale=7)
    text(ax, 0.722, 0.158, "verify", fontsize=5.8, color=ORANGE)
    box(ax, 0.755, 0.100, 0.090, 0.060, edge=GREEN, radius=0.008)
    text(ax, 0.800, 0.130, "+ c₂ to bank", fontsize=6.2, fontweight="bold", color=GREEN)
    text(ax, 0.915, 0.145, "replay support only", fontsize=6.1, fontweight="bold", color=INK)
    text(ax, 0.915, 0.108, "never enters PPO", fontsize=6.1, fontweight="bold", color=RED)


def render() -> None:
    fig = plt.figure(figsize=(7.35, 2.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    xs = (0.018, 0.349, 0.680)
    draw_discover(ax, xs[0])
    draw_retain(ax, xs[1])
    draw_rebalance(ax, xs[2])
    arrow(ax, (0.324, 0.620), (0.344, 0.620), color=PURPLE, lw=1.2, scale=9)
    arrow(ax, (0.655, 0.620), (0.675, 0.620), color=BLUE, lw=1.2, scale=9)
    controller(ax)
    text(ax, 0.982, 0.975, "executed correct modes only", fontsize=6.1, color=MUTED, ha="right", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    render()
