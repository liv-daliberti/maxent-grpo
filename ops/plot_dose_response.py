#!/usr/bin/env python3
"""Plot the aggregation-temperature response by environment and model scale.

Countdown and graph coloring occupy separate side-by-side columns, with 0.5B
and 3B in aligned rows. pass@8 and coverage@8 share an axis because both are
treatment effects in percentage points.

The inputs are the single-domain regressions written by
ops/analyze_countdown_comparative.py. Run with system Python (matplotlib):

  python ops/plot_dose_response.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FINE = ROOT / "var/artifacts/cdfine_regression.json"
COUNTDOWN_05B = ROOT / "var/artifacts/cdcomp4_countdown_only_regression.json"
COUNTDOWN_3B = ROOT / "var/artifacts/cd3b_regression.json"
GRAPH_05B = ROOT / "var/artifacts/gccomp1_graph_only_regression.json"
GRAPH_3B = ROOT / "var/artifacts/gc3b_regression.json"
OUT = ROOT / "paper/figures/dose_response"

TAU = {
    "xdr_tau0p0001": 0.0001,
    "xdr_tau0p001": 0.001,
    "xdr_tau0p01": 0.01,
    "xdr_tau0p02": 0.02,
    "xdr_tau0p03": 0.03,
    "xdr_tau0p04": 0.04,
    "xdr_tau0p05": 0.05,
    "xdr_tau0p1": 0.1,
    "xdr_tau0p25": 0.25,
    "xdr_tau0p5": 0.5,
    "xdr_tau1": 1.0,
    "xdr_tau2": 2.0,
}

OUTCOMES = [
    ("pass@8", "pass@8", "#31688e", "o"),
    ("coverage@8", "coverage@8", "#b2632e", "^"),
]
INK = "#1a1a1a"

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.7,
    }
)


def rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)


def series(
    sources: list[tuple[list[dict], str]], outcome: str
) -> list[tuple[float, float, float, float, str]]:
    points = []
    for data, round_tag in sources:
        for row in data:
            if row["outcome"] != outcome or row["treatment"] not in TAU:
                continue
            points.append(
                (
                    TAU[row["treatment"]],
                    100 * row["gamma"],
                    100 * row["ci_low"],
                    100 * row["ci_high"],
                    round_tag,
                )
            )
    return sorted(points)


def special_row(data: list[dict], outcome: str, treatment: str):
    for row in data:
        if row["outcome"] == outcome and row["treatment"] == treatment:
            return (
                100 * row["gamma"],
                100 * row["ci_low"],
                100 * row["ci_high"],
            )
    return None


def draw_observed_panel(
    ax,
    sources: list[tuple[list[dict], str]],
    *,
    boundary_data: list[dict] | None = None,
    adaptive_data: list[dict] | None = None,
) -> None:
    ax.axhline(0, color="#999999", lw=0.7, ls=(0, (3, 3)), zorder=1)
    for outcome, _label, hue, marker in OUTCOMES:
        points = series(sources, outcome)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        low = [point[1] - point[2] for point in points]
        high = [point[3] - point[1] for point in points]
        ax.errorbar(
            xs,
            ys,
            yerr=[low, high],
            color=hue,
            ecolor=hue,
            lw=1.25,
            elinewidth=0.8,
            capsize=1.8,
            zorder=3,
        )
        for point in points:
            filled = point[4] == "later"
            ax.plot(
                point[0],
                point[1],
                marker=marker,
                ms=4.2,
                mec=hue,
                mfc=hue if filled else "white",
                mew=0.9,
                zorder=4,
            )

        boundary = (
            special_row(boundary_data, outcome, "xdr_tau0")
            if boundary_data is not None
            else None
        )
        if boundary:
            value, ci_low, ci_high = boundary
            ax.errorbar(
                [2.6e-5],
                [value],
                yerr=[[value - ci_low], [ci_high - value]],
                fmt=marker,
                ms=4.2,
                color=hue,
                mfc="white",
                ecolor=hue,
                elinewidth=0.8,
                capsize=1.8,
                zorder=4,
            )

        adaptive = (
            special_row(adaptive_data, outcome, "xdr_adapt")
            if adaptive_data is not None
            else None
        )
        if adaptive:
            value, ci_low, ci_high = adaptive
            ax.errorbar(
                [3.6],
                [value],
                yerr=[[value - ci_low], [ci_high - value]],
                fmt="D",
                ms=4.0,
                color=hue,
                ecolor=hue,
                elinewidth=0.8,
                capsize=1.8,
                zorder=4,
            )

    if boundary_data is not None:
        ax.annotate(
            r"$\tau_{\rm agg}{=}0$",
            (2.6e-5, ax.get_ylim()[1]),
            xytext=(1, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=6.8,
            color="#555555",
        )
    if adaptive_data is not None:
        ax.annotate(
            r"adaptive $\tau_{\rm agg}$",
            (3.6, ax.get_ylim()[1]),
            xytext=(-1, -3),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=6.8,
            color="#555555",
        )

    ax.set_xscale("log")
    ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0])
    ax.set_xticklabels(
        [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"]
    )
    ax.set_xlim(1.5e-5, 5.6)
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.7)


def main() -> None:
    fine = rows(FINE)
    countdown_05b = rows(COUNTDOWN_05B)
    countdown_3b = rows(COUNTDOWN_3B)
    graph_05b = rows(GRAPH_05B)
    graph_3b = rows(GRAPH_3B)

    panels = [
        (
            "Countdown",
            [
                ([(fine, "later"), (countdown_05b, "earlier")], fine, fine),
                ([(countdown_3b, "later")], None, None),
            ],
        ),
        (
            "Graph coloring",
            [
                ([(graph_05b, "earlier")], None, None),
                ([(graph_3b, "later")], None, None),
            ],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.1), sharey="row", squeeze=False)
    for col, (environment, observed) in enumerate(panels):
        for row, (sources, boundary_data, adaptive_data) in enumerate(observed):
            ax = axes[row][col]
            draw_observed_panel(
                ax,
                sources,
                boundary_data=boundary_data,
                adaptive_data=adaptive_data,
            )
            scale = "0.5B" if row == 0 else "3B"
            title = f"{environment}\n{scale}" if row == 0 else scale
            ax.set_title(title, fontsize=8.3, pad=4)
            if row == 0:
                ax.tick_params(axis="x", which="both", labelbottom=False)
            else:
                ax.set_xlabel(r"aggregation temperature $\tau_{\rm agg}$")
            if col == 0:
                ax.set_ylabel(r"$\Delta$ from Dr.GRPO (pts)")

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="-",
            color=OUTCOMES[0][2],
            ms=4.2,
            label="pass@8",
        ),
        plt.Line2D(
            [],
            [],
            marker="^",
            ls="-",
            color=OUTCOMES[1][2],
            ms=4.2,
            label="coverage@8",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color="#555555",
            mfc="#555555",
            ms=4.2,
            label="later round / 3B",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color="#555555",
            mfc="white",
            ms=4.2,
            label="original 0.5B grid",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.2,
        handlelength=1.5,
    )
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.965))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
