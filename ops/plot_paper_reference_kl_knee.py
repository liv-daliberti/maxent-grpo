#!/usr/bin/env python3
"""Reference-KL coefficient sweeps and categorical reference summaries.

Each domain panel shows seed-mean terminal PCMD and pass@8 on the same
[0, 1] scale. Horizontal rules show initial PCMD, when reportable, and Re:Dr
PCMD. The vertical rule substitutes mean initial single-draw correctness
into the categorical beta-star formula; it is neither a fitted knee nor a
prediction for pass@8. Green and red identify the two sides of this plug-in
threshold. Reference PCMD is not an upper bound on neural checkpoints.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

SOURCE = ROOT / "paper/results/reference_kl_comparison.json"
OUT = ROOT / "paper/figures/reference_kl_knee"

DOMAINS = (
    ("graph_coloring", "Graph"),
    ("countdown", "Countdown"),
    ("python_factors", "Python"),
    ("mathir", "MathIR"),
    ("pantry_plan", "Pantry"),
)
# Shading encodes only the categorical plug-in stationary calculation.
# Domain-average initial correctness is not a promptwise prediction, and
# pass@8 is a different estimand from stationary single-draw correctness.
SAFE = "#E7F5EA"
COSTLY = "#FBEAE7"
BREADTH = style.METRIC        # the breadth series
CORRECT = style.CONTROL       # the correctness series
CEILING = style.MUTED         # initial PCMD, not a neural upper bound
KNEE = style.ADD_ON           # categorical plug-in threshold
REPLAY = style.METHOD         # Re:Dr's breadth, for the crossing


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    style.apply_rcparams()
    # Authored at half the shared canvas and included at half \linewidth, so
    # the plate keeps the common scale factor and its type matches every other
    # figure. Five domains in a 2x3 grid leave one cell, which the legend takes
    # rather than a sixth strip of whitespace below the panels.
    figure, grid = plt.subplots(
        3, 2,
        figsize=(style.WIDTH / 2, style.panel_height(3, per_row=0.86)),
        sharey=True, sharex=True,
    )
    axes = [grid[0][0], grid[0][1], grid[1][0], grid[1][1], grid[2][0]]
    spare = grid[2][1]
    for axis, (key, label) in zip(axes, DOMAINS):
        row = payload["domains"].get(key, {})
        kl = row.get("kl", {})
        betas = sorted(float(b) for b in kl)
        style.style_axis(axis, title=label)

        # initial-policy PCMD and the Re:Dr comparison
        frozen = row.get("frozen_pmd")
        if frozen is not None:
            axis.axhline(frozen, color=CEILING, lw=0.9, ls=(0, (1, 1.6)), zorder=1)
        replay = (row.get("replay") or {}).get("pmd")
        if replay is not None:
            axis.axhline(replay, color=REPLAY, lw=1.0, ls=(0, (5, 2)), zorder=1)
        knee = row.get("beta_star")
        # The threshold may fall beyond the measured coefficients. The axis
        # displays the plug-in calculation without extrapolating the data.
        if knee is not None:
            axis.axvspan(6e-4, knee, facecolor=SAFE, zorder=0)
            axis.axvspan(knee, 1.6, facecolor=COSTLY, zorder=0)
            axis.axvline(knee, color=KNEE, lw=1.0, ls=(0, (3, 1.5)), zorder=1)

        for series, colour, marker in ((("pmd"), BREADTH, "o"),
                                       (("pass8"), CORRECT, "s")):
            xs = [b for b in betas if kl[f"{b}"].get(series) is not None]
            ys = [kl[f"{b}"][series] for b in xs]
            axis.plot(xs, ys, color=colour, lw=style.MEAN_LW, marker=marker,
                      markersize=3.4, markerfacecolor=style.WHITE,
                      markeredgewidth=1.0, zorder=3, clip_on=False)

        axis.set_xscale("log")
        axis.set_xlim(6e-4, 1.6)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
        axis.set_xticklabels(["$.001$", "$.01$", "$.1$", "$1$"])
    # The left bottom panel labels the shared coefficient axis. A second
    # label above the spare-cell legend would overlap its first entry.
    for axis in (axes[4],):
        axis.set_xlabel(r"$\beta$", fontsize=style.FONT, color=style.INK,
                        labelpad=1)
    for axis in (axes[0], axes[2], axes[4]):
        axis.set_ylabel("terminal value", fontsize=style.FONT, color=style.INK)

    handles = [
        Line2D([], [], color=BREADTH, lw=style.MEAN_LW, marker="o", markersize=3.4,
               markerfacecolor=style.WHITE, label="reference-KL PCMD"),
        Line2D([], [], color=CORRECT, lw=style.MEAN_LW, marker="s", markersize=3.4,
               markerfacecolor=style.WHITE, label="reference-KL pass@8"),
        Line2D([], [], color=CEILING, lw=0.9, ls=(0, (1, 1.6)),
               label="initial-policy PCMD"),
        Line2D([], [], color=REPLAY, lw=1.0, ls=(0, (5, 2)), label="Re:Dr PCMD"),
        Line2D([], [], color=KNEE, lw=1.0, ls=(0, (3, 1.5)),
               label=r"categorical $\beta^\star$"),
    ]
    # The spare cell keeps the grid's x axis so the column reads as one ruler,
    # and carries the key where a sixth panel would have been.
    spare.set_facecolor(style.WHITE)
    for side in spare.spines:
        spare.spines[side].set_visible(False)
    axes[3].tick_params(labelbottom=True)
    spare.tick_params(which="both", left=False, bottom=False, labelleft=False,
                      labelbottom=False)
    spare.grid(False)
    spare.set_xlabel("")
    spare.legend(handles=handles, labels=[h.get_label() for h in handles],
                 loc="center", frameon=False, fontsize=style.FONT - 0.6,
                 handlelength=2.0, labelspacing=0.45, borderpad=0.2)
    figure.subplots_adjust(left=0.115, right=0.99, top=0.94, bottom=0.075,
                           wspace=0.14, hspace=0.30)
    style.save(figure, OUT)
    print(f"wrote {OUT.relative_to(ROOT)}.pdf/.png")
    for key, label in DOMAINS:
        row = payload["domains"].get(key, {})
        print(f"  {label:<10} frozen={row.get('frozen_pmd')} "
              f"beta*={row.get('beta_star')} betas={sorted(row.get('kl', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
