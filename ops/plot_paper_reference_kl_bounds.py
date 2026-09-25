#!/usr/bin/env python3
"""App. Q: attained breadth against the bound each method's own result predicts.

One panel. The comparison between an anchor and a memory is not between two
numbers but between two different bounds, and this is the plate that says so.

The x position of a point is what that method can reach in that domain.
An anchor is bounded by the frozen policy's own success-conditional breadth
(Proposition "The retained conditional is the reference's"); a memory is
bounded by uniform rehearsal over the keys its bank actually holds,
1 - 1/E|B| (Corollary "Conditional full-coverage categorical limit"), which is
measured per arm because the fresh objective decides what discovery banks. The
y position is what it actually reached. The diagonal is attainment.

Reading the plate: points near the diagonal took what their bound allowed, and
the horizontal spread within a domain is the whole comparison --- whichever
method sits further right had more available to it there, before either was
run. A point below the diagonal did not collect what was on offer, and in
every such case the shortfall is a bank that never filled.

The anchor is shown at the largest swept coefficient below that domain's
beta* = c_G/(-logit mu(C)), the window inside which the correctness identity
of the same proposition puts stationary correctness above one half. Reading it
at a larger coefficient would credit it with breadth that the companion figure
shows is paid for in correctness.
"""
from __future__ import annotations

import json
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
OUT = ROOT / "paper/figures/reference_kl_bounds"
SHORT = {"graph_coloring": "Graph", "countdown": "Countd.",
         "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "Pantry"}
# Validated together under --pairs all; see paper_style.FRONTIER_FIVE.
ANCHOR, REDR, REMAX = style.CONTROL, style.METHOD, style.ABLATION


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    style.apply_rcparams()
    figure, axis = plt.subplots(figsize=(style.WIDTH * 0.46, style.WIDTH * 0.40))
    style.style_axis(axis)
    axis.plot([0, 1], [0, 1], color=style.MUTED, lw=0.9, ls=(0, (4, 2)), zorder=1)
    axis.text(0.62, 0.66, "attains its bound", fontsize=style.FONT - 0.6,
              color=style.MUTED, rotation=38, ha="center", va="center", zorder=1)

    series = {ANCHOR: [], REDR: [], REMAX: []}
    for key, row in payload["domains"].items():
        # the anchor, read inside its own coefficient window
        bound, knee = row.get("frozen_pmd"), row.get("beta_star")
        if bound is not None and knee is not None:
            inside = [float(b) for b in row.get("kl", {}) if float(b) < knee]
            if inside:
                best = max(inside)
                series[ANCHOR].append((bound, row["kl"][f"{best}"]["pmd"], SHORT[key]))
        for colour, arm, ceil in ((REDR, "replay", "replay_ceiling"),
                                  (REMAX, "remax", "remax_ceiling")):
            value, limit = row.get(arm), row.get(ceil)
            if value and limit:
                series[colour].append((limit, value["pmd"], SHORT[key]))

    marks = {ANCHOR: "o", REDR: "s", REMAX: "^"}
    for colour, points in series.items():
        axis.scatter([p[0] for p in points], [p[1] for p in points],
                     s=26, facecolor=style.WHITE, edgecolor=colour,
                     linewidths=1.3, marker=marks[colour], zorder=3, clip_on=False)
    # label only the anchor points, so each domain is named once
    for bound, value, name in series[ANCHOR]:
        axis.annotate(name, (bound, value), textcoords="offset points",
                      xytext=(4, -7), fontsize=style.FONT - 1, color=style.INK,
                      zorder=4)

    axis.set_xlim(-0.02, 1.0)
    axis.set_ylim(-0.02, 1.0)
    axis.set_xlabel("bound its own result predicts", fontsize=style.FONT, color=style.INK)
    axis.set_ylabel("attained terminal PCMD", fontsize=style.FONT, color=style.INK)

    handles = [
        Line2D([], [], color=ANCHOR, marker="o", ls="none", markersize=5,
               markerfacecolor=style.WHITE, markeredgewidth=1.3,
               label=r"reference KL, $\beta<\beta^\ast$"),
        Line2D([], [], color=REDR, marker="s", ls="none", markersize=5,
               markerfacecolor=style.WHITE, markeredgewidth=1.3, label="Re:Dr"),
        Line2D([], [], color=REMAX, marker="^", ls="none", markersize=5,
               markerfacecolor=style.WHITE, markeredgewidth=1.3, label="Re:Max"),
    ]
    style.bottom_legend(figure, handles, [h.get_label() for h in handles],
                        y=-0.02, ncol=3)
    figure.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.22)
    style.save(figure, OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")
    for colour, name in ((ANCHOR, "anchor"), (REDR, "Re:Dr"), (REMAX, "Re:Max")):
        for bound, value, dom in series[colour]:
            print(f"  {name:<7}{dom:<9} bound {bound:.3f} -> attained {value:.3f}"
                  f"  ({value/bound*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
