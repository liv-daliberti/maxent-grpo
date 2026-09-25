#!/usr/bin/env python3
"""App. Q: the anchor and the memory at a glance, on both axes at once.

Two panels, one row, the arms shared down the side. Left is breadth, right is
correctness, and the pair is the whole comparison: a coefficient large enough
to bring an anchor's breadth up to a memory's is a coefficient that has already
spent the correctness the memory keeps.

The rule on the left panel is the frozen policy's own \\pmd{}, averaged over
the same domains. Proposition "The retained conditional is the reference's"
makes it the anchor's bound, and the anchor tracks it: no coefficient here
clears it by more than twice the .023 cross-pool measurement gap of
amendment 1. It is drawn on the memory arms too, where it is not a bound --- a
bank is limited by what discovery puts in it, not by what the reference had,
and Graph is the domain that separates the two, with Re:Dr .232 above the rule
there against the anchor's .015.

Averages are over the domains that carry a frozen bound and the full
coefficient range, so every arm is read on the same set. Per-domain curves,
where the coefficient's cost is visible one domain at a time, are the
companion plate.
"""
from __future__ import annotations

import json
import statistics
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
OUT = ROOT / "paper/figures/reference_kl_summary"
NOFRESH, ANCHOR, MEMORY = style.CONTROL, style.COMPARATOR, style.METHOD


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    domains = {k: r for k, r in payload["domains"].items()
               if r.get("frozen_pmd") is not None and "0.1" in r.get("kl", {})}
    keys = sorted(domains)

    def mean(pick):
        values = [pick(domains[k]) for k in keys]
        return statistics.fmean(values) if all(v is not None for v in values) else None

    arms = [("Dr.GRPO", NOFRESH,
             lambda r: r["control"]["pmd"], lambda r: r["control"]["pass8"])]
    for beta in ("0.001", "0.01", "0.04", "0.1"):
        arms.append((rf"KL $\beta={beta.lstrip('0')}$", ANCHOR,
                     (lambda b: lambda r: r["kl"][b]["pmd"])(beta),
                     (lambda b: lambda r: r["kl"][b]["pass8"])(beta)))
    arms += [("Re:Dr", MEMORY, lambda r: r["replay"]["pmd"], lambda r: r["replay"]["pass8"]),
             ("Re:Max", MEMORY, lambda r: r["remax"]["pmd"], lambda r: r["remax"]["pass8"])]

    style.apply_rcparams()
    figure, axes = plt.subplots(
        1, 2, figsize=(style.WIDTH * 0.74, style.WIDTH * 0.30), sharey=True)
    positions = list(range(len(arms)))[::-1]
    frozen = mean(lambda r: r["frozen_pmd"])

    for axis, index, label in ((axes[0], 2, r"terminal \textsc{pcmd}"),
                               (axes[1], 3, "terminal pass@8")):
        style.style_axis(axis)
        if axis is axes[0]:
            axis.axvline(frozen, color=style.MUTED, lw=1.0, ls=(0, (4, 2)), zorder=1)
            axis.text(frozen, len(arms) - 0.35, " frozen PCMD",
                      fontsize=style.FONT - 1, color=style.MUTED, va="top", ha="left")
        for y, arm in zip(positions, arms):
            value = mean(arm[index])
            axis.plot([0, value], [y, y], color=arm[1], lw=0.9, alpha=0.45, zorder=2)
            axis.plot([value], [y], marker="o", markersize=5.2, color=arm[1],
                      markerfacecolor=style.WHITE, markeredgewidth=1.5, zorder=3)
            axis.annotate(f"{value:.2f}", (value, y), textcoords="offset points",
                          xytext=(7, -2.6), fontsize=style.FONT - 1, color=style.INK)
        axis.set_xlim(0, 1.0)
        axis.set_xlabel(label.replace(r"\textsc{pcmd}", "PCMD"),
                        fontsize=style.FONT, color=style.INK)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([a[0] for a in arms], fontsize=style.FONT)
    axes[0].set_ylim(-0.7, len(arms) - 0.3)

    handles = [Line2D([], [], color=c, marker="o", ls="none", markersize=5,
                      markerfacecolor=style.WHITE, markeredgewidth=1.5, label=t)
               for c, t in ((NOFRESH, "no retention"), (ANCHOR, "anchor"),
                            (MEMORY, "memory"))]
    style.bottom_legend(figure, handles, [h.get_label() for h in handles],
                        y=-0.04, ncol=3)
    figure.subplots_adjust(left=0.155, right=0.985, top=0.97, bottom=0.30, wspace=0.09)
    style.save(figure, OUT)
    print(f"wrote {OUT.relative_to(ROOT)}   domains={keys}")
    print(f"  frozen PCMD bound {frozen:.3f}")
    for arm in arms:
        print(f"  {arm[0]:<16} pcmd {mean(arm[2]):.3f}   pass@8 {mean(arm[3]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
