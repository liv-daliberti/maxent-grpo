#!/usr/bin/env python3
"""Dose-response figure: xDr.GRPO effect vs temperature on 0.5B Countdown.

Reads the regression JSONs written by ops/analyze_countdown_comparative.py for
the fine-tau round (cdfine: tau 0.01-0.04 + mode-adaptive) and the original
grid (cdcomp4_stable, countdown-only: tau 0.05-2), each estimated against its
own matched Dr.GRPO baseline, and plots the treatment effect with 95% CIs for
pass@8 and coverage@8 on a log-tau axis. Run with system python3 (matplotlib).

  python3 ops/plot_dose_response.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FINE = ROOT / "var/artifacts/cdfine_regression.json"
GRID = ROOT / "var/artifacts/cdcomp4_countdown_only_regression.json"
OUT = ROOT / "paper/figures/dose_response"

TAU = {
    "xdr_tau0p0001": 0.0001,
    "xdr_tau0p001": 0.001,
    "xdr_tau0p01": 0.01, "xdr_tau0p02": 0.02, "xdr_tau0p03": 0.03,
    "xdr_tau0p04": 0.04, "xdr_tau0p05": 0.05, "xdr_tau0p1": 0.1,
    "xdr_tau0p25": 0.25, "xdr_tau0p5": 0.5, "xdr_tau1": 1.0, "xdr_tau2": 2.0,
}

HUE = "#31688e"      # single treatment hue (one entity across doses)
ADAPT = "#b2632e"    # mode-adaptive arm: different entity, second fixed hue
INK = "#1a1a1a"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.7,
})


def rows(path):
    return json.load(open(path))


def series(data, outcome, round_tag):
    out = []
    for r in data:
        if r["outcome"] != outcome or r["treatment"] not in TAU:
            continue
        out.append((TAU[r["treatment"]], 100 * r["gamma"],
                    100 * r["ci_low"], 100 * r["ci_high"], round_tag))
    return out


def adapt_row(data, outcome):
    for r in data:
        if r["outcome"] == outcome and r["treatment"] == "xdr_adapt":
            return (100 * r["gamma"], 100 * r["ci_low"], 100 * r["ci_high"])
    return None


def main():
    fine, grid = rows(FINE), rows(GRID)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharex=True)
    for ax, outcome, label in (
        (axes[0], "pass@8", r"$\Delta$ pass@8 (points)"),
        (axes[1], "coverage@8", r"$\Delta$ coverage@8 (points)"),
    ):
        pts = sorted(series(fine, outcome, "fine") + series(grid, outcome, "grid"))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo = [p[1] - p[2] for p in pts]
        hi = [p[3] - p[1] for p in pts]
        ax.axhline(0, color="#999999", lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.errorbar(xs, ys, yerr=[lo, hi], color=HUE, ecolor=HUE,
                    lw=1.4, elinewidth=0.9, capsize=2.0, zorder=3)
        for p in pts:  # open markers for the original grid, filled for fine round
            filled = p[4] == "fine"
            ax.plot(p[0], p[1], marker="o", ms=4.5, mec=HUE,
                    mfc=HUE if filled else "white", mew=1.0, zorder=4)
        z = next(
            (r for r in fine
             if r["outcome"] == outcome and r["treatment"] == "xdr_tau0"),
            None,
        )
        if z:
            zg, zl, zh = 100 * z["gamma"], 100 * z["ci_low"], 100 * z["ci_high"]
            ax.errorbar([2.6e-5], [zg], yerr=[[zg - zl], [zh - zg]], fmt="s",
                        ms=4.5, color=HUE, mfc="white", ecolor=HUE,
                        elinewidth=0.9, capsize=2.0, zorder=4)
            ax.annotate(r"$\tau{=}0$", (2.6e-5, zg), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=HUE,
                        annotation_clip=False)
        a = adapt_row(fine, outcome)
        if a:
            ax.errorbar([3.6], [a[0]], yerr=[[a[0] - a[1]], [a[2] - a[0]]],
                        fmt="D", ms=4.5, color=ADAPT, ecolor=ADAPT,
                        elinewidth=0.9, capsize=2.0, zorder=4)
            ax.annotate("adaptive", (3.6, a[0]), textcoords="offset points",
                        xytext=(-1, 7), ha="right", fontsize=7, color=ADAPT,
                        annotation_clip=False)
        ax.set_xscale("log")
        ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0])
        ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"])
        ax.set_xlim(1.5e-5, 5.6)
        ax.set_xlabel(r"temperature $\tau$")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.7)
    handles = [
        plt.Line2D([], [], marker="o", ls="-", color=HUE, mfc=HUE, ms=4.5,
                   lw=1.4, label=r"fine round ($\tau\leq 0.04$)"),
        plt.Line2D([], [], marker="o", ls="-", color=HUE, mfc="white", ms=4.5,
                   lw=1.4, label=r"original grid ($\tau\geq 0.05$)"),
        plt.Line2D([], [], marker="D", ls="", color=ADAPT, ms=4.5,
                   label="mode-adaptive"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="upper right",
                   handlelength=1.6, borderaxespad=0.1)
    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
