#!/usr/bin/env python3
"""App. A.3: the bank decomposition against measured prompts.

Equation (7) splits the uniform replay loss into a banked-mass term and a
within-bank balance term. The prose then makes a claim that is hard to see in
the algebra: at fixed mass and key count, expected ``distinct@K`` is largest
under a uniform allocation, while the probability of drawing *any* one of those
keys depends on the mass alone and does not move.

The bands are that closed form. The points are measured: per-prompt verified
keys at the terminal checkpoint of the Qwen2.5-0.5B replay arms, one panel per
benchmark domain, read from ``replay_bank_decomposition_measured.json`` and
bound to it by hash in the paper contract.

Panels are split by domain because a mode means something different in each --
a colouring, a computation route, a return vector -- so the number of keys a
prompt can occupy is a property of the domain, and pooling the five would hide
exactly the variation the bands are drawn over. Panel washes are the five tints
of the domain cards in Fig. 2.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

OUT = ROOT / "paper/figures/replay_bank_decomposition"
MEASURED = ROOT / "paper/results/replay_bank_decomposition_measured.json"
MACROS = ROOT / "paper/results/replay_bank_decomposition_macros.tex"

# The bank capacity the experiments actually run at (Table 5, M = 16), so the
# drawn ceiling 1 - 1/k is the one the implemented bank can reach.
K_BANK = 16
# The evaluation draw budget every headline number fixes.
K_DRAWS = 8
# Three banked-mass levels, low to high. The claim is about the shape at each
# fixed level, not about moving between them.
BANKED_MASS = (0.25, 0.50, 0.75)
STEPS = 401

# Domain order and tints copied from the cards in Fig. 2, so a panel here can be
# matched back to the domain's card by colour alone.
DOMAINS = (
    ("graph_coloring", "Graph", "#E8F1FA"),
    ("countdown", "Countdown", "#E7FBF6"),
    ("python_factors", "Python", "#EFFBE7"),
    ("mathir", "MathIR", "#E7FBEE"),
    ("pantry_plan", "Pantry", "#E7ECFB"),
)


def allocation(t: float, k: int = K_BANK) -> list[float]:
    """Point mass at ``t = 0``, uniform at ``t = 1``, convex in between."""
    return [(1.0 - t) * (1.0 if i == 0 else 0.0) + t / k for i in range(k)]


def within_bank_pmd(weights) -> float:
    """The paper's breadth statistic applied inside the bank: 1 - sum w^2."""
    return 1.0 - sum(w * w for w in weights)


def banked_distinct(weights, mass: float, draws: int = K_DRAWS) -> float:
    """Expected distinct banked keys in ``draws``: sum_c [1 - (1 - p_c)^K]."""
    return sum(1.0 - (1.0 - mass * w) ** draws for w in weights)


def any_banked(mass: float, draws: int = K_DRAWS) -> float:
    """Probability that some banked key is drawn: 1 - (1 - P_B)^K."""
    return 1.0 - (1.0 - mass) ** draws


def measured_points():
    """Per-prompt (breadth, distinct@8, verified mass, occupied keys)."""
    payload = json.loads(MEASURED.read_text())
    pts = []
    for row in payload["rows"]:
        if row["verified"] < 2:
            # PCMD needs two verified draws; below that a prompt has no
            # allocation to place on this axis at all.
            continue
        total = row["verified"]
        q = [c / total for c in row["counts"]]
        pts.append({"x": 1.0 - sum(v * v for v in q),
                    "y": row["distinct8"],
                    "mass": total / row["samples"],
                    "k": row["k"],
                    "domain": row["domain"]})
    return payload, pts


def bands_for(pts, mass, lo, hi):
    """Closed-form band over the key counts these prompts actually realize."""
    inside = [p for p in pts if lo <= p["mass"] <= hi]
    ks = sorted(p["k"] for p in inside)
    k_lo = max(ks[int(0.10 * (len(ks) - 1))] if ks else 2, 2)
    k_hi = max(ks[-1] if ks else 4, k_lo + 1)
    ts = [i / (STEPS - 1) for i in range(STEPS)]

    def curve(k):
        xs, ys = [], []
        for t in ts:
            w = allocation(t, k)
            xs.append(within_bank_pmd(w))
            ys.append(banked_distinct(w, mass))
        return xs, ys

    def y_at(k, target):
        xs, ys = curve(k)
        if target > xs[-1] + 1e-12:
            return None          # k keys cannot reach this breadth at all
        j = 0
        while j + 1 < len(xs) and xs[j + 1] < target:
            j += 1
        if j + 1 >= len(xs):
            return ys[-1]
        span = xs[j + 1] - xs[j]
        u = 0.0 if span <= 0 else (target - xs[j]) / span
        return ys[j] + u * (ys[j + 1] - ys[j])

    grid = [i / 200 * (1.0 - 1.0 / k_hi) for i in range(201)]
    band_lo, band_hi = [], []
    for g in grid:
        vals = [v for v in (y_at(k, g) for k in range(k_lo, k_hi + 1))
                if v is not None]
        band_lo.append(min(vals))
        band_hi.append(max(vals))
    return inside, grid, band_lo, band_hi, (k_lo, k_hi), ks


def main() -> None:
    style.apply_rcparams()
    cmap = style.sequential_cmap()
    colours = [cmap(v) for v in (0.45, 0.68, 0.95)]
    bins = [(0.17, 0.33), (0.42, 0.58), (0.67, 0.83)]

    payload, pts = measured_points()

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(style.WIDTH, 2.35),
                             sharex=True, sharey=True)
    record = []
    for axis, (key, label, tint) in zip(axes, DOMAINS):
        here = [p for p in pts if p["domain"] == key]
        axis.set_facecolor(tint)
        entry = {"domain": key, "label": label, "prompts": len(here), "bins": []}
        for (lo, hi), mass, colour in zip(bins, BANKED_MASS, colours):
            inside, grid, band_lo, band_hi, kband, ks = bands_for(here, mass, lo, hi)
            axis.fill_between(grid, band_lo, band_hi, color=colour, alpha=0.22,
                              linewidth=0, zorder=1)
            axis.plot(grid, band_hi, color=colour, linewidth=1.1, zorder=2)
            axis.scatter([p["x"] for p in inside], [p["y"] for p in inside],
                         s=6, color=colour, alpha=0.60, linewidths=0, zorder=3)
            entry["bins"].append({"mass": mass, "range": [lo, hi],
                                  "prompts": len(inside), "k_band": list(kband),
                                  "k_observed": [ks[0], ks[-1]] if ks else None})
        axis.set_title(label, fontsize=style.FONT - 1.6, color=style.INK, pad=2.0)
        style.style_axis(axis)
        axis.set_xlim(0.0, 1.0 - 1.0 / K_BANK)
        record.append(entry)

    axes[0].set_ylabel(f"distinct verified\nmodes in {K_DRAWS} draws")
    fig.supxlabel(r"within-prompt breadth $1-\sum_c q_c^2$",
                  fontsize=style.FONT, color=style.INK, y=-0.09)

    style.bottom_legend(
        fig,
        [Line2D([], [], color=c, linewidth=1.6) for c in colours]
        + [Line2D([], [], color=style.INK, marker="o", linestyle="none",
                  markersize=3.0)],
        [rf"closed form, $P\approx{m:.2f}$" for m in BANKED_MASS]
        + ["one held-out prompt"],
        y=-0.30,
        ncol=4,
    )

    style.save(fig, OUT)
    plt.close(fig)

    MACROS.write_text(
        "% Generated by ops/plot_paper_replay_bank_decomposition.py; "
        "do not hand edit.\n"
        + "\\newcommand{\\MDbankprompts}{" + f"{len(pts):,}" + "}\n")

    OUT.with_suffix(".json").write_text(json.dumps({
        "schema": "paper-figure-derived-v1",
        "kind": "derived-with-measurement",
        "curves": "closed forms printed in App. A.3, evaluated over "
                  "w(t) = (1 - t) e_1 + t U_k.",
        "measured": {
            "source": MEASURED.relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(MEASURED.read_bytes()).hexdigest(),
            "scope": payload["scope"],
            "prompts_plotted": len(pts),
            "circularity_note": payload["derivation"]["circularity_note"],
        },
        "builder": {"path": "ops/plot_paper_replay_bank_decomposition.py"},
        "panels": record,
        "bank_capacity": K_BANK,
        "capacity_note": "observed occupied keys stay far below a capacity of "
                         "16, so what bounds these policies is how many modes "
                         "they find, not how many the bank can hold.",
        "draws": K_DRAWS,
        "outputs": {
            extension: {
                "path": OUT.with_suffix("." + extension)
                           .relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(
                    OUT.with_suffix("." + extension).read_bytes()).hexdigest(),
            }
            for extension in ("pdf", "png")
        },
    }, indent=1) + "\n")
    print(json.dumps({"event": "built", "output": str(OUT.with_suffix(".pdf")),
                      "prompts": len(pts),
                      "panels": {e["label"]: e["prompts"] for e in record}}))


if __name__ == "__main__":
    main()
