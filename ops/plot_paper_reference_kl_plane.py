#!/usr/bin/env python3
"""Terminal correctness and diversity for reference KL and replay.

The panel equally averages the domains with reportable initial PCMD and
uses coefficients available in every included domain. Replay and MaxRL
points use the same domain set. Lines connect measured means without
uncertainty bands. Initial-policy rules describe
a reference, not upper bounds on neural performance. Only one replay dose
is represented, so this figure does not compare dose sensitivity.
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
OUT = ROOT / "paper/figures/reference_kl_plane"
# The validated five; see paper_style.FRONTIER_FIVE.
KL, DRGRPO, REDR, MAXRL, REMAX = (style.COMPARATOR, style.CONTROL, style.METHOD,
                                  style.ADD_ON, style.ABLATION)
# Washes compare correctness with the measured KL curve at equal diversity.
# They sit below every rule, curve and marker.
BEYOND, SHORT = "#EAF4EC", "#FBECEC"



def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    # Initial PCMD is displayed only with at least thirty defined prompts.
    # This selects Graph, MathIR, and Pantry; the other two domains remain in
    # the domain-level table and coefficient figure.
    domains = {k: r for k, r in payload["domains"].items()
               if r.get("frozen_pmd") is not None and (r.get("frozen_defined") or 0) >= 30}
    keys = sorted(domains)
    # Use the intersection of measured coefficients across the included domains.
    betas = sorted({b for k in keys for b in domains[k]["kl"]},
                   key=float)
    betas = [b for b in betas if all(b in domains[k]["kl"] for k in keys)]

    def mean(pick):
        values = [pick(domains[k]) for k in keys]
        return statistics.fmean(values) if all(v is not None for v in values) else None

    style.apply_rcparams()
    figure, axis = plt.subplots(figsize=(style.WIDTH * 0.50, style.WIDTH * 0.38))
    style.style_axis(axis)

    frozen = mean(lambda r: r["frozen_pmd"])
    frozen_x = mean(lambda r: r["frozen_pass8"])
    axis.axhline(frozen, color=style.MUTED, lw=1.0, ls=(0, (4, 2)), zorder=1)
    # The frozen policy's own accuracy, so the plate says where training starts
    # on both axes rather than only on one. Their crossing is the base model.
    if frozen_x is not None:
        axis.axvline(frozen_x, color=style.MUTED, lw=1.0, ls=(0, (1, 2)), zorder=1)


    # The beta = 0 point is the control with a zero replay derivative.
    # Optimizer updates and fresh rollouts match, but total compute is not
    # equalized or reallocated to effective baseline training.
    labels = ["0", *betas]
    xs = [mean(lambda r: r["control"]["pass8"])] + [
        mean((lambda b: lambda r: r["kl"][b]["pass8"])(b)) for b in betas]
    ys = [mean(lambda r: r["control"]["pmd"])] + [
        mean((lambda b: lambda r: r["kl"][b]["pmd"])(b)) for b in betas]
    axis.plot(xs, ys, color=KL, lw=style.MEAN_LW, marker="o", markersize=4.2,
              markerfacecolor=style.WHITE, markeredgewidth=1.3, zorder=3)
    # Two coincident points print as one dot. A wider ring around the outer
    # one makes the pair legible without moving either off its value.
    if (len(xs) >= 2 and abs(xs[-1] - xs[-2]) < 0.01
            and abs(ys[-1] - ys[-2]) < 0.01):
        axis.plot([xs[-1]], [ys[-1]], marker="o", markersize=8.4, color=KL,
                  markerfacecolor="none", markeredgewidth=1.3, zorder=3)
    # One offset for all six put labels on the curve, on the breadth rule, and
    # on the key. These are placed per coefficient instead: the direction each
    # one leans is the direction that is empty at that point.
    # The accuracy rule cuts the left half of the plane, so every label but the
    # control's leans right, into the empty side. The control is the one point
    # left of the rule, so its label leans left into the margin.
    nudge = {"0": ((-6, 1), "right"), "0.001": ((9, -4), "left"),
             "0.01": ((5, 10), "left"), "0.04": ((11, 1), "left"),
             "0.1": ((-2, -13), "center"), "0.2": ((9, 7), "left")}
    # The final two points nearly coincide at this figure's scale. A shared
    # annotation keeps both coefficient labels legible.
    same = (len(xs) >= 2 and abs(xs[-1] - xs[-2]) < 0.01
            and abs(ys[-1] - ys[-2]) < 0.01)
    for index, (beta, x, y) in enumerate(zip(labels, xs, ys)):
        if same and index == len(labels) - 1:
            continue
        if same and index == len(labels) - 2:
            text = (r"$\beta=" + labels[-2].lstrip("0") + r"$, $"
                    + labels[-1].lstrip("0") + r"$ (overlapping)")
        else:
            text = r"$\beta=" + (beta.lstrip("0") or "0") + r"$"
        offset, align = nudge.get(beta, ((-6, 8), "right"))
        axis.annotate(text, (x, y), textcoords="offset points", xytext=offset,
                      fontsize=style.FONT - 1.2, color=KL, ha=align, zorder=4)

    # These names reach the key verbatim; "(ours)" matches how the level-gain
    # figure marks the same two arms.
    points = (("MaxRL", MAXRL, "D", lambda r: r["maxrl"]),
              ("Re:Dr (ours)", REDR, "^", lambda r: r["replay"]),
              ("Re:Max (ours)", REMAX, "v", lambda r: r["remax"]))
    for name, colour, marker, pick in points:
        x, y = mean(lambda r: pick(r)["pass8"]), mean(lambda r: pick(r)["pmd"])
        axis.plot([x], [y], marker=marker, markersize=6.2, color=colour,
                  markerfacecolor=style.WHITE, markeredgewidth=1.6, zorder=4)

    axis.set_xlim(0.40, 0.90)
    axis.set_ylim(-0.02, 0.62)

    # Follow the measured polyline, including its small final reversal, rather
    # than the initial-policy rules. Do not extrapolate a KL comparison beyond
    # the observed diversity range. Both washes share the same curve boundary.
    x0, x1 = axis.get_xlim()
    axis.fill_betweenx(ys, x0, xs, facecolor=SHORT, edgecolor="none", zorder=0)
    axis.fill_betweenx(ys, xs, x1, facecolor=BEYOND, edgecolor="none", zorder=0)

    axis.set_xlabel("terminal pass@8", fontsize=style.FONT, color=style.INK)
    axis.set_ylabel("terminal PCMD", fontsize=style.FONT, color=style.INK)

    # One key rather than six labels scattered over the plane: colour and
    # marker carry the arm, and the beta labels are left on the curve because
    # they mark positions along it rather than name a series. The lower left is
    # where Dr.GRPO and MaxRL sit, so the key takes the empty lower right.
    keys_ = [
        Line2D([], [], color=KL, lw=style.MEAN_LW, marker="o", markersize=4.2,
               markerfacecolor=style.WHITE, markeredgewidth=1.3,
               label=r"Dr.GRPO $+$ reference KL"),
        Line2D([], [], color=style.MUTED, lw=1.0, ls=(0, (4, 2)),
               label="base model PCMD"),
        Line2D([], [], color=style.MUTED, lw=1.0, ls=(0, (1, 2)),
               label="base model pass@8"),
    ]
    keys_ += [Line2D([], [], color=colour, lw=0, marker=marker, markersize=6.2,
                     markerfacecolor=style.WHITE, markeredgewidth=1.6,
                     label=name)
              for name, colour, marker, _ in points]
    axis.legend(handles=keys_, loc="lower right", frameon=True, framealpha=0.92,
                edgecolor=style.MUTED, fontsize=style.FONT - 0.6,
                handlelength=2.0, labelspacing=0.45, borderpad=0.5)
    figure.subplots_adjust(left=0.135, right=0.99, top=0.97, bottom=0.155)
    style.save(figure, OUT)
    print(f"wrote {OUT.relative_to(ROOT)}   domains={keys}")
    print(f"  base model PCMD {frozen:.3f}")
    for beta, x, y in zip(labels, xs, ys):
        print(f"  KL beta={beta:<6} pass@8 {x:.3f}  pcmd {y:.3f}")
    for name, _, _, pick in points:
        print(f"  {name:<9} pass@8 {mean(lambda r: pick(r)['pass8']):.3f}  "
              f"pcmd {mean(lambda r: pick(r)['pmd']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
