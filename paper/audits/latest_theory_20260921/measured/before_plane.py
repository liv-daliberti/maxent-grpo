#!/usr/bin/env python3
"""App. Q: the anchor's trade-off curve, and where the memory arms sit on it.

One panel, both axes measured: correctness across, breadth up. A method is
better up and to the right, so the plate answers the comparison directly
rather than asking the reader to hold two bar charts side by side.

The anchor is a curve because its coefficient is a dial. It runs right and up
while beta is small, then turns back left: past the per-domain knee the
coefficient keeps buying breadth and starts paying correctness for it, which
is the cost Proposition "The retained conditional is the reference's" makes
explicit through its correctness identity. The memory arms are single points
because a replay dose is not a dial in the same sense --- Corollary
"Conditional full-coverage categorical limit" gives the same destination for
every positive dose.

The rule is the base model's own success-conditional breadth. The same
proposition bounds an anchor by it. It does not bound a memory, which is
limited by what discovery banks instead, and the per-domain plate shows Graph
as the domain where that difference is visible.
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
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

SOURCE = ROOT / "paper/results/reference_kl_comparison.json"
OUT = ROOT / "paper/figures/reference_kl_plane"
# The validated five; see paper_style.FRONTIER_FIVE.
KL, DRGRPO, REDR, MAXRL, REMAX = (style.COMPARATOR, style.CONTROL, style.METHOD,
                                  style.ADD_ON, style.ABLATION)
# The anchor's curve is a frontier, and the two washes say which side of it a
# point falls on. Same pair as the knee plate, so one reading carries across.
BEYOND = "#E7F5EA"
SHORT = "#FBEAE7"


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    # A base-model bound has to be measurable to be drawn. The resample's own
    # bar is thirty defined prompts, and a frozen policy that rarely succeeds
    # clears it nowhere: Countdown and PythonFactors are excluded here for that
    # reason, not for their results. Their curves are in the per-domain plate.
    domains = {k: r for k, r in payload["domains"].items()
               if r.get("frozen_pmd") is not None and (r.get("frozen_defined") or 0) >= 30}
    keys = sorted(domains)
    # every coefficient the whole set carries, so the curve grows as cells land
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


    # The compute-matched control is this family at beta = 0: same objective,
    # same bank scaffolding with a zero derivative, one knob unturned. It opens
    # the curve rather than standing apart from it.
    labels = ["0", *betas]
    xs = [mean(lambda r: r["control"]["pass8"])] + [
        mean((lambda b: lambda r: r["kl"][b]["pass8"])(b)) for b in betas]
    ys = [mean(lambda r: r["control"]["pmd"])] + [
        mean((lambda b: lambda r: r["kl"][b]["pmd"])(b)) for b in betas]
    axis.plot(xs, ys, color=KL, lw=style.MEAN_LW, marker="o", markersize=4.2,
              markerfacecolor=style.WHITE, markeredgewidth=1.3, zorder=3)
    # One offset for all six put labels on the curve, on the breadth rule, and
    # on the key. These are placed per coefficient instead: the direction each
    # one leans is the direction that is empty at that point.
    # The accuracy rule cuts the left half of the plane, so every label but the
    # control's leans right, into the empty side. The control is the one point
    # left of the rule, so its label leans left into the margin.
    nudge = {"0": ((-6, 1), "right"), "0.001": ((9, -4), "left"),
             "0.01": ((5, 10), "left"), "0.04": ((11, 1), "left"),
             # far enough left to finish before the accuracy rule rather than
             # straddle it, and below the breadth rule
             "0.1": ((-19, -13), "right"), "0.2": ((8, 9), "left"),
             "0.3": ((12, 6), "left")}
    for beta, x, y in zip(labels, xs, ys):
        text = rf"$\beta={beta.lstrip('0') or '0'}$"
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

    # PCMD rises monotonically with beta, so the curve is a function of
    # breadth: at each breadth level it gives the pass@8 the anchor manages
    # there. Green is the side that beats it at the same breadth, red the side
    # that does not. Past the swept range the endpoints are simply held, which
    # is why the wash extends to the frame without the curve doing so.
    (x_lo, x_hi), (y_lo, y_hi) = axis.get_xlim(), axis.get_ylim()
    edge_y = [y_lo, *ys, y_hi]
    edge_x = [xs[0], *xs, xs[-1]]
    axis.fill_betweenx(edge_y, edge_x, x_hi, facecolor=BEYOND, zorder=0)
    axis.fill_betweenx(edge_y, x_lo, edge_x, facecolor=SHORT, zorder=0)
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
               label="base model accuracy"),
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
