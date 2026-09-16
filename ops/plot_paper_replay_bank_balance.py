#!/usr/bin/env python3
"""Schematic for App. A.3: why replay balances within the bank.

Three states of one prompt's verified modes, drawn as overlapping regions
whose area is the conditional probability the policy puts on each mode. The
base model spreads correct probability over several modes; training on the
verifier alone concentrates it onto one; replaying banked keys uniformly holds
every banked mode above a floor.

This plate is illustrative. The probabilities are chosen to show the three
states legibly, not measured, and the record it writes says so: nothing in the
paper cites it as evidence. The measured version of the same contrast is the
base-grid table and the retention comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

OUT = ROOT / "paper/figures/replay_bank_balance"
LABELS = ("A", "B", "C", "D")
# Centres chosen so the regions overlap: a prompt's modes are alternative
# solutions to one problem, not disjoint populations.
CENTRES = ((0.37, 0.59), (0.62, 0.63), (0.43, 0.36), (0.67, 0.39))
# Area is the probability the policy *produces* that verified mode, not a
# conditional distribution over modes, so the four areas need not sum to a
# constant: under replay every banked mode can grow at once, which is what the
# measured pass@8-and-breadth result looks like.
# Each number is the probability the policy produces that verified mode at
# least once in a draw, so they are per-mode probabilities in [0, 1] whose sum
# may exceed 1 -- the events are not exclusive across samples.
#
# Totals carry the paper's result and not just its breadth half: RLVR-only
# training raises how often the policy is right while collapsing onto one way
# of being right, so its total area exceeds the base model's even as three
# modes shrink to almost nothing. Replay raises the total further and spreads
# it. A cannot simply be enlarged to show this: it is a probability, so 0.95 is
# near its ceiling, and the rise has to come from the base panel being lower.
STATES = (
    ("Base", (0.22, 0.19, 0.16, 0.13)),          # total 0.70
    ("RLVR-only", (0.95, 0.05, 0.04, 0.03)),     # total 1.07, concentrated on A
    ("+ Replay", (0.68, 0.61, 0.56, 0.51)),      # total 2.36, spread
)
FLOOR = 0.22  # the probability floor the replayed modes are held above


# Area carries the probability, so radius goes as its square root. The constant
# is set by the largest mass any panel shows, so the biggest blob still clears
# its panel edge and the three panels stay comparable to each other.
SCALE = 0.235 / max(p for _, probabilities in STATES for p in probabilities) ** 0.5


def radius(probability: float) -> float:
    return SCALE * probability ** 0.5


def draw_state(axis, probabilities, *, floors: bool) -> None:
    for (x, y), probability, colour, label in zip(
            CENTRES, probabilities, style.MODE_RAMP, LABELS):
        r = radius(probability)
        axis.add_patch(Circle((x, y), r, facecolor=colour, alpha=.45,
                              edgecolor=colour, linewidth=.9, zorder=2))
        if floors:
            axis.add_patch(Circle((x, y), radius(FLOOR), facecolor="none",
                                  edgecolor=colour, linewidth=.7,
                                  linestyle=(0, (2, 1.6)), alpha=.9, zorder=3))
        axis.text(x, y, label, ha="center", va="center", zorder=4,
                  fontsize=style.FONT - 3.2, color=style.WHITE,
                  fontweight="bold")


def _legend(fig) -> None:
    """A key under the panels: the four correct modes, then the wash.

    The wash is the part a reader is most likely to misread. Without naming it,
    the three panels look like a partition of correct answers, and the blue
    shrinking under "+ Replay" reads as something lost rather than as less
    error.

    Built through the legend machinery rather than placed by hand: it centres
    the row itself, and its markers stay round. Drawing circles on a wide, short
    strip in axes coordinates flattens them into dashes.
    """
    class _Bubble(HandlerBase):
        """A legend handle that is the panel's mark: a filled circle lettered
        in white, not a marker with the letter set beside it."""

        def __init__(self, letter: str, colour: str):
            super().__init__()
            self.letter, self.colour = letter, colour

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            centre = (width / 2 - xdescent, height / 2 - ydescent)
            radius = min(width, height) / 2
            circle = Circle(centre, radius, facecolor=to_rgba(self.colour, .45),
                            edgecolor=self.colour, linewidth=.9, transform=trans)
            text = Text(centre[0], centre[1], self.letter, ha='center',
                        va='center', color=style.WHITE, fontweight='bold',
                        fontsize=fontsize - 1.0, transform=trans)
            return [circle, text]

    blank = Line2D([], [], linestyle='none', marker='none')
    modes = [Line2D([], [], linestyle='none', marker='o', markersize=7.5,
                    markerfacecolor=to_rgba(colour, .45), markeredgecolor=colour,
                    markeredgewidth=.9, label='')
             for label, colour in zip(LABELS, style.MODE_RAMP)]
    bubbles = {handle: _Bubble(label, colour)
               for handle, label, colour in zip(modes, LABELS, style.MODE_RAMP)}
    wash = Patch(facecolor=style.PANEL, edgecolor=style.GRID, linewidth=.8,
                 label='incorrect output space')
    legend = fig.legend(
        handles=[blank, *modes, wash],
        labels=['correct modes', *['' for _ in LABELS], 'incorrect output space'],
        handler_map=bubbles,
        loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=False,
        fontsize=style.FONT - 2.0, handletextpad=.35, columnspacing=.9,
        handlelength=1.9, handleheight=1.9, borderaxespad=0)
    for text in legend.get_texts():
        text.set_color(style.INK)


def main() -> None:
    style.apply_rcparams()
    # Taller than the panels alone need: the extra strip carries a legend, so a
    # reader does not have to infer from the prose that the wash is error mass.
    fig, axes = plt.subplots(1, 3, figsize=(3.25, 1.40))
    fig.subplots_adjust(wspace=.02, bottom=.20)
    for axis, (title, probabilities) in zip(axes, STATES):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                      boxstyle="round,pad=0.01,rounding_size=0.03",
                                      facecolor=style.PANEL, edgecolor=style.GRID,
                                      linewidth=.8, zorder=0))
        draw_state(axis, probabilities, floors=title.startswith("+"))
        # The wash outside the mode regions is the rest of the policy's
        # probability: responses that do not verify. Without naming it a reader
        # can read the panel as a partition of correct answers and take the
        # shrinking blue in "+ Replay" for a loss rather than for less error.
        # Named once, on the first panel, because it means the same in all three.
        axis.set_title(title, fontsize=style.FONT - 2.2, color=style.INK, pad=1.5)

    _legend(fig)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    OUT.with_suffix(".json").write_text(json.dumps({
        "schema": "paper-figure-schematic-v1",
        "kind": "schematic",
        "data": "illustrative; no measured source. Probabilities are chosen for "
                "legibility and are not estimates.",
        "unfilled_area": "the panel wash outside the mode regions is unverified "
                         "(incorrect) probability mass; labelled on the first panel.",
        "builder": {"path": "ops/plot_paper_replay_bank_balance.py"},
        "states": {title: dict(zip(LABELS, probabilities))
                   for title, probabilities in STATES},
        "floor": FLOOR,
        "totals": {title: round(sum(p), 3) for title, p in STATES},
        "totals_note": "total area rises Base -> RLVR-only -> + Replay: accuracy "
                       "improves in both, while only replay keeps the modes apart.",
        "outputs": {"pdf": f"{OUT.name}.pdf", "png": f"{OUT.name}.png"},
    }, indent=1) + "\n")
    print(json.dumps({"event": "built", "output": str(OUT.with_suffix('.pdf'))}))


if __name__ == "__main__":
    main()
