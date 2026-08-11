#!/usr/bin/env python3
"""Render verified replay as one horizontal row of components.
Four equal columns, read left to right: verify outputs, store one exemplar per
observed mode, revisit prompt-local banks, and replay their exemplars uniformly.
Colour names a verified execution mode,
not an arm, so it comes from ``paper_style.MODE_RAMP``: a known key is slot 0,
a newly observed key is slot 1, and a mechanism action is slot 2, exactly as in
`plot_paper_collapse_toy.py` and the ModeBench examples, so a colour means the
same thing in every figure of the paper.

The canvas stays 13.2in wide because the column layout below is hand-placed in
inches on it. Type is sized through ``paper_style.font_for_canvas`` so that,
after \\includegraphics scales this canvas to \\textwidth, the labels land the
same size on paper as the text in every other figure.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

OUT = ROOT / "paper/figures/verified_replay_mechanism"

# Inches on a canvas whose single axes spans it one-to-one.
WIDTH = 13.2  # the collapse story's canvas, so both figures print at one scale

FONT = style.font_for_canvas(WIDTH)
MONO = "DejaVu Sans Mono"

INK = style.INK
MUTED = style.MUTED
GRID = style.GRID
FRAME = style.MUTED
PANEL = style.PANEL
WHITE = style.WHITE

KEY_SEEN = style.MODE_RAMP[0]  # a key the bank already holds
KEY_NEW = style.MODE_RAMP[1]  # a newly observed verified key
ACTION = style.MODE_RAMP[2]  # a pressure the method applies
INVALID = style.INVALID  # off-ramp on purpose, as in the collapse story

style.apply_rcparams(font_size=FONT)
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "mathtext.fontset": "dejavusans",
    }
)
MARGIN = 0.14
GAP = 0.14
COLUMNS = 4
COL_W = (WIDTH - 2 * MARGIN - (COLUMNS - 1) * GAP) / COLUMNS
PAD = 0.14
INNER = COL_W - 2 * PAD

# Inter-line gaps are typographic, not structural: they were set as ~1.2x the
# original 17pt line height, so they scale with the type rather than staying
# fixed in inches. Leaving them fixed while the type shrank pushed the two
# lines of a card title far enough apart that pdftotext stopped reading them as
# one line, which the paper's figure contract checks for.
_TYPE_SCALE = FONT / 17.0
LINE = 0.28 * _TYPE_SCALE
CHIP_H = 0.36
# Each card carries three bands, not four: heading, drawing, one takeaway. The
# card used to run a muted subtitle above the drawing as well, which said the
# same thing as the takeaway under it in different words, cost a band of height
# on every card, and in the replay card overlapped the box beneath it. The
# distinguishing half of each subtitle moved into the takeaway line instead.
BODY_TOP = 0.80
BODY_H = 1.42
FOOTER_ZONE = 0.50
CARD_H = BODY_TOP + BODY_H + FOOTER_ZONE
HEIGHT = CARD_H + 0.20


def box(ax, x, y, w, h, *, face=WHITE, edge=GRID, radius=0.05, lw=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            clip_on=False,
            zorder=1,
        )
    )


def text(ax, x, y, value, **kwargs):
    defaults = {"fontsize": FONT, "color": INK, "ha": "center", "va": "center", "zorder": 3}
    defaults.update(kwargs)
    return ax.text(x, y, value, **defaults)


def fits(fig, artist, limit: float, where: str) -> None:
    width = artist.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    assert width <= limit + 1e-6, f"{where}: {width:.2f}in exceeds {limit:.2f}in"


def arrow(ax, start, end, *, color=MUTED, lw=1.3, scale=11):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            color=color,
            linewidth=lw,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=2,
        )
    )


def chip(fig, ax, x, center, lines, *, accent, face=WHITE, color=None, mono=False,
         width=None, bold=True, pad=0.16):
    """Centred rounded chip holding one or two lines, sized to the column."""

    width = INNER if width is None else width
    height = len(lines) * LINE + 0.10
    box(ax, x, center - height / 2, width, height, face=face, edge=accent, radius=0.05, lw=1.3)
    for index, line in enumerate(lines):
        artist = text(
            ax,
            x + width / 2,
            center - (index - (len(lines) - 1) / 2) * LINE,
            line,
            color=accent if color is None else color,
            fontweight="bold" if bold else "normal",
            fontfamily=MONO if mono else "sans-serif",
        )
        fits(fig, artist, width - pad, line)
    return height


def header(fig, ax, x, step, title, size):
    """Badge and title on one line; the stage kicker is the badge itself.

    The card previously spent three bands on its heading: a badge row, a
    ``STEP 1`` kicker beside it, and the title stacked underneath. The kicker
    said nothing the badge letter and the column order did not already say, and
    the band it occupied pushed every card half an inch taller. Title text now
    sits on the badge's own line, and the whole heading is one line.
    """

    top = HEIGHT - 0.10
    box(ax, x, top - 0.40, 0.32, 0.30, face=INK, edge=INK, radius=0.05)
    text(ax, x + 0.16, top - 0.25, step[0], color=WHITE, fontweight="bold")
    # Two lines at the card's own type size, not one line shrunk to fit. The
    # column is 1.77in wide beside the badge and the longest title needs 2.45in
    # on one line, so a single line can only be bought by making the heading
    # smaller than the text it heads. Wrapping instead keeps the heading at full
    # size and puts it in space the card was leaving blank.
    for index, line in enumerate(title):
        artist = text(
            ax,
            x + 0.42,
            top - 0.17 - index * LINE,
            line,
            fontweight="bold",
            ha="left",
            fontsize=size,
        )
        fits(fig, artist, INNER - 0.42, line)


def heading_size(fig, ax) -> float:
    """One type size for all four headings: the largest that fits every line.

    Sizing each card independently is easy and wrong: it turns a typographic
    accident into apparent emphasis. The titles are fixed copy
    checked by the figure contract, so the type yields to the longest of them
    and every card then shares it.
    """

    limit = INNER - 0.42
    size = FONT
    while size > 0.55 * FONT:
        widest = 0.0
        for _, title, _ in COLUMN_SPECS:
            for line in title:
                probe = text(
                    ax, 0, 0, line, fontweight="bold", ha="left", fontsize=size,
                )
                widest = max(
                    widest,
                    probe.get_window_extent(
                        renderer=fig.canvas.get_renderer()
                    ).width
                    / fig.dpi,
                )
                probe.remove()
        if widest <= limit:
            return size
        size -= 0.25
    raise AssertionError("no heading size fits every card")


def footer(fig, ax, x, lines):
    """The card's one takeaway, carrying what its drawing cannot show."""

    base = HEIGHT - 0.10 - CARD_H
    for index, line in enumerate(lines):
        artist = text(
            ax,
            x + INNER / 2,
            base + 0.36 - index * LINE,
            line,
            fontweight="bold",
        )
        fits(fig, artist, INNER, line)


def body_top() -> float:
    """Where every card's drawing starts; all four are top-aligned to it."""

    return HEIGHT - 0.10 - BODY_TOP


def draw_score(fig, ax, x):
    """Three rollouts meet one validator; only verified outputs are admitted."""

    # Top-aligned like every other card's drawing: centring these three chips in
    # the body zone instead left this card's takeaway line crowded while the
    # others had air above theirs.
    top = body_top()
    first = top - CHIP_H / 2
    rows = (
        ("y₁→c₁", "seen mode", KEY_SEEN, WHITE),
        ("y₂→c₂", "new mode", KEY_NEW, WHITE),
        ("y₃→⊥", "invalid", MUTED, INVALID),
    )
    for index, (rollout, verdict, accent, face) in enumerate(rows):
        yy = first - index * (CHIP_H + 0.12)
        box(ax, x, yy - CHIP_H / 2, INNER, CHIP_H, face=face, edge=accent, radius=0.05, lw=1.3)
        left = text(ax, x + 0.10, yy, rollout, color=INK, fontweight="bold", fontfamily=MONO, ha="left")
        right = text(ax, x + INNER - 0.10, yy, verdict, color=accent, fontweight="bold", ha="right")
        # The rollout and its verdict share one chip, so they are checked
        # together: they must leave a visible gap, not merely fit.
        used = sum(
            artist.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
            for artist in (left, right)
        )
        assert used <= INNER - 0.32, f"{rollout} {verdict}: {used:.2f}in leaves no gap"
    footer(fig, ax, x, ["one validator gives", "reward and identity"])


def draw_retain(fig, ax, x):
    """A discovered key is written to a bank that outlives the group."""

    top = body_top()
    first = top - CHIP_H / 2
    chip(fig, ax, x, first, ["{c₁}  at start"], accent=KEY_SEEN, color=INK)
    arrow(ax, (x + INNER / 2, first - CHIP_H / 2 - 0.03), (x + INNER / 2, first - CHIP_H / 2 - 0.23), color=KEY_NEW)
    text(ax, x + INNER / 2 + 0.10, first - CHIP_H / 2 - 0.13, "+ c₂", color=KEY_NEW, fontweight="bold", ha="left")
    second = first - CHIP_H - 0.26
    chip(fig, ax, x, second, ["{c₁, c₂}  kept"], accent=KEY_NEW, color=INK)
    footer(fig, ax, x, ["one exemplar per key,", "in a lasting bank"])


def draw_schedule(fig, ax, x):
    """A deterministic cursor gives one prompt-local bank each update."""

    top = body_top()
    centers = (top - 0.24, top - 0.71, top - 1.18)
    for index, center in enumerate(centers, start=1):
        chip(
            fig,
            ax,
            x,
            center,
            [f"B{index}"],
            accent=ACTION if index == 2 else FRAME,
            color=INK,
            face=PANEL if index != 2 else WHITE,
            width=0.84,
        )
        if index == 2:
            text(
                ax,
                x + 1.03,
                center,
                "← τ",
                color=ACTION,
                fontweight="bold",
                ha="left",
            )
    footer(fig, ax, x, ["one prompt-local bank", "per optimizer step"])


def draw_uniform_replay(fig, ax, x):
    """Every exemplar in the scheduled verified bank receives equal weight.

    The two-line exemplar chip is centred on its own half-height rather than on
    ``CHIP_H``: the card's other chips hold one line, and reusing their centre
    for this one pushed it above the body zone and into the band above.
    """

    top = body_top()
    stack_h = 2 * LINE + 0.10
    first = top - stack_h / 2
    chip(fig, ax, x, first, ["e₁ ∈ c₁", "e₂ ∈ c₂"], accent=KEY_NEW, color=INK)
    arrow(
        ax,
        (x + INNER / 2, first - stack_h / 2 - 0.03),
        (x + INNER / 2, first - stack_h / 2 - 0.25),
        color=ACTION,
    )
    chip(
        fig,
        ax,
        x,
        first - stack_h / 2 - 0.45,
        ["Lrep = mean(−sθ)"],
        accent=ACTION,
        color=INK,
        mono=True,
        pad=0.10,
    )
    footer(fig, ax, x, ["every banked mode", "keeps a gradient"])


COLUMN_SPECS = (
    (("A", "STEP 1"), ("Verify", "outputs"), draw_score),
    (("B", "STEP 2"), ("Store", "exemplars"), draw_retain),
    (("C", "STEP 3"), ("Revisit", "recurrently"), draw_schedule),
    (("D", "STEP 4"), ("Replay", "uniformly"), draw_uniform_replay),
)


def render() -> None:
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_axis_off()
    fig.canvas.draw()

    top = HEIGHT - 0.10
    lefts = [MARGIN + index * (COL_W + GAP) for index in range(COLUMNS)]
    for left in lefts:
        box(ax, left, top - CARD_H, COL_W, CARD_H, face=PANEL, edge=GRID, radius=0.09, lw=1.1)

    head_size = heading_size(fig, ax)
    for left, (step, title, draw) in zip(lefts, COLUMN_SPECS):
        header(fig, ax, left + PAD, step, title, head_size)
        draw(fig, ax, left + PAD)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    render()
