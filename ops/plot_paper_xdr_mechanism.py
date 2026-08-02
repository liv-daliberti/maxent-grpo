#!/usr/bin/env python3
"""Render xGRPO as one horizontal row of components.

Five equal columns, read left to right: three pipeline steps (discover, retain,
rebalance) followed by the two modulators that act on them (the self-referenced
controller and the singleton escape). Colour names a verified execution mode,
not an arm, so it comes from ``paper_style.MODE_RAMP``: a known key is slot 0,
a new key is slot 1, and a mechanism action is slot 2, exactly as in
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

OUT = ROOT / "paper/figures/xdr_mechanism"

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
KEY_NEW = style.MODE_RAMP[1]  # a key this group discovered
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
COLUMNS = 5
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
SUB_LINE = 0.26 * _TYPE_SCALE
CHIP_H = 0.36
HEAD_TO_TITLE = 0.62
BODY_TOP = 1.74
FOOTER_ZONE = 0.62
BODY_H = 1.90
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


def header(fig, ax, x, step, title):
    top = HEIGHT - 0.10
    box(ax, x, top - 0.40, 0.32, 0.30, face=INK, edge=INK, radius=0.05)
    text(ax, x + 0.16, top - 0.25, step[0], color=WHITE, fontweight="bold")
    text(ax, x + 0.42, top - 0.25, step[1], color=MUTED, ha="left")
    for index, line in enumerate(title):
        artist = text(
            ax,
            x + INNER / 2,
            top - HEAD_TO_TITLE - 0.16 - index * LINE,
            line,
            fontweight="bold",
        )
        fits(fig, artist, INNER, line)


def subtitle(fig, ax, x, lines):
    top = HEIGHT - 0.10
    for index, line in enumerate(lines):
        artist = text(
            ax,
            x + INNER / 2,
            top - 1.42 - index * SUB_LINE,
            line,
            color=MUTED,
        )
        fits(fig, artist, INNER, line)


def footer(fig, ax, x, lines):
    base = HEIGHT - 0.10 - CARD_H
    for index, line in enumerate(lines):
        artist = text(
            ax,
            x + INNER / 2,
            base + 0.44 - index * LINE,
            line,
            fontweight="bold",
        )
        fits(fig, artist, INNER, line)


def body_zone() -> tuple[float, float]:
    top = HEIGHT - 0.10 - BODY_TOP
    return top, top - BODY_H


def draw_discover(fig, ax, x):
    """Three rollouts meet one validator; only what it accepts earns anything."""

    subtitle(fig, ax, x, ["one validator:", "reward + identity"])
    top, bottom = body_zone()
    center = (top + bottom) / 2
    rows = (
        ("y₁→c₁", "seen", KEY_SEEN, WHITE),
        ("y₂→c₂", "new +ν", KEY_NEW, WHITE),
        ("y₃→⊥", "invalid", MUTED, INVALID),
    )
    for index, (rollout, verdict, accent, face) in enumerate(rows):
        yy = center + (1 - index) * (CHIP_H + 0.12)
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
    footer(fig, ax, x, ["a bonus needs", "V to accept"])


def draw_retain(fig, ax, x):
    """A discovered key is written to a bank that outlives the group."""

    subtitle(fig, ax, x, ["a new key joins", "a lasting bank"])
    top, _ = body_zone()
    first = top - CHIP_H / 2
    chip(fig, ax, x, first, ["{c₁}  at start"], accent=KEY_SEEN, color=INK)
    arrow(ax, (x + INNER / 2, first - CHIP_H / 2 - 0.03), (x + INNER / 2, first - CHIP_H / 2 - 0.23), color=KEY_NEW)
    text(ax, x + INNER / 2 + 0.10, first - CHIP_H / 2 - 0.13, "+ c₂", color=KEY_NEW, fontweight="bold", ha="left")
    second = first - CHIP_H - 0.26
    chip(fig, ax, x, second, ["{c₁, c₂}  kept"], accent=KEY_NEW, color=INK)
    arrow(ax, (x + INNER / 2, second - CHIP_H / 2 - 0.03), (x + INNER / 2, second - CHIP_H / 2 - 0.23), color=ACTION)
    chip(
        fig,
        ax,
        x,
        second - CHIP_H - 0.44,
        ["global replay,", "round-robin"],
        accent=ACTION,
        color=INK,
    )
    footer(fig, ax, x, ["it acts again", "before its turn"])


def draw_rebalance(fig, ax, x):
    """Replay keeps correct mass up and stops one key from owning it."""

    subtitle(fig, ax, x, ["replay holds P⁺,", "evens the keys"])
    top, bottom = body_zone()
    base = top - 0.60
    groups = ((x + 0.20, "before", (0.46, 0.15)), (x + INNER - 0.92, "after", (0.32, 0.32)))
    for left, label, heights in groups:
        for index, (height, accent) in enumerate(zip(heights, (KEY_SEEN, KEY_NEW))):
            box(
                ax,
                left + index * 0.40,
                base,
                0.30,
                height,
                face=accent,
                edge=accent,
                radius=0.03,
                lw=0,
            )
        text(ax, left + 0.35, base - 0.18, label, color=MUTED)
    arrow(ax, (x + INNER / 2 - 0.16, base + 0.22), (x + INNER / 2 + 0.16, base + 0.22), color=ACTION)
    chip(fig, ax, x, top - 1.28, ["verified mass"], accent=ACTION, color=INK)
    chip(fig, ax, x, top - 1.78, ["known-key KL"], accent=ACTION, color=INK)
    footer(fig, ax, x, ["high P⁺, no", "key monopoly"])


def draw_control(fig, ax, x):
    """Each sensor raises exactly one pressure, and only while it is slipping."""

    subtitle(fig, ax, x, ["each sensor sets", "one pressure"])
    top, bottom = body_zone()
    center = (top + bottom) / 2
    rows = (
        ("mode entropy ↓", "β rare/new ↑", KEY_SEEN),
        ("surprisal ↑", "µ mass ↑", KEY_NEW),
        ("bank entropy ↓", "α balance ↑", ACTION),
    )
    for index, (sensor, pressure, accent) in enumerate(rows):
        # Sensor over pressure inside one chip, so the pairing is a single
        # object rather than two lines that happen to sit together.
        yy = center + (1 - index) * 0.66
        box(ax, x, yy - 0.29, INNER, 0.58, face=WHITE, edge=accent, radius=0.05, lw=1.3)
        artist = text(ax, x + INNER / 2, yy + 0.14, sensor, color=MUTED)
        fits(fig, artist, INNER - 0.16, sensor)
        artist = text(ax, x + INNER / 2, yy - 0.14, "→ " + pressure, color=accent, fontweight="bold")
        fits(fig, artist, INNER - 0.16, pressure)
    footer(fig, ax, x, ["pressure tracks", "its own sensor"])


def draw_escape(fig, ax, x):
    """A bank stuck at one key may add a verified alternative to replay only."""

    subtitle(fig, ax, x, ["fires only when", "one bank, one key"])
    top, bottom = body_zone()
    first = top - CHIP_H / 2
    chip(fig, ax, x, first, ["{c₁}  alone"], accent=KEY_SEEN, color=INK)
    arrow(ax, (x + INNER / 2, first - CHIP_H / 2 - 0.04), (x + INNER / 2, first - CHIP_H / 2 - 0.34), color=ACTION)
    text(ax, x + INNER / 2 + 0.10, first - CHIP_H / 2 - 0.19, "verify", color=ACTION, ha="left")
    chip(fig, ax, x, first - CHIP_H - 0.38, ["+ c₂ to bank"], accent=KEY_NEW, color=INK)
    chip(
        fig,
        ax,
        x,
        bottom + 0.36,
        ["replay support,", "never enters PPO"],
        accent=FRAME,
        color=INK,
        face=WHITE,
        bold=False,
        pad=0.10,
    )
    footer(fig, ax, x, ["one new key,", "after warmup"])


COLUMN_SPECS = (
    (("A", "STEP 1"), ("Discover", "verified modes"), draw_discover),
    (("B", "STEP 2"), ("Retain every", "discovery"), draw_retain),
    (("C", "STEP 3"), ("Preserve mass", "+ rebalance"), draw_rebalance),
    (("D", "CONTROL"), ("Self-referenced", "control"), draw_control),
    (("E", "ESCAPE"), ("Singleton", "escape"), draw_escape),
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

    for index, (left, (step, title, draw)) in enumerate(zip(lefts, COLUMN_SPECS)):
        header(fig, ax, left + PAD, step, title)
        draw(fig, ax, left + PAD)
        if index == 2:
            # The three pipeline steps read left to right on their own; the
            # rule marks where the modulators that act on them begin.
            divider = left + COL_W + GAP / 2
            ax.plot(
                [divider, divider],
                [top - CARD_H + 0.30, top - 0.30],
                color=FRAME,
                linewidth=1.0,
                linestyle=(0, (2, 3)),
                zorder=0,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    render()
