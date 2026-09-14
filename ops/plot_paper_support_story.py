#!/usr/bin/env python3
"""Explain the paper's two-bottleneck verified-support story in one diagram.

Both paths share one column grid: POLICY SAMPLES, REWARD (+ KEY), a split
stage, an UPDATE, and the resulting SUPPORT. The reward-only path has no split
stage of its own, so its reward box simply spans that column too -- the same
"card spans N columns" trick as the ModeBench card grid -- which is what keeps
the two rows' boxes lined up edge to edge instead of drifting the way five
hand-placed boxes and six hand-placed boxes otherwise would. Each stage also
carries its own fill colour, shared by both rows, so a reader can match
POLICY SAMPLES-to-POLICY SAMPLES or UPDATE-to-UPDATE by colour alone before
reading a word of either box.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402


OUT = ROOT / "paper/figures/verified_support_story"

style.apply_rcparams()

INK = style.INK
MUTED = style.MUTED
GRID = style.GRID
PANEL = style.PANEL
WHITE = style.WHITE
RETENTION = style.METHOD  # teal: this row's own mechanism, so it also names the row
MODE_A, MODE_B, MODE_C, _ = style.MODE_RAMP

# The reward-only path is deliberately colourless in its row label and arrows:
# reward-only training cannot tell modes apart, and that message is carried by
# content (fewer chips, "B is gone from the batch"), not by painting the row a
# "danger" hue that would have to compete with the mode-identity palette.
# MaxRL gets a fresh green rather than the arm palette's ADD_ON: this figure
# never shows an experimental arm, and ADD_ON is the exact hex already spoken
# for by mode B (``style.MODE_RAMP[1]``) -- reusing it here would make one
# swatch mean two different things in the same figure. PATH_NEUTRAL and
# DISCOVERY are local to this figure and not exported.
PATH_NEUTRAL = "#42525E"
DISCOVERY = "#2F9E44"

# One fill per pipeline stage, shared by both rows: the same colour marks
# POLICY SAMPLES whether it feeds the reward-only path or Re:MaxRL, so a
# reader can match stage to stage by colour before reading either box. Hues
# are chosen clear of every colour that already means something else in this
# figure -- mode A/B/C (violet/rose/orange), MaxRL green, and Replay teal --
# each with a >=20 degree buffer, then lightened to the same pale, low-chroma
# band as the paper's own PANEL wash so no stage reads as more "important"
# than another; every stage clears >=3:1 against every mode and mechanism
# colour that can appear on top of it.
STAGE_FILL = {
    "samples": PANEL,  # unchanged: ties this figure to the paper's shared wash
    "reward": "#FBF9EA",
    "update": "#FBEAFB",
    "support": "#F1FBEA",
}
STAGE_EDGE = MUTED

# --- figure geometry ---------------------------------------------------
# Fixed here (not just at savefig time) because a badge drawn as a circle in
# these data units only renders as an on-page circle if the two axes carry
# the same physical inches-per-unit; see ``ASPECT`` below.
WIDTH_IN = style.WIDTH
HEIGHT_IN = 3.05
XLIM = (0.0, 1.0)
YLIM = (0.06, 1.0)
# inches-per-x-unit divided by inches-per-y-unit. The axes here is not square
# in data units (a wide, short canvas), so a Circle patch -- equal radius in
# both data axes -- would draw as an oval. Ellipses compensate by this factor
# so a badge is round on the page regardless of the canvas's own proportions.
ASPECT = (WIDTH_IN * (YLIM[1] - YLIM[0])) / (HEIGHT_IN * (XLIM[1] - XLIM[0]))


def rounded_box(ax, x, y, w, h, *, face=WHITE, edge=GRID, lw=1.0, radius=0.028):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=MUTED, lw=1.15, dashed=False):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        color=color,
        linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else "solid",
        connectionstyle="arc3,rad=0",
        shrinkA=1,
        shrinkB=1,
        clip_on=False,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def orthogonal_arrow(ax, start, end, *, corner_x, color=MUTED, lw=1.15):
    """Route an arrow with square, 90-degree bends, corner shared by its pair."""

    ax.plot(
        [start[0], corner_x, corner_x],
        [start[1], start[1], end[1]],
        color=color,
        linewidth=lw,
        solid_capstyle="round",
        solid_joinstyle="round",
        clip_on=False,
        zorder=1,
    )
    arrow(ax, (corner_x, end[1]), end, color=color, lw=lw)


def label(ax, x, y, value, **kwargs):
    defaults = {
        "ha": "center",
        "va": "center",
        "fontsize": style.FONT,
        "color": INK,
        "zorder": 3,
    }
    defaults.update(kwargs)
    return ax.text(x, y, value, **defaults)


def chip(ax, x, y, value, color, *, radius=0.0165):
    """A mode-identity badge: a true circle, matching Figure 2's badges.

    Drawn as an ``Ellipse`` rather than a ``Circle``: this axes is wider than
    it is tall in data units, so a patch with equal x/y radius would render
    as a flattened oval on the page. Compensating by ``ASPECT`` keeps it round.
    """

    ax.add_patch(
        Ellipse(
            (x, y), 2 * radius, 2 * radius * ASPECT,
            facecolor=WHITE, edgecolor=color, linewidth=1.3, zorder=2,
        )
    )
    label(ax, x, y, value, color=color, fontweight="bold")
    return radius * 2


def chip_row(ax, left, y, values, colors, *, radius=0.0165, gap=0.006):
    """Chips laid left to right from ``left``; returns the row's total width."""

    step = radius * 2 + gap
    for index, (value, color) in enumerate(zip(values, colors)):
        chip(ax, left + radius + index * step, y, value, color, radius=radius)
    return len(values) * step - gap


def chip_row_width(n, *, radius=0.0165, gap=0.006):
    return n * (radius * 2 + gap) - gap


def centered_chip_row(ax, cx, y, values, colors, *, radius=0.0165, gap=0.006):
    """A chip row centred on ``cx`` instead of hand-placed from a left edge.

    A fixed left inset (``cx - some eyeballed offset``) only happens to clear
    the box's own edge for whichever chip count and radius it was tuned
    against; a four-chip row at the box's normal radius overflowed the box by
    exactly this mistake. Centring from the row's own computed width is
    correct for any chip count or radius without re-tuning an offset by eye.
    """

    width = chip_row_width(len(values), radius=radius, gap=gap)
    return chip_row(ax, cx - width / 2, y, values, colors, radius=radius, gap=gap)


def stack(ax, cx, box_top, box_bottom, drawers, *, title_gap=0.075, bottom_pad=0.035, line_gap=0.070):
    """Centre a tightly, evenly spaced block of ``drawers`` below a box's title.

    Every box titles itself the same way (a bold line just under its top
    edge) and then has some number of content rows -- a subtitle, a coloured
    finding, a chip row. The rows keep one fixed spacing between them
    (``line_gap``, wide enough that a chip row never brushes the text above
    it) and the resulting block is centred in whatever room the box has below
    its title. Spreading rows across the *full* remaining height instead --
    the first draft of this helper -- pins two rows to the box's extreme top
    and bottom with a dead gap between them the moment the box is taller than
    its content needs, which is the "cramped title, then a lot of empty
    space, then a cramped bottom" look this exists to avoid.
    """

    content_top = box_top - title_gap
    content_bottom = box_bottom + bottom_pad
    n = len(drawers)
    center = (content_top + content_bottom) / 2
    start = center + (n - 1) * line_gap / 2
    ys = [start - index * line_gap for index in range(n)]
    for y, draw_fn in zip(ys, drawers):
        draw_fn(ax, cx, y)


def text_row(value, **kwargs):
    return lambda ax, cx, y: label(ax, cx, y, value, **kwargs)


# --- column grid --------------------------------------------------------
# Five columns wide enough for their busiest row: samples, reward(+key), the
# split, an update, and the resulting support. Weights, not equal widths,
# because "REWARD + KEY" and "SUPPORT MAINTAINED + EXPANDED" carry more text
# than the split stage's two one-word titles.
MARGIN = 0.014
# Only the column-1/column-2 gutter carries the rotated "reward"/"key" labels
# in their own whitespace (see the fork-in arrows below); the other three
# gutters separate boxes with nothing living between them, so they can run
# narrower and hand that space to the columns, which is where every measured
# overflow in this figure has come from.
GUTTERS = (0.028, 0.040, 0.028, 0.028)
WEIGHTS = (0.90, 1.05, 0.87, 1.23, 1.00)
_UNIT = (1 - 2 * MARGIN - sum(GUTTERS)) / sum(WEIGHTS)
_WIDTHS = [w * _UNIT for w in WEIGHTS]
_LEFTS = []
_cursor = MARGIN
for _index, _w in enumerate(_WIDTHS):
    _LEFTS.append(_cursor)
    _cursor += _w + (GUTTERS[_index] if _index < len(GUTTERS) else 0.0)


def column(index: int, span: int = 1) -> tuple[float, float]:
    """Left edge and width of column ``index``, optionally spanning forward."""

    left = _LEFTS[index]
    right = _LEFTS[index + span - 1] + _WIDTHS[index + span - 1]
    return left, right - left


def gutter_mid(index: int) -> float:
    """X midway through the gutter after column ``index``, for fork corners."""

    left, width = column(index)
    return left + width + GUTTERS[index] / 2


def draw() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    axis.set_xlim(*XLIM)
    axis.set_ylim(*YLIM)
    axis.axis("off")

    def fixed_label(ax, x, y, value, **kwargs):
        return label(ax, x, y, value, **kwargs)

    def row_stack(cx, box_top, box_bottom, drawers, **kwargs):
        stack(axis, cx, box_top, box_bottom, drawers, **kwargs)

    # ---------------- Row 1: reward-only path ----------------
    row1_top, row1_h = 0.895, 0.275
    row1_bottom = row1_top - row1_h
    row1_label_y = row1_top + 0.045

    label(
        axis, MARGIN, row1_label_y, "REWARD-ONLY PATH",
        ha="left", fontsize=style.SMALL_FONT, color=PATH_NEUTRAL, fontweight="bold",
    )

    x, w = column(0)
    cx = x + w / 2
    rounded_box(axis, x, row1_bottom, w, row1_h, face=STAGE_FILL["samples"], edge=STAGE_EDGE)
    fixed_label(axis, cx, row1_top - 0.045, "POLICY SAMPLES", fontsize=style.SMALL_FONT, fontweight="bold")
    row_stack(cx, row1_top, row1_bottom, [
        lambda ax, cx, y: centered_chip_row(
            axis, cx, y, ("A", "A", "B", r"$\bot$"), (MODE_A, MODE_A, MODE_B, MUTED), radius=0.0145,
        ),
    ])

    # Spans the split column too: the reward-only path never forks, so its one
    # reward box simply reaches as far as Re:MaxRL's two forked boxes do,
    # which is what keeps column 4 lined up between the rows.
    x, w = column(1, span=2)
    cx = x + w / 2
    rounded_box(axis, x, row1_bottom, w, row1_h, face=STAGE_FILL["reward"], edge=STAGE_EDGE)
    fixed_label(axis, cx, row1_top - 0.045, "BINARY REWARD", fontsize=style.SMALL_FONT, fontweight="bold")
    row_stack(cx, row1_top, row1_bottom, [
        text_row("reward:  1   1   1   0", fontweight="bold"),
        text_row("mode identity discarded", fontsize=style.SMALL_FONT, color=MUTED),
    ])

    x, w = column(3)
    cx = x + w / 2
    rounded_box(axis, x, row1_bottom, w, row1_h, face=STAGE_FILL["update"], edge=STAGE_EDGE)
    fixed_label(axis, cx, row1_top - 0.045, "MODE-BLIND UPDATE", fontsize=style.SMALL_FONT, fontweight="bold")
    row_stack(cx, row1_top, row1_bottom, [
        text_row("A in 2 of 3 correct rows", color=MODE_A, fontweight="bold"),
        text_row("B in 1 of 3 correct rows", color=MODE_B, fontweight="bold"),
    ])

    x, w = column(4)
    cx = x + w / 2
    rounded_box(axis, x, row1_bottom, w, row1_h, face=STAGE_FILL["support"], edge=STAGE_EDGE)
    fixed_label(axis, cx, row1_top - 0.045, "SUPPORT NARROWS", fontsize=style.SMALL_FONT, fontweight="bold")
    row_stack(cx, row1_top, row1_bottom, [
        lambda ax, cx, y: chip(axis, cx, y, "A", MODE_A),
        text_row("B is gone from the batch", fontsize=style.SMALL_FONT, color=MUTED),
    ])

    row1_mid = row1_bottom + row1_h * 0.5
    for start_col, end_col in ((0, 1), (1, 3), (3, 4)):
        start_x = column(start_col)[0] + column(start_col)[1]
        end_x = column(end_col)[0]
        arrow(axis, (start_x + 0.004, row1_mid), (end_x - 0.004, row1_mid), color=PATH_NEUTRAL, dashed=True)

    # ---------------- Row 2: Re:MaxRL ----------------
    row2_top = row1_bottom - 0.115
    row2_h = 0.335
    row2_bottom = row2_top - row2_h
    row2_label_y = row2_top + 0.045

    # "Re:MaxRL" keeps its own mixed-case brand spelling even inside this
    # otherwise-all-caps label, matching how it is written everywhere else in
    # the paper -- the way "REWARD-ONLY PATH" beside it does not get its own
    # brand treatment because it is a plain description, not a method name.
    label(
        axis, MARGIN, row2_label_y, "Re:MaxRL (ours)",
        ha="left", fontsize=style.SMALL_FONT, color=RETENTION, fontweight="bold",
    )

    x, w = column(0)
    cx = x + w / 2
    col0_right = x + w
    rounded_box(axis, x, row2_bottom, w, row2_h, face=STAGE_FILL["samples"], edge=STAGE_EDGE)
    fixed_label(axis, cx, row2_top - 0.045, "POLICY SAMPLES", fontsize=style.SMALL_FONT, fontweight="bold")
    row_stack(cx, row2_top, row2_bottom, [
        lambda ax, cx, y: centered_chip_row(
            axis, cx, y, ("A", "A", "B", r"$\bot$"), (MODE_A, MODE_A, MODE_B, MUTED), radius=0.0145,
        ),
    ])

    x, w = column(1)
    col1_left = x
    col1_right = x + w
    cx = x + w / 2
    rounded_box(axis, x, row2_bottom, w, row2_h, face=STAGE_FILL["reward"], edge=STAGE_EDGE)
    fixed_label(axis, cx, row2_top - 0.045, "REWARD + KEY", fontsize=style.SMALL_FONT, fontweight="bold")
    row_stack(cx, row2_top, row2_bottom, [
        text_row("one validator returns both", fontsize=style.SMALL_FONT, color=MUTED),
        text_row("reward:  1  1  1  0", fontweight="bold"),
        lambda ax, cx, y: centered_chip_row(
            axis, cx, y, ("A", "A", "B", r"$\bot$"), (MODE_A, MODE_A, MODE_B, MUTED), radius=0.0145,
        ),
    ])

    fork_gap = 0.022
    fork_h = (row2_h - fork_gap) / 2
    x, w = column(2)
    col2_left = x
    cx = x + w / 2
    maxrl_top = row2_top
    rounded_box(axis, x, maxrl_top - fork_h, w, fork_h, face="#F0F9EE", edge=DISCOVERY, lw=1.3)
    fixed_label(axis, cx, maxrl_top - 0.030, "MAXRL", fontsize=style.SMALL_FONT, color=DISCOVERY, fontweight="bold")
    row_stack(cx, maxrl_top, maxrl_top - fork_h, [
        text_row("fresh search learns", fontsize=style.SMALL_FONT, fontweight="bold"),
        lambda ax, cx, y: chip(axis, cx, y, "C", MODE_C, radius=0.0145),
    ], title_gap=0.050, bottom_pad=0.032, line_gap=0.052)

    replay_bottom = row2_bottom
    replay_top = replay_bottom + fork_h
    rounded_box(axis, x, replay_bottom, w, fork_h, face="#F1FAFA", edge=RETENTION, lw=1.3)
    fixed_label(axis, cx, replay_top - 0.030, "REPLAY", fontsize=style.SMALL_FONT, color=RETENTION, fontweight="bold")
    row_stack(cx, replay_top, replay_bottom, [
        text_row("maintains", fontsize=style.SMALL_FONT, fontweight="bold"),
        lambda ax, cx, y: centered_chip_row(axis, cx, y, ("A", "B"), (MODE_A, MODE_B), radius=0.0145),
    ], title_gap=0.050, bottom_pad=0.032, line_gap=0.052)
    col2_right = x + w

    x, w = column(3)
    col3_left = x
    cx = x + w / 2
    rounded_box(axis, x, row2_bottom, w, row2_h, face=STAGE_FILL["update"], edge=STAGE_EDGE)
    fixed_label(axis, cx, row2_top - 0.045, "Re:MaxRL UPDATE", fontsize=style.SMALL_FONT, fontweight="bold")
    # Mirrors MODE-BLIND UPDATE's own shape exactly: one coloured, descriptive
    # line per component, so the two UPDATE boxes read as the same kind of
    # thing instead of one being two sentences and the other two bare labels.
    row_stack(cx, row2_top, row2_bottom, [
        text_row("MaxRL adds mode C", color=DISCOVERY, fontweight="bold"),
        text_row("Replay keeps modes A, B", color=RETENTION, fontweight="bold"),
    ])
    col3_right = x + w

    x, w = column(4)
    cx = x + w / 2
    rounded_box(axis, x, row2_bottom, w, row2_h, face=STAGE_FILL["support"], edge=STAGE_EDGE)
    fixed_label(axis, cx, row2_top - 0.045, "SUPPORT MAINTAINED", fontsize=style.SMALL_FONT, fontweight="bold")
    fixed_label(axis, cx, row2_top - 0.086, "+ EXPANDED", fontsize=style.SMALL_FONT, fontweight="bold")
    row_stack(cx, row2_top, row2_bottom, [
        text_row("A + B kept, C added", color=RETENTION, fontweight="bold"),
        lambda ax, cx, y: centered_chip_row(axis, cx, y, ("A", "B", "C"), (MODE_A, MODE_B, MODE_C)),
    ], title_gap=0.116)

    row2_mid = row2_bottom + row2_h * 0.5
    arrow(axis, (col0_right + 0.004, row2_mid), (col1_left - 0.004, row2_mid), color=MUTED)

    # Reward and key leave the same box as two separate channels, not one
    # stub that later splits, so each gets its own line from the start.
    fork_in_x = gutter_mid(1)
    orthogonal_arrow(
        axis, (col1_right, row2_mid + 0.014), (col2_left + 0.014, maxrl_top - fork_h * 0.5),
        corner_x=fork_in_x, color=DISCOVERY,
    )
    orthogonal_arrow(
        axis, (col1_right, row2_mid - 0.014), (col2_left + 0.014, replay_bottom + fork_h * 0.5),
        corner_x=fork_in_x, color=RETENTION,
    )
    # Rotated to run alongside the line rather than across it: the gutter here
    # is only wide enough for a word's height, not its width, so upright text
    # cannot avoid the line and both neighbouring boxes at once.
    label(
        axis, fork_in_x - 0.006, row2_mid + 0.026, "reward",
        ha="center", va="bottom", rotation=90, fontsize=style.SMALL_FONT, color=DISCOVERY,
    )
    label(
        axis, fork_in_x - 0.006, row2_mid - 0.026, "key",
        ha="center", va="top", rotation=90, fontsize=style.SMALL_FONT, color=RETENTION,
    )

    fork_out_x = gutter_mid(2)
    orthogonal_arrow(
        axis, (col2_right, maxrl_top - fork_h * 0.5), (col3_left, row2_mid + 0.028),
        corner_x=fork_out_x, color=DISCOVERY,
    )
    orthogonal_arrow(
        axis, (col2_right, replay_bottom + fork_h * 0.5), (col3_left, row2_mid - 0.028),
        corner_x=fork_out_x, color=RETENTION,
    )

    arrow(axis, (col3_right + 0.004, row2_mid), (column(4)[0] - 0.004, row2_mid), color=MUTED)

    figure.tight_layout(pad=0.05)
    return figure


def main() -> None:
    figure = draw()
    style.save(figure, OUT, dpi=300)
    plt.close(figure)
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
