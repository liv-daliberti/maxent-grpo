#!/usr/bin/env python3
"""One visual language for every figure in ``paper/main.tex``.

Before this module the manuscript carried three unrelated styles: the cool
navy small-multiples of the E72 figures, a warm cream language in the
explanatory figures, and a blue/orange trajectory style inherited from the E68
progress surface. The last one is the reason this file exists. It painted
matched Dr.GRPO blue and xGRPO orange, while the E72 figures painted matched
Dr.GRPO orange and xGRPO teal, so a reader who learned ``orange'' in
Figure~\\ref{fig:modes-per-epoch} met the opposite arm under the same colour in
Figure~\\ref{fig:replay-ablation-curves}. Colour has to follow the entity, so
the arm palette is defined once, here, and imported everywhere.

Geometry is the second thing this centralises, and it is what makes the E72
figures legible where the older ones are not. ``iclr2026_conference.sty`` sets
\\textwidth to 5.5in. A figure authored 7.35in wide and included at \\linewidth
is scaled by .75, so its 7.2pt type lands at about 5.4pt on the page. A figure
authored 11.5in wide is scaled by .48 and the same nominal type lands near
3.9pt --- unreadable in print, and the only difference is the authoring canvas.
``WIDTH`` is therefore the one canvas width every figure is built on, and
``panel_height`` keeps the per-panel aspect of the reference figure as the row
count changes.

The palettes are validated rather than chosen by eye:

- Arms (``CONTROL``/``ABLATION``/``METHOD``) report chroma FAIL on the method
  teal and weak tritan separation from the control orange. This is a known and
  accepted trade: the pair is the manuscript's established identity. It is
  legal only because identity is never colour alone --- every arm also carries
  its own dash pattern from ``ARM_DASH``. Do not draw an arm as a solid line of
  its colour with no other cue.
- ``MODE_RAMP`` passes every check under ``--pairs all`` (worst deutan dE 6.6,
  normal-vision dE 15.0), which the plasma ramp it replaces did not: that one
  failed the lightness band at both ends and put ``other valid`` on a yellow
  with 1.46:1 contrast against white. All-pairs is the right test because these
  categories meet as touching segments of a stacked bar.

Both results are reproducible with the validator in the dataviz skill:

    python3 validate_palette.py "#C76A3A,#6C5CE7,#087F8C" --mode light
    python3 validate_palette.py "#7048E8,#C2255C,#C76A3A,#0E8F86" \\
        --mode light --pairs all
"""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib as mpl

# --- canvas -----------------------------------------------------------------
# The authoring width of the reference figure. Every figure uses it so that all
# figures land on the page at one common scale factor (5.5/7.35 = .748).
WIDTH = 7.35

# \textwidth in iclr2026_conference.sty. Anything included at \linewidth is
# scaled to this, whatever canvas it was authored on.
TEXTWIDTH = 5.5

# Height of the reference figure's two metric rows, used to hold the per-panel
# aspect constant when a figure has a different number of rows.
_REFERENCE_ROWS = 2
_REFERENCE_HEIGHT = 3.5

# --- ink --------------------------------------------------------------------
INK = "#19324A"  # titles, labels, any text
MUTED = "#607487"  # spines and tick marks; recedes behind the data
GRID = "#D8E2EA"  # grid lines, lighter still
PANEL = "#E8F1FA"  # pale blue panel wash; every mark clears 3:1 on it
# Worst mark-on-wash contrast is the control orange at 3.31:1; ink text sits at
# 11.5:1. Re-check with the dataviz validator before darkening this.
WHITE = "#FFFFFF"

# --- arms -------------------------------------------------------------------
# Identity is colour + dash. Never one without the other.
CONTROL = "#C76A3A"  # matched Dr.GRPO
METHOD = "#087F8C"  # xGRPO
ABLATION = "#6C5CE7"  # B3a and other remove-one arms
# A fourth arm slot, for add-one arms that must coexist with the three above in
# one panel (E83's semantic-MaxEnt-without-replay against E81's with it).
# Chosen by the dataviz validator under `--pairs all`, not by eye: every pair
# in {CONTROL, ABLATION, METHOD, ADD_ON} clears CVD dE 8.9 (worst, deutan) and
# the normal-vision floor at 15.0, in both light and dark. An olive candidate
# looked fine on the adjacent-pair test and failed all-pairs against CONTROL at
# dE 4.3 protan, which is why the all-pairs run is the one that counts here.
ADD_ON = "#C2255C"
# A fifth arm slot, for the adaptive-coefficient variant of an add-one arm.
# Deliberately a sibling of ABLATION rather than a new hue: both are semantic
# MaxEnt, fixed and adapted, so a related colour with its own dash reads as a
# variant rather than an unrelated treatment. It is the only purple far enough
# from ABLATION to clear the checks (dE 12.7 deutan, 15.4 normal); every lighter
# variant collided with it.
#
# LIGHT MODE ONLY. Under `--pairs all` it clears CVD separation and the
# normal-vision floor in light, and does not become the binding pair on either.
# Against a dark surface it fails the lightness band and lands at 2.12:1
# contrast. The manuscript figures are print artifacts on white, so this is
# accepted knowingly; do not reuse ADAPTIVE in a dark-mode context.
ADAPTIVE = "#7B1FA2"

# A dose variant of an arm is the *same* entity at a different setting, so it
# keeps its mechanism's hue and separates on dash alone. This is the one case
# where sharing a colour is correct rather than a collision: E90 is verified
# replay, dosed by bank occupancy instead of by a fixed coefficient. Giving it
# a sixth hue would say "unrelated treatment", which is the opposite of true.
METHOD_DOSE_DASH = (0, (7, 1.5, 2, 1.5))

ARM_DASH: dict[str, Any] = {
    CONTROL: (0, (5, 1.6)),
    ABLATION: (0, (1.6, 1.4)),
    ADD_ON: (0, (4, 1.2, 1, 1.2)),
    ADAPTIVE: (0, (6, 1.2, 1, 1.2, 1, 1.2)),
    METHOD: "solid",
}

# --- mode identity ----------------------------------------------------------
# For the explanatory figures, where colour names a verified execution mode
# rather than an arm. Order is the legend order.
MODE_RAMP = ("#7048E8", "#C2255C", "#C76A3A", "#0E8F86")
# Not a mode; a neutral off-ramp that must recede. Stepped down from #E5E9ED
# when the wash went pale blue: against that wash the old grey sat at 1.07:1
# and the legend swatch and the top bar segment both vanished. This still
# recedes far behind every mode colour, which clear 3.3:1 or better.
INVALID = "#CBD6E0"

# --- line weights -----------------------------------------------------------
MEAN_LW = 1.35  # a five-seed mean
SEED_LW = 0.5  # an individual seed, when shown at all
BAND_ALPHA = 0.13  # seed-range fill
GRID_LW = 0.55
SPINE_LW = 0.55

# --- type -------------------------------------------------------------------
FONT = 7.2  # body text inside a figure
TITLE_FONT = 7.6  # panel titles
LABEL_FONT = 7.0  # axis labels
SMALL_FONT = 6.4  # annotations that must not compete with the data


def apply_rcparams(font_size: float = FONT) -> None:
    """Install the shared defaults. Call once, before creating a figure."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": SPINE_LW,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,  # embed as Type 42 so the text stays selectable
            "ps.fonttype": 42,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def font_for_canvas(canvas_width: float, size: float = FONT) -> float:
    """Type size that lands on the page at ``size`` from a wider canvas.

    The schematic figures are drawn on an oversized canvas on purpose: their
    layout is hand-placed in inches, and rebuilding it at ``WIDTH`` would mean
    re-tuning every coordinate. That is fine --- \\includegraphics scales any
    canvas to \\linewidth --- but it means their nominal type size is not
    comparable to a figure authored at ``WIDTH``. Feed the canvas width through
    here and the text lands the same size on paper as everything else.
    """
    return size * canvas_width / WIDTH


def panel_height(rows: int, *, per_row: float | None = None) -> float:
    """Figure height that holds the reference per-panel aspect at ``rows``."""
    if per_row is None:
        per_row = _REFERENCE_HEIGHT / _REFERENCE_ROWS
    return per_row * rows


def style_axis(axis, *, grid: str = "both", title: str | None = None) -> None:
    """Recessive grid, no top/right spines, small ticks. The house treatment."""
    if grid in {"both", "x", "y"}:
        axis.grid(
            True,
            axis=grid if grid in {"x", "y"} else "both",
            color=GRID,
            linewidth=GRID_LW,
            zorder=0,
        )
        axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_linewidth(SPINE_LW)
    axis.tick_params(length=2.0, width=SPINE_LW, pad=1.5, labelsize=FONT)
    if title is not None:
        axis.set_title(title, fontsize=TITLE_FONT, color=INK, pad=3)


def bottom_legend(
    figure,
    handles: Iterable,
    labels: Iterable,
    *,
    y: float = -0.055,
    ncol: int | None = None,
) -> None:
    """One legend for the whole figure, below it, unboxed."""
    labels = list(labels)
    figure.legend(
        list(handles),
        labels,
        loc="lower center",
        ncol=ncol if ncol is not None else len(labels),
        frameon=False,
        fontsize=FONT,
        bbox_to_anchor=(0.5, y),
        handlelength=2.6,
    )


def save(figure, path, *, png: bool = True, dpi: int = 240) -> None:
    """Write the figure with the reference's tight bounds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    if png:
        figure.savefig(
            path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02
        )
