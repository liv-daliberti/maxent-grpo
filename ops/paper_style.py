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

# An *external* estimator we compare against, rather than one of our own arms.
# Ordinary GRPO used to be drawn at #2E6FBB, which the all-pairs validator puts
# at dE 10.0 (normal vision) from the method teal --- so the single pair the
# paper's claim rests on, GRPO against Re:Dr, was the hardest pair in
# the figure to tell apart. This deeper blue clears every check in the frontier
# five below. It is a comparator hue, not an arm slot: do not use it for an
# ablation or a dose variant of one of ours.
COMPARATOR = "#00509E"

# The five identities that meet in one panel in the pass-8 frontier and the
# direct-comparator figures: matched Dr.GRPO (CONTROL), GRPO (COMPARATOR),
# Re:Dr (METHOD), UCPO (ABLATION), RLEP-Dr (ADD_ON). Validated together
# under `--pairs all`, which is the right test because all five are scattered
# into the same axes:
#
#     node validate_palette.js "#C76A3A,#6C5CE7,#087F8C,#C2255C,#00509E" \
#         --mode light --pairs all
#
# Worst CVD dE 8.9 (deutan, ADD_ON vs METHOD), normal-vision floor exactly 15.0
# (ADD_ON vs CONTROL), every slot inside the lightness band and over 3:1 on the
# surface. The one standing FAIL is the METHOD teal's chroma, the documented
# trade recorded above; it is why every one of these five also carries its own
# marker shape. UCPO and RLEP-Dr borrow the ABLATION and ADD_ON hues because
# they never share a panel with the semantic arms that own those slots --- if
# that ever changes, re-run the validator on the combined set rather than
# cycling a sixth hue.
FRONTIER_FIVE = (CONTROL, COMPARATOR, METHOD, ABLATION, ADD_ON)

# A few figures colour by *metric* rather than by arm: the sustained-AUC panels
# plot one arm's effect on correctness, on distinct modes, and on the adjusted
# difference, and only the first panel labels the rows, so colour is what
# carries the metric across the other four. Those figures had the correctness
# series at #2E6FBB, which the validator puts at dE 10.0 (normal) from the
# METHOD teal used for the distinct-mode series --- two of the three metrics in
# the same panel, hard to tell apart. This olive clears the in-panel trio
# {METRIC, METHOD, ADD_ON} outright --- worst normal-vision dE 15.1, worst CVD
# dE 8.2 deutan:
#
#     node validate_palette.js "#4E6910,#087F8C,#C2255C" --mode light --pairs all
#
# Across the whole vocabulary it is not fully independent: against CONTROL it
# sits at dE 7.5 protan, inside the 6--8 band that is legal only with a second
# channel. That is acceptable here and nowhere else, because METRIC is never
# drawn in a panel that also draws an arm, and every series in these figures
# carries its own marker shape. It is a metric slot, never an arm. METHOD and
# ADD_ON are reused for the other two metric series on the same grounds; if an
# arm is ever added to one of those figures, re-run the validator on the
# combined set rather than reaching for a seventh hue.
METRIC = "#4E6910"

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
    COMPARATOR: (0, (1.2, 1.3)),
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

# --- magnitude --------------------------------------------------------------
# One hue, light to dark, in the manuscript's own blue. This exists because its
# absence was being filled by ``cmap="viridis"`` in the mechanism diagnostics:
# a rainbow ramp is the wrong encoding for magnitude, it shares no hue with
# anything else in the paper, and its bright yellow top end took white cell
# labels down to about 1.1:1 --- the largest values in those grids were the
# ones a reader could not read. Lightness here is strictly monotonic, so the
# ramp still orders correctly in greyscale and under every CVD simulation.
SEQUENTIAL_STOPS = (
    "#F1F6FA",
    "#C3DAEA",
    "#8DB8D6",
    "#4E8CB8",
    "#256A94",
    "#19324A",
)


def sequential_cmap(name: str = "paper_blues"):
    """The manuscript's magnitude ramp as a matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(SEQUENTIAL_STOPS))


def _relative_luminance(hex_color: str) -> float:
    raw = hex_color.lstrip("#")
    channels = []
    for offset in (0, 2, 4):
        value = int(raw[offset : offset + 2], 16) / 255
        channels.append(
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(one: str, other: str) -> float:
    """WCAG contrast between two opaque hex colours."""
    high, low = sorted(
        (_relative_luminance(one), _relative_luminance(other)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def cell_ink(cell_color: Any) -> str:
    """Ink for a label drawn *on* a filled cell: whichever reads better.

    Picking one fixed label colour for a whole heatmap guarantees that one end
    of the ramp is unreadable. Choosing per cell keeps every number above 3.5:1
    on ``SEQUENTIAL_STOPS``, where the crossover falls at the ramp's midpoint.
    Accepts anything matplotlib can resolve to a colour, including an RGBA
    tuple straight from a colormap call.
    """
    from matplotlib.colors import to_hex

    resolved = to_hex(cell_color)
    return WHITE if contrast_ratio(resolved, WHITE) >= contrast_ratio(resolved, INK) else INK


# --- line weights -----------------------------------------------------------
MEAN_LW = 1.35  # a five-seed mean
SEED_LW = 0.5  # an individual seed, when shown at all
BAND_ALPHA = 0.13  # seed-range fill
GRID_LW = 0.55
PAPER_GRID_AXIS = "both"
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
    """Recessive two-axis grid, no top/right spines, and small ticks.

    ``grid`` remains for backwards compatibility, but any enabled paper grid
    draws both vertical and horizontal guides. This is a visual invariant.
    """
    if grid in {"both", "x", "y"}:
        axis.grid(
            True,
            axis=PAPER_GRID_AXIS,
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


def dominant_note(notes: Iterable[str]) -> str | None:
    """The provenance note shared by most panels, or ``None`` if none is.

    A grid of fifteen small multiples that stamps ``n=5 - 8p - terminal`` into
    every panel has spent fifteen slots of the reader's attention to say one
    thing, and the one panel where it reads ``n=1`` --- the only panel where the
    note carries information --- looks exactly like the fourteen that do not.
    Hoist the common case into the subtitle and draw the badge only where a
    panel departs from it, so the annotation marks an exception rather than
    decorating the rule.

    Returns ``None`` when no note covers more than half the panels; in that case
    the notes genuinely differ and each panel should keep its own.
    """
    counts: dict[str, int] = {}
    total = 0
    for note in notes:
        counts[note] = counts.get(note, 0) + 1
        total += 1
    if not counts:
        return None
    note, hits = max(counts.items(), key=lambda item: item[1])
    return note if hits * 2 > total else None


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
