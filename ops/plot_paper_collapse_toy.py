#!/usr/bin/env python3
"""Render the paper's model-backed Qwen2.5-3B Graph Coloring example.

The figure is not a simulation. It mechanically selects one prompt from the
four available Qwen2.5-3B-Instruct Dr.GRPO/Re:Max pairs (seeds 70--73)
using their fixed-seed evaluation samples. Duplicate records caused by resumes
are resolved by retaining the last (step, draw_index) record, exactly as a
checkpoint snapshot.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox
from matplotlib.transforms import offset_copy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

# Inches. The layout below is hand-placed on this canvas, so it stays put and
# the type is scaled to it instead.
CANVAS_WIDTH = 13.2
CANVAS_HEIGHT = 5.32

PAIR_DRAWS = {
    70: {
        "drgrpo": ROOT / "var/data/xdr_qwen25_3b_instruct_grpo_compute_matched_"
        "e80r1_qwen3b_aligned_graph_control_s70/debug_job30277372/"
        "eval_mode_coverage_draws.jsonl",
        "replay_maxrl": ROOT / "var/data/xdr_qwen25_3b_instruct_maxrl_verified_"
        "replay_e118q3_graph_replay_maxrl_s70/debug_job31010883/"
        "eval_mode_coverage_draws.jsonl",
    },
    71: {
        "drgrpo": ROOT / "var/data/xdr_qwen25_3b_instruct_grpo_compute_matched_"
        "e80r1_qwen3b_aligned_graph_control_s71/debug_job30277374/"
        "eval_mode_coverage_draws.jsonl",
        "replay_maxrl": ROOT / "var/data/xdr_qwen25_3b_instruct_maxrl_verified_"
        "replay_e118q3_graph_replay_maxrl_s71/debug_job31010885/"
        "eval_mode_coverage_draws.jsonl",
    },
    72: {
        "drgrpo": ROOT / "var/data/xdr_qwen25_3b_instruct_grpo_compute_matched_"
        "e80r1_qwen3b_aligned_graph_control_s72/debug_job30277376/"
        "eval_mode_coverage_draws.jsonl",
        "replay_maxrl": ROOT / "var/data/xdr_qwen25_3b_instruct_maxrl_verified_"
        "replay_e118q3_graph_replay_maxrl_s72/debug_job31010887/"
        "eval_mode_coverage_draws.jsonl",
    },
    73: {
        "drgrpo": ROOT / "var/data/xdr_qwen25_3b_instruct_grpo_compute_matched_"
        "e80r1_qwen3b_aligned_graph_control_s73/debug_job30277378/"
        "eval_mode_coverage_draws.jsonl",
        "replay_maxrl": ROOT / "var/data/xdr_qwen25_3b_instruct_maxrl_verified_"
        "replay_e118q3_graph_replay_maxrl_s73/debug_job31010889/"
        "eval_mode_coverage_draws.jsonl",
    },
}
OUT = ROOT / "paper/figures/modecollapse_story"
AUDIT = ROOT / "var/artifacts/paper_graph_collapse_toy.json"

# One type size for every label in the figure, so nothing in the printed panel
# reads as a second-class annotation. Sized through the shared helper so that,
# once this 13.2in canvas is scaled to \textwidth, the labels match the type in
# every other figure.
FONT = style.font_for_canvas(CANVAS_WIDTH)

INK = style.INK
MUTED = style.MUTED
GRID = style.GRID
FRAME = style.MUTED
PANEL = style.PANEL
WHITE = style.WHITE

# One tint per card -- A, B, C -- instead of A's own warm cream sitting next
# to one flat blue wash shared by B and C: that pairing read as two unrelated
# figures glued together. B and C fill one shared card split down the seam
# between their panels rather than two separate boxes, so the two stay
# visually "touching" exactly as before. B and C's hues sit in the one wide
# gap the paper's palette otherwise leaves open (MODE_RAMP claims
# violet/rose/orange/teal, roughly 40-156 degrees is free of all four).
# Panel A instead takes the same blue family as PantryPlan's card in
# Figure 2: pulling it out of the green family it first shared with C reads
# as clearly distinct at a glance, and blue is safe for panel A specifically
# because it is the one card that also holds the vertex-colour ramp below
# (see ``NODE_COLORS``), which is warm sepia and so sits nowhere near blue.
CARD_A = "#E8EDFA"  # blue, matching Figure 2's PantryPlan card
CARD_B = "#FAF9E8"  # gold
CARD_C = "#EFFAE8"  # yellow-green

# Two scales share this figure and must never be confused. The *series* scale
# identifies executed answer modes and is the shared ``MODE_RAMP``; it is the
# only saturated thing in the figure, in the bar panels and their legend. The
# *vertex* scale is the puzzle's own three colours in panel A. It is a
# high-contrast achromatic ramp, while the solution-mode scale is saturated
# and multi-hued. Explicit labels in the two cards make these independent
# encodings readable without requiring the reader to infer the distinction.
MODE_COLORS = {
    "31212": style.MODE_RAMP[0],  # shared initial mode
    "33112": style.MODE_RAMP[1],  # most frequent shared initial mode
    "31312": style.MODE_RAMP[2],  # shared initial and terminal mode
}
OTHER = style.MODE_RAMP[3]  # any further verified mode
INVALID = style.INVALID  # invalid response — off-ramp on purpose, so it recedes
# Legend labels carry their series colour. Every mode fill clears 3:1 against
# the surface under the validated ramp, so unlike the plasma ramp this replaced
# --- whose yellow sat at 1.46:1 and needed a hand-picked darker gold --- each
# label can simply wear its own series colour. Only the neutral invalid swatch
# still borrows muted ink, because it is not a series.
LABEL_COLORS = {
    "mode A": MODE_COLORS["31212"],
    "mode B": MODE_COLORS["33112"],
    "mode C": MODE_COLORS["31312"],
    "other valid": OTHER,
    "invalid": MUTED,
}
# The checkpoints the bar panels show, named once so the figure and the audit
# record cannot drift apart. The 3B cohorts evaluate on a half-pass grid; this
# subset keeps the requested early checkpoints and the first 32/32 endpoint.
STEPS_PER_PASS = 192
DISPLAY_STEPS = [0, 96, 288, 384, 768, 864, 1152, 1248]
END_PASS = DISPLAY_STEPS[-1] / STEPS_PER_PASS

# A warm neutral ramp, not the previous cool blue-gray one: the old
# 1/2/3 steps (#E8EDF2/#77838F/#17212B) all sat at the same ~210 degree hue as
# PANEL, INVALID, and the sequential magnitude ramp, so this figure's one
# achromatic scale was quietly the same colour as three other things it has
# nothing to do with. Warm sepia keeps the same lightness steps -- so the
# ramp still orders the same way in grayscale and reads the same size of
# jump between paints -- while stopping the vertex scale from reading as
# "more blue" anywhere in the figure. Paint 1 still stays clear of both the
# white "uncoloured" node and the near-white invalid segment above: distinct
# by hue (warm vs. INVALID's cool gray-blue), the same way the original ramp
# relied on hue rather than on a passing contrast ratio between two very
# pale fills.
NODE_COLORS = {1: "#EDE8DE", 2: "#836D54", 3: "#2C231C"}
VERTEX_EDGE = "#4A3F33"
NODE_TEXT = {1: INK, 2: WHITE, 3: WHITE}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",
            "Helvetica",
            "Arial",
            "Liberation Sans",
        ],
        "font.size": FONT,
        "axes.titlesize": FONT,
        "axes.labelsize": FONT,
        "xtick.labelsize": FONT,
        "ytick.labelsize": FONT,
        "legend.fontsize": FONT,
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": FRAME,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.facecolor": "none",
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded_box(ax, x, y, w, h, *, face=WHITE, edge=GRID, radius=0.018, lw=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def panel_label(ax, letter: str, title: str) -> None:
    """Letter plus title on one baseline, offset in points so the gap between
    them is the same in every panel regardless of how wide the panel is."""

    # Both anchor their *bottom* at the same height, so a title wrapped onto
    # two lines grows upward and stays clear of the panel contents instead of
    # dropping its second line onto them. The letter is then lifted by however
    # many extra lines the title has, in points, so it sits level with the
    # title's first line rather than its last.
    lines = title.count("\n") + 1
    lift = (lines - 1) * 1.25 * FONT
    for text, x_offset, y_offset in (
        (letter, 0.0, lift),
        (title, 1.4 * FONT, 0.0),
    ):
        ax.text(
            0.0,
            1.045,
            text,
            transform=offset_copy(
                ax.transAxes, fig=ax.figure, x=x_offset, y=y_offset, units="points"
            ),
            fontsize=FONT,
            fontweight="bold",
            color=INK,
            ha="left",
            va="bottom",
            linespacing=1.25,
        )


def load_snapshots(path: Path) -> tuple[dict[int, dict[int, dict]], dict[int, dict]]:
    records: dict[tuple[int, int], dict] = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["evaluation_kind"] != "fixed_seed_sampled_k_neutral":
                continue
            assert record["sample_count"] == 8
            records[(int(record["step"]), int(record["draw_index"]))] = record

    by_step: dict[int, dict[int, dict]] = {}
    metadata: dict[int, dict] = {}
    for (step, draw_index), record in sorted(records.items()):
        assert draw_index in range(4)
        by_step.setdefault(step, {})[draw_index] = record
        for prompt in record["prompts"]:
            metadata[int(prompt["prompt_index"])] = prompt
    for step, draws in by_step.items():
        assert sorted(draws) == [0, 1, 2, 3], (path, step, sorted(draws))
    return by_step, metadata


def correct_counts(draws: dict[int, dict], prompt_index: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for draw_index in range(4):
        prompt = draws[draw_index]["prompts"][prompt_index]
        for key, reward in zip(prompt["answer_keys"], prompt["rewards"]):
            if float(reward) > 0 and key is not None:
                counts[str(key).split(":")[-1]] += 1
    return counts


def pass_at_8(draws: dict[int, dict], prompt_index: int) -> float:
    successes = 0
    for draw_index in range(4):
        prompt = draws[draw_index]["prompts"][prompt_index]
        successes += int(any(float(reward) > 0 for reward in prompt["rewards"]))
    return successes / 4


def select_prompt(
    pairs: dict[int, dict[str, dict[int, dict[int, dict]]]],
) -> tuple[int, int]:
    start, displayed_endpoint = 0, 1248
    candidates = []
    for seed, snapshots in sorted(pairs.items()):
        control = snapshots["drgrpo"]
        replay_maxrl = snapshots["replay_maxrl"]
        if not (
            start in control
            and start in replay_maxrl
            and displayed_endpoint in control
            and displayed_endpoint in replay_maxrl
        ):
            continue
        prompt_count = len(control[start][0]["prompts"])
        for prompt_index in range(prompt_count):
            initial_control = correct_counts(control[start], prompt_index)
            initial_replay = correct_counts(replay_maxrl[start], prompt_index)
            control_final = correct_counts(control[displayed_endpoint], prompt_index)
            replay_final = correct_counts(
                replay_maxrl[displayed_endpoint], prompt_index
            )
            reference = json.loads(
                control[start][0]["prompts"][prompt_index]["reference"]
            )
            if (
                initial_control == initial_replay
                and len(initial_control) == 3
                and sum(initial_control.values()) < 20
                and pass_at_8(control[displayed_endpoint], prompt_index) == 1
                and pass_at_8(replay_maxrl[displayed_endpoint], prompt_index) == 1
                and len(control_final) == 1
                and sum(replay_final.values()) == 32
                and min(
                    len(correct_counts(replay_maxrl[step], prompt_index))
                    for step in DISPLAY_STEPS
                ) >= 3
                and len(replay_final) > len(control_final)
            ):
                candidates.append(
                    (
                        min(initial_control.values()),
                        min(
                            sum(control_final.values()),
                            sum(replay_final.values()),
                        ),
                        len(replay_final),
                        sum(initial_control.values()),
                        -seed,
                        -prompt_index,
                        seed,
                        prompt_index,
                    )
                )
    assert candidates
    seed, prompt_index = max(candidates)[-2:]
    # Freeze the mechanically selected illustration so source changes fail loudly.
    assert (seed, prompt_index) == (70, 84), (seed, prompt_index)
    return seed, prompt_index


def parse_reference(prompt: dict) -> dict:
    reference = json.loads(prompt["reference"])
    assert reference["verifier"] == "graph_coloring"
    assert reference["num_completions"] == 6
    return reference


def assert_valid_coloring(key: str, reference: dict) -> None:
    colors = [int(color) for color in key]
    assert len(colors) == len(reference["partial_colors"])
    for vertex, fixed_color in enumerate(reference["partial_colors"]):
        if fixed_color is not None:
            assert colors[vertex] == fixed_color
    for left, right in reference["edges"]:
        assert colors[left - 1] != colors[right - 1]


def enumerate_valid_colorings(reference: dict) -> list[str]:
    """Every completion the verifier accepts, in lexicographic order.

    Panel A names three of these Option A/B/C and then jumps to the final
    letter. They are counted from the instance rather than asserted from the
    prompt text, and checked against the verifier's own ``num_completions``.
    """

    fixed = reference["partial_colors"]
    keys: list[str] = []
    for assignment in itertools.product("123", repeat=len(fixed)):
        colors = [int(color) for color in assignment]
        if any(
            value is not None and colors[vertex] != value
            for vertex, value in enumerate(fixed)
        ):
            continue
        if any(colors[a - 1] == colors[b - 1] for a, b in reference["edges"]):
            continue
        keys.append("".join(assignment))
    assert len(keys) == reference["num_completions"], (
        f"enumerated {len(keys)} colorings but the verifier reports "
        f"{reference['num_completions']}"
    )
    return keys


def last_option(reference: dict, shown: tuple[str, ...]) -> str:
    """The last valid coloring displayed at the bottom of panel A.

    Options A/B/C are the three modes the bar panels track, so they take the
    first three letters. The omitted modes take the intervening letters; the
    final row is a real accepted coloring, not a placeholder.
    """

    rest = [key for key in enumerate_valid_colorings(reference) if key not in shown]
    assert len(rest) == reference["num_completions"] - len(shown)
    return rest[-1]


# Vertex geometry for panel A, in the graph inset's own axes fraction. The
# prompt graph is two components -- the 4--1--5 path and the isolated 2--3
# edge -- and the layout now says so: one component per row, the lower row
# offset to sit under the upper row's edge midpoints. The earlier hand-placed
# scatter met node 1 with three edges at nearly the same angle and pushed
# node 5 off on its own, which read as an accident rather than as structure.
GRAPH_POSITIONS = {
    4: (0.12, 0.775),
    1: (0.50, 0.775),
    5: (0.88, 0.775),
    2: (0.31, 0.225),
    3: (0.69, 0.225),
}
# Points^2, as ``scatter`` reads it: a 30pt disc, slightly smaller than the
# old 1000 so the two rows keep air between them.
GRAPH_NODE_AREA = 900.0
GRAPH_EDGE_INK = "#8C9AA8"
# A blank vertex is the thing the prompt asks the model to fill in, so it wears
# a dashed ring; the two pre-painted vertices wear the solid vertex-scale
# outline. The distinction is now visible in the drawing instead of resting on
# "white means unassigned" alone.
GRAPH_BLANK_EDGE = "#8FA0B4"
# Printed air between an edge end and the ring it runs into. Edges are trimmed
# to this instead of being drawn under the discs, so every edge ends on a
# visible gap at the same distance from every node.
GRAPH_EDGE_GAP_PT = 4.0


def _edge_endpoints(ax, start, end, radius_pt: float, gap_pt: float):
    """The edge shortened at both ends to clear the two node discs.

    Worked in display pixels and converted back, so the trim is the same
    printed length on both axes even though the inset is far wider than it
    is tall, and is independent of the DPI each output format is saved at.
    """

    to_display = ax.transData.transform
    to_data = ax.transData.inverted().transform
    first = np.asarray(to_display(start), dtype=float)
    last = np.asarray(to_display(end), dtype=float)
    span = last - first
    length = float(np.hypot(*span))
    inset = (radius_pt + gap_pt) * ax.figure.dpi / 72.0
    if length <= 2.0 * inset:
        return None
    unit = span / length
    return np.array([to_data(first + unit * inset), to_data(last - unit * inset)])


def draw_partial_graph(ax, reference: dict) -> None:
    positions = GRAPH_POSITIONS
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    radius_pt = math.sqrt(GRAPH_NODE_AREA) / 2.0
    for left, right in reference["edges"]:
        segment = _edge_endpoints(
            ax, positions[left], positions[right], radius_pt, GRAPH_EDGE_GAP_PT
        )
        if segment is None:  # pragma: no cover - layout guard
            raise AssertionError(f"edge {left}-{right} is shorter than its nodes")
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=GRAPH_EDGE_INK,
            lw=2.0,
            solid_capstyle="round",
            clip_on=False,
            zorder=1,
        )
    for vertex, color in enumerate(reference["partial_colors"], start=1):
        x, y = positions[vertex]
        blank = color is None
        face = WHITE if blank else NODE_COLORS[color]
        ax.scatter(
            [x],
            [y],
            s=GRAPH_NODE_AREA,
            facecolor=face,
            edgecolor=GRAPH_BLANK_EDGE if blank else VERTEX_EDGE,
            lw=1.4,
            linestyle=(0, (2.4, 1.9)) if blank else "solid",
            clip_on=False,
            zorder=2,
        )
        ax.text(
            x,
            y,
            str(vertex),
            ha="center",
            va="center",
            fontsize=FONT,
            fontweight="bold",
            color=MUTED if blank else NODE_TEXT[color],
            zorder=3,
        )


def make_card_axes(fig):
    """An inch-scaled axes behind everything, for drawing the group cards.

    Inch coordinates rather than figure fractions so a corner radius is the
    same 0.09in in both directions instead of following the figure's aspect.
    """

    width, height = fig.get_size_inches()
    card = fig.add_axes([0, 0, 1, 1], zorder=-1)
    card.set_xlim(0, width)
    card.set_ylim(0, height)
    card.set_axis_off()
    return card


def group_extent(fig, artists) -> Bbox:
    """Tight bounds of a group of artists, in inches on the canvas."""
    boxes = []
    for artist in artists:
        try:
            box = artist.get_tightbbox(fig.canvas.get_renderer())
        except TypeError:  # legends take no renderer argument on some versions
            box = artist.get_window_extent()
        if box is not None:
            boxes.append(box.transformed(fig.dpi_scale_trans.inverted()))
    return Bbox.union(boxes)


def draw_group_card(
    card,
    extent: Bbox,
    *,
    top: float,
    bottom: float,
    pad: float = 0.17,
    face: str = PANEL,
):
    """One rounded pale-blue card behind a group of panels.

    The figure carries two of these --- the prompt panel is one object and the
    two trajectory panels with their shared key are another --- rather than a
    single wash behind everything, so the grouping is visible before any label
    is read. ``top`` and ``bottom`` are supplied by the caller and shared by
    both cards, so the two are exactly the same height however deep their
    contents happen to run; only their widths follow their contents.
    """

    patch = FancyBboxPatch(
        (extent.x0 - pad, bottom),
        extent.width + 2 * pad,
        top - bottom,
        boxstyle="round,pad=0.0,rounding_size=0.09",
        facecolor=face,
        edgecolor=GRID,
        linewidth=1.1,
        clip_on=False,
        zorder=-1,
    )
    card.add_patch(patch)
    return patch


def draw_split_card(
    card,
    extent: Bbox,
    seam_x: float,
    *,
    top: float,
    bottom: float,
    pad: float = 0.17,
    face_left: str = PANEL,
    face_right: str = PANEL,
):
    """One card, one outline, two fills either side of ``seam_x``.

    Panels B and C are still one grouped object -- same outline, same
    "touching" adjacency as the single-colour card this replaces -- so
    nothing about their spacing changes; only the fill now marks which half
    is which. The outline is a third, unfilled patch drawn last so the seam
    between the two fills can never nick it: an outline that is itself half
    of the coloured fill (rather than repeated whole around each colour)
    always risks the second fill's edge sitting fractionally over the first
    fill's own stroke.
    """

    box = (extent.x0 - pad, bottom, extent.width + 2 * pad, top - bottom)
    base = FancyBboxPatch(
        box[:2], box[2], box[3],
        boxstyle="round,pad=0.0,rounding_size=0.09",
        facecolor=face_right, edgecolor="none", clip_on=False, zorder=-2,
    )
    card.add_patch(base)
    left = FancyBboxPatch(
        (box[0], box[1]), seam_x - box[0], box[3],
        boxstyle="square,pad=0.0",
        facecolor=face_left, edgecolor="none", clip_on=False, zorder=-2,
    )
    left.set_clip_path(base)
    card.add_patch(left)
    outline = FancyBboxPatch(
        box[:2], box[2], box[3],
        boxstyle="round,pad=0.0,rounding_size=0.09",
        facecolor="none", edgecolor=GRID, linewidth=1.1, clip_on=False, zorder=-1,
    )
    card.add_patch(outline)
    return outline


def draw_option_row(ax, y: float, label: str, key: str, color: str | None = None) -> None:
    # The option name carries its own series colour, so a row in panel A and
    # its segment in panels B and C are joined by colour as well as by name.
    # The circles beside it stay on the achromatic vertex-colour scale: they
    # are assignments within a mode, not mode identities.
    ax.text(
        0.0,
        y,
        label,
        transform=ax.transAxes,
        fontsize=FONT,
        fontweight="bold",
        color=color if color is not None else MODE_COLORS[key],
        ha="left",
        va="center",
    )
    for index, digit in enumerate(key):
        # A wider pitch plus smaller markers keeps the printed bubbles apart.
        x = 0.350 + index * 0.145
        ax.scatter(
            [x],
            [y],
            transform=ax.transAxes,
            s=420,
            facecolor=NODE_COLORS[int(digit)],
            edgecolor=VERTEX_EDGE,
            lw=1.0,
            clip_on=False,
            zorder=3,
        )
        ax.text(
            x,
            y,
            str(index + 1),
            transform=ax.transAxes,
            fontsize=FONT,
            fontweight="bold",
            color=NODE_TEXT[int(digit)],
            ha="center",
            va="center",
            zorder=4,
        )


# Wrapped to three short lines rather than two long ones: panel A is about
# 3.8in wide on the canvas, and the two-line form ran past its own card and
# over the neighbouring one.
PROMPT_QUESTION = (
    "Color the three blank nodes;\n"
    "connected nodes must differ."
)


def render_prompt_panel(ax, reference: dict) -> None:
    ax.set_axis_off()
    # The question is panel A's own title now rather than a banner over all
    # three panels: it describes the prompt, which is what A shows, and B and C
    # are trajectories of two methods answering it.
    panel_label(ax, "A", PROMPT_QUESTION)
    # The graph starts below 1.0 rather than at it, so the three-line question
    # above has clear air under it instead of sitting on the top node.
    graph_ax = ax.inset_axes([-0.008, 0.487, 0.70, 0.415])
    draw_partial_graph(graph_ax, reference)
    ax.text(
        0.0,
        0.390,
        "CIRCLE FILL = VERTEX COLOR",
        transform=ax.transAxes,
        fontsize=FONT,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
    )
    ax.text(
        0.0,
        0.305,
        f"OPTION LABEL = SOLUTION MODE ({reference['num_completions']} VALID)",
        transform=ax.transAxes,
        fontsize=FONT * 0.78,
        fontweight="normal",
        color=MUTED,
        ha="left",
        va="center",
    )
    options = [
        ("Option A", "31212", None),
        ("Option B", "33112", None),
        ("Option C", "31312", None),
    ]
    # The count is carried by the rows themselves --- three named, an ellipsis,
    # then the twelfth --- instead of by a "12 valid solutions" callout. Option
    # L is one of the nine the figure does not draw, so it wears the same
    # colour those nine wear in the bar panels: "other valid".
    last_key = last_option(reference, tuple(key for _, key, _ in options))
    last_label = f"Option {chr(ord('A') + reference['num_completions'] - 1)}"
    assert_valid_coloring(last_key, reference)
    for y, (label, key, color) in zip((0.170, 0.010, -0.150), options):
        draw_option_row(ax, y, label, key, color)
    ax.text(
        0.055,
        -0.275,
        "⋮",
        transform=ax.transAxes,
        fontsize=FONT * 1.15,
        fontweight="bold",
        color=MUTED,
        ha="center",
        va="center",
    )
    draw_option_row(ax, -0.405, last_label, last_key, OTHER)





def render_method_trajectory(
    ax,
    snapshots: dict[int, dict[int, dict]],
    prompt_index: int,
    *,
    letter: str,
    title: str,
    show_ylabel: bool,
) -> dict:
    panel_label(ax, letter, title)
    steps = DISPLAY_STEPS
    centers = np.arange(len(steps), dtype=float)
    # Stacked bottom-up in legend order so the reading order of the stack and
    # the reading order of the legend agree.
    keys = list(MODE_COLORS)
    counts_by_step = [correct_counts(snapshots[step], prompt_index) for step in steps]
    audit = {
        str(step): {
            "training_pass": step / STEPS_PER_PASS,
            "correct": sum(counts.values()),
            "distinct": len(counts),
            "counts": dict(sorted(counts.items())),
        }
        for step, counts in zip(steps, counts_by_step)
    }
    bottoms = np.zeros(len(steps))
    for key in keys:
        values = np.array([counts.get(key, 0) for counts in counts_by_step])
        ax.bar(centers, values, width=0.60, bottom=bottoms,
               color=MODE_COLORS[key], edgecolor=WHITE, linewidth=1.1)
        bottoms += values
    other = np.array([sum(value for mode, value in counts.items() if mode not in keys)
                      for counts in counts_by_step])
    ax.bar(centers, other, width=0.60, bottom=bottoms,
           color=OTHER, edgecolor=WHITE, linewidth=1.1)
    bottoms += other
    ax.bar(centers, 32 - bottoms, width=0.60, bottom=bottoms,
           color=INVALID, edgecolor=WHITE, linewidth=1.1)
    ax.bar(centers, np.full(len(steps), 32), width=0.60, bottom=0,
           facecolor="none", edgecolor=FRAME, linewidth=0.9, zorder=4)
    for x, counts in zip(centers, counts_by_step):
        ax.text(x, 33.2, str(len(counts)), fontsize=FONT, fontweight="bold",
                color=INK, ha="center", va="bottom")
    ax.set_ylim(0, 37.6)
    ax.set_xlim(-0.55, len(steps) - 0.45)
    # Labelled from the same list the bars are built from. These were a
    # hardcoded copy and silently kept naming the old checkpoints when the
    # window moved.
    ax.set_xticks(centers, [f"{step / STEPS_PER_PASS:g}" for step in steps])
    ax.set_xlabel(f"training pass (of {END_PASS:g})", labelpad=4)
    ax.set_yticks([0, 8, 16, 24, 32])
    if show_ylabel:
        ax.set_ylabel("fixed-seed samples (of 32)", labelpad=4)
    else:
        ax.set_yticklabels([])
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.0, width=0.8, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    return audit

def main() -> None:
    pairs = {}
    metadata = {}
    for seed, paths in sorted(PAIR_DRAWS.items()):
        control, control_meta = load_snapshots(paths["drgrpo"])
        replay, replay_meta = load_snapshots(paths["replay_maxrl"])
        assert control_meta.keys() == replay_meta.keys()
        pairs[seed] = {"drgrpo": control, "replay_maxrl": replay}
        metadata[seed] = control_meta
    seed, prompt_index = select_prompt(pairs)
    control = pairs[seed]["drgrpo"]
    replay = pairs[seed]["replay_maxrl"]
    control_meta = metadata[seed]
    reference = parse_reference(control_meta[prompt_index])
    displayed_keys = set()
    for snapshots in (control, replay):
        for records in snapshots.values():
            displayed_keys.update(correct_counts(records, prompt_index))
    for key in displayed_keys:
        assert_valid_coloring(key, reference)

    initial_control = correct_counts(control[0], prompt_index)
    initial_replay = correct_counts(replay[0], prompt_index)
    assert initial_control == initial_replay
    assert initial_control == Counter({"31212": 6, "33112": 8, "31312": 2})

    # Drawn at the full text width, so the canvas is wide relative to its
    # height and every element keeps its printed font size while gaining room.
    fig = plt.figure(figsize=(CANVAS_WIDTH, CANVAS_HEIGHT))
    card_axes = make_card_axes(fig)
    # The question moved into panel A's title, so there is no figure-wide
    # banner and the panels start higher.
    # A compact gutter separates the task/example card from the paired
    # training-trajectory card without making the two stories feel detached.
    ratios = [4.05, 1.65, 4.02, 0.17, 4.02]
    # The right edge leaves room for the B/C card's 0.17in pad, which at 0.992
    # pushed the card past the canvas and clipped it.
    grid_left, grid_right = 0.014, 0.987
    grid = fig.add_gridspec(
        1, 5, width_ratios=ratios,
        left=grid_left, right=grid_right, top=0.815, bottom=0.330, wspace=0.0,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in (0, 2, 4)]
    # Figure-fraction edges of the B/C block, derived from the gridspec rather
    # than restated, so the key stays centred on it if the ratios ever change.
    unit = (grid_right - grid_left) / sum(ratios)
    BC_LEFT = grid_left + sum(ratios[:2]) * unit
    BC_RIGHT = grid_right
    render_prompt_panel(axes[0], reference)
    control_trajectory = render_method_trajectory(
        axes[1], control, prompt_index, letter="B",
        title="Qwen2.5-3B\nDr.GRPO",
        show_ylabel=True,
    )
    replay_trajectory = render_method_trajectory(
        axes[2], replay, prompt_index, letter="C",
        title="Qwen2.5-3B\nRe:Max (ours)",
        show_ylabel=False,
    )

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color,
               markeredgecolor="none", markersize=11, label=label)
        for label, color in [
            ("mode A", MODE_COLORS["31212"]),
            ("mode B", MODE_COLORS["33112"]),
            ("mode C", MODE_COLORS["31312"]),
            ("other valid", OTHER),
            ("invalid", INVALID),
        ]
    ]
    # The key describes the bar segments in B and C only --- panel A has no
    # stacked bars --- so it is centred on the B/C block rather than on the
    # whole figure, and lives inside their card.
    bc_centre = (BC_LEFT + BC_RIGHT) / 2
    fig.legend(
        handles=legend,
        loc="lower center",
        # Tucked up against the bar panels: at 0.004 it sat a visible band
        # below them and read as a separate object rather than their key.
        bbox_to_anchor=(bc_centre, 0.070),
        ncol=5,
        frameon=False,
        fontsize=FONT,
        handletextpad=0.45,
        columnspacing=1.4,
    )
    for entry in fig.legends[-1].get_texts():
        entry.set_color(LABEL_COLORS[entry.get_text()])
        entry.set_fontweight("bold")

    # Every extent below is measured after a draw rather than guessed: the
    # legend's height in particular is only known once it is laid out, and it
    # sets how far the B/C card has to reach.
    fig.canvas.draw()
    keys = fig.legends[-1]
    width, height = fig.get_size_inches()
    # Centring the key on the B/C block can push it off the canvas, because its
    # width is set by its five labels and not by the block. Measure it and pull
    # it back inside if it overhangs; anything drawn past the canvas edge is
    # simply not rendered.
    key_box = keys.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    overhang = max(0.0, key_box.x1 - (width - 0.10)) - max(0.0, 0.10 - key_box.x0)
    if abs(overhang) > 1e-3:
        keys.set_bbox_to_anchor((bc_centre - overhang / width, 0.070))
        fig.canvas.draw()
    # Trim to the legend's *text*, not its bounding box: the box carries the
    # legend's internal padding, which put an extra 0.07in of wash under the
    # visible labels on top of whatever margin was asked for.
    label_bottom = min(
        text.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).y0
        for text in keys.get_texts()
    )
    # Panel A's contents sit higher than the bar panels' --- its three-line
    # question reaches further above the axes than their one-line titles, and
    # its last option row stops above their key. Centre the prompt axes on the
    # bar block before the cards are drawn, so one rectangle can hug both
    # tightly instead of the pair being level in height but offset on the page.
    prompt_extent = group_extent(fig, [axes[0]])
    bars_probe = group_extent(fig, [axes[1], axes[2], *keys.get_texts()])
    drop = ((prompt_extent.y0 + prompt_extent.y1) - (bars_probe.y0 + bars_probe.y1)) / 2
    if abs(drop) > 0.01:
        box = axes[0].get_position()
        axes[0].set_position(
            [box.x0, box.y0 - drop / height, box.width, box.height]
        )
        fig.canvas.draw()
    prompt_extent = group_extent(fig, [axes[0]])
    # The key belongs to the bar panels, so their card has to hold it: it sets
    # the card's width whenever the five labels run wider than the two panels,
    # and its measured text bottom sets how far the card reaches down.
    bars_extent = group_extent(fig, [axes[1], axes[2], *keys.get_texts()])
    bars_extent = Bbox.from_extents(
        bars_extent.x0,
        min(bars_extent.y0, label_bottom),
        bars_extent.x1,
        bars_extent.y1,
    )
    # Each card hugs its own contents, then the shorter of the two is grown
    # upward until the heights match. Forcing a single top *and* bottom across
    # both instead left the bar card with a band of empty wash above it (the
    # prompt's three-line question sets the top) and the prompt card with the
    # same band below it (the key sets the bottom). Equal height was the
    # requirement; a shared baseline was not.
    # With the two blocks centred on each other, one shared rectangle hugs both
    # without leaving a band above the bars or below the prompt.
    pad = 0.17
    card_top = max(prompt_extent.y1, bars_extent.y1) + pad
    card_bottom = max(0.0, min(prompt_extent.y0, bars_extent.y0) - pad)
    a_top = b_top = card_top
    a_bottom = b_bottom = card_bottom
    a_card = draw_group_card(
        card_axes,
        prompt_extent,
        top=card_top,
        bottom=card_bottom,
        face=CARD_A,
    )
    # The seam sits in the middle of B and C's own gutter (not the legend's,
    # which spans both): B and C keep exactly the gap the gridspec already
    # gives them, only the fill either side of its midpoint now differs.
    b_extent = group_extent(fig, [axes[1]])
    c_extent = group_extent(fig, [axes[2]])
    seam_x = (b_extent.x1 + c_extent.x0) / 2
    draw_split_card(
        card_axes, bars_extent, seam_x,
        top=card_top, bottom=card_bottom, face_left=CARD_B, face_right=CARD_C,
    )
    # The arrow occupies only the white gutter: it both reinforces the split
    # and reads as "task/solution modes -> observed training trajectories."
    gutter_left = a_card.get_x() + a_card.get_width()
    gutter_right = bars_extent.x0 - pad
    card_axes.text(
        (gutter_left + gutter_right) / 2,
        (card_top + card_bottom) / 2,
        "→",
        fontsize=FONT * 1.55,
        fontweight="bold",
        color=MUTED,
        ha="center",
        va="center",
    )

    if os.environ.get("STORY_LAYOUT_DEBUG"):
        gap = (bars_extent.x0 - 0.17) - (prompt_extent.x1 + 0.17)
        print(
            f"[layout] canvas={width:.2f}x{height:.2f}in\n"
            f"[layout] A card   x {prompt_extent.x0 - 0.17:6.2f} .."
            f" {prompt_extent.x1 + 0.17:6.2f}\n"
            f"[layout] BC card  x {bars_extent.x0 - 0.17:6.2f} .."
            f" {bars_extent.x1 + 0.17:6.2f}\n"
            f"[layout] gap between cards = {gap:.3f}in\n"
            f"[layout] A content y {prompt_extent.y0:.2f} .. {prompt_extent.y1:.2f}"
            f"  (h {prompt_extent.height:.2f})\n"
            f"[layout] BC content y {bars_extent.y0:.2f} .. {bars_extent.y1:.2f}"
            f"  (h {bars_extent.height:.2f})\n"
            f"[layout] A card   y {a_bottom:.2f} .. {a_top:.2f}"
            f"  (h {a_top - a_bottom:.2f})\n"
            f"[layout] BC card  y {b_bottom:.2f} .. {b_top:.2f}"
            f"  (h {b_top - b_bottom:.2f})",
            file=sys.stderr,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Crop to the cards themselves rather than the whole canvas, so removing the
    # figure-wide banner does not leave a band of white above them.
    crop = Bbox.from_extents(
        0.0, card_bottom - 0.012, width, min(height, card_top + 0.05)
    )
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches=crop, pad_inches=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=260, bbox_inches=crop, pad_inches=0.035)
    plt.close(fig)

    audit = {
        "schema": "paper_graph_collapse_toy_v26",
        "model": "Qwen2.5-3B-Instruct",
        "seed": seed,
        "layout_contract": {
            "panels": ["A", "B", "C"],
            "panel_titles": [
                PROMPT_QUESTION.replace("\n", " "),
                "Qwen2.5-3B Dr.GRPO",
                "Qwen2.5-3B Re:Max",
            ],
            "steps": DISPLAY_STEPS,
            "end_pass": END_PASS,
            "paired_bars": False,
            "grouping": "warm task card -> blue trajectory card",
            "color_encodings": {
                "panel_a_circle_fill": "vertex_color",
                "panels_bc_bar_fill": "solution_mode",
            },
        },
        "sampling": {
            "temperature": 1.0,
            "draws_per_checkpoint": 4,
            "samples_per_draw": 8,
            "samples_per_checkpoint": 32,
        },
        "selection": {
            "status": "post_hoc_illustration",
            "rule": (
                "Across available Qwen2.5-3B seeds 70--73, among Graph "
                "Coloring prompts whose paired step-0 correct samples match "
                "with exactly three modes and fewer than 20/32 correct, "
                "require all four terminal K=8 draws to succeed in both arms, "
                "a singleton terminal Dr.GRPO endpoint, a 32/32 Re:Max "
                "endpoint, and no fewer than three Re:Max "
                "modes at any displayed checkpoint; maximize "
                "the least frequent initial mode, then the smaller terminal "
                "correct-sample count, Re:Max terminal modes, and initial "
                "correct samples, then choose the lowest seed and prompt index."
            ),
            "selection_step": 3072,
            "selection_pass": 8.0,
            "prompt_index": prompt_index,
        },
        "prompt": {
            "instance_id": reference["instance_id"],
            "edges": reference["edges"],
            "partial_colors": reference["partial_colors"],
            "verifier_valid_completions": reference["num_completions"],
        },
        "initial": {
            "step": 0,
            "correct": sum(initial_control.values()),
            "distinct": len(initial_control),
            "counts": dict(sorted(initial_control.items())),
        },
        "terminal_pass8": {
            "drgrpo": pass_at_8(control[DISPLAY_STEPS[-1]], prompt_index),
            "replay_maxrl": pass_at_8(
                replay[DISPLAY_STEPS[-1]], prompt_index
            ),
        },
        "drgrpo_trajectory": control_trajectory,
        "replay_maxrl_trajectory": replay_trajectory,
        "sources": {
            "drgrpo": {
                "path": str(PAIR_DRAWS[seed]["drgrpo"].relative_to(ROOT)),
                "sha256": sha256(PAIR_DRAWS[seed]["drgrpo"]),
            },
            "replay_maxrl": {
                "path": str(PAIR_DRAWS[seed]["replay_maxrl"].relative_to(ROOT)),
                "sha256": sha256(PAIR_DRAWS[seed]["replay_maxrl"]),
            },
        },
        "selection_pool_sources": {
            str(pool_seed): {
                arm: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                }
                for arm, path in paths.items()
            }
            for pool_seed, paths in sorted(PAIR_DRAWS.items())
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}.{{pdf,png}} and {AUDIT}")


if __name__ == "__main__":
    main()
