#!/usr/bin/env python3
"""Render the paper's model-backed Graph Coloring collapse example.

The figure is not a simulation. It reads fixed-seed samples emitted during the
matched Qwen2.5-0.5B-Instruct Graph Coloring runs used by the paper. Duplicate
evaluation records caused by resumptions are resolved by retaining the last
record for each (step, draw_index), exactly as a checkpoint snapshot.
"""

from __future__ import annotations

import hashlib
import itertools
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

DR_DRAWS = (
    ROOT
    / "var/data/xdr_qwen25_0p5b_instruct_grpo_"
    "gce61r1_e58_vs_grpo_05b_12ep_grpo_s43/"
    "debug_job30126333/eval_mode_coverage_draws.jsonl"
)
XDR_DRAWS = (
    ROOT
    / "var/data/xdr_qwen25_0p5b_instruct_verified_first_global_replay_canonical_"
    "gce61r1_e58_vs_grpo_05b_12ep_"
    "verified_first_global_replay_canonical_s43/"
    "debug_job30126334/eval_mode_coverage_draws.jsonl"
)
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
# The pale wash every figure in the paper sits on; here it is the canvas
# itself, so the three figures read as one surface.
PANEL = style.PANEL
WHITE = style.WHITE

# Two scales share this figure and must never be confused. The *series* scale
# identifies executed answer modes and is the shared ``MODE_RAMP``; it is the
# only saturated thing in the figure, in the bar panels and their legend. The
# *paint* scale is the puzzle's own three colours in panel A. It is
# deliberately a neutral slate ramp, off the mode ramp entirely: a paint is
# part of the question, not one of the measured modes, and a reader must never
# read a node's fill as a bar's series colour.
MODE_COLORS = {
    "33221": style.MODE_RAMP[0],  # Option A
    "31223": style.MODE_RAMP[1],  # Option B
    "32213": style.MODE_RAMP[2],  # Option C
}
OTHER = style.MODE_RAMP[3]  # any further verified mode
INVALID = style.INVALID  # invalid response — off-ramp on purpose, so it recedes
# Legend labels carry their series colour. Every mode fill clears 3:1 against
# the surface under the validated ramp, so unlike the plasma ramp this replaced
# --- whose yellow sat at 1.46:1 and needed a hand-picked darker gold --- each
# label can simply wear its own series colour. Only the neutral invalid swatch
# still borrows muted ink, because it is not a series.
LABEL_COLORS = {
    "Option A": MODE_COLORS["33221"],
    "Option B": MODE_COLORS["31223"],
    "Option C": MODE_COLORS["32213"],
    "other valid": OTHER,
    "invalid": MUTED,
}
# The checkpoints the bar panels show, named once so the figure and the audit
# record cannot drift apart. Through six epochs rather than four: Dr.GRPO holds
# 32/32 on exactly one mode from step 288 to 2064 without a break, so the longer
# window shows the collapse as a plateau rather than a moment, and the historical treatment is at
# its widest at the end of it --- 29/32 across seven modes at 1152, against
# 27/32 across six at 768. Five epochs was measured and is worse than four
# (26/32, six modes), so the window skips it.
DISPLAY_STEPS = [0, 48, 96, 192, 384, 768, 1152]
END_EPOCH = DISPLAY_STEPS[-1] // 192

NODE_COLORS = {1: "#C9D6E2", 2: "#6E8599", 3: "#263D51"}  # slate: light/mid/deep
# Paint 1 stays clear of both the white "uncoloured" node and the near-white
# invalid segment above, so no fill in the figure reads as two things.
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


def select_prompt(
    dr: dict[int, dict[int, dict]],
    xdr: dict[int, dict[int, dict]],
) -> int:
    start, displayed_endpoint = 0, 384
    assert start in dr and start in xdr
    assert displayed_endpoint in dr and displayed_endpoint in xdr
    candidates = []
    prompt_count = len(dr[start][0]["prompts"])
    for prompt_index in range(prompt_count):
        initial = correct_counts(dr[start], prompt_index)
        dr_final = correct_counts(dr[displayed_endpoint], prompt_index)
        xdr_final = correct_counts(xdr[displayed_endpoint], prompt_index)
        if len(initial) >= 3 and len(dr_final) == 1 and len(xdr_final) >= 2:
            candidates.append(
                (
                    len(xdr_final),
                    sum(xdr_final.values()),
                    sum(initial.values()),
                    -prompt_index,
                    prompt_index,
                )
            )
    assert candidates
    prompt_index = max(candidates)[-1]
    # Freeze the mechanically selected illustration so source changes fail loudly.
    assert prompt_index == 94, prompt_index
    return prompt_index


def parse_reference(prompt: dict) -> dict:
    reference = json.loads(prompt["reference"])
    assert reference["verifier"] == "graph_coloring"
    assert reference["num_completions"] == 12
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

    Panel A names three of these Option A/B/C and then jumps to Option L. That
    lettering is only honest if there really are twelve, so they are counted
    here from the instance rather than asserted from the prompt text, and the
    count is checked against the verifier's own ``num_completions``.
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


def twelfth_option(reference: dict, shown: tuple[str, ...]) -> str:
    """The coloring panel A labels Option L.

    Options A/B/C are the three modes the bar panels track, so they take the
    first three letters; the nine the figure does not draw take D through L in
    lexicographic order. Option L is therefore the last of those nine --- a
    real accepted coloring, not a placeholder for one.
    """

    rest = [key for key in enumerate_valid_colorings(reference) if key not in shown]
    assert len(rest) == reference["num_completions"] - len(shown)
    return rest[-1]


def draw_partial_graph(ax, reference: dict) -> None:
    positions = {
        1: (0.12, 0.80),
        2: (0.14, 0.13),
        3: (0.49, 0.86),
        4: (0.90, 0.12),
        5: (0.83, 0.55),
    }
    for left, right in reference["edges"]:
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        ax.plot([x1, x2], [y1, y2], color="#9AA8B5", lw=1.6, zorder=1)
    for vertex, color in enumerate(reference["partial_colors"], start=1):
        x, y = positions[vertex]
        face = NODE_COLORS[color] if color is not None else WHITE
        ax.scatter(
            [x],
            [y],
            s=1000,
            facecolor=face,
            edgecolor=INK if color is not None else "#9AA8B5",
            lw=1.2,
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
            color=NODE_TEXT[color] if color is not None else MUTED,
            zorder=3,
        )
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 0.98)
    ax.set_axis_off()


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


def draw_group_card(card, extent: Bbox, *, top: float, bottom: float, pad: float = 0.17):
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
        facecolor=PANEL,
        edgecolor=GRID,
        linewidth=1.1,
        clip_on=False,
        zorder=-1,
    )
    card.add_patch(patch)
    return patch


def draw_option_row(ax, y: float, label: str, key: str, color: str | None = None) -> None:
    # The option name carries its own series colour, so a row in panel A and
    # its segment in panels B and C are joined by colour as well as by name.
    # The circles beside it stay on the slate paint scale: they are the
    # puzzle's colours, not modes.
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
        x = 0.345 + index * 0.138
        ax.scatter(
            [x],
            [y],
            transform=ax.transAxes,
            s=690,
            facecolor=NODE_COLORS[int(digit)],
            edgecolor=WHITE,
            lw=1.1,
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
    "How can we color the three\n"
    "uncolored nodes so that connected\n"
    "nodes get different colors?"
)


def render_prompt_panel(ax, reference: dict) -> None:
    ax.set_axis_off()
    # The question is panel A's own title now rather than a banner over all
    # three panels: it describes the prompt, which is what A shows, and B and C
    # are trajectories of two methods answering it.
    panel_label(ax, "A", PROMPT_QUESTION)
    # The graph starts below 1.0 rather than at it, so the three-line question
    # above has clear air under it instead of sitting on the top node.
    graph_ax = ax.inset_axes([0.159, 0.515, 0.58, 0.365])
    draw_partial_graph(graph_ax, reference)
    ax.text(
        0.0,
        0.330,
        "Twelve valid colorings",
        transform=ax.transAxes,
        fontsize=FONT,
        color=MUTED,
        ha="left",
        va="center",
    )
    options = [
        ("Option A", "33221", None),
        ("Option B", "31223", None),
        ("Option C", "32213", None),
    ]
    # The count is carried by the rows themselves --- three named, an ellipsis,
    # then the twelfth --- instead of by a "12 valid solutions" callout. Option
    # L is one of the nine the figure does not draw, so it wears the same
    # colour those nine wear in the bar panels: "other valid".
    last_key = twelfth_option(reference, tuple(key for _, key, _ in options))
    assert_valid_coloring(last_key, reference)
    for y, (label, key, color) in zip((0.200, 0.065, -0.070), options):
        draw_option_row(ax, y, label, key, color)
    ax.text(
        0.055,
        -0.190,
        "⋮",
        transform=ax.transAxes,
        fontsize=FONT * 1.15,
        fontweight="bold",
        color=MUTED,
        ha="center",
        va="center",
    )
    draw_option_row(ax, -0.320, "Option L", last_key, OTHER)





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
    keys = ["33221", "31223", "32213"]
    counts_by_step = [correct_counts(snapshots[step], prompt_index) for step in steps]
    audit = {
        str(step): {
            "epoch": step / 192,
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
    ax.set_xlim(-0.55, 6.55)
    # Labelled from the same list the bars are built from. These were a
    # hardcoded copy and silently kept naming the old checkpoints when the
    # window moved.
    ax.set_xticks(centers, [str(step) for step in steps])
    ax.set_xlabel(f"optimizer step ({END_EPOCH} epochs)", labelpad=4)
    ax.set_yticks([0, 8, 16, 24, 32])
    if show_ylabel:
        ax.set_ylabel("fixed-seed samples (of 32)", labelpad=4)
    else:
        ax.set_yticklabels([])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.0, width=0.8, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    return audit

def main() -> None:
    dr, dr_meta = load_snapshots(DR_DRAWS)
    xdr, _ = load_snapshots(XDR_DRAWS)
    prompt_index = select_prompt(dr, xdr)
    reference = parse_reference(dr_meta[prompt_index])
    displayed_keys = set()
    for snapshots in (dr, xdr):
        for records in snapshots.values():
            displayed_keys.update(correct_counts(records, prompt_index))
    for key in displayed_keys:
        assert_valid_coloring(key, reference)

    initial_dr = correct_counts(dr[0], prompt_index)
    initial_xdr = correct_counts(xdr[0], prompt_index)
    assert initial_dr == initial_xdr
    assert initial_dr == Counter({"31223": 10, "32213": 5, "33221": 4})

    # Drawn at the full text width, so the canvas is wide relative to its
    # height and every element keeps its printed font size while gaining room.
    fig = plt.figure(figsize=(CANVAS_WIDTH, CANVAS_HEIGHT))
    card_axes = make_card_axes(fig)
    # The question moved into panel A's title, so there is no figure-wide
    # banner and the panels start higher.
    ratios = [4.05, 1.32, 4.02, 0.17, 4.02]
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
    dr_trajectory = render_method_trajectory(
        axes[1], dr, prompt_index, letter="B", title="GRPO",
        show_ylabel=True,
    )
    xdr_trajectory = render_method_trajectory(
        axes[2], xdr, prompt_index, letter="C", title="historical treatment",
        show_ylabel=False,
    )

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color,
               markeredgecolor="none", markersize=11, label=label)
        for label, color in [
            ("Option A", MODE_COLORS["33221"]),
            ("Option B", MODE_COLORS["31223"]),
            ("Option C", MODE_COLORS["32213"]),
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
    draw_group_card(card_axes, prompt_extent, top=card_top, bottom=card_bottom)
    draw_group_card(card_axes, bars_extent, top=card_top, bottom=card_bottom)

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
        "schema": "paper_graph_collapse_toy_v16",
        "model": "Qwen2.5-0.5B-Instruct",
        "seed": 43,
        "layout_contract": {
            "panels": ["A", "B", "C"],
            "panel_titles": [
                PROMPT_QUESTION.replace("\n", " "),
                "GRPO",
                "historical treatment",
            ],
            "steps": DISPLAY_STEPS,
            "end_epoch": END_EPOCH,
            "paired_bars": False,
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
                "Among prompts with >=3 observed valid modes at step 0, "
                "a singleton Dr.GRPO distribution at epoch 2, and >=2 "
                "historical-treatment modes there, maximize historical-treatment epoch-2 observed "
                "modes, then historical-treatment epoch-2 correct samples, then initial "
                "correct samples, then choose the lowest prompt index."
            ),
            "selection_step": 384,
            "selection_epoch": 2.0,
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
            "correct": sum(initial_dr.values()),
            "distinct": len(initial_dr),
            "counts": dict(sorted(initial_dr.items())),
        },
        "drgrpo_trajectory": dr_trajectory,
        "xdrgrpo_trajectory": xdr_trajectory,
        "sources": {
            "drgrpo": {
                "path": str(DR_DRAWS.relative_to(ROOT)),
                "sha256": sha256(DR_DRAWS),
            },
            "xdrgrpo": {
                "path": str(XDR_DRAWS.relative_to(ROOT)),
                "sha256": sha256(XDR_DRAWS),
            },
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}.{{pdf,png}} and {AUDIT}")


if __name__ == "__main__":
    main()
