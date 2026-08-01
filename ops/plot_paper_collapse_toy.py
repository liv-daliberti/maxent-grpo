#!/usr/bin/env python3
"""Render the paper's model-backed Graph Coloring collapse example.

The figure is not a simulation. It reads fixed-seed samples emitted during the
matched Qwen2.5-0.5B-Instruct Graph Coloring runs used by the paper. Duplicate
evaluation records caused by resumptions are resolved by retaining the last
record for each (step, draw_index), exactly as a checkpoint snapshot.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import offset_copy

ROOT = Path(__file__).resolve().parents[1]
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

# One type size for every label in the figure, so nothing in the printed
# panel reads as a second-class annotation.
FONT = 17.0

INK = "#1B2733"
MUTED = "#5B6B7B"
GRID = "#F0DEC6"
FRAME = "#C7AE8E"
# The pale orange every figure in the paper sits on; here it is the canvas
# itself, so the three figures read as one surface.
PANEL = "#FEF4E7"
WHITE = "#FFFFFF"

# Two scales share this figure and must never be confused. The *series* scale
# identifies executed answer modes; it is `plt.cm.plasma`, quoted here as fixed
# hex so the printed figure never moves with a matplotlib release, and it is
# the only saturated thing in the figure, in the bar panels and their legend.
# The *paint* scale is the puzzle's own three colours in panel A. It is
# deliberately a neutral slate ramp, off the plasma ramp entirely: a paint is
# part of the question, not one of the measured modes, and a reader must never
# read a node's fill as a bar's series colour.
MODE_COLORS = {
    "33221": "#41049D",  # Option A — plasma 0.10
    "31223": "#BF3984",  # Option B — plasma 0.45
    "32213": "#F2844B",  # Option C — plasma 0.70
}
OTHER = "#FCCE25"  # any further verified mode — plasma 0.90
INVALID = "#E5E9ED"  # invalid response — off-ramp on purpose, so it recedes
# Legend labels carry their series colour. Two of the five fills are too light
# to set type in at print size, so those labels use the legible sibling of the
# same family: a deeper gold for the yellow swatch, muted ink for the grey one.
LABEL_COLORS = {
    "Option A": MODE_COLORS["33221"],
    "Option B": MODE_COLORS["31223"],
    "Option C": MODE_COLORS["32213"],
    "other valid": "#A17605",
    "invalid": MUTED,
}
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

    for text, transform in (
        (letter, ax.transAxes),
        (
            title,
            offset_copy(
                ax.transAxes, fig=ax.figure, x=1.4 * FONT, y=0, units="points"
            ),
        ),
    ):
        ax.text(
            0.0,
            1.045,
            text,
            transform=transform,
            fontsize=FONT,
            fontweight="bold",
            color=INK,
            ha="left",
            va="bottom",
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


def draw_canvas_card(fig) -> None:
    """The whole figure is one rounded card, matching the paper's other two.

    Drawn in an inch-scaled axes behind everything so the corner radius is the
    same 0.09in in both directions instead of following the figure's aspect.
    """

    width, height = fig.get_size_inches()
    card = fig.add_axes([0, 0, 1, 1], zorder=-1)
    card.set_xlim(0, width)
    card.set_ylim(0, height)
    card.set_axis_off()
    card.add_patch(
        FancyBboxPatch(
            (0.02, 0.02),
            width - 0.04,
            height - 0.04,
            boxstyle="round,pad=0.0,rounding_size=0.09",
            facecolor=PANEL,
            edgecolor=GRID,
            linewidth=1.1,
            clip_on=False,
        )
    )


def draw_option_row(ax, y: float, label: str, key: str) -> None:
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
        color=MODE_COLORS[key],
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


def render_prompt_panel(ax, reference: dict) -> None:
    ax.set_axis_off()
    panel_label(ax, "A", "One prompt, many answers")
    # The prompt question is the figure title, so panel A spends its full
    # upper half on the graph instead of repeating the question.
    graph_ax = ax.inset_axes([0.0, 0.53, 0.58, 0.47])
    draw_partial_graph(graph_ax, reference)
    rounded_box(ax, 0.612, 0.680, 0.345, 0.235, face=WHITE, edge=GRID)
    ax.text(
        0.7845,
        0.7975,
        "12 valid\nsolutions",
        transform=ax.transAxes,
        fontsize=FONT,
        fontweight="bold",
        color=INK,
        ha="center",
        va="center",
        linespacing=1.25,
    )
    ax.text(
        0.0,
        0.440,
        "Three valid colorings:",
        transform=ax.transAxes,
        fontsize=FONT,
        color=MUTED,
        ha="left",
        va="center",
    )
    options = [
        ("Option A", "33221"),
        ("Option B", "31223"),
        ("Option C", "32213"),
    ]
    for y, (label, key) in zip((0.310, 0.155, 0.000), options):
        draw_option_row(ax, y, label, key)





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
    steps = [0, 48, 96, 192, 384, 576, 768]
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
    ax.set_xticks(centers, ["0", "48", "96", "192", "384", "576", "768"])
    ax.set_xlabel("optimizer step (4 epochs)", labelpad=4)
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
    fig = plt.figure(figsize=(13.2, 5.60))
    draw_canvas_card(fig)
    fig.text(
        0.034,
        0.985,
        "How can we color the three uncolored nodes so that connected nodes "
        "get different colors?",
        fontsize=FONT,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    grid = fig.add_gridspec(
        1, 5, width_ratios=[4.35, 0.65, 3.74, 0.17, 3.74],
        left=0.034, right=0.992, top=0.845, bottom=0.270, wspace=0.0,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in (0, 2, 4)]
    render_prompt_panel(axes[0], reference)
    dr_trajectory = render_method_trajectory(
        axes[1], dr, prompt_index, letter="B", title="GRPO",
        show_ylabel=True,
    )
    xdr_trajectory = render_method_trajectory(
        axes[2], xdr, prompt_index, letter="C", title="x-mode GRPO",
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
    # Sits in the bottom band the gridspec reserves for it, clear of the
    # x-axis labels above.
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=5,
        frameon=False,
        fontsize=FONT,
        handletextpad=0.45,
        columnspacing=2.0,
    )
    for entry in fig.legends[-1].get_texts():
        entry.set_color(LABEL_COLORS[entry.get_text()])
        entry.set_fontweight("bold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(
        OUT.with_suffix(".png"),
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)

    audit = {
        "schema": "paper_graph_collapse_toy_v16",
        "model": "Qwen2.5-0.5B-Instruct",
        "seed": 43,
        "layout_contract": {
            "panels": ["A", "B", "C"],
            "panel_titles": [
                "One prompt, many answers",
                "GRPO",
                "x-mode GRPO",
            ],
            "steps": [0, 48, 96, 192, 384, 576, 768],
            "end_epoch": 4,
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
                "xGRPO modes there, maximize xGRPO epoch-2 observed "
                "modes, then xGRPO epoch-2 correct samples, then initial "
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
