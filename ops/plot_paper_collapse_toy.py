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

INK = "#19324A"
MUTED = "#607487"
GRID = "#D8E2EA"
PANEL = "#F6F9FB"
WHITE = "#FFFFFF"
INVALID = "#D8DEE5"
DR = "#C76A3A"
XDR = "#087F8C"

MODE_COLORS = {
    "31223": "#D95F45",
    "33221": "#6C5CE7",
    "32213": "#2A9D8F",
}
OTHER = "#4C78A8"
NODE_COLORS = {1: "#F2C14E", 2: "#2A9D8F", 3: "#7B6FD0"}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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


def panel_label(ax, letter: str, title: str, color: str) -> None:
    ax.text(
        0.0,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=color,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.105,
        1.04,
        title,
        transform=ax.transAxes,
        fontsize=10.2,
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
        ax.plot([x1, x2], [y1, y2], color=MUTED, lw=1.45, zorder=1)
    for vertex, color in enumerate(reference["partial_colors"], start=1):
        x, y = positions[vertex]
        face = NODE_COLORS[color] if color is not None else WHITE
        edge = INK if color is not None else MUTED
        ax.scatter(
            [x],
            [y],
            s=430,
            facecolor=face,
            edgecolor=edge,
            lw=1.35,
            clip_on=False,
            zorder=2,
        )
        ax.text(
            x,
            y,
            str(vertex),
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=WHITE if color is not None else MUTED,
            zorder=3,
        )
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 0.98)
    ax.set_axis_off()


def draw_option_row(ax, y: float, label: str, key: str) -> None:
    ax.text(
        0.02,
        y,
        label,
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=MODE_COLORS[key],
        ha="left",
        va="center",
    )
    for index, digit in enumerate(key):
        x = 0.35 + index * 0.11
        ax.scatter(
            [x],
            [y],
            transform=ax.transAxes,
            s=70,
            facecolor=NODE_COLORS[int(digit)],
            edgecolor=WHITE,
            lw=0.6,
            clip_on=False,
            zorder=3,
        )
        ax.text(
            x,
            y,
            str(index + 1),
            transform=ax.transAxes,
            fontsize=5.8,
            fontweight="bold",
            color=INK if int(digit) == 1 else WHITE,
            ha="center",
            va="center",
            zorder=4,
        )
    ax.text(
        0.97,
        y,
        "✓",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=XDR,
        ha="right",
        va="center",
    )


def render_prompt_panel(ax, reference: dict) -> None:
    ax.set_axis_off()
    panel_label(ax, "A", "One prompt, many answers", MODE_COLORS["33221"])
    # The prompt question is the figure title, so panel A spends its full
    # upper half on the graph instead of repeating the question.
    graph_ax = ax.inset_axes([0.0, 0.38, 0.63, 0.58])
    draw_partial_graph(graph_ax, reference)
    rounded_box(ax, 0.69, 0.60, 0.28, 0.21, face="#F0F7F6", edge="#B9DCD7")
    ax.text(
        0.83,
        0.705,
        "12 valid\nsolutions",
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=XDR,
        ha="center",
        va="center",
    )
    ax.text(
        0.0,
        0.30,
        "Three valid options (numbers identify vertices):",
        transform=ax.transAxes,
        fontsize=6.7,
        color=MUTED,
        ha="left",
        va="center",
    )
    options = [
        ("Option A", "33221"),
        ("Option B", "31223"),
        ("Option C", "32213"),
    ]
    for y, (label, key) in zip((0.215, 0.105, -0.005), options):
        draw_option_row(ax, y, label, key)





def render_method_trajectory(
    ax,
    snapshots: dict[int, dict[int, dict]],
    prompt_index: int,
    *,
    letter: str | None,
    title: str,
    accent: str,
    show_ylabel: bool,
) -> dict:
    if letter is not None:
        panel_label(ax, letter, title, accent)
    else:
        ax.text(
            0.0, 1.04, title, transform=ax.transAxes, fontsize=10.2,
            fontweight="bold", color=INK, ha="left", va="bottom",
        )
    steps = [0, 48, 96, 192, 384, 576, 768]
    centers = np.arange(len(steps), dtype=float)
    keys = ["31223", "33221", "32213"]
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
        ax.bar(centers, values, width=0.62, bottom=bottoms,
               color=MODE_COLORS[key], edgecolor=WHITE, linewidth=0.45)
        bottoms += values
    other = np.array([sum(value for mode, value in counts.items() if mode not in keys)
                      for counts in counts_by_step])
    ax.bar(centers, other, width=0.62, bottom=bottoms,
           color=OTHER, edgecolor=WHITE, linewidth=0.45)
    bottoms += other
    ax.bar(centers, 32 - bottoms, width=0.62, bottom=bottoms,
           color=INVALID, edgecolor=WHITE, linewidth=0.45)
    ax.bar(centers, np.full(len(steps), 32), width=0.62, bottom=0,
           facecolor="none", edgecolor=accent, linewidth=1.05, zorder=4)
    for x, counts in zip(centers, counts_by_step):
        ax.text(x, 32.7, str(len(counts)), fontsize=6.5, fontweight="bold",
                color=accent, ha="center", va="bottom")
    ax.set_ylim(0, 38.5)
    ax.set_xlim(-0.50, 6.50)
    ax.set_xticks(
        centers,
        ["Step 0", "48", "96", "192", "384", "576", "768\n(epoch 4)"],
        fontsize=6.2,
    )
    ax.set_xlabel("optimizer step", fontsize=7.0, labelpad=1)
    ax.set_yticks([0, 8, 16, 24, 32])
    if show_ylabel:
        ax.set_ylabel("fixed-seed samples (of 32)", fontsize=7.0, labelpad=1)
    else:
        ax.set_yticklabels([])
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
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
    fig = plt.figure(figsize=(9.00, 2.72))
    fig.text(
        0.51,
        0.975,
        "How can we color the three uncolored nodes so that connected nodes "
        "get different colors?",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
        ha="center",
        va="top",
    )
    grid = fig.add_gridspec(
        1, 3, width_ratios=[1.18, 1.00, 1.00],
        left=0.022, right=0.995, top=0.795, bottom=0.135, wspace=0.20,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    render_prompt_panel(axes[0], reference)
    dr_trajectory = render_method_trajectory(
        axes[1], dr, prompt_index, letter="B", title="Dr.GRPO",
        accent=DR, show_ylabel=True,
    )
    xdr_trajectory = render_method_trajectory(
        axes[2], xdr, prompt_index, letter=None, title="xGRPO",
        accent=XDR, show_ylabel=False,
    )

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color,
               markeredgecolor="none", markersize=6, label=label)
        for label, color in [
            ("Option A", MODE_COLORS["33221"]),
            ("Option B", MODE_COLORS["31223"]),
            ("Option C", MODE_COLORS["32213"]),
            ("other valid", OTHER),
            ("invalid", INVALID),
        ]
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.51, -0.075),
        ncol=5,
        frameon=False,
        fontsize=7.0,
        handletextpad=0.35,
        columnspacing=1.0,
    )

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
        "schema": "paper_graph_collapse_toy_v15",
        "model": "Qwen2.5-0.5B-Instruct",
        "seed": 43,
        "layout_contract": {
            "panels": ["A", "B"],
            "panel_B_facets": ["Dr.GRPO", "xGRPO"],
            "steps": [0, 48, 96, 192, 384, 576, 768],
            "end_epoch": 4,
            "paired_bars": False,
            "panel_C": False,
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
