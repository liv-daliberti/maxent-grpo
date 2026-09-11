#!/usr/bin/env python3
"""Show why the three maze environments cannot discriminate between arms.

Top row: mean # distinct correct@8 per arm. The two arms sit on top of each
other in all three domains.

Bottom row: greedy pass@1 broken out per evaluation maze, averaged over all ten
runs. This is the reason for the top row -- PointMaze and its geometry shift
leave three of four mazes unsolved by every seed of both arms for all twelve
passes, while AntMaze solves all four from initialization. Floor and ceiling
respectively; neither leaves any dynamic range for a method comparison.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/maze_domain_degeneracy"

INK = "#19324A"
MUTED = "#607487"
GRID = "#D8E2EA"
CONTROL_COLOR = "#0057A8"
TREATMENT_COLOR = "#D55E00"
# Validated all-pairs (scripts/validate_palette.js, light surface): CVD PASS,
# normal-vision floor PASS. Fixed order, never cycled.
MAZE_COLORS = ["#0057A8", "#E69F00", "#008A5A", "#CC79A7"]

ARMS = [("grpo", "matched Dr.GRPO", CONTROL_COLOR),
        ("verified_first_global_replay_canonical", "xGRPO", TREATMENT_COLOR)]
SEEDS = range(43, 48)
DOMAINS = [
    ("point_maze_stage_b_05b_12pass", "PointMaze"),
    ("point_maze_geometry_shift_stage_b_05b_12pass", "PointMaze geometry shift"),
    ("ant_maze_stage_b_05b_12pass", "AntMaze"),
]
PASSES = 12


def evals(domain: str, arm: str, seed: int) -> list[dict]:
    path = ROOT / f"var/artifacts/{domain}_{arm}_s{seed}.metrics.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return [r for r in rows if "greedy" in r]


def mean_series(runs: list[list[dict]], key: str) -> list[float]:
    """Average `key` across runs at each evaluation coordinate."""
    if not runs:
        return []
    n = min(len(r) for r in runs)
    out = []
    for i in range(n):
        vals = [r[i][key] for r in runs if key in r[i]]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.6))

    for col, (domain, label) in enumerate(DOMAINS):
        # ---- top: distinct@8 by arm -------------------------------------
        ax = axes[0][col]
        for arm, arm_label, color in ARMS:
            runs = [evals(domain, arm, s) for s in SEEDS]
            runs = [r for r in runs if r]
            series = mean_series(runs, "distinct8")
            if not series:
                continue
            x = [i * PASSES / (len(series) - 1) for i in range(len(series))]
            lo = [min(r[i]["distinct8"] for r in runs) for i in range(len(series))]
            hi = [max(r[i]["distinct8"] for r in runs) for i in range(len(series))]
            ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
            ax.plot(x, series, color=color, linewidth=2.0, label=arm_label,
                    solid_capstyle="round")
        ax.set_title(label, color=INK, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("mean # distinct correct@8" if col == 0 else "",
                      color=INK, fontsize=9)
        ax.set_ylim(bottom=0)

        # ---- bottom: per-maze greedy ------------------------------------
        ax2 = axes[1][col]
        all_runs = [evals(domain, arm, s) for arm, _, _ in ARMS for s in SEEDS]
        all_runs = [r for r in all_runs if r]
        maze_keys = sorted(
            k for k in all_runs[0][-1]
            if k.startswith("eval/") and k.endswith("/greedy")
        )
        drawn = []
        for idx, key in enumerate(maze_keys):
            name = key.split("/")[1]
            series = mean_series(all_runs, key)
            if not series:
                continue
            x = [i * PASSES / (len(series) - 1) for i in range(len(series))]
            color = MAZE_COLORS[idx % len(MAZE_COLORS)]
            ax2.plot(x, series, color=color, linewidth=2.0,
                     solid_capstyle="round")
            drawn.append((round(series[-1], 3), max(series), name, color, x[-1]))

        # Coincident lines get one shared label instead of a stack of collided
        # ones: identical endpoints mean an identical story.
        groups: dict[float, list] = {}
        for end, peak, name, color, xend in drawn:
            groups.setdefault(end, []).append((peak, name, color, xend))
        for end, members in sorted(groups.items()):
            peak = max(m[0] for m in members)
            color = members[0][2]
            xend = members[0][3]
            ax2.plot([xend], [end], marker="o", markersize=5, color=color,
                     zorder=5)
            if len(members) == 1:
                text = f" {members[0][1]}"
            elif peak <= 0:
                text = f" {len(members)} mazes — never solved,\n  any seed, either arm"
            else:
                text = f" all {len(members)} mazes"
            ax2.annotate(text, (xend, end), color=INK, fontsize=7.8,
                         va="center", ha="left", annotation_clip=False)
        ax2.set_ylabel("greedy pass@1 per maze" if col == 0 else "",
                       color=INK, fontsize=9)
        ax2.set_ylim(-0.05, 1.18)
        ax2.set_xlabel("training passes", color=INK, fontsize=9)

        for a in (ax, ax2):
            a.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_color(GRID)
            a.tick_params(colors=MUTED, labelsize=8)
            a.set_xlim(0, PASSES * 1.30)
            a.set_xticks([0, 3, 6, 9, 12])

    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left",
                      labelcolor=INK)
    fig.suptitle(
        "The three maze environments cannot separate the arms",
        color=INK, fontsize=13.5, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.935,
        "Top: the arms overlap everywhere.  Bottom: PointMaze and its geometry "
        "shift never solve three of four mazes in any seed of either arm (floor); "
        "AntMaze solves all four from initialization (ceiling).",
        ha="center", color=MUTED, fontsize=8.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.png", dpi=190, facecolor="white")
    fig.savefig(f"{OUT}.pdf", facecolor="white")
    print(f"wrote {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
