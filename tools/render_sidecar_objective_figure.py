#!/usr/bin/env python3
"""Render a paper-focused sidecar figure for Dr.GRPO vs Dr.GRPO-Explorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


RUN_ROOT_DEFAULT = Path(
    "var/artifacts/full_eval_pairs/"
    "full_eval_richsidecar_Qwen2.5-1.5B-Instruct_math_fair_mltheory_"
    "tau0p25_beta0p08_no_template_livepass8_parity_20260326_095838"
)

GRPO_COLOR = "#1f77b4"
EXPLORER_COLOR = "#ff7f0e"
TOP_CORRECT_COLOR = "#4C78A8"
OTHER_CORRECT_COLOR = "#9ec9f5"
INCORRECT_COLOR = "#d0d0d0"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _smooth(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _series(data: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in data:
        step = row.get("step")
        value = row.get(key)
        if isinstance(step, int) and isinstance(value, (int, float)):
            xs.append(step)
            ys.append(float(value))
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    parser.add_argument("--output-prefix", type=Path, default=Path("var/artifacts/plots/drgrpo_explorer_sidecar_objective"))
    parser.add_argument("--window", type=int, default=11)
    args = parser.parse_args()

    run_root = args.run_root
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    within = _load_json(run_root / "listwise_vs_grpo_within_prompt_mass.summary.json")
    prompt = _load_json(run_root / "listwise_vs_grpo_prompt_dynamics.summary.json")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.3, 1.3]},
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.24, wspace=0.14)
    ax0, ax1, ax2 = axes

    methods = ["Dr.GRPO", "Dr.GRPO-Explorer"]
    top_vals = [within["grpo"]["top1_mass_mean"], within["listwise"]["top1_mass_mean"]]
    other_vals = [within["grpo"]["mean_other_correct_mass"], within["listwise"]["mean_other_correct_mass"]]
    incorrect_vals = [within["grpo"]["mean_incorrect_mass"], within["listwise"]["mean_incorrect_mass"]]
    ypos = [1, 0]

    ax0.barh(ypos, top_vals, color=TOP_CORRECT_COLOR, height=0.36)
    ax0.barh(ypos, other_vals, left=top_vals, color=OTHER_CORRECT_COLOR, height=0.36)
    ax0.barh(
        ypos,
        incorrect_vals,
        left=[a + b for a, b in zip(top_vals, other_vals)],
        color=INCORRECT_COLOR,
        height=0.36,
    )
    ax0.set_xlim(0.0, 1.0)
    ax0.set_yticks(ypos)
    ax0.set_yticklabels(methods)
    ax0.set_xlabel("Average update mass")
    ax0.set_title("Pooled Update-Mass Composition")
    ax0.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax0.set_axisbelow(True)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    for y, t, o, i in zip(ypos, top_vals, other_vals, incorrect_vals):
        if t > 0.08:
            ax0.text(t / 2, y, f"{t:.2f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if o > 0.08:
            ax0.text(t + o / 2, y, f"{o:.2f}", ha="center", va="center", color="#17324d", fontsize=10, fontweight="bold")
        if i > 0.08:
            ax0.text(t + o + i / 2, y, f"{i:.2f}", ha="center", va="center", color="#374151", fontsize=10, fontweight="bold")

    def draw_line_panel(ax, title: str, ylabel: str, key: str, ylim: tuple[float, float]) -> None:
        gx, gy = _series(prompt["grpo"], key)
        lx, ly = _series(prompt["listwise"], key)
        ax.plot(gx, gy, color=GRPO_COLOR, alpha=0.22, linewidth=1.6)
        ax.plot(lx, ly, color=EXPLORER_COLOR, alpha=0.22, linewidth=1.6)
        ax.plot(gx, _smooth(gy, args.window), color=GRPO_COLOR, linewidth=2.8)
        ax.plot(lx, _smooth(ly, args.window), color=EXPLORER_COLOR, linewidth=2.8)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    draw_line_panel(ax1, "Effective Active Rollouts", "exp(entropy)", "mean_effective_rollouts", (1.0, 8.15))
    draw_line_panel(ax2, "Incorrect-Mass Share", "Mass on incorrect rollouts", "mean_incorrect_mass", (0.0, 0.52))

    fig.suptitle(
        "How Groupwise Exploration Changes Prompt-Local Update Behavior",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    method_handles = [
        Line2D([0], [0], color=GRPO_COLOR, lw=2.8, label="Dr.GRPO"),
        Line2D([0], [0], color=EXPLORER_COLOR, lw=2.8, label="Dr.GRPO-Explorer"),
    ]
    fill_handles = [
        Patch(facecolor=TOP_CORRECT_COLOR, label="Top-ranked correct rollout"),
        Patch(facecolor=OTHER_CORRECT_COLOR, label="Other correct rollouts"),
        Patch(facecolor=INCORRECT_COLOR, label="Incorrect rollouts"),
    ]
    ax1.legend(handles=method_handles, loc="lower left", frameon=False)
    fig.legend(handles=fill_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.05))

    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")

    manifest = {
        "run_root": str(run_root),
        "window": args.window,
        "grpo": {
            "top1_mass_mean": within["grpo"]["top1_mass_mean"],
            "other_correct_mass_mean": within["grpo"]["mean_other_correct_mass"],
            "incorrect_mass_mean": within["grpo"]["mean_incorrect_mass"],
            "latest_effective_rollouts": prompt["grpo"][-1]["mean_effective_rollouts"],
            "latest_incorrect_mass": prompt["grpo"][-1]["mean_incorrect_mass"],
            "latest_step": prompt["grpo"][-1]["step"],
        },
        "explorer": {
            "top1_mass_mean": within["listwise"]["top1_mass_mean"],
            "other_correct_mass_mean": within["listwise"]["mean_other_correct_mass"],
            "incorrect_mass_mean": within["listwise"]["mean_incorrect_mass"],
            "latest_effective_rollouts": prompt["listwise"][-1]["mean_effective_rollouts"],
            "latest_incorrect_mass": prompt["listwise"][-1]["mean_incorrect_mass"],
            "latest_step": prompt["listwise"][-1]["step"],
        },
    }
    output_prefix.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
