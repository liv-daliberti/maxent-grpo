#!/usr/bin/env python3
"""Render E49D's finite-action MATH quality and mechanism dashboard."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[2]
PREFIX = {
    "toy": "e49d_maximal_support_math_toy_05b_v1",
    "full": "e49d_maximal_support_math_full_05b_v1",
}
POOL = {"toy": 50, "full": 384}
COLORS = {"grpo": "#555555", "online_canonical_haarnoja": "#0072B2"}
LABELS = {
    "grpo": "Matched gated Dr.GRPO",
    "online_canonical_haarnoja": "E49D normalized canonical Haarnoja",
}


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _curve(stage: str, path: pathlib.Path) -> list[dict]:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops/exp_scaling/parse_scaling_curve.py"),
            "--stamp-prefix",
            PREFIX[stage],
            "--prompt-pool-size",
            str(POOL[stage]),
            "--num-samples",
            "16",
            "--max-training-passes",
            "3",
            "--eval-splits",
            "math",
            "--out",
            str(path),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _fraction(row, numerator: str):
    positive = _number(row.get("math_strategy_validator_positive_rows"))
    value = _number(row.get(numerator))
    return value / positive if positive and value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(PREFIX), required=True)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    output = args.out or (
        ROOT
        / "paper/figures"
        / f"e49d_maximal_support_math_{args.stage}_live.png"
    )
    curve_path = (
        ROOT
        / "var/artifacts"
        / f"e49d_maximal_support_math_{args.stage}_curve.json"
    )
    rows = _curve(args.stage, curve_path)
    by_arm = {
        arm: sorted(
            [
                row
                for row in rows
                if row.get("arm") == arm
                and row.get("split") == "math"
                and int(row.get("seed", -1)) == 45
            ],
            key=lambda row: float(row.get("training_passes") or 0),
        )
        for arm in COLORS
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    for arm, arm_rows in by_arm.items():
        x = [_number(row.get("training_passes")) for row in arm_rows]
        ax.plot(
            x,
            [_number(row.get("greedy")) for row in arm_rows],
            color=COLORS[arm],
            marker="o",
            label=f"{LABELS[arm]} greedy",
        )
        ax.plot(
            x,
            [_number(row.get("pass8")) for row in arm_rows],
            color=COLORS[arm],
            marker="s",
            linestyle="--",
            alpha=0.85,
            label=f"{LABELS[arm]} pass@8",
        )
    ax.set(title="A. Raw held-out MATH quality", xlabel="Prompt epochs", ylabel="Accuracy")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for arm, arm_rows in by_arm.items():
        ax.plot(
            [_number(row.get("training_passes")) for row in arm_rows],
            [
                _number(row.get("verified_discovery_mean_support_per_prompt"))
                for row in arm_rows
            ],
            color=COLORS[arm],
            marker="o",
            label=f"{LABELS[arm]} mean support",
        )
    treatment = by_arm["online_canonical_haarnoja"]
    ax.plot(
        [_number(row.get("training_passes")) for row in treatment],
        [
            _number(
                row.get(
                    "online_canonical_support_at_least_two_prompt_fraction"
                )
            )
            for row in treatment
        ],
        color="#009E73",
        marker="s",
        linestyle="--",
        label="Treatment prompts with ≥2 routes",
    )
    ax.axhline(2, color="#999999", linestyle=":", linewidth=1)
    ax.set(
        title="B. Audited finite strategy support",
        xlabel="Prompt epochs",
        ylabel="Mean support / fraction",
    )
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    for arm, arm_rows in by_arm.items():
        x = [_number(row.get("training_passes")) for row in arm_rows]
        ax.plot(
            x,
            [_fraction(row, "math_strategy_accepted_rows") for row in arm_rows],
            color=COLORS[arm],
            marker="o",
            label=f"{LABELS[arm]} accepted",
        )
    if treatment:
        x = [_number(row.get("training_passes")) for row in treatment]
        ax.plot(
            x,
            [
                _fraction(row, "math_strategy_rejected_contract_rows")
                for row in treatment
            ],
            color="#D55E00",
            linestyle="--",
            label="Bad/missing declared combo",
        )
        ax.plot(
            x,
            [
                (
                    (_number(row.get("math_strategy_rejected_integrity_rows")) or 0)
                    + (_number(row.get("math_strategy_rejected_ambiguous_rows")) or 0)
                )
                / _number(row.get("math_strategy_validator_positive_rows"))
                if _number(row.get("math_strategy_validator_positive_rows"))
                else None
                for row in treatment
            ],
            color="#CC79A7",
            linestyle="-.",
            label="Execution/integrity rejected",
        )
    ax.set(
        title="C. Exact execution gate",
        xlabel="Prompt epochs",
        ylabel="Fraction of answer-positive rows",
    )
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)

    x = [_number(row.get("training_passes")) for row in treatment]
    ax = axes[1, 0]
    ax.plot(
        x,
        [
            _number(row.get("online_canonical_dual_normalized_entropy_ema"))
            for row in treatment
        ],
        color=COLORS["online_canonical_haarnoja"],
        marker="o",
    )
    ax.axhline(0.8, color="#D55E00", linestyle="--", label="Target 0.80")
    ax.set(
        title="D. Normalized strategy entropy",
        xlabel="Prompt epochs",
        ylabel=r"$H(q_x)/\log |B_x^+|$ EMA",
    )
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(
        x,
        [
            _number(row.get("online_canonical_dual_next_alpha"))
            for row in treatment
        ],
        color=COLORS["online_canonical_haarnoja"],
        marker="o",
    )
    ax.axhline(0.10, color="#999999", linestyle=":", linewidth=1)
    ax.axhline(0.50, color="#999999", linestyle=":", linewidth=1)
    ax.set(
        title="E. E46 Haarnoja response",
        xlabel="Prompt epochs",
        ylabel=r"Next $\alpha$",
    )
    ax.set_ylim(0.09, 0.51)

    ax = axes[1, 2]
    for arm, arm_rows in by_arm.items():
        x_arm = [_number(row.get("training_passes")) for row in arm_rows]
        ratios = []
        for row in arm_rows:
            raw = _number(row.get("math_strategy_raw_task_reward_mean"))
            gated = _number(row.get("math_strategy_gated_task_reward_mean"))
            ratios.append(gated / raw if raw and gated is not None else None)
        ax.plot(
            x_arm,
            ratios,
            color=COLORS[arm],
            marker="o",
            label=f"{LABELS[arm]} gate retention",
        )
    ax.plot(
        x,
        [
            _number(
                row.get("online_canonical_exploration_to_task_rms_ratio")
            )
            for row in treatment
        ],
        color="#009E73",
        marker="s",
        linestyle="--",
        label="Treatment exploration/task RMS",
    )
    ax.set(
        title="F. Contract and exploration signal",
        xlabel="Prompt epochs",
        ylabel="Ratio",
    )
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)

    menu_summary = (
        ROOT
        / "var/artifacts"
        / f"e49d_math_strategy_menu_{args.stage}_v1/generation_summary.json"
    )
    menu_status = "pending"
    if menu_summary.is_file():
        menu_status = (
            "double-audit pass"
            if json.loads(menu_summary.read_text(encoding="utf-8")).get("pass")
            else "failed"
        )
    figure.suptitle(
        "E49D finite-action, execution-gated MATH — "
        f"{args.stage} — menus: {menu_status}",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    if output.suffix.lower() == ".png":
        figure.savefig(output.with_suffix(".pdf"))
    print(output)


if __name__ == "__main__":
    main()
