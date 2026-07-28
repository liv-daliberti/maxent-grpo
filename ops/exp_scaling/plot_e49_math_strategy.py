#!/usr/bin/env python3
"""Render the E49B matched MATH quality/mechanism dashboard."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PREFIX = {
    "toy": "e49b_math_strategy_toy_05b_v1",
    "full": "e49b_math_strategy_full_05b_v1",
}
POOL = {"toy": 50, "full": 384}
COLORS = {"grpo": "#555555", "online_canonical_haarnoja": "#0072B2"}
LABELS = {
    "grpo": "Matched Dr.GRPO",
    "online_canonical_haarnoja": "E46 strategy MaxEnt",
}


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _curve(stage: str, path: Path) -> list[dict]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("toy", "full"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or (
        ROOT / "paper/figures" / f"e49b_math_strategy_{args.stage}_live.png"
    )
    curve_path = (
        ROOT / "var/artifacts" / f"e49b_math_strategy_{args.stage}_curve.json"
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
            alpha=0.8,
            label=f"{LABELS[arm]} pass@8",
        )
    ax.set(title="A. Held-out quality", xlabel="Prompt epochs", ylabel="Accuracy")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

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
            label=LABELS[arm],
        )
    ax.axhline(2, color="#999999", linestyle=":", linewidth=1)
    ax.set(
        title="B. Validated strategy support",
        xlabel="Prompt epochs",
        ylabel="Mean strategies / solved prompt",
    )
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    for arm, arm_rows in by_arm.items():
        acceptance = []
        integrity = []
        ambiguity = []
        disagreement = []
        for row in arm_rows:
            positive = _number(row.get("math_strategy_validator_positive_rows"))
            accepted = _number(row.get("math_strategy_accepted_rows"))
            invalid = _number(row.get("math_strategy_rejected_integrity_rows"))
            ambiguous = _number(row.get("math_strategy_rejected_ambiguous_rows"))
            rejected = _number(
                row.get("math_strategy_rejected_disagreement_rows")
            )
            acceptance.append(
                accepted / positive if positive and accepted is not None else None
            )
            integrity.append(
                invalid / positive
                if positive and invalid is not None
                else None
            )
            ambiguity.append(
                ambiguous / positive
                if positive and ambiguous is not None
                else None
            )
            disagreement.append(
                rejected / positive if positive and rejected is not None else None
            )
        x = [_number(row.get("training_passes")) for row in arm_rows]
        ax.plot(
            x,
            acceptance,
            color=COLORS[arm],
            marker="o",
            label=f"{LABELS[arm]} accepted",
        )
        if arm == "online_canonical_haarnoja":
            ax.plot(
                x,
                integrity,
                color="#009E73",
                linestyle="-.",
                label="Invalid derivation",
            )
            ax.plot(
                x,
                ambiguity,
                color="#D55E00",
                linestyle="--",
                label="Ambiguous/omitted",
            )
            ax.plot(
                x,
                disagreement,
                color="#CC79A7",
                linestyle=":",
                label="Conflicting reps",
            )
    ax.set(
        title="C. Canonicalizer gate",
        xlabel="Prompt epochs",
        ylabel="Fraction of validator-positive rows",
    )
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)

    treatment = by_arm["online_canonical_haarnoja"]
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
        title="D. Normalized bank entropy",
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
        title="E. Haarnoja response",
        xlabel="Prompt epochs",
        ylabel=r"Next $\alpha$",
    )
    ax.set_ylim(0.09, 0.51)

    ax = axes[1, 2]
    ax.plot(
        x,
        [
            _number(
                row.get("online_canonical_exploration_to_task_rms_ratio")
            )
            for row in treatment
        ],
        color=COLORS["online_canonical_haarnoja"],
        marker="o",
        label="Exploration / task RMS",
    )
    ax.plot(
        x,
        [
            _number(row.get("online_canonical_new_outcome_row_fraction"))
            for row in treatment
        ],
        color="#009E73",
        marker="s",
        linestyle="--",
        label="New-strategy row fraction",
    )
    ax.set(
        title="F. Live exploration signal",
        xlabel="Prompt epochs",
        ylabel="Ratio / fraction",
    )
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    calibration_path = (
        ROOT
        / "var/artifacts/e47w_pairwise_math_strategy_calibration_v1/analysis.json"
    )
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.is_file()
        else {}
    )
    figure.suptitle(
        "E49B proof-gated MATH strategy MaxEnt — "
        f"{args.stage} — E47W gate: {calibration.get('gate_status', 'pending')}",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    if output.suffix.lower() == ".png":
        figure.savefig(output.with_suffix(".pdf"))
    print(output)


if __name__ == "__main__":
    main()
