#!/usr/bin/env python3
"""Render the frozen E69 five-area confirmatory paper panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GATE3 = ROOT / "var/artifacts/e69_gate3_confirmatory_audit_latest.json"
GATE4 = ROOT / "var/artifacts/e69_gate4_math500_analysis.json"
OUT_PNG = ROOT / "paper/figures/e69_five_area_confirmatory.png"
OUT_PDF = ROOT / "paper/figures/e69_five_area_confirmatory.pdf"
DOMAINS = (
    ("graph_coloring", "Graph"),
    ("countdown", "Countdown"),
    ("python_factor", "Python"),
    ("mathir", "MathIR"),
)
COLORS = {
    "greedy": "#285f9e",
    "mean8": "#d07a1f",
    "pass8": "#2b8c68",
    "distinct8": "#8f4b9e",
}
LABELS = {
    "greedy": "Greedy",
    "mean8": "Mean@8",
    "pass8": "Pass@8",
    "distinct8": "Distinct@8",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate3 = json.loads(GATE3.read_text(encoding="utf-8"))
    gate4 = json.loads(GATE4.read_text(encoding="utf-8"))
    if (
        gate3.get("status") != "complete"
        or gate3.get("summary", {}).get("integrity_violations") != 0
        or gate4.get("status") != "complete"
        or gate4.get("summary", {}).get("integrity_violations") != 0
    ):
        raise SystemExit("E69 paper panel requires clean complete Gate 3 and Gate 4")

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(12.5, 2.8),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.92]},
        constrained_layout=True,
    )
    markers = {"greedy": "o", "pass8": "s", "distinct8": "^"}
    for axis, (domain, title) in zip(axes[:4], DOMAINS):
        passes = list(range(7))
        for metric in ("greedy", "pass8", "distinct8"):
            values = [
                gate3["aggregate_deltas"][domain][str(pass_index)][metric]
                for pass_index in passes
            ]
            axis.plot(
                passes,
                values,
                color=COLORS[metric],
                marker=markers[metric],
                markersize=3.5,
                linewidth=1.4,
                label=LABELS[metric],
            )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.axhline(
            -0.02,
            color="#999999",
            linewidth=0.7,
            linestyle=":",
        )
        axis.set_title(title)
        axis.set_xticks((0, 2, 4, 6))
        axis.set_xlabel("Prompt passes")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0].set_ylabel("E69 − compute-matched Dr.GRPO")

    math_axis = axes[4]
    x_positions = list(range(3))
    for x_position, metric in zip(x_positions, ("greedy", "mean8", "pass8")):
        row = gate4["paired_deltas"][metric]
        math_axis.errorbar(
            [x_position],
            [row["mean"]],
            yerr=[
                [row["mean"] - row["lower"]],
                [row["upper"] - row["mean"]],
            ],
            color=COLORS[metric],
            marker="D",
            markersize=5,
            capsize=3,
            linewidth=1.2,
        )
        offsets = (-0.10, 0.0, 0.10)
        for offset, seed in zip(offsets, (43, 44, 45)):
            math_axis.scatter(
                [x_position + offset],
                [row["seeds"][str(seed)]],
                color=COLORS[metric],
                marker=".",
                s=15,
                alpha=0.7,
                zorder=3,
            )
    math_axis.axhline(0, color="#555555", linewidth=0.8)
    math_axis.axhline(
        -0.02,
        color="#999999",
        linewidth=0.7,
        linestyle=":",
    )
    math_axis.set_xticks(x_positions, ("Greedy", "Mean@8", "Pass@8"), rotation=30)
    math_axis.set_title("Held-out MATH-500")
    math_axis.set_xlabel("Terminal metric")
    math_axis.grid(axis="y", color="#dddddd", linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    classification = gate4["final_classification"]["status"].replace("_", " ")
    figure.suptitle(
        "E69 verified-route successor: paired three-seed effects "
        f"(frozen classification: {classification})",
        y=1.08,
        fontsize=10.5,
    )
    figure.text(
        0.995,
        -0.02,
        "E69 provenance: "
        f"Gate3 { _sha256(GATE3)[:10] } · Gate4 { _sha256(GATE4)[:10] }",
        ha="right",
        va="top",
        fontsize=6.5,
        color="#666666",
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(figure)
    print(f"[e69-panel] wrote {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()
