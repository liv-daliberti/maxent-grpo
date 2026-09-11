#!/usr/bin/env python3
"""Render E61's expanding four-domain, three-seed comparison."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/e61_e58_vs_grpo_05b_12ep_live"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
BLUE = "#0057A8"
ORANGE = "#D55E00"
SEEDS = (43, 44, 45)
SEED_STYLES = {43: "-", 44: (0, (5, 2)), 45: (0, (1.5, 1.5))}
DOMAINS = (
    (
        "Graph coloring",
        "gce61_e58_vs_grpo_05b_12ep",
        192,
    ),
    (
        "Countdown",
        "cde61_e58_vs_grpo_05b_12ep",
        384,
    ),
    (
        "Python factors",
        "pye61_e58_vs_grpo_05b_12ep",
        384,
    ),
    (
        "MathIR action menu",
        "mie61_e58_vs_grpo_05b_12ep",
        384,
    ),
)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_points(prefix: str, steps_per_pass: int) -> list[dict[str, float]]:
    path = ROOT / f"var/artifacts/{prefix}_scaling_curve.json"
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    deduplicated: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in rows:
        arm = row.get("arm")
        seed = row.get("seed")
        step = row.get("step")
        if (
            arm not in (CONTROL, TREATMENT)
            or seed not in SEEDS
            or row.get("split") != "multi_answer"
            or not isinstance(step, (int, float))
            or not _finite(row.get("distinct8"))
            or not _finite(row.get("pass8"))
            or not _finite(row.get("mean8"))
        ):
            continue
        point = {
            "arm": arm,
            "seed": int(seed),
            "step": int(step),
            "passes": float(step) / steps_per_pass,
            "distinct8": float(row["distinct8"]),
            "pass8": float(row["pass8"]),
            "mean8": float(row["mean8"]),
        }
        point["excess"] = point["distinct8"] - point["pass8"]
        deduplicated[(arm, int(seed), int(step))] = point
    return [deduplicated[key] for key in sorted(deduplicated)]


def _series(
    points: list[dict[str, float]],
    arm: str,
    seed: int,
    metric: str,
) -> tuple[list[float], list[float]]:
    selected = sorted(
        (
            row
            for row in points
            if row["arm"] == arm and row["seed"] == seed
        ),
        key=lambda row: row["step"],
    )
    return (
        [row["passes"] for row in selected],
        [row[metric] for row in selected],
    )


def _complete_mean(
    points: list[dict[str, float]],
    arm: str,
    metric: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    by_step: dict[int, dict[int, dict[str, float]]] = {}
    for row in points:
        if row["arm"] == arm:
            by_step.setdefault(int(row["step"]), {})[int(row["seed"])] = row
    xs: list[float] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for step in sorted(by_step):
        seed_rows = by_step[step]
        if set(seed_rows) != set(SEEDS):
            continue
        values = [seed_rows[seed][metric] for seed in SEEDS]
        xs.append(seed_rows[SEEDS[0]]["passes"])
        means.append(statistics.fmean(values))
        lows.append(min(values))
        highs.append(max(values))
    return xs, means, lows, highs


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#dddddd", lw=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.7, labelsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def main() -> None:
    domain_points = {
        label: _load_points(prefix, steps_per_pass)
        for label, prefix, steps_per_pass in DOMAINS
    }
    all_points = [
        point for points in domain_points.values() for point in points
    ]
    latest_pass = max((row["passes"] for row in all_points), default=0.0)
    x_upper = max(0.5, latest_pass + max(0.08, latest_pass * 0.06))

    fig, axes = plt.subplots(4, 3, figsize=(13.6, 11.2), sharex=True)
    columns = (
        ("distinct8", "mean # distinct correct@8"),
        ("excess", "excess multiplicity: distinct@8 − pass@8"),
        ("mean8", "mean correctness@8"),
    )

    for row_index, (domain, _, _) in enumerate(DOMAINS):
        points = domain_points[domain]
        for column_index, (metric, title) in enumerate(columns):
            ax = axes[row_index, column_index]
            for arm, color, marker, arm_zorder in (
                (CONTROL, BLUE, "o", 2),
                (TREATMENT, ORANGE, "D", 3),
            ):
                for seed in SEEDS:
                    xs, ys = _series(points, arm, seed, metric)
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        ls=SEED_STYLES[seed],
                        lw=1.15,
                        marker=marker,
                        ms=2.8 if arm == CONTROL else 3.1,
                        markerfacecolor=color if arm == CONTROL else "none",
                        markeredgecolor=color,
                        markeredgewidth=1.05,
                        alpha=0.68,
                        zorder=arm_zorder,
                    )
                xs, means, lows, highs = _complete_mean(points, arm, metric)
                if xs:
                    ax.fill_between(
                        xs,
                        lows,
                        highs,
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                        zorder=1,
                    )
                    ax.plot(
                        xs,
                        means,
                        color=color,
                        lw=2.8,
                        marker=marker,
                        ms=4.2 if arm == CONTROL else 4.6,
                        markerfacecolor=color if arm == CONTROL else "none",
                        markeredgecolor=color,
                        markeredgewidth=1.35,
                        alpha=1.0,
                        zorder=5 if arm == TREATMENT else 4,
                    )
            if not points:
                ax.text(
                    0.5,
                    0.5,
                    "starting — evaluation not emitted yet",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                    fontsize=8,
                )
            ax.set_title(title, fontsize=9)
            ax.set_ylim(bottom=0.0)
            ax.set_xlim(0.0, x_upper)
            _style_axis(ax)
            if column_index == 0:
                ax.set_ylabel(domain, fontsize=9, fontweight="bold")
        axes[row_index, 0].text(
            0.985,
            0.05,
            f"{len(points)} parsed eval records",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#666666",
        )

    for ax in axes[-1]:
        ax.set_xlabel("training passes (axis expands with landed data)", fontsize=8)

    handles = [
        Line2D(
            [],
            [],
            color=BLUE,
            lw=2.8,
            marker="o",
            label="matched Dr.GRPO",
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            lw=2.8,
            marker="D",
            markerfacecolor="none",
            markeredgewidth=1.35,
            label="E58 global verified replay",
        ),
        Line2D(
            [],
            [],
            color="#555555",
            lw=1.15,
            ls=SEED_STYLES[43],
            label="seed 43",
        ),
        Line2D(
            [],
            [],
            color="#555555",
            lw=1.15,
            ls=SEED_STYLES[44],
            label="seed 44",
        ),
        Line2D(
            [],
            [],
            color="#555555",
            lw=1.15,
            ls=SEED_STYLES[45],
            label="seed 45",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.975),
    )
    complete_means = sum(
        len(_complete_mean(points, arm, "distinct8")[0])
        for points in domain_points.values()
        for arm in (CONTROL, TREATMENT)
    )
    fig.suptitle(
        "E61 live — 3-seed matched Dr.GRPO vs literal E58; "
        "12 passes/run; four executable domains\n"
        f"latest landed evaluation={latest_pass:.2f} passes; "
        f"complete 3-seed arm/domain points={complete_means}; "
        "thick lines are 3-seed means, shaded bands are seed ranges",
        fontsize=11,
        y=1.005,
    )
    fig.text(
        0.5,
        0.006,
        "Training uses no gold support, desired mode count, desired entropy, "
        "or evaluation feedback. E58 coefficients are unprojected.",
        ha="center",
        fontsize=7.7,
        color="#444444",
    )
    fig.tight_layout(rect=(0.03, 0.025, 0.99, 0.945))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[e61-plot] wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
