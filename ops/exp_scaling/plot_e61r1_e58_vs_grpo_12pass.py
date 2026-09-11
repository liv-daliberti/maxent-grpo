#!/usr/bin/env python3
"""Render E61-R1's expanding four-domain, three-seed comparison."""

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

from plot_e64_math500_realism import load_e64_aggregate_points


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/e61r1_e58_vs_grpo_05b_12ep_live"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
REPAIR = "verified_entropy_gated_singleton_escape_canonical"
PLUMBING_CONTROL = "same_plumbing_actuator_off_control"
BLUE = "#0057A8"
ORANGE = "#D55E00"
PURPLE = "#7A3E9D"
GREEN = "#008A5A"
SEEDS = (43, 44, 45)
SEED_STYLES = {43: "-", 44: (0, (5, 2)), 45: (0, (1.5, 1.5))}
DOMAINS = (
    (
        "Graph coloring",
        "gce61r1_e58_vs_grpo_05b_12ep",
        "gce68_separated_support_actuator_05b_12ep",
        192,
    ),
    (
        "Countdown",
        "cde61r1_e58_vs_grpo_05b_12ep",
        "cde68_separated_support_actuator_05b_12ep",
        384,
    ),
    (
        "Python factors",
        "pye61r1_e58_vs_grpo_05b_12ep",
        "pye68_separated_support_actuator_05b_12ep",
        384,
    ),
    (
        "MathIR action menu",
        "mie61r1_e58_vs_grpo_05b_12ep",
        "mie68_separated_support_actuator_05b_12ep",
        384,
    ),
    (
        "Held-out MATH-500 transfer",
        "e64_math500_realism",
        None,
        384,
    ),
)
PLUMBING_PREFIX = {
    "Graph coloring": "gce66_same_plumbing_control_05b_12ep",
    "Countdown": "cde66_same_plumbing_control_05b_12ep",
    "Python factors": "pye66_same_plumbing_control_05b_12ep",
    "MathIR action menu": "mie66_same_plumbing_control_05b_12ep",
}
# The named paper figure stays on the prospectively frozen checkpoint surface:
# ten ModeBench anchors and seven MATH-500 anchors. A separate diagnostic
# wrapper overrides only the MATH display surface to show every integer epoch.
MODEBENCH_CHECKPOINT_PASSES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12)
MATH500_DISPLAY_PASSES = (0, 2, 4, 6, 8, 10, 12)
MATH500_DISPLAY_DESCRIPTION = (
    "the 7 registered even-pass anchors"
)
E64_PREFIX = "e64_math500_realism"
E64_DOMAIN = "Held-out MATH-500 transfer"
E68_AUDIT = (
    ROOT
    / "var/artifacts/e68_separated_support_actuator_ablation_audit_latest.json"
)
CHECKPOINT_COVERAGE_AUDIT = (
    ROOT / "var/artifacts/e65_fixed_checkpoint_coverage_audit_latest.json"
)
EVAL_CADENCE_AUDIT = (
    ROOT / "var/artifacts/e65_eval_cadence_audit_latest.json"
)
E68_AUDIT_DOMAINS = {
    "Graph coloring": "graph_coloring",
    "Countdown": "countdown",
    "Python factors": "python_factor",
    "MathIR action menu": "mathir",
}
OPEN_SET_ENTROPY = (
    "semantic_shannon_success_conditioned_signed_open_set_entropy_ema"
)
OPEN_SET_COEFFICIENT = (
    "semantic_shannon_success_conditioned_signed_open_set_next_coefficient"
)
MASS_COEFFICIENT = "canonical_replay_mass_next_alpha"
BALANCE_COEFFICIENT = "canonical_replay_next_alpha"
TIGHT_CONTROLLER_METRICS = {
    OPEN_SET_ENTROPY,
    OPEN_SET_COEFFICIENT,
    MASS_COEFFICIENT,
    BALANCE_COEFFICIENT,
}
QUALITY_METRICS = {"greedy", "mean8", "pass8", "distinct8"}
PASSIVE_CONTROL_METRICS = {
    "online_canonical_mean_support_per_prompt",
    "online_canonical_tracked_outcomes",
}
PANELS = (
    ("greedy", "neutral pass@1", False),
    ("mean8", "neutral mean@8", False),
    ("pass8", "neutral pass@8", False),
    ("distinct8", "mean # distinct correct@8", True),
    (OPEN_SET_ENTROPY, "open-set predictive entropy EMA", False),
    (OPEN_SET_COEFFICIENT, r"next semantic coefficient $\beta$", False),
    (MASS_COEFFICIENT, r"next verified-mass coefficient $\mu$", False),
    (BALANCE_COEFFICIENT, r"next known-mode coefficient $\alpha$", False),
    ("canonical_replay_balance_loss", "verified replay KL", False),
    ("canonical_replay_available_modes", "replayed verified modes", True),
    (
        "online_canonical_new_outcome_row_fraction",
        "new verified outcome fraction",
        False,
    ),
    (
        "online_canonical_mean_support_per_prompt",
        "mean verified support per prompt",
        False,
    ),
    (
        "online_canonical_tracked_outcomes",
        "cumulative verified discoveries",
        True,
    ),
)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _repair_escape_counts() -> dict[str, int]:
    if not E68_AUDIT.is_file():
        return {}
    payload = json.loads(E68_AUDIT.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for label, audit_domain in E68_AUDIT_DOMAINS.items():
        runs = payload.get("domains", {}).get(audit_domain, {}).get("runs", [])
        counts[label] = sum(int(run.get("interventions", 0)) for run in runs)
    return counts


def _audit_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_points(
    prefix: str,
    steps_per_pass: int,
    *,
    arm_alias: str | None = None,
) -> list[dict[str, Any]]:
    path = ROOT / f"var/artifacts/{prefix}_scaling_curve.json"
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    deduplicated: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        raw_arm = row.get("arm")
        arm = arm_alias if raw_arm == TREATMENT and arm_alias else raw_arm
        seed = row.get("seed")
        step = row.get("step")
        if (
            arm not in (CONTROL, TREATMENT, PLUMBING_CONTROL, REPAIR)
            or seed not in SEEDS
            or not isinstance(step, (int, float))
            or row.get("split") != "multi_answer"
        ):
            continue
        key = (arm, int(seed), int(step))
        point = deduplicated.setdefault(
            key,
            {
                "arm": arm,
                "seed": int(seed),
                "step": int(step),
                "passes": float(step) / steps_per_pass,
            },
        )
        # Scaling artifacts contain both evaluation rows and more-frequent
        # mechanism-only rows. Merge their finite numeric fields at a shared
        # step so the full monitor can show both without allowing the
        # unique-answer split to overwrite multi-answer quality.
        for metric, value in row.items():
            if _finite(value):
                point[metric] = float(value)
    return [deduplicated[key] for key in sorted(deduplicated)]


def _paper_checkpoint_points(
    domain: str,
    points: list[dict[str, Any]],
    steps_per_pass: int,
) -> list[dict[str, Any]]:
    passes = (
        MATH500_DISPLAY_PASSES
        if domain == E64_DOMAIN
        else MODEBENCH_CHECKPOINT_PASSES
    )
    allowed_steps = {int(value * steps_per_pass) for value in passes}
    return [
        point
        for point in points
        if int(point["step"]) in allowed_steps
    ]


def _series(
    points: list[dict[str, Any]],
    arm: str,
    seed: int,
    metric: str,
) -> tuple[list[float], list[float]]:
    selected = sorted(
        (
            row
            for row in points
            if (
                row["arm"] == arm
                and row["seed"] == seed
                and _finite(row.get(metric))
            )
        ),
        key=lambda row: row["step"],
    )
    return (
        [row["passes"] for row in selected],
        [float(row[metric]) for row in selected],
    )


def _complete_mean(
    points: list[dict[str, Any]],
    arm: str,
    metric: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    by_step: dict[int, dict[int, dict[str, Any]]] = {}
    for row in points:
        if row["arm"] == arm and _finite(row.get(metric)):
            by_step.setdefault(int(row["step"]), {})[int(row["seed"])] = row
    xs: list[float] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for step in sorted(by_step):
        seed_rows = by_step[step]
        if set(seed_rows) != set(SEEDS):
            continue
        values = [float(seed_rows[seed][metric]) for seed in SEEDS]
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


def _set_y_limits(
    ax: plt.Axes,
    metric: str,
    plotted_values: list[float],
) -> None:
    if not plotted_values:
        ax.set_ylim(0.0, 1.0)
        return
    low = min(plotted_values)
    high = max(plotted_values)
    if metric in TIGHT_CONTROLLER_METRICS:
        span = high - low
        padding = max(span * 0.12, abs(high) * 0.025, 1e-4)
        ax.set_ylim(max(0.0, low - padding), high + padding)
        return
    upper = high * 1.08 if high > 0 else 1.0
    ax.set_ylim(0.0, upper)


def main() -> None:
    escape_counts = _repair_escape_counts()
    coverage_audit = _audit_summary(CHECKPOINT_COVERAGE_AUDIT)
    cadence_audit = _audit_summary(EVAL_CADENCE_AUDIT)
    coverage_summary = coverage_audit.get("summary", {})
    raw_domain_points = {
        label: (
            load_e64_aggregate_points()
            if prefix == E64_PREFIX
            else _load_points(prefix, steps_per_pass)
        )
        for label, prefix, _, steps_per_pass in DOMAINS
    }
    for label, _, repair_prefix, steps_per_pass in DOMAINS:
        if repair_prefix is not None:
            raw_domain_points[label].extend(
                _load_points(repair_prefix, steps_per_pass)
            )
            raw_domain_points[label].extend(
                _load_points(
                    PLUMBING_PREFIX[label],
                    steps_per_pass,
                    arm_alias=PLUMBING_CONTROL,
                )
            )
    domain_points = {
        label: _paper_checkpoint_points(
            label,
            raw_domain_points[label],
            steps_per_pass,
        )
        for label, _, _, steps_per_pass in DOMAINS
    }
    all_points = [
        point for points in domain_points.values() for point in points
    ]
    latest_pass = max((row["passes"] for row in all_points), default=0.0)
    latest_training_pass = max(
        (
            row["passes"]
            for points in raw_domain_points.values()
            for row in points
        ),
        default=0.0,
    )

    fig, axes = plt.subplots(
        len(DOMAINS),
        len(PANELS),
        figsize=(31.5, 11.4),
        sharex="row",
        squeeze=False,
    )

    for row_index, (domain, _, _, _) in enumerate(DOMAINS):
        points = domain_points[domain]
        domain_latest_pass = max(
            (float(point["passes"]) for point in points),
            default=0.0,
        )
        x_upper = max(
            0.5,
            domain_latest_pass
            + max(0.08, domain_latest_pass * 0.06),
        )
        for column_index, (metric, title, integer_ticks) in enumerate(PANELS):
            ax = axes[row_index, column_index]
            plotted_values: list[float] = []
            plotted_anything = False
            plot_arms = (
                (CONTROL, TREATMENT, PLUMBING_CONTROL, REPAIR)
                if metric in QUALITY_METRICS
                or metric in PASSIVE_CONTROL_METRICS
                else (TREATMENT, PLUMBING_CONTROL, REPAIR)
            )
            for arm, color, marker, arm_zorder in (
                (CONTROL, BLUE, "o", 2),
                (TREATMENT, ORANGE, "D", 3),
                (PLUMBING_CONTROL, PURPLE, "s", 4),
                (REPAIR, GREEN, "^", 5),
            ):
                if arm not in plot_arms:
                    continue
                for seed in SEEDS:
                    xs, ys = _series(points, arm, seed, metric)
                    plotted_anything = plotted_anything or bool(xs)
                    plotted_values.extend(ys)
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
                    plotted_anything = True
                    plotted_values.extend(lows)
                    plotted_values.extend(highs)
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
                        zorder=6 if arm == REPAIR else 5,
                    )
            if not plotted_anything:
                if domain == E64_DOMAIN and metric == "distinct8":
                    pending_text = (
                        "N/A — correctness transfer\n"
                        "has one verified answer class"
                    )
                elif domain == E64_DOMAIN and metric == BALANCE_COEFFICIENT:
                    pending_text = (
                        "N/A — known-mode balance\n"
                        "is structurally ineligible"
                    )
                else:
                    pending_text = (
                        "awaiting evaluation"
                        if metric in QUALITY_METRICS
                        else "awaiting mechanism telemetry"
                    )
                ax.text(
                    0.5,
                    0.5,
                    pending_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                    fontsize=6.5,
                )
            ax.set_title(title, fontsize=7.5)
            _set_y_limits(ax, metric, plotted_values)
            ax.set_xlim(0.0, x_upper)
            _style_axis(ax)
            if integer_ticks:
                ax.yaxis.set_major_locator(
                    MaxNLocator(5, integer=True)
                )
            if column_index == 0:
                ax.set_ylabel(domain, fontsize=8.2, fontweight="bold")
            if (
                domain == E64_DOMAIN
                and metric == "canonical_replay_available_modes"
            ):
                ax.text(
                    0.98,
                    0.96,
                    "verified answer classes",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=5.5,
                    color="#666666",
                )
        eval_metric = "greedy" if domain == E64_DOMAIN else "distinct8"
        axes[row_index, 0].text(
            0.985,
            0.05,
            (
                f"{sum(_finite(point.get(eval_metric)) for point in points)} "
                "eval records"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#666666",
        )
        if domain in escape_counts:
            axes[row_index, 0].text(
                0.015,
                0.95,
                f"E68 audited escapes={escape_counts[domain]}",
                transform=axes[row_index, 0].transAxes,
                ha="left",
                va="top",
                fontsize=5.8,
                color=GREEN,
            )

    for ax in axes[-1]:
        ax.set_xlabel("training passes", fontsize=7)

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
            color=PURPLE,
            lw=2.8,
            marker="s",
            markerfacecolor="none",
            markeredgewidth=1.35,
            label="E66 same-plumbing actuator-off control",
        ),
        Line2D(
            [],
            [],
            color=GREEN,
            lw=2.8,
            marker="^",
            markerfacecolor="none",
            markeredgewidth=1.35,
            label="E68 separated-support entropy-gated actuator",
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
        ncol=7,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.985),
    )
    complete_means = sum(
        len(
            _complete_mean(
                points,
                arm,
                "greedy" if domain == E64_DOMAIN else "distinct8",
            )[0]
        )
        for domain, points in domain_points.items()
        for arm in (CONTROL, TREATMENT, PLUMBING_CONTROL, REPAIR)
    )
    fig.suptitle(
        "Five-domain confirmation — Dr.GRPO, historical E58, "
        "same-plumbing E66 control, and separated-support E68 actuator; "
        "3 seeds × 12 passes\n"
        f"latest paper checkpoint={latest_pass:.2f} passes "
        f"(training telemetry={latest_training_pass:.2f}); "
        f"complete 3-seed arm/domain points={complete_means}; "
        "fixed checkpoint cells="
        f"{coverage_summary.get('landed_checkpoint_cells', 0)}/"
        f"{coverage_summary.get('expected_checkpoint_cells', 174)}; "
        f"eval cadence={cadence_audit.get('status', 'not available')}; "
        f"audited E68 singleton escapes={sum(escape_counts.values())}; "
        "thick lines are 3-seed means, shaded bands are seed ranges",
        fontsize=10.5,
        y=1.015,
    )
    fig.text(
        0.5,
        0.006,
        "ModeBench rows show their 10 fixed paper anchors. Held-out MATH-500 "
        "is evaluated every quarter epoch and this row shows "
        f"{MATH500_DISPLAY_DESCRIPTION}; its primary gate remains frozen to "
        "those same 7 anchors. Every row's x-axis expands as those points land. "
        "Training uses no gold support, desired mode "
        "count, desired entropy, or evaluation feedback; E58's three "
        "self-warmup coefficients remain unprojected. MATH-500 is held out; "
        "its mode-balance cells are N/A rather than inferred from prose. "
        "E66 is E68's same-plumbing causal comparator. E68 escape counts "
        "are support-only admissions and never PPO rows.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0.025, 0.035, 0.995, 0.935))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[e61r1-plot] wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
