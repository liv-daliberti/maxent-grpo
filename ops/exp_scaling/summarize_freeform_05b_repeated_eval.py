#!/usr/bin/env python3
"""Summarize and plot E30's four fixed K=8 evaluation draws without smoothing."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "var/artifacts/freeform_05b_repeated_eval_v1"
OUT_JSON = EVAL_ROOT / "freeform_05b_fixed_k8x4_findings.json"
OUT_MD = EVAL_ROOT / "freeform_05b_fixed_k8x4_findings.md"
OUT_FIGURE = ROOT / "paper/figures/freeform_05b_fixed_k8x4_diagnostic"
PREVIEW = EVAL_ROOT / "freeform_05b_fixed_k8x4_diagnostic.png"

ENVIRONMENTS = {
    "Countdown": "cde30_freeform_05b_fixed_k8x4_v1",
    "Graph coloring": "gce30_freeform_05b_fixed_k8x4_v1",
}
TRAJECTORY_INPUTS = {
    "Countdown": {
        "grpo": "cde22_freeform_conditional_dual_05b_v2_scaling_curve.json",
        "maxent_dual": "cde27_freeform_conditional_dual_05b_v1_scaling_curve.json",
    },
    "Graph coloring": {
        "grpo": "gce22_freeform_conditional_dual_05b_v2_scaling_curve.json",
        "maxent_dual": "gce27_freeform_conditional_dual_05b_v1_scaling_curve.json",
    },
}
ARMS = {"grpo": "Dr.GRPO", "maxent_dual": "Conditional-token MaxEnt"}
METRICS = {
    "pass8": "any_correct_at_k",
    "mean8": "mean_at_k",
    "coverage8": "mode_coverage_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}
METRIC_LABELS = {
    "pass8": "pass@8",
    "mean8": "mean@8",
    "coverage8": "coverage@8",
    "distinct8": "distinct@8",
}
COLORS = {"grpo": "#3f3f3f", "maxent_dual": "#6A3D9A"}
MARKERS = {43: "o", 44: "s", 45: "^"}


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def load_records() -> list[dict]:
    records = []
    for environment, stamp in ENVIRONMENTS.items():
        pattern = re.compile(re.escape(stamp) + r"_e(\d+)_coverage_summary\.json")
        for path in sorted(EVAL_ROOT.glob(f"{stamp}_e*_coverage_summary.json")):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            eval_seed = int(match.group(1))
            payload = json.loads(path.read_text(encoding="utf-8"))
            for checkpoint in payload["checkpoints"]:
                alias_match = re.fullmatch(r"(.+)_s(\d+)", checkpoint["alias"])
                if alias_match is None:
                    continue
                arm, train_seed_text = alias_match.groups()
                metrics = checkpoint["splits"]["multi_answer"]["metrics"]
                records.append(
                    {
                        "environment": environment,
                        "arm": arm,
                        "train_seed": int(train_seed_text),
                        "eval_seed": eval_seed,
                        **{
                            short: float(metrics[source])
                            for short, source in METRICS.items()
                        },
                    }
                )
    return records


def summarize_trajectory_roughness() -> dict:
    """Measure raw adjacent-checkpoint movement in the clean source curves."""
    output = {}
    artifact_root = ROOT / "var/artifacts"
    for environment, arm_paths in TRAJECTORY_INPUTS.items():
        output[environment] = {}
        for arm, filename in arm_paths.items():
            rows = json.loads((artifact_root / filename).read_text(encoding="utf-8"))
            rows = [row for row in rows if row["arm"] == arm]
            output[environment][arm] = {}
            for metric in METRICS:
                by_seed = {}
                for seed in sorted({row["seed"] for row in rows}):
                    seed_rows = sorted(
                        (
                            row
                            for row in rows
                            if row["seed"] == seed and row.get(metric) is not None
                        ),
                        key=lambda row: row["training_passes"],
                    )
                    adjacent = [
                        abs(next_row[metric] - row[metric])
                        for row, next_row in zip(seed_rows, seed_rows[1:])
                    ]
                    if adjacent:
                        by_seed[str(seed)] = float(np.mean(adjacent))
                output[environment][arm][metric] = {
                    "mean_absolute_adjacent_difference": float(
                        np.mean(list(by_seed.values()))
                    ),
                    "by_train_seed": by_seed,
                    "source": filename,
                }
    return output


def summarize(records: list[dict]) -> dict:
    output = {
        "design": {},
        "cells": {},
        "effects": {},
        "trajectory_roughness": summarize_trajectory_roughness(),
    }
    for environment, stamp in ENVIRONMENTS.items():
        environment_rows = [row for row in records if row["environment"] == environment]
        output["design"][environment] = {
            "stamp": stamp,
            "train_seeds": sorted({row["train_seed"] for row in environment_rows}),
            "eval_seeds": sorted({row["eval_seed"] for row in environment_rows}),
            "k": 8,
            "engine": "vllm_v0",
        }
        output["cells"][environment] = {}
        for arm in ARMS:
            arm_rows = [row for row in environment_rows if row["arm"] == arm]
            metric_summaries = {}
            for metric in METRICS:
                raw = [
                    {
                        "train_seed": row["train_seed"],
                        "eval_seed": row["eval_seed"],
                        "value": row[metric],
                    }
                    for row in arm_rows
                ]
                by_train_seed = defaultdict(list)
                for row in arm_rows:
                    by_train_seed[row["train_seed"]].append(row[metric])
                train_seed_means = {
                    str(seed): float(np.mean(values))
                    for seed, values in sorted(by_train_seed.items())
                }
                within_draw_sds = {
                    str(seed): _sample_std(values)
                    for seed, values in sorted(by_train_seed.items())
                }
                values = [item["value"] for item in raw]
                metric_summaries[metric] = {
                    "mean": float(np.mean(values)),
                    "raw_min": float(np.min(values)),
                    "raw_max": float(np.max(values)),
                    "total_sd": _sample_std(values),
                    "mean_within_train_seed_eval_draw_sd": float(
                        np.mean(list(within_draw_sds.values()))
                    ),
                    "mean_within_train_seed_eval_draw_se": float(
                        np.mean(list(within_draw_sds.values())) / 2.0
                    ),
                    "between_train_seed_sd_of_draw_means": _sample_std(
                        list(train_seed_means.values())
                    ),
                    "train_seed_means": train_seed_means,
                    "within_train_seed_eval_draw_sds": within_draw_sds,
                    "raw_draws": raw,
                }
            output["cells"][environment][arm] = metric_summaries

        regression_path = EVAL_ROOT / f"{stamp}_regression.json"
        regressions = json.loads(regression_path.read_text(encoding="utf-8"))
        output["effects"][environment] = {
            row["outcome"]: row for row in regressions
        }
    return output


def write_markdown(summary: dict) -> None:
    lines = [
        "# E30 fixed K=8 × 4 evaluation findings",
        "",
        "Every cell contains 12 raw estimates (3 training seeds × 4 fixed evaluation seeds).",
        "No curve or draw is smoothed. `MC SD` averages the within-checkpoint SD across",
        "the four evaluation seeds; `train SD` is the SD of the three checkpoint means.",
        "",
    ]
    for environment in ENVIRONMENTS:
        lines.extend(
            [
                f"## {environment}",
                "",
                "| Arm | Metric | Mean | Raw range | MC SD | MC SE | Train SD |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for arm, arm_label in ARMS.items():
            for metric, metric_label in METRIC_LABELS.items():
                row = summary["cells"][environment][arm][metric]
                lines.append(
                    f"| {arm_label} | {metric_label} | {row['mean']:.4f} | "
                    f"[{row['raw_min']:.4f}, {row['raw_max']:.4f}] | "
                    f"{row['mean_within_train_seed_eval_draw_sd']:.4f} | "
                    f"{row['mean_within_train_seed_eval_draw_se']:.4f} | "
                    f"{row['between_train_seed_sd_of_draw_means']:.4f} |"
                )
        lines.extend(
            [
                "",
                "| Contrast (MaxEnt − Dr.GRPO) | Estimate | Prompt-clustered 95% CI | Run-cluster p |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for outcome in ("pass@8", "mean@8", "coverage@8", "distinct@8", "pass@1 (greedy)"):
            row = summary["effects"][environment][outcome]
            lines.append(
                f"| {outcome} | {row['gamma']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                f"{row['p_run_cluster']:.4f} |"
            )
        lines.extend(
            [
                "",
                "| Arm | Metric | Adjacent-checkpoint MAAD | MAAD / terminal MC SD |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for arm, arm_label in ARMS.items():
            for metric, metric_label in METRIC_LABELS.items():
                roughness = summary["trajectory_roughness"][environment][arm][metric][
                    "mean_absolute_adjacent_difference"
                ]
                mc_sd = summary["cells"][environment][arm][metric][
                    "mean_within_train_seed_eval_draw_sd"
                ]
                ratio = roughness / mc_sd if mc_sd > 0 else float("inf")
                ratio_text = f"{ratio:.1f}×" if np.isfinite(ratio) else "∞"
                lines.append(
                    f"| {arm_label} | {metric_label} | {roughness:.4f} | "
                    f"{ratio_text} |"
                )
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(summary: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.1), squeeze=False)
    for row_index, environment in enumerate(ENVIRONMENTS):
        for column_index, (metric, metric_label) in enumerate(METRIC_LABELS.items()):
            axis = axes[row_index][column_index]
            for arm_index, arm in enumerate(ARMS):
                cell = summary["cells"][environment][arm][metric]
                for seed_index, train_seed in enumerate((43, 44, 45)):
                    draws = [
                        item
                        for item in cell["raw_draws"]
                        if item["train_seed"] == train_seed
                    ]
                    draws.sort(key=lambda item: item["eval_seed"])
                    center = arm_index + (seed_index - 1) * 0.13
                    offsets = np.linspace(-0.035, 0.035, len(draws))
                    axis.scatter(
                        center + offsets,
                        [item["value"] for item in draws],
                        color=COLORS[arm],
                        marker=MARKERS[train_seed],
                        s=24,
                        alpha=0.72,
                        linewidths=0,
                        zorder=3,
                    )
                axis.vlines(
                    arm_index,
                    cell["raw_min"],
                    cell["raw_max"],
                    color=COLORS[arm],
                    lw=1.2,
                    alpha=0.65,
                    zorder=2,
                )
                axis.scatter(
                    [arm_index],
                    [cell["mean"]],
                    color=COLORS[arm],
                    marker="D",
                    s=42,
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=4,
                )
            axis.set_xticks([0, 1], ["Dr.GRPO", "MaxEnt"])
            axis.set_title(metric_label)
            if column_index == 0:
                axis.set_ylabel(environment)
            if metric != "distinct8":
                axis.set_ylim(0, 1)
            else:
                axis.set_ylim(bottom=0)
            axis.grid(axis="y", color="#dddddd", lw=0.6)
            axis.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D(
            [],
            [],
            color="#555555",
            marker=MARKERS[seed],
            linestyle="none",
            label=f"train seed {seed}",
        )
        for seed in (43, 44, 45)
    ]
    handles.append(
        plt.Line2D([], [], color="#555555", marker="D", linestyle="none", label="12-draw mean")
    )
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center")
    fig.suptitle(
        "0.5B free-form terminal evaluation: four fixed K=8 draws per training seed",
        y=0.95,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.015,
        "All raw draw estimates shown; vertical whiskers are raw ranges; no smoothing.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.91))
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE.with_suffix(".pdf"))
    fig.savefig(OUT_FIGURE.with_suffix(".png"), dpi=200)
    fig.savefig(PREVIEW, dpi=200)
    plt.close(fig)


def main() -> None:
    records = load_records()
    if len(records) != 48:
        raise SystemExit(f"expected 48 checkpoint/draw records, found {len(records)}")
    summary = summarize(records)
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary)
    plot(summary)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_FIGURE}.pdf/.png and {PREVIEW}")


if __name__ == "__main__":
    main()
