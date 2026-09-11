#!/usr/bin/env python3
"""Render the active E26 MATH dual against its historical E21 control.

This deliberately mirrors the canonical-action MaxEnt paper style: method
colors are identical, individual seeds are faint, and the available-seed mean
is emphasized. Each arm is read from its explicit immutable prefix; cancelled
E21 treatments and aborted E26 drafts are never pooled into the active figure.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREFIX = "mte26_math_freeform_conditional_dual_high_entropy_05b_v2"
DEFAULT_CONTROL_PREFIX = "mte21_math_conditional_token_05b_v4"
DEFAULT_OUTPUT = ROOT / "paper/figures/e26_freeform_math_maxent_live"
DEFAULT_CURVE = ROOT / "var/artifacts/mte26_math_freeform_conditional_dual_high_entropy_05b_v2_live_curve.json"
DEFAULT_COMPUTE_OUTPUT = ROOT / "paper/figures/compute_divergence_math_maxent"
DEFAULT_COMPUTE_PREVIEW = ROOT / "var/artifacts/divergence_math_maxent_latest.png"
TRAIN_PROMPTS = 8515.0
EXPECTED_SEEDS = (43, 44, 45)

METHODS = (
    ("grpo", r"Historical E21 Dr.GRPO ($\alpha=0$)", "#666666"),
    ("maxent_dual", "E26-v2 aggressive Haarnoja dual", "#009E73"),
)
METHOD_KEYS = frozenset(method for method, _label, _color in METHODS)
PANELS = (
    ("pass1", "MATH-500 pass@1", "eval", (0.0, 1.0)),
    ("pass8", "MATH-500 pass@8", "eval", (0.0, 1.0)),
    ("mean8", "MATH-500 mean@8", "eval", (0.0, 1.0)),
    ("reward", "Training reward (16-step mean)", "train", (0.0, 1.0)),
    ("length", "Response tokens (16-step mean)", "train", (0.0, 1024.0)),
    ("no_eos", "No-EOS rate (16-step mean)", "train", (0.0, 1.0)),
)
COMPUTE_METHODS = (
    ("grpo", "E21 Dr.GRPO (historical reference)", "#767676", (0, (4, 2))),
    (
        "maxent_dual",
        "E26-v2 base-preserving Haarnoja dual (125% target)",
        "#6A3D9A",
        (0, (5, 1.5, 1, 1.5)),
    ),
)
COMPUTE_PANELS = (
    ("pass1", "pass@1", "eval", (0.0, 1.0)),
    ("pass8", "pass@8", "eval", (0.0, 1.0)),
    ("mean8", "mean@8", "eval", (0.0, 1.0)),
    ("reward", "training reward", "train", (0.0, 1.0)),
    ("length", "response tokens", "train", (0.0, 1024.0)),
    ("no_eos", "no-EOS rate", "train", (0.0, 1.0)),
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _step(row: dict[str, Any]) -> int:
    value = row.get("trainer/global_step", row.get("misc/global_step", -1))
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # A live writer may leave one incomplete final line.  Earlier complete
    # records remain valid and are retained.
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _discover_attempt(data_root: Path, prefix: str, arm: str, seed: int) -> tuple[Path, list[dict[str, Any]]] | None:
    suffix = f"_{prefix}_{arm}_s{seed}"
    candidates: list[tuple[int, int, Path, list[dict[str, Any]]]] = []
    for run_dir in data_root.glob(f"*{suffix}"):
        for metrics_path in run_dir.glob("debug_*/train_metrics.jsonl"):
            rows = _read_metrics(metrics_path)
            reached = max((_step(row) for row in rows), default=-1)
            candidates.append((reached, len(rows), metrics_path, rows))
    if not candidates:
        return None
    _reached, _count, path, rows = max(
        candidates, key=lambda item: (item[0], item[1], str(item[2]))
    )
    return path, rows


def _training_pass(row: dict[str, Any]) -> float | None:
    # One learner step consumes one source prompt and produces 16 sampled
    # candidates.  ``misc/prompt_consumed`` counts those candidate rows, so
    # using it here would make the x-axis advance sixteen times too quickly.
    step = _step(row)
    if step < 0:
        return None
    return min(1.0, max(0.0, step / TRAIN_PROMPTS))


def _rolling(points: list[tuple[float, float]], window: int = 16) -> list[tuple[float, float]]:
    points = sorted(points)
    values: list[float] = []
    output: list[tuple[float, float]] = []
    for x_value, value in points:
        values.append(value)
        output.append((x_value, sum(values[-window:]) / len(values[-window:])))
    return output


def _extract(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = defaultdict(list)
    eval_keys = {
        "pass1": "eval/math/accuracy",
        "pass8": "eval/math/sampled_any_correct_at_8",
        "mean8": "eval/math/sampled_mean_at_8",
    }
    seen_eval: dict[tuple[str, float], tuple[int, float]] = {}
    train_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x_value = _training_pass(row)
        if x_value is None:
            continue
        step = _step(row)
        for metric, key in eval_keys.items():
            value = _finite(row.get(key))
            if value is not None:
                dedupe_key = (metric, round(x_value, 8))
                previous = seen_eval.get(dedupe_key)
                if previous is None or step >= previous[0]:
                    seen_eval[dedupe_key] = (step, value)

        reward = _finite(row.get("actor/rewards"))
        length = _finite(row.get("actor/response_tok_len"))
        no_eos = _finite(row.get("actor/no_eos_count"))
        batch = _finite(row.get("actor/num_data"))
        if reward is not None:
            train_points["reward"].append((x_value, reward))
        if length is not None:
            train_points["length"].append((x_value, length))
        if no_eos is not None and batch is not None and batch > 0:
            train_points["no_eos"].append((x_value, no_eos / batch))

    for (metric, x_value), (_step_value, value) in seen_eval.items():
        curves[metric].append((x_value, value))
    for metric in eval_keys:
        curves[metric].sort()
    for metric in ("reward", "length", "no_eos"):
        curves[metric] = _rolling(train_points[metric])
    return dict(curves)


def _mean_curve(
    seed_curves: dict[int, list[tuple[float, float]]], *, minimum_seeds: int = 1
) -> list[tuple[float, float, int]]:
    by_x: dict[float, list[float]] = defaultdict(list)
    for points in seed_curves.values():
        for x_value, value in points:
            by_x[round(x_value, 8)].append(value)
    return [
        (x_value, sum(values) / len(values), len(values))
        for x_value, values in sorted(by_x.items())
        if len(values) >= minimum_seeds
    ]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_savefig(fig: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        fig.savefig(temporary, **kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_compute_companion(
    all_curves: dict[tuple[str, int], dict[str, list[tuple[float, float]]]],
    output: Path,
    latest_preview: Path,
) -> None:
    """Render MATH in the same visual language as compute-divergence plots."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.edgecolor": "#1a1a1a",
            "axes.labelcolor": "#1a1a1a",
            "xtick.color": "#1a1a1a",
            "ytick.color": "#1a1a1a",
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0), sharex=True, squeeze=False)
    have_any = False
    have_treatment = any(
        arm == "maxent_dual" and any(points for points in metrics.values())
        for (arm, _seed), metrics in all_curves.items()
    )
    for axis, (metric, title, kind, limits) in zip(axes.flat, COMPUTE_PANELS):
        latest_metric_pass = -1.0
        for arm, _label, color, line_style in COMPUTE_METHODS:
            seed_curves = {
                seed: all_curves.get((arm, seed), {}).get(metric, [])
                for seed in EXPECTED_SEEDS
            }
            seed_curves = {
                seed: points for seed, points in seed_curves.items() if points
            }
            for points in seed_curves.values():
                have_any = True
                xs, ys = zip(*points)
                latest_metric_pass = max(latest_metric_pass, max(xs))
                sparse = len(points) <= 3
                axis.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=0.8 if sparse else 0.55,
                    alpha=0.6 if sparse else 0.3,
                    marker="o" if sparse else None,
                    markersize=2.3,
                    zorder=2,
                )
            mean_points = _mean_curve(seed_curves, minimum_seeds=2)
            if mean_points:
                xs = [point[0] for point in mean_points]
                ys = [point[1] for point in mean_points]
                axis.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=1.65,
                    linestyle=line_style,
                    marker="o" if len(xs) <= 25 else None,
                    markersize=2.5,
                    zorder=4,
                )
        axis.set_title(title, loc="left", fontsize=8.2)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(*limits)
        axis.xaxis.set_major_locator(MultipleLocator(0.25))
        for boundary in (0.25, 0.5, 0.75, 1.0):
            axis.axvline(
                boundary,
                color="#b8b8b8",
                linewidth=0.45,
                linestyle=":",
                zorder=0,
            )
        axis.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.7)
        if kind == "eval" and latest_metric_pass <= 1e-9 and metric == "pass8":
            axis.text(
                0.97,
                0.94,
                "INITIAL EVAL ONLY\nNEXT AT 0.25 PASS",
                transform=axis.transAxes,
                ha="right",
                va="top",
                color="#777777",
                fontsize=6.4,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.8,
                },
            )
    if not have_any:
        axes[0, 1].text(
            0.5,
            0.5,
            "WAITING FOR MATH TELEMETRY",
            transform=axes[0, 1].transAxes,
            ha="center",
            va="center",
            color="#777777",
            fontsize=7.3,
        )
    elif not have_treatment:
        axes[0, 1].text(
            0.5,
            0.5,
            "E26-v2 PENDING FIRST\nSHARED TELEMETRY",
            transform=axes[0, 1].transAxes,
            ha="center",
            va="center",
            color="#777777",
            fontsize=7.3,
        )

    handles = [
        Line2D([], [], color=color, linewidth=1.8, linestyle=line_style, label=label)
        for _arm, label, color, line_style in COMPUTE_METHODS
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=7.2,
        ncol=len(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.0,
    )
    fig.text(
        0.5,
        0.905,
        "MATH-500 · Qwen2.5-0.5B",
        ha="center",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.5,
        0.855,
        "E26-v2 treatment-only · Free-form token-policy MaxEnt",
        ha="center",
        fontsize=8.5,
        weight="bold",
        color="#1a1a1a",
    )
    fig.supxlabel(
        "training passes over the 8,515-prompt pool", fontsize=8.2, y=0.025
    )
    fig.text(
        0.5,
        0.006,
        "E21 Dr.GRPO is historical only; E26-v2 is the active treatment. Faint = seed, bold = mean of at least two seeds.",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.12,
        top=0.80,
        wspace=0.34,
        hspace=0.38,
    )
    _atomic_savefig(fig, output.with_suffix(".pdf"))
    _atomic_savefig(fig, output.with_suffix(".png"), dpi=200)
    _atomic_savefig(fig, latest_preview, dpi=200)
    plt.close(fig)


def build(
    prefix: str,
    control_prefix: str,
    data_root: Path,
    output: Path,
    curve_path: Path,
    compute_output: Path,
    compute_preview: Path,
) -> None:
    all_curves: dict[tuple[str, int], dict[str, list[tuple[float, float]]]] = {}
    source_paths: dict[tuple[str, int], str] = {}
    tidy: list[dict[str, Any]] = []
    for arm, _label, _color in METHODS:
        arm_prefix = control_prefix if arm == "grpo" else prefix
        for seed in EXPECTED_SEEDS:
            attempt = _discover_attempt(data_root, arm_prefix, arm, seed)
            if attempt is None:
                continue
            source, rows = attempt
            source_paths[(arm, seed)] = str(source.resolve())
            metrics = _extract(rows)
            all_curves[(arm, seed)] = metrics
            for metric, points in metrics.items():
                for x_value, value in points:
                    tidy.append(
                        {
                            "arm": arm,
                            "metric": metric,
                            "prefix": arm_prefix,
                            "seed": seed,
                            "training_passes": x_value,
                            "value": value,
                        }
                    )
    _atomic_json(curve_path, tidy)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.2), sharex=True, squeeze=False)
    axes_flat = list(axes.flat)
    have_any = False
    have_treatment = any(
        arm == "maxent_dual" and any(points for points in metrics.values())
        for (arm, _seed), metrics in all_curves.items()
    )
    for axis, (metric, title, kind, limits) in zip(axes_flat, PANELS):
        for arm, _label, color in METHODS:
            seed_curves = {
                seed: all_curves.get((arm, seed), {}).get(metric, [])
                for seed in EXPECTED_SEEDS
            }
            seed_curves = {seed: points for seed, points in seed_curves.items() if points}
            for seed, points in seed_curves.items():
                have_any = True
                xs, ys = zip(*points)
                axis.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=0.7,
                    alpha=0.22,
                    marker="o" if kind == "eval" else None,
                    markersize=2.5,
                )
            mean_points = _mean_curve(seed_curves)
            if mean_points:
                xs = [point[0] for point in mean_points]
                ys = [point[1] for point in mean_points]
                axis.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=1.8,
                    marker="o" if kind == "eval" else None,
                    markersize=3.2,
                    zorder=4,
                )
        axis.set_title(title, fontweight="semibold", pad=5)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(*limits)
        axis.xaxis.set_major_locator(MultipleLocator(0.25))
        for boundary in (0.25, 0.5, 0.75, 1.0):
            axis.axvline(boundary, color="#B8B8B8", linewidth=0.45, linestyle=":", zorder=0)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if metric in {"pass1", "pass8", "mean8", "reward", "no_eos"}:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for axis in axes[-1, :]:
        axis.set_xlabel("Training passes over 8,515 admitted prompts")

    if not have_any:
        axes_flat[0].text(
            0.5,
            0.5,
            "Waiting for E26-v2 telemetry",
            ha="center",
            va="center",
            transform=axes_flat[0].transAxes,
            color="#666666",
        )
    elif not have_treatment:
        axes_flat[0].text(
            0.5,
            0.5,
            "E26-v2 pending first shared telemetry",
            ha="center",
            va="center",
            transform=axes_flat[0].transAxes,
            color="#666666",
        )

    handles = [
        Line2D([0], [0], color=color, linewidth=2.0, label=label)
        for _arm, label, color in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.6,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(
        "MATH-500 · Qwen2.5-0.5B · free-form conditional-token MaxEnt",
        y=0.992,
        fontsize=11.0,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.965,
        f"E26-v2 treatment-only cohort · historical E21 reference · faint = seed, bold = available-seed mean · {timestamp}",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    fig.text(
        0.5,
        0.012,
        "Unrestricted token policy (not canonical-action MaxEnt); MATH-500 remains evaluation-only.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0.04, 0.035, 0.99, 0.925), h_pad=1.1, w_pad=1.0)
    _atomic_savefig(fig, output.with_suffix(".pdf"), bbox_inches="tight")
    _atomic_savefig(fig, output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    _render_compute_companion(all_curves, compute_output, compute_preview)
    print(
        json.dumps(
            {
                "compute_pdf": str(compute_output.with_suffix('.pdf').resolve()),
                "compute_png": str(compute_output.with_suffix('.png').resolve()),
                "compute_preview": str(compute_preview.resolve()),
                "curve": str(curve_path.resolve()),
                "pdf": str(output.with_suffix('.pdf').resolve()),
                "png": str(output.with_suffix('.png').resolve()),
                "run_attempts": len(source_paths),
                "points": len(tidy),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--control-prefix", default=DEFAULT_CONTROL_PREFIX)
    parser.add_argument("--data-root", type=Path, default=ROOT / "var/data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--curve", type=Path, default=DEFAULT_CURVE)
    parser.add_argument(
        "--compute-output", type=Path, default=DEFAULT_COMPUTE_OUTPUT
    )
    parser.add_argument(
        "--compute-preview", type=Path, default=DEFAULT_COMPUTE_PREVIEW
    )
    args = parser.parse_args()
    build(
        args.prefix,
        args.control_prefix,
        args.data_root,
        args.output,
        args.curve,
        args.compute_output,
        args.compute_preview,
    )


if __name__ == "__main__":
    main()
