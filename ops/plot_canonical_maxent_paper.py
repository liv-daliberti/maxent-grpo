#!/usr/bin/env python3
"""Plot scale-separated canonical-action MaxEnt trajectories for the paper.

Each scale is one two-row figure, with one domain per row and the four
evaluation metrics in columns.  The manuscript centers the 3B figure and uses
the completed 0.5B cohort as an appendix replication.  No visual averaging or
line-style encoding is allowed across tasks or scales.  Every figure contains
only finite canonical-action policies:
canonical Dr.GRPO (alpha=0), fixed-coefficient MaxEnt, proportional entropy
control, and Haarnoja-style dual control.  Matched 0.5B and 3B canonical
Dr.GRPO artifacts are added whenever they exist.  Missing
runs/checkpoints are omitted rather than imputed.

Inputs are the tidy curve JSON files produced by
``ops/exp_scaling/parse_scaling_curve.py``.  Regenerate with::

    python ops/plot_canonical_maxent_paper.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "paper/figures/canonical_maxent_trajectories"
EXPECTED_SEEDS = frozenset({43, 44, 45})
MAX_TRAINING_PASSES = 5.0
INTERIM_3B_RESULT = (
    ROOT / "paper/results/canonical_maxent_vs_reward_only_3b_interim.json"
)

DOMAINS = (
    (
        "Graph coloring",
        "graph_coloring",
        ROOT / "var/artifacts/gce16_canonical_maxent_05b_v2_scaling_curve.json",
        ROOT / "var/artifacts/gce17_canonical_maxent_3b_v5_scaling_curve.json",
        ROOT / "var/artifacts/gce19_canonical_drgrpo_05b_v1_scaling_curve.json",
        ROOT / "var/artifacts/gce18_canonical_drgrpo_3b_v1_scaling_curve.json",
    ),
    (
        "Countdown",
        "countdown",
        ROOT / "var/artifacts/cde16_canonical_maxent_05b_v2_scaling_curve.json",
        ROOT / "var/artifacts/cde17_canonical_maxent_3b_v5_scaling_curve.json",
        ROOT / "var/artifacts/cde19_canonical_drgrpo_05b_v1_scaling_curve.json",
        ROOT / "var/artifacts/cde18_canonical_drgrpo_3b_v1_scaling_curve.json",
    ),
)


def _load_common_3b_horizons() -> dict[str, float]:
    """Read plot markers from the frozen artifact that supports the paper."""

    payload = json.loads(INTERIM_3B_RESULT.read_text(encoding="utf-8"))
    domain_results = payload["domains"]
    horizons: dict[str, float] = {}
    for domain, domain_slug, *_paths in DOMAINS:
        horizon = float(domain_results[domain_slug]["common_horizon_passes"])
        if not 0.0 <= horizon <= MAX_TRAINING_PASSES:
            raise ValueError(f"invalid frozen 3B horizon for {domain}: {horizon}")
        horizons[domain] = horizon
    return horizons


COMMON_3B_HORIZONS = _load_common_3b_horizons()

METHODS = (
    ("grpo", r"Canonical Dr.GRPO ($\alpha=0$)", "#666666"),
    ("maxent", "Fixed", "#0072B2"),
    ("maxent_control", "Proportional", "#D55E00"),
    ("maxent_dual", "Haarnoja dual", "#009E73"),
)
METHOD_KEYS = frozenset(item[0] for item in METHODS)
METRICS = (
    ("greedy", "pass@1", (0.0, 1.0)),
    ("pass8", "pass@8", (0.0, 1.0)),
    ("coverage8", "coverage@8", (0.0, 0.5)),
    ("distinct8", "distinct@8", (0.0, 3.0)),
)
SCALES = (("0.5B", "05b"), ("3B", "3b"))


def _read_json_array(path: Path, *, required: bool, retries: int = 4) -> list[dict[str, Any]]:
    """Read a live curve artifact, retrying a concurrent partial publication."""

    error: Exception | None = None
    for attempt in range(retries):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"expected a JSON array, got {type(payload).__name__}")
            return [row for row in payload if isinstance(row, dict)]
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(0.15 * (attempt + 1))

    message = f"could not read curve artifact {path}: {error}"
    if required:
        raise RuntimeError(message) from error
    # Optional live artifacts may not have landed yet.  An absent baseline is
    # genuinely missing data, so omit it without manufacturing a placeholder.
    return []


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_source(path: Path, *, required: bool) -> list[dict[str, Any]]:
    """Filter one curve file to eligible multi-answer analytical records."""

    rows = _read_json_array(path, required=required)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split") != "multi_answer" or row.get("arm") not in METHOD_KEYS:
            continue
        try:
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        passes = _finite_float(row.get("training_passes"))
        if seed not in EXPECTED_SEEDS or passes is None:
            continue
        if not -1e-9 <= passes <= MAX_TRAINING_PASSES + 1e-9:
            continue
        clean = dict(row)
        clean["seed"] = seed
        clean["training_passes"] = min(MAX_TRAINING_PASSES, max(0.0, passes))
        filtered.append(clean)
    return filtered


def _deduplicate(
    rows: list[dict[str, Any]], metric: str
) -> dict[tuple[str, int], list[tuple[float, float]]]:
    """Return sorted seed curves, choosing one record per checkpoint."""

    points: dict[tuple[str, int, float], tuple[float, float]] = {}
    for row in rows:
        value = _finite_float(row.get(metric))
        if value is None:
            continue
        arm = str(row["arm"])
        seed = int(row["seed"])
        x_value = float(row["training_passes"])
        step = _finite_float(row.get("step")) or -1.0
        key = (arm, seed, x_value)
        previous = points.get(key)
        if previous is None or step >= previous[0]:
            points[key] = (step, value)

    curves: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for (arm, seed, x_value), (_, value) in points.items():
        curves[(arm, seed)].append((x_value, value))
    for key in curves:
        curves[key].sort()
    return curves


def _mean_curve(
    curves: dict[tuple[str, int], list[tuple[float, float]]],
    arm: str,
    *,
    minimum_seeds: int = 2,
) -> list[tuple[float, float]]:
    """Average only observed values; never replace a missing seed with zero."""

    by_x: dict[float, list[float]] = defaultdict(list)
    for (candidate_arm, _seed), points in curves.items():
        if candidate_arm != arm:
            continue
        for x_value, value in points:
            by_x[round(x_value, 8)].append(value)
    return [
        (x_value, sum(values) / len(values))
        for x_value, values in sorted(by_x.items())
        if len(values) >= minimum_seeds
    ]


def _axis_upper(
    default_limits: tuple[float, float],
    *curve_sets: dict[tuple[str, int], list[tuple[float, float]]],
) -> float:
    values = [
        value
        for curves in curve_sets
        for points in curves.values()
        for _x_value, value in points
        if value >= 0
    ]
    observed = max(values, default=0.0)
    upper = max(default_limits[1], observed * 1.08)
    if default_limits[1] <= 1.0:
        upper = min(1.0, upper)
    return upper


def _atomic_savefig(fig: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        fig.savefig(temporary, **kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cell_output(output_prefix: Path, domain_slug: str, scale_slug: str) -> Path:
    return output_prefix.with_name(
        f"{output_prefix.name}_{domain_slug}_{scale_slug}"
    )


def _plot_metric_axis(
    axis: Any,
    cell_curves: dict[tuple[str, int], list[tuple[float, float]]],
    metric_label: str,
    default_limits: tuple[float, float],
    metric_upper: float,
    *,
    show_xlabel: bool,
) -> None:
    """Draw seed and available-seed-mean curves for one metric panel."""

    for arm, _label, color in METHODS:
        for seed in sorted(EXPECTED_SEEDS):
            points = cell_curves.get((arm, seed), [])
            if not points:
                continue
            xs, ys = zip(*points)
            axis.plot(
                xs,
                ys,
                color=color,
                linewidth=0.65,
                alpha=0.22,
                zorder=1,
            )
        mean_points = _mean_curve(cell_curves, arm)
        if mean_points:
            xs, ys = zip(*mean_points)
            axis.plot(
                xs,
                ys,
                color=color,
                linewidth=2.05,
                solid_capstyle="round",
                zorder=3,
            )

    axis.set_title(metric_label, fontweight="semibold", pad=5)
    axis.set_xlim(0.0, MAX_TRAINING_PASSES)
    axis.set_ylim(default_limits[0], metric_upper)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.55, alpha=0.75)
    axis.grid(axis="x", color="#eeeeee", linewidth=0.45, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.5, width=0.6)
    if show_xlabel:
        axis.set_xlabel("Training passes over prompt pool")


def _legend_handles(rows: list[dict[str, Any]]) -> list[Line2D]:
    present_arms = {
        str(row["arm"])
        for row in rows
        if row.get("arm") in METHOD_KEYS
    }
    handles = [
        Line2D([0], [0], color=color, linewidth=2.2, label=label)
        for arm, label, color in METHODS
        if arm in present_arms
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#777777",
                linewidth=0.65,
                alpha=0.45,
                label="Seed",
            ),
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=2.05,
                label="Available-seed mean",
            ),
        ]
    )
    return handles


def _build_scale_figure(
    output_prefix: Path,
    data: dict[tuple[str, str], list[dict[str, Any]]],
    curves: dict[tuple[str, str, str], dict[tuple[str, int], list[tuple[float, float]]]],
    metric_uppers: dict[str, float],
    *,
    scale: str,
    scale_slug: str,
    interim: bool,
) -> list[Path]:
    """Build one two-domain trajectory figure without pooling domains."""

    fig, axes = plt.subplots(
        nrows=len(DOMAINS),
        ncols=len(METRICS),
        figsize=(8.4, 4.75),
        sharex=True,
        squeeze=False,
    )
    combined_rows: list[dict[str, Any]] = []
    for row_index, (domain, _domain_slug, *_paths) in enumerate(DOMAINS):
        combined_rows.extend(data[(domain, scale)])
        for axis, (metric, metric_label, default_limits) in zip(
            axes[row_index], METRICS
        ):
            _plot_metric_axis(
                axis,
                curves[(domain, scale, metric)],
                metric_label,
                default_limits,
                metric_uppers[metric],
                show_xlabel=False,
            )
            if interim:
                axis.axvline(
                    COMMON_3B_HORIZONS[domain],
                    color="#222222",
                    linestyle=(0, (3, 2)),
                    linewidth=0.8,
                    alpha=0.72,
                    zorder=2,
                )
            axis.tick_params(labelsize=7.5)
            if row_index == 0:
                axis.title.set_fontsize(9.5)
            else:
                axis.set_title("")
        axes[row_index, 0].set_ylabel(
            domain,
            fontsize=9.5,
            fontweight="semibold",
            labelpad=11,
        )

    present_arms = {
        str(row["arm"])
        for row in combined_rows
        if row.get("arm") in METHOD_KEYS
    }
    handles = [
        Line2D([0], [0], color=color, linewidth=2.2, label=label)
        for arm, label, color in METHODS
        if arm in present_arms
    ]
    if interim:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#222222",
                linestyle=(0, (3, 2)),
                linewidth=0.9,
                label="Paired horizon",
            )
        )
    fig.suptitle(
        f"Qwen2.5-{scale} · canonical-action RL",
        y=0.985,
        fontsize=11.0,
        fontweight="semibold",
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=len(handles),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.15,
        fontsize=8.0,
    )
    fig.supxlabel("Training passes over prompt pool", y=0.025, fontsize=8.5)
    if interim:
        fig.text(
            0.995,
            0.012,
            "Incomplete horizons are not imputed.",
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#555555",
        )
    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.11,
        top=0.84,
        wspace=0.22,
        hspace=0.30,
    )
    output_base = output_prefix.with_name(f"{output_prefix.name}_{scale_slug}")
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    _atomic_savefig(fig, pdf_path, bbox_inches="tight")
    _atomic_savefig(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def build_figure(output_prefix: Path) -> list[Path]:
    """Build the main 3B figure and the appendix 0.5B replication figure."""

    data: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for domain, _slug, e16_path, e17_path, e19_path, e18_path in DOMAINS:
        data[(domain, "0.5B")] = _load_source(e16_path, required=True)
        data[(domain, "0.5B")].extend(_load_source(e19_path, required=False))
        data[(domain, "3B")] = _load_source(e17_path, required=False)
        data[(domain, "3B")].extend(_load_source(e18_path, required=False))

    curves = {
        (domain, scale, metric): _deduplicate(rows, metric)
        for (domain, scale), rows in data.items()
        for metric, _label, _limits in METRICS
    }
    metric_uppers = {
        metric: _axis_upper(
            limits,
            *(curves[(domain, scale, metric)] for domain, _slug, *_ in DOMAINS for scale, _scale_slug in SCALES),
        )
        for metric, _label, limits in METRICS
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11.0,
            "axes.titlesize": 13.0,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    outputs: list[Path] = []
    outputs.extend(
        _build_scale_figure(
            output_prefix,
            data,
            curves,
            metric_uppers,
            scale="3B",
            scale_slug="3b",
            interim=True,
        )
    )
    outputs.extend(
        _build_scale_figure(
            output_prefix,
            data,
            curves,
            metric_uppers,
            scale="0.5B",
            scale_slug="05b",
            interim=False,
        )
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output prefix; combined 3b and 05b suffixes are added "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in build_figure(args.output_base.resolve()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
