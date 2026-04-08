#!/usr/bin/env python3
"""Aggregate multiseed SEED eval summaries and render CI curve figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METRIC_SPECS: dict[str, tuple[str, str]] = {
    "pass_at_1": ("avg", "Pass@1"),
    "pass_at_8": ("pass_at_8_avg", "Pass@8"),
    "mean_at_8": ("mean_at_8_avg", "Mean@8"),
}
TEMPLATE_ORDER = ["no", "qwen_math", "r1"]
TEMPLATE_LABELS = {
    "no": "no",
    "qwen_math": "qwen",
    "r1": "r1",
}
TEMPLATE_COLORS = {
    "no": "#1f77b4",
    "qwen_math": "#d97706",
    "r1": "#16a34a",
}
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=None,
        help="Curve figure path. Defaults to <results-root>/multiseed_curves.svg.",
    )
    parser.add_argument(
        "--template-grid-svg",
        type=Path,
        default=None,
        help=(
            "3x3 template/metric grid figure path. Defaults to "
            "<results-root>/multiseed_curves_template_grid.svg."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Aggregate JSON path. Defaults to <results-root>/multiseed_curves.summary.json.",
    )
    parser.add_argument(
        "--rows-tsv",
        type=Path,
        default=None,
        help="Flat TSV rows path. Defaults to <results-root>/multiseed_curves.tsv.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Defaults to the results-root directory name.",
    )
    return parser.parse_args(argv)


def _template_sort_key(template: str) -> tuple[int, str]:
    try:
        return (TEMPLATE_ORDER.index(template), template)
    except ValueError:
        return (len(TEMPLATE_ORDER), template)


def _template_label(template: str) -> str:
    return TEMPLATE_LABELS.get(template, template)


def _template_color(template: str) -> str:
    return TEMPLATE_COLORS.get(template, "#6b7280")


def _parse_seed_dir(name: str) -> int | None:
    normalized = name.strip()
    for prefix in ("seed_", "seed-"):
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :]
            if suffix.lstrip("-").isdigit():
                return int(suffix)
    if normalized.startswith("seed") and normalized[4:].lstrip("-").isdigit():
        return int(normalized[4:])
    return None


def _parse_step_dir(name: str) -> int | None:
    normalized = name.strip()
    for prefix in ("step", "step_", "step-"):
        if normalized.startswith(prefix):
            digits = normalized[len(prefix) :]
            if digits.isdigit():
                return int(digits)
    return None


def _candidate_summary_paths(results_root: Path) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in (
        "seed_*/*/*/seed_paper_eval_sharded.summary.json",
        "seed-*/*/*/seed_paper_eval_sharded.summary.json",
    ):
        for path in sorted(results_root.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            matches.append(path)
    return matches


def load_summary_records(results_root: Path) -> list[dict[str, Any]]:
    results_root = results_root.resolve()
    records: list[dict[str, Any]] = []
    for summary_path in _candidate_summary_paths(results_root):
        rel_parts = summary_path.relative_to(results_root).parts
        if len(rel_parts) != 4:
            continue
        seed_dir, step_dir, template, _ = rel_parts
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        seed = payload.get("sampling_seed")
        if seed is None:
            seed = _parse_seed_dir(seed_dir)
        step = _parse_step_dir(step_dir)
        if step is None:
            continue
        try:
            seed_value = None if seed is None else int(seed)
        except (TypeError, ValueError):
            seed_value = _parse_seed_dir(seed_dir)
        record = {
            "sampling_seed": seed_value,
            "step": int(step),
            "template": str(template),
            "summary_path": str(summary_path),
        }
        for metric_name, (summary_key, _) in METRIC_SPECS.items():
            raw_value = payload.get(summary_key)
            try:
                record[metric_name] = None if raw_value is None else float(raw_value)
            except (TypeError, ValueError):
                record[metric_name] = None
        records.append(record)
    if not records:
        raise FileNotFoundError(
            f"No merged seed summaries found under {results_root}"
        )
    return sorted(
        records,
        key=lambda row: (
            _template_sort_key(str(row["template"])),
            int(row["step"]),
            -1 if row["sampling_seed"] is None else int(row["sampling_seed"]),
        ),
    )


def _critical_t_95(sample_count: int) -> float:
    if sample_count <= 1:
        return 0.0
    degrees = sample_count - 1
    if degrees in T_CRITICAL_95:
        return T_CRITICAL_95[degrees]
    return 1.96


def _confidence_interval_95(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean = float(sum(values) / len(values))
    if len(values) <= 1:
        return mean, mean
    margin = (
        _critical_t_95(len(values))
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )
    return (
        max(0.0, mean - float(margin)),
        min(1.0, mean + float(margin)),
    )


def aggregate_records(
    records: list[dict[str, Any]],
    *,
    results_root: Path,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[tuple[int | None, float]]] = {}
    templates = sorted(
        {str(record["template"]) for record in records},
        key=_template_sort_key,
    )
    seeds = sorted(
        {
            int(record["sampling_seed"])
            for record in records
            if record.get("sampling_seed") is not None
        }
    )

    for record in records:
        for metric_name in METRIC_SPECS:
            value = record.get(metric_name)
            if value is None:
                continue
            key = (metric_name, str(record["template"]), int(record["step"]))
            grouped.setdefault(key, []).append(
                (
                    None
                    if record.get("sampling_seed") is None
                    else int(record["sampling_seed"]),
                    float(value),
                )
            )

    metric_payloads: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for metric_name, (_, label) in METRIC_SPECS.items():
        template_payloads: dict[str, list[dict[str, Any]]] = {}
        for template in templates:
            steps = sorted(
                step
                for grouped_metric, grouped_template, step in grouped
                if grouped_metric == metric_name and grouped_template == template
            )
            if not steps:
                continue
            points: list[dict[str, Any]] = []
            for step in steps:
                values_by_seed = sorted(
                    grouped[(metric_name, template, step)],
                    key=lambda item: (
                        -1 if item[0] is None else item[0],
                        item[1],
                    ),
                )
                values = [value for _, value in values_by_seed]
                ci_low, ci_high = _confidence_interval_95(values)
                mean = float(sum(values) / len(values))
                point = {
                    "step": int(step),
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "seed_count": len(values),
                    "seeds": [
                        int(seed)
                        for seed, _ in values_by_seed
                        if seed is not None
                    ],
                    "values": values,
                }
                points.append(point)
                rows.append(
                    {
                        "metric": metric_name,
                        "template": template,
                        "step": int(step),
                        "mean": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "seed_count": len(values),
                        "seeds": ",".join(str(seed) for seed in point["seeds"]),
                        "values": ",".join(f"{value:.10f}" for value in values),
                    }
                )
            template_payloads[template] = points
        metric_payloads[metric_name] = {
            "label": label,
            "templates": template_payloads,
        }

    return {
        "results_root": str(results_root.resolve()),
        "summary_count": len(records),
        "seeds": seeds,
        "templates": templates,
        "metrics": metric_payloads,
        "rows": rows,
    }


def write_summary_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "template",
        "step",
        "mean",
        "ci_low",
        "ci_high",
        "seed_count",
        "seeds",
        "values",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def _plot_points(
    ax: Any,
    points: list[dict[str, Any]],
    *,
    color: str,
) -> None:
    xs = [int(point["step"]) for point in points]
    ys = [float(point["mean"]) for point in points]
    ci_lows = [float(point["ci_low"]) for point in points]
    ci_highs = [float(point["ci_high"]) for point in points]
    ax.fill_between(xs, ci_lows, ci_highs, color=color, alpha=0.18)
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2.0,
        marker="o",
        markersize=3.8,
    )


def plot_curves(payload: dict[str, Any], output_path: Path, *, title: str) -> None:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True, sharey=True)

    for ax, metric_name in zip(axes, METRIC_SPECS):
        metric_payload = payload["metrics"][metric_name]
        for template in payload["templates"]:
            points = metric_payload["templates"].get(template, [])
            if not points:
                continue
            _plot_points(ax, points, color=_template_color(template))
        ax.set_title(metric_payload["label"])
        ax.set_xlabel("Checkpoint step")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Suite average")

    legend_templates = [
        template
        for template in payload["templates"]
        if any(
            payload["metrics"][metric_name]["templates"].get(template)
            for metric_name in METRIC_SPECS
        )
    ]
    if legend_templates:
        handles = [
            Line2D(
                [0],
                [0],
                color=_template_color(template),
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                label=_template_label(template),
            )
            for template in legend_templates
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, 1.07),
        )

    fig.suptitle(title, y=1.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=output_path.suffix.lstrip(".") or "svg", dpi=200)
    plt.close(fig)


def plot_template_metric_grid(
    payload: dict[str, Any],
    output_path: Path,
    *,
    title: str,
) -> None:
    _configure_matplotlib()
    templates = [
        template
        for template in payload["templates"]
        if any(
            payload["metrics"][metric_name]["templates"].get(template)
            for metric_name in METRIC_SPECS
        )
    ]
    metric_names = list(METRIC_SPECS)
    fig, axes = plt.subplots(
        len(templates),
        len(metric_names),
        figsize=(13.2, 11.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    if len(templates) == 1 and len(metric_names) == 1:
        axes_grid = [[axes]]
    elif len(templates) == 1:
        axes_grid = [list(axes)]
    elif len(metric_names) == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = axes

    for row_index, template in enumerate(templates):
        for col_index, metric_name in enumerate(metric_names):
            ax = axes_grid[row_index][col_index]
            points = payload["metrics"][metric_name]["templates"].get(template, [])
            if points:
                _plot_points(ax, points, color=_template_color(template))
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            ax.set_title(
                f"{_template_label(template)} • {payload['metrics'][metric_name]['label']}"
            )
            if row_index == len(templates) - 1:
                ax.set_xlabel("Checkpoint step")
            if col_index == 0:
                ax.set_ylabel("Suite average")

    fig.suptitle(title, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=output_path.suffix.lstrip(".") or "svg", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    results_root = args.results_root.resolve()
    output_svg = args.output_svg or (results_root / "multiseed_curves.svg")
    template_grid_svg = args.template_grid_svg or (
        results_root / "multiseed_curves_template_grid.svg"
    )
    summary_json = args.summary_json or (
        results_root / "multiseed_curves.summary.json"
    )
    rows_tsv = args.rows_tsv or (results_root / "multiseed_curves.tsv")
    title = args.title or f"SEED multiseed curves: {results_root.name}"

    records = load_summary_records(results_root)
    payload = aggregate_records(records, results_root=results_root)
    write_summary_json(payload, summary_json)
    write_rows_tsv(payload["rows"], rows_tsv)
    plot_curves(payload, output_svg, title=title)
    plot_template_metric_grid(payload, template_grid_svg, title=title)

    print(f"Wrote summary json: {summary_json}")
    print(f"Wrote rows tsv: {rows_tsv}")
    print(f"Wrote figure: {output_svg}")
    print(f"Wrote template grid figure: {template_grid_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
