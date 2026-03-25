#!/usr/bin/env python3
"""Render a live dashboard for the in-progress listwise tau/beta sweep."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import listwise_sweep_report as report


STATUS_FILL = {
    "pending": "#f3f4f6",
    "running_no_summary": "#e0f2fe",
    "running_with_summary": "#eff6ff",
    "done": "#f8fafc",
    "other": "#f5f5f5",
}

METRIC_PALETTE = [
    "#f7fcf5",
    "#e5f5e0",
    "#c7e9c0",
    "#a1d99b",
    "#74c476",
    "#41ab5d",
    "#238b45",
]

LENGTH_PALETTE = [
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the current state of an in-progress listwise tau/beta sweep."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "var" / "artifacts" / "seed_paper_eval" / "live",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Requested dashboard artifact path. The script always writes both SVG and PNG versions.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="Listwise Tau/Beta Sweep",
        help="Top-level title for the dashboard.",
    )
    return parser.parse_args()


def _load_scheduler_states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    try:
        completed = subprocess.run(
            [
                "squeue",
                "-h",
                "-j",
                ",".join(job_ids),
                "-o",
                "%i|%T",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    states: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        job_id, state = line.split("|", 1)
        states[job_id.strip()] = state.strip().upper()
    return states


def _status_bucket(row: dict[str, Any], scheduler_state: str | None) -> str:
    if scheduler_state == "RUNNING":
        return "running_with_summary" if row.get("avg") is not None else "running_no_summary"
    if scheduler_state is not None:
        return "pending"
    if row.get("avg") is not None:
        return "done"
    return "pending"


def _status_label(row: dict[str, Any], scheduler_state: str | None) -> str:
    step = row.get("step")
    if step is None:
        step = row.get("train_step")
    if scheduler_state == "RUNNING":
        if step is not None:
            return f"step {int(step)} RUN"
        return "RUN"
    if scheduler_state is not None:
        if scheduler_state == "PENDING":
            return "PENDING"
        return scheduler_state
    if step is not None:
        return f"step {int(step)} DONE" if row.get("step") is not None else f"step {int(step)}"
    return "PENDING"


def _augment_rows(
    rows: list[dict[str, Any]],
    scheduler_states: dict[str, str],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for row in rows:
        scheduler_state = scheduler_states.get(str(row["job_id"]))
        copied = dict(row)
        copied["scheduler_state"] = scheduler_state
        copied["status_bucket"] = _status_bucket(row, scheduler_state)
        copied["status_label"] = _status_label(row, scheduler_state)
        augmented.append(copied)
    return augmented


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"done": 0, "running": 0, "pending": 0}
    for row in rows:
        bucket = str(row["status_bucket"])
        if bucket.startswith("running"):
            counts["running"] += 1
        elif bucket == "done":
            counts["done"] += 1
        else:
            counts["pending"] += 1
    return counts


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    available = [row for row in rows if row.get("avg") is not None]
    if not available:
        return None
    available.sort(key=report.rank_key, reverse=True)
    return available[0]


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_range(rows: list[dict[str, Any]], metric: str) -> tuple[float, float] | None:
    values = [_metric_value(row, metric) for row in rows]
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    lo = min(filtered)
    hi = max(filtered)
    if math.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    return lo, hi


def _interp_palette(value: float, lo: float, hi: float, palette: list[str], *, reverse: bool = False) -> str:
    frac = 0.5 if math.isclose(lo, hi) else (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    if reverse:
        frac = 1.0 - frac
    idx = int(round(frac * (len(palette) - 1)))
    return palette[idx]


def _text(x: float, y: float, text: str, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {extra}>{html.escape(text)}</text>'


def _line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {extra}/>'


def _rect(x: float, y: float, width: float, height: float, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" {extra}/>'


def _panel(parts: list[str], x0: float, y0: float, width: float, height: float, title: str) -> tuple[float, float, float, float]:
    parts.append(_rect(x0, y0, width, height, fill="white", stroke="#d4d4d8", **{"stroke-width": "1"}))
    parts.append(_text(x0 + 12, y0 + 22, title, **{"font-size": "15", "font-weight": "700"}))
    left = x0 + 66
    top = y0 + 42
    plot_w = width - 88
    plot_h = height - 82
    return left, top, plot_w, plot_h


def _draw_heatmap(
    parts: list[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    rows: list[dict[str, Any]],
    taus: list[float],
    betas: list[float],
    metric: str,
    reverse_colors: bool = False,
    value_format: str = ".4f",
    subtitle: str,
) -> None:
    left, top, plot_w, plot_h = _panel(parts, x0, y0, width, height, title)
    parts.append(_text(left, y0 + height - 10, subtitle, **{"font-size": "11", "fill": "#666"}))
    cell_w = plot_w / max(len(betas), 1)
    cell_h = plot_h / max(len(taus), 1)
    row_map = {(float(row["tau"]), float(row["beta"])): row for row in rows}
    metric_span = _metric_range(rows, metric)

    for col_idx, beta in enumerate(betas):
        x = left + col_idx * cell_w + cell_w / 2.0
        parts.append(_text(x, top - 10, f"β={beta:.2f}", **{"font-size": "11", "text-anchor": "middle", "fill": "#444"}))
    for row_idx, tau in enumerate(taus):
        y = top + row_idx * cell_h + cell_h / 2.0 + 4
        parts.append(_text(left - 10, y, f"τ={tau:.2f}", **{"font-size": "11", "text-anchor": "end", "fill": "#444"}))

    best = _best_row(rows)
    best_key = None if best is None else (float(best["tau"]), float(best["beta"]))

    for row_idx, tau in enumerate(taus):
        for col_idx, beta in enumerate(betas):
            x = left + col_idx * cell_w
            y = top + row_idx * cell_h
            row = row_map.get(
                (tau, beta),
                {
                    "tau": tau,
                    "beta": beta,
                    "avg": None,
                    "pass_at_8_avg": None,
                    "mean_at_8_avg": None,
                    "avg_len_mean": None,
                    "status_bucket": "pending",
                    "status_label": "PENDING",
                },
            )
            value = _metric_value(row, metric)
            fill = STATUS_FILL[str(row["status_bucket"])]
            if value is not None and metric_span is not None:
                fill = _interp_palette(
                    value,
                    metric_span[0],
                    metric_span[1],
                    LENGTH_PALETTE if metric == "avg_len_mean" else METRIC_PALETTE,
                    reverse=reverse_colors,
                )
            stroke = "#94a3b8"
            stroke_width = "1"
            if best_key == (tau, beta) and value is not None:
                stroke = "#b45309"
                stroke_width = "2"
            parts.append(
                _rect(
                    x,
                    y,
                    cell_w - 6,
                    cell_h - 6,
                    fill=fill,
                    stroke=stroke,
                    **{"stroke-width": stroke_width, "rx": "6", "ry": "6"},
                )
            )
            center_x = x + (cell_w - 6) / 2.0
            if value is None:
                value_text = "--"
            else:
                value_text = format(value, value_format)
            parts.append(
                _text(
                    center_x,
                    y + 24,
                    value_text,
                    **{"font-size": "14", "font-weight": "700", "text-anchor": "middle", "fill": "#111827"},
                )
            )
            parts.append(
                _text(
                    center_x,
                    y + 42,
                    str(row["status_label"]),
                    **{"font-size": "10", "text-anchor": "middle", "fill": "#374151"},
                )
            )


def _draw_leaderboard(
    parts: list[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    rows: list[dict[str, Any]],
) -> None:
    left, top, plot_w, plot_h = _panel(parts, x0, y0, width, height, title)
    completed = [row for row in rows if row.get("avg") is not None]
    completed.sort(key=report.rank_key, reverse=True)
    counts = _status_counts(rows)
    best = _best_row(rows)

    summary_lines = [
        f"Complete: {counts['done']}  Running: {counts['running']}  Pending: {counts['pending']}",
    ]
    if best is not None:
        summary_lines.append(
            "Best so far: "
            f"τ={float(best['tau']):.2f}, β={float(best['beta']):.2f}, "
            f"avg={float(best['avg']):.4f}, pass@8={float(best['pass_at_8_avg']):.4f}, "
            f"seeds={int(best.get('seed_completed_count', 1))}/{int(best.get('seed_count', 1))}"
        )
    else:
        summary_lines.append("Best so far: no completed evals yet.")
    summary_lines.append(
        "Cells show the mean over completed seeds at each sweep point; "
        "status reports completed versus total seeds and falls back to the latest train step."
    )

    for idx, line in enumerate(summary_lines):
        parts.append(_text(left, top + 18 * idx, line, **{"font-size": "12", "fill": "#374151"}))

    table_top = top + 76
    headers = ["Rank", "(τ, β)", "seeds", "step", "avg", "pass@8", "mean@8", "avg len", "status"]
    col_x = [
        left,
        left + 52,
        left + 136,
        left + 208,
        left + 268,
        left + 348,
        left + 438,
        left + 528,
        left + 632,
    ]
    for x, header in zip(col_x, headers):
        parts.append(_text(x, table_top, header, **{"font-size": "11", "font-weight": "700", "fill": "#111827"}))
    parts.append(_line(left, table_top + 8, left + min(plot_w, 650), table_top + 8, stroke="#d4d4d8", **{"stroke-width": "1"}))

    for idx, row in enumerate(completed[:5], start=1):
        y = table_top + idx * 22
        values = [
            str(idx),
            f"({float(row['tau']):.2f}, {float(row['beta']):.2f})",
            f"{int(row.get('seed_completed_count', 1))}/{int(row.get('seed_count', 1))}",
            str(int(row["step"])) if row.get("step") is not None else "",
            f"{float(row['avg']):.4f}",
            f"{float(row['pass_at_8_avg']):.4f}",
            f"{float(row['mean_at_8_avg']):.4f}",
            f"{float(row['avg_len_mean']):.1f}",
            str(row["status_label"]),
        ]
        for x, value in zip(col_x, values):
            parts.append(_text(x, y, value, **{"font-size": "11", "fill": "#374151"}))

    legend_y = y0 + height - 42
    legend_items = [
        ("done", "Completed"),
        ("running_with_summary", "Running with latest eval"),
        ("pending", "Pending / no eval yet"),
    ]
    cursor_x = left
    for bucket, label in legend_items:
        parts.append(_rect(cursor_x, legend_y - 12, 16, 12, fill=STATUS_FILL[bucket], stroke="#94a3b8", **{"stroke-width": "1"}))
        parts.append(_text(cursor_x + 22, legend_y - 2, label, **{"font-size": "11", "fill": "#374151"}))
        cursor_x += 180


def _plot_svg(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    width = 1420
    height = 1020
    outer = 26
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        _text(outer, 28, title, **{"font-size": "20", "font-weight": "700", "fill": "#111827"}),
        _text(
            outer,
            50,
            "Live automl dashboard for the listwise tau/beta sweep. Pending cells are intentionally left unscored.",
            **{"font-size": "12", "fill": "#4b5563"},
        ),
    ]
    panel_gap = 22
    top_y = 74
    half_w = (width - 2 * outer - panel_gap) / 2.0
    panel_h = 330
    bottom_y = top_y + panel_h + panel_gap
    last_y = bottom_y + panel_h + panel_gap
    leaderboard_h = height - last_y - outer

    taus = sorted({float(row["tau"]) for row in rows})
    betas = sorted({float(row["beta"]) for row in rows})

    _draw_heatmap(
        parts,
        x0=outer,
        y0=top_y,
        width=half_w,
        height=panel_h,
        title="Current Avg",
        rows=rows,
        taus=taus,
        betas=betas,
        metric="avg",
        subtitle="Higher is better. Value shown is the mean across completed seeds for that point.",
    )
    _draw_heatmap(
        parts,
        x0=outer + half_w + panel_gap,
        y0=top_y,
        width=half_w,
        height=panel_h,
        title="Current Pass@8 Avg",
        rows=rows,
        taus=taus,
        betas=betas,
        metric="pass_at_8_avg",
        subtitle="Higher is better. This tracks sampled success over 8 responses per prompt, averaged across completed seeds.",
    )
    _draw_heatmap(
        parts,
        x0=outer,
        y0=bottom_y,
        width=half_w,
        height=panel_h,
        title="Current Mean@8 Avg",
        rows=rows,
        taus=taus,
        betas=betas,
        metric="mean_at_8_avg",
        subtitle="Higher is better. This captures average success rate across the 8 sampled responses, averaged across completed seeds.",
    )
    _draw_heatmap(
        parts,
        x0=outer + half_w + panel_gap,
        y0=bottom_y,
        width=half_w,
        height=panel_h,
        title="Current Avg Output Length",
        rows=rows,
        taus=taus,
        betas=betas,
        metric="avg_len_mean",
        reverse_colors=True,
        value_format=".0f",
        subtitle="Lower is better if quality holds. Darker cells are shorter completions on average across completed seeds.",
    )
    _draw_leaderboard(
        parts,
        x0=outer,
        y0=last_y,
        width=width - 2 * outer,
        height=leaderboard_h,
        title="Leaderboard And Status",
        rows=rows,
    )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _artifact_paths(output_path: Path) -> tuple[Path, Path]:
    suffix = output_path.suffix.lower()
    if suffix == ".png":
        return output_path.with_suffix(".svg"), output_path
    if suffix == ".svg":
        return output_path, output_path.with_suffix(".png")
    svg_path = output_path.with_suffix(".svg")
    return svg_path, svg_path.with_suffix(".png")


def _rasterize_svg(svg_path: Path, png_path: Path) -> None:
    magick = shutil.which("magick")
    convert = shutil.which("convert")
    if magick is not None:
        cmd = [magick, "-background", "white", str(svg_path), str(png_path)]
    elif convert is not None:
        cmd = [convert, "-background", "white", str(svg_path), str(png_path)]
    else:
        raise RuntimeError(
            "No SVG rasterizer found. Install ImageMagick (`convert` or `magick`) to emit PNG output."
        )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _summary_payload(
    rows: list[dict[str, Any]],
    title: str,
    *,
    svg_path: Path | None = None,
    png_path: Path | None = None,
) -> dict[str, Any]:
    counts = _status_counts(rows)
    best = _best_row(rows)
    payload: dict[str, Any] = {
        "title": title,
        "counts": counts,
        "rows": rows,
        "artifacts": {
            "svg_path": None if svg_path is None else str(svg_path),
            "png_path": None if png_path is None else str(png_path),
        },
    }
    if best is not None:
        payload["best"] = {
            "tau": float(best["tau"]),
            "beta": float(best["beta"]),
            "step": int(best["step"]) if best.get("step") is not None else None,
            "seed_count": int(best.get("seed_count", 1)),
            "seed_completed_count": int(best.get("seed_completed_count", 1)),
            "avg": float(best["avg"]),
            "pass_at_8_avg": float(best["pass_at_8_avg"]),
            "mean_at_8_avg": float(best["mean_at_8_avg"]),
            "avg_len_mean": float(best["avg_len_mean"]),
            "job_id": str(best["job_id"]),
            "run_name": str(best["run_name"]),
            "status_label": str(best["status_label"]),
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args()
    entries = report.load_manifest(args.manifest)
    rows = [report.build_row(entry, args.results_root) for entry in entries]
    scheduler_states = _load_scheduler_states([str(entry.job_id) for entry in entries])
    plot_rows = report.aggregate_rows(_augment_rows(rows, scheduler_states))
    svg_path, png_path = _artifact_paths(args.output)
    _plot_svg(plot_rows, svg_path, args.title)
    _rasterize_svg(svg_path, png_path)
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            _summary_payload(plot_rows, args.title, svg_path=svg_path, png_path=png_path),
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
