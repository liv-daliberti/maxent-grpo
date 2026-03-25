#!/usr/bin/env python3
"""Render a live interim dashboard for the current full GRPO vs listwise run."""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UNIFORM_GROUP_ENTROPY = math.log(8.0)


@dataclass
class MetricPoint:
    step: int
    reward: float | None
    mean_length: float | None
    clipped_ratio: float | None
    informative_share: float | None
    weight_entropy: float | None
    active_group_frac: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an interim training dashboard for the live GRPO vs listwise full run."
    )
    parser.add_argument("--grpo-log", type=Path, required=True)
    parser.add_argument("--listwise-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="GRPO vs Listwise: Full Run Interim Dashboard",
    )
    parser.add_argument("--grpo-label", default="GRPO")
    parser.add_argument("--listwise-label", default="Listwise")
    return parser.parse_args()


def _text(x: float, y: float, text: str, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {extra}>{html.escape(text)}</text>'


def _line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {extra}/>'


def _rect(x: float, y: float, width: float, height: float, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" {extra}/>'


def _circle(cx: float, cy: float, r: float, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" {extra}/>'


def _parse_metric_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith("{") or "'train/loss/total'" not in stripped:
        return None
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_series(path: Path) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    if not path.exists():
        return points
    for raw_line in path.read_text().splitlines():
        payload = _parse_metric_line(raw_line)
        if payload is None:
            continue
        reward_zero_std = _float_or_none(payload.get("frac_reward_zero_std"))
        informative_share = None if reward_zero_std is None else max(0.0, min(1.0, 1.0 - reward_zero_std))
        points.append(
            MetricPoint(
                step=len(points) + 1,
                reward=_float_or_none(payload.get("reward")),
                mean_length=_float_or_none(payload.get("completions/mean_length")),
                clipped_ratio=_float_or_none(payload.get("completions/clipped_ratio")),
                informative_share=informative_share,
                weight_entropy=_float_or_none(payload.get("weight_entropy")),
                active_group_frac=_float_or_none(payload.get("maxent/listwise_active_group_frac")),
            )
        )
    return points


def _series_values(points: list[MetricPoint], key: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for point in points:
        value = getattr(point, key)
        if value is None:
            continue
        values.append((point.step, float(value)))
    return values


def _metric_range(series_groups: list[list[tuple[int, float]]], *, pad_frac: float = 0.08) -> tuple[float, float]:
    values = [value for group in series_groups for _, value in group]
    if not values:
        return (0.0, 1.0)
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        if abs(lo) < 1e-9:
            return (-1.0, 1.0)
        span = abs(lo) * 0.1
        return (lo - span, hi + span)
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def _step_range(series_groups: list[list[tuple[int, float]]]) -> tuple[int, int]:
    steps = [step for group in series_groups for step, _ in group]
    if not steps:
        return (0, 1)
    return (min(steps), max(steps))


def _polyline_points(
    series: list[tuple[int, float]],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_min: int,
    x_max: int,
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    if not series:
        return []
    x_span = max(1, x_max - x_min)
    y_span = y_max - y_min if not math.isclose(y_min, y_max) else 1.0
    coords: list[tuple[float, float]] = []
    for step, value in series:
        x = x0 + ((step - x_min) / x_span) * width
        y = y0 + height - ((value - y_min) / y_span) * height
        coords.append((x, y))
    return coords


def _format_metric(value: float | None, *, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _draw_axes(
    parts: list[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_min: int,
    x_max: int,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
    y_tick_count: int = 5,
) -> None:
    parts.append(_line(x0, y0 + height, x0 + width, y0 + height, stroke="#71717a", **{"stroke-width": "1"}))
    parts.append(_line(x0, y0, x0, y0 + height, stroke="#71717a", **{"stroke-width": "1"}))
    for i in range(y_tick_count):
        frac = i / max(1, y_tick_count - 1)
        y = y0 + height - frac * height
        value = y_min + frac * (y_max - y_min)
        parts.append(_line(x0, y, x0 + width, y, stroke="#e4e4e7", **{"stroke-width": "1"}))
        parts.append(_text(x0 - 8, y + 4, f"{value:.2f}", **{"font-size": "11", "text-anchor": "end", "fill": "#52525b"}))
    for step in range(x_min, x_max + 1, max(1, math.ceil((x_max - x_min + 1) / 6))):
        frac = 0.0 if x_max == x_min else (step - x_min) / (x_max - x_min)
        x = x0 + frac * width
        parts.append(_line(x, y0 + height, x, y0 + height + 6, stroke="#71717a", **{"stroke-width": "1"}))
        parts.append(_text(x, y0 + height + 20, str(step), **{"font-size": "11", "text-anchor": "middle", "fill": "#52525b"}))
    parts.append(_text(x0 + width / 2, y0 + height + 42, x_label, **{"font-size": "12", "text-anchor": "middle", "fill": "#3f3f46"}))
    parts.append(
        f'<text x="{x0 - 48:.1f}" y="{y0 + height / 2:.1f}" font-size="12" text-anchor="middle" fill="#3f3f46" transform="rotate(-90 {x0 - 48:.1f} {y0 + height / 2:.1f})">{html.escape(y_label)}</text>'
    )


def _draw_series(
    parts: list[str],
    series: list[tuple[int, float]],
    *,
    color: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_min: int,
    x_max: int,
    y_min: float,
    y_max: float,
) -> None:
    coords = _polyline_points(
        series,
        x0=x0,
        y0=y0,
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    if not coords:
        return
    if len(coords) > 1:
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    for x, y in coords:
        parts.append(_circle(x, y, 3.6, fill=color, stroke="white", **{"stroke-width": "1"}))


def _panel_frame(parts: list[str], x0: float, y0: float, width: float, height: float, title: str, subtitle: str | None = None) -> tuple[float, float, float, float]:
    parts.append(_rect(x0, y0, width, height, fill="white", stroke="#d4d4d8", **{"stroke-width": "1"}))
    parts.append(_text(x0 + 14, y0 + 22, title, **{"font-size": "15", "font-weight": "700", "fill": "#111827"}))
    if subtitle:
        parts.append(_text(x0 + 14, y0 + 40, subtitle, **{"font-size": "11", "fill": "#52525b"}))
    top = y0 + 52
    left = x0 + 66
    plot_w = width - 88
    plot_h = height - 98
    return left, top, plot_w, plot_h


def _draw_legend(parts: list[str], x: float, y: float, grpo_label: str, listwise_label: str) -> None:
    parts.append(_line(x, y, x + 20, y, stroke="#2563eb", **{"stroke-width": "3"}))
    parts.append(_text(x + 28, y + 4, grpo_label, **{"font-size": "12", "fill": "#1f2937"}))
    parts.append(_line(x + 120, y, x + 140, y, stroke="#dc2626", **{"stroke-width": "3"}))
    parts.append(_text(x + 148, y + 4, listwise_label, **{"font-size": "12", "fill": "#1f2937"}))


def _latest(points: list[MetricPoint]) -> MetricPoint | None:
    if not points:
        return None
    return points[-1]


def _summary_payload(
    title: str,
    grpo_points: list[MetricPoint],
    listwise_points: list[MetricPoint],
    args: argparse.Namespace,
) -> dict[str, Any]:
    grpo_latest = _latest(grpo_points)
    listwise_latest = _latest(listwise_points)
    return {
        "title": title,
        "grpo_label": args.grpo_label,
        "listwise_label": args.listwise_label,
        "uniform_group_entropy": UNIFORM_GROUP_ENTROPY,
        "grpo": {
            "points": len(grpo_points),
            "latest_step": None if grpo_latest is None else grpo_latest.step,
            "latest_reward": None if grpo_latest is None else grpo_latest.reward,
            "latest_mean_length": None if grpo_latest is None else grpo_latest.mean_length,
            "latest_clipped_ratio": None if grpo_latest is None else grpo_latest.clipped_ratio,
            "latest_informative_share": None if grpo_latest is None else grpo_latest.informative_share,
            "log_path": str(args.grpo_log),
        },
        "listwise": {
            "points": len(listwise_points),
            "latest_step": None if listwise_latest is None else listwise_latest.step,
            "latest_reward": None if listwise_latest is None else listwise_latest.reward,
            "latest_mean_length": None if listwise_latest is None else listwise_latest.mean_length,
            "latest_clipped_ratio": None if listwise_latest is None else listwise_latest.clipped_ratio,
            "latest_informative_share": None if listwise_latest is None else listwise_latest.informative_share,
            "latest_weight_entropy": None if listwise_latest is None else listwise_latest.weight_entropy,
            "latest_active_group_frac": None if listwise_latest is None else listwise_latest.active_group_frac,
            "latest_concentration": None
            if listwise_latest is None or listwise_latest.weight_entropy is None
            else 1.0 - (listwise_latest.weight_entropy / UNIFORM_GROUP_ENTROPY),
            "log_path": str(args.listwise_log),
        },
        "note": "Live training-only dashboard. Held-out SEED eval artifacts are still pending.",
    }


def _plot_svg(
    title: str,
    grpo_points: list[MetricPoint],
    listwise_points: list[MetricPoint],
    output: Path,
    args: argparse.Namespace,
) -> None:
    width = 1500
    height = 1020
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    parts.append(_rect(0, 0, width, height, fill="#fafafa"))
    parts.append(_text(40, 50, title, **{"font-size": "28", "font-weight": "700", "fill": "#111827"}))
    parts.append(
        _text(
            40,
            78,
            "Current live training traces from the full GRPO vs listwise pair. These are interim rollout-side metrics, not held-out benchmark evals.",
            **{"font-size": "14", "fill": "#52525b"},
        )
    )
    _draw_legend(parts, 40, 102, args.grpo_label, args.listwise_label)

    grpo_latest = _latest(grpo_points)
    listwise_latest = _latest(listwise_points)

    cards = [
        ("GRPO latest reward", _format_metric(None if grpo_latest is None else grpo_latest.reward)),
        ("Listwise latest reward", _format_metric(None if listwise_latest is None else listwise_latest.reward)),
        ("GRPO mean length", _format_metric(None if grpo_latest is None else grpo_latest.mean_length, decimals=1)),
        ("Listwise mean length", _format_metric(None if listwise_latest is None else listwise_latest.mean_length, decimals=1)),
        ("GRPO informative share", _format_metric(None if grpo_latest is None else grpo_latest.informative_share)),
        ("Listwise informative share", _format_metric(None if listwise_latest is None else listwise_latest.informative_share)),
    ]
    card_x = 40.0
    for label, value in cards:
        parts.append(_rect(card_x, 126, 215, 62, rx="10", ry="10", fill="white", stroke="#d4d4d8", **{"stroke-width": "1"}))
        parts.append(_text(card_x + 14, 149, label, **{"font-size": "12", "fill": "#52525b"}))
        parts.append(_text(card_x + 14, 175, value, **{"font-size": "24", "font-weight": "700", "fill": "#111827"}))
        card_x += 228

    panel_w = 690.0
    panel_h = 320.0
    left_x = 40.0
    right_x = 770.0
    top_y = 220.0
    bottom_y = 570.0

    reward_grpo = _series_values(grpo_points, "reward")
    reward_listwise = _series_values(listwise_points, "reward")
    reward_x_min, reward_x_max = _step_range([reward_grpo, reward_listwise])
    reward_y_min, reward_y_max = _metric_range([reward_grpo, reward_listwise], pad_frac=0.12)
    x0, y0, plot_w, plot_h = _panel_frame(parts, left_x, top_y, panel_w, panel_h, "Reward Over Training", "Higher is better on current rollout batches.")
    _draw_axes(parts, x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=reward_x_min, x_max=reward_x_max, y_min=reward_y_min, y_max=reward_y_max, x_label="Logged optimizer step", y_label="reward")
    _draw_series(parts, reward_grpo, color="#2563eb", x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=reward_x_min, x_max=reward_x_max, y_min=reward_y_min, y_max=reward_y_max)
    _draw_series(parts, reward_listwise, color="#dc2626", x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=reward_x_min, x_max=reward_x_max, y_min=reward_y_min, y_max=reward_y_max)

    length_grpo = _series_values(grpo_points, "mean_length")
    length_listwise = _series_values(listwise_points, "mean_length")
    len_x_min, len_x_max = _step_range([length_grpo, length_listwise])
    len_y_min, len_y_max = _metric_range([length_grpo, length_listwise], pad_frac=0.15)
    x0, y0, plot_w, plot_h = _panel_frame(parts, right_x, top_y, panel_w, panel_h, "Mean Completion Length", "Current rollout lengths before held-out eval.")
    _draw_axes(parts, x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=len_x_min, x_max=len_x_max, y_min=len_y_min, y_max=len_y_max, x_label="Logged optimizer step", y_label="tokens")
    _draw_series(parts, length_grpo, color="#2563eb", x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=len_x_min, x_max=len_x_max, y_min=len_y_min, y_max=len_y_max)
    _draw_series(parts, length_listwise, color="#dc2626", x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=len_x_min, x_max=len_x_max, y_min=len_y_min, y_max=len_y_max)

    info_grpo = _series_values(grpo_points, "informative_share")
    info_listwise = _series_values(listwise_points, "informative_share")
    info_x_min, info_x_max = _step_range([info_grpo, info_listwise])
    x0, y0, plot_w, plot_h = _panel_frame(
        parts,
        left_x,
        bottom_y,
        panel_w,
        panel_h,
        "Informative Prompt-Group Share",
        "1 - frac_reward_zero_std. Higher means more prompt groups have actual within-group signal.",
    )
    _draw_axes(parts, x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=info_x_min, x_max=info_x_max, y_min=0.0, y_max=1.0, x_label="Logged optimizer step", y_label="share")
    ref_y = y0 + plot_h - 0.5 * plot_h
    parts.append(_line(x0, ref_y, x0 + plot_w, ref_y, stroke="#a1a1aa", **{"stroke-width": "1", "stroke-dasharray": "4 4"}))
    parts.append(_text(x0 + plot_w - 6, ref_y - 6, "0.50", **{"font-size": "11", "text-anchor": "end", "fill": "#71717a"}))
    _draw_series(parts, info_grpo, color="#2563eb", x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=info_x_min, x_max=info_x_max, y_min=0.0, y_max=1.0)
    _draw_series(parts, info_listwise, color="#dc2626", x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=info_x_min, x_max=info_x_max, y_min=0.0, y_max=1.0)

    listwise_concentration = [
        (point.step, max(0.0, min(1.0, 1.0 - (point.weight_entropy / UNIFORM_GROUP_ENTROPY))))
        for point in listwise_points
        if point.weight_entropy is not None
    ]
    listwise_active = _series_values(listwise_points, "active_group_frac")
    conc_x_min, conc_x_max = _step_range([listwise_concentration, listwise_active])
    x0, y0, plot_w, plot_h = _panel_frame(
        parts,
        right_x,
        bottom_y,
        panel_w,
        panel_h,
        "Listwise Weighting Internals",
        "Concentration = 1 - weight_entropy / log(8). Higher means listwise is moving away from uniform mass.",
    )
    _draw_axes(parts, x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=conc_x_min, x_max=conc_x_max, y_min=0.0, y_max=1.0, x_label="Logged optimizer step", y_label="share")
    _draw_series(parts, listwise_active, color="#dc2626", x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=conc_x_min, x_max=conc_x_max, y_min=0.0, y_max=1.0)
    _draw_series(parts, listwise_concentration, color="#7c3aed", x0=x0, y0=y0, width=plot_w, height=plot_h, x_min=conc_x_min, x_max=conc_x_max, y_min=0.0, y_max=1.0)
    parts.append(_line(x0 + 6, y0 + 16, x0 + 26, y0 + 16, stroke="#dc2626", **{"stroke-width": "3"}))
    parts.append(_text(x0 + 34, y0 + 20, "active-group share", **{"font-size": "12", "fill": "#1f2937"}))
    parts.append(_line(x0 + 190, y0 + 16, x0 + 210, y0 + 16, stroke="#7c3aed", **{"stroke-width": "3"}))
    parts.append(_text(x0 + 218, y0 + 20, "mass concentration", **{"font-size": "12", "fill": "#1f2937"}))

    footer_y = 940.0
    parts.append(_text(40, footer_y, f"GRPO log: {args.grpo_log}", **{"font-size": "11", "fill": "#52525b"}))
    parts.append(_text(40, footer_y + 18, f"Listwise log: {args.listwise_log}", **{"font-size": "11", "fill": "#52525b"}))
    parts.append(
        _text(
            40,
            footer_y + 42,
            "Interpretation: this dashboard compares live rollout-side behavior only. The full SEED-paper held-out eval chain is still waiting on the train jobs to finish.",
            **{"font-size": "12", "fill": "#3f3f46"},
        )
    )

    parts.append("</svg>")
    output.write_text("\n".join(parts))


def _write_png_from_svg(svg_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    convert = shutil.which("convert")
    if convert is None:
        return
    subprocess.run([convert, str(svg_path), str(png_path)], check=True)


def main() -> None:
    args = _parse_args()
    grpo_points = _load_series(args.grpo_log)
    listwise_points = _load_series(args.listwise_log)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = args.output.with_suffix(".svg")
    png_path = args.output.with_suffix(".png")
    _plot_svg(args.title, grpo_points, listwise_points, svg_path, args)
    _write_png_from_svg(svg_path, png_path)

    summary = _summary_payload(args.title, grpo_points, listwise_points, args)
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
