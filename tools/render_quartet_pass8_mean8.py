#!/usr/bin/env python3
"""Render official quartet pass@8 / mean@8 tables and learning curves.

This script reads official SEED-style summary JSON files from the live rich-sidecar
parity run, selects one checkpoint per method by best pooled greedy pass@1
(``avg``), and emits:

- a 3-panel learning-curve figure for pooled pass@1 / pass@8 / mean@8
- a pass@8 LaTeX table snippet
- a mean@8 LaTeX table snippet
- a JSON manifest recording the selected checkpoints and metrics
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


STEP_RE = re.compile(r"step-(\d+)")
BENCHMARK_ORDER = [
    ("aime", "AIME24"),
    ("amc", "AMC"),
    ("math", "MATH500"),
    ("minerva", "Minerva"),
    ("olympiad_bench", "OlympiadBench"),
]
SERIES_LABELS = {
    "grpo": "Dr.GRPO",
    "listwise": "Dr.GRPO-Explorer",
}
SERIES_COLORS = {
    "grpo": "#1f77b4",
    "listwise": "#2ca02c",
}
SVG_FONT_FAMILY = "'Times New Roman', 'Nimbus Roman', serif"


@dataclass
class SummaryPoint:
    step: int
    avg: float
    pass_at_8_avg: float
    mean_at_8_avg: float
    results: Dict[str, float]
    pass_at_8: Dict[str, float]
    mean_at_8: Dict[str, float]
    summary_path: str


def _parse_step(path: Path) -> int:
    match = STEP_RE.search(path.as_posix())
    if match is None:
        raise ValueError(f"Could not parse step from {path}")
    return int(match.group(1))


def _load_series(run_dir: Path) -> List[SummaryPoint]:
    points: List[SummaryPoint] = []
    for summary_path in sorted(run_dir.glob("step-*/*.summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = payload.get("results") or {}
        pass_at_8 = payload.get("pass_at_8") or {}
        mean_at_8 = payload.get("mean_at_8") or {}
        if len(results) != 5 or len(pass_at_8) != 5 or len(mean_at_8) != 5:
            continue
        if (
            payload.get("avg") is None
            or payload.get("pass_at_8_avg") is None
            or payload.get("mean_at_8_avg") is None
        ):
            continue
        points.append(
            SummaryPoint(
                step=_parse_step(summary_path),
                avg=float(payload["avg"]),
                pass_at_8_avg=float(payload["pass_at_8_avg"]),
                mean_at_8_avg=float(payload["mean_at_8_avg"]),
                results={str(k): float(v) for k, v in results.items()},
                pass_at_8={str(k): float(v) for k, v in pass_at_8.items()},
                mean_at_8={str(k): float(v) for k, v in mean_at_8.items()},
                summary_path=str(summary_path),
            )
        )
    points.sort(key=lambda item: item.step)
    return points


def _select_best(points: List[SummaryPoint]) -> SummaryPoint:
    if not points:
        raise ValueError("No summary points available for selection.")
    return max(points, key=lambda item: (item.avg, item.pass_at_8_avg, item.mean_at_8_avg, item.step))


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _write_metric_table(
    output_path: Path,
    metric_name: str,
    pooled_label: str,
    value_attr: str,
    grpo_point: SummaryPoint,
    listwise_point: SummaryPoint,
) -> None:
    metric_key = "pass_at_8" if value_attr == "pass_at_8_avg" else "mean_at_8"
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Benchmark} & \textbf{Dr.GRPO} & \textbf{Dr.GRPO-Explorer} & \textbf{Token MaxEnt} & \textbf{SEED-Dr.GRPO} \\",
        r"    \midrule",
        f"    Selected step & {grpo_point.step} & {listwise_point.step} & --- & --- \\\\",
        r"    \midrule",
    ]
    for key, label in BENCHMARK_ORDER:
        grpo_value = getattr(grpo_point, metric_key)[key]
        listwise_value = getattr(listwise_point, metric_key)[key]
        lines.append(
            f"    {label:<14} & {_pct(grpo_value)} & {_pct(listwise_value)} & --- & --- \\\\"
        )
    lines.extend(
        [
            r"    \midrule",
            f"    {pooled_label} & {_pct(getattr(grpo_point, value_attr))} & {_pct(getattr(listwise_point, value_attr))} & --- & --- \\\\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{2mm}",
            (
                r"  \caption{\textbf{Best-so-far "
                + metric_name
                + r" results on the current five-benchmark quartet suite.} "
                r"Entries are official evaluator percentages. The Dr.GRPO and "
                r"Dr.GRPO-Explorer columns use the best pooled \texttt{pass@1} "
                r"checkpoint available within the current pass@8-enabled parity run; "
                r"the Token MaxEnt and SEED-Dr.GRPO columns are left blank as "
                r"placeholders for future runs.}"
            ),
            (
                r"  \label{tab:quartet-"
                + ("pass8" if value_attr == "pass_at_8_avg" else "mean8")
                + r"-results}"
            ),
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _find_point_by_step(points: List[SummaryPoint], step: int) -> SummaryPoint:
    for point in points:
        if point.step == step:
            return point
    raise ValueError(f"Missing step {step} in series.")


def _write_matched_step_table(
    output_path: Path,
    matched_step: int,
    grpo_point: SummaryPoint,
    listwise_point: SummaryPoint,
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{7pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Metric} & \textbf{Dr.GRPO} & \textbf{Dr.GRPO-Explorer} \\",
        r"    \midrule",
        f"    Training step & {matched_step} & {matched_step} \\\\",
        f"    Pooled pass@1 & {_pct(grpo_point.avg)} & {_pct(listwise_point.avg)} \\\\",
        f"    Pooled pass@8 & {_pct(grpo_point.pass_at_8_avg)} & {_pct(listwise_point.pass_at_8_avg)} \\\\",
        f"    Pooled mean@8 & {_pct(grpo_point.mean_at_8_avg)} & {_pct(listwise_point.mean_at_8_avg)} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{2mm}",
        (
            r"  \caption{\textbf{Matched-step comparison on the current five-benchmark quartet suite.} "
            r"Both methods are evaluated at the last shared checkpoint step so the comparison uses "
            r"the same training budget on both sides.}"
        ),
        r"  \label{tab:quartet-matched-step-results}",
        r"\end{table}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    if bold:
        candidates = [
            "/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _polyline_points(
    points: List[SummaryPoint],
    metric_attr: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_max: float,
) -> List[tuple[float, float]]:
    coords: List[tuple[float, float]] = []
    x_span = max(float(x_max) - float(x_min), 1.0)
    for item in points:
        x_ratio = (float(item.step) - float(x_min)) / x_span
        x_ratio = max(0.0, min(1.0, x_ratio))
        y_value = float(getattr(item, metric_attr))
        y_ratio = 0.0 if y_max <= 0 else y_value / y_max
        coords.append((x0 + x_ratio * width, y0 + height - y_ratio * height))
    return coords


def _draw_selected_circle(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: str) -> None:
    x, y = xy
    r = 7
    draw.ellipse((x - r, y - r, x + r, y + r), fill="white", outline=color, width=3)


def _plot_curves(
    output_png: Path,
    output_svg: Path,
    grpo_points: List[SummaryPoint],
    listwise_points: List[SummaryPoint],
    grpo_best: SummaryPoint,
    listwise_best: SummaryPoint,
) -> None:
    width = 1800
    height = 880
    outer = 92
    top = 84
    bottom = 230
    panel_gap = 62
    panel_w = (width - 2 * outer - 2 * panel_gap) / 3.0
    panel_h = height - top - bottom
    title_font = _load_font(40, bold=True)
    axis_font = _load_font(34, bold=True)
    tick_font = _load_font(30)
    legend_font = _load_font(32)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    metric_specs = [
        ("avg", "Pooled pass@1"),
        ("pass_at_8_avg", "Pooled pass@8"),
        ("mean_at_8_avg", "Pooled mean@8"),
    ]
    series = {
        "grpo": grpo_points,
        "listwise": listwise_points,
    }
    best_points = {
        "grpo": grpo_best,
        "listwise": listwise_best,
    }

    svg_parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2:.1f}" y="48" text-anchor="middle" '
            f'font-family="{SVG_FONT_FAMILY}" font-size="40" font-weight="700">'
            "Official 5-task learning curves for the pass@8-enabled parity run"
            "</text>"
        ),
    ]

    all_steps = sorted(
        {
            item.step
            for item in grpo_points + listwise_points
            if item.step > 0
        }
    )
    if all_steps:
        x_min = min(all_steps)
        x_max = max(all_steps)
    else:
        x_min = min((item.step for item in grpo_points + listwise_points), default=0)
        x_max = max((item.step for item in grpo_points + listwise_points), default=0)
    visible_series = {
        family: [item for item in points if item.step >= x_min]
        for family, points in series.items()
    }
    x_ticks = [x_min]
    for xtick in range(50, int(x_max) + 1, 50):
        if xtick > x_min:
            x_ticks.append(xtick)
    if x_ticks[-1] != x_max:
        x_ticks.append(int(x_max))

    for idx, (metric_attr, title) in enumerate(metric_specs):
        x0 = outer + idx * (panel_w + panel_gap)
        y0 = top
        y_max = max(
            max((getattr(item, metric_attr) for item in grpo_points), default=0.0),
            max((getattr(item, metric_attr) for item in listwise_points), default=0.0),
            0.01,
        )
        y_max *= 1.08

        # grid and axes
        for grid_idx in range(6):
            gy = y0 + panel_h - (panel_h * grid_idx / 5.0)
            draw.line((x0, gy, x0 + panel_w, gy), fill="#e5e7eb", width=1)
            value = y_max * grid_idx / 5.0
            label = f"{100.0 * value:.1f}"
            bbox = draw.textbbox((0, 0), label, font=tick_font)
            label_w = bbox[2] - bbox[0]
            label_h = bbox[3] - bbox[1]
            draw.text((x0 - 18 - label_w, gy - label_h / 2), label, fill="#4b5563", font=tick_font)
            svg_parts.append(
                f'<line x1="{x0:.1f}" y1="{gy:.1f}" x2="{x0 + panel_w:.1f}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text x="{x0 - 14:.1f}" y="{gy + 10:.1f}" text-anchor="end" '
                f'font-family="{SVG_FONT_FAMILY}" font-size="30" fill="#4b5563">{escape(label)}</text>'
            )

        draw.line((x0, y0, x0, y0 + panel_h), fill="#111827", width=2)
        draw.line((x0, y0 + panel_h, x0 + panel_w, y0 + panel_h), fill="#111827", width=2)
        title_bbox = draw.textbbox((0, 0), title, font=axis_font)
        title_w = title_bbox[2] - title_bbox[0]
        draw.text((x0 + panel_w / 2 - title_w / 2, y0 - 54), title, fill="#111827", font=axis_font)
        xlabel = "Training step"
        xlabel_bbox = draw.textbbox((0, 0), xlabel, font=tick_font)
        xlabel_w = xlabel_bbox[2] - xlabel_bbox[0]
        draw.text((x0 + panel_w / 2 - xlabel_w / 2, y0 + panel_h + 28), xlabel, fill="#111827", font=tick_font)
        svg_parts.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0 + panel_h:.1f}" stroke="#111827" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<line x1="{x0:.1f}" y1="{y0 + panel_h:.1f}" x2="{x0 + panel_w:.1f}" y2="{y0 + panel_h:.1f}" stroke="#111827" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<text x="{x0 + panel_w/2:.1f}" y="{y0 - 16:.1f}" text-anchor="middle" '
            f'font-family="{SVG_FONT_FAMILY}" font-size="34" font-weight="700" fill="#111827">{escape(title)}</text>'
        )
        svg_parts.append(
            f'<text x="{x0 + panel_w/2:.1f}" y="{y0 + panel_h + 60:.1f}" text-anchor="middle" '
            f'font-family="{SVG_FONT_FAMILY}" font-size="30" fill="#111827">Training step</text>'
        )

        x_span = max(float(x_max) - float(x_min), 1.0)
        for xtick in x_ticks:
            tx = x0 + ((float(xtick) - float(x_min)) / x_span) * panel_w
            draw.line((tx, y0 + panel_h, tx, y0 + panel_h + 5), fill="#111827", width=1)
            tick_label = str(int(xtick))
            tick_bbox = draw.textbbox((0, 0), tick_label, font=tick_font)
            tick_w = tick_bbox[2] - tick_bbox[0]
            draw.text((tx - tick_w / 2, y0 + panel_h + 60), tick_label, fill="#4b5563", font=tick_font)
            svg_parts.append(
                f'<line x1="{tx:.1f}" y1="{y0 + panel_h:.1f}" x2="{tx:.1f}" y2="{y0 + panel_h + 5:.1f}" stroke="#111827" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text x="{tx:.1f}" y="{y0 + panel_h + 94:.1f}" text-anchor="middle" '
                f'font-family="{SVG_FONT_FAMILY}" font-size="30" fill="#4b5563">{int(xtick)}</text>'
            )

        for family in ("grpo", "listwise"):
            points = visible_series[family]
            color = SERIES_COLORS[family]
            poly = _polyline_points(points, metric_attr, x0, y0, panel_w, panel_h, x_min, x_max, y_max)
            if len(poly) >= 2:
                draw.line(poly, fill=color, width=4)
            for px, py in poly:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline=color)
            svg_poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in poly)
            svg_parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{svg_poly}"/>'
            )
            for px, py in poly:
                svg_parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.8" fill="{color}" stroke="{color}"/>'
            )
            selected = best_points[family]
            selected_xy = _polyline_points(
                [selected], metric_attr, x0, y0, panel_w, panel_h, x_min, x_max, y_max
            )[0]
            _draw_selected_circle(draw, selected_xy, color)
            svg_parts.append(
                f'<circle cx="{selected_xy[0]:.1f}" cy="{selected_xy[1]:.1f}" r="7" fill="white" stroke="{color}" stroke-width="3"/>'
            )

    legend_items = list(("grpo", "listwise"))
    legend_gap = 280
    legend_y = height - 94
    legend_x = width / 2 - legend_gap / 2
    for idx, family in enumerate(legend_items):
        y = legend_y
        x = legend_x + idx * legend_gap
        color = SERIES_COLORS[family]
        label = SERIES_LABELS[family]
        draw.line((x, y, x + 56, y), fill=color, width=4)
        draw.ellipse((x + 20 - 4, y - 4, x + 20 + 4, y + 4), fill=color, outline=color)
        draw.text((x + 72, y - 18), label, fill="#111827", font=legend_font)
        svg_parts.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 56:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="4"/>'
        )
        svg_parts.append(
            f'<circle cx="{x + 20:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{x + 72:.1f}" y="{y + 10:.1f}" font-family="{SVG_FONT_FAMILY}" font-size="32" fill="#111827">{escape(label)}</text>'
        )
    svg_parts.append("</svg>")

    img.save(output_png)
    output_svg.write_text("\n".join(svg_parts) + "\n", encoding="utf-8")


def _build_manifest(
    output_path: Path,
    grpo_dir: Path,
    listwise_dir: Path,
    grpo_points: List[SummaryPoint],
    listwise_points: List[SummaryPoint],
    grpo_best: SummaryPoint,
    listwise_best: SummaryPoint,
    output_png: Path,
    output_svg: Path,
    pass8_table_path: Path,
    mean8_table_path: Path,
    matched_step: int,
    matched_step_table_path: Path,
    grpo_matched: SummaryPoint,
    listwise_matched: SummaryPoint,
) -> None:
    payload = {
        "selection_rule": "best pooled pass@1 (avg) among official 5-task pass@8-enabled summaries",
        "grpo_dir": str(grpo_dir),
        "listwise_dir": str(listwise_dir),
        "grpo_best": asdict(grpo_best),
        "listwise_best": asdict(listwise_best),
        "grpo_series_steps": [point.step for point in grpo_points],
        "listwise_series_steps": [point.step for point in listwise_points],
        "output_png": str(output_png),
        "output_svg": str(output_svg),
        "pass8_table": str(pass8_table_path),
        "mean8_table": str(mean8_table_path),
        "matched_step": matched_step,
        "matched_step_table": str(matched_step_table_path),
        "grpo_matched": asdict(grpo_matched),
        "listwise_matched": asdict(listwise_matched),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grpo-dir",
        default=(
            "var/artifacts/seed_paper_eval/live/"
            "full_eval_richsidecar_Qwen2.5-1.5B-Instruct_math_fair_mltheory_tau0p25_beta0p08_"
            "no_template_livepass8_parity_20260326_095838-grpo"
        ),
    )
    parser.add_argument(
        "--listwise-dir",
        default=(
            "var/artifacts/seed_paper_eval/live/"
            "full_eval_richsidecar_Qwen2.5-1.5B-Instruct_math_fair_mltheory_tau0p25_beta0p08_"
            "no_template_livepass8_parity_20260326_095838-listwise"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="var/artifacts/plots/quartet_pass8_mean8_learning_curves",
    )
    parser.add_argument(
        "--paper-dir",
        default="var/artifacts/paper",
    )
    args = parser.parse_args()

    grpo_dir = Path(args.grpo_dir)
    listwise_dir = Path(args.listwise_dir)
    output_prefix = Path(args.output_prefix)
    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    grpo_points = _load_series(grpo_dir)
    listwise_points = _load_series(listwise_dir)
    if not grpo_points:
        raise SystemExit(f"No pass@8-enabled official summaries found in {grpo_dir}")
    if not listwise_points:
        raise SystemExit(f"No pass@8-enabled official summaries found in {listwise_dir}")

    grpo_best = _select_best(grpo_points)
    listwise_best = _select_best(listwise_points)

    output_png = output_prefix.with_suffix(".png")
    output_svg = output_prefix.with_suffix(".svg")
    manifest_path = output_prefix.with_suffix(".json")
    pass8_table_path = paper_dir / "quartet_pass8_table.tex"
    mean8_table_path = paper_dir / "quartet_mean8_table.tex"
    common_steps = sorted({point.step for point in grpo_points} & {point.step for point in listwise_points})
    if not common_steps:
        raise SystemExit("No shared checkpoint steps between GRPO and listwise series.")
    matched_step = max(common_steps)
    matched_step_table_path = paper_dir / f"quartet_matched_step_{matched_step}.tex"
    grpo_matched = _find_point_by_step(grpo_points, matched_step)
    listwise_matched = _find_point_by_step(listwise_points, matched_step)

    _plot_curves(output_png, output_svg, grpo_points, listwise_points, grpo_best, listwise_best)
    _write_metric_table(
        output_path=pass8_table_path,
        metric_name=r"\texttt{pass@8}",
        pooled_label="Pooled pass@8",
        value_attr="pass_at_8_avg",
        grpo_point=grpo_best,
        listwise_point=listwise_best,
    )
    _write_metric_table(
        output_path=mean8_table_path,
        metric_name=r"\texttt{mean@8}",
        pooled_label="Pooled mean@8",
        value_attr="mean_at_8_avg",
        grpo_point=grpo_best,
        listwise_point=listwise_best,
    )
    _write_matched_step_table(
        output_path=matched_step_table_path,
        matched_step=matched_step,
        grpo_point=grpo_matched,
        listwise_point=listwise_matched,
    )
    _build_manifest(
        output_path=manifest_path,
        grpo_dir=grpo_dir,
        listwise_dir=listwise_dir,
        grpo_points=grpo_points,
        listwise_points=listwise_points,
        grpo_best=grpo_best,
        listwise_best=listwise_best,
        output_png=output_png,
        output_svg=output_svg,
        pass8_table_path=pass8_table_path,
        mean8_table_path=mean8_table_path,
        matched_step=matched_step,
        matched_step_table_path=matched_step_table_path,
        grpo_matched=grpo_matched,
        listwise_matched=listwise_matched,
    )


if __name__ == "__main__":
    main()
