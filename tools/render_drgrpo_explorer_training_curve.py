#!/usr/bin/env python3
"""Render a clean training curve for Dr.GRPO vs Dr.GRPO-Explorer.

The figure is built from the current parity run, using checkpoint
``trainer_state.json`` histories when available and extending them with fresher
local W&B ``output.log`` points when the run has advanced beyond the last saved
checkpoint. It plots the per-step training reward metric
``rewards/seed_paper_boxed_accuracy_reward_math/mean`` against training epoch,
with raw traces and causal moving-average traces.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont


RUN_ROOT_DEFAULT = (
    "var/data/full_eval_pairs/"
    "full_eval_richsidecar_Qwen2.5-1.5B-Instruct_math_fair_mltheory_"
    "tau0p25_beta0p08_no_template_livepass8_parity_20260326_095838"
)
WANDB_RUNS_ROOT = Path("var/wandb/runs/wandb")
METRIC_KEY = "rewards/seed_paper_boxed_accuracy_reward_math/mean"
SVG_FONT_FAMILY = "'Times New Roman', 'Nimbus Roman', serif"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
COLORS = {
    "grpo": "#1f77b4",
    "explorer": "#ff7f0e",
}
LABELS = {
    "grpo": "Dr.GRPO",
    "explorer": "Dr.GRPO-Explorer",
}


@dataclass
class CurvePoint:
    step: int
    epoch: float
    value: float


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


def _hex_to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def _latest_checkpoint_dir(method_dir: Path) -> Path:
    checkpoints = []
    for child in method_dir.glob("checkpoint-*"):
        try:
            step = int(child.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if child.is_dir():
            checkpoints.append((step, child))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {method_dir}")
    checkpoints.sort()
    return checkpoints[-1][1]


def _load_curve(checkpoint_dir: Path) -> List[CurvePoint]:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    points: List[CurvePoint] = []
    for item in payload.get("log_history", []):
        step = item.get("step")
        epoch = item.get("epoch")
        value = item.get(METRIC_KEY)
        if isinstance(step, int) and isinstance(epoch, (int, float)) and isinstance(value, (int, float)):
            points.append(CurvePoint(step=step, epoch=float(epoch), value=float(value)))
    if not points:
        raise ValueError(f"No {METRIC_KEY!r} entries found in {trainer_state_path}")
    points.sort(key=lambda point: point.step)
    return points


def _parse_cli_args(args: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    idx = 0
    while idx < len(args):
        key = args[idx]
        if key.startswith("--") and idx + 1 < len(args) and not args[idx + 1].startswith("--"):
            parsed[key] = args[idx + 1]
            idx += 2
            continue
        idx += 1
    return parsed


def _find_wandb_output_log(run_root: Path, method_dirname: str) -> Path | None:
    target_output_dir = str((run_root / method_dirname).resolve())
    if not WANDB_RUNS_ROOT.exists():
        return None
    for metadata_path in WANDB_RUNS_ROOT.glob("run-*/files/wandb-metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cli_args = _parse_cli_args(payload.get("args", []))
        output_dir = cli_args.get("--output_dir")
        if not output_dir:
            continue
        output_dir_path = Path(output_dir)
        if not output_dir_path.is_absolute():
            output_dir_path = (Path.cwd() / output_dir_path).resolve()
        else:
            output_dir_path = output_dir_path.resolve()
        if str(output_dir_path) != target_output_dir:
            continue
        output_log_path = metadata_path.with_name("output.log")
        if output_log_path.exists():
            return output_log_path
    return None


def _load_curve_from_wandb(output_log_path: Path) -> List[CurvePoint]:
    points: List[CurvePoint] = []
    with output_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                payload = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            epoch = payload.get("epoch")
            value = payload.get(METRIC_KEY)
            if not isinstance(epoch, (int, float)) or not isinstance(value, (int, float)):
                continue
            points.append(CurvePoint(step=len(points) + 1, epoch=float(epoch), value=float(value)))
    return points


def _merge_curves(checkpoint_points: Sequence[CurvePoint], wandb_points: Sequence[CurvePoint]) -> List[CurvePoint]:
    if not wandb_points:
        return list(checkpoint_points)
    if len(wandb_points) <= len(checkpoint_points):
        return list(checkpoint_points)
    merged = list(checkpoint_points)
    merged.extend(wandb_points[len(checkpoint_points) :])
    return merged


def _causal_moving_average(points: Sequence[CurvePoint], window: int) -> List[CurvePoint]:
    values = [point.value for point in points]
    out: List[CurvePoint] = []
    for idx, point in enumerate(points):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(CurvePoint(step=point.step, epoch=point.epoch, value=sum(chunk) / len(chunk)))
    return out


def _nice_bounds(values: Sequence[float]) -> tuple[float, float, float]:
    vmin = min(values)
    vmax = max(values)
    spread = max(vmax - vmin, 0.02)
    pad = max(0.01, spread * 0.08)
    lower = math.floor((vmin - pad) * 100.0) / 100.0
    upper = math.ceil((vmax + pad) * 100.0) / 100.0
    tick = 0.02 if upper - lower > 0.12 else 0.01
    return lower, upper, tick


def _x_ticks(max_epoch: float) -> List[float]:
    if max_epoch <= 2.0:
        tick = 0.25
    elif max_epoch <= 4.0:
        tick = 0.5
    else:
        tick = 1.0
    ticks: List[float] = []
    current = tick
    while current <= max_epoch + 1e-9:
        ticks.append(round(current, 2))
        current += tick
    if not ticks or abs(ticks[-1] - max_epoch) > 1e-6:
        ticks.append(round(max_epoch, 2))
    return ticks


def _coord(
    epoch: float,
    value: float,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float]:
    x_span = max(x_max - x_min, 1e-8)
    x_ratio = (epoch - x_min) / x_span
    x_ratio = max(0.0, min(1.0, x_ratio))
    y_ratio = 0.0 if y_max <= y_min else (value - y_min) / (y_max - y_min)
    x = x0 + x_ratio * width
    y = y0 + height - y_ratio * height
    return x, y


def _build_svg_polyline(points: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _render_plot(
    *,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
    manifest_path: Path,
    run_root: Path,
    grpo_raw: Sequence[CurvePoint],
    explorer_raw: Sequence[CurvePoint],
    grpo_smooth: Sequence[CurvePoint],
    explorer_smooth: Sequence[CurvePoint],
    window: int,
    grpo_checkpoint: Path,
    explorer_checkpoint: Path,
    grpo_wandb_log: Path | None,
    explorer_wandb_log: Path | None,
) -> None:
    width = 1400
    height = 1400
    left = 170
    right = 95
    top = 160
    bottom = 340
    plot_w = width - left - right
    plot_h = height - top - bottom

    title_font = _load_font(50, bold=True)
    axis_font = _load_font(48, bold=True)
    tick_font = _load_font(34)
    legend_font = _load_font(46)

    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    min_epoch = min(grpo_raw[0].epoch, explorer_raw[0].epoch)
    max_epoch = max(grpo_raw[-1].epoch, explorer_raw[-1].epoch)
    all_values = [point.value for point in grpo_raw] + [point.value for point in explorer_raw]
    y_min, y_max, y_tick = _nice_bounds(all_values)
    x_ticks = _x_ticks(max_epoch)

    svg_parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    title = "Training Reward Curve: Dr.GRPO vs Dr.GRPO-Explorer"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_w) / 2, 40), title, fill="#111827", font=title_font)
    svg_parts.append(
        f'<text x="{width/2:.1f}" y="74" text-anchor="middle" font-family="{SVG_FONT_FAMILY}" '
        'font-size="52" font-weight="700" fill="#111827">'
        f"{escape(title)}</text>"
    )

    # grid and axes
    y = y_min
    while y <= y_max + 1e-9:
        _, gy = _coord(min_epoch, y, x0=left, y0=top, width=plot_w, height=plot_h, x_min=min_epoch, x_max=max_epoch, y_min=y_min, y_max=y_max)
        draw.line((left, gy, left + plot_w, gy), fill="#e5e7eb", width=2)
        label = f"{y:.2f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text((left - 28 - w, gy - h / 2), label, fill="#6b7280", font=tick_font)
        svg_parts.append(
            f'<line x1="{left:.1f}" y1="{gy:.1f}" x2="{left + plot_w:.1f}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<text x="{left - 14:.1f}" y="{gy + 10:.1f}" text-anchor="end" font-family="{SVG_FONT_FAMILY}" '
            f'font-size="36" fill="#6b7280">{escape(label)}</text>'
        )
        y += y_tick

    draw.line((left, top, left, top + plot_h), fill="#111827", width=3)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#111827", width=3)
    svg_parts.append(
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{top + plot_h:.1f}" stroke="#111827" stroke-width="3"/>'
    )
    svg_parts.append(
        f'<line x1="{left:.1f}" y1="{top + plot_h:.1f}" x2="{left + plot_w:.1f}" y2="{top + plot_h:.1f}" stroke="#111827" stroke-width="3"/>'
    )

    for xtick in x_ticks:
        gx, _ = _coord(xtick, y_min, x0=left, y0=top, width=plot_w, height=plot_h, x_min=min_epoch, x_max=max_epoch, y_min=y_min, y_max=y_max)
        draw.line((gx, top + plot_h, gx, top + plot_h + 10), fill="#111827", width=2)
        label = f"{xtick:.1f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        w = bbox[2] - bbox[0]
        draw.text((gx - w / 2, top + plot_h + 22), label, fill="#6b7280", font=tick_font)
        svg_parts.append(
            f'<line x1="{gx:.1f}" y1="{top + plot_h:.1f}" x2="{gx:.1f}" y2="{top + plot_h + 10:.1f}" stroke="#111827" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<text x="{gx:.1f}" y="{top + plot_h + 58:.1f}" text-anchor="middle" font-family="{SVG_FONT_FAMILY}" '
            f'font-size="36" fill="#6b7280">{escape(label)}</text>'
        )

    x_label = "Training epoch"
    y_label = "Seed boxed accuracy reward (mean)"
    x_bbox = draw.textbbox((0, 0), x_label, font=axis_font)
    x_w = x_bbox[2] - x_bbox[0]
    draw.text((left + plot_w / 2 - x_w / 2, height - 250), x_label, fill="#111827", font=axis_font)
    svg_parts.append(
        f'<text x="{left + plot_w/2:.1f}" y="{height - 214:.1f}" text-anchor="middle" font-family="{SVG_FONT_FAMILY}" '
        f'font-size="50" font-weight="700" fill="#111827">{escape(x_label)}</text>'
    )

    y_center = top + plot_h / 2
    y_bbox = draw.textbbox((0, 0), y_label, font=axis_font)
    y_w = y_bbox[2] - y_bbox[0]
    y_h = y_bbox[3] - y_bbox[1]
    y_img = Image.new("RGBA", (y_w + 8, y_h + 8), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_img)
    y_draw.text((4, 4), y_label, fill="#111827", font=axis_font)
    y_rotated = y_img.rotate(90, expand=True)
    img.alpha_composite(y_rotated, (24, int(y_center - y_rotated.height / 2)))
    svg_parts.append(
        f'<text x="44" y="{y_center:.1f}" text-anchor="middle" transform="rotate(-90 44 {y_center:.1f})" '
        f'font-family="{SVG_FONT_FAMILY}" font-size="50" font-weight="700" fill="#111827">{escape(y_label)}</text>'
    )

    def draw_series(name: str, raw: Sequence[CurvePoint], smooth: Sequence[CurvePoint]) -> None:
        color = COLORS[name]
        raw_rgba = _hex_to_rgba(color, 170)
        smooth_rgba = _hex_to_rgba(color, 255)
        raw_xy = [
            _coord(point.epoch, point.value, x0=left, y0=top, width=plot_w, height=plot_h, x_min=min_epoch, x_max=max_epoch, y_min=y_min, y_max=y_max)
            for point in raw
        ]
        smooth_xy = [
            _coord(point.epoch, point.value, x0=left, y0=top, width=plot_w, height=plot_h, x_min=min_epoch, x_max=max_epoch, y_min=y_min, y_max=y_max)
            for point in smooth
        ]
        if len(raw_xy) >= 2:
            raw_layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            raw_draw = ImageDraw.Draw(raw_layer)
            raw_draw.line(raw_xy, fill=raw_rgba, width=4)
            img.alpha_composite(raw_layer)
            svg_parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-opacity="0.67" stroke-width="4" points="{_build_svg_polyline(raw_xy)}"/>'
            )
        if len(smooth_xy) >= 2:
            smooth_layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            smooth_draw = ImageDraw.Draw(smooth_layer)
            smooth_draw.line(smooth_xy, fill=smooth_rgba, width=7)
            img.alpha_composite(smooth_layer)
            svg_parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-opacity="1.0" stroke-width="7" points="{_build_svg_polyline(smooth_xy)}"/>'
            )
        last_x, last_y = smooth_xy[-1]
        endpoint_rgba = _hex_to_rgba(color, 255)
        draw.ellipse((last_x - 8, last_y - 8, last_x + 8, last_y + 8), fill=(255, 255, 255, 255), outline=endpoint_rgba, width=4)
        svg_parts.append(
            f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="8" fill="white" stroke="{color}" stroke-width="4"/>'
        )

    draw_series("grpo", grpo_raw, grpo_smooth)
    draw_series("explorer", explorer_raw, explorer_smooth)

    legend_x = left + 28
    legend_y = top + 34
    legend_row_gap = 62
    legend_box_x0 = legend_x - 24
    legend_box_y0 = legend_y - 32
    legend_box_x1 = legend_x + 485
    legend_box_y1 = legend_y + legend_row_gap + 42
    draw.rounded_rectangle(
        (legend_box_x0, legend_box_y0, legend_box_x1, legend_box_y1),
        radius=16,
        fill=(255, 255, 255, 235),
        outline=(209, 213, 219, 255),
        width=2,
    )
    svg_parts.append(
        f'<rect x="{legend_box_x0:.1f}" y="{legend_box_y0:.1f}" width="{legend_box_x1 - legend_box_x0:.1f}" '
        f'height="{legend_box_y1 - legend_box_y0:.1f}" rx="16" ry="16" fill="white" fill-opacity="0.92" '
        f'stroke="#d1d5db" stroke-width="2"/>'
    )
    for idx, name in enumerate(("grpo", "explorer")):
        x = legend_x
        y = legend_y + idx * legend_row_gap
        color = COLORS[name]
        label = LABELS[name]
        draw.line((x, y, x + 70, y), fill=color, width=6)
        draw.ellipse((x + 26 - 5, y - 5, x + 26 + 5, y + 5), fill=color, outline=color)
        draw.text((x + 92, y - 23), label, fill="#111827", font=legend_font)
        svg_parts.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 70:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="6"/>'
        )
        svg_parts.append(
            f'<circle cx="{x + 26:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{x + 92:.1f}" y="{y + 12:.1f}" font-family="{SVG_FONT_FAMILY}" font-size="48" fill="#111827">{escape(label)}</text>'
        )

    img.convert("RGB").save(output_png)
    img.convert("RGB").save(output_pdf, resolution=300.0)
    svg_parts.append("</svg>")
    output_svg.write_text("\n".join(svg_parts) + "\n", encoding="utf-8")

    manifest = {
        "run_root": str(run_root),
        "metric_key": METRIC_KEY,
        "smoothing_window": window,
        "grpo_checkpoint": str(grpo_checkpoint),
        "explorer_checkpoint": str(explorer_checkpoint),
        "grpo_wandb_log": str(grpo_wandb_log) if grpo_wandb_log else None,
        "explorer_wandb_log": str(explorer_wandb_log) if explorer_wandb_log else None,
        "grpo_latest": asdict(grpo_raw[-1]),
        "explorer_latest": asdict(explorer_raw[-1]),
        "output_png": str(output_png),
        "output_pdf": str(output_pdf),
        "output_svg": str(output_svg),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=RUN_ROOT_DEFAULT)
    parser.add_argument("--output-prefix", default="var/artifacts/plots/drgrpo_vs_explorer_training_curve")
    parser.add_argument("--window", type=int, default=9)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    grpo_checkpoint = _latest_checkpoint_dir(run_root / "grpo")
    explorer_checkpoint = _latest_checkpoint_dir(run_root / "listwise")
    grpo_checkpoint_curve = _load_curve(grpo_checkpoint)
    explorer_checkpoint_curve = _load_curve(explorer_checkpoint)
    grpo_wandb_log = _find_wandb_output_log(run_root, "grpo")
    explorer_wandb_log = _find_wandb_output_log(run_root, "listwise")
    grpo_wandb_curve = _load_curve_from_wandb(grpo_wandb_log) if grpo_wandb_log else []
    explorer_wandb_curve = _load_curve_from_wandb(explorer_wandb_log) if explorer_wandb_log else []
    grpo_raw = _merge_curves(grpo_checkpoint_curve, grpo_wandb_curve)
    explorer_raw = _merge_curves(explorer_checkpoint_curve, explorer_wandb_curve)
    grpo_smooth = _causal_moving_average(grpo_raw, args.window)
    explorer_smooth = _causal_moving_average(explorer_raw, args.window)

    _render_plot(
        output_png=output_prefix.with_suffix(".png"),
        output_pdf=output_prefix.with_suffix(".pdf"),
        output_svg=output_prefix.with_suffix(".svg"),
        manifest_path=output_prefix.with_suffix(".json"),
        run_root=run_root,
        grpo_raw=grpo_raw,
        explorer_raw=explorer_raw,
        grpo_smooth=grpo_smooth,
        explorer_smooth=explorer_smooth,
        window=args.window,
        grpo_checkpoint=grpo_checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        grpo_wandb_log=grpo_wandb_log,
        explorer_wandb_log=explorer_wandb_log,
    )


if __name__ == "__main__":
    main()
