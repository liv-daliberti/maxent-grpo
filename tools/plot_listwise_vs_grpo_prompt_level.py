#!/usr/bin/env python3
"""Prompt-level visualization of how listwise differs from Dr.GRPO."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import plot_listwise_vs_grpo_distribution as dist


COLORS = {
    "grpo": "#2563eb",
    "listwise": "#16a34a",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a clearer prompt-level comparison between Dr.GRPO and listwise "
            "using rich completion sidecars."
        )
    )
    parser.add_argument("--grpo-run", required=True)
    parser.add_argument("--listwise-run", required=True)
    parser.add_argument("--wandb-root", default=str(dist.DEFAULT_WANDB_ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--q-temperature", type=float, default=2.0)
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


def _panel(parts: List[str], x0: float, y0: float, width: float, height: float, title: str, subtitle: str) -> Tuple[float, float, float, float]:
    parts.append(_rect(x0, y0, width, height, fill="white", stroke="#d4d4d8", **{"stroke-width": "1"}))
    parts.append(_text(x0 + 12, y0 + 22, title, **{"font-size": "15", "font-weight": "700", "fill": "#111827"}))
    parts.append(_text(x0 + 12, y0 + 39, subtitle, **{"font-size": "11", "fill": "#52525b"}))
    left = x0 + 58
    top = y0 + 54
    plot_w = width - 80
    plot_h = height - 90
    return left, top, plot_w, plot_h


def _entropy(values: Sequence[float]) -> float:
    filtered = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0]
    if not filtered:
        return float("nan")
    total = sum(filtered)
    norm = [v / total for v in filtered]
    return float(-sum(v * math.log(v) for v in norm))


def _top1_mass(record: dist.GroupRecord) -> float:
    return float(dist._sorted_mass_by_reward_rank(record)[0])


def _incorrect_mass(record: dist.GroupRecord) -> float:
    return float(sum(float(m) for r, m in zip(record.rewards, record.mass) if float(r) <= 0.0))


def _effective_rollouts(record: dist.GroupRecord) -> float:
    ent = _entropy(record.mass)
    if not math.isfinite(ent):
        return float("nan")
    return float(math.exp(ent))


def _informative_records(records: Sequence[dist.GroupRecord]) -> List[dist.GroupRecord]:
    return [record for record in records if dist._is_informative(record)]


def _ecdf(values: Sequence[float]) -> List[Tuple[float, float]]:
    filtered = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not filtered:
        return []
    n = len(filtered)
    return [(value, (idx + 1) / n) for idx, value in enumerate(filtered)]


def _agg_by_correct_count(records: Sequence[dist.GroupRecord], fn) -> List[Tuple[int, float, int]]:
    buckets: Dict[int, List[float]] = {}
    for record in records:
        correct_count = dist._count_correct(record)
        buckets.setdefault(correct_count, []).append(float(fn(record)))
    points = []
    for correct_count in sorted(buckets):
        vals = buckets[correct_count]
        points.append((correct_count, sum(vals) / len(vals), len(vals)))
    return points


def _draw_axes(
    parts: List[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_ticks: Sequence[Tuple[float, str]],
    y_min: float,
    y_max: float,
    y_label: str,
    x_label: str,
) -> None:
    parts.append(_line(x0, y0 + height, x0 + width, y0 + height, stroke="#333", **{"stroke-width": "1.2"}))
    parts.append(_line(x0, y0, x0, y0 + height, stroke="#333", **{"stroke-width": "1.2"}))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + height - frac * height
        value = y_min + frac * (y_max - y_min)
        parts.append(_line(x0, y, x0 + width, y, stroke="#ececec", **{"stroke-width": "1"}))
        parts.append(_text(x0 - 8, y + 4, f"{value:.2f}", **{"font-size": "10", "text-anchor": "end", "fill": "#666"}))
    for x_val, label in x_ticks:
        x = x0 + x_val * width
        parts.append(_line(x, y0 + height, x, y0 + height + 5, stroke="#333", **{"stroke-width": "1"}))
        parts.append(_text(x, y0 + height + 16, label, **{"font-size": "10", "text-anchor": "middle", "fill": "#666"}))
    parts.append(_text(x0 + width / 2, y0 + height + 34, x_label, **{"font-size": "11", "text-anchor": "middle"}))
    parts.append(
        _text(
            x0 - 38,
            y0 + height / 2,
            y_label,
            **{
                "font-size": "11",
                "text-anchor": "middle",
                "transform": f"rotate(-90 {x0 - 38:.1f} {y0 + height / 2:.1f})",
            },
        )
    )


def _draw_ecdf(
    parts: List[str],
    values: Sequence[float],
    *,
    color: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
) -> None:
    series = _ecdf(values)
    if not series:
        return
    x_span = max(x_max - x_min, 1e-8)
    points = []
    for x_val, y_frac in series:
        x = x0 + ((x_val - x_min) / x_span) * width
        y = y0 + height - y_frac * height
        points.append((x, y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')


def _draw_conditional_line(
    parts: List[str],
    points: Sequence[Tuple[int, float, int]],
    *,
    color: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
) -> None:
    if not points:
        return
    x_span = 7.0
    y_span = max(y_max - y_min, 1e-8)
    coords = []
    for correct_count, value, count in points:
        x = x0 + ((correct_count - 1) / x_span) * width
        y = y0 + height - ((value - y_min) / y_span) * height
        coords.append((x, y, count))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
    for x, y, count in coords:
        r = 3.0 + min(count, 12) * 0.18
        parts.append(_circle(x, y, r, fill=color, stroke="white", **{"stroke-width": "1"}))


def _plot_svg(
    *,
    grpo_records: Sequence[dist.GroupRecord],
    listwise_records: Sequence[dist.GroupRecord],
    output_path: Path,
) -> dict:
    grpo_info = _informative_records(grpo_records)
    listwise_info = _informative_records(listwise_records)

    grpo_incorrect = [_incorrect_mass(r) for r in grpo_info]
    listwise_incorrect = [_incorrect_mass(r) for r in listwise_info]
    grpo_eff = [_effective_rollouts(r) for r in grpo_info]
    listwise_eff = [_effective_rollouts(r) for r in listwise_info]
    grpo_top1_by_k = _agg_by_correct_count(grpo_info, _top1_mass)
    listwise_top1_by_k = _agg_by_correct_count(listwise_info, _top1_mass)
    grpo_bad_by_k = _agg_by_correct_count(grpo_info, _incorrect_mass)
    listwise_bad_by_k = _agg_by_correct_count(listwise_info, _incorrect_mass)

    width = 1200
    height = 860
    outer = 28
    panel_gap = 20
    panel_w = (width - 2 * outer - panel_gap) / 2
    panel_h = (height - 118 - panel_gap - outer) / 2

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(outer, 30, "Dr.GRPO vs Listwise: Prompt-Level Update Behavior", **{"font-size": "22", "font-weight": "700"}),
        _text(
            outer,
            52,
            "Each point is built from actual 8-rollout prompt groups in the current rich-sidecar run. These panels focus on prompt-level mass allocation, not benchmark eval.",
            **{"font-size": "12", "fill": "#444"},
        ),
    ]

    # legend
    parts.append(_line(outer, 76, outer + 24, 76, stroke=COLORS["grpo"], **{"stroke-width": "3"}))
    parts.append(_text(outer + 32, 80, "Dr.GRPO", **{"font-size": "12"}))
    parts.append(_line(outer + 120, 76, outer + 144, 76, stroke=COLORS["listwise"], **{"stroke-width": "3"}))
    parts.append(_text(outer + 152, 80, "Listwise", **{"font-size": "12"}))

    # Panel 1: incorrect mass ECDF
    left, top, plot_w, plot_h = _panel(
        parts,
        outer,
        96,
        panel_w,
        panel_h,
        "Mass Left On Incorrect Rollouts",
        "ECDF over informative prompt groups. Further left is more selective.",
    )
    _draw_axes(
        parts,
        x0=left,
        y0=top,
        width=plot_w,
        height=plot_h,
        x_ticks=[(0.0, "0.0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")],
        y_min=0.0,
        y_max=1.0,
        y_label="fraction of prompt groups",
        x_label="mass on incorrect rollouts",
    )
    _draw_ecdf(parts, grpo_incorrect, color=COLORS["grpo"], x0=left, y0=top, width=plot_w, height=plot_h, x_min=0.0, x_max=1.0)
    _draw_ecdf(parts, listwise_incorrect, color=COLORS["listwise"], x0=left, y0=top, width=plot_w, height=plot_h, x_min=0.0, x_max=1.0)

    # Panel 2: effective rollouts ECDF
    left, top, plot_w, plot_h = _panel(
        parts,
        outer + panel_w + panel_gap,
        96,
        panel_w,
        panel_h,
        "Effective Number Of Active Rollouts",
        "ECDF of exp(entropy). 1 = single-winner update, 8 = uniform over all rollouts.",
    )
    _draw_axes(
        parts,
        x0=left,
        y0=top,
        width=plot_w,
        height=plot_h,
        x_ticks=[(0.0, "1"), (0.142857, "2"), (0.428571, "4"), (1.0, "8")],
        y_min=0.0,
        y_max=1.0,
        y_label="fraction of prompt groups",
        x_label="effective active rollouts",
    )
    # reference at 8
    parts.append(_line(left + plot_w, top, left + plot_w, top + plot_h, stroke="#a3a3a3", **{"stroke-width": "1", "stroke-dasharray": "4 4"}))
    _draw_ecdf(parts, grpo_eff, color=COLORS["grpo"], x0=left, y0=top, width=plot_w, height=plot_h, x_min=1.0, x_max=8.0)
    _draw_ecdf(parts, listwise_eff, color=COLORS["listwise"], x0=left, y0=top, width=plot_w, height=plot_h, x_min=1.0, x_max=8.0)

    # Panel 3: top1 mass vs correct count
    left, top, plot_w, plot_h = _panel(
        parts,
        outer,
        96 + panel_h + panel_gap,
        panel_w,
        panel_h,
        "Top-Ranked Rollout Mass vs Group Difficulty",
        "Conditioned on how many of the 8 rollouts were correct. Lower means mass is spread out more.",
    )
    _draw_axes(
        parts,
        x0=left,
        y0=top,
        width=plot_w,
        height=plot_h,
        x_ticks=[((k - 1) / 7.0, str(k)) for k in range(1, 9)],
        y_min=0.0,
        y_max=1.0,
        y_label="average mass on top-ranked rollout",
        x_label="# correct rollouts in prompt group",
    )
    _draw_conditional_line(parts, grpo_top1_by_k, color=COLORS["grpo"], x0=left, y0=top, width=plot_w, height=plot_h, y_min=0.0, y_max=1.0)
    _draw_conditional_line(parts, listwise_top1_by_k, color=COLORS["listwise"], x0=left, y0=top, width=plot_w, height=plot_h, y_min=0.0, y_max=1.0)

    # Panel 4: incorrect mass vs correct count
    left, top, plot_w, plot_h = _panel(
        parts,
        outer + panel_w + panel_gap,
        96 + panel_h + panel_gap,
        panel_w,
        panel_h,
        "Incorrect Mass vs Group Difficulty",
        "Conditioned on how many of the 8 rollouts were correct. This directly shows how much update mass remains on wrong answers.",
    )
    _draw_axes(
        parts,
        x0=left,
        y0=top,
        width=plot_w,
        height=plot_h,
        x_ticks=[((k - 1) / 7.0, str(k)) for k in range(1, 9)],
        y_min=0.0,
        y_max=1.0,
        y_label="average mass on incorrect rollouts",
        x_label="# correct rollouts in prompt group",
    )
    _draw_conditional_line(parts, grpo_bad_by_k, color=COLORS["grpo"], x0=left, y0=top, width=plot_w, height=plot_h, y_min=0.0, y_max=1.0)
    _draw_conditional_line(parts, listwise_bad_by_k, color=COLORS["listwise"], x0=left, y0=top, width=plot_w, height=plot_h, y_min=0.0, y_max=1.0)

    parts.append(
        _text(
            outer,
            height - 18,
            f"Informative groups used: GRPO {len(grpo_info)}/{len(grpo_records)}, Listwise {len(listwise_info)}/{len(listwise_records)}.",
            **{"font-size": "11", "fill": "#555"},
        )
    )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")

    return {
        "grpo": {
            "total_groups": len(grpo_records),
            "informative_groups": len(grpo_info),
            "incorrect_mass_ecdf_n": len(grpo_incorrect),
            "effective_rollouts_mean": sum(grpo_eff) / len(grpo_eff) if grpo_eff else float("nan"),
            "top1_by_correct_count": grpo_top1_by_k,
            "incorrect_by_correct_count": grpo_bad_by_k,
        },
        "listwise": {
            "total_groups": len(listwise_records),
            "informative_groups": len(listwise_info),
            "incorrect_mass_ecdf_n": len(listwise_incorrect),
            "effective_rollouts_mean": sum(listwise_eff) / len(listwise_eff) if listwise_eff else float("nan"),
            "top1_by_correct_count": listwise_top1_by_k,
            "incorrect_by_correct_count": listwise_bad_by_k,
        },
    }


def main() -> int:
    args = _parse_args()
    wandb_root = Path(args.wandb_root)
    grpo_dir = dist._resolve_run(str(args.grpo_run), wandb_root)
    listwise_dir = dist._resolve_run(str(args.listwise_run), wandb_root)
    grpo_records, _ = dist._load_records(
        grpo_dir,
        label="grpo",
        q_temperature=float(args.q_temperature),
        include_neutral_groups=True,
        max_groups=int(args.max_groups),
    )
    listwise_records, _ = dist._load_records(
        listwise_dir,
        label="listwise",
        q_temperature=float(args.q_temperature),
        include_neutral_groups=True,
        max_groups=int(args.max_groups),
    )
    output_path = Path(args.output)
    summary = _plot_svg(
        grpo_records=grpo_records,
        listwise_records=listwise_records,
        output_path=output_path,
    )
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
