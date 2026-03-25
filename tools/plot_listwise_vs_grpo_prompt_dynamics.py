#!/usr/bin/env python3
"""Plot prompt-level mass dynamics over training for Dr.GRPO vs listwise."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
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


@dataclass
class StepDynamics:
    step: int
    total_groups: int
    informative_groups: int
    informative_share: float
    mean_incorrect_mass: float
    mean_top1_mass: float
    mean_effective_rollouts: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a cleaner time-series view of prompt-level mass dynamics."
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


def _entropy(values: Sequence[float]) -> float:
    filtered = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0]
    if not filtered:
        return float("nan")
    total = sum(filtered)
    normalized = [v / total for v in filtered]
    return float(-sum(v * math.log(v) for v in normalized))


def _incorrect_mass(record: dist.GroupRecord) -> float:
    return float(sum(float(m) for r, m in zip(record.rewards, record.mass) if float(r) <= 0.0))


def _top1_mass(record: dist.GroupRecord) -> float:
    masses = dist._sorted_mass_by_reward_rank(record)
    return float(masses[0]) if masses else float("nan")


def _effective_rollouts(record: dist.GroupRecord) -> float:
    ent = _entropy(record.mass)
    if not math.isfinite(ent):
        return float("nan")
    return float(math.exp(ent))


def _aggregate_dynamics(records: Sequence[dist.GroupRecord]) -> List[StepDynamics]:
    by_step: Dict[int, List[dist.GroupRecord]] = {}
    for record in records:
        by_step.setdefault(int(record.step), []).append(record)

    results: List[StepDynamics] = []
    for step in sorted(by_step):
        step_records = by_step[step]
        informative = [record for record in step_records if dist._is_informative(record)]
        informative_share = (len(informative) / len(step_records)) if step_records else float("nan")

        def _mean(values: Sequence[float]) -> float:
            vals = [float(v) for v in values if math.isfinite(float(v))]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        results.append(
            StepDynamics(
                step=step,
                total_groups=len(step_records),
                informative_groups=len(informative),
                informative_share=informative_share,
                mean_incorrect_mass=_mean([_incorrect_mass(record) for record in informative]),
                mean_top1_mass=_mean([_top1_mass(record) for record in informative]),
                mean_effective_rollouts=_mean([_effective_rollouts(record) for record in informative]),
            )
        )
    return results


def _panel(parts: List[str], x0: float, y0: float, width: float, height: float, title: str, subtitle: str) -> Tuple[float, float, float, float]:
    parts.append(_rect(x0, y0, width, height, fill="white", stroke="#d4d4d8", **{"stroke-width": "1"}))
    parts.append(_text(x0 + 12, y0 + 22, title, **{"font-size": "15", "font-weight": "700", "fill": "#111827"}))
    parts.append(_text(x0 + 12, y0 + 39, subtitle, **{"font-size": "11", "fill": "#52525b"}))
    left = x0 + 58
    top = y0 + 54
    plot_w = width - 80
    plot_h = height - 88
    return left, top, plot_w, plot_h


def _metric_bounds(grpo: Sequence[Tuple[int, float]], listwise: Sequence[Tuple[int, float]], *, floor: float | None = None, ceil: float | None = None) -> Tuple[float, float]:
    values = [v for _, v in list(grpo) + list(listwise) if math.isfinite(v)]
    if not values:
        lo, hi = 0.0, 1.0
    else:
        lo = min(values)
        hi = max(values)
        if math.isclose(lo, hi):
            span = abs(lo) * 0.1 if abs(lo) > 1e-9 else 1.0
            lo -= span
            hi += span
        else:
            pad = (hi - lo) * 0.08
            lo -= pad
            hi += pad
    if floor is not None:
        lo = min(lo, floor)
        lo = max(lo, floor) if ceil is not None and floor == ceil else max(lo, floor)
        lo = floor
    if ceil is not None:
        hi = ceil
    return lo, hi


def _draw_axes(
    parts: List[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_steps: Sequence[int],
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
    x_min = min(x_steps) if x_steps else 1
    x_max = max(x_steps) if x_steps else 1
    span = max(x_max - x_min, 1)
    for step in x_steps:
        x = x0 + ((step - x_min) / span) * width
        parts.append(_line(x, y0 + height, x, y0 + height + 5, stroke="#333", **{"stroke-width": "1"}))
        parts.append(_text(x, y0 + height + 16, str(step), **{"font-size": "10", "text-anchor": "middle", "fill": "#666"}))
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


def _draw_series(
    parts: List[str],
    *,
    points: Sequence[Tuple[int, float]],
    color: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_steps: Sequence[int],
    y_min: float,
    y_max: float,
) -> None:
    if not points:
        return
    x_min = min(x_steps) if x_steps else 1
    x_max = max(x_steps) if x_steps else 1
    span = max(x_max - x_min, 1)
    y_span = max(y_max - y_min, 1e-8)
    coords = []
    for step, value in points:
        if not math.isfinite(value):
            continue
        x = x0 + ((step - x_min) / span) * width
        y = y0 + height - ((value - y_min) / y_span) * height
        coords.append((x, y))
    if not coords:
        return
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
    for x, y in coords:
        parts.append(_circle(x, y, 3.5, fill=color, stroke="white", **{"stroke-width": "1"}))


def _plot_svg(
    *,
    grpo_steps: Sequence[StepDynamics],
    listwise_steps: Sequence[StepDynamics],
    output_path: Path,
) -> dict:
    width = 1200
    height = 860
    outer = 28
    panel_gap = 20
    panel_w = (width - 2 * outer - panel_gap) / 2
    panel_h = (height - 118 - panel_gap - outer) / 2

    step_union = sorted({item.step for item in grpo_steps} | {item.step for item in listwise_steps})
    grpo_info = [(item.step, item.informative_share) for item in grpo_steps]
    listwise_info = [(item.step, item.informative_share) for item in listwise_steps]
    grpo_bad = [(item.step, item.mean_incorrect_mass) for item in grpo_steps]
    listwise_bad = [(item.step, item.mean_incorrect_mass) for item in listwise_steps]
    grpo_top1 = [(item.step, item.mean_top1_mass) for item in grpo_steps]
    listwise_top1 = [(item.step, item.mean_top1_mass) for item in listwise_steps]
    grpo_eff = [(item.step, item.mean_effective_rollouts) for item in grpo_steps]
    listwise_eff = [(item.step, item.mean_effective_rollouts) for item in listwise_steps]

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(outer, 30, "Dr.GRPO vs Listwise: How Prompt-Level Behavior Changes Over Training", **{"font-size": "22", "font-weight": "700"}),
        _text(
            outer,
            52,
            "Each point aggregates actual 8-rollout prompt groups from one training step in the current rich-sidecar run.",
            **{"font-size": "12", "fill": "#444"},
        ),
        _text(
            outer,
            69,
            "This figure is meant to answer how the update behavior evolves, not just how one pooled snapshot differs.",
            **{"font-size": "12", "fill": "#444"},
        ),
    ]
    parts.append(_line(outer, 88, outer + 24, 88, stroke=COLORS["grpo"], **{"stroke-width": "3"}))
    parts.append(_text(outer + 32, 92, "Dr.GRPO", **{"font-size": "12"}))
    parts.append(_line(outer + 120, 88, outer + 144, 88, stroke=COLORS["listwise"], **{"stroke-width": "3"}))
    parts.append(_text(outer + 152, 92, "Listwise", **{"font-size": "12"}))

    panels = [
        (
            outer,
            108,
            "Informative-Group Share",
            "Fraction of prompt groups at that step with mixed rewards. Higher means more groups have real rank signal.",
            grpo_info,
            listwise_info,
            0.0,
            1.0,
            "share",
        ),
        (
            outer + panel_w + panel_gap,
            108,
            "Incorrect-Mass Share",
            "Average mass left on wrong rollouts among informative groups. Lower is more selective.",
            grpo_bad,
            listwise_bad,
            0.0,
            1.0,
            "mass on incorrect rollouts",
        ),
        (
            outer,
            108 + panel_h + panel_gap,
            "Top-1 Rollout Mass",
            "Average mass on the highest-ranked rollout among informative groups. Higher means more winner-take-all behavior.",
            grpo_top1,
            listwise_top1,
            0.0,
            1.0,
            "mass on top-ranked rollout",
        ),
        (
            outer + panel_w + panel_gap,
            108 + panel_h + panel_gap,
            "Effective Active Rollouts",
            "Average exp(entropy) among informative groups. 1 means single-winner; 8 means nearly uniform.",
            grpo_eff,
            listwise_eff,
            1.0,
            8.0,
            "effective active rollouts",
        ),
    ]

    for x0, y0, title, subtitle, grpo_pts, listwise_pts, y_lo, y_hi, y_label in panels:
        left, top, plot_w, plot_h = _panel(parts, x0, y0, panel_w, panel_h, title, subtitle)
        _draw_axes(
            parts,
            x0=left,
            y0=top,
            width=plot_w,
            height=plot_h,
            x_steps=step_union,
            y_min=y_lo,
            y_max=y_hi,
            y_label=y_label,
            x_label="training step",
        )
        _draw_series(parts, points=grpo_pts, color=COLORS["grpo"], x0=left, y0=top, width=plot_w, height=plot_h, x_steps=step_union, y_min=y_lo, y_max=y_hi)
        _draw_series(parts, points=listwise_pts, color=COLORS["listwise"], x0=left, y0=top, width=plot_w, height=plot_h, x_steps=step_union, y_min=y_lo, y_max=y_hi)

    parts.append(
        _text(
            outer,
            height - 18,
            f"Steps available now: GRPO {len(grpo_steps)} steps, Listwise {len(listwise_steps)} steps. Re-run later for smoother trajectories.",
            **{"font-size": "11", "fill": "#555"},
        )
    )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")

    return {
        "grpo": [item.__dict__ for item in grpo_steps],
        "listwise": [item.__dict__ for item in listwise_steps],
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
    summary = _plot_svg(
        grpo_steps=_aggregate_dynamics(grpo_records),
        listwise_steps=_aggregate_dynamics(listwise_records),
        output_path=Path(args.output),
    )
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
