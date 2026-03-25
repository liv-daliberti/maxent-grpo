#!/usr/bin/env python3
"""Plot within-prompt rollout mass over time for Dr.GRPO versus listwise."""

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
    "grpo": "#1f77b4",
    "listwise": "#2ca02c",
    "top1": "#4c78a8",
    "other_correct": "#9ecae9",
    "incorrect": "#d9d9d9",
    "neutral": "#bdbdbd",
}


@dataclass
class StepSummary:
    step: int
    informative_groups: int
    neutral_groups: int
    mean_top1_mass: float
    mean_other_correct_mass: float
    mean_incorrect_mass: float
    mean_correct_count: float
    mean_group_size: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot how rollout update mass evolves over training steps for "
            "Dr.GRPO versus listwise."
        )
    )
    parser.add_argument("--grpo-run", required=True)
    parser.add_argument("--listwise-run", required=True)
    parser.add_argument(
        "--wandb-root",
        default=str(dist.DEFAULT_WANDB_ROOT),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--q-temperature", type=float, default=2.0)
    return parser.parse_args()


def _sorted_order(record: dist.GroupRecord) -> List[int]:
    return sorted(
        range(len(record.rewards)),
        key=lambda idx: (-float(record.rewards[idx]), idx),
    )


def _step_summary(record: dist.GroupRecord) -> Tuple[bool, float, float, float, int, int]:
    informative = dist._is_informative(record)
    group_size = len(record.rewards)
    correct_count = dist._count_correct(record)
    if not informative:
        return False, 0.0, 0.0, 0.0, correct_count, group_size

    order = _sorted_order(record)
    top1_mass = float(record.mass[order[0]]) if order else 0.0
    correct_total = sum(
        float(mass)
        for reward, mass in zip(record.rewards, record.mass)
        if float(reward) > 0.0
    )
    incorrect_mass = sum(
        float(mass)
        for reward, mass in zip(record.rewards, record.mass)
        if float(reward) <= 0.0
    )
    other_correct_mass = max(correct_total - top1_mass, 0.0)
    return True, top1_mass, other_correct_mass, incorrect_mass, correct_count, group_size


def _aggregate_steps(records: Sequence[dist.GroupRecord]) -> List[StepSummary]:
    by_step: Dict[int, List[dist.GroupRecord]] = {}
    for record in records:
        by_step.setdefault(int(record.step), []).append(record)

    summaries: List[StepSummary] = []
    for step in sorted(by_step):
        step_records = by_step[step]
        informative_groups = 0
        neutral_groups = 0
        top1_vals: List[float] = []
        other_vals: List[float] = []
        incorrect_vals: List[float] = []
        correct_counts: List[int] = []
        group_sizes: List[int] = []
        for record in step_records:
            informative, top1, other_correct, incorrect, correct_count, group_size = _step_summary(record)
            if informative:
                informative_groups += 1
                top1_vals.append(top1)
                other_vals.append(other_correct)
                incorrect_vals.append(incorrect)
            else:
                neutral_groups += 1
            correct_counts.append(correct_count)
            group_sizes.append(group_size)
        summaries.append(
            StepSummary(
                step=step,
                informative_groups=informative_groups,
                neutral_groups=neutral_groups,
                mean_top1_mass=sum(top1_vals) / len(top1_vals) if top1_vals else 0.0,
                mean_other_correct_mass=sum(other_vals) / len(other_vals) if other_vals else 0.0,
                mean_incorrect_mass=sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else 0.0,
                mean_correct_count=sum(correct_counts) / len(correct_counts) if correct_counts else 0.0,
                mean_group_size=sum(group_sizes) / len(group_sizes) if group_sizes else 0.0,
            )
        )
    return summaries


def _text(x: float, y: float, text: str, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {extra}>{html.escape(text)}</text>'


def _line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {extra}/>'


def _panel(parts: List[str], x0: float, y0: float, width: float, height: float, title: str) -> Tuple[float, float, float, float]:
    parts.append(
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" height="{height:.1f}" '
        'fill="white" stroke="#d0d0d0" stroke-width="1"/>'
    )
    parts.append(_text(x0 + 12, y0 + 22, title, **{"font-size": "15", "font-weight": "700"}))
    left = x0 + 52
    top = y0 + 42
    plot_w = width - 76
    plot_h = height - 96
    return left, top, plot_w, plot_h


def _draw_method_panel(
    parts: List[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    summaries: Sequence[StepSummary],
) -> None:
    left, top, plot_w, plot_h = _panel(parts, x0, y0, width, height, title)
    parts.append(_line(left, top, left, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    parts.append(_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        parts.append(_line(left, y, left + plot_w, y, stroke="#ececec", **{"stroke-width": "1"}))
        parts.append(_text(left - 8, y + 4, f"{frac:.2f}", **{"font-size": "10", "text-anchor": "end", "fill": "#666"}))

    steps = [item.step for item in summaries]
    max_step = max(steps or [1])
    slot_w = plot_w / max(max_step, 1)
    bar_w = slot_w * 0.58
    for summary in summaries:
        center = left + (summary.step - 0.5) * slot_w
        bar_left = center - bar_w / 2.0
        if summary.neutral_groups > 0 and summary.informative_groups == 0:
            parts.append(
                f'<rect x="{bar_left:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{plot_h:.1f}" '
                f'fill="{COLORS["neutral"]}" opacity="0.55" stroke="#888" stroke-width="0.6"/>'
            )
            parts.append(
                _text(
                    center,
                    top + plot_h / 2.0 + 4,
                    "neutral",
                    **{"font-size": "11", "text-anchor": "middle", "fill": "#444", "font-weight": "700"},
                )
            )
        else:
            running = 0.0
            for value, fill in (
                (summary.mean_top1_mass, COLORS["top1"]),
                (summary.mean_other_correct_mass, COLORS["other_correct"]),
                (summary.mean_incorrect_mass, COLORS["incorrect"]),
            ):
                if value <= 0.0:
                    continue
                y = top + plot_h - min(running + value, 1.0) * plot_h
                h = value * plot_h
                parts.append(
                    f'<rect x="{bar_left:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}" stroke="none"/>'
                )
                running += value
            parts.append(
                _text(
                    center,
                    top - 8,
                    f"{summary.mean_correct_count:.1f}/{summary.mean_group_size:.0f}",
                    **{"font-size": "10", "text-anchor": "middle", "fill": "#444"},
                )
            )
        parts.append(_text(center, top + plot_h + 16, str(summary.step), **{"font-size": "10", "text-anchor": "middle", "fill": "#666"}))
    parts.append(_text(left + plot_w / 2.0, y0 + height - 16, "Training step", **{"font-size": "11", "text-anchor": "middle"}))
    parts.append(
        _text(
            left - 38,
            top + plot_h / 2.0,
            "Update mass",
            **{
                "font-size": "11",
                "text-anchor": "middle",
                "transform": f"rotate(-90 {left - 38:.1f} {top + plot_h / 2.0:.1f})",
            },
        )
    )


def _plot_svg(
    grpo_steps: Sequence[StepSummary],
    listwise_steps: Sequence[StepSummary],
    output_path: Path,
) -> None:
    width = 1180
    height = 620
    outer_pad = 28
    title_h = 74
    panel_gap = 24
    panel_h = (height - title_h - outer_pad - panel_gap - 18) / 2.0
    panel_w = width - 2 * outer_pad

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(outer_pad, 24, "Update-Mass Distribution Over Time", **{"font-size": "18", "font-weight": "700"}),
        _text(
            outer_pad,
            44,
            "Each bar is one training step. Blue = top correct rollout, light blue = other correct rollouts, grey = incorrect rollouts.",
            **{"font-size": "12", "fill": "#444"},
        ),
        _text(
            outer_pad,
            60,
            "Solid stacked bars mean the prompt group had mixed rewards. Grey full bars mean all 8 rewards were identical, so there was no rank signal.",
            **{"font-size": "12", "fill": "#444"},
        ),
    ]

    _draw_method_panel(
        parts,
        x0=outer_pad,
        y0=title_h,
        width=panel_w,
        height=panel_h,
        title="Dr.GRPO",
        summaries=grpo_steps,
    )
    _draw_method_panel(
        parts,
        x0=outer_pad,
        y0=title_h + panel_h + panel_gap,
        width=panel_w,
        height=panel_h,
        title="Listwise",
        summaries=listwise_steps,
    )

    legend_y = height - 12
    legend_x = outer_pad
    for idx, (fill, label) in enumerate(
        [
            (COLORS["top1"], "Top-ranked correct rollout"),
            (COLORS["other_correct"], "Other correct rollouts"),
            (COLORS["incorrect"], "Incorrect rollouts"),
            (COLORS["neutral"], "Neutral step (no rank signal)"),
        ]
    ):
        x = legend_x + idx * 235
        parts.append(f'<rect x="{x:.1f}" y="{legend_y - 13:.1f}" width="12" height="12" fill="{fill}" stroke="#666" stroke-width="0.4"/>')
        parts.append(_text(x + 18, legend_y - 2, label, **{"font-size": "11"}))

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


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
    grpo_steps = _aggregate_steps(grpo_records)
    listwise_steps = _aggregate_steps(listwise_records)
    output_path = Path(args.output)
    _plot_svg(grpo_steps, listwise_steps, output_path)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "grpo": [summary.__dict__ for summary in grpo_steps],
            "listwise": [summary.__dict__ for summary in listwise_steps],
            "run_dirs": {
                "grpo": str(grpo_dir),
                "listwise": str(listwise_dir),
            },
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
