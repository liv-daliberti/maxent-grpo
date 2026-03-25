#!/usr/bin/env python3
"""Plot held-out pass@8 rollout mass for Dr.GRPO versus listwise."""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


COLORS = {
    "grpo": "#1f77b4",
    "listwise": "#2ca02c",
    "top1": "#4c78a8",
    "other_correct": "#9ecae9",
    "incorrect": "#d9d9d9",
}


@dataclass
class PromptGroup:
    task_name: str
    prompt_index: int
    rewards: List[float]
    mass: List[float]
    informative: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot held-out full-suite pass@8 distribution for GRPO vs listwise."
    )
    parser.add_argument("--grpo-pass8-json", required=True)
    parser.add_argument("--listwise-pass8-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--listwise-tau", type=float, default=0.5)
    parser.add_argument("--listwise-beta", type=float, default=0.08)
    parser.add_argument("--listwise-q-temperature", type=float, default=2.0)
    parser.add_argument(
        "--listwise-len-norm-ref",
        action="store_true",
        default=True,
    )
    return parser.parse_args()


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_val = max(values)
    exp_vals = [math.exp(v - max_val) for v in values]
    denom = sum(exp_vals) or 1.0
    return [val / denom for val in exp_vals]


def _normalize_positive(values: Sequence[float]) -> List[float]:
    positives = [max(float(v), 0.0) for v in values]
    denom = sum(positives)
    if denom <= 0.0:
        return [float("nan")] * len(positives)
    return [v / denom for v in positives]


def _load_groups(
    path: Path,
    *,
    method: str,
    listwise_tau: float,
    listwise_beta: float,
    listwise_q_temperature: float,
    listwise_len_norm_ref: bool,
) -> List[PromptGroup]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups: List[PromptGroup] = []
    for entry in payload:
        task_name = str(entry["task_name"])
        prompt_index = int(entry["prompt_index"])
        samples = list(entry.get("samples") or [])
        rewards = [float(sample.get("reward", 0.0)) for sample in samples]
        informative = bool(rewards) and (max(rewards) - min(rewards) > 1e-8)
        if method == "grpo":
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            advantages = [reward - mean_reward for reward in rewards]
            mass = _normalize_positive(advantages)
        else:
            if not informative:
                mass = [1.0 / len(rewards)] * len(rewards) if rewards else []
            else:
                q = _softmax(
                    [reward / max(listwise_q_temperature, 1e-8) for reward in rewards]
                )
                log_terms = [math.log(max(val, 1e-12)) / max(listwise_tau, 1e-8) for val in q]
                if listwise_beta > 0.0:
                    ref_terms: List[float] = []
                    usable_ref = True
                    for sample in samples:
                        token_count = sample.get("token_count")
                        logprob_sum = sample.get("logprob_sum")
                        if token_count in (None, 0) or logprob_sum is None:
                            usable_ref = False
                            break
                        denom = float(token_count) if listwise_len_norm_ref else 1.0
                        ref_terms.append(float(logprob_sum) / max(denom, 1e-8))
                    if usable_ref:
                        log_terms = [
                            term + (listwise_beta * ref_term) / max(listwise_tau, 1e-8)
                            for term, ref_term in zip(log_terms, ref_terms)
                        ]
                mass = _softmax(log_terms)
        groups.append(
            PromptGroup(
                task_name=task_name,
                prompt_index=prompt_index,
                rewards=rewards,
                mass=mass,
                informative=informative,
            )
        )
    return groups


def _avg_mass_by_rank(groups: Sequence[PromptGroup]) -> List[float]:
    ranked: List[List[float]] = []
    for group in groups:
        if not group.informative:
            continue
        order = sorted(range(len(group.rewards)), key=lambda idx: (-group.rewards[idx], idx))
        ranked.append([group.mass[idx] for idx in order])
    if not ranked:
        return []
    width = max(len(row) for row in ranked)
    out: List[float] = []
    for idx in range(width):
        vals = [row[idx] for row in ranked if idx < len(row)]
        out.append(sum(vals) / len(vals))
    return out


def _mass_breakdown(groups: Sequence[PromptGroup]) -> Tuple[float, float, float]:
    top_vals: List[float] = []
    other_vals: List[float] = []
    incorrect_vals: List[float] = []
    for group in groups:
        if not group.informative:
            continue
        order = sorted(range(len(group.rewards)), key=lambda idx: (-group.rewards[idx], idx))
        top_mass = group.mass[order[0]] if order else 0.0
        correct_mass = sum(m for r, m in zip(group.rewards, group.mass) if r > 0.0)
        incorrect_mass = sum(m for r, m in zip(group.rewards, group.mass) if r <= 0.0)
        top_vals.append(float(top_mass))
        other_vals.append(float(max(correct_mass - top_mass, 0.0)))
        incorrect_vals.append(float(incorrect_mass))
    if not top_vals:
        return 0.0, 0.0, 0.0
    n = len(top_vals)
    return (
        sum(top_vals) / n,
        sum(other_vals) / n,
        sum(incorrect_vals) / n,
    )


def _task_informative_fraction(groups: Sequence[PromptGroup]) -> Dict[str, Tuple[int, int]]:
    stats: Dict[str, Tuple[int, int]] = {}
    counts: Dict[str, List[int]] = {}
    for group in groups:
        bucket = counts.setdefault(group.task_name, [0, 0])
        bucket[0] += 1
        bucket[1] += int(group.informative)
    for task_name, (total, informative) in counts.items():
        stats[task_name] = (int(total), int(informative))
    return stats


def _text(x: float, y: float, text: str, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {extra}>{html.escape(text)}</text>'


def _line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
    extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {extra}/>'


def _plot_svg(
    *,
    grpo_groups: Sequence[PromptGroup],
    listwise_groups: Sequence[PromptGroup],
    output_path: Path,
) -> dict[str, object]:
    tasks = ["aime", "amc", "math", "minerva", "olympiad_bench"]
    grpo_task_stats = _task_informative_fraction(grpo_groups)
    listwise_task_stats = _task_informative_fraction(listwise_groups)
    grpo_rank = _avg_mass_by_rank(grpo_groups)
    listwise_rank = _avg_mass_by_rank(listwise_groups)
    grpo_breakdown = _mass_breakdown(grpo_groups)
    listwise_breakdown = _mass_breakdown(listwise_groups)

    width = 1320
    height = 470
    outer_pad = 28
    panel_gap = 26
    title_h = 68
    panel_w = (width - 2 * outer_pad - 2 * panel_gap) / 3.0
    panel_h = height - title_h - outer_pad - 24
    panel_y = title_h

    def panel_origin(idx: int) -> Tuple[float, float]:
        return outer_pad + idx * (panel_w + panel_gap), panel_y

    def panel_frame(x0: float, y0: float, title: str) -> List[str]:
        return [
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="white" stroke="#d0d0d0" stroke-width="1"/>',
            _text(x0 + 12, y0 + 20, title, **{"font-size": "15", "font-weight": "700"}),
        ]

    def plot_rect(x0: float, y0: float) -> Tuple[float, float, float, float]:
        return x0 + 46, y0 + 36, panel_w - 62, panel_h - 76

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(outer_pad, 24, "Held-Out Full-Suite pass@8 Distribution", **{"font-size": "18", "font-weight": "700"}),
        _text(
            outer_pad,
            44,
            "Computed from saved SEED pass@8 outputs on AIME, AMC, MATH500, Minerva, and OlympiadBench.",
            **{"font-size": "12", "fill": "#444"},
        ),
        _text(
            outer_pad,
            60,
            "Listwise weights use reward-softmax q plus the saved sequence log-probs when available; GRPO mass is the normalized positive centered reward.",
            **{"font-size": "12", "fill": "#444"},
        ),
    ]

    # Panel 1: informative fraction by task.
    x0, y0 = panel_origin(0)
    parts.extend(panel_frame(x0, y0, "Informative Prompt Groups By Task"))
    left, top, plot_w, plot_h = plot_rect(x0, y0)
    parts.append(_line(left, top, left, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    parts.append(_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        parts.append(_line(left, y, left + plot_w, y, stroke="#ececec", **{"stroke-width": "1"}))
        parts.append(_text(left - 8, y + 4, f"{frac:.2f}", **{"font-size": "10", "text-anchor": "end", "fill": "#666"}))
    slot_w = plot_w / max(len(tasks), 1)
    bar_w = slot_w * 0.28
    for idx, task in enumerate(tasks):
        center = left + idx * slot_w + slot_w / 2.0
        g_total, g_info = grpo_task_stats.get(task, (0, 0))
        l_total, l_info = listwise_task_stats.get(task, (0, 0))
        g_frac = (g_info / g_total) if g_total else 0.0
        l_frac = (l_info / l_total) if l_total else 0.0
        gy = top + plot_h - g_frac * plot_h
        ly = top + plot_h - l_frac * plot_h
        parts.append(f'<rect x="{center - bar_w - 4:.1f}" y="{gy:.1f}" width="{bar_w:.1f}" height="{top + plot_h - gy:.1f}" fill="{COLORS["grpo"]}"/>')
        parts.append(f'<rect x="{center + 4:.1f}" y="{ly:.1f}" width="{bar_w:.1f}" height="{top + plot_h - ly:.1f}" fill="{COLORS["listwise"]}"/>')
        parts.append(_text(center, top + plot_h + 16, task, **{"font-size": "10", "text-anchor": "middle", "fill": "#666"}))
        parts.append(_text(center, top + plot_h + 30, f"G {g_info}/{g_total} | L {l_info}/{l_total}", **{"font-size": "9", "text-anchor": "middle", "fill": "#666"}))

    # Panel 2: rank mass.
    x0, y0 = panel_origin(1)
    parts.extend(panel_frame(x0, y0, "Average Mass By Reward Rank"))
    left, top, plot_w, plot_h = plot_rect(x0, y0)
    y_max = max(grpo_rank + listwise_rank + [1.0 / 8.0, 1e-6])
    max_rank = max(len(grpo_rank), len(listwise_rank))
    parts.append(_line(left, top, left, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    parts.append(_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        parts.append(_line(left, y, left + plot_w, y, stroke="#ececec", **{"stroke-width": "1"}))
        parts.append(_text(left - 8, y + 4, f"{frac * y_max:.2f}", **{"font-size": "10", "text-anchor": "end", "fill": "#666"}))
    uniform_y = top + plot_h - ((1.0 / 8.0) / y_max) * plot_h
    parts.append(_line(left, uniform_y, left + plot_w, uniform_y, stroke="#999", **{"stroke-width": "1.6", "stroke-dasharray": "6 5"}))
    parts.append(_text(left + plot_w - 4, uniform_y - 6, "Uniform 1/8", **{"font-size": "10", "text-anchor": "end", "fill": "#666"}))

    def poly(values: Sequence[float], color: str) -> None:
        if not values:
            return
        pts = []
        for idx, value in enumerate(values, start=1):
            x = left + (idx - 1) * (plot_w / max(max_rank - 1, 1))
            y = top + plot_h - (value / y_max) * plot_h
            pts.append((x, y))
            parts.append(_text(x, top + plot_h + 16, str(idx), **{"font-size": "10", "text-anchor": "middle", "fill": "#666"}))
        coord = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polyline points="{coord}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="{color}"/>')

    poly(grpo_rank, COLORS["grpo"])
    poly(listwise_rank, COLORS["listwise"])

    # Panel 3: overall stacked mass breakdown.
    x0, y0 = panel_origin(2)
    parts.extend(panel_frame(x0, y0, "Overall Update-Mass Breakdown"))
    left, top, plot_w, plot_h = plot_rect(x0, y0)
    parts.append(_line(left, top, left, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    parts.append(_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        parts.append(_line(left, y, left + plot_w, y, stroke="#ececec", **{"stroke-width": "1"}))
        parts.append(_text(left - 8, y + 4, f"{frac:.2f}", **{"font-size": "10", "text-anchor": "end", "fill": "#666"}))
    labels = ["Dr.GRPO", "Listwise"]
    slot_w = plot_w / 2.0
    bar_w = slot_w * 0.42
    for idx, (label, breakdown) in enumerate(zip(labels, [grpo_breakdown, listwise_breakdown], strict=True)):
        center = left + idx * slot_w + slot_w / 2.0
        bar_left = center - bar_w / 2.0
        running = 0.0
        for value, fill in (
            (breakdown[0], COLORS["top1"]),
            (breakdown[1], COLORS["other_correct"]),
            (breakdown[2], COLORS["incorrect"]),
        ):
            if value <= 0.0:
                continue
            y = top + plot_h - min(running + value, 1.0) * plot_h
            h = value * plot_h
            parts.append(f'<rect x="{bar_left:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}" stroke="none"/>')
            running += value
        parts.append(_text(center, top + plot_h + 16, label, **{"font-size": "10", "text-anchor": "middle", "fill": "#666"}))

    legend_x = left
    legend_y = y0 + panel_h - 16
    for idx, (fill, label) in enumerate(
        [
            (COLORS["top1"], "Top-ranked correct rollout"),
            (COLORS["other_correct"], "Other correct rollouts"),
            (COLORS["incorrect"], "Incorrect rollouts"),
        ]
    ):
        x = legend_x + idx * 170
        parts.append(f'<rect x="{x:.1f}" y="{legend_y - 10:.1f}" width="12" height="12" fill="{fill}" stroke="#666" stroke-width="0.4"/>')
        parts.append(_text(x + 18, legend_y, label, **{"font-size": "11"}))

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")

    return {
        "grpo": {
            "total_groups": len(grpo_groups),
            "informative_groups": sum(int(group.informative) for group in grpo_groups),
            "task_stats": {task: {"total": total, "informative": info} for task, (total, info) in grpo_task_stats.items()},
            "avg_mass_by_rank": grpo_rank,
            "breakdown": {
                "top1": grpo_breakdown[0],
                "other_correct": grpo_breakdown[1],
                "incorrect": grpo_breakdown[2],
            },
        },
        "listwise": {
            "total_groups": len(listwise_groups),
            "informative_groups": sum(int(group.informative) for group in listwise_groups),
            "task_stats": {task: {"total": total, "informative": info} for task, (total, info) in listwise_task_stats.items()},
            "avg_mass_by_rank": listwise_rank,
            "breakdown": {
                "top1": listwise_breakdown[0],
                "other_correct": listwise_breakdown[1],
                "incorrect": listwise_breakdown[2],
            },
        },
    }


def main() -> int:
    args = _parse_args()
    grpo_groups = _load_groups(
        Path(args.grpo_pass8_json),
        method="grpo",
        listwise_tau=float(args.listwise_tau),
        listwise_beta=float(args.listwise_beta),
        listwise_q_temperature=float(args.listwise_q_temperature),
        listwise_len_norm_ref=bool(args.listwise_len_norm_ref),
    )
    listwise_groups = _load_groups(
        Path(args.listwise_pass8_json),
        method="listwise",
        listwise_tau=float(args.listwise_tau),
        listwise_beta=float(args.listwise_beta),
        listwise_q_temperature=float(args.listwise_q_temperature),
        listwise_len_norm_ref=bool(args.listwise_len_norm_ref),
    )
    summary = _plot_svg(
        grpo_groups=grpo_groups,
        listwise_groups=listwise_groups,
        output_path=Path(args.output),
    )
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
