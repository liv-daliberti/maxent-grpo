#!/usr/bin/env python3
"""Plot accuracy-aware prompt-level dynamics over training for Dr.GRPO vs listwise."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import plot_listwise_vs_grpo_prompt_dynamics as dyn
from tools import plot_listwise_vs_grpo_distribution as dist


COLORS = {
    "grpo": "#2563eb",
    "listwise": "#16a34a",
}


@dataclass
class StepMetrics:
    step: int
    total_groups: int
    informative_groups: int
    informative_share: float
    rollout_accuracy: float
    prompt_pass_at_8: float
    mean_correct_rollouts: float
    mean_incorrect_mass: float
    mean_top1_mass: float
    mean_effective_rollouts: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render accuracy-aware training dynamics for Dr.GRPO vs listwise."
    )
    parser.add_argument("--grpo-run", required=True)
    parser.add_argument("--listwise-run", required=True)
    parser.add_argument("--wandb-root", default=str(dist.DEFAULT_WANDB_ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--q-temperature", type=float, default=2.0)
    return parser.parse_args()


def _mean(values: Sequence[float]) -> float:
    valid = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(valid) / len(valid)) if valid else float("nan")


def _aggregate_metrics(records: Sequence[dist.GroupRecord]) -> List[StepMetrics]:
    by_step: Dict[int, List[dist.GroupRecord]] = {}
    for record in records:
        by_step.setdefault(int(record.step), []).append(record)

    results: List[StepMetrics] = []
    for step in sorted(by_step):
        step_records = by_step[step]
        informative = [record for record in step_records if dist._is_informative(record)]
        rollout_rewards = [float(reward) for record in step_records for reward in record.rewards]
        correct_counts = [sum(1 for reward in record.rewards if float(reward) > 0.0) for record in step_records]
        pass_at_8 = [1.0 if count > 0 else 0.0 for count in correct_counts]
        results.append(
            StepMetrics(
                step=step,
                total_groups=len(step_records),
                informative_groups=len(informative),
                informative_share=(len(informative) / len(step_records)) if step_records else float("nan"),
                rollout_accuracy=_mean(rollout_rewards),
                prompt_pass_at_8=_mean(pass_at_8),
                mean_correct_rollouts=_mean(correct_counts),
                mean_incorrect_mass=_mean([dyn._incorrect_mass(record) for record in informative]),
                mean_top1_mass=_mean([dyn._top1_mass(record) for record in informative]),
                mean_effective_rollouts=_mean([dyn._effective_rollouts(record) for record in informative]),
            )
        )
    return results


def _write_png_from_svg(svg_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    convert = shutil.which("convert")
    if convert is None:
        return
    subprocess.run([convert, "-background", "white", str(svg_path), str(png_path)], check=True)


def _plot_svg(
    *,
    grpo_steps: Sequence[StepMetrics],
    listwise_steps: Sequence[StepMetrics],
    output_path: Path,
) -> dict:
    width = 1400
    height = 960
    outer = 28
    panel_gap_x = 18
    panel_gap_y = 18
    header_h = 96
    footer_h = 30
    panel_w = (width - 2 * outer - 2 * panel_gap_x) / 3
    panel_h = (height - header_h - footer_h - 2 * outer - panel_gap_y) / 2

    step_union = sorted({item.step for item in grpo_steps} | {item.step for item in listwise_steps})

    metric_map = {
        "rollout_accuracy": (
            [(item.step, item.rollout_accuracy) for item in grpo_steps],
            [(item.step, item.rollout_accuracy) for item in listwise_steps],
            0.0,
            1.0,
            "Rollout Accuracy",
            "Average correctness across all sampled rollouts at that training step.",
            "mean reward / rollout",
        ),
        "prompt_pass_at_8": (
            [(item.step, item.prompt_pass_at_8) for item in grpo_steps],
            [(item.step, item.prompt_pass_at_8) for item in listwise_steps],
            0.0,
            1.0,
            "Prompt Pass@8",
            "Fraction of prompt groups at that step with at least one correct rollout.",
            "share of groups with >=1 correct",
        ),
        "informative_share": (
            [(item.step, item.informative_share) for item in grpo_steps],
            [(item.step, item.informative_share) for item in listwise_steps],
            0.0,
            1.0,
            "Informative-Group Share",
            "Fraction of groups with mixed rewards. Higher means more real rank signal.",
            "share of mixed-reward groups",
        ),
        "mean_incorrect_mass": (
            [(item.step, item.mean_incorrect_mass) for item in grpo_steps],
            [(item.step, item.mean_incorrect_mass) for item in listwise_steps],
            0.0,
            1.0,
            "Incorrect-Mass Share",
            "Average update mass left on wrong rollouts among informative groups.",
            "mass on incorrect rollouts",
        ),
        "mean_top1_mass": (
            [(item.step, item.mean_top1_mass) for item in grpo_steps],
            [(item.step, item.mean_top1_mass) for item in listwise_steps],
            0.0,
            1.0,
            "Top-1 Rollout Mass",
            "Average mass on the highest-ranked rollout among informative groups.",
            "mass on top-ranked rollout",
        ),
        "mean_effective_rollouts": (
            [(item.step, item.mean_effective_rollouts) for item in grpo_steps],
            [(item.step, item.mean_effective_rollouts) for item in listwise_steps],
            1.0,
            8.0,
            "Effective Active Rollouts",
            "Average exp(entropy) among informative groups. 1 means single-winner; 8 means nearly uniform.",
            "effective active rollouts",
        ),
    }

    panels = [
        ("rollout_accuracy", outer, outer + header_h),
        ("prompt_pass_at_8", outer + panel_w + panel_gap_x, outer + header_h),
        ("informative_share", outer + 2 * (panel_w + panel_gap_x), outer + header_h),
        ("mean_incorrect_mass", outer, outer + header_h + panel_h + panel_gap_y),
        ("mean_top1_mass", outer + panel_w + panel_gap_x, outer + header_h + panel_h + panel_gap_y),
        ("mean_effective_rollouts", outer + 2 * (panel_w + panel_gap_x), outer + header_h + panel_h + panel_gap_y),
    ]

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        dyn._text(outer, 30, "Dr.GRPO vs Listwise: Accuracy And Update Dynamics Over Training", **{"font-size": "24", "font-weight": "700"}),
        dyn._text(
            outer,
            52,
            "Top row shows actual step-level correctness on the training prompt groups. Bottom row shows how each method distributes update mass.",
            **{"font-size": "12", "fill": "#444"},
        ),
        dyn._text(
            outer,
            69,
            "All points come from the same rich completion records used for the within-prompt plots.",
            **{"font-size": "12", "fill": "#444"},
        ),
    ]
    parts.append(dyn._line(outer, 88, outer + 24, 88, stroke=COLORS["grpo"], **{"stroke-width": "3"}))
    parts.append(dyn._text(outer + 32, 92, "Dr.GRPO", **{"font-size": "12"}))
    parts.append(dyn._line(outer + 120, 88, outer + 144, 88, stroke=COLORS["listwise"], **{"stroke-width": "3"}))
    parts.append(dyn._text(outer + 152, 92, "Listwise", **{"font-size": "12"}))

    for key, x0, y0 in panels:
        grpo_pts, listwise_pts, y_lo, y_hi, title, subtitle, y_label = metric_map[key]
        left, top, plot_w, plot_h = dyn._panel(parts, x0, y0, panel_w, panel_h, title, subtitle)
        dyn._draw_axes(
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
        dyn._draw_series(
            parts,
            points=grpo_pts,
            color=COLORS["grpo"],
            x0=left,
            y0=top,
            width=plot_w,
            height=plot_h,
            x_steps=step_union,
            y_min=y_lo,
            y_max=y_hi,
        )
        dyn._draw_series(
            parts,
            points=listwise_pts,
            color=COLORS["listwise"],
            x0=left,
            y0=top,
            width=plot_w,
            height=plot_h,
            x_steps=step_union,
            y_min=y_lo,
            y_max=y_hi,
        )

    grpo_last = grpo_steps[-1] if grpo_steps else None
    listwise_last = listwise_steps[-1] if listwise_steps else None
    footer = (
        f"Latest available: GRPO step {grpo_last.step if grpo_last else 'NA'} "
        f"(acc={grpo_last.rollout_accuracy:.3f}, pass@8={grpo_last.prompt_pass_at_8:.3f}) | "
        f"Listwise step {listwise_last.step if listwise_last else 'NA'} "
        f"(acc={listwise_last.rollout_accuracy:.3f}, pass@8={listwise_last.prompt_pass_at_8:.3f})"
        if grpo_last and listwise_last
        else "Latest available metrics unavailable."
    )
    parts.append(dyn._text(outer, height - 18, footer, **{"font-size": "11", "fill": "#555"}))
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

    output_path = Path(args.output)
    summary = _plot_svg(
        grpo_steps=_aggregate_metrics(grpo_records),
        listwise_steps=_aggregate_metrics(listwise_records),
        output_path=output_path.with_suffix(".svg"),
    )
    _write_png_from_svg(output_path.with_suffix(".svg"), output_path.with_suffix(".png"))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
