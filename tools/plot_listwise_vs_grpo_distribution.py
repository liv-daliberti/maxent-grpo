#!/usr/bin/env python3
"""Plot how listwise redistributes rollout mass relative to plain Dr.GRPO.

This script is designed to work with the local W&B completion tables emitted by
the training loop. It prefers exact per-sample columns added to newer runs:

- ``reward_total``
- ``q_mass``
- ``update_weight_raw``
- ``update_mass_proxy``

For older runs that only contain reward/advantage tables, it falls back to:

- GRPO mass: normalize positive grouped advantages
- Listwise mass: q-targets reconstructed as ``softmax(reward_total / q_temperature)``

The resulting figure focuses on within-prompt rollout distributions rather than
top-line reward curves:

1. Which prompt groups were informative versus neutral.
2. Average rollout mass by reward rank within informative prompt groups.
3. How each method splits update mass across top-ranked, other-correct, and
   incorrect rollouts.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


RUN_FILE_RE = re.compile(r"(?:rich_completions|completions)_(\d+)_")
SIDECAR_FILE_RE = re.compile(r"(?:rich_completions|completions)_step_(\d+)\.json$")
DEFAULT_WANDB_ROOT = Path("var/wandb/runs/wandb")
COLORS = {
    "grpo": "#1f77b4",
    "listwise": "#2ca02c",
}


@dataclass
class GroupRecord:
    step: int
    prompt_index: int
    rewards: List[float]
    mass: List[float]


@dataclass
class AnalysisResult:
    label: str
    total_groups: int
    groups_used: int
    neutral_groups: int
    mass_source: str
    avg_mass_by_rank: List[float]
    entropies: List[float]
    mean_correct_mass: float
    mean_other_correct_mass: float
    mean_incorrect_mass: float
    top1_mass_mean: float
    step_statuses: List[Tuple[int, bool, int, int]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot within-prompt rollout mass distributions for plain Dr.GRPO "
            "versus listwise MaxEnt."
        )
    )
    parser.add_argument(
        "--grpo-run",
        required=True,
        help="W&B run directory or run_name substring for the GRPO run.",
    )
    parser.add_argument(
        "--listwise-run",
        required=True,
        help="W&B run directory or run_name substring for the listwise run.",
    )
    parser.add_argument(
        "--wandb-root",
        default=str(DEFAULT_WANDB_ROOT),
        help=f"Root containing local W&B runs. Default: {DEFAULT_WANDB_ROOT}",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path (.png or .pdf).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON path for numeric summary output.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Optional cap on prompt groups per run (0 = use all).",
    )
    parser.add_argument(
        "--q-temperature",
        type=float,
        default=2.0,
        help="Fallback listwise q temperature when q_mass is absent. Default: 2.0",
    )
    parser.add_argument(
        "--include-neutral-groups",
        action="store_true",
        help="Include prompt groups with zero reward variance. Default skips them.",
    )
    return parser.parse_args()


def _read_run_name(run_dir: Path) -> Optional[str]:
    meta_path = run_dir / "files" / "wandb-metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    args = meta.get("args")
    if not isinstance(args, list):
        return None
    for idx, arg in enumerate(args[:-1]):
        if arg == "--run_name":
            value = args[idx + 1]
            if isinstance(value, str):
                return value
    return None


def _read_output_dir(run_dir: Path) -> Optional[Path]:
    meta_path = run_dir / "files" / "wandb-metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    args = meta.get("args")
    if not isinstance(args, list):
        return None
    for idx, arg in enumerate(args[:-1]):
        if arg == "--output_dir":
            value = args[idx + 1]
            if isinstance(value, str) and value.strip():
                return Path(value.strip())
    return None


def _resolve_run(spec: str, wandb_root: Path) -> Path:
    candidate = Path(spec)
    if candidate.exists():
        return candidate
    matches: List[Tuple[str, Path]] = []
    for run_dir in sorted(wandb_root.glob("run-*")):
        run_name = _read_run_name(run_dir)
        if run_name is None:
            continue
        if spec == run_name or spec in run_name:
            matches.append((run_name, run_dir))
    if not matches:
        raise FileNotFoundError(f"No local W&B run matched '{spec}'.")
    exact = [item for item in matches if item[0] == spec]
    if len(exact) == 1:
        return exact[0][1]
    if len(matches) == 1:
        return matches[0][1]
    rendered = ", ".join(name for name, _ in matches[:6])
    raise ValueError(
        f"Ambiguous run spec '{spec}'. Matches include: {rendered}"
    )


def _completion_tables(run_dir: Path) -> List[Path]:
    table_dir = run_dir / "files" / "media" / "table"
    tables: List[Path] = []
    if table_dir.exists():
        tables = list(table_dir.glob("rich_completions_*.table.json"))
        if not tables:
            tables = list(table_dir.glob("completions_*.table.json"))
    sidecars: List[Path] = []
    output_dir = _read_output_dir(run_dir)
    if output_dir is not None:
        sidecar_dir = output_dir / "rich_completions"
        if sidecar_dir.exists():
            sidecars = list(sidecar_dir.glob("rich_completions_step_*.json"))
            if not sidecars:
                sidecars = list(sidecar_dir.glob("completions_step_*.json"))
    if sidecars and (not tables or len(sidecars) > len(tables)):
        tables = sidecars

    def _step_key(path: Path) -> tuple[int, str]:
        match = RUN_FILE_RE.search(path.name)
        if match is not None:
            return (int(match.group(1)), path.name)
        sidecar_match = SIDECAR_FILE_RE.search(path.name)
        if sidecar_match is not None:
            return (int(sidecar_match.group(1)), path.name)
        return (math.inf, path.name)

    tables.sort(
        key=_step_key,
    )
    return tables


def _softmax(values: Sequence[float], temperature: float) -> List[float]:
    if not values:
        return []
    temp = max(float(temperature), 1e-8)
    arr = np.asarray(values, dtype=np.float64) / temp
    arr = arr - np.max(arr)
    probs = np.exp(arr)
    denom = float(np.sum(probs))
    if denom <= 0.0 or not math.isfinite(denom):
        return [float("nan")] * len(values)
    return [float(val / denom) for val in probs]


def _normalize_positive(values: Sequence[float]) -> List[float]:
    positives = [max(float(val), 0.0) for val in values]
    denom = sum(positives)
    if denom > 0.0:
        return [val / denom for val in positives]
    return [float("nan")] * len(values)


def _entropy(values: Sequence[float]) -> float:
    filtered = [float(val) for val in values if isinstance(val, (int, float)) and math.isfinite(float(val)) and float(val) > 0.0]
    if not filtered:
        return float("nan")
    total = sum(filtered)
    if total <= 0.0:
        return float("nan")
    normalized = [val / total for val in filtered]
    return float(-sum(val * math.log(val) for val in normalized))


def _extract_reward_column(columns: Sequence[str]) -> Optional[str]:
    if "reward_total" in columns:
        return "reward_total"
    preferred = "reward/seed_paper_boxed_accuracy_reward_math"
    if preferred in columns:
        return preferred
    reward_cols = [col for col in columns if col.startswith("reward/")]
    return reward_cols[0] if reward_cols else None


def _iter_group_rows(obj: dict) -> Iterable[List[dict]]:
    columns = obj.get("columns")
    data = obj.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        return
    col_idx = {str(col): idx for idx, col in enumerate(columns)}

    if "prompt_index" in col_idx and "step" in col_idx:
        grouped: Dict[Tuple[int, int], List[dict]] = {}
        for row in data:
            if not isinstance(row, list):
                continue
            payload = {
                str(col): row[idx] if idx < len(row) else None
                for col, idx in col_idx.items()
            }
            try:
                key = (int(payload.get("step", 0)), int(payload.get("prompt_index", -1)))
            except (TypeError, ValueError):
                continue
            grouped.setdefault(key, []).append(payload)
        for key in sorted(grouped.keys()):
            rows = grouped[key]
            rows.sort(
                key=lambda item: int(item.get("completion_index", 0) or 0)
            )
            yield rows
        return

    current: List[dict] = []
    current_key: Tuple[object, object] | None = None
    for row in data:
        if not isinstance(row, list):
            continue
        payload = {
            str(col): row[idx] if idx < len(row) else None
            for col, idx in col_idx.items()
        }
        key = (payload.get("step"), payload.get("prompt"))
        if current and key != current_key:
            yield current
            current = []
        current_key = key
        current.append(payload)
    if current:
        yield current


def _group_record_from_rows(
    rows: Sequence[dict],
    *,
    label: str,
    q_temperature: float,
    include_neutral_groups: bool,
) -> Optional[Tuple[GroupRecord, str]]:
    if not rows:
        return None
    reward_col = _extract_reward_column(list(rows[0].keys()))
    if reward_col is None:
        return None

    rewards: List[float] = []
    advantages: List[float] = []
    q_values: List[float] = []
    proxy_values: List[float] = []
    prompt_index = -1
    step = 0
    has_q = True
    has_proxy = True
    for row in rows:
        try:
            rewards.append(float(row.get(reward_col, float("nan"))))
        except (TypeError, ValueError):
            return None
        try:
            advantages.append(float(row.get("advantage", float("nan"))))
        except (TypeError, ValueError):
            advantages.append(float("nan"))
        try:
            q_values.append(float(row.get("q_mass", float("nan"))))
        except (TypeError, ValueError):
            q_values.append(float("nan"))
        try:
            proxy_values.append(float(row.get("update_mass_proxy", float("nan"))))
        except (TypeError, ValueError):
            proxy_values.append(float("nan"))
        try:
            prompt_index = int(row.get("prompt_index", prompt_index))
        except (TypeError, ValueError):
            pass
        try:
            step = int(row.get("step", step))
        except (TypeError, ValueError):
            pass

    reward_min = min(rewards) if rewards else 0.0
    reward_max = max(rewards) if rewards else 0.0
    if not include_neutral_groups and abs(reward_max - reward_min) <= 1e-8:
        return None

    has_q = all(math.isfinite(val) for val in q_values)
    has_proxy = all(math.isfinite(val) for val in proxy_values)

    if label == "grpo":
        mass = proxy_values if has_proxy else _normalize_positive(advantages)
        source = "logged_proxy" if has_proxy else "positive_advantage_proxy"
    else:
        if has_proxy:
            mass = proxy_values
            source = "logged_proxy"
        elif has_q:
            mass = q_values
            source = "logged_q_mass"
        else:
            mass = _softmax(rewards, q_temperature)
            source = "softmax_reward_fallback"

    if not mass or not any(math.isfinite(val) for val in mass):
        if include_neutral_groups and abs(reward_max - reward_min) <= 1e-8 and rewards:
            uniform = [1.0 / len(rewards)] * len(rewards)
            return (
                GroupRecord(
                    step=step,
                    prompt_index=prompt_index,
                    rewards=rewards,
                    mass=uniform,
                ),
                "neutral_uniform_placeholder",
            )
        return None
    return GroupRecord(step=step, prompt_index=prompt_index, rewards=rewards, mass=mass), source


def _load_records(
    run_dir: Path,
    *,
    label: str,
    q_temperature: float,
    include_neutral_groups: bool,
    max_groups: int,
) -> Tuple[List[GroupRecord], str]:
    records: List[GroupRecord] = []
    source_counts: Dict[str, int] = {}
    for table_path in _completion_tables(run_dir):
        try:
            obj = json.loads(table_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for rows in _iter_group_rows(obj):
            parsed = _group_record_from_rows(
                rows,
                label=label,
                q_temperature=q_temperature,
                include_neutral_groups=include_neutral_groups,
            )
            if parsed is None:
                continue
            record, source = parsed
            records.append(record)
            source_counts[source] = source_counts.get(source, 0) + 1
            if max_groups > 0 and len(records) >= max_groups:
                break
        if max_groups > 0 and len(records) >= max_groups:
            break
    if not records:
        raise RuntimeError(
            "No usable prompt groups found for "
            f"{label} in {run_dir}. Older completion tables often lack "
            "prompt-group identity, so rerun with the current logging code "
            "that emits prompt_index/group_size/q_mass/update_weight_raw."
        )
    source = max(source_counts.items(), key=lambda item: item[1])[0]
    return records, source


def _sorted_mass_by_reward_rank(record: GroupRecord) -> List[float]:
    order = sorted(
        range(len(record.rewards)),
        key=lambda idx: (-float(record.rewards[idx]), idx),
    )
    return [float(record.mass[idx]) for idx in order]


def _is_informative(record: GroupRecord) -> bool:
    if not record.rewards:
        return False
    return abs(max(record.rewards) - min(record.rewards)) > 1e-8


def _count_correct(record: GroupRecord) -> int:
    return sum(1 for reward in record.rewards if float(reward) > 0.0)


def _analyze(
    label: str,
    all_records: Sequence[GroupRecord],
    records: Sequence[GroupRecord],
    mass_source: str,
) -> AnalysisResult:
    sorted_groups = [_sorted_mass_by_reward_rank(record) for record in records if record.mass]
    max_len = max(len(group) for group in sorted_groups)
    avg_mass_by_rank: List[float] = []
    for rank in range(max_len):
        rank_vals = [group[rank] for group in sorted_groups if rank < len(group)]
        avg_mass_by_rank.append(float(np.mean(rank_vals)) if rank_vals else float("nan"))
    entropies = [_entropy(record.mass) for record in records]
    correct_mass_vals = []
    other_correct_mass_vals = []
    incorrect_mass_vals = []
    top1_vals = []
    for record in records:
        top1_vals.append(_sorted_mass_by_reward_rank(record)[0])
        correct_mass = sum(
            float(mass)
            for reward, mass in zip(record.rewards, record.mass)
            if float(reward) > 0.0
        )
        incorrect_mass = sum(
            float(mass)
            for reward, mass in zip(record.rewards, record.mass)
            if float(reward) <= 0.0
        )
        correct_mass_vals.append(correct_mass)
        other_correct_mass_vals.append(max(correct_mass - top1_vals[-1], 0.0))
        incorrect_mass_vals.append(incorrect_mass)
    step_statuses = sorted(
        [
            (
                int(record.step),
                _is_informative(record),
                _count_correct(record),
                len(record.rewards),
            )
            for record in all_records
        ],
        key=lambda item: item[0],
    )
    return AnalysisResult(
        label=label,
        total_groups=len(all_records),
        groups_used=len(records),
        neutral_groups=max(len(all_records) - len(records), 0),
        mass_source=mass_source,
        avg_mass_by_rank=avg_mass_by_rank,
        entropies=[val for val in entropies if math.isfinite(val)],
        mean_correct_mass=float(np.mean(correct_mass_vals)) if correct_mass_vals else float("nan"),
        mean_other_correct_mass=float(np.mean(other_correct_mass_vals))
        if other_correct_mass_vals
        else float("nan"),
        mean_incorrect_mass=float(np.mean(incorrect_mass_vals)) if incorrect_mass_vals else float("nan"),
        top1_mass_mean=float(np.mean(top1_vals)) if top1_vals else float("nan"),
        step_statuses=step_statuses,
    )


def _plot(result_grpo: AnalysisResult, result_listwise: AnalysisResult, output_path: Path) -> None:
    if output_path.suffix.lower() == ".svg":
        _plot_svg(result_grpo, result_listwise, output_path)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)

    ax = axes[0]
    max_step = max(
        [step for step, _, _, _ in result_grpo.step_statuses + result_listwise.step_statuses]
        or [1]
    )
    method_rows = [
        ("Dr.GRPO", result_grpo, COLORS["grpo"]),
        ("Listwise", result_listwise, COLORS["listwise"]),
    ]
    for row_idx, (_, result, color) in enumerate(method_rows):
        y = len(method_rows) - 1 - row_idx
        for step, informative, num_correct, group_size in result.step_statuses:
            face = color if informative else "#d9d9d9"
            ax.add_patch(
                Rectangle(
                    (step - 0.42, y - 0.28),
                    0.84,
                    0.56,
                    facecolor=face,
                    edgecolor="#666666",
                    linewidth=0.8,
                )
            )
            ax.text(
                step,
                y,
                f"{num_correct}/{group_size}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if informative else "#444444",
                fontweight="600",
            )
    ax.set_xlim(0.4, max_step + 0.6)
    ax.set_ylim(-0.7, 1.5)
    ax.set_xticks(range(1, max_step + 1))
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Dr.GRPO", "Listwise"])
    ax.set_title("Which Prompt Groups Carry Rank Signal?")
    ax.set_xlabel("Training step")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(
        handles=[
            Patch(facecolor="#888888", edgecolor="#666666", label="Mixed rewards: used later"),
            Patch(facecolor="#d9d9d9", edgecolor="#666666", label="All rewards equal: dropped later"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    ax.text(
        0.0,
        -0.23,
        (
            f"Later panels use only mixed-reward groups: "
            f"GRPO {result_grpo.groups_used}/{result_grpo.total_groups}, "
            f"Listwise {result_listwise.groups_used}/{result_listwise.total_groups}."
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )

    ax = axes[1]
    x_grpo = np.arange(1, len(result_grpo.avg_mass_by_rank) + 1)
    x_listwise = np.arange(1, len(result_listwise.avg_mass_by_rank) + 1)
    ax.plot(
        x_grpo,
        result_grpo.avg_mass_by_rank,
        marker="o",
        linewidth=2.0,
        color=COLORS["grpo"],
        label="Dr.GRPO",
    )
    ax.plot(
        x_listwise,
        result_listwise.avg_mass_by_rank,
        marker="o",
        linewidth=2.0,
        color=COLORS["listwise"],
        label="Listwise",
    )
    ax.axhline(1.0 / 8.0, linestyle="--", linewidth=1.4, color="#999999", label="Uniform 1/8")
    ax.set_title("Among Informative Groups, Where Does Mass Go?")
    ax.set_xlabel("Reward Rank Within Prompt Group")
    ax.set_ylabel("Mass")
    ax.set_xticks(range(1, max(len(x_grpo), len(x_listwise)) + 1))
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False)

    ax = axes[2]
    categories = ["Dr.GRPO", "Listwise"]
    xpos = np.arange(len(categories))
    width = 0.58
    top1_vals = [result_grpo.top1_mass_mean, result_listwise.top1_mass_mean]
    other_correct_vals = [
        result_grpo.mean_other_correct_mass,
        result_listwise.mean_other_correct_mass,
    ]
    incorrect_vals = [result_grpo.mean_incorrect_mass, result_listwise.mean_incorrect_mass]
    ax.bar(xpos, top1_vals, width=width, color="#4c78a8", label="Top-ranked correct rollout")
    ax.bar(
        xpos,
        other_correct_vals,
        width=width,
        bottom=top1_vals,
        color="#9ecae9",
        label="Other correct rollouts",
    )
    ax.bar(
        xpos,
        incorrect_vals,
        width=width,
        bottom=np.asarray(top1_vals) + np.asarray(other_correct_vals),
        color="#d9d9d9",
        label="Incorrect rollouts",
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("How Each Method Uses Its Update Mass")
    ax.set_ylabel("Average Mass")
    ax.grid(alpha=0.25, linestyle=":", axis="y")
    ax.legend(frameon=False)

    fig.suptitle(
        (
            "Within-Prompt Rollout Mass: Listwise vs Dr.GRPO\n"
            "Numbers inside boxes are correct rollouts / total rollouts. Grey boxes are neutral groups with no rank signal."
        ),
        y=1.04,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_svg(
    result_grpo: AnalysisResult,
    result_listwise: AnalysisResult,
    output_path: Path,
) -> None:
    width = 1320
    height = 500
    outer_pad = 28
    panel_gap = 26
    title_h = 70
    panel_w = (width - 2 * outer_pad - 2 * panel_gap) / 3.0
    panel_h = height - title_h - outer_pad - 26
    panel_y = title_h

    def _panel_origin(idx: int) -> Tuple[float, float]:
        return outer_pad + idx * (panel_w + panel_gap), panel_y

    def _panel_frame(x0: float, y0: float, title: str) -> List[str]:
        return [
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="white" stroke="#d0d0d0" stroke-width="1"/>',
            f'<text x="{x0 + 12:.1f}" y="{y0 + 20:.1f}" font-size="15" font-weight="600">{html.escape(title)}</text>',
        ]

    def _plot_rect(x0: float, y0: float) -> Tuple[float, float, float, float]:
        left = x0 + 44
        top = y0 + 34
        w = panel_w - 60
        h = panel_h - 86
        return left, top, w, h

    def _polyline(points: Sequence[Tuple[float, float]], color: str) -> str:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def _scatter(points: Sequence[Tuple[float, float]], color: str) -> str:
        return "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="{color}"/>'
            for x, y in points
        )

    def _line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
        extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
        return (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {extra}/>'
        )

    def _text(x: float, y: float, text: str, **attrs: str) -> str:
        extra = " ".join(f'{key}="{value}"' for key, value in attrs.items())
        return f'<text x="{x:.1f}" y="{y:.1f}" {extra}>{html.escape(text)}</text>'

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(
            outer_pad,
            22,
            "Within-Prompt Rollout Mass: Listwise vs Dr.GRPO",
            **{"font-size": "18", "font-weight": "700"},
        ),
        _text(
            outer_pad,
            42,
            "Numbers inside boxes are correct rollouts / total rollouts. Grey boxes are neutral groups with no rank signal.",
            **{"font-size": "12", "fill": "#444"},
        ),
        _text(
            outer_pad,
            58,
            f"Mass source: GRPO {result_grpo.mass_source} | Listwise {result_listwise.mass_source}",
            **{"font-size": "12", "fill": "#444"},
        ),
    ]

    # Panel 1: which steps were informative.
    x0, y0 = _panel_origin(0)
    parts.extend(_panel_frame(x0, y0, "Which Prompt Groups Carry Rank Signal?"))
    left, top, plot_w, plot_h = _plot_rect(x0, y0)
    max_step = max(
        [step for step, _, _, _ in result_grpo.step_statuses + result_listwise.step_statuses]
        or [1]
    )
    parts.append(_line(left, top + plot_h - 10, left + plot_w, top + plot_h - 10, stroke="#333", **{"stroke-width": "1.2"}))
    slot_w = plot_w / max(max_step, 1)
    row_y = {
        "grpo": top + plot_h * 0.30,
        "listwise": top + plot_h * 0.72,
    }
    rect_h = 42.0
    for method, result, color in (
        ("grpo", result_grpo, COLORS["grpo"]),
        ("listwise", result_listwise, COLORS["listwise"]),
    ):
        y = row_y[method]
        parts.append(_text(left - 10, y + 5, "Dr.GRPO" if method == "grpo" else "Listwise", **{"font-size": "12", "text-anchor": "end"}))
        for step, informative, num_correct, group_size in result.step_statuses:
            x = left + (step - 1) * slot_w + 4
            rect_w = slot_w - 8
            fill = color if informative else "#d9d9d9"
            text_fill = "white" if informative else "#444"
            parts.append(
                f'<rect x="{x:.1f}" y="{y - rect_h/2:.1f}" width="{rect_w:.1f}" height="{rect_h:.1f}" '
                f'fill="{fill}" stroke="#666" stroke-width="0.8" rx="4" ry="4"/>'
            )
            parts.append(_text(x + rect_w / 2.0, y + 5, f"{num_correct}/{group_size}", **{"font-size": "12", "text-anchor": "middle", "fill": text_fill, "font-weight": "700"}))
    for step in range(1, max_step + 1):
        x = left + (step - 0.5) * slot_w
        parts.append(_text(x, top + plot_h - 18, str(step), **{"font-size": "10", "text-anchor": "middle", "fill": "#666"}))
    legend_x = left
    legend_y = y0 + panel_h - 26
    parts.append(f'<rect x="{legend_x:.1f}" y="{legend_y - 10:.1f}" width="12" height="12" fill="#888888" stroke="#666"/>')
    parts.append(_text(legend_x + 18, legend_y, "Mixed rewards: used in later panels", **{"font-size": "11"}))
    parts.append(f'<rect x="{legend_x + 210:.1f}" y="{legend_y - 10:.1f}" width="12" height="12" fill="#d9d9d9" stroke="#666"/>')
    parts.append(_text(legend_x + 228, legend_y, "All rewards equal: dropped later", **{"font-size": "11"}))
    parts.append(
        _text(
            left,
            y0 + panel_h - 6,
            (
                f"Later panels use only mixed-reward groups: "
                f"GRPO {result_grpo.groups_used}/{result_grpo.total_groups}, "
                f"Listwise {result_listwise.groups_used}/{result_listwise.total_groups}."
            ),
            **{"font-size": "11", "fill": "#444"},
        )
    )

    # Panel 2: average mass by reward rank.
    x0, y0 = _panel_origin(1)
    parts.extend(_panel_frame(x0, y0, "Among Informative Groups, Where Does Mass Go?"))
    left, top, plot_w, plot_h = _plot_rect(x0, y0)
    max_rank = max(len(result_grpo.avg_mass_by_rank), len(result_listwise.avg_mass_by_rank))
    y_max = max(
        1e-6,
        max(result_grpo.avg_mass_by_rank + result_listwise.avg_mass_by_rank + [1.0 / 8.0]),
    )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        parts.append(_line(left, y, left + plot_w, y, stroke="#ececec", **{"stroke-width": "1"}))
        parts.append(_text(left - 8, y + 4, f"{frac * y_max:.2f}", **{"font-size": "10", "text-anchor": "end", "fill": "#666"}))
    parts.append(_line(left, top, left, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    parts.append(_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    uniform_y = top + plot_h - ((1.0 / 8.0) / y_max) * plot_h
    parts.append(_line(left, uniform_y, left + plot_w, uniform_y, stroke="#999", **{"stroke-width": "1.6", "stroke-dasharray": "6 5"}))
    parts.append(_text(left + plot_w - 4, uniform_y - 6, "Uniform 1/8", **{"font-size": "10", "text-anchor": "end", "fill": "#666"}))
    grpo_pts: List[Tuple[float, float]] = []
    listwise_pts: List[Tuple[float, float]] = []
    for idx, value in enumerate(result_grpo.avg_mass_by_rank, start=1):
        x = left + (idx - 1) * (plot_w / max(max_rank - 1, 1))
        y = top + plot_h - (value / y_max) * plot_h
        grpo_pts.append((x, y))
        parts.append(_text(x, top + plot_h + 16, str(idx), **{"font-size": "10", "text-anchor": "middle", "fill": "#666"}))
    for idx, value in enumerate(result_listwise.avg_mass_by_rank, start=1):
        x = left + (idx - 1) * (plot_w / max(max_rank - 1, 1))
        y = top + plot_h - (value / y_max) * plot_h
        listwise_pts.append((x, y))
    if grpo_pts:
        parts.append(_polyline(grpo_pts, COLORS["grpo"]))
        parts.append(_scatter(grpo_pts, COLORS["grpo"]))
    if listwise_pts:
        parts.append(_polyline(listwise_pts, COLORS["listwise"]))
        parts.append(_scatter(listwise_pts, COLORS["listwise"]))
    parts.append(_text(left + plot_w / 2.0, y0 + panel_h - 12, "Reward rank within prompt group", **{"font-size": "11", "text-anchor": "middle"}))
    parts.append(_text(left - 34, top + plot_h / 2.0, "Average mass", **{"font-size": "11", "text-anchor": "middle", "transform": f"rotate(-90 {left - 34:.1f} {top + plot_h / 2.0:.1f})"}))

    # Panel 3: stacked mass destination bars.
    x0, y0 = _panel_origin(2)
    parts.extend(_panel_frame(x0, y0, "How Each Method Uses Its Update Mass"))
    left, top, plot_w, plot_h = _plot_rect(x0, y0)
    parts.append(_line(left, top, left, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    parts.append(_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#333", **{"stroke-width": "1.2"}))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        parts.append(_line(left, y, left + plot_w, y, stroke="#ececec", **{"stroke-width": "1"}))
        parts.append(_text(left - 8, y + 4, f"{frac:.2f}", **{"font-size": "10", "text-anchor": "end", "fill": "#666"}))
    labels = ["Dr.GRPO", "Listwise"]
    top_vals = [result_grpo.top1_mass_mean, result_listwise.top1_mass_mean]
    other_vals = [result_grpo.mean_other_correct_mass, result_listwise.mean_other_correct_mass]
    bad_vals = [result_grpo.mean_incorrect_mass, result_listwise.mean_incorrect_mass]
    slot_w = plot_w / len(labels)
    bar_w = slot_w * 0.42
    fill_top = "#4c78a8"
    fill_other = "#9ecae9"
    fill_bad = "#d9d9d9"
    for idx, label in enumerate(labels):
        center = left + idx * slot_w + slot_w / 2.0
        bar_left = center - bar_w / 2.0
        running = 0.0
        for value, fill in (
            (top_vals[idx], fill_top),
            (other_vals[idx], fill_other),
            (bad_vals[idx], fill_bad),
        ):
            y = top + plot_h - max(min(running + value, 1.0), 0.0) * plot_h
            h = max(min(value, 1.0), 0.0) * plot_h
            parts.append(f'<rect x="{bar_left:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}" stroke="none"/>')
            running += value
        parts.append(_text(center, top + plot_h + 16, label, **{"font-size": "10", "text-anchor": "middle", "fill": "#666"}))
    parts.append(_text(left - 34, top + plot_h / 2.0, "Average mass", **{"font-size": "11", "text-anchor": "middle", "transform": f"rotate(-90 {left - 34:.1f} {top + plot_h / 2.0:.1f})"}))
    legend_x = left
    legend_y = y0 + panel_h - 26
    parts.append(f'<rect x="{legend_x:.1f}" y="{legend_y - 10:.1f}" width="12" height="12" fill="{fill_top}"/>')
    parts.append(_text(legend_x + 18, legend_y, "Top-ranked correct rollout", **{"font-size": "11"}))
    parts.append(f'<rect x="{legend_x + 178:.1f}" y="{legend_y - 10:.1f}" width="12" height="12" fill="{fill_other}"/>')
    parts.append(_text(legend_x + 196, legend_y, "Other correct rollouts", **{"font-size": "11"}))
    parts.append(f'<rect x="{legend_x + 348:.1f}" y="{legend_y - 10:.1f}" width="12" height="12" fill="{fill_bad}"/>')
    parts.append(_text(legend_x + 366, legend_y, "Incorrect rollouts", **{"font-size": "11"}))

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    wandb_root = Path(args.wandb_root)
    grpo_dir = _resolve_run(str(args.grpo_run), wandb_root)
    listwise_dir = _resolve_run(str(args.listwise_run), wandb_root)
    grpo_all_records, _ = _load_records(
        grpo_dir,
        label="grpo",
        q_temperature=float(args.q_temperature),
        include_neutral_groups=True,
        max_groups=int(args.max_groups),
    )
    listwise_all_records, _ = _load_records(
        listwise_dir,
        label="listwise",
        q_temperature=float(args.q_temperature),
        include_neutral_groups=True,
        max_groups=int(args.max_groups),
    )
    grpo_records, grpo_source = _load_records(
        grpo_dir,
        label="grpo",
        q_temperature=float(args.q_temperature),
        include_neutral_groups=bool(args.include_neutral_groups),
        max_groups=int(args.max_groups),
    )
    listwise_records, listwise_source = _load_records(
        listwise_dir,
        label="listwise",
        q_temperature=float(args.q_temperature),
        include_neutral_groups=bool(args.include_neutral_groups),
        max_groups=int(args.max_groups),
    )
    result_grpo = _analyze("grpo", grpo_all_records, grpo_records, grpo_source)
    result_listwise = _analyze(
        "listwise",
        listwise_all_records,
        listwise_records,
        listwise_source,
    )
    _plot(result_grpo, result_listwise, Path(args.output))

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "grpo": {
                "total_groups": result_grpo.total_groups,
                "groups_used": result_grpo.groups_used,
                "neutral_groups": result_grpo.neutral_groups,
                "mass_source": result_grpo.mass_source,
                "avg_mass_by_rank": result_grpo.avg_mass_by_rank,
                "mean_entropy": float(np.mean(result_grpo.entropies)) if result_grpo.entropies else float("nan"),
                "mean_correct_mass": result_grpo.mean_correct_mass,
                "mean_other_correct_mass": result_grpo.mean_other_correct_mass,
                "mean_incorrect_mass": result_grpo.mean_incorrect_mass,
                "top1_mass_mean": result_grpo.top1_mass_mean,
                "step_statuses": result_grpo.step_statuses,
                "run_dir": str(grpo_dir),
            },
            "listwise": {
                "total_groups": result_listwise.total_groups,
                "groups_used": result_listwise.groups_used,
                "neutral_groups": result_listwise.neutral_groups,
                "mass_source": result_listwise.mass_source,
                "avg_mass_by_rank": result_listwise.avg_mass_by_rank,
                "mean_entropy": float(np.mean(result_listwise.entropies)) if result_listwise.entropies else float("nan"),
                "mean_correct_mass": result_listwise.mean_correct_mass,
                "mean_other_correct_mass": result_listwise.mean_other_correct_mass,
                "mean_incorrect_mass": result_listwise.mean_incorrect_mass,
                "top1_mass_mean": result_listwise.top1_mass_mean,
                "step_statuses": result_listwise.step_statuses,
                "run_dir": str(listwise_dir),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
