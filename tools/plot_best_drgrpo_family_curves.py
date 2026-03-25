#!/usr/bin/env python3
"""Plot the best retained local run for each Dr.GRPO-family method.

The script scans local W&B run directories, infers the method family from the
recorded run name, parses metric dicts from ``files/output.log``, selects the
best run per family by maximum pooled greedy ``pass@1`` across the evaluation
suite, and emits a single-panel figure of pooled greedy ``pass@1`` vs epoch.

This is designed for the current local workflow where some live runs only have
``output.log`` available and not a full W&B history export.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
FAMILY_ORDER = ["grpo", "seed", "maxent", "listwise"]
FAMILY_LABELS = {
    "grpo": "A. Dr.GRPO",
    "seed": "B. SEED-Dr.GRPO",
    "maxent": "C. Token MaxEnt Dr.GRPO",
    "listwise": "D. Listwise MaxEnt Dr.GRPO",
}
FAMILY_COLORS = {
    "grpo": "#1f77b4",
    "seed": "#d4a000",
    "listwise": "#2ca02c",
    "maxent": "#9467bd",
}


@dataclass
class CurvePoint:
    epoch: float
    value: float


@dataclass
class RunRecord:
    run_dir: str
    run_name: str
    family: str
    best_eval_pooled_accuracy: Optional[float]
    train_accuracy_points: List[CurvePoint]
    eval_points: List[CurvePoint]


def _clean_text(raw: str) -> str:
    return ANSI_RE.sub("", raw).replace("\r", "")


def _parse_dict_lines(output_log: Path) -> Iterable[dict]:
    with output_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = _clean_text(raw_line).strip()
            if "{" not in line or "}" not in line:
                continue
            start = line.find("{")
            end = line.rfind("}")
            if start < 0 or end <= start:
                continue
            payload = line[start : end + 1]
            try:
                parsed = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, dict):
                yield parsed


def _infer_run_name(run_dir: Path) -> Optional[str]:
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


def _infer_family(run_name: Optional[str]) -> Optional[str]:
    if not run_name:
        return None
    for family in FAMILY_ORDER:
        if run_name.endswith(f"-{family}"):
            return family
    return None


def _build_run_record(run_dir: Path) -> Optional[RunRecord]:
    output_log = run_dir / "files" / "output.log"
    if not output_log.exists():
        return None
    run_name = _infer_run_name(run_dir)
    family = _infer_family(run_name)
    if family is None:
        return None

    train_accuracy_points: List[CurvePoint] = []
    eval_points: List[CurvePoint] = []
    best_eval: Optional[float] = None

    for metrics in _parse_dict_lines(output_log):
        epoch_raw = metrics.get("epoch", 0.0)
        try:
            epoch = float(epoch_raw)
        except (TypeError, ValueError):
            epoch = 0.0

        eval_value = math.nan
        eval_key = None
        if "eval_pass_at_1" in metrics:
            eval_key = "eval_pass_at_1"
        elif "eval/pass_at_1" in metrics:
            eval_key = "eval/pass_at_1"
        if eval_key is not None:
            try:
                eval_value = float(metrics[eval_key])
            except (TypeError, ValueError):
                eval_value = math.nan

        if math.isfinite(eval_value):
            eval_points.append(CurvePoint(epoch=epoch, value=eval_value))
            if best_eval is None or eval_value > best_eval:
                best_eval = eval_value

        if "train/loss/total" in metrics and "rewards/accuracy_reward/mean" in metrics:
            try:
                train_accuracy = float(metrics["rewards/accuracy_reward/mean"])
            except (TypeError, ValueError):
                train_accuracy = math.nan
            if math.isfinite(train_accuracy):
                train_accuracy_points.append(CurvePoint(epoch=epoch, value=train_accuracy))

    if best_eval is None and not train_accuracy_points and not eval_points:
        return None

    return RunRecord(
        run_dir=str(run_dir),
        run_name=run_name or run_dir.name,
        family=family,
        best_eval_pooled_accuracy=best_eval,
        train_accuracy_points=train_accuracy_points,
        eval_points=eval_points,
    )


def _pick_best_run(records: Iterable[RunRecord]) -> Dict[str, RunRecord]:
    best: Dict[str, RunRecord] = {}
    for record in records:
        current = best.get(record.family)
        if current is None:
            best[record.family] = record
            continue

        current_key = (
            -math.inf
            if current.best_eval_pooled_accuracy is None
            else current.best_eval_pooled_accuracy,
            len(current.train_accuracy_points),
            len(current.eval_points),
            current.run_dir,
        )
        candidate_key = (
            -math.inf
            if record.best_eval_pooled_accuracy is None
            else record.best_eval_pooled_accuracy,
            len(record.train_accuracy_points),
            len(record.eval_points),
            record.run_dir,
        )
        if candidate_key > current_key:
            best[record.family] = record
    return best


def _configure_matplotlib() -> None:
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    if "Times New Roman" in installed_fonts:
        font_family = "Times New Roman"
    elif "Nimbus Roman" in installed_fonts:
        font_family = "Nimbus Roman"
    else:
        font_family = "serif"
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def _moving_average(points: List[CurvePoint], window: int = 15) -> List[CurvePoint]:
    if not points:
        return []
    if len(points) <= 2:
        return points
    span = max(3, min(window, len(points)))
    half = span // 2
    smoothed: List[CurvePoint] = []
    for idx, point in enumerate(points):
        start = max(0, idx - half)
        end = min(len(points), idx + half + 1)
        window_points = points[start:end]
        mean_value = sum(p.value for p in window_points) / len(window_points)
        smoothed.append(CurvePoint(epoch=point.epoch, value=mean_value))
    return smoothed


def _plot_runs(best_runs: Dict[str, RunRecord], output_path: Path) -> None:
    _configure_matplotlib()
    fig, ax_eval = plt.subplots(1, 1, figsize=(4.9, 4.1), constrained_layout=True)
    fig.suptitle("Qwen2.5-Math-1.5B", y=1.06)

    for family in FAMILY_ORDER:
        record = best_runs.get(family)
        if record is None:
            continue
        color = FAMILY_COLORS[family]
        label = FAMILY_LABELS[family]

        if record.eval_points:
            ax_eval.plot(
                [p.epoch for p in record.eval_points],
                [p.value for p in record.eval_points],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=color,
                label=label,
            )
        else:
            ax_eval.plot(
                [],
                [],
                linewidth=1.8,
                color=color,
                label=f"{label} (no retained eval points)",
            )

    ax_eval.set_title("Pooled Greedy Pass @1 Across 5 Benchmarks")
    ax_eval.set_xlabel("Epoch")
    ax_eval.set_ylabel("Pooled Accuracy")
    ax_eval.grid(True, alpha=0.25)
    ax_eval.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-root",
        default="wandb",
        help="Local W&B run directory root.",
    )
    parser.add_argument(
        "--output",
        default="var/artifacts/plots/best_drgrpo_family_curves.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--manifest",
        default="var/artifacts/plots/best_drgrpo_family_curves.json",
        help="JSON manifest path describing selected runs.",
    )
    args = parser.parse_args()

    wandb_root = Path(args.wandb_root)
    records = [
        record
        for run_dir in sorted(wandb_root.glob("run-*"))
        if (record := _build_run_record(run_dir)) is not None
    ]
    best_runs = _pick_best_run(records)
    if not best_runs:
        raise SystemExit("No parsable local W&B runs were found.")

    output_path = Path(args.output)
    _plot_runs(best_runs, output_path)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selected_runs": {
            family: asdict(record) for family, record in sorted(best_runs.items())
        }
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved figure to {output_path}")
    print(f"Saved PDF to {output_path.with_suffix('.pdf')}")
    print(f"Saved manifest to {manifest_path}")
    for family in FAMILY_ORDER:
        record = best_runs.get(family)
        if record is None:
            continue
        best_eval = (
            "n/a"
            if record.best_eval_pooled_accuracy is None
            else f"{record.best_eval_pooled_accuracy:.4f}"
        )
        print(
            f"{family}: run={record.run_name} | best_pooled_eval_accuracy={best_eval} | "
            f"train_accuracy_points={len(record.train_accuracy_points)} | "
            f"eval_points={len(record.eval_points)}"
        )


if __name__ == "__main__":
    main()
