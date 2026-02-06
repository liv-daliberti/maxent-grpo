#!/usr/bin/env python3
"""Summarize math-eval artifacts into Pass@k tables."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

DATASET_ORDER = ["math_500", "aime24", "aime25", "amc", "minerva"]
STEP_PATTERN = re.compile(r"checkpoint[-_](\d+)")


@dataclass
class SeedSummary:
    model_id: str
    label: str
    style: str
    dataset: str
    temperature: Optional[float]
    seed: int
    total: int
    pass_at_1: float
    pass_at_k: float
    avg_pass_at_k: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Pass@1/Pass@k/Avg@k tables from inference artifacts."
    )
    parser.add_argument(
        "--artifact-root",
        default="var/artifacts/inference",
        help="Root directory containing inference artifact JSONL files.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Number of decimal places for percentages.",
    )
    parser.add_argument(
        "--min-datasets",
        type=int,
        default=1,
        help="Only display checkpoints that cover at least this many datasets.",
    )
    parser.add_argument(
        "--plot-name",
        default="plot/math_eval_graphic.png",
        help="Relative path (under artifact root) for the summary plot.",
    )
    return parser.parse_args()


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _score_flags(scores: Sequence[object]) -> Tuple[bool, bool]:
    if not scores:
        return False, False
    numeric = [_to_float(score) for score in scores]
    pass1 = numeric[0] >= 1.0 if numeric else False
    passk = any(val >= 1.0 for val in numeric)
    return pass1, passk


def dataset_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (DATASET_ORDER.index(name), name)
    except ValueError:
        return (len(DATASET_ORDER), name)


def extract_step(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    match = STEP_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def derive_family(model_id: str) -> Tuple[str, str]:
    path = Path(model_id)
    checkpoint = path.name or model_id
    family = path.parent.name or path.name or model_id
    return family, checkpoint


def load_seed_summary(path: Path) -> Optional[SeedSummary]:
    total = 0
    pass1_hits = 0.0
    passk_hits = 0.0
    avgk_sum = 0.0
    meta: Dict[str, object] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not meta:
                meta = payload
            scores = payload.get("scores") or []
            pass1_ok, passk_ok = _score_flags(scores)
            total += 1
            pass1_hits += 1.0 if pass1_ok else 0.0
            passk_hits += 1.0 if passk_ok else 0.0
            score_list = [_to_float(score) for score in scores] if scores else []
            avgk_sum += sum(score_list) / len(score_list) if score_list else 0.0
    if not total or not meta:
        return None
    dataset = str(
        meta.get("dataset_id")
        or meta.get("dataset_name")
        or meta.get("split")
        or "dataset"
    )
    temperature = meta.get("temperature")
    temperature_value: Optional[float]
    if temperature is None or temperature == "na":
        temperature_value = None
    else:
        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError):
            temperature_value = None
    return SeedSummary(
        model_id=str(meta.get("model_id") or meta.get("model_label") or "model"),
        label=str(meta.get("model_label") or meta.get("model_id") or path.stem),
        style=str(meta.get("style") or ""),
        dataset=dataset,
        temperature=temperature_value,
        seed=int(meta.get("seed", -1)),
        total=total,
        pass_at_1=pass1_hits / total,
        pass_at_k=passk_hits / total,
        avg_pass_at_k=avgk_sum / total,
    )


def collect_seed_summaries(root: Path) -> List[SeedSummary]:
    summaries: List[SeedSummary] = []
    for json_path in sorted(root.rglob("*.jsonl")):
        summary = load_seed_summary(json_path)
        if summary:
            summaries.append(summary)
    # LightEval saves per-run JSON blobs; map best-effort accuracy to pass@1.
    for json_path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        results: Dict[str, Dict[str, Union[float, int, str, dict]]] = data.get(
            "results", {}
        )
        model_id = str(
            data.get("model_name")
            or data.get("model_id")
            or json_path.parent.name
            or "model"
        )
        label = json_path.stem
        style = str(data.get("style", "")) if isinstance(data, dict) else ""
        for task_name, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            acc = None
            for key in ("acc", "accuracy", "exact_match", "score"):
                if key in metrics:
                    try:
                        acc = float(metrics[key])
                        break
                    except (TypeError, ValueError):
                        acc = None
            if acc is None:
                continue
            summaries.append(
                SeedSummary(
                    model_id=model_id,
                    label=label,
                    style=style,
                    dataset=str(task_name),
                    temperature=None,
                    seed=-1,
                    total=1,
                    pass_at_1=acc,
                    pass_at_k=acc,
                    avg_pass_at_k=acc,
                )
            )
    return summaries


def format_pct(value: Optional[float], precision: int) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value * 100:.{precision}f}%"


def build_rows(
    summaries: Iterable[SeedSummary], min_datasets: int
) -> Tuple[List[dict], List[str]]:
    grouped: Dict[
        Tuple[str, str, str, Optional[float]],
        Dict[str, Dict[str, List[float]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta: Dict[
        Tuple[str, str, str, Optional[float]],
        Dict[str, object],
    ] = {}
    for summary in summaries:
        key = (summary.model_id, summary.label, summary.style, summary.temperature)
        ds_metrics = grouped[key][summary.dataset]
        ds_metrics["pass_at_1"].append(summary.pass_at_1)
        ds_metrics["pass_at_k"].append(summary.pass_at_k)
        ds_metrics["avg_pass_at_k"].append(summary.avg_pass_at_k)
        if key not in meta:
            family, checkpoint = derive_family(summary.model_id)
            meta[key] = {
                "family": family,
                "checkpoint": checkpoint,
                "label": summary.label,
                "style": summary.style,
                "temperature": summary.temperature,
                "step": extract_step(checkpoint)
                or extract_step(summary.label)
                or math.inf,
            }
    dataset_names = sorted(
        {summary.dataset for summary in summaries},
        key=dataset_sort_key,
    )
    rows: List[dict] = []
    for key, datasets in grouped.items():
        model_meta = meta[key]
        if len(datasets) < max(1, min_datasets):
            continue
        metrics: Dict[str, Dict[str, Optional[float]]] = {}
        for dataset, stats in datasets.items():
            metrics[dataset] = {
                name: sum(values) / len(values) if values else None
                for name, values in stats.items()
            }
        rows.append(
            {
                **model_meta,
                "metrics": metrics,
            }
        )
    rows.sort(key=lambda row: (row["family"], row["step"], row["checkpoint"]))
    return rows, dataset_names


def render_table(
    rows: List[dict], dataset_names: List[str], metric: str, precision: int
) -> str:
    headers = ["family", "ckpt", "style", "temp", *dataset_names, "avg"]
    table_rows: List[List[str]] = []
    for row in rows:
        metric_values = [
            row["metrics"].get(dataset, {}).get(metric) for dataset in dataset_names
        ]
        avg_value: Optional[float]
        valid = [value for value in metric_values if value is not None]
        avg_value = sum(valid) / len(valid) if valid else None
        temp_val = row.get("temperature")
        temp_str = "n/a" if temp_val is None else f"{temp_val:.2f}"
        table_rows.append(
            [
                row.get("family", ""),
                row.get("checkpoint", ""),
                row.get("style", ""),
                temp_str,
                *[format_pct(value, precision) for value in metric_values],
                format_pct(avg_value, precision),
            ]
        )
    if not table_rows:
        return "No rows matched the requested filters."
    col_widths = [
        max(len(headers[idx]), *(len(r[idx]) for r in table_rows))
        for idx in range(len(headers))
    ]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells))

    lines = [fmt_row(headers)]
    lines.append("  ".join("-" * width for width in col_widths))
    lines.extend(fmt_row(row) for row in table_rows)
    return "\n".join(lines)


def _average_metric(
    row: dict, dataset_names: List[str], metric: str
) -> Optional[float]:
    values = [row["metrics"].get(dataset, {}).get(metric) for dataset in dataset_names]
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _build_series(
    rows: List[dict], dataset_names: List[str], metric: str
) -> Dict[Tuple[str, str, Optional[float]], List[Tuple[int, float]]]:
    grouped: Dict[Tuple[str, str, Optional[float]], List[Tuple[int, float]]] = (
        defaultdict(list)
    )
    for row in rows:
        step_val = row.get("step")
        if step_val is None or not math.isfinite(step_val):
            continue
        avg_val = _average_metric(row, dataset_names, metric)
        if avg_val is None:
            continue
        key = (row.get("family", ""), row.get("style", ""), row.get("temperature"))
        grouped[key].append((int(step_val), avg_val))
    for key, points in list(grouped.items()):
        points.sort(key=lambda pair: pair[0])
        grouped[key] = points
    return grouped


def render_plots(
    rows: List[dict],
    dataset_names: List[str],
    metrics: List[Tuple[str, str]],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot generation.")
        return
    series_data = {
        metric_key: _build_series(rows, dataset_names, metric_key)
        for _, metric_key in metrics
    }
    if not any(series_data.values()):
        print("No numeric data available to plot.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5 * len(metrics), 4), sharex=True
    )
    if len(metrics) == 1:
        axes = [axes]
    legend_map: Dict[str, object] = {}
    for ax, (title, metric_key) in zip(axes, metrics):
        ax.set_title(title)
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel("avg across datasets")
        for (family, style, temp), points in series_data.get(metric_key, {}).items():
            if not points:
                continue
            xs, ys = zip(*points)
            label_parts = [family or "model"]
            if style:
                label_parts.append(style)
            if temp is not None:
                label_parts.append(f"T={temp:.2f}")
            label = " | ".join(label_parts)
            (line,) = ax.plot(xs, ys, marker="o", label=label)
            legend_map.setdefault(label, line)
        ax.grid(alpha=0.3)
    if legend_map:
        legend_labels = list(legend_map.keys())
        legend_handles = [legend_map[lbl] for lbl in legend_labels]
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(3, len(set(legend_labels))),
            bbox_to_anchor=(0.5, -0.1),
        )
        fig.subplots_adjust(bottom=0.15)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Wrote comparison plot to {output_path}")


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root)
    if not root.exists():
        raise SystemExit(f"Artifact root {root} does not exist.")
    summaries = collect_seed_summaries(root)
    if not summaries:
        raise SystemExit(f"No inference artifacts found under {root}.")
    rows, dataset_names = build_rows(summaries, args.min_datasets)
    metrics = [
        ("Pass@1", "pass_at_1"),
        ("Pass@8", "pass_at_k"),
        ("Avg@8", "avg_pass_at_k"),
    ]
    for title, metric in metrics:
        print(f"\n{title}")
        print(render_table(rows, dataset_names, metric, args.precision))
    output_path = Path(args.artifact_root) / args.plot_name
    render_plots(rows, dataset_names, metrics, output_path)


if __name__ == "__main__":
    main()
