#!/usr/bin/env python3
"""Render a best-so-far quartet pass@1 table for a specific parity run pair."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"step-(\d+)")
BENCHMARK_KEYS = [
    ("aime", "AIME24"),
    ("amc", "AMC"),
    ("math", "MATH500"),
    ("minerva", "MIN"),
    ("olympiad_bench", "OLY"),
]


@dataclass
class SummaryPoint:
    step: int
    avg: float
    pass_at_8_avg: float | None
    mean_at_8_avg: float | None
    results: dict[str, float]
    pass_at_8: dict[str, float]
    mean_at_8: dict[str, float]
    summary_path: str


def _parse_step(path: Path) -> int:
    match = STEP_RE.search(path.as_posix())
    if match is None:
        raise ValueError(f"Could not parse step from {path}")
    return int(match.group(1))


def _load_summary(path: Path) -> SummaryPoint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SummaryPoint(
        step=_parse_step(path),
        avg=float(payload["avg"]),
        pass_at_8_avg=(
            float(payload["pass_at_8_avg"]) if payload.get("pass_at_8_avg") is not None else None
        ),
        mean_at_8_avg=(
            float(payload["mean_at_8_avg"]) if payload.get("mean_at_8_avg") is not None else None
        ),
        results={str(k): float(v) for k, v in (payload.get("results") or {}).items()},
        pass_at_8={str(k): float(v) for k, v in (payload.get("pass_at_8") or {}).items()},
        mean_at_8={str(k): float(v) for k, v in (payload.get("mean_at_8") or {}).items()},
        summary_path=str(path),
    )


def _load_series(run_dir: Path) -> list[SummaryPoint]:
    points: list[SummaryPoint] = []
    for summary_path in sorted(run_dir.glob("step-*/*.summary.json")):
        try:
            point = _load_summary(summary_path)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue
        if len(point.results) != 5:
            continue
        points.append(point)
    points.sort(key=lambda item: item.step)
    return points


def _select_best(points: list[SummaryPoint]) -> SummaryPoint:
    if not points:
        raise ValueError("No summary points available.")
    return max(
        points,
        key=lambda item: (
            item.avg,
            item.pass_at_8_avg if item.pass_at_8_avg is not None else -1.0,
            item.mean_at_8_avg if item.mean_at_8_avg is not None else -1.0,
            item.step,
        ),
    )


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _table_row(label: str, step: int | str, values: dict[str, float], avg: float) -> str:
    parts = [label, str(step)]
    for key, _ in BENCHMARK_KEYS:
        parts.append(_pct(values[key]))
    parts.append(_pct(avg))
    return "    " + " & ".join(parts) + r" \\"


def _write_table(
    output_path: Path,
    *,
    base_label: str,
    base_step: int,
    base_values: dict[str, float],
    base_avg: float,
    grpo_point: SummaryPoint,
    listwise_point: SummaryPoint,
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \begin{tabular}{lccccccc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Step} & \textbf{AIME24} & \textbf{AMC} & \textbf{MATH500} & \textbf{MIN} & \textbf{OLY} & \textbf{Pooled pass@1} \\",
        r"    \midrule",
        _table_row(base_label, base_step, base_values, base_avg),
        _table_row("Dr.GRPO-1.5B", grpo_point.step, grpo_point.results, grpo_point.avg),
        _table_row(
            "Dr.GRPO-Explorer-1.5B",
            listwise_point.step,
            listwise_point.results,
            listwise_point.avg,
        ),
        r"    Token MaxEnt-1.5B                    & --- & ---  & ---  & ---  & ---  & ---  & ---  \\",
        r"    SEED-Dr.GRPO-1.5B                    & --- & ---  & ---  & ---  & ---  & ---  & ---  \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{2mm}",
        r"  \caption{\textbf{Official \texttt{pass@1} results on the current five-benchmark quartet suite.}",
        r"  Entries are official evaluator percentages from the current live rich-sidecar parity run.",
        r"  The \textbf{Qwen2.5-1.5B-Instruct (base)} row is the pre-finetuning checkpoint evaluated at step 0.",
        r"  For \textbf{Dr.GRPO} and \textbf{Dr.GRPO-Explorer}, we report the checkpoint with best pooled \texttt{pass@1} within the same live run.}",
        r"  \label{tab:quartet-pass1-results}",
        r"\end{table*}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _maybe_fetch_wandb_run(entity: str, project: str, display_name: str) -> dict[str, Any] | None:
    try:
        import wandb
    except Exception:
        return None
    try:
        api = wandb.Api(timeout=20)
        runs = list(api.runs(f"{entity}/{project}", {"display_name": display_name}, per_page=2))
    except Exception:
        return None
    if not runs:
        return None
    run = runs[0]
    return {
        "id": run.id,
        "name": run.name,
        "display_name": run.display_name,
        "group": run.group,
        "state": run.state,
        "url": getattr(run, "url", None),
        "summary": {
            "paper_eval/latest_step": run.summary.get("paper_eval/latest_step"),
            "paper_eval/avg": run.summary.get("paper_eval/avg"),
            "paper_eval/aime": run.summary.get("paper_eval/aime"),
            "paper_eval/amc": run.summary.get("paper_eval/amc"),
            "paper_eval/math": run.summary.get("paper_eval/math"),
            "paper_eval/minerva": run.summary.get("paper_eval/minerva"),
            "paper_eval/olympiad_bench": run.summary.get("paper_eval/olympiad_bench"),
            "paper_eval/pass_at_8_avg": run.summary.get("paper_eval/pass_at_8_avg"),
            "paper_eval/mean_at_8_avg": run.summary.get("paper_eval/mean_at_8_avg"),
        },
    }


def _build_manifest(
    output_path: Path,
    *,
    base_label: str,
    base_step: int,
    base_values: dict[str, float],
    base_avg: float,
    base_summary_path: str,
    grpo_dir: Path,
    listwise_dir: Path,
    grpo_series: list[SummaryPoint],
    listwise_series: list[SummaryPoint],
    grpo_best: SummaryPoint,
    listwise_best: SummaryPoint,
    grpo_wandb: dict[str, Any] | None,
    listwise_wandb: dict[str, Any] | None,
) -> None:
    payload = {
        "table_kind": "quartet_pass1_bestsofar",
        "selection_rule": "best pooled pass@1 (avg) among official 5-task summaries",
        "base_row": {
            "label": base_label,
            "step": base_step,
            "avg": base_avg,
            "results": base_values,
            "summary_path": base_summary_path,
        },
        "grpo_dir": str(grpo_dir),
        "listwise_dir": str(listwise_dir),
        "grpo_best": asdict(grpo_best),
        "listwise_best": asdict(listwise_best),
        "grpo_series": [asdict(point) for point in grpo_series],
        "listwise_series": [asdict(point) for point in listwise_series],
        "wandb": {
            "grpo": grpo_wandb,
            "listwise": listwise_wandb,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo-dir", required=True)
    parser.add_argument("--listwise-dir", required=True)
    parser.add_argument("--grpo-run-name", required=True)
    parser.add_argument("--listwise-run-name", required=True)
    parser.add_argument(
        "--base-summary",
        default=(
            "var/artifacts/seed_paper_eval/live/"
            "full_eval_richsidecar_Qwen2.5-1.5B-Instruct_math_fair_mltheory_tau0p25_beta0p08_"
            "no_template_livepass8_parity_20260326_095838-grpo/step-000000/"
            "seed_paper_eval_20260326T135040Z.summary.json"
        ),
    )
    parser.add_argument("--base-label", default="Qwen2.5-1.5B-Instruct (base)")
    parser.add_argument("--wandb-entity", default="ogd3-princeton-university")
    parser.add_argument("--wandb-project", default="huggingface")
    parser.add_argument(
        "--output-prefix",
        default="var/artifacts/paper/quartet_pass1_bestsofar_20260328_174335",
    )
    args = parser.parse_args()

    grpo_dir = Path(args.grpo_dir)
    listwise_dir = Path(args.listwise_dir)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    base_summary_path = Path(args.base_summary)
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    base_values = {str(k): float(v) for k, v in (base_summary.get("results") or {}).items()}
    base_avg = float(base_summary["avg"])

    grpo_series = _load_series(grpo_dir)
    listwise_series = _load_series(listwise_dir)
    grpo_best = _select_best(grpo_series)
    listwise_best = _select_best(listwise_series)

    grpo_wandb = _maybe_fetch_wandb_run(args.wandb_entity, args.wandb_project, args.grpo_run_name)
    listwise_wandb = _maybe_fetch_wandb_run(
        args.wandb_entity, args.wandb_project, args.listwise_run_name
    )

    table_path = output_prefix.with_suffix(".tex")
    manifest_path = output_prefix.with_suffix(".json")
    _write_table(
        table_path,
        base_label=args.base_label,
        base_step=0,
        base_values=base_values,
        base_avg=base_avg,
        grpo_point=grpo_best,
        listwise_point=listwise_best,
    )
    _build_manifest(
        manifest_path,
        base_label=args.base_label,
        base_step=0,
        base_values=base_values,
        base_avg=base_avg,
        base_summary_path=str(base_summary_path),
        grpo_dir=grpo_dir,
        listwise_dir=listwise_dir,
        grpo_series=grpo_series,
        listwise_series=listwise_series,
        grpo_best=grpo_best,
        listwise_best=listwise_best,
        grpo_wandb=grpo_wandb,
        listwise_wandb=listwise_wandb,
    )
    print(table_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
