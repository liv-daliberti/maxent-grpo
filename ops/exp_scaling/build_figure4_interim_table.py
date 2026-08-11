#!/usr/bin/env python3
"""Freeze four-metric arm means for the dated interim Figure 4 snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIGURE_SNAPSHOT = ROOT / "paper/figures/figure4_interim_20260806.json"
OUTPUT = ROOT / "paper/results/figure4_interim_20260806_table.json"
EXPECTED_DRAWS = 4

FAMILY_SOURCES = {
    "Qwen2.5-0.5B": {
        "static": ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
        "point": ROOT
        / "var/artifacts/e78pm_point_maze_verified_replay_only_05b_jobs.json",
    },
    "Falcon3-1B": {
        "static": ROOT
        / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json",
        "point": ROOT
        / "var/artifacts/e79pm_falcon_point_maze_verified_replay_jobs.json",
    },
    "Qwen2.5-3B": {
        "static": ROOT
        / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
        "point": None,
    },
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite: {value!r}")
    return float(value)


def _latest_summary(domain_record: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Deepest paired checkpoint that reflects training, never pass 0.

    At pass 0 both arms are the same untrained model, so every metric matches
    exactly. Reporting that as a panel would print a row of exact ties and read
    as "the arms are indistinguishable here" when in fact neither arm has taken
    a step. A panel enters the table only once a paired checkpoint exists past
    initialization.
    """

    summaries = {
        key: value
        for key, value in domain_record.get("paired_summary_by_pass", {}).items()
        if float(key) > 0
    }
    if not summaries:
        return None
    return max(summaries.items(), key=lambda item: float(item[0]))


def _static_sampled(run_dir: Path, *, step: int) -> dict[str, float]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if (
                    row.get("evaluation_kind")
                    != "fixed_seed_sampled_k_neutral"
                    or row.get("step") != step
                ):
                    continue
                draw = row.get("draw_index")
                metrics = row.get("metrics")
                if isinstance(draw, int) and isinstance(metrics, dict):
                    records[draw] = metrics
    if sorted(records) != list(range(EXPECTED_DRAWS)):
        raise RuntimeError(
            f"{run_dir}: step {step} has draws {sorted(records)}, "
            f"expected 0--{EXPECTED_DRAWS - 1}"
        )
    fields = {
        "pass8": "any_correct_at_k",
        "mean8": "mean_at_k",
        "distinct8": "distinct_correct_modes_at_k",
    }
    return {
        output: sum(
            _finite(records[draw].get(source), name=f"{run_dir}:{step}:{source}")
            for draw in range(EXPECTED_DRAWS)
        )
        / EXPECTED_DRAWS
        for output, source in fields.items()
    }


def _static_pass1(run_dir: Path, *, step: int) -> float:
    candidates = sorted(
        run_dir.glob(f"debug_job*/eval_results/{step}_multi_answer.json")
    )
    if not candidates:
        raise RuntimeError(f"{run_dir}: missing greedy evaluation at step {step}")
    rows = _json(candidates[-1])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{candidates[-1]}: greedy evaluation is empty")
    scores = []
    for index, row in enumerate(rows):
        row_scores = row.get("scores") if isinstance(row, dict) else None
        if not isinstance(row_scores, list) or len(row_scores) != 1:
            raise RuntimeError(
                f"{candidates[-1]} row {index}: expected one greedy score"
            )
        scores.append(_finite(row_scores[0], name=f"greedy score row {index}"))
    return sum(scores) / len(scores)


def _point_metrics(path: Path, *, step: int) -> dict[str, float]:
    selected = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (
                row.get("schema") == "point-maze-waypoint-pilot-evaluation-v1"
                and row.get("learning_round") == step
            ):
                selected = row
    if selected is None:
        raise RuntimeError(f"{path}: missing PointMaze evaluation at step {step}")
    return {
        "pass1": None,
        "pass8": _finite(selected.get("pass8"), name=f"{path}:pass8"),
        "mean8": _finite(selected.get("mean8"), name=f"{path}:mean8"),
        "distinct8": _finite(selected.get("distinct8"), name=f"{path}:distinct8"),
    }


def _run_index(ledger: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (str(run.get("domain", "point_maze")), str(run["arm"]), int(run["seed"])): run
        for run in ledger["runs"]
    }


def _mean(values: list[float | None], *, metric: str) -> float | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise RuntimeError(f"metric {metric} mixes missing and observed values")
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric)


def main() -> None:
    snapshot = _json(FIGURE_SNAPSHOT)
    output: dict[str, Any] = {
        "schema": "figure4_interim_four_metric_table_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figure_snapshot": str(FIGURE_SNAPSHOT),
        "figure_snapshot_generated_at": snapshot["generated_at"],
        "families": {},
    }
    for family, family_snapshot in snapshot["families"].items():
        sources = FAMILY_SOURCES[family]
        static_ledger = _json(sources["static"])
        static_index = _run_index(static_ledger)
        point_ledger = _json(sources["point"]) if sources["point"] else None
        point_index = _run_index(point_ledger) if point_ledger else {}
        family_out: dict[str, Any] = {"domains": {}}
        for domain, domain_record in family_snapshot["domains"].items():
            latest = _latest_summary(domain_record)
            if latest is None:
                continue
            pass_key, figure_summary = latest
            paired_seeds = [int(seed) for seed in figure_summary["paired_seeds"]]
            source_ledger = point_ledger if domain == "point_maze" else static_ledger
            if source_ledger is None:
                raise RuntimeError(f"{family}/{domain}: missing source ledger")
            step = round(float(pass_key) * int(source_ledger["train_rows"]))
            domain_out: dict[str, Any] = {
                "pass": float(pass_key),
                "step": step,
                "paired_seeds": paired_seeds,
                "arms": {},
            }
            for arm in ("control", "replay"):
                per_seed: dict[str, Any] = {}
                for seed in paired_seeds:
                    index = point_index if domain == "point_maze" else static_index
                    run = index[(domain, arm, seed)]
                    if domain == "point_maze":
                        metrics = _point_metrics(Path(run["metrics_path"]), step=step)
                    else:
                        run_dir = Path(run["run_dir"])
                        metrics = _static_sampled(run_dir, step=step)
                        metrics["pass1"] = _static_pass1(run_dir, step=step)
                    per_seed[str(seed)] = metrics
                means = {
                    metric: _mean(
                        [per_seed[str(seed)][metric] for seed in paired_seeds],
                        metric=metric,
                    )
                    for metric in ("pass1", "pass8", "mean8", "distinct8")
                }
                domain_out["arms"][arm] = {"means": means, "per_seed": per_seed}
            for arm in ("control", "replay"):
                observed = domain_out["arms"][arm]["means"]["distinct8"]
                expected = float(figure_summary[f"{arm}_mean"])
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(
                        f"{family}/{domain}/{arm}: distinct8 {observed} "
                        f"does not reproduce Figure 4 {expected}"
                    )
            family_out["domains"][domain] = domain_out
        output["families"][family] = family_out
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
