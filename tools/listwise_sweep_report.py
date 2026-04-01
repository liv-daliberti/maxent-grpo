from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class SweepEntry:
    tau: float
    beta: float
    seed: int | None
    job_id: str
    job_name: str
    run_name: str
    output_dir: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank listwise tau/beta sweep runs from SEED paper-eval summaries."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_repo_root() / "var" / "artifacts" / "seed_paper_eval" / "live",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    return parser


def load_manifest(path: Path) -> list[SweepEntry]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for row in reader:
            rows.append(
                SweepEntry(
                    tau=float(row["tau"]),
                    beta=float(row["beta"]),
                    seed=_maybe_int(row.get("seed")),
                    job_id=row["job_id"],
                    job_name=row["job_name"],
                    run_name=row["run_name"],
                    output_dir=row["output_dir"],
                )
            )
    return rows


def latest_summary_path(results_root: Path, run_name: str) -> Path | None:
    run_dir = results_root / run_name
    if not run_dir.exists():
        return None
    summaries = sorted(run_dir.glob("step-*/seed_paper_eval_*.summary.json"))
    if not summaries:
        return None
    return max(
        summaries,
        key=lambda path: (
            int(path.parent.name.split("-")[-1]),
            path.name,
        ),
    )


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def latest_training_step(output_dir: str) -> int | None:
    run_dir = Path(output_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    steps: list[int] = []
    trainer_state_path = run_dir / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            trainer_state = None
        if isinstance(trainer_state, dict):
            step = _maybe_int(trainer_state.get("global_step"))
            if step is not None:
                steps.append(step)
    for checkpoint_dir in run_dir.glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        step = _maybe_int(checkpoint_dir.name.split("-")[-1])
        if step is not None:
            steps.append(step)
    return max(steps) if steps else None


def _mean_mapping(values: Any) -> float | None:
    if not isinstance(values, dict) or not values:
        return None
    parsed: list[float] = []
    for value in values.values():
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    return mean(parsed) if parsed else None


def build_row(entry: SweepEntry, results_root: Path) -> dict[str, Any]:
    summary_path = latest_summary_path(results_root, entry.run_name)
    train_step = latest_training_step(entry.output_dir)
    row: dict[str, Any] = {
        "tau": entry.tau,
        "beta": entry.beta,
        "seed": entry.seed,
        "job_id": entry.job_id,
        "job_name": entry.job_name,
        "run_name": entry.run_name,
        "output_dir": entry.output_dir,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "step": None,
        "train_step": train_step,
        "avg": None,
        "pass_at_1_avg": None,
        "pass_at_8_avg": None,
        "mean_at_8_avg": None,
        "avg_len_mean": None,
        "status": "pending" if summary_path is None else "ok",
    }
    if summary_path is None:
        return row
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row["step"] = int(summary_path.parent.name.split("-")[-1])
    row["avg"] = summary.get("avg")
    row["pass_at_1_avg"] = summary.get("avg")
    row["pass_at_8_avg"] = summary.get("pass_at_8_avg")
    row["mean_at_8_avg"] = summary.get("mean_at_8_avg")
    row["avg_len_mean"] = _mean_mapping(summary.get("avg_lens"))
    return row


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return mean(values) if values else None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["tau"]), float(row["beta"]))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (_, _), group in sorted(grouped.items()):
        if len(group) == 1:
            single = dict(group[0])
            single["seed_count"] = 1
            single["seed_completed_count"] = 1 if single.get("avg") is not None else 0
            seed = single.get("seed")
            single["seeds"] = [] if seed is None else [seed]
            aggregated.append(single)
            continue

        completed = [row for row in group if row.get("avg") is not None]
        seeds = sorted(
            seed
            for seed in (_maybe_int(row.get("seed")) for row in group)
            if seed is not None
        )
        job_ids = [str(row["job_id"]) for row in group]
        job_names = [str(row["job_name"]) for row in group]
        run_names = [str(row["run_name"]) for row in group]
        output_dirs = [str(row["output_dir"]) for row in group]
        summary_paths = [
            str(row["summary_path"])
            for row in group
            if row.get("summary_path") is not None
        ]
        step_values = [
            step for step in (_maybe_int(row.get("step")) for row in completed) if step is not None
        ]
        train_step_values = [
            step for step in (_maybe_int(row.get("train_step")) for row in group) if step is not None
        ]
        running_rows = sum(
            1
            for row in group
            if str(row.get("status_bucket", "")).startswith("running")
        )
        if len(completed) == len(group):
            status = "ok"
            status_bucket = "done"
        elif running_rows > 0:
            status = "partial"
            status_bucket = "running_with_summary" if completed else "running_no_summary"
        elif completed:
            status = "partial"
            status_bucket = "running_with_summary"
        else:
            status = "pending"
            status_bucket = "pending"
        aggregated.append(
            {
                "tau": float(group[0]["tau"]),
                "beta": float(group[0]["beta"]),
                "seed": None,
                "seeds": seeds,
                "seed_count": len(group),
                "seed_completed_count": len(completed),
                "job_id": ",".join(job_ids),
                "job_ids": job_ids,
                "job_name": ",".join(job_names),
                "job_names": job_names,
                "run_name": f"{len(group)} seeds @ tau={float(group[0]['tau']):.2f},beta={float(group[0]['beta']):.2f}",
                "run_names": run_names,
                "output_dir": output_dirs[0],
                "output_dirs": output_dirs,
                "summary_path": summary_paths[0] if summary_paths else None,
                "summary_paths": summary_paths,
                "step": max(step_values) if step_values else None,
                "train_step": max(train_step_values) if train_step_values else None,
                "avg": _mean_metric(completed, "avg"),
                "pass_at_1_avg": _mean_metric(completed, "pass_at_1_avg"),
                "pass_at_8_avg": _mean_metric(completed, "pass_at_8_avg"),
                "mean_at_8_avg": _mean_metric(completed, "mean_at_8_avg"),
                "avg_len_mean": _mean_metric(completed, "avg_len_mean"),
                "status": status,
                "status_bucket": status_bucket,
                "status_label": f"{len(completed)}/{len(group)} done",
                "scheduler_state": (
                    "RUNNING"
                    if any(str(row.get("scheduler_state")) == "RUNNING" for row in group)
                    else None
                ),
            }
        )
    return aggregated


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    avg = (
        float(row["pass_at_1_avg"])
        if row.get("pass_at_1_avg") is not None
        else (float(row["avg"]) if row.get("avg") is not None else float("-inf"))
    )
    pass_at_8_avg = (
        float(row["pass_at_8_avg"])
        if row.get("pass_at_8_avg") is not None
        else float("-inf")
    )
    mean_at_8_avg = (
        float(row["mean_at_8_avg"])
        if row.get("mean_at_8_avg") is not None
        else float("-inf")
    )
    avg_len_mean = (
        -float(row["avg_len_mean"])
        if row.get("avg_len_mean") is not None
        else float("-inf")
    )
    return (avg, pass_at_8_avg, mean_at_8_avg, avg_len_mean)


def format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "rank",
        "tau",
        "beta",
        "seeds",
        "step",
        "pass_at_1_avg",
        "pass_at_8_avg",
        "mean_at_8_avg",
        "avg_len_mean",
        "job_id",
        "run_name",
        "status",
    ]
    lines = ["\t".join(headers)]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "\t".join(
                [
                    str(idx),
                    f"{row['tau']:.4f}",
                    f"{row['beta']:.4f}",
                    (
                        f"{int(row.get('seed_completed_count', 1))}/{int(row.get('seed_count', 1))}"
                    ),
                    "" if row["step"] is None else str(row["step"]),
                    ""
                    if row.get("pass_at_1_avg") is None
                    else f"{float(row['pass_at_1_avg']):.4f}",
                    ""
                    if row["pass_at_8_avg"] is None
                    else f"{float(row['pass_at_8_avg']):.4f}",
                    ""
                    if row["mean_at_8_avg"] is None
                    else f"{float(row['mean_at_8_avg']):.4f}",
                    ""
                    if row["avg_len_mean"] is None
                    else f"{float(row['avg_len_mean']):.1f}",
                    str(row["job_id"]),
                    str(row["run_name"]),
                    str(row["status"]),
                ]
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = load_manifest(args.manifest)
    rows = aggregate_rows([build_row(entry, args.results_root) for entry in entries])
    rows.sort(key=rank_key, reverse=True)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(format_table(rows))
        completed_rows = [row for row in rows if row.get("avg") is not None]
        if completed_rows:
            best = completed_rows[0]
            print(
                "\nBest:",
                f"tau={best['tau']:.4f}",
                f"beta={best['beta']:.4f}",
                f"seeds={best.get('seed_completed_count', 1)}/{best.get('seed_count', 1)}",
                f"pass_at_1_avg={best.get('pass_at_1_avg', best['avg'])}",
                f"pass_at_8_avg={best['pass_at_8_avg']}",
                f"mean_at_8_avg={best['mean_at_8_avg']}",
                f"run_name={best['run_name']}",
            )
        elif rows:
            print("\nBest: no completed eval summaries yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
