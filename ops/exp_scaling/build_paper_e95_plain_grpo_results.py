#!/usr/bin/env python3
"""Freeze every available Falcon plain-GRPO trajectory checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "var/artifacts/e95_plain_grpo_Falcon3-1B_jobs.json"
DEFAULT_OUTPUT = ROOT / "paper/results/e95_falcon_plain_grpo_reportable.json"
MODEL = "Falcon3-1B"
SCALE = "falcon1b"
DOMAINS = (
    "graph_coloring", "countdown", "python_factors", "mathir",
    "pantry_plan",
)
SEEDS = (55, 56, 57, 58, 59)
EXPECTED_DRAWS = 4
EVALUATION_KIND = "fixed_seed_sampled_k_neutral"
METRICS = {
    "pass8": "any_correct_at_k",
    "distinct8": "distinct_correct_modes_at_k",
    "mean8": "mean_at_k",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_prefix(path: Path) -> tuple[bytes, dict[str, Any]]:
    byte_length = path.stat().st_size
    with path.open("rb") as handle:
        data = handle.read(byte_length)
    if len(data) != byte_length:
        raise RuntimeError(f"short read while freezing {path}")
    return data, {
        "path": str(path.relative_to(ROOT)),
        "byte_length": byte_length,
        "sha256": sha256_bytes(data),
    }


def run_trajectory(
    run: dict[str, Any],
    *,
    interval: int,
    target: int,
    train_rows: int,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Return every complete sampled checkpoint from one registered run."""

    records: dict[tuple[int, int], dict[str, float]] = {}
    prefix_sources: list[dict[str, Any]] = []
    run_dir = Path(str(run["run_dir"]))
    paths = sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl"))
    for path in paths:
        data, source = frozen_prefix(path)
        source.update({
            "domain": str(run["domain"]),
            "seed": int(run["seed"]),
            "job_id": int(run["job_id"]),
        })
        prefix_sources.append(source)
        for raw_line in data.splitlines():
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if row.get("evaluation_kind") != EVALUATION_KIND:
                continue
            step, draw = row.get("step"), row.get("draw_index")
            metrics = row.get("metrics")
            if (
                not isinstance(step, int)
                or draw not in range(EXPECTED_DRAWS)
                or step < 0
                or step > target
                or step % interval
                or not isinstance(metrics, dict)
            ):
                continue
            values = {
                key: float(metrics[field])
                for key, field in METRICS.items()
                if isinstance(metrics.get(field), (int, float))
                and math.isfinite(float(metrics[field]))
            }
            if len(values) == len(METRICS):
                records[(step, int(draw))] = values

    trajectory: dict[str, dict[str, float]] = {}
    for step in range(0, target + 1, interval):
        if any(
            (step, draw) not in records for draw in range(EXPECTED_DRAWS)
        ):
            continue
        trajectory[str(step / train_rows)] = {
            metric: sum(
                records[(step, draw)][metric]
                for draw in range(EXPECTED_DRAWS)
            ) / EXPECTED_DRAWS
            for metric in METRICS
        }
    return trajectory, prefix_sources

def build(ledger_path: Path) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema") != "e95_plain_grpo_control_jobs_v2":
        raise RuntimeError("unexpected E95 plain-GRPO ledger schema")
    if ledger.get("family") != MODEL or tuple(ledger.get("seeds", ())) != SEEDS:
        raise RuntimeError("E95 Falcon model or seed contract drifted")
    interval = int(ledger["checkpoint_interval_steps"])
    target = int(ledger["target_steps"])
    train_rows = int(ledger["train_rows"])
    runs = {
        (str(run["domain"]), int(run["seed"])): run
        for run in ledger["runs"]
    }
    domains: dict[str, Any] = {}
    prefix_sources: list[dict[str, Any]] = []
    for domain in DOMAINS:
        trajectories: dict[str, Any] = {}
        for seed in SEEDS:
            run = runs.get((domain, seed))
            if run is None:
                continue
            trajectory, run_sources = run_trajectory(
                run, interval=interval, target=target, train_rows=train_rows
            )
            if trajectory:
                trajectories[str(seed)] = trajectory
            prefix_sources.extend(run_sources)
        all_passes = sorted({
            float(pass_key)
            for trajectory in trajectories.values()
            for pass_key in trajectory
        })
        summary_by_pass: dict[str, Any] = {}
        for training_pass in all_passes:
            pass_key = str(training_pass)
            available_seeds = [
                seed for seed in SEEDS
                if pass_key in trajectories.get(str(seed), {})
            ]
            if not available_seeds:
                continue
            summary_by_pass[pass_key] = {
                "seeds": available_seeds,
                "n": len(available_seeds),
                "metrics": {
                    metric: {
                        "mean": sum(
                            trajectories[str(seed)][pass_key][metric]
                            for seed in available_seeds
                        ) / len(available_seeds),
                        "range": [
                            min(trajectories[str(seed)][pass_key][metric] for seed in available_seeds),
                            max(trajectories[str(seed)][pass_key][metric] for seed in available_seeds),
                        ],
                        "per_seed": {
                            str(seed): trajectories[str(seed)][pass_key][metric]
                            for seed in available_seeds
                        },
                    }
                    for metric in METRICS
                },
            }
        if summary_by_pass:
            domains[domain] = {
                "seeds_with_any_checkpoint": sorted(map(int, trajectories)),
                "summary_by_pass": summary_by_pass,
            }
    return {
        "schema": "paper-e95-falcon-plain-grpo-available-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "all available sampled checkpoints; no minimum seed count",
        "model": MODEL,
        "scale": SCALE,
        "domains": domains,
        "registered_seeds": list(SEEDS),
        "checkpoint_interval_steps": interval,
        "target_steps": target,
        "train_rows": train_rows,
        "draws_per_checkpoint": EXPECTED_DRAWS,
        "evaluation_kind": EVALUATION_KIND,
        "metrics": METRICS,
        "ledger": {
            "path": str(ledger_path.relative_to(ROOT)),
            "sha256": sha256_path(ledger_path),
        },
        "prefix_sources": prefix_sources,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.ledger.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
