#!/usr/bin/env python3
"""Report outcome-blind execution status for the preregistered E105 cohort.

This monitor deliberately reads training progress, completion receipts, saved
checkpoint names, and scheduler state only.  It never opens evaluation files,
so checking a running campaign cannot leak an outcome before the registered
paired analysis is complete.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import status_e78 as shared  # noqa: E402
import launch_e105_group_centered_semantic_repair_full_three_scale as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / launch.LEDGER
SCHEMA = "e105_group_centered_semantic_repair_full_three_scale_jobs_v1"
# The launcher is the frozen design authority. Duplicating these constants in
# a monitor already caused the Falcon seed set to drift once; aliases make a
# future launch-design change visible here automatically instead of rejecting
# a scientifically valid ledger after release.
ARM = launch.ARM
SCALE_SEEDS = launch.SCALE_SEEDS
DOMAINS = launch.DOMAINS
TARGET_STEPS = launch.TARGET_STEPS
TRAIN_ROWS = launch.TRAIN_ROWS
PASSES = launch.PASSES
CHECKPOINT_INTERVAL = launch.CHECKPOINT_INTERVAL
FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "REVOKED",
    "TIMEOUT",
}


def expected_cells() -> set[tuple[str, str, int]]:
    return {
        (scale, domain, seed)
        for scale, seeds in SCALE_SEEDS.items()
        for domain in DOMAINS
        for seed in seeds
    }


def validate_repaired_python_pairs(
    payload: dict[str, Any], runs: list[dict[str, Any]]
) -> None:
    if payload.get("python_comparator_repaired") is not True:
        raise RuntimeError("E105 ledger does not require repaired Python comparators")
    expected_path = (ROOT / launch.REPAIRED_PYTHON_COMPARATOR_LEDGER).resolve()
    recorded_path = Path(
        str(payload.get("repaired_python_comparator_ledger", ""))
    ).resolve()
    if recorded_path != expected_path or not recorded_path.is_file():
        raise RuntimeError("E105 ledger names an absent or unexpected E109 ledger")
    if payload.get("repaired_python_comparator_ledger_sha256") != (
        launch.e104.digest(recorded_path)
    ):
        raise RuntimeError("E105 repaired Python comparator ledger digest mismatch")
    repaired = json.loads(recorded_path.read_text(encoding="utf-8"))
    checks = {
        "schema": repaired.get("schema")
        == "e109_repaired_python_replay_comparators_jobs_v1",
        "released": repaired.get("released") is True,
        "pointmaze_excluded": repaired.get("pointmaze") == "excluded",
        "domain": repaired.get("domain") == "python_factors",
        "semantic_disabled": repaired.get("semantic_coefficient") == 0.0,
        "snapshot": repaired.get("snapshot_sha256")
        == payload.get("snapshot_sha256"),
        "parser_surface": repaired.get("parser_surface_version")
        == payload.get("python_response_surface_version"),
        "target_steps": repaired.get("target_steps") == TARGET_STEPS,
        "checkpoint_interval": repaired.get("checkpoint_interval_steps")
        == CHECKPOINT_INTERVAL,
        "qwen3_a6000_seeds": repaired.get("qwen3_a6000_seeds") == [73, 74],
        "qwen3_placement_path": Path(
            str(repaired.get("qwen3_paired_placement_artifact", ""))
        ).resolve()
        == Path(
            str(payload.get("qwen3_paired_placement_artifact", ""))
        ).resolve(),
        "qwen3_placement_digest": repaired.get(
            "qwen3_paired_placement_artifact_sha256"
        )
        == payload.get("qwen3_paired_placement_artifact_sha256"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"E109 comparator ledger contract failed: {failed}")
    repaired_runs = list(repaired.get("runs", []))
    expected = {
        (scale, "python_factors", seed)
        for scale, seeds in SCALE_SEEDS.items()
        for seed in seeds
    }
    repaired_index = {
        (str(run["scale"]), str(run["domain"]), int(run["seed"])): run
        for run in repaired_runs
        if run.get("arm") == "replay"
    }
    if (
        len(repaired_runs) != 15
        or set(repaired_index) != expected
        or any(run.get("arm") != "replay" for run in repaired_runs)
    ):
        raise RuntimeError("E109 does not contain the exact 15 Python replay cells")
    repaired_job_ids = {int(run["job_id"]) for run in repaired_runs}
    for run in runs:
        key = (str(run["scale"]), str(run["domain"]), int(run["seed"]))
        paired = run["paired_replay"]
        if key[1] == "python_factors":
            expected_pair = {
                field: repaired_index[key][field]
                for field in ("job_id", "run_stamp", "run_dir")
            }
            if paired != expected_pair:
                raise RuntimeError(f"E105 Python pair does not match E109: {key}")
        elif int(paired["job_id"]) in repaired_job_ids:
            raise RuntimeError(f"E105 non-Python pair incorrectly uses E109: {key}")


def load_and_validate_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"E105 is not released: its ledger does not exist at {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "schema": payload.get("schema") == SCHEMA,
        "released": payload.get("released") is True,
        "pointmaze_excluded": payload.get("pointmaze") == "excluded",
        "models": payload.get("models") == list(SCALE_SEEDS),
        "domains": payload.get("domains") == list(DOMAINS),
        "seeds": payload.get("seeds")
        == {scale: list(seeds) for scale, seeds in SCALE_SEEDS.items()},
        "arms": payload.get("arms") == [ARM],
        "target_steps": payload.get("target_steps") == TARGET_STEPS,
        "train_rows": payload.get("train_rows") == TRAIN_ROWS,
        "passes": payload.get("passes") == PASSES,
        "checkpoint_interval": payload.get("checkpoint_interval_steps")
        == CHECKPOINT_INTERVAL,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"E105 ledger contract failed: {failed}")

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise RuntimeError("E105 ledger runs are absent or malformed")
    try:
        cells = {
            (str(run["scale"]), str(run["domain"]), int(run["seed"]))
            for run in runs
        }
        job_ids = [int(run["job_id"]) for run in runs]
        run_stamps = [str(run["run_stamp"]) for run in runs]
        run_dirs = [str(run["run_dir"]) for run in runs]
        comparator_job_ids = [int(run["paired_replay"]["job_id"]) for run in runs]
        comparator_run_stamps = [
            str(run["paired_replay"]["run_stamp"]) for run in runs
        ]
        comparator_run_dirs = [
            str(run["paired_replay"]["run_dir"]) for run in runs
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("E105 run record is malformed") from exc
    if len(runs) != 75 or cells != expected_cells():
        raise RuntimeError("E105 ledger does not contain the exact 75-cell grid")
    if any(run.get("arm") != ARM for run in runs):
        raise RuntimeError("E105 ledger contains another treatment arm")
    for label, values in (
        ("job ids", job_ids),
        ("run stamps", run_stamps),
        ("run directories", run_dirs),
        ("paired comparator job ids", comparator_job_ids),
        ("paired comparator run stamps", comparator_run_stamps),
        ("paired comparator run directories", comparator_run_dirs),
    ):
        if len(values) != len(set(values)):
            raise RuntimeError(f"E105 ledger contains duplicate {label}")
    if set(job_ids) & set(comparator_job_ids):
        raise RuntimeError("E105 treatment and comparator job ids overlap")
    validate_repaired_python_pairs(payload, runs)
    return payload


def progress_row(
    run: dict[str, Any],
    *,
    role: str,
    scheduler_states: dict[int, str],
) -> dict[str, Any]:
    job_id = int(run["job_id"])
    run_dir = Path(str(run["run_dir"]))
    metrics_step = min(shared.run_step(run_dir), TARGET_STEPS)
    receipt_step = min(shared.receipt_step(run_dir), TARGET_STEPS)
    step = max(metrics_step, receipt_step)
    checkpoint = min(shared.checkpoint_step(run_dir), TARGET_STEPS)
    state = shared.normalize_state(scheduler_states.get(job_id, "NOT_IN_QUEUE"))
    complete = step >= TARGET_STEPS
    issues: list[str] = []
    if state in FAILURE_STATES and not complete:
        issues.append(f"scheduler_{state.lower()}")
    if state == "COMPLETED" and not complete:
        issues.append("scheduler_complete_without_terminal_training_receipt")
    if checkpoint > step and not complete:
        issues.append("checkpoint_ahead_of_observed_training_step")
    return {
        "role": role,
        "scale": str(run["scale"]),
        "domain": str(run["domain"]),
        "seed": int(run["seed"]),
        "job_id": job_id,
        "run_stamp": str(run["run_stamp"]),
        "run_dir": str(run_dir),
        "state": "COMPLETED" if complete else state,
        "step": step,
        "checkpoint": checkpoint,
        "terminal_receipt_step": receipt_step,
        "complete": complete,
        "issues": issues,
    }


def snapshot(
    ledger: dict[str, Any],
    *,
    scheduler_states: dict[int, str] | None = None,
) -> dict[str, Any]:
    runs = ledger["runs"]
    comparator_runs = [
        {
            "scale": run["scale"],
            "domain": run["domain"],
            "seed": run["seed"],
            **run["paired_replay"],
        }
        for run in runs
    ]
    if scheduler_states is None:
        scheduler_states = shared.scheduler_states(
            [int(run["job_id"]) for run in runs + comparator_runs]
        )

    rows = [
        progress_row(run, role="treatment", scheduler_states=scheduler_states)
        for run in runs
    ]
    comparator_rows = [
        progress_row(run, role="paired_replay", scheduler_states=scheduler_states)
        for run in comparator_runs
    ]

    counts = Counter(str(row["state"]) for row in rows)
    comparator_counts = Counter(str(row["state"]) for row in comparator_rows)
    completed = sum(bool(row["complete"]) for row in rows)
    comparator_completed = sum(bool(row["complete"]) for row in comparator_rows)
    issue_count = sum(bool(row["issues"]) for row in rows)
    comparator_issue_count = sum(bool(row["issues"]) for row in comparator_rows)
    return {
        "schema": "e105_outcome_blind_status_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pointmaze": "excluded",
        "outcomes_read": False,
        "target_steps": TARGET_STEPS,
        "cells": len(rows),
        "completed": completed,
        "realized_steps": sum(int(row["step"]) for row in rows),
        "total_steps": len(rows) * TARGET_STEPS,
        "state_counts": dict(sorted(counts.items())),
        "issue_cells": issue_count,
        "paired_comparator_cells": len(comparator_rows),
        "paired_comparator_completed": comparator_completed,
        "paired_comparator_realized_steps": sum(
            int(row["step"]) for row in comparator_rows
        ),
        "paired_comparator_total_steps": len(comparator_rows) * TARGET_STEPS,
        "paired_comparator_state_counts": dict(sorted(comparator_counts.items())),
        "paired_comparator_issue_cells": comparator_issue_count,
        "ready_for_registered_analysis": (
            completed == 75
            and comparator_completed == 75
            and issue_count == 0
            and comparator_issue_count == 0
        ),
        "rows": rows,
        "paired_comparator_rows": comparator_rows,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "E105 outcome-blind status",
        (
            f"cells={report['cells']} complete={report['completed']} "
            f"steps={report['realized_steps']:,}/{report['total_steps']:,} "
            f"issues={report['issue_cells']}"
        ),
        f"states={json.dumps(report['state_counts'], sort_keys=True)}",
        (
            f"paired_comparators={report['paired_comparator_cells']} "
            f"complete={report['paired_comparator_completed']} "
            f"steps={report['paired_comparator_realized_steps']:,}/"
            f"{report['paired_comparator_total_steps']:,} "
            f"issues={report['paired_comparator_issue_cells']}"
        ),
        (
            "paired_comparator_states="
            f"{json.dumps(report['paired_comparator_state_counts'], sort_keys=True)}"
        ),
        (
            "ready_for_registered_analysis="
            f"{str(report['ready_for_registered_analysis']).lower()}"
        ),
    ]
    problem_rows = [
        row
        for row in report["rows"] + report["paired_comparator_rows"]
        if row["issues"]
    ]
    for row in problem_rows:
        lines.append(
            f"ISSUE {row['role']} {row['scale']} {row['domain']} s{row['seed']} "
            f"job={row['job_id']} state={row['state']} "
            f"step={row['step']} {','.join(row['issues'])}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "return nonzero unless all 75 treatment and 75 paired comparator "
            "cells are cleanly complete"
        ),
    )
    args = parser.parse_args()
    try:
        report = snapshot(load_and_validate_ledger(args.ledger.resolve()))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report))
    if args.require_complete and not report["ready_for_registered_analysis"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
