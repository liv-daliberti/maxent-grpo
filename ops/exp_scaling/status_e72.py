#!/usr/bin/env python3
"""Print E72 campaign progress: where each cohort is, and how far it moved.

Every figure is measured, not estimated from wall clock: training progress is
the sum of realized optimizer steps, and frontier progress counts completion
markers. Each run appends a snapshot to a history file, so the ``was`` column
is the previous invocation rather than a remembered number, and the rate is
derived from the gap between them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# --- cohort sizes, from the registered protocols -----------------------------
B3A_RUNS = 25
STEPS_PER_RUN = 4608
B3A_TOTAL_STEPS = B3A_RUNS * STEPS_PER_RUN

# 50 trained checkpoints plus 10 base-model references (one per domain and GPU
# model) = 60 measurable runs, times the temperatures each stage sweeps.
FRONTIER_RUNS = 60
FRONTIER_STAGES = {"a": FRONTIER_RUNS * 6, "b": FRONTIER_RUNS * 3, "c": FRONTIER_RUNS * 2}
FRONTIER_TOTAL_CELLS = sum(FRONTIER_STAGES.values())

# --- cost model, for the compute-share rows ----------------------------------
# Observed on this cohort: ~640 optimizer steps/hour/run, and ~3 minutes per
# eval-only cell including model load.
HOURS_PER_TRAINING_RUN = STEPS_PER_RUN / 640
HOURS_PER_CELL = 0.05
B3A_GPU_HOURS = B3A_RUNS * HOURS_PER_TRAINING_RUN
FRONTIER_GPU_HOURS = FRONTIER_TOTAL_CELLS * HOURS_PER_CELL
# Wave A of the baseline suite is five training arms of this size, plus Tier 0.
WAVE_A_ARMS = 5
WAVE_A_GPU_HOURS = WAVE_A_ARMS * B3A_GPU_HOURS + FRONTIER_GPU_HOURS

# Every training cohort in the campaign, newest last. A cohort is 25 runs
# unless it pairs two arms on a fresh seed set, which the confirmation does.
COHORTS: tuple[tuple[str, str, int], ...] = (
    ("B3a  replay gradient removed", "verified_first_replay_gradient_ablation_*_b3a_s*", 25),
    ("B1a  discovery credit removed", "verified_first_replay_only_ablation_*_b1a_s*", 25),
    # Both arms of the confirmation, matched pairwise on fresh seeds.
    ("B1a confirmation  seeds 48-52", "*conf_s*", 50),
)
HISTORY = "var/artifacts/e72_status_history.jsonl"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def last_global_step(metrics_path: Path) -> int:
    """Read the final logged optimizer step without parsing the whole file."""
    try:
        with metrics_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 200_000))
            tail = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return 0
    for line in reversed(tail):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if "misc/global_step" in record:
            return int(record["misc/global_step"])
    return 0


def cohort_progress(root: Path, pattern: str, expected_runs: int) -> dict[str, Any]:
    steps = 0
    progressing = 0
    complete = 0
    for run_dir in sorted(root.glob(f"var/data/xdr_qwen25_0p5b_instruct_{pattern}")):
        best = max(
            (
                last_global_step(Path(path))
                for path in glob.glob(str(run_dir / "debug_job*" / "train_metrics.jsonl"))
            ),
            default=0,
        )
        steps += best
        progressing += best > 0
        complete += (run_dir / "TRAINING_COMPLETE.json").is_file()
    return {
        "steps": steps,
        "total_steps": expected_runs * STEPS_PER_RUN,
        "runs_progressing": progressing,
        "runs_complete": complete,
        "runs_total": expected_runs,
    }


def frontier_progress(root: Path) -> dict[str, Any]:
    per_stage = {}
    for stage, expected in FRONTIER_STAGES.items():
        stage_root = root / "var" / "data" / "e72_frontier" / stage
        done = len(list(stage_root.glob("*/*/*/*/EVAL_ONLY_COMPLETE.json"))) if stage_root.is_dir() else 0
        per_stage[stage] = {"done": done, "expected": expected}
    return {
        "per_stage": per_stage,
        "done": sum(entry["done"] for entry in per_stage.values()),
        "total": FRONTIER_TOTAL_CELLS,
    }


def queue_counts(user: str) -> dict[str, int]:
    """Live scheduler state, keyed by the job-name prefix each launcher uses."""
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%j %T"],
            capture_output=True, text=True, check=False, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    counts: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        if name.startswith("e72b1aconf") or name.startswith("e72xgrpoconf"):
            key = "conf"
        elif name.startswith("e72b1a"):
            key = "b1a"
        elif name.startswith("e72b3a"):
            key = "b3a"
        elif name.startswith("e72f"):
            key = "cells"
        else:
            continue
        if state in ("RUNNING", "PENDING"):
            counts[f"{key}_{state.lower()}"] = counts.get(f"{key}_{state.lower()}", 0) + 1
    return counts


def pct(done: float, total: float) -> float:
    return 100.0 * done / total if total else 0.0


def render(current: dict[str, Any], previous: dict[str, Any] | None) -> str:
    frontier = current["frontier"]
    frontier_pct = pct(frontier["done"], frontier["total"])
    cohorts = current["cohorts"]

    previous_cohorts = (previous or {}).get("cohorts", {})

    def prev_pct(name: str) -> float | None:
        entry = previous_cohorts.get(name)
        if not entry:
            return None
        return pct(entry["steps"], entry["total_steps"])

    lines = [
        "",
        f"  {'cohort':<30} {'now':>6}   {'was':>6}   detail",
        f"  {'-' * 30} {'-' * 6}   {'-' * 6}   {'-' * 40}",
    ]

    done_gpu_hours = frontier_pct / 100 * FRONTIER_GPU_HOURS
    planned_gpu_hours = FRONTIER_GPU_HOURS

    for label, _, _ in COHORTS:
        entry = cohorts.get(label)
        if entry is None:
            continue
        now = pct(entry["steps"], entry["total_steps"])
        before = prev_pct(label)
        before_text = f"{before:5.1f}%" if before is not None else "    --"
        state = (
            "done"
            if entry["runs_complete"] == entry["runs_total"]
            else "not started"
            if entry["runs_progressing"] == 0
            else "running"
        )
        lines.append(
            f"  {label:<30} {now:5.1f}%   {before_text}   "
            f"{entry['runs_complete']}/{entry['runs_total']} terminal, "
            f"{entry['steps']:,} steps ({state})"
        )
        gpu = entry["runs_total"] * HOURS_PER_TRAINING_RUN
        done_gpu_hours += now / 100 * gpu
        planned_gpu_hours += gpu

    lines.append(
        f"  {'Frontier cells':<30} {frontier_pct:5.1f}%   "
        + (
            f"{pct(previous['frontier']['done'], previous['frontier']['total']):5.1f}%"
            if previous
            else "    --"
        )
        + "   "
        + "  ".join(
            f"{stage}:{entry['done']}/{entry['expected']}"
            for stage, entry in frontier["per_stage"].items()
        )
    )
    lines += [
        "",
        f"  launched work: {pct(done_gpu_hours, planned_gpu_hours):.1f}% "
        f"(~{done_gpu_hours:.0f} / {planned_gpu_hours:.0f} GPU-hours)",
        f"  Wave A + Tier 0: {pct(done_gpu_hours, WAVE_A_GPU_HOURS):.1f}% "
        f"(~{done_gpu_hours:.0f} / {WAVE_A_GPU_HOURS:.0f} GPU-hours, {WAVE_A_ARMS} arms)",
    ]

    queue = current.get("queue", {})
    if queue:
        lines.append(
            "  queue: "
            + "  ".join(
                f"{key.replace('_', ' ')} {value}" for key, value in sorted(queue.items())
            )
        )

    # A snapshot written before the multi-cohort schema has no per-cohort steps
    # to difference against; reporting a rate from it would invent one.
    if previous and previous_cohorts:
        hours = (current["unix"] - previous["unix"]) / 3600
        gained = sum(e["steps"] for e in cohorts.values()) - sum(
            e["steps"] for e in previous_cohorts.values()
        )
        remaining = sum(
            e["total_steps"] - e["steps"]
            for label, e in cohorts.items()
            if e["runs_progressing"] > 0 and e["runs_complete"] < e["runs_total"]
        )
        if hours > 0.01 and gained > 0:
            rate = gained / hours
            eta = f" -> ~{remaining / rate:.1f} h for cohorts in flight" if remaining else ""
            lines.append(
                f"  rate:  {rate:,.0f} steps/hour since last check "
                f"({hours:.2f} h ago){eta}"
            )
        elif hours > 0.01:
            lines.append(f"  rate:  no training movement in the last {hours:.2f} h")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=os.environ.get("USER", ""))
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="print without appending a snapshot (leaves the 'was' column alone)",
    )
    args = parser.parse_args()

    current = {
        "unix": time.time(),
        "cohorts": {
            label: cohort_progress(root, pattern, runs)
            for label, pattern, runs in COHORTS
        },
        "frontier": frontier_progress(root),
        "queue": queue_counts(args.user) if args.user else {},
    }

    history_path = root / HISTORY
    previous = None
    if history_path.is_file():
        for line in history_path.read_text().splitlines():
            try:
                previous = json.loads(line)
            except ValueError:
                continue

    print(render(current, previous))

    if not args.no_record:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(current, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
