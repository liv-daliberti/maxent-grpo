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

FRONTIER_STAGES = {"a": 360, "b": 150, "c": 120}
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

B3A_GLOB = "var/data/xdr_qwen25_0p5b_instruct_verified_first_replay_gradient_ablation_*"
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


def b3a_progress(root: Path) -> dict[str, Any]:
    steps = 0
    progressing = 0
    complete = 0
    for run_dir in sorted(root.glob(B3A_GLOB.replace("var/data/", "var/data/"))):
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
        "total_steps": B3A_TOTAL_STEPS,
        "runs_progressing": progressing,
        "runs_complete": complete,
        "runs_total": B3A_RUNS,
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
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%j %T"],
            capture_output=True, text=True, check=False, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    counts = {"b3a_running": 0, "b3a_pending": 0, "cells_running": 0, "cells_pending": 0}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        key = "b3a" if name.startswith("e72b3a") else "cells" if name.startswith("e72f") else None
        if key is None:
            continue
        if state == "RUNNING":
            counts[f"{key}_running"] += 1
        elif state == "PENDING":
            counts[f"{key}_pending"] += 1
    return counts


def pct(done: float, total: float) -> float:
    return 100.0 * done / total if total else 0.0


def render(current: dict[str, Any], previous: dict[str, Any] | None) -> str:
    b3a, frontier = current["b3a"], current["frontier"]
    b3a_pct = pct(b3a["steps"], b3a["total_steps"])
    frontier_pct = pct(frontier["done"], frontier["total"])

    done_gpu_hours = (
        b3a_pct / 100 * B3A_GPU_HOURS + frontier_pct / 100 * FRONTIER_GPU_HOURS
    )
    session_pct = pct(done_gpu_hours, B3A_GPU_HOURS + FRONTIER_GPU_HOURS)
    suite_pct = pct(done_gpu_hours, WAVE_A_GPU_HOURS)

    def was(path: tuple[str, ...], value: float) -> str:
        if not previous:
            return "--"
        node: Any = previous
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if not isinstance(node, (int, float)):
            return "--"
        return f"{node:.1f}%" if abs(value - node) < 100 else "--"

    prev_pcts = {}
    if previous:
        prev_b3a = pct(previous["b3a"]["steps"], previous["b3a"]["total_steps"])
        prev_front = pct(previous["frontier"]["done"], previous["frontier"]["total"])
        prev_gpu = (
            prev_b3a / 100 * B3A_GPU_HOURS + prev_front / 100 * FRONTIER_GPU_HOURS
        )
        prev_pcts = {
            "b3a": prev_b3a,
            "frontier": prev_front,
            "session": pct(prev_gpu, B3A_GPU_HOURS + FRONTIER_GPU_HOURS),
            "suite": pct(prev_gpu, WAVE_A_GPU_HOURS),
        }

    def row(label: str, now: float, key: str, detail: str) -> str:
        before = f"{prev_pcts[key]:5.1f}%" if key in prev_pcts else "    --"
        return f"  {label:<26} {now:5.1f}%   {before}   {detail}"

    stages = frontier["per_stage"]
    lines = [
        "",
        f"  {'scope':<26} {'now':>6}   {'was':>6}   detail",
        f"  {'-' * 26} {'-' * 6}   {'-' * 6}   {'-' * 44}",
        row(
            "B3a cohort",
            b3a_pct,
            "b3a",
            f"{b3a['steps']:,} / {b3a['total_steps']:,} steps, "
            f"{b3a['runs_complete']}/{b3a['runs_total']} runs terminal",
        ),
        row(
            "Frontier cells",
            frontier_pct,
            "frontier",
            "  ".join(
                f"{stage}:{entry['done']}/{entry['expected']}"
                for stage, entry in stages.items()
            ),
        ),
        row(
            "This session's compute",
            session_pct,
            "session",
            f"~{done_gpu_hours:.0f} / {B3A_GPU_HOURS + FRONTIER_GPU_HOURS:.0f} GPU-hours",
        ),
        row(
            "Wave A + Tier 0",
            suite_pct,
            "suite",
            f"~{done_gpu_hours:.0f} / {WAVE_A_GPU_HOURS:.0f} GPU-hours "
            f"({WAVE_A_ARMS} arms)",
        ),
    ]

    queue = current.get("queue", {})
    if queue:
        lines += [
            "",
            f"  queue: B3a {queue.get('b3a_running', 0)} running / "
            f"{queue.get('b3a_pending', 0)} pending  |  "
            f"cells {queue.get('cells_running', 0)} running / "
            f"{queue.get('cells_pending', 0)} pending",
        ]

    if previous:
        hours = (current["unix"] - previous["unix"]) / 3600
        gained = b3a["steps"] - previous["b3a"]["steps"]
        if hours > 0.01 and gained > 0:
            rate = gained / hours
            remaining = b3a["total_steps"] - b3a["steps"]
            lines.append(
                f"  rate:  {rate:,.0f} B3a steps/hour since last check "
                f"({hours:.2f} h ago) -> ~{remaining / rate:.1f} h at this rate"
            )
        elif hours > 0.01:
            lines.append(f"  rate:  no B3a movement in the last {hours:.2f} h")
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
        "b3a": b3a_progress(root),
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
