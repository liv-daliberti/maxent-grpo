#!/usr/bin/env python3
"""Report read-only progress for the E78 verified-replay-only campaign.

Progress comes from realized optimizer steps in each run's metrics log.  The
scheduler query explains which cells are running or waiting; it is not used to
estimate training progress.  Saved-checkpoint counts use the registered
half-pass grid (192 optimizer steps).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
POINT_LEDGER = (
    ROOT / "var/artifacts/e78pm_point_maze_verified_replay_only_05b_jobs.json"
)
# E81 adds one arm to this design and inherits both E78 arms unchanged, so its
# progress is reported here beside the comparators it depends on.
E81_LEDGER = ROOT / "var/artifacts/e81_semantic_maxent_verified_replay_05b_jobs.json"
DOMAIN_LABELS = {
    "graph_coloring": "Graph",
    "countdown": "Countdown",
    "python_factors": "Python",
    "mathir": "MathIR",
    "pantry_plan": "Pantry",
}
STATE_ORDER = (
    "RUNNING",
    "PENDING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NOT_IN_QUEUE",
    "UNKNOWN",
)


def max_global_step(metrics_path: Path) -> int:
    """Return the deepest optimizer step visible in a bounded log tail."""

    try:
        with metrics_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 1_000_000))
            lines = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return 0

    best = 0
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        step = record.get("misc/global_step", record.get("trainer/global_step"))
        if step is not None:
            best = max(best, int(step))
    return best


def run_step(run_dir: Path) -> int:
    return max(
        (
            max_global_step(path)
            for path in run_dir.glob("debug_job*/train_metrics.jsonl")
        ),
        default=0,
    )

VERL_CONSOLE_STEP = re.compile(rb"(?:^|\n)step:(\d+)\s+-")


def verl_console_step(log_path: Path) -> int:
    """Return the deepest upstream verl LocalLogger step in a bounded tail."""

    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 2_000_000))
            payload = handle.read()
    except OSError:
        return 0
    return max(
        (int(match.group(1)) for match in VERL_CONSOLE_STEP.finditer(payload)),
        default=0,
    )


# Each PointMaze cohort stamps its own experiment id into the schema, exactly as
# the rolling-checkpoint schema does; matching one literal reported every other
# cohort as having taken no optimizer steps at all.
POINT_TRAINING_SCHEMA = re.compile(r"^[a-z0-9]+-point-maze-training-v1$")


def point_run_step(metrics_path: Path) -> int:
    """Return the deepest E78-PM optimizer step in its append-only metrics."""

    try:
        with metrics_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 2_000_000))
            lines = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return 0
    best = 0
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if POINT_TRAINING_SCHEMA.match(str(row.get("schema", ""))):
            best = max(best, int(row.get("optimizer_step", 0)))
    return best


POINT_CHECKPOINT_SCHEMA = re.compile(
    r"^e\d+pm-point-maze-rolling-checkpoint-v1$"
)

# Every PointMaze-family domain stamps its own evaluation schema
# (point-maze-waypoint-pilot-..., point-maze-tour-...). Four separate readers
# hardcoded one literal and silently read zero rows from every other domain:
# the panel drew empty and the table raised "missing evaluation at step N".
# Consumers import this rather than spelling the schema out again.
POINT_EVAL_SCHEMA = re.compile(r"^point-maze-[a-z0-9-]+-evaluation-v1$")

# Full-length panel titles, shared by the figure and the manuscript table. Two
# copies of this map drifted apart and the renderer raised KeyError on a domain
# the figure had already drawn; consumers import this rather than restate it.
DOMAIN_TITLES = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
    "point_maze": "PointMaze",
    # A separate domain from PointMaze, not a newer version of it.
    "point_maze_tour": "PointMaze Tour",
}


def point_checkpoint_step(path: Path) -> int:
    # Each PointMaze cohort stamps its own experiment id into the schema
    # (e78pm-, e79pm-, ...). Matching one literal silently reported every other
    # cohort as having saved no checkpoints at all.
    try:
        row = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not POINT_CHECKPOINT_SCHEMA.match(str(row.get("schema", ""))):
        return 0
    return int(row.get("update", 0))


def checkpoint_step(run_dir: Path) -> int:
    best = 0
    for path in run_dir.glob("debug_job*/checkpoints/step_*"):
        try:
            best = max(best, int(path.name.removeprefix("step_")))
        except ValueError:
            continue
    for path in run_dir.glob("debug_job*/checkpoints/latest"):
        try:
            value = path.read_text(encoding="utf-8").strip()
            best = max(best, int(value.removeprefix("step_")))
        except (OSError, ValueError):
            continue
    return best


def receipt_step(run_dir: Path) -> int:
    """Return the terminal step a completion receipt records, or 0.

    ``run_step`` reads a bounded tail of the metrics log, which assumes the log
    is ordered by step. A cell that was requeued after finishing appends a
    second training segment, so its tail holds *earlier* steps than the file's
    maximum and the tail read under-reports badly. The receipt is written once,
    at the end, and states the terminal step outright.
    """

    candidates = [run_dir / "TRAINING_COMPLETE.json"]
    candidates.extend(sorted(run_dir.glob("debug_job*/TRAINING_COMPLETE.json")))
    best = 0
    for path in candidates:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if row.get("schema") == "oat_zero_training_complete_v1":
            try:
                best = max(best, int(row.get("terminal_step", 0)))
            except (TypeError, ValueError):
                continue
        elif row.get("schema") == "e113r4_official_verl_dapo_training_complete_v1":
            try:
                best = max(best, int(row.get("total_training_steps", 0)))
            except (TypeError, ValueError):
                continue
    return best


def is_complete(run_dir: Path, step: int, target: int) -> bool:
    return step >= target or any(
        run_dir.glob("debug_job*/TRAINING_COMPLETE.json")
    ) or (run_dir / "TRAINING_COMPLETE.json").is_file()


def normalize_state(value: str) -> str:
    return value.split()[0].split("+")[0].upper()


def scheduler_states(job_ids: list[int]) -> dict[int, str]:
    """Read live states from squeue, then terminal states from sacct."""

    if not job_ids:
        return {}
    joined = ",".join(str(job_id) for job_id in job_ids)
    states: dict[int, str] = {}
    try:
        live = subprocess.run(
            ["squeue", "-h", "-j", joined, "-o", "%i|%T"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        live = None
    if live is not None and live.returncode == 0:
        for line in live.stdout.splitlines():
            job_id, separator, state = line.partition("|")
            if separator and job_id.isdigit():
                states[int(job_id)] = normalize_state(state)

    missing = [job_id for job_id in job_ids if job_id not in states]
    if not missing:
        return states
    try:
        history = subprocess.run(
            [
                "sacct",
                "-n",
                "-X",
                "-j",
                ",".join(str(job_id) for job_id in missing),
                "--format=JobIDRaw,State",
                "--parsable2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        history = None
    if history is not None and history.returncode == 0:
        wanted = set(missing)
        for line in history.stdout.splitlines():
            job_id, separator, state = line.partition("|")
            if separator and job_id.isdigit() and int(job_id) in wanted:
                states[int(job_id)] = normalize_state(state)
    return states


def load_snapshot(ledger_path: Path) -> dict[str, object]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    target = int(ledger["target_steps"])
    steps_per_pass = int(ledger["train_rows"])
    interval = int(ledger["checkpoint_interval_steps"])
    runs = ledger["runs"]
    states = scheduler_states([int(run["job_id"]) for run in runs])

    rows: list[dict[str, object]] = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        log_step = (
            verl_console_step(Path(str(run["log_path"])))
            if run.get("log_path")
            else 0
        )
        step = min(max(run_step(run_dir), receipt_step(run_dir), log_step), target)
        checkpoint = min(checkpoint_step(run_dir), target)
        complete = is_complete(run_dir, step, target)
        state = states.get(int(run["job_id"]), "NOT_IN_QUEUE")
        if complete:
            state = "COMPLETED"
        rows.append(
            {
                "arm": run["arm"],
                "checkpoint": checkpoint,
                "domain": run["domain"],
                "job_id": int(run["job_id"]),
                "seed": int(run["seed"]),
                "state": state,
                "step": step,
            }
        )
    return {
        "arms": ledger.get("arms")
        or sorted({str(run["arm"]) for run in runs}),
        "checkpoint_interval": interval,
        "domains": ledger.get("domains")
        or sorted({str(run["domain"]) for run in runs}),
        "passes": int(ledger["passes"]),
        "rows": rows,
        "steps_per_pass": steps_per_pass,
        "target": target,
    }


def load_point_snapshot(ledger_path: Path) -> dict[str, object]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    target = int(ledger["target_steps"])
    steps_per_pass = int(ledger["train_rows"])
    runs = ledger["runs"]
    raw_prepare = ledger.get("prepare_job_id", ledger.get("shared_data_prepare_job_id"))
    # Cohorts whose data was materialized offline have no prepare job to query.
    prepare_job_id = None if raw_prepare is None else int(raw_prepare)
    warmstart = ledger.get("warmstart")
    warmstart_job_id = (
        int(warmstart["job_id"]) if isinstance(warmstart, dict) else None
    )
    job_ids = [int(run["job_id"]) for run in runs]
    if prepare_job_id is not None:
        job_ids.insert(0, prepare_job_id)
    if warmstart_job_id is not None:
        job_ids.append(warmstart_job_id)
    states = scheduler_states(job_ids)
    rows = []
    for run in runs:
        step = min(point_run_step(Path(run["metrics_path"])), target)
        checkpoint = min(
            point_checkpoint_step(Path(run["checkpoint_dir"])), target
        )
        try:
            complete = (
                json.loads(
                    Path(run["receipt_path"]).read_text(encoding="utf-8")
                ).get("status")
                == "complete"
            )
        except (OSError, ValueError):
            complete = False
        state = states.get(int(run["job_id"]), "NOT_IN_QUEUE")
        if complete:
            state = "COMPLETED"
            step = target
        rows.append(
            {
                "arm": run["arm"],
                "checkpoint": checkpoint,
                "domain": "point_maze",
                "job_id": int(run["job_id"]),
                "seed": int(run["seed"]),
                "state": state,
                "step": step,
            }
        )
    return {
        "arms": ledger["arms"],
        "checkpoint_interval": int(ledger["checkpoint_interval_steps"]),
        "experiment": str(ledger.get("experiment", "E78-PM")),
        "passes": int(ledger["passes"]),
        "prepare_job_id": prepare_job_id,
        "prepare_state": (
            None if prepare_job_id is None
            else states.get(prepare_job_id, "NOT_IN_QUEUE")
        ),
        "warmstart_job_id": warmstart_job_id,
        "warmstart_state": (
            states.get(warmstart_job_id, "NOT_IN_QUEUE")
            if warmstart_job_id is not None
            else None
        ),
        "rows": rows,
        "steps_per_pass": steps_per_pass,
        "target": target,
    }


def state_counts(rows: list[dict[str, object]]) -> Counter[str]:
    return Counter(str(row["state"]) for row in rows)


def state_summary(counts: Counter[str]) -> str:
    ordered = [state for state in STATE_ORDER if counts[state]]
    ordered.extend(sorted(set(counts) - set(STATE_ORDER)))
    return " | ".join(f"{state.lower()} {counts[state]}" for state in ordered)


def render(snapshot: dict[str, object]) -> str:
    rows = snapshot["rows"]
    target = int(snapshot["target"])
    steps_per_pass = int(snapshot["steps_per_pass"])
    interval = int(snapshot["checkpoint_interval"])
    total = len(rows) * target
    realized = sum(int(row["step"]) for row in rows)
    observed = sum(int(row["step"]) > 0 for row in rows)
    complete = sum(int(row["step"]) >= target for row in rows)

    lines = [
        time.strftime("E78 status  %Y-%m-%d %H:%M:%S %Z"),
        (
            f"scheduler: {state_summary(state_counts(rows))}"
            if rows
            else "scheduler: no registered runs"
        ),
        (
            f"training:  {realized:,}/{total:,} optimizer steps "
            f"({100 * realized / total:.2f}%) | "
            f"started {observed}/{len(rows)} | terminal {complete}/{len(rows)}"
        ),
        (
            f"equivalent full-cohort depth: "
            f"{realized / (len(rows) * steps_per_pass):.2f}/"
            f"{snapshot['passes']} passes"
        ),
        "",
        "domain     arm      R  PD done   mean pass      range   saved ckpt",
        "---------- ------- -- --- ---- ----------- ---------- ------------",
    ]
    for domain in snapshot["domains"]:
        for arm in snapshot["arms"]:
            group = [
                row
                for row in rows
                if row["domain"] == domain and row["arm"] == arm
            ]
            counts = state_counts(group)
            passes = [int(row["step"]) / steps_per_pass for row in group]
            saved = max((int(row["checkpoint"]) for row in group), default=0)
            lines.append(
                f"{DOMAIN_LABELS.get(str(domain), str(domain)):<10} "
                f"{str(arm):<7} {counts['RUNNING']:>2} {counts['PENDING']:>3} "
                f"{counts['COMPLETED']:>4} {sum(passes) / len(passes):>8.2f} "
                f"{min(passes):>4.1f}-{max(passes):<4.1f} "
                f"{saved / steps_per_pass:>8.1f}"
            )

    checkpoint_parts = []
    for step in range(interval, target + 1, interval):
        count = sum(int(row["checkpoint"]) >= step for row in rows)
        checkpoint_parts.append(f"{step / steps_per_pass:g}: {count}/{len(rows)}")
        if count == 0:
            break
    lines.extend(
        [
            "",
            "registered checkpoint reach: " + " | ".join(checkpoint_parts),
            "live pass uses metrics; saved ckpt is the deepest retained half-pass file.",
        ]
    )
    return "\n".join(lines)


def render_point(snapshot: dict[str, object]) -> str:
    rows = snapshot["rows"]
    target = int(snapshot["target"])
    steps_per_pass = int(snapshot["steps_per_pass"])
    interval = int(snapshot["checkpoint_interval"])
    realized = sum(int(row["step"]) for row in rows)
    total = len(rows) * target
    counts = state_counts(rows)
    heading = f"{snapshot['experiment']} prospective PointMaze extension"
    lines = [
        heading,
        "-" * len(heading),
        (
            f"data certification: {snapshot['prepare_state'].lower()} "
            f"(job {snapshot['prepare_job_id']})"
        ),
    ]
    if snapshot.get("warmstart_job_id") is not None:
        lines.append(
            f"Falcon warm start: {str(snapshot['warmstart_state']).lower()} "
            f"(job {snapshot['warmstart_job_id']})"
        )
    lines.extend([
        f"scheduler: {state_summary(counts)}",
        (
            f"training:  {realized:,}/{total:,} optimizer steps "
            f"({100 * realized / total:.2f}%) | "
            f"started {sum(int(row['step']) > 0 for row in rows)}/{len(rows)} | "
            f"terminal {sum(int(row['step']) >= target for row in rows)}/{len(rows)}"
        ),
        "arm      R  PD done   mean pass      range   saved ckpt",
        "------- -- --- ---- ----------- ---------- ------------",
    ])
    for arm in snapshot["arms"]:
        group = [row for row in rows if row["arm"] == arm]
        arm_counts = state_counts(group)
        passes = [int(row["step"]) / steps_per_pass for row in group]
        saved = max((int(row["checkpoint"]) for row in group), default=0)
        lines.append(
            f"{str(arm):<7} {arm_counts['RUNNING']:>2} "
            f"{arm_counts['PENDING']:>3} {arm_counts['COMPLETED']:>4} "
            f"{sum(passes) / len(passes):>8.2f} "
            f"{min(passes):>4.1f}-{max(passes):<4.1f} "
            f"{saved / steps_per_pass:>8.1f}"
        )
    reach = []
    for step in range(interval, target + 1, interval):
        count = sum(int(row["checkpoint"]) >= step for row in rows)
        reach.append(f"{step / steps_per_pass:g}: {count}/{len(rows)}")
        if count == 0:
            break
    lines.append("registered checkpoint reach: " + " | ".join(reach))
    return "\n".join(lines)


def pair_depth(
    ledger_path: Path, target: int
) -> dict[tuple[str, int, str], int]:
    """Return realized steps for every run in a comparator ledger."""

    if not ledger_path.is_file():
        return {}
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    depth: dict[tuple[str, int, str], int] = {}
    for run in ledger["runs"]:
        run_dir = Path(run["run_dir"])
        step = min(run_step(run_dir), target)
        if is_complete(run_dir, step, target):
            step = target
        depth[(str(run["domain"]), int(run["seed"]), str(run["arm"]))] = step
    return depth


def e78_pair_depth(target: int) -> dict[tuple[str, int, str], int]:
    """Realized steps for every E78 run, keyed by domain/seed/arm."""

    return pair_depth(DEFAULT_LEDGER, target)


def render_semantic(
    snapshot: dict[str, object],
    pairs: dict[tuple[str, int, str], int],
    *,
    label: str = "E81 semantic-MaxEnt extension",
) -> str:
    """Render a semantic arm's depth and how much of it is analysably paired."""

    rows = snapshot["rows"]
    target = int(snapshot["target"])
    steps_per_pass = int(snapshot["steps_per_pass"])
    interval = int(snapshot["checkpoint_interval"])
    realized = sum(int(row["step"]) for row in rows)
    total = len(rows) * target
    coefficient = snapshot.get("semantic_coefficient", 0.10)
    heading = (
        f"{label} (eta = {coefficient:g}, "
        f"verified replay {snapshot.get('replay_weight', 0.10):g})"
    )

    def paired(domain: str, seed: int) -> bool:
        semantic = max(
            (
                int(row["step"])
                for row in rows
                if row["domain"] == domain and int(row["seed"]) == seed
            ),
            default=0,
        )
        return (
            min(
                semantic,
                pairs.get((domain, seed, "replay"), 0),
                pairs.get((domain, seed, "control"), 0),
            )
            >= target
        )

    lines = [
        heading,
        "-" * len(heading),
        f"scheduler: {state_summary(state_counts(rows))}",
        (
            f"training:  {realized:,}/{total:,} optimizer steps "
            f"({100 * realized / total:.2f}%) | "
            f"started {sum(int(row['step']) > 0 for row in rows)}/{len(rows)} | "
            f"terminal {sum(int(row['step']) >= target for row in rows)}/{len(rows)}"
        ),
        "domain      R  PD done   mean pass      range   saved ckpt  paired",
        "---------- -- --- ---- ----------- ---------- ------------ -------",
    ]
    ready = 0
    for domain in snapshot["domains"]:
        group = [row for row in rows if row["domain"] == domain]
        if not group:
            continue
        counts = state_counts(group)
        passes = [int(row["step"]) / steps_per_pass for row in group]
        saved = max((int(row["checkpoint"]) for row in group), default=0)
        complete = sum(paired(str(domain), int(row["seed"])) for row in group)
        ready += complete
        lines.append(
            f"{DOMAIN_LABELS.get(str(domain), str(domain)):<10} "
            f"{counts['RUNNING']:>2} {counts['PENDING']:>3} "
            f"{counts['COMPLETED']:>4} {sum(passes) / len(passes):>8.2f} "
            f"{min(passes):>4.1f}-{max(passes):<4.1f} "
            f"{saved / steps_per_pass:>8.1f} {f'{complete}/{len(group)}':>10}"
        )

    reach = []
    for step in range(interval, target + 1, interval):
        count = sum(int(row["checkpoint"]) >= step for row in rows)
        reach.append(f"{step / steps_per_pass:g}: {count}/{len(rows)}")
        if count == 0:
            break
    lines.extend(
        [
            "registered checkpoint reach: " + " | ".join(reach),
            (
                f"paired cells with semantic, replay, and control all at "
                f"pass {snapshot['passes']}: {ready}/{len(rows)} "
                f"(comparator arms inherited, never re-run)"
            ),
        ]
    )
    return "\n".join(lines)


def load_semantic_snapshot(ledger_path: Path) -> dict[str, object]:
    snapshot = load_snapshot(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    snapshot["semantic_coefficient"] = float(ledger.get("semantic_coefficient", 0.10))
    snapshot["replay_weight"] = float(ledger.get("replay_weight", 0.10))
    return snapshot


def semantic_section(
    ledger_path: Path, pair_ledger_path: Path, label: str
) -> str | None:
    """Render one semantic arm beside the comparator ledger it pairs against."""

    if not ledger_path.is_file():
        return None
    snapshot = load_semantic_snapshot(ledger_path)
    pairs = pair_depth(pair_ledger_path, int(snapshot["target"]))
    return render_semantic(snapshot, pairs, label=label)


# Alias retained so existing callers keep working.
load_e81_snapshot = load_semantic_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        metavar="SECONDS",
        help="refresh continuously at this interval; Ctrl-C exits",
    )
    args = parser.parse_args()
    while True:
        report = render(load_snapshot(args.ledger.resolve()))
        if POINT_LEDGER.is_file():
            report += "\n\n" + render_point(load_point_snapshot(POINT_LEDGER))
        section = semantic_section(
            E81_LEDGER, DEFAULT_LEDGER, "E81 semantic-MaxEnt extension"
        )
        if section:
            report += "\n\n" + section
        print(report, flush=True)
        if args.watch <= 0:
            return
        print(flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
