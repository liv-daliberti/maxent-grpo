#!/usr/bin/env python3
"""Write the job ledger for the PointMaze Tour admission-gate cells.

These cells are *not* paper evidence. They are the stage-2 collapse gate and the
stage-3 matched smoke from ``docs/point_maze_tour_v2_redesign_plan.md``, run on
development maps to decide whether the redesigned sixth domain is worth a
cohort at all. They are registered so campaign_stats can report their depth
alongside everything else, and marked unplotted with that reason.

Regenerate after submitting the gate jobs:

    python ops/exp_scaling/register_point_maze_tour_gate.py \\
        --control 43:<jobid> 44:<jobid> --replay 43:<jobid>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"
LEDGER = ARTIFACTS / "point_maze_tour_gate_jobs.json"

TRAIN_ROWS = 384
CHECKPOINT_INTERVAL = 384


def scheduler_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def pair(value: str) -> tuple[int, int]:
    seed, _, job = value.partition(":")
    if not job:
        raise argparse.ArgumentTypeError("expected seed:jobid")
    return int(seed), int(job)


def run_row(arm: str, seed: int, job_id: int, stage: str) -> dict[str, object]:
    stem = f"tour_{stage}_{arm}_s{seed}"
    return {
        "arm": arm,
        "domain": "point_maze_tour",
        "seed": seed,
        "job_id": job_id,
        "stage": stage,
        "metrics_path": str(ARTIFACTS / f"{stem}.metrics.jsonl"),
        "receipt_path": str(ARTIFACTS / f"{stem}.json"),
        # These gate cells do not checkpoint; the reader reports 0 for a missing
        # directory, which is the honest value.
        "checkpoint_dir": str(ROOT / "var/checkpoints" / stem),
        "scheduler_record": scheduler_record(job_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=pair, nargs="+", required=True)
    parser.add_argument("--replay", type=pair, nargs="*", default=[])
    parser.add_argument("--data-root", default=str(ROOT / "var/data/point_maze_tour_k5"))
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--stage-control", default="stage2k5")
    parser.add_argument("--stage-replay", default="stage23k5")
    args = parser.parse_args()

    runs = [run_row("control", s, j, args.stage_control) for s, j in args.control]
    runs += [run_row("replay", s, j, args.stage_replay) for s, j in args.replay]

    ledger = {
        "schema": "point_maze_tour_gate_jobs_v1",
        "experiment": "TOUR-GATE",
        "purpose": (
            "stage-2 collapse gate and stage-3 matched smoke on development "
            "maps; admission evidence for the redesigned sixth domain, not a "
            "paper estimate"
        ),
        "protocol": "docs/point_maze_tour_v2_redesign_plan.md",
        "arms": sorted({row["arm"] for row in runs}),
        "seeds": sorted({int(row["seed"]) for row in runs}),
        "domains": ["point_maze_tour"],
        "passes": args.passes,
        "train_rows": TRAIN_ROWS,
        "dev_rows": 64,
        "eval_rows": 128,
        "target_steps": args.passes * TRAIN_ROWS,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
        "evaluation_split": "dev",
        "data_root": args.data_root,
        "data_identity_sha256": None,
        "initial_model": "Qwen2.5-0.5B-Instruct (untouched; no warm start)",
        "gate_threshold": {
            "metric": "distinct8",
            "rule": "control must fall by >= 0.50 from its pass-0 value",
        },
        "released": True,
        "runs": runs,
    }
    identity = ARTIFACTS / "point_maze_tour_v1r1_identity.json"
    source = Path(args.data_root) / "identity.json"
    if source.is_file():
        import hashlib

        ledger["data_identity_sha256"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    LEDGER.write_text(
        json.dumps(ledger, indent=1, sort_keys=True) + "\n", encoding="ascii"
    )
    print(f"wrote {LEDGER} with {len(runs)} cells")


if __name__ == "__main__":
    main()
