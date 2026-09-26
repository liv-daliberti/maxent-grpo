#!/usr/bin/env python3
"""Configure or launch the balanced PointMaze repair-v2 viability gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import launch_point_maze_algorithm_repair_v1_viability as base


ROOT = Path(__file__).resolve().parents[2]
base.PROTOCOL = (
    ROOT / "paper/preregistration/point_maze_algorithm_repair_v2_20260730.md"
)
base.BATCH = ROOT / "ops/slurm/evaluate_point_maze_algorithm_repair_v2.slurm"
base.DATA = ROOT / "var/data/point_maze_algorithm_repair_v2"
base.ADMISSION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v2_admission_audit.json"
)
base.OUTPUT = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v2_viability.json"
)
base.IDENTITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v2_viability_identity.json"
)
base.SUBMISSION = (
    ROOT
    / "var/artifacts/point_maze_algorithm_repair_v2_viability_submission.json"
)

_snapshot_tree = base.snapshot_tree
_atomic = base.atomic


def snapshot_tree(source: Path, _prefix: str):
    return _snapshot_tree(source, "point_repair_v2_source")


def atomic(path: Path, payload: Any) -> None:
    if path == base.IDENTITY:
        payload = {
            **payload,
            "schema": "point-maze-algorithm-repair-v2-viability-identity",
            "seed": 76520,
            "balanced_medium_hard_slate": True,
            "v1_development_outcome_loaded": True,
            "v2_final_seed": False,
        }
    elif path == base.SUBMISSION:
        payload = {
            **payload,
            "schema": "point-maze-algorithm-repair-v2-viability-submission",
        }
    _atomic(path, payload)


base.snapshot_tree = snapshot_tree
base.atomic = atomic


if __name__ == "__main__":
    base.main()
