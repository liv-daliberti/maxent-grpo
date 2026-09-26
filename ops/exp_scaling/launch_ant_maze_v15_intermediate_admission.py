#!/usr/bin/env python3
"""Configure or launch the prospective AntMaze v15 admission gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import launch_ant_maze_v14_hard_admission as base


ROOT = Path(__file__).resolve().parents[2]
base.PROTOCOL = (
    ROOT
    / "paper/preregistration/ant_maze_v15_intermediate_admission_20260730.md"
)
base.MAKER = ROOT / "ops/make_ant_maze_mode_data_v15_intermediate.py"
base.AUDITOR = ROOT / "ops/audit_ant_maze_mode_data_v15_intermediate.py"
base.BATCH = (
    ROOT / "ops/slurm/admit_ant_maze_modebench_v15_intermediate.slurm"
)
base.DATA = ROOT / "var/data/ant_maze_modebench_v15_intermediate"
base.AUDIT = (
    ROOT
    / "var/artifacts/ant_maze_modebench_v15_intermediate_admission_audit.json"
)
base.IDENTITY = (
    ROOT / "var/artifacts/ant_maze_v15_intermediate_admission_identity.json"
)
base.SUBMISSION = (
    ROOT / "var/artifacts/ant_maze_v15_intermediate_admission_submission.json"
)

_snapshot_tree = base.snapshot_tree
_atomic = base.atomic


def snapshot_tree(source: Path, _prefix: str):
    return _snapshot_tree(source, "ant_v15_intermediate_source")


def atomic(path: Path, payload: Any) -> None:
    if path == base.IDENTITY:
        payload = {
            **payload,
            "schema": "ant-maze-v15-intermediate-admission-identity-v1",
            "map_size": 13,
            "central_obstacle_shape": [3, 3],
            "route_decisions": 8,
            "max_actions": 20,
            "action_repeat": 400,
            "v14_route_outcome_loaded": True,
            "v15_model_sampled": False,
            "post_v15_outcome_map_substitution": False,
        }
    elif path == base.SUBMISSION:
        payload = {
            **payload,
            "schema": "ant-maze-v15-intermediate-admission-submission-v1",
        }
    _atomic(path, payload)


base.snapshot_tree = snapshot_tree
base.atomic = atomic


if __name__ == "__main__":
    base.main()
