#!/usr/bin/env python3
"""Relaunch AntMaze v14-hard with immutable Python source snapshots."""

from pathlib import Path

import launch_ant_maze_v14_hard_admission_r1 as r1


ROOT = Path(__file__).resolve().parents[2]
base = r1.base
base.PROTOCOL = (
    ROOT / "paper/preregistration/ant_maze_v14_hard_admission_r2_20260730.md"
)
base.BATCH = ROOT / "ops/slurm/admit_ant_maze_modebench_v14_hard_r2.slurm"
base.IDENTITY = ROOT / "var/artifacts/ant_maze_v14_hard_admission_r2_identity.json"
base.SUBMISSION = (
    ROOT / "var/artifacts/ant_maze_v14_hard_admission_r2_submission.json"
)
_snapshot_tree = base.snapshot_tree


def snapshot_tree(source: Path, _prefix: str):
    return _snapshot_tree(source, "ant_v14_hard_r2_source")


base.snapshot_tree = snapshot_tree


if __name__ == "__main__":
    base.main()

