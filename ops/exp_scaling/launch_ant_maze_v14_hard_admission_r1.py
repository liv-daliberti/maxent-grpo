#!/usr/bin/env python3
"""Relaunch AntMaze v14-hard after protocol-identity isolation repair."""

from pathlib import Path

import launch_ant_maze_v14_hard_admission as base


ROOT = Path(__file__).resolve().parents[2]
base.PROTOCOL = (
    ROOT / "paper/preregistration/ant_maze_v14_hard_admission_r1_20260730.md"
)
base.MAKER = ROOT / "ops/make_ant_maze_mode_data_v14_hard_r1.py"
base.AUDITOR = ROOT / "ops/audit_ant_maze_mode_data_v14_hard_r1.py"
base.BATCH = ROOT / "ops/slurm/admit_ant_maze_modebench_v14_hard_r1.slurm"
base.IDENTITY = ROOT / "var/artifacts/ant_maze_v14_hard_admission_r1_identity.json"
base.SUBMISSION = (
    ROOT / "var/artifacts/ant_maze_v14_hard_admission_r1_submission.json"
)


if __name__ == "__main__":
    base.main()

