#!/usr/bin/env python3
"""Schedule the self-contained frozen Ant v15/v17 admission launcher."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "ops/exp_scaling") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops/exp_scaling"))

import schedule_ant_maze_v15_controller_v17_r1 as v1  # noqa: E402


v1.DEPENDENCY_BATCH = (
    ROOT
    / "ops/slurm/"
    "launch_ant_maze_v15_controller_v17_r3_after_controller.slurm"
)
v1.LAUNCHER = (
    ROOT
    / "ops/exp_scaling/"
    "launch_ant_maze_v15_controller_v17_r3_admission.py"
)
v1.MANIFEST = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v17_r3_dependent_launcher_identity.json"
)
v1.FILES = (
    ROOT
    / "paper/preregistration/"
    "ant_maze_v15_controller_v17_r1_admission_20260730.md",
    ROOT / "ops/make_ant_maze_mode_data_v15_controller_v17_r3.py",
    ROOT / "ops/audit_ant_maze_mode_data_v15_controller_v17_r3.py",
    ROOT / "ops/make_ant_maze_mode_data.py",
    ROOT / "ops/audit_ant_maze_mode_data.py",
    v1.LAUNCHER,
    ROOT
    / "ops/exp_scaling/"
    "launch_ant_maze_v15_controller_v17_r1_admission.py",
    ROOT / "ops/exp_scaling/launch_ant_maze_v14_hard_admission.py",
    ROOT / "ops/slurm/admit_ant_maze_modebench_v15_controller_v17_r1.slurm",
    v1.DEPENDENCY_BATCH,
    Path(__file__).resolve(),
    ROOT / "ops/exp_scaling/schedule_ant_maze_v15_controller_v17_r1.py",
    ROOT / "src/oat_drgrpo/ant_maze_worker_v17.py",
    ROOT / "src/oat_drgrpo/ant_maze_worker_v17_r1.py",
    ROOT / "src/oat_drgrpo/maze_modebench_worker_v17_r1.py",
    ROOT / "src/oat_drgrpo/maze_modebench_process_v17_r1.py",
)


if __name__ == "__main__":
    v1.main()
