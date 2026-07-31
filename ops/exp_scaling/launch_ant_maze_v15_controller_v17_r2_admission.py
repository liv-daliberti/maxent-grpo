#!/usr/bin/env python3
"""Launch the timeout-corrected frozen Ant v15/v17 admission binding."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "ops", ROOT / "ops/exp_scaling", ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import launch_ant_maze_v15_controller_v17_r1_admission as v1  # noqa: E402


v1.LAUNCHER = Path(__file__).resolve()
v1.base.MAKER = (
    ROOT / "ops/make_ant_maze_mode_data_v15_controller_v17_r2.py"
)
v1.base.AUDITOR = (
    ROOT / "ops/audit_ant_maze_mode_data_v15_controller_v17_r2.py"
)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        v1.controller_identity()
    v1.base.main()
