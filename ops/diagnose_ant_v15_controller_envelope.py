#!/usr/bin/env python3
"""Replay candidate v15 routes and print raw controller-envelope diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.ant_maze_worker_v12 import (  # noqa: E402
    controller_receipt_sha256,
    execute_ant_v12_raw,
)
from oat_drgrpo.maze_modebench import (  # noqa: E402
    ANT_ACTIONS,
    ANT_MAZE_ACTION_VERSION,
    ANT_MAZE_VERIFIER,
    parse_maze_action_spec,
)
from oat_drgrpo.maze_runtime_identity import maze_runtime_identity  # noqa: E402


def main() -> None:
    maze_map = [[1] * 13]
    maze_map.extend([[1] + [0] * 11 + [1] for _ in range(11)])
    maze_map.append([1] * 13)
    for row in range(5, 8):
        for column in range(5, 8):
            maze_map[row][column] = 1
    runtime = maze_runtime_identity()
    spec = {
        "verifier": ANT_MAZE_VERIFIER,
        "maze_action_version": ANT_MAZE_ACTION_VERSION,
        "environment_id": "AntMaze_UMaze-v5",
        "environment_sha256": runtime["ant_environment_sha256"],
        "controller_sha256": controller_receipt_sha256(),
        "map_id": "ant_v15_controller_envelope",
        "maze_map": maze_map,
        "reset_cell": [6, 4],
        "goal_cell": [6, 8],
        "reset_seed": 108_500,
        "min_actions": 2,
        "max_actions": 20,
        "action_repeat": 400,
        "action_tokens": list(ANT_ACTIONS),
        "success_threshold": 0.5,
        "max_segment_length": 1.0,
        "bounds_xy": [[-24.0, 24.0], [-24.0, 24.0]],
        "route_gates": [
            {
                "id": "upper",
                "axis": "x",
                "coordinate": 0.0,
                "span": [1.0, 20.0],
                "hysteresis": 0.1,
            },
            {
                "id": "lower",
                "axis": "x",
                "coordinate": 0.0,
                "span": [-20.0, -1.0],
                "hysteresis": 0.1,
            },
        ],
    }
    spec["spec_sha256"] = parse_maze_action_spec(spec).spec_sha256
    candidates = (
        "N N E E E E S S",
        "S S E E E E N N",
        "N E E E E S",
        "S E E E E N",
    )
    rows = []
    for candidate in candidates:
        execution = execute_ant_v12_raw(candidate, spec)
        trajectory = execution["trajectory_xy"]
        rows.append(
            {
                "candidate": candidate,
                "success": execution["success"],
                "final_goal_distance": execution["final_goal_distance"],
                "simulator_steps": execution["simulator_steps"],
                "segment_steps": execution["segment_steps"],
                "nominal_targets_xy": execution["nominal_targets_xy"],
                "start_xy": trajectory[0],
                "finish_xy": trajectory[-1],
                "x_range": [
                    min(point[0] for point in trajectory),
                    max(point[0] for point in trajectory),
                ],
                "y_range": [
                    min(point[1] for point in trajectory),
                    max(point[1] for point in trajectory),
                ],
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
