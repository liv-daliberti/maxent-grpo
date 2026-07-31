from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from oat_drgrpo.ant_maze_worker_v8 import RECEIPT_SHA256, controller_identity
from oat_drgrpo.maze_modebench import (
    ANT_ACTIONS,
    ANT_MAZE_ACTION_VERSION,
    ANT_MAZE_VERIFIER,
)
from oat_drgrpo.maze_modebench_process import MazeVerifierProcess


ROOT = Path(__file__).resolve().parents[1]
WORKER_PYTHON = ROOT / "var/maze_runtime/venv/bin/python"
RECEIPT = ROOT / "var/maze_runtime/controllers/ant_waypoint_v8.evaluation.json"
MODEL = ROOT / "var/maze_runtime/controllers/ant_waypoint_v8.zip"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _spec() -> dict:
    identity = json.loads(
        subprocess.check_output(
            [
                str(WORKER_PYTHON),
                "-c",
                (
                    "import json,sys;sys.path.insert(0,'src');"
                    "from oat_drgrpo.maze_runtime_identity import "
                    "maze_runtime_identity;"
                    "print(json.dumps(maze_runtime_identity()))"
                ),
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()[-1]
    )
    spec = {
        "verifier": ANT_MAZE_VERIFIER,
        "maze_action_version": ANT_MAZE_ACTION_VERSION,
        "environment_id": "AntMaze_UMaze-v5",
        "environment_sha256": identity["ant_environment_sha256"],
        "controller_sha256": RECEIPT_SHA256,
        "map_id": "v8_excluded_adjacent_worker_test",
        "maze_map": [[1] * 5]
        + [[1, 0, 0, 0, 1] for _ in range(3)]
        + [[1] * 5],
        "reset_cell": [2, 1],
        "goal_cell": [2, 2],
        "reset_seed": 92_801,
        "min_actions": 1,
        "max_actions": 4,
        "action_repeat": 400,
        "action_tokens": list(ANT_ACTIONS),
        "success_threshold": 0.5,
        "max_segment_length": 1.0,
        "bounds_xy": [[-8.0, 8.0], [-8.0, 8.0]],
        "route_gates": [
            {
                "id": "straight",
                "axis": "x",
                "coordinate": -2.0,
                "span": [-2.0, 2.0],
                "hysteresis": 0.05,
            }
        ],
    }
    spec["spec_sha256"] = _canonical_sha256(spec)
    return spec


@pytest.mark.skipif(
    not WORKER_PYTHON.is_file() or not RECEIPT.is_file() or not MODEL.is_file(),
    reason="pinned Ant v8 runtime/controller is unavailable",
)
def test_ant_v8_identity_and_excluded_adjacent_worker_fixture():
    assert controller_identity()["status"] == "pass"
    verifier = MazeVerifierProcess(
        worker_python=WORKER_PYTHON,
        timeout_seconds=25.0,
    )
    try:
        result = verifier.validate_with_execution("E", _spec())
    finally:
        verifier.close()
    assert result is not None
    validation, execution = result
    assert validation.directed_gates == ("straight+",)
    assert execution["final_goal_distance"] <= 0.5
