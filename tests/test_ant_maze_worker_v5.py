from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from oat_drgrpo.maze_modebench import (
    ANT_ACTIONS,
    ANT_MAZE_ACTION_VERSION,
    ANT_MAZE_VERIFIER,
)
from oat_drgrpo.maze_modebench_process import MazeVerifierProcess


ROOT = Path(__file__).resolve().parents[1]
WORKER_PYTHON = ROOT / "var/maze_runtime/venv/bin/python"
CONTROLLER_RECEIPT = (
    ROOT / "var/maze_runtime/controllers/ant_heading_v5.evaluation.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "controller_sha256": _sha256(CONTROLLER_RECEIPT),
        "map_id": "adjacent_east_worker_test",
        "maze_map": [[1] * 5]
        + [[1, 0, 0, 0, 1] for _ in range(3)]
        + [[1] * 5],
        "reset_cell": [2, 1],
        "goal_cell": [2, 2],
        "reset_seed": 82001,
        "min_actions": 2,
        "max_actions": 8,
        "action_repeat": 100,
        "action_tokens": list(ANT_ACTIONS),
        "success_threshold": 0.5,
        "max_segment_length": 0.2,
        "bounds_xy": [[-8.0, 8.0], [-8.0, 8.0]],
        "route_gates": [
            {
                "id": "lower",
                "axis": "x",
                "coordinate": -1.5,
                "span": [-0.1, 0.4],
                "hysteresis": 0.05,
            },
            {
                "id": "upper",
                "axis": "x",
                "coordinate": -1.5,
                "span": [0.65, 1.1],
                "hysteresis": 0.05,
            },
        ],
    }
    spec["spec_sha256"] = _canonical_sha256(spec)
    return spec


@pytest.mark.skipif(
    not WORKER_PYTHON.is_file() or not CONTROLLER_RECEIPT.is_file(),
    reason="pinned Ant v5 runtime/controller is unavailable",
)
def test_external_ant_v5_worker_executes_two_distinct_language_routes():
    verifier = MazeVerifierProcess(
        worker_python=WORKER_PYTHON,
        timeout_seconds=25.0,
    )
    try:
        lower = verifier.validate("E S E SE", _spec())
        upper = verifier.validate("E SE E S", _spec())
    finally:
        verifier.close()

    assert lower is not None and lower.directed_gates == ("lower+",)
    assert upper is not None and upper.directed_gates == ("upper+",)
    assert lower.canonical_key != upper.canonical_key
    assert ANT_MAZE_ACTION_VERSION in lower.canonical_key
