from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from oat_drgrpo.point_maze_interactive_process import (
    PointMazeInteractiveProcess,
)


ROOT = Path(__file__).resolve().parents[1]
WORKER_PYTHON = ROOT / "var/maze_runtime/venv/bin/python"
DATA_ROOT = ROOT / "var/data/point_maze_modebench_v1"


@pytest.mark.skipif(
    not WORKER_PYTHON.is_file() or not DATA_ROOT.is_dir(),
    reason="pinned PointMaze runtime/data are unavailable",
)
def test_stepwise_worker_preserves_two_admitted_route_identities():
    from datasets import load_from_disk

    row = load_from_disk(str(DATA_ROOT / "dev"))["multi_answer"][0]
    spec = json.loads(row["answer"])
    identity = json.loads((DATA_ROOT / "identity.json").read_text())
    certification = next(
        record
        for record in identity["certification"]
        if record["map_id"] == spec["map_id"]
    )
    programs = [
        route["program"].split() for route in certification["routes"]
    ]
    process = PointMazeInteractiveProcess(worker_python=WORKER_PYTHON)
    try:
        resets = process.reset_batch(
            [
                {"session_id": f"route-{index}", "spec": spec}
                for index in range(2)
            ]
        )
        assert all(len(result["velocity_xy"]) == 2 for result in resets)
        final = [None, None]
        for step_index in range(max(map(len, programs))):
            requests = [
                {
                    "session_id": f"route-{index}",
                    "action": program[step_index],
                }
                for index, program in enumerate(programs)
                if step_index < len(program) and final[index] is None
            ]
            for result in process.step_batch(requests):
                assert len(result["velocity_xy"]) == 2
                index = int(result["session_id"].split("-")[-1])
                if result["done"]:
                    final[index] = result
        recycled = process.reset_batch(
            [{"session_id": "route-0", "spec": spec}]
        )
        assert recycled[0]["session_id"] == "route-0"
    finally:
        process.close()

    assert all(result is not None and result["success"] for result in final)
    assert len({result["canonical_key"] for result in final}) == 2
    assert all(result["validation_error"] is None for result in final)
