from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from oat_drgrpo.maze_modebench import MazeModeBenchError
from oat_drgrpo.point_maze_waypoint import (
    POINT_WAYPOINT_ACTION_VERSION,
    POINT_WAYPOINT_CONTROLLER_SHA256,
    find_point_waypoint_route_programs,
    legal_point_waypoint_actions,
    make_point_waypoint_spec,
    parse_point_waypoint_spec,
    render_point_waypoint_problem,
)
from oat_drgrpo.point_maze_waypoint_data import (
    DEFAULT_POINT_WAYPOINT_SPLIT_COUNTS,
    generate_point_waypoint_tasks,
)
from oat_drgrpo.point_maze_waypoint_policy import (
    point_waypoint_allowed_labels,
    point_waypoint_transition_sha256,
    render_point_waypoint_prompt,
)
from oat_drgrpo.point_maze_waypoint_process import PointMazeWaypointProcess


ROOT = Path(__file__).resolve().parents[1]
WORKER_PYTHON = ROOT / "var/maze_runtime/venv/bin/python"


def _three_route_spec(environment_sha256: str = "a" * 64):
    size = 9
    maze_map = [
        [int(row in {0, size - 1} or column in {0, size - 1}) for column in range(size)]
        for row in range(size)
    ]
    openings = (2, 4, 6)
    for row in range(1, size - 1):
        if row not in openings:
            maze_map[row][size // 2] = 1
    center = (size - 1) / 2.0
    gates = [
        {
            "id": f"corridor_{row}",
            "axis": "x",
            "coordinate": 0.0,
            "span": [center - row - 0.4, center - row + 0.4],
            "hysteresis": 0.1,
        }
        for row in openings
    ]
    return make_point_waypoint_spec(
        environment_sha256=environment_sha256,
        map_id="waypoint_test_3route",
        maze_map=maze_map,
        reset_cell=(4, 1),
        goal_cell=(4, 7),
        reset_seed=88001,
        route_gates=gates,
        max_actions=32,
    )


def _runtime_environment_sha256() -> str:
    source = (
        "import json,sys;"
        "sys.path.insert(0,'src');"
        "from oat_drgrpo.maze_runtime_identity import maze_runtime_identity;"
        "print(json.dumps(maze_runtime_identity(),sort_keys=True))"
    )
    completed = subprocess.run(
        [str(WORKER_PYTHON), "-c", source],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.splitlines()[-1])["point_environment_sha256"]


def test_waypoint_spec_finds_three_simple_routes_and_binds_controller():
    raw = _three_route_spec()
    spec = parse_point_waypoint_spec(raw)
    assert raw["waypoint_action_version"] == POINT_WAYPOINT_ACTION_VERSION
    assert raw["controller_sha256"] == POINT_WAYPOINT_CONTROLLER_SHA256
    assert raw["route_identity_rule"] == "exactly-one-directed-gate"
    assert legal_point_waypoint_actions(spec, spec.base_spec.reset_cell) == (
        "N",
        "E",
        "S",
    )
    programs = find_point_waypoint_route_programs(raw)
    assert set(programs) == {"corridor_2+", "corridor_4+", "corridor_6+"}
    assert all("," not in route for route in programs)


def test_waypoint_spec_hash_fails_closed():
    raw = _three_route_spec()
    raw["max_actions"] += 1
    with pytest.raises(MazeModeBenchError, match="hash"):
        parse_point_waypoint_spec(raw)


def test_waypoint_prompt_lists_only_legal_moves_and_full_horizon_state():
    raw = _three_route_spec()
    problem = render_point_waypoint_problem(raw)
    observation = {
        "current_cell": [4, 1],
        "previous_cell": None,
        "goal_cell": [4, 7],
        "achieved_goal": [-3.0, 0.0],
        "desired_goal": [3.0, 0.0],
        "velocity_xy": [0.0, 0.0],
        "remaining_actions": 32,
        "allowed_actions": ["N", "E", "S"],
    }
    assert point_waypoint_allowed_labels(observation["allowed_actions"]) == (
        "A",
        "B",
        "C",
    )
    prompt = render_point_waypoint_prompt(problem, observation)
    assert "current_cell=(4,1)" in prompt
    assert "previous_cell=START" in prompt
    assert "remaining_waypoints=32" in prompt
    assert "A: N" in prompt and "B: E" in prompt and "C: S" in prompt
    assert "D: W" not in prompt


def test_waypoint_transition_hash_includes_logical_state_and_support():
    before = {
        "achieved_goal": [-3.0, 0.0],
        "desired_goal": [3.0, 0.0],
        "velocity_xy": [0.0, 0.0],
        "current_cell": [4, 1],
        "previous_cell": None,
        "goal_cell": [4, 7],
        "remaining_actions": 32,
        "allowed_actions": ["N", "E", "S"],
    }
    after = {
        **before,
        "achieved_goal": [-2.0, 0.0],
        "current_cell": [4, 2],
        "previous_cell": [4, 1],
        "remaining_actions": 31,
        "allowed_actions": ["N", "E", "S", "W"],
    }
    baseline = point_waypoint_transition_sha256(before=before, action="E", after=after)
    changed = {
        **after,
        "allowed_actions": ["N", "E", "S"],
    }
    assert baseline != point_waypoint_transition_sha256(
        before=before, action="E", after=changed
    )


def test_pilot_generator_has_nonoverlapping_three_to_five_mode_splits():
    tasks = generate_point_waypoint_tasks(
        environment_sha256="a" * 64,
        split_counts={"train": 4, "dev": 3, "eval": 3},
    )
    assert DEFAULT_POINT_WAYPOINT_SPLIT_COUNTS == {
        "train": 64,
        "dev": 32,
        "eval": 64,
    }
    fingerprints = [
        task.instance_fingerprint for rows in tasks.values() for task in rows
    ]
    assert len(fingerprints) == len(set(fingerprints)) == 10
    replacement = generate_point_waypoint_tasks(
        environment_sha256="a" * 64,
        split_counts={"train": 4, "dev": 3, "eval": 3},
        seed=88_101,
        excluded_fingerprints=fingerprints,
    )
    replacement_fingerprints = {
        task.instance_fingerprint for rows in replacement.values() for task in rows
    }
    assert replacement_fingerprints.isdisjoint(fingerprints)
    assert all(
        3 <= len(task.route_programs) <= 5 for rows in tasks.values() for task in rows
    )
    for rows in tasks.values():
        rotations = [int(task.spec["map_id"].rsplit("_r", 1)[1]) for task in rows]
        assert rotations == [index % 4 for index in range(len(rows))]


def test_point_trainer_records_and_renormalizes_per_decision_support():
    source = (ROOT / "ops/train_point_maze_interactive_paired_smoke_v1.py").read_text()
    assert "allowed_token_ids_by_prompt" in source
    assert 'slot.get("allowed_token_ids", fallback_support)' in source
    assert '"allowed_token_ids": decision.allowed_token_ids' in source
    assert "selected action lies outside its recorded support" in source


@pytest.mark.skipif(
    not WORKER_PYTHON.is_file(),
    reason="pinned PointMaze runtime is unavailable",
)
def test_three_waypoint_routes_execute_with_distinct_canonical_keys():
    raw = _three_route_spec(_runtime_environment_sha256())
    programs = find_point_waypoint_route_programs(raw)
    results = []
    with PointMazeWaypointProcess(worker_python=WORKER_PYTHON) as worker:
        reset = worker.reset_batch([{"session_id": "legal-support", "spec": raw}])[0]
        assert reset["allowed_actions"] == ["N", "E", "S"]
        worker.abort_batch(["legal-support"])
        for index, program in enumerate(programs.values()):
            results.append(
                worker.execute_program(
                    session_id=f"route-{index}",
                    spec=raw,
                    actions=program.split(),
                )
            )
    assert all(result["success"] for result in results)
    assert all(result["validation_error"] is None for result in results)
    assert len({result["canonical_key"] for result in results}) == 3
