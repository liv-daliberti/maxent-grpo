from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oat_drgrpo.math_grader import (
    boxed_reward_fn,
    validated_modebench_exploration_identity,
    validated_modebench_outcome_key,
)
from oat_drgrpo.maze_modebench import (
    ANT_ACTIONS,
    ANT_MAZE_ACTION_VERSION,
    ANT_MAZE_VERIFIER,
    MAZE_ACTION_VERSION,
    POINT_ACTIONS,
    POINT_MAZE_VERIFIER,
    MazeModeBenchError,
    parse_maze_action_program,
    parse_maze_action_spec,
    validate_maze_execution,
)
from oat_drgrpo.maze_modebench_process import MazeVerifierProcess


ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _point_spec() -> dict:
    return {
        "verifier": POINT_MAZE_VERIFIER,
        "maze_action_version": MAZE_ACTION_VERSION,
        "environment_id": "PointMaze_Custom-v3",
        "environment_sha256": _sha("point-environment"),
        "controller_sha256": None,
        "map_id": "two_corridors_v1",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        "reset_cell": [3, 1],
        "goal_cell": [3, 5],
        "reset_seed": 43001,
        "min_actions": 2,
        "max_actions": 16,
        "action_repeat": 5,
        "action_tokens": list(POINT_ACTIONS),
        "success_threshold": 0.5,
        "max_segment_length": 1.1,
        "bounds_xy": [[-3.0, 3.0], [-2.0, 2.0]],
        "route_gates": [
            {
                "id": "upper",
                "axis": "x",
                "coordinate": 0.0,
                "span": [0.4, 1.6],
                "hysteresis": 0.1,
            },
            {
                "id": "lower",
                "axis": "x",
                "coordinate": 0.0,
                "span": [-1.6, -0.4],
                "hysteresis": 0.1,
            },
        ],
    }


def _execution(spec: dict, candidate: str, trajectory: list[list[float]], **updates):
    parsed = parse_maze_action_spec(spec)
    tokens = parse_maze_action_program(candidate, parsed)
    record = {
        "environment_sha256": parsed.environment_sha256,
        "controller_sha256": parsed.controller_sha256,
        "spec_sha256": parsed.spec_sha256,
        "reset_seed": parsed.reset_seed,
        "action_tokens": list(tokens),
        "success": True,
        "final_goal_distance": 0.2,
        "trajectory_xy": trajectory,
        "simulator_steps": min(10, len(tokens) * parsed.action_repeat),
    }
    record.update(updates)
    return record


UPPER_PATH = [
    [-2.0, 1.0],
    [-1.0, 1.0],
    [-0.2, 1.0],
    [-0.05, 1.0],
    [0.05, 1.0],
    [0.2, 1.0],
    [1.0, 1.0],
]
LOWER_PATH = [
    [-2.0, -1.0],
    [-1.0, -1.0],
    [-0.2, -1.0],
    [0.2, -1.0],
    [1.0, -1.0],
]


def test_point_action_program_collapses_only_delimiter_and_case_formatting():
    spec = parse_maze_action_spec(_point_spec())

    assert parse_maze_action_program("n, E  coast", spec) == ("N", "E", "COAST")
    assert parse_maze_action_program("N E COAST", spec) == ("N", "E", "COAST")
    with pytest.raises(MazeModeBenchError, match="unknown action token"):
        parse_maze_action_program("N JUMP", spec)
    with pytest.raises(MazeModeBenchError, match="length"):
        parse_maze_action_program("N", spec)


def test_successful_upper_and_lower_paths_have_distinct_verified_route_keys():
    spec = _point_spec()
    candidate = "E E"

    upper = validate_maze_execution(
        candidate,
        spec,
        _execution(spec, candidate, UPPER_PATH),
    )
    lower = validate_maze_execution(
        candidate,
        spec,
        _execution(spec, candidate, LOWER_PATH),
    )

    assert upper.directed_gates == ("upper+",)
    assert lower.directed_gates == ("lower+",)
    assert upper.canonical_key != lower.canonical_key
    assert upper.canonical_key.startswith(
        f"{POINT_MAZE_VERIFIER}:{MAZE_ACTION_VERSION}:two_corridors_v1:"
    )


def test_gate_hysteresis_collapses_centerline_jitter_without_inventing_modes():
    spec = _point_spec()
    candidate = "E E E"
    jittered = [
        [-0.3, 1.0],
        [-0.05, 1.0],
        [0.04, 1.0],
        [-0.04, 1.0],
        [0.06, 1.0],
        [0.3, 1.0],
    ]

    validation = validate_maze_execution(
        candidate,
        spec,
        _execution(spec, candidate, jittered),
    )

    assert validation.directed_gates == ("upper+",)


def test_route_gate_recrossing_is_rejected_as_a_non_simple_route():
    spec = _point_spec()
    candidate = "E E E"
    recrossing = [
        [-0.3, 1.0],
        [0.3, 1.0],
        [-0.3, 1.0],
        [0.3, 1.0],
    ]

    with pytest.raises(MazeModeBenchError, match="recrosses"):
        validate_maze_execution(
            candidate,
            spec,
            _execution(spec, candidate, recrossing),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"success": False}, "unsuccessful"),
        ({"final_goal_distance": 0.6}, "goal threshold"),
        ({"environment_sha256": _sha("wrong")}, "environment hash"),
        ({"reset_seed": 1}, "reset seed"),
        ({"action_tokens": ["W", "W"]}, "action program"),
    ],
)
def test_execution_identity_fails_closed(updates, message):
    spec = _point_spec()
    candidate = "E E"
    execution = _execution(spec, candidate, UPPER_PATH, **updates)

    with pytest.raises(MazeModeBenchError, match=message):
        validate_maze_execution(candidate, spec, execution)


def test_teleport_and_out_of_bounds_trajectories_are_rejected():
    spec = _point_spec()
    candidate = "E E"

    with pytest.raises(MazeModeBenchError, match="teleport"):
        validate_maze_execution(
            candidate,
            spec,
            _execution(spec, candidate, [[-2.0, 1.0], [0.2, 1.0]]),
        )
    with pytest.raises(MazeModeBenchError, match="outside maze bounds"):
        validate_maze_execution(
            candidate,
            spec,
            _execution(spec, candidate, [[-2.0, 1.0], [-2.5, 3.0]]),
        )


def test_ant_spec_requires_frozen_controller_and_exact_heading_alphabet():
    spec = _point_spec()
    spec.update(
        {
            "verifier": ANT_MAZE_VERIFIER,
            "maze_action_version": ANT_MAZE_ACTION_VERSION,
            "environment_id": "AntMaze_Custom-v5",
            "environment_sha256": _sha("ant-environment"),
            "controller_sha256": _sha("frozen-controller"),
            "action_tokens": list(ANT_ACTIONS),
        }
    )

    parsed = parse_maze_action_spec(spec)

    assert parsed.controller_sha256 == _sha("frozen-controller")
    assert parsed.action_tokens == ANT_ACTIONS
    assert "STOP" not in parsed.action_tokens
    spec["controller_sha256"] = None
    with pytest.raises(MazeModeBenchError, match="controller_sha256"):
        parse_maze_action_spec(spec)


def test_spec_hash_rejects_unbound_claim():
    spec = _point_spec()
    spec["spec_sha256"] = _sha("not-the-spec")

    with pytest.raises(MazeModeBenchError, match="canonical spec"):
        parse_maze_action_spec(spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reset_cell", [0, 0], "free maze cell"),
        ("goal_cell", [3, 1], "must differ"),
        ("maze_map", [[0] * 5 for _ in range(5)], "borders must be walls"),
    ],
)
def test_maze_geometry_contract_fails_closed(field, value, message):
    spec = _point_spec()
    spec[field] = value

    with pytest.raises(MazeModeBenchError, match=message):
        parse_maze_action_spec(spec)


def _runtime_point_spec() -> tuple[Path, dict]:
    worker_python = ROOT / "var/maze_runtime/venv/bin/python"
    if not worker_python.is_file():
        return worker_python, _point_spec()
    import subprocess

    identity = json.loads(
        subprocess.check_output(
            [
                str(worker_python),
                "-c",
                (
                    "import json,sys;sys.path.insert(0, chr(115)+chr(114)+chr(99));"
                    "from oat_drgrpo.maze_runtime_identity import maze_runtime_identity;"
                    "print(json.dumps(maze_runtime_identity()))"
                ),
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()[-1]
    )
    spec = _point_spec()
    spec.update(
        {
            "environment_id": "PointMaze_UMaze-v3",
            "environment_sha256": identity["point_environment_sha256"],
            "min_actions": 2,
            "max_actions": 64,
            "action_repeat": 5,
            "success_threshold": 0.45,
            "max_segment_length": 0.2,
            "bounds_xy": [[-3.5, 3.5], [-3.5, 3.5]],
            "route_gates": [
                {
                    "id": "upper",
                    "axis": "x",
                    "coordinate": 0.0,
                    "span": [0.4, 3.0],
                    "hysteresis": 0.1,
                },
                {
                    "id": "lower",
                    "axis": "x",
                    "coordinate": 0.0,
                    "span": [-3.0, -0.4],
                    "hysteresis": 0.1,
                },
            ],
        }
    )
    return worker_python, spec


def test_external_pointmaze_worker_executes_two_distinct_language_routes():
    worker_python, spec = _runtime_point_spec()
    if not worker_python.is_file():
        pytest.skip("pinned maze runtime is not installed")
    upper_program = " ".join(["N"] * 7 + ["E"] * 18 + ["S"] * 7)
    lower_program = " ".join(["S"] * 7 + ["E"] * 18 + ["N"] * 7)
    verifier = MazeVerifierProcess(worker_python=worker_python)
    try:
        upper = verifier.validate(upper_program, spec)
        lower = verifier.validate(lower_program, spec)

        assert upper is not None
        assert lower is not None
        assert upper.directed_gates == ("upper+",)
        assert lower.directed_gates == ("lower+",)
        assert upper.canonical_key != lower.canonical_key
        assert verifier._process is not None
    finally:
        verifier.close()


def test_pointmaze_reward_and_mode_key_come_from_same_external_execution_boundary(
    monkeypatch,
):
    worker_python, spec = _runtime_point_spec()
    if not worker_python.is_file():
        pytest.skip("pinned maze runtime is not installed")
    monkeypatch.setenv("OAT_ZERO_MAZE_WORKER_PYTHON", str(worker_python))
    program = " ".join(["N"] * 7 + ["E"] * 18 + ["S"] * 7)
    reference = json.dumps(spec, sort_keys=True)

    info, reward = boxed_reward_fn(program, reference)
    key = validated_modebench_outcome_key(program, reference)
    identity = validated_modebench_exploration_identity(program, reference)

    assert info == {"formatted": True}
    assert reward == 1.0
    assert key is not None and key.endswith(":upper+")
    assert identity is not None
    assert identity.endpoint_key.endswith(":goal:two_corridors_v1")
    assert identity.route_signature == key
