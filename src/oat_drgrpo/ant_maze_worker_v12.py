"""Grid-anchored AntMaze executor for the admitted v11 controller."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from . import ant_maze_worker_v10 as executor
from .maze_modebench import (
    ANT_MAZE_VERIFIER,
    parse_maze_action_program,
    parse_maze_action_spec,
    validate_maze_execution,
)
from .maze_runtime_identity import maze_runtime_identity


ROOT = executor.ROOT
MODEL_PATH = ROOT / "var/maze_runtime/controllers/ant_waypoint_v11.zip"
RECEIPT_PATH = (
    ROOT / "var/maze_runtime/controllers/ant_waypoint_v11.evaluation.json"
)
TRAINING_IDENTITY_PATH = (
    ROOT / "var/artifacts/ant_waypoint_controller_v11_identity.json"
)
WAYPOINT_DISTANCE = 4.0
WAYPOINT_SUCCESS_THRESHOLD = 0.45
TARGETING_VERSION = "initial-grid-cumulative-v12"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _executor_identity() -> dict[str, Any]:
    return {
        "targeting_version": TARGETING_VERSION,
        "worker_source_sha256": _sha256(Path(__file__)),
        "controller_receipt_sha256": _sha256(RECEIPT_PATH),
        "controller_model_sha256": _sha256(MODEL_PATH),
        "controller_training_identity_sha256": _sha256(TRAINING_IDENTITY_PATH),
        "waypoint_distance": WAYPOINT_DISTANCE,
        "waypoint_success_threshold": WAYPOINT_SUCCESS_THRESHOLD,
    }


def controller_receipt_sha256() -> str:
    controller_identity()
    return _canonical_sha256(_executor_identity())


def controller_identity() -> dict[str, Any]:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    training_identity = json.loads(
        TRAINING_IDENTITY_PATH.read_text(encoding="utf-8")
    )
    if (
        receipt.get("status") != "pass"
        or receipt.get("decision") != "admitted_to_fresh_maze_route_gate_v11"
        or receipt.get("seed") != 73011
        or receipt.get("timesteps") != 2_000_000
    ):
        raise RuntimeError("Ant v12 requires the exact admitted v11 controller")
    if (
        training_identity.get("job_id") != 30198291
        or training_identity.get("seed") != 73011
    ):
        raise RuntimeError("Ant v12 training identity differs from admitted v11")
    checks = receipt.get("checks", {})
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("Ant v12 received a failed v11 controller check")
    if receipt.get("hashes", {}).get("model_sha256") != _sha256(MODEL_PATH):
        raise RuntimeError("Ant v12 receipt does not bind the v11 model")
    if (
        receipt.get("training_waypoint_distances") != [WAYPOINT_DISTANCE]
        or receipt.get("waypoint_distance") != WAYPOINT_DISTANCE
        or receipt.get("success_threshold") != WAYPOINT_SUCCESS_THRESHOLD
    ):
        raise RuntimeError("Ant v12 local waypoint contract drift")
    route_identity_path = os.environ.get("OAT_ZERO_PROTOCOL_IDENTITY")
    if route_identity_path:
        route_identity = json.loads(
            Path(route_identity_path).read_text(encoding="utf-8")
        )
        expected = _executor_identity()
        if route_identity.get("schema_version") != (
            "ant-maze-v12-route-generation-identity-v1"
        ):
            raise RuntimeError("Ant v12 route identity schema mismatch")
        for key, value in expected.items():
            if route_identity.get(key) != value:
                raise RuntimeError(f"Ant v12 route identity does not bind {key}")
    return receipt


def _model():
    executor.MODEL_PATH = MODEL_PATH
    executor.RECEIPT_PATH = RECEIPT_PATH
    executor.TRAINING_IDENTITY_PATH = TRAINING_IDENTITY_PATH
    executor.controller_identity = controller_identity
    executor.controller_receipt_sha256 = controller_receipt_sha256
    return executor._model()


def _heading(token: str):
    import numpy as np

    diagonal = 1.0 / math.sqrt(2.0)
    return np.asarray(
        {
            "N": (0.0, 1.0),
            "NE": (diagonal, diagonal),
            "E": (1.0, 0.0),
            "SE": (diagonal, -diagonal),
            "S": (0.0, -1.0),
            "SW": (-diagonal, -diagonal),
            "W": (-1.0, 0.0),
            "NW": (-diagonal, diagonal),
        }[token],
        dtype=np.float32,
    )


def execute_ant_v12_raw(candidate: str, raw_spec: Mapping[str, Any]) -> dict[str, Any]:
    import gymnasium as gym
    import gymnasium_robotics
    import numpy as np

    gym.register_envs(gymnasium_robotics)
    spec = parse_maze_action_spec(raw_spec)
    if spec.verifier != ANT_MAZE_VERIFIER:
        raise ValueError("AntMaze v12 received a non-AntMaze spec")
    runtime = maze_runtime_identity()
    if spec.environment_sha256 != runtime["ant_environment_sha256"]:
        raise ValueError("AntMaze environment hash differs from installed runtime")
    if spec.controller_sha256 != controller_receipt_sha256():
        raise ValueError("AntMaze controller hash differs from frozen v12 executor")
    tokens = parse_maze_action_program(candidate, spec)
    model = _model()
    env = gym.make(
        spec.environment_id,
        maze_map=[list(row) for row in spec.maze_map],
        reward_type="sparse",
        continuing_task=False,
        reset_target=False,
        max_episode_steps=len(tokens) * spec.action_repeat,
    )
    segment_steps = []
    nominal_targets = []
    try:
        env.unwrapped.position_noise_range = 0.0
        observation, info = env.reset(
            seed=spec.reset_seed,
            options={
                "reset_cell": list(spec.reset_cell),
                "goal_cell": list(spec.goal_cell),
            },
        )
        initial = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32).copy()
        cumulative_heading = np.zeros(2, dtype=np.float32)
        trajectory = [observation["achieved_goal"].astype(float).tolist()]
        success = bool(info.get("success", False))
        simulator_steps = 0
        terminated = truncated = False
        for token in tokens:
            cumulative_heading += _heading(token)
            target = initial + WAYPOINT_DISTANCE * cumulative_heading
            nominal_targets.append(target.astype(float).tolist())
            before_segment = simulator_steps
            for _ in range(spec.action_repeat):
                current = np.asarray(
                    env.unwrapped.data.qpos[:2], dtype=np.float32
                )
                relative = np.clip(
                    (target - current) / WAYPOINT_DISTANCE, -1.0, 1.0
                )
                controller_input = np.concatenate(
                    [
                        np.asarray(observation["observation"], dtype=np.float32),
                        relative,
                    ]
                )
                action, _ = model.predict(controller_input, deterministic=True)
                observation, _reward, terminated, truncated, info = env.step(action)
                simulator_steps += 1
                trajectory.append(
                    observation["achieved_goal"].astype(float).tolist()
                )
                success = bool(info.get("success", False))
                current = np.asarray(
                    env.unwrapped.data.qpos[:2], dtype=np.float32
                )
                waypoint_reached = float(np.linalg.norm(target - current)) <= (
                    WAYPOINT_SUCCESS_THRESHOLD
                )
                if success or waypoint_reached or terminated or truncated:
                    break
            segment_steps.append(simulator_steps - before_segment)
            if success or terminated or truncated:
                break
        distance = float(
            np.linalg.norm(
                observation["achieved_goal"] - observation["desired_goal"]
            )
        )
    finally:
        env.close()
    return {
        "environment_sha256": spec.environment_sha256,
        "controller_sha256": spec.controller_sha256,
        "spec_sha256": spec.spec_sha256,
        "reset_seed": spec.reset_seed,
        "action_tokens": list(tokens),
        "targeting_version": TARGETING_VERSION,
        "nominal_targets_xy": nominal_targets,
        "segment_steps": segment_steps,
        "success": success,
        "final_goal_distance": distance,
        "trajectory_xy": trajectory,
        "simulator_steps": simulator_steps,
    }


def execute_ant_v12(candidate: str, raw_spec: Mapping[str, Any]):
    execution = execute_ant_v12_raw(candidate, raw_spec)
    return validate_maze_execution(candidate, raw_spec, execution), execution
