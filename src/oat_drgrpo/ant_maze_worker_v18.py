"""Stable-handoff AntMaze executor for the prospectively gated v18 controller."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .maze_modebench import (
    ANT_MAZE_VERIFIER,
    parse_maze_action_program,
    parse_maze_action_spec,
    validate_maze_execution,
)
from .maze_runtime_identity import maze_runtime_identity


ROOT = Path(
    os.environ.get(
        "OAT_ZERO_REPO_ROOT",
        Path(__file__).resolve().parents[2],
    )
)
MODEL_PATH = ROOT / "var/maze_runtime/controllers/ant_stable_handoff_v18.zip"
RECEIPT_PATH = (
    ROOT
    / "var/maze_runtime/controllers/ant_stable_handoff_v18.evaluation.json"
)
TRAINING_IDENTITY_PATH = (
    ROOT / "var/artifacts/ant_stable_handoff_controller_v18_identity.json"
)
WAYPOINT_DISTANCE = 4.0
WAYPOINT_SUCCESS_THRESHOLD = 0.45
STABLE_PLANAR_SPEED = 1.0
TARGETING_VERSION = "initial-grid-cumulative-stable-handoff-v18"
EXPECTED_CONTROLLER_JOB = 30205570
_MODEL = None


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
        "controller_training_identity_sha256": _sha256(
            TRAINING_IDENTITY_PATH
        ),
        "waypoint_distance": WAYPOINT_DISTANCE,
        "waypoint_success_threshold": WAYPOINT_SUCCESS_THRESHOLD,
        "stable_planar_speed": STABLE_PLANAR_SPEED,
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
        receipt.get("schema_version")
        != "ant-waypoint-controller-v18-evaluation-v1"
        or receipt.get("status") != "pass"
        or receipt.get("decision")
        != "admitted_to_fresh_maze_route_gate_v18"
        or receipt.get("seed") != 73018
        or receipt.get("timesteps") != 6_000_000
        or receipt.get("workers") != 8
        or receipt.get("learning_rate") != 2e-7
    ):
        raise RuntimeError("Ant v18 worker requires the passing frozen v18 gate")
    if (
        training_identity.get("schema_version")
        != "ant-stable-handoff-controller-v18-identity-v1"
        or training_identity.get("job_id") != EXPECTED_CONTROLLER_JOB
        or training_identity.get("seed") != 73018
        or training_identity.get("timesteps") != 6_000_000
        or training_identity.get("development_episode_count") != 96
        or training_identity.get("stable_planar_speed")
        != STABLE_PLANAR_SPEED
    ):
        raise RuntimeError("Ant v18 training identity drift")
    checks = receipt.get("checks", {})
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("Ant v18 worker received a failed controller check")
    summary = receipt.get("evaluation", {}).get("summary", {})
    maximum_arrival_speed = summary.get("maximum_arrival_speed")
    if (
        summary.get("episodes") != 96
        or maximum_arrival_speed is None
        or float(maximum_arrival_speed) > STABLE_PLANAR_SPEED + 1e-6
    ):
        raise RuntimeError("Ant v18 stable-arrival gate drift")
    hashes = receipt.get("hashes", {})
    if hashes.get("model_sha256") != _sha256(MODEL_PATH):
        raise RuntimeError("Ant v18 receipt does not bind its model")
    if (
        hashes.get("training_source_sha256")
        != training_identity.get("trainer_sha256")
        or hashes.get("initial_model_sha256")
        != training_identity.get("initial_model_sha256")
    ):
        raise RuntimeError("Ant v18 receipt does not bind its frozen training")
    if (
        receipt.get("training_waypoint_distances") != [WAYPOINT_DISTANCE]
        or receipt.get("waypoint_distance") != WAYPOINT_DISTANCE
        or receipt.get("success_threshold") != WAYPOINT_SUCCESS_THRESHOLD
    ):
        raise RuntimeError("Ant v18 waypoint contract drift")
    route_identity_path = os.environ.get("OAT_ZERO_PROTOCOL_IDENTITY")
    if route_identity_path:
        route_identity = json.loads(
            Path(route_identity_path).read_text(encoding="utf-8")
        )
        expected = _executor_identity()
        if route_identity.get("schema_version") != (
            "ant-maze-v18-stable-route-generation-identity-v1"
        ):
            raise RuntimeError("Ant v18 route identity schema mismatch")
        for key, value in expected.items():
            if route_identity.get(key) != value:
                raise RuntimeError(
                    f"Ant v18 route identity does not bind {key}"
                )
    return receipt


def _model():
    global _MODEL
    if _MODEL is None:
        from stable_baselines3 import PPO

        controller_identity()
        _MODEL = PPO.load(MODEL_PATH, device="cpu")
    return _MODEL


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


def execute_ant_v18_raw(
    candidate: str,
    raw_spec: Mapping[str, Any],
) -> dict[str, Any]:
    import gymnasium as gym
    import gymnasium_robotics
    import numpy as np

    gym.register_envs(gymnasium_robotics)
    spec = parse_maze_action_spec(raw_spec)
    if spec.verifier != ANT_MAZE_VERIFIER:
        raise ValueError("AntMaze v18 received a non-AntMaze spec")
    runtime = maze_runtime_identity()
    if spec.environment_sha256 != runtime["ant_environment_sha256"]:
        raise ValueError("AntMaze environment hash differs from installed runtime")
    if spec.controller_sha256 != controller_receipt_sha256():
        raise ValueError("AntMaze controller hash differs from frozen v18 executor")
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
    segment_steps: list[int] = []
    segment_arrival_speeds: list[float] = []
    nominal_targets: list[list[float]] = []
    success = False
    try:
        env.unwrapped.position_noise_range = 0.0
        observation, _info = env.reset(
            seed=spec.reset_seed,
            options={
                "reset_cell": list(spec.reset_cell),
                "goal_cell": list(spec.goal_cell),
            },
        )
        initial = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32).copy()
        cumulative_heading = np.zeros(2, dtype=np.float32)
        trajectory = [observation["achieved_goal"].astype(float).tolist()]
        simulator_steps = 0
        terminated = truncated = False
        final_planar_speed = float(
            np.linalg.norm(np.asarray(env.unwrapped.data.qvel[:2]))
        )
        for token_index, token in enumerate(tokens):
            cumulative_heading += _heading(token)
            target = initial + WAYPOINT_DISTANCE * cumulative_heading
            nominal_targets.append(target.astype(float).tolist())
            before_segment = simulator_steps
            waypoint_reached = False
            environment_success = False
            for _ in range(spec.action_repeat):
                current = np.asarray(
                    env.unwrapped.data.qpos[:2], dtype=np.float32
                )
                relative = np.clip(
                    (target - current) / WAYPOINT_DISTANCE,
                    -1.0,
                    1.0,
                )
                controller_input = np.concatenate(
                    [
                        np.asarray(observation["observation"], dtype=np.float32),
                        relative,
                    ]
                )
                action, _ = model.predict(controller_input, deterministic=True)
                observation, _reward, terminated, truncated, info = env.step(
                    action
                )
                simulator_steps += 1
                trajectory.append(
                    observation["achieved_goal"].astype(float).tolist()
                )
                current = np.asarray(
                    env.unwrapped.data.qpos[:2], dtype=np.float32
                )
                final_planar_speed = float(
                    np.linalg.norm(
                        np.asarray(env.unwrapped.data.qvel[:2], dtype=np.float32)
                    )
                )
                within_radius = float(np.linalg.norm(target - current)) <= (
                    WAYPOINT_SUCCESS_THRESHOLD
                )
                waypoint_reached = (
                    within_radius
                    and final_planar_speed <= STABLE_PLANAR_SPEED
                )
                environment_success = bool(info.get("success", False))
                if waypoint_reached or terminated or truncated:
                    break
            segment_steps.append(simulator_steps - before_segment)
            if waypoint_reached:
                segment_arrival_speeds.append(final_planar_speed)
            final_token = token_index == len(tokens) - 1
            if final_token:
                success = waypoint_reached and environment_success
            if not waypoint_reached or terminated or truncated:
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
        "waypoint_success_threshold": WAYPOINT_SUCCESS_THRESHOLD,
        "stable_planar_speed_threshold": STABLE_PLANAR_SPEED,
        "segment_arrival_speeds": segment_arrival_speeds,
        "nominal_targets_xy": nominal_targets,
        "segment_steps": segment_steps,
        "success": success,
        "final_goal_distance": distance,
        "final_planar_speed": final_planar_speed,
        "trajectory_xy": trajectory,
        "simulator_steps": simulator_steps,
    }


def execute_ant_v18(candidate: str, raw_spec: Mapping[str, Any]):
    execution = execute_ant_v18_raw(candidate, raw_spec)
    return validate_maze_execution(candidate, raw_spec, execution), execution
