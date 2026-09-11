"""Hash-bound AntMaze executor for the admitted v8 waypoint controller."""

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
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()
MODEL_PATH = ROOT / "var/maze_runtime/controllers/ant_waypoint_v8.zip"
RECEIPT_PATH = (
    ROOT / "var/maze_runtime/controllers/ant_waypoint_v8.evaluation.json"
)
RECEIPT_SHA256 = (
    "fc2961d74a1fdfc8f3d2c1936ea61041cff65333ada0e6ebcd571b44919b18f1"
)
MODEL_SHA256 = (
    "77e780dfff1147bc2f542c1761f6efeaa2305fd5ddb625244ebf8e82b3fa871d"
)
WAYPOINT_DISTANCE = 4.0
WAYPOINT_SUCCESS_THRESHOLD = 0.45
_MODEL = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def controller_identity() -> dict[str, Any]:
    """Load and verify the immutable v8 controller receipt and model."""

    if _sha256(RECEIPT_PATH) != RECEIPT_SHA256:
        raise RuntimeError("Ant v8 controller receipt hash mismatch")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "pass"
        or receipt.get("decision") != "admitted_to_fresh_maze_route_gate_v8"
    ):
        raise RuntimeError("Ant v8 controller is not admitted")
    if _sha256(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("Ant v8 controller model hash mismatch")
    if receipt.get("hashes", {}).get("model_sha256") != MODEL_SHA256:
        raise RuntimeError("Ant v8 receipt does not bind the controller model")
    if receipt.get("training_waypoint_distances") != [WAYPOINT_DISTANCE]:
        raise RuntimeError("Ant v8 receipt has a different waypoint distance")
    if receipt.get("success_threshold") != WAYPOINT_SUCCESS_THRESHOLD:
        raise RuntimeError("Ant v8 receipt has a different waypoint threshold")
    return receipt


def _model():
    global _MODEL
    controller_identity()
    if _MODEL is None:
        from stable_baselines3 import PPO

        _MODEL = PPO.load(MODEL_PATH, device="cpu")
    return _MODEL


def _heading(token: str):
    import numpy as np

    diagonal = 1.0 / math.sqrt(2.0)
    values = {
        "N": (0.0, 1.0),
        "NE": (diagonal, diagonal),
        "E": (1.0, 0.0),
        "SE": (diagonal, -diagonal),
        "S": (0.0, -1.0),
        "SW": (-diagonal, -diagonal),
        "W": (-1.0, 0.0),
        "NW": (-diagonal, diagonal),
    }
    return np.asarray(values[token], dtype=np.float32)


def execute_ant_v8_raw(
    candidate: str,
    raw_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute commands as successive four-unit relative waypoint targets."""

    import gymnasium as gym
    import gymnasium_robotics
    import numpy as np

    gym.register_envs(gymnasium_robotics)
    spec = parse_maze_action_spec(raw_spec)
    if spec.verifier != ANT_MAZE_VERIFIER:
        raise ValueError("AntMaze executor received a non-AntMaze spec")
    runtime = maze_runtime_identity()
    if spec.environment_sha256 != runtime["ant_environment_sha256"]:
        raise ValueError("AntMaze environment hash differs from installed runtime")
    if spec.controller_sha256 != RECEIPT_SHA256:
        raise ValueError("AntMaze controller hash differs from frozen v8 receipt")
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
    try:
        env.unwrapped.position_noise_range = 0.0
        observation, info = env.reset(
            seed=spec.reset_seed,
            options={
                "reset_cell": list(spec.reset_cell),
                "goal_cell": list(spec.goal_cell),
            },
        )
        trajectory = [observation["achieved_goal"].astype(float).tolist()]
        success = bool(info.get("success", False))
        simulator_steps = 0
        terminated = truncated = False
        for token in tokens:
            current = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32)
            target = current + WAYPOINT_DISTANCE * _heading(token)
            waypoint_reached = False
            for _ in range(spec.action_repeat):
                current = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32)
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
                observation, _reward, terminated, truncated, info = env.step(action)
                simulator_steps += 1
                trajectory.append(
                    observation["achieved_goal"].astype(float).tolist()
                )
                success = bool(info.get("success", False))
                current = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32)
                waypoint_reached = (
                    float(np.linalg.norm(target - current))
                    <= WAYPOINT_SUCCESS_THRESHOLD
                )
                if success or waypoint_reached or terminated or truncated:
                    break
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
        "success": success,
        "final_goal_distance": distance,
        "trajectory_xy": trajectory,
        "simulator_steps": simulator_steps,
    }


def execute_ant_v8(candidate: str, raw_spec: Mapping[str, Any]):
    """Execute and validate one v8 program in the pinned AntMaze runtime."""

    execution = execute_ant_v8_raw(candidate, raw_spec)
    return validate_maze_execution(candidate, raw_spec, execution), execution
