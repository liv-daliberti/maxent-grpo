"""Hash-bound AntMaze executor for the admitted v5 macro controller."""

from __future__ import annotations

import hashlib
import json
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
MODEL_PATH = ROOT / "var/maze_runtime/controllers/ant_heading_v1.zip"
RECEIPT_PATH = ROOT / "var/maze_runtime/controllers/ant_heading_v5.evaluation.json"
RECEIPT_SHA256 = (
    "324d3301b8e21b4dbbfcc2e6b9a87aba479bd2da5aa3a040cff24adebaf8828e"
)
MODEL_SHA256 = (
    "526b669bb14cf8a07b44a1c725f94411632a3ac8a8e84966db09336cc96e4b0f"
)
_MODEL = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def controller_identity() -> dict[str, Any]:
    """Load and verify the immutable v5 controller receipt."""

    if _sha256(RECEIPT_PATH) != RECEIPT_SHA256:
        raise RuntimeError("Ant v5 controller receipt hash mismatch")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    if receipt.get("status") != "pass":
        raise RuntimeError("Ant v5 controller is not admitted")
    if _sha256(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("Ant v5 controller model hash mismatch")
    if receipt.get("hashes", {}).get("model_sha256") != MODEL_SHA256:
        raise RuntimeError("Ant v5 receipt does not bind the controller model")
    return receipt


def _model():
    global _MODEL
    receipt = controller_identity()
    if _MODEL is None:
        from stable_baselines3 import PPO

        _MODEL = PPO.load(MODEL_PATH, device="cpu")
    return _MODEL, receipt["macro_identity"]


def execute_ant_v5_raw(
    candidate: str,
    raw_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one program and retain its raw record even on goal failure."""

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
        raise ValueError("AntMaze controller hash differs from frozen v5 receipt")
    tokens = parse_maze_action_program(candidate, spec)
    model, identity = _model()
    names = tuple(str(value) for value in identity["command_names"])
    commands = np.asarray(identity["commands"], dtype=np.float32)
    macros = tuple(
        tuple(int(index) for index in macro) for macro in identity["macros"]
    )
    token_to_index = {name: index for index, name in enumerate(names)}
    if tuple(token_to_index) != spec.action_tokens:
        raise RuntimeError("Ant v5 macro alphabet differs from the maze spec")

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
            macro = macros[token_to_index[token]]
            for primitive_step in range(spec.action_repeat):
                segment = min(
                    primitive_step * len(macro) // spec.action_repeat,
                    len(macro) - 1,
                )
                command = commands[macro[segment]]
                controller_input = np.concatenate(
                    [
                        np.asarray(observation["observation"], dtype=np.float32),
                        command,
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
                success = bool(info.get("success", False))
                if success or terminated or truncated:
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
    execution = {
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
    return execution


def execute_ant_v5(candidate: str, raw_spec: Mapping[str, Any]):
    """Execute and validate one program in the pinned AntMaze runtime."""

    execution = execute_ant_v5_raw(candidate, raw_spec)
    return validate_maze_execution(candidate, raw_spec, execution), execution
