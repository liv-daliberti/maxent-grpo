"""Networkless JSON-lines worker for 0.5B maze action programs."""

from __future__ import annotations

import json
import math
import os
import signal
import sys
from typing import Any, Mapping

from .maze_modebench import (
    ANT_MAZE_VERIFIER,
    POINT_MAZE_VERIFIER,
    parse_maze_action_program,
    parse_maze_action_spec,
    validate_maze_execution,
)
from .maze_runtime_identity import maze_runtime_identity
from .ant_maze_worker_v5 import execute_ant_v5
from .ant_maze_worker_v5 import RECEIPT_SHA256 as ANT_V5_RECEIPT_SHA256
from .ant_maze_worker_v8 import execute_ant_v8
from .ant_maze_worker_v8 import RECEIPT_SHA256 as ANT_V8_RECEIPT_SHA256
from .ant_maze_worker_v9 import controller_receipt_sha256 as ant_v9_receipt_sha256
from .ant_maze_worker_v9 import execute_ant_v9
from .ant_maze_worker_v10 import controller_receipt_sha256 as ant_v10_receipt_sha256
from .ant_maze_worker_v10 import execute_ant_v10
from .ant_maze_worker_v11 import controller_receipt_sha256 as ant_v11_receipt_sha256
from .ant_maze_worker_v11 import execute_ant_v11
from .ant_maze_worker_v12 import controller_receipt_sha256 as ant_v12_receipt_sha256
from .ant_maze_worker_v12 import execute_ant_v12


WORKER_TIMEOUT_SECONDS = float(
    os.environ.get("OAT_ZERO_MAZE_WORKER_TIMEOUT_SECONDS", "20.0")
)
if not math.isfinite(WORKER_TIMEOUT_SECONDS) or WORKER_TIMEOUT_SECONDS <= 0:
    raise ValueError("maze worker timeout must be finite and positive")


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("maze action-program execution timed out")


def _point_action(token: str):
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
        "COAST": (0.0, 0.0),
    }
    return np.asarray(values[token], dtype=np.float32)


def _execute_point(candidate: str, raw_spec: Mapping[str, Any]):
    import gymnasium as gym
    import gymnasium_robotics
    import numpy as np

    gym.register_envs(gymnasium_robotics)
    spec = parse_maze_action_spec(raw_spec)
    if spec.verifier != POINT_MAZE_VERIFIER:
        raise ValueError("PointMaze executor received a non-PointMaze spec")
    runtime = maze_runtime_identity()
    if spec.environment_sha256 != runtime["point_environment_sha256"]:
        raise ValueError("PointMaze environment hash differs from installed runtime")
    tokens = parse_maze_action_program(candidate, spec)
    env = gym.make(
        spec.environment_id,
        maze_map=[list(row) for row in spec.maze_map],
        reward_type="sparse",
        continuing_task=False,
        reset_target=False,
        max_episode_steps=len(tokens) * spec.action_repeat,
    )
    try:
        # Gymnasium-Robotics 1.4.2 forwards this constructor argument to the
        # nested PointEnv, so bind it on the MazeEnv before reset instead.
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
        for token in tokens:
            action = _point_action(token)
            for _ in range(spec.action_repeat):
                observation, _reward, terminated, truncated, info = env.step(action)
                simulator_steps += 1
                achieved = observation["achieved_goal"].astype(float)
                trajectory.append(achieved.tolist())
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
        "controller_sha256": None,
        "spec_sha256": spec.spec_sha256,
        "reset_seed": spec.reset_seed,
        "action_tokens": list(tokens),
        "success": success,
        "final_goal_distance": distance,
        "trajectory_xy": trajectory,
        "simulator_steps": simulator_steps,
    }
    return validate_maze_execution(candidate, raw_spec, execution), execution


def execute(candidate: str, raw_spec: Mapping[str, Any]):
    spec = parse_maze_action_spec(raw_spec)
    if spec.verifier == POINT_MAZE_VERIFIER:
        return _execute_point(candidate, raw_spec)
    if spec.verifier == ANT_MAZE_VERIFIER:
        if spec.controller_sha256 == ANT_V5_RECEIPT_SHA256:
            return execute_ant_v5(candidate, raw_spec)
        if spec.controller_sha256 == ANT_V8_RECEIPT_SHA256:
            return execute_ant_v8(candidate, raw_spec)
        # V12 route identities are intentionally incompatible with the older
        # controller-specific route schemas.  Resolve the newest bound
        # controller before asking any older validator to inspect that identity.
        if spec.controller_sha256 == ant_v12_receipt_sha256():
            return execute_ant_v12(candidate, raw_spec)
        if spec.controller_sha256 == ant_v9_receipt_sha256():
            return execute_ant_v9(candidate, raw_spec)
        if spec.controller_sha256 == ant_v10_receipt_sha256():
            return execute_ant_v10(candidate, raw_spec)
        if spec.controller_sha256 == ant_v11_receipt_sha256():
            return execute_ant_v11(candidate, raw_spec)
        raise ValueError("unsupported AntMaze controller identity")
    raise ValueError("unsupported maze verifier")


def main() -> None:
    signal.signal(signal.SIGALRM, _timeout)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            signal.setitimer(signal.ITIMER_REAL, WORKER_TIMEOUT_SECONDS)
            validation, execution = execute(request["candidate"], request["spec"])
            payload = {
                "valid": True,
                "canonical_key": validation.canonical_key,
                "directed_gates": list(validation.directed_gates),
                "action_tokens": list(validation.action_tokens),
                "simulator_steps": validation.simulator_steps,
                "execution": execution,
            }
        except Exception as error:
            payload = {"valid": False, "error": type(error).__name__}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
