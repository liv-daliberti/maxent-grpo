"""Persistent networkless PointMaze worker for stepwise language policies."""

from __future__ import annotations

import json
import math
import signal
import sys
from typing import Any, Mapping

from .maze_modebench import (
    POINT_MAZE_VERIFIER,
    parse_maze_action_spec,
    validate_maze_execution,
)
from .maze_runtime_identity import maze_runtime_identity


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("interactive PointMaze worker request timed out")


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


class InteractivePointWorker:
    """Own simulator sessions without exposing simulator objects to the LM."""

    def __init__(self) -> None:
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        self.gym = gym
        self.sessions: dict[str, dict[str, Any]] = {}
        self.runtime = maze_runtime_identity()

    def close(self) -> None:
        for session in self.sessions.values():
            session["env"].close()
        self.sessions.clear()

    def reset(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        session_id = str(raw["session_id"])
        if not session_id or session_id in self.sessions:
            raise ValueError("interactive session ID is empty or duplicated")
        raw_spec = raw["spec"]
        spec = parse_maze_action_spec(raw_spec)
        if spec.verifier != POINT_MAZE_VERIFIER:
            raise ValueError("interactive worker received a non-PointMaze spec")
        if spec.environment_sha256 != self.runtime["point_environment_sha256"]:
            raise ValueError("PointMaze environment hash differs from runtime")
        env = self.gym.make(
            spec.environment_id,
            maze_map=[list(row) for row in spec.maze_map],
            reward_type="sparse",
            continuing_task=False,
            reset_target=False,
            max_episode_steps=spec.max_actions * spec.action_repeat,
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
        except BaseException:
            env.close()
            raise
        achieved = observation["achieved_goal"].astype(float).tolist()
        desired = observation["desired_goal"].astype(float).tolist()
        velocity = observation["observation"][2:4].astype(float).tolist()
        self.sessions[session_id] = {
            "env": env,
            "raw_spec": raw_spec,
            "spec": spec,
            "tokens": [],
            "trajectory": [achieved],
            "steps": 0,
            "done": False,
        }
        response = {
            "session_id": session_id,
            "achieved_goal": achieved,
            "desired_goal": desired,
            "velocity_xy": velocity,
            "success": bool(info.get("success", False)),
            "remaining_actions": spec.max_actions,
        }
        return response

    def step(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        session_id = str(raw["session_id"])
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("unknown interactive session")
        if session["done"]:
            raise ValueError("interactive session is already terminal")
        spec = session["spec"]
        token = str(raw["action"]).upper()
        if token not in spec.action_tokens:
            raise ValueError("action is outside the frozen PointMaze alphabet")
        if len(session["tokens"]) >= spec.max_actions:
            raise ValueError("interactive action horizon is exhausted")

        env = session["env"]
        session["tokens"].append(token)
        success = terminated = truncated = False
        for _ in range(spec.action_repeat):
            observation, _reward, terminated, truncated, info = env.step(
                _point_action(token)
            )
            session["steps"] += 1
            achieved_array = observation["achieved_goal"].astype(float)
            session["trajectory"].append(achieved_array.tolist())
            success = bool(info.get("success", False))
            if success or terminated or truncated:
                break

        achieved = observation["achieved_goal"].astype(float).tolist()
        desired = observation["desired_goal"].astype(float).tolist()
        velocity = observation["observation"][2:4].astype(float).tolist()
        distance = float(
            np.linalg.norm(
                observation["achieved_goal"] - observation["desired_goal"]
            )
        )
        horizon = len(session["tokens"]) >= spec.max_actions
        done = bool(success or terminated or truncated or horizon)
        canonical_key = None
        directed_gates: list[str] = []
        validation_error = None
        if done:
            session["done"] = True
            if success and len(session["tokens"]) >= spec.min_actions:
                execution = {
                    "environment_sha256": spec.environment_sha256,
                    "controller_sha256": None,
                    "spec_sha256": spec.spec_sha256,
                    "reset_seed": spec.reset_seed,
                    "action_tokens": list(session["tokens"]),
                    "success": True,
                    "final_goal_distance": distance,
                    "trajectory_xy": session["trajectory"],
                    "simulator_steps": session["steps"],
                }
                try:
                    validation = validate_maze_execution(
                        " ".join(session["tokens"]),
                        session["raw_spec"],
                        execution,
                    )
                    canonical_key = validation.canonical_key
                    directed_gates = list(validation.directed_gates)
                except Exception as error:
                    validation_error = f"{type(error).__name__}: {error}"
            env.close()

        response = {
            "session_id": session_id,
            "achieved_goal": achieved,
            "desired_goal": desired,
            "velocity_xy": velocity,
            "success": success,
            "done": done,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "horizon_exhausted": bool(horizon and not success),
            "final_goal_distance": distance,
            "remaining_actions": spec.max_actions - len(session["tokens"]),
            "action_count": len(session["tokens"]),
            "simulator_steps": session["steps"],
            "canonical_key": canonical_key,
            "directed_gates": directed_gates,
            "validation_error": validation_error,
        }
        if done:
            self.sessions.pop(session_id)
        return response


def main() -> None:
    signal.signal(signal.SIGALRM, _timeout)
    worker = InteractivePointWorker()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                signal.setitimer(signal.ITIMER_REAL, 60.0)
                command = request.get("command")
                if command == "reset_batch":
                    results = [worker.reset(row) for row in request["sessions"]]
                    payload = {"ok": True, "results": results}
                elif command == "step_batch":
                    results = [worker.step(row) for row in request["steps"]]
                    payload = {"ok": True, "results": results}
                elif command == "close":
                    worker.close()
                    payload = {"ok": True}
                else:
                    raise ValueError("unsupported interactive worker command")
            except Exception as error:
                payload = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
            sys.stdout.flush()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
