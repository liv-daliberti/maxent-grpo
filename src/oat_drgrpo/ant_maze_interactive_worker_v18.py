"""Persistent trusted AntMaze v18 worker for one-token language decisions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
from typing import Any, Mapping

from . import ant_maze_worker_v18 as v18
from .maze_modebench import (
    ANT_MAZE_VERIFIER,
    parse_maze_action_spec,
    validate_maze_execution,
)
from .maze_runtime_identity import maze_runtime_identity


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("interactive AntMaze v18 worker request timed out")


def _without_protocol_identity(function):
    identity = os.environ.pop("OAT_ZERO_PROTOCOL_IDENTITY", None)
    try:
        return function()
    finally:
        if identity is not None:
            os.environ["OAT_ZERO_PROTOCOL_IDENTITY"] = identity


class InteractiveAntWorkerV18:
    def __init__(self) -> None:
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        self.gym = gym
        self.runtime = maze_runtime_identity()
        self.model = _without_protocol_identity(v18._model)
        self.controller_sha256 = _without_protocol_identity(
            v18.controller_receipt_sha256
        )
        self.sessions: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        for session in self.sessions.values():
            session["env"].close()
        self.sessions.clear()

    @staticmethod
    def _observation(
        session_id: str,
        session: dict[str, Any],
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> dict[str, Any]:
        env = session["env"]
        spec = session["spec"]
        return {
            "session_id": session_id,
            "achieved_goal": observation["achieved_goal"].astype(float).tolist(),
            "desired_goal": observation["desired_goal"].astype(float).tolist(),
            "velocity_xy": env.unwrapped.data.qvel[:2].astype(float).tolist(),
            "success": bool(info.get("success", False)),
            "remaining_actions": spec.max_actions - len(session["tokens"]),
        }

    def reset(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        session_id = str(raw["session_id"])
        if not session_id or session_id in self.sessions:
            raise ValueError("interactive AntMaze v18 session ID is empty or duplicated")
        raw_spec = raw["spec"]
        spec = parse_maze_action_spec(raw_spec)
        if spec.verifier != ANT_MAZE_VERIFIER:
            raise ValueError("interactive v18 worker received a non-AntMaze spec")
        if spec.environment_sha256 != self.runtime["ant_environment_sha256"]:
            raise ValueError("AntMaze environment hash differs from runtime")
        if spec.controller_sha256 != self.controller_sha256:
            raise ValueError("AntMaze controller hash differs from v18")
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
        initial = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32).copy()
        session = {
            "env": env,
            "raw_spec": raw_spec,
            "spec": spec,
            "tokens": [],
            "trajectory": [observation["achieved_goal"].astype(float).tolist()],
            "steps": 0,
            "done": False,
            "initial": initial,
            "cumulative_heading": np.zeros(2, dtype=np.float32),
            "nominal_targets": [],
            "segment_steps": [],
            "segment_arrival_speeds": [],
            "observation": observation,
        }
        self.sessions[session_id] = session
        return self._observation(session_id, session, observation, info)

    def step(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        session_id = str(raw["session_id"])
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("unknown interactive AntMaze v18 session")
        if session["done"]:
            raise ValueError("interactive AntMaze v18 session is terminal")
        spec = session["spec"]
        token = str(raw["action"]).upper()
        if token not in spec.action_tokens:
            raise ValueError("action is outside the frozen AntMaze alphabet")
        if len(session["tokens"]) >= spec.max_actions:
            raise ValueError("interactive AntMaze v18 horizon is exhausted")

        env = session["env"]
        session["tokens"].append(token)
        session["cumulative_heading"] += v18._heading(token)
        target = (
            session["initial"]
            + v18.WAYPOINT_DISTANCE * session["cumulative_heading"]
        )
        session["nominal_targets"].append(target.astype(float).tolist())
        before = session["steps"]
        environment_success = terminated = truncated = False
        waypoint_reached = False
        final_planar_speed = float(
            np.linalg.norm(np.asarray(env.unwrapped.data.qvel[:2]))
        )
        info: dict[str, Any] = {}
        observation = session["observation"]
        for _ in range(spec.action_repeat):
            current = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32)
            relative = np.clip(
                (target - current) / v18.WAYPOINT_DISTANCE,
                -1.0,
                1.0,
            )
            controller_input = np.concatenate(
                [
                    np.asarray(observation["observation"], dtype=np.float32),
                    relative,
                ]
            )
            action, _ = self.model.predict(controller_input, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(action)
            session["observation"] = observation
            session["steps"] += 1
            session["trajectory"].append(
                observation["achieved_goal"].astype(float).tolist()
            )
            current = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32)
            final_planar_speed = float(
                np.linalg.norm(
                    np.asarray(env.unwrapped.data.qvel[:2], dtype=np.float32)
                )
            )
            within_radius = float(np.linalg.norm(target - current)) <= (
                v18.WAYPOINT_SUCCESS_THRESHOLD
            )
            waypoint_reached = (
                within_radius
                and final_planar_speed <= v18.STABLE_PLANAR_SPEED
            )
            environment_success = bool(info.get("success", False))
            if waypoint_reached or terminated or truncated:
                break

        session["segment_steps"].append(session["steps"] - before)
        if waypoint_reached:
            session["segment_arrival_speeds"].append(final_planar_speed)
        distance = float(
            np.linalg.norm(
                observation["achieved_goal"] - observation["desired_goal"]
            )
        )
        horizon = len(session["tokens"]) >= spec.max_actions
        stable_route_failure = not waypoint_reached
        success = waypoint_reached and environment_success
        done = bool(success or stable_route_failure or terminated or truncated or horizon)
        canonical_key = None
        directed_gates: list[str] = []
        validation_error = None
        if done:
            session["done"] = True
            if success and len(session["tokens"]) >= spec.min_actions:
                execution = {
                    "environment_sha256": spec.environment_sha256,
                    "controller_sha256": spec.controller_sha256,
                    "spec_sha256": spec.spec_sha256,
                    "reset_seed": spec.reset_seed,
                    "action_tokens": list(session["tokens"]),
                    "targeting_version": v18.TARGETING_VERSION,
                    "waypoint_success_threshold": (
                        v18.WAYPOINT_SUCCESS_THRESHOLD
                    ),
                    "stable_planar_speed_threshold": v18.STABLE_PLANAR_SPEED,
                    "segment_arrival_speeds": session[
                        "segment_arrival_speeds"
                    ],
                    "nominal_targets_xy": session["nominal_targets"],
                    "segment_steps": session["segment_steps"],
                    "success": True,
                    "final_goal_distance": distance,
                    "final_planar_speed": final_planar_speed,
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
        public = self._observation(session_id, session, observation, info)
        response = {
            **public,
            "done": done,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "horizon_exhausted": bool(horizon and not success),
            "stable_route_failure": bool(stable_route_failure),
            "final_goal_distance": distance,
            "final_planar_speed": final_planar_speed,
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
    worker = InteractiveAntWorkerV18()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                signal.setitimer(signal.ITIMER_REAL, 600.0)
                command = request.get("command")
                if command == "reset_batch":
                    payload = {
                        "ok": True,
                        "results": [
                            worker.reset(row) for row in request["sessions"]
                        ],
                    }
                elif command == "step_batch":
                    payload = {
                        "ok": True,
                        "results": [
                            worker.step(row) for row in request["steps"]
                        ],
                    }
                elif command == "close":
                    worker.close()
                    payload = {"ok": True}
                else:
                    raise ValueError("unsupported interactive AntMaze v18 command")
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
