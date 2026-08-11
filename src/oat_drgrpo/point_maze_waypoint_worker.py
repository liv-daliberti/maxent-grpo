"""Persistent networkless worker for sequential PointMaze waypoint policies."""

from __future__ import annotations

import json
import signal
import sys
from typing import Any, Mapping

from .maze_runtime_identity import maze_runtime_identity
from .point_maze_waypoint import (
    POINT_WAYPOINT_VERIFIER,
    advance_point_waypoint_cell,
    legal_point_waypoint_actions,
    parse_point_waypoint_spec,
    validate_point_waypoint_execution,
)


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("interactive PointMaze waypoint request timed out")


class InteractivePointWaypointWorker:
    """Own MuJoCo sessions and expose only public, auditable state."""

    def __init__(self) -> None:
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        self.gym = gym
        self.runtime = maze_runtime_identity()
        self.sessions: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        for session in self.sessions.values():
            session["env"].close()
        self.sessions.clear()

    def abort(self, session_id: str) -> None:
        session = self.sessions.pop(str(session_id), None)
        if session is not None:
            session["env"].close()

    @staticmethod
    def _public(
        session_id: str,
        session: Mapping[str, Any],
        *,
        success: bool,
        done: bool,
    ) -> dict[str, Any]:
        observation = session["observation"]
        spec = session["spec"]
        current = session["current_cell"]
        allowed = () if done else legal_point_waypoint_actions(spec, current)
        return {
            "session_id": session_id,
            "achieved_goal": observation["achieved_goal"].astype(float).tolist(),
            "desired_goal": observation["desired_goal"].astype(float).tolist(),
            "velocity_xy": observation["observation"][2:4].astype(float).tolist(),
            "current_cell": list(current),
            "previous_cell": (
                None
                if session["previous_cell"] is None
                else list(session["previous_cell"])
            ),
            "goal_cell": list(spec.base_spec.goal_cell),
            "allowed_actions": list(allowed),
            "remaining_actions": spec.max_actions - len(session["tokens"]),
            "success": bool(success),
            "done": bool(done),
        }

    def reset(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        session_id = str(raw["session_id"])
        if not session_id or session_id in self.sessions:
            raise ValueError("Point waypoint session ID is empty or duplicated")
        raw_spec = raw["spec"]
        spec = parse_point_waypoint_spec(raw_spec)
        if raw_spec.get("verifier") != POINT_WAYPOINT_VERIFIER:
            raise ValueError("waypoint worker received the wrong verifier")
        if (
            spec.base_spec.environment_sha256
            != self.runtime["point_environment_sha256"]
        ):
            raise ValueError("PointMaze environment hash differs from runtime")
        env = self.gym.make(
            spec.base_spec.environment_id,
            maze_map=[list(row) for row in spec.base_spec.maze_map],
            reward_type="sparse",
            continuing_task=False,
            reset_target=False,
            max_episode_steps=spec.max_actions * spec.max_steps_per_action,
        )
        try:
            env.unwrapped.position_noise_range = 0.0
            observation, info = env.reset(
                seed=spec.base_spec.reset_seed,
                options={
                    "reset_cell": list(spec.base_spec.reset_cell),
                    "goal_cell": list(spec.base_spec.goal_cell),
                },
            )
        except BaseException:
            env.close()
            raise
        achieved = observation["achieved_goal"].astype(float).tolist()
        session = {
            "env": env,
            "raw_spec": dict(raw_spec),
            "spec": spec,
            "tokens": [],
            "target_cells": [],
            "trajectory": [achieved],
            "segment_steps": [],
            "steps": 0,
            "current_cell": spec.base_spec.reset_cell,
            "previous_cell": None,
            "observation": observation,
        }
        self.sessions[session_id] = session
        return self._public(
            session_id,
            session,
            success=bool(info.get("success", False)),
            done=False,
        )

    def step(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        session_id = str(raw["session_id"])
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("unknown Point waypoint session")
        spec = session["spec"]
        if len(session["tokens"]) >= spec.max_actions:
            raise ValueError("Point waypoint horizon is exhausted")
        token = str(raw["action"]).upper()
        allowed = legal_point_waypoint_actions(spec, session["current_cell"])
        if token not in allowed:
            raise ValueError("action is not a legal adjacent free-cell move")

        env = session["env"]
        target_cell = advance_point_waypoint_cell(spec, session["current_cell"], token)
        target = env.unwrapped.maze.cell_rowcol_to_xy(np.asarray(target_cell))
        session["tokens"].append(token)
        session["target_cells"].append(target_cell)
        before_steps = session["steps"]
        observation = session["observation"]
        info: dict[str, Any] = {}
        terminated = truncated = environment_success = waypoint_reached = False
        final_speed = float(np.linalg.norm(observation["observation"][2:4]))
        for _ in range(spec.max_steps_per_action):
            state = observation["observation"]
            position = state[:2]
            velocity = state[2:4]
            action = np.clip(
                spec.kp * (target - position) - spec.kd * velocity,
                -1.0,
                1.0,
            )
            observation, _reward, terminated, truncated, info = env.step(action)
            session["observation"] = observation
            session["steps"] += 1
            session["trajectory"].append(
                observation["achieved_goal"].astype(float).tolist()
            )
            position_error = float(
                np.linalg.norm(observation["achieved_goal"] - target)
            )
            final_speed = float(np.linalg.norm(observation["observation"][2:4]))
            waypoint_reached = (
                position_error <= spec.position_tolerance
                and final_speed <= spec.velocity_tolerance
            )
            environment_success = bool(info.get("success", False))
            if waypoint_reached or environment_success or terminated or truncated:
                break

        segment_steps = session["steps"] - before_steps
        session["segment_steps"].append(segment_steps)
        success = bool(environment_success and target_cell == spec.base_spec.goal_cell)
        stable_waypoint_failure = bool(not waypoint_reached and not success)
        if waypoint_reached or success:
            session["previous_cell"] = session["current_cell"]
            session["current_cell"] = target_cell
        horizon = len(session["tokens"]) >= spec.max_actions
        done = bool(
            success or stable_waypoint_failure or terminated or truncated or horizon
        )
        distance = float(
            np.linalg.norm(observation["achieved_goal"] - observation["desired_goal"])
        )
        canonical_key = None
        directed_gates: list[str] = []
        validation_error = None
        if done and success and len(session["tokens"]) >= spec.min_actions:
            execution = {
                "environment_sha256": spec.base_spec.environment_sha256,
                "controller_sha256": spec.controller_sha256,
                "spec_sha256": spec.spec_sha256,
                "reset_seed": spec.base_spec.reset_seed,
                "action_tokens": list(session["tokens"]),
                "target_cells": [list(cell) for cell in session["target_cells"]],
                "logical_final_cell": list(session["current_cell"]),
                "success": True,
                "stable_waypoint_failure": False,
                "final_goal_distance": distance,
                "final_speed": final_speed,
                "trajectory_xy": session["trajectory"],
                "segment_steps": session["segment_steps"],
                "simulator_steps": session["steps"],
            }
            try:
                validation = validate_point_waypoint_execution(
                    " ".join(session["tokens"]),
                    session["raw_spec"],
                    execution,
                )
                canonical_key = validation.canonical_key
                directed_gates = list(validation.directed_gates)
            except Exception as error:
                validation_error = f"{type(error).__name__}: {error}"

        response = {
            **self._public(
                session_id,
                session,
                success=success,
                done=done,
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "horizon_exhausted": bool(horizon and not success),
            "stable_waypoint_failure": stable_waypoint_failure,
            "waypoint_reached": bool(waypoint_reached),
            "target_cell": list(target_cell),
            "final_goal_distance": distance,
            "final_speed": final_speed,
            "action_count": len(session["tokens"]),
            "segment_steps": segment_steps,
            "simulator_steps": session["steps"],
            "canonical_key": canonical_key,
            "directed_gates": directed_gates,
            "validation_error": validation_error,
        }
        if done:
            env.close()
            self.sessions.pop(session_id)
        return response


def main() -> None:
    signal.signal(signal.SIGALRM, _timeout)
    worker = InteractivePointWaypointWorker()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                signal.setitimer(signal.ITIMER_REAL, 120.0)
                command = request.get("command")
                if command == "reset_batch":
                    results = [worker.reset(row) for row in request["sessions"]]
                    payload = {"ok": True, "results": results}
                elif command == "step_batch":
                    results = [worker.step(row) for row in request["steps"]]
                    payload = {"ok": True, "results": results}
                elif command == "abort_batch":
                    for session_id in request["session_ids"]:
                        worker.abort(str(session_id))
                    payload = {"ok": True}
                elif command == "close":
                    worker.close()
                    payload = {"ok": True}
                else:
                    raise ValueError("unsupported Point waypoint worker command")
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
