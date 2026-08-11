"""Persistent networkless worker for option-level PointMaze Tour policies.

One request executes one leg: the policy names an unvisited landmark, the
pinned planner produces the shortest grid path to it, and the frozen PD adapter
drives that path carrying momentum through intermediate cells.  After the last
landmark the worker drives the closing leg to the goal without consulting the
policy, so an episode contains exactly ``K`` policy decisions.
"""

from __future__ import annotations

import json
import signal
import sys
from typing import Any, Mapping

from .maze_runtime_identity import maze_runtime_identity
from .point_maze_tour import (
    POINT_TOUR_VERIFIER,
    parse_point_tour_spec,
    plan_leg_cells,
    validate_point_tour_execution,
)


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("interactive PointMaze tour request timed out")


class InteractivePointTourWorker:
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
        chosen = list(session["chosen"])
        remaining = [
            identifier for identifier in spec.landmark_ids if identifier not in chosen
        ]
        return {
            "session_id": session_id,
            "achieved_goal": observation["achieved_goal"].astype(float).tolist(),
            "desired_goal": observation["desired_goal"].astype(float).tolist(),
            "velocity_xy": observation["observation"][2:4].astype(float).tolist(),
            "current_cell": list(session["current_cell"]),
            "goal_cell": list(spec.base_spec.goal_cell),
            "landmark_ids": list(spec.landmark_ids),
            "landmark_cells": [list(cell) for cell in spec.landmark_cells],
            "visited": list(session["visited"]),
            "chosen": chosen,
            # The menu is over landmarks not yet *named*, so an episode always
            # emits exactly one permutation and always makes K decisions.
            "allowed_landmarks": [] if done else remaining,
            "remaining_steps": max(0, spec.tour_step_budget - session["steps"]),
            "simulator_steps": int(session["steps"]),
            "success": bool(success),
            "done": bool(done),
        }

    def reset(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        session_id = str(raw["session_id"])
        if not session_id or session_id in self.sessions:
            raise ValueError("Point tour session ID is empty or duplicated")
        raw_spec = raw["spec"]
        spec = parse_point_tour_spec(raw_spec)
        if raw_spec.get("verifier") != POINT_TOUR_VERIFIER:
            raise ValueError("tour worker received the wrong verifier")
        if spec.base_spec.environment_sha256 != self.runtime["point_environment_sha256"]:
            raise ValueError("PointMaze environment hash differs from runtime")
        env = self.gym.make(
            spec.base_spec.environment_id,
            maze_map=[list(row) for row in spec.base_spec.maze_map],
            reward_type="sparse",
            continuing_task=False,
            reset_target=False,
            max_episode_steps=spec.base_spec.max_actions * spec.base_spec.action_repeat,
        )
        try:
            env.unwrapped.position_noise_range = 0.0
            observation, _info = env.reset(
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
            "chosen": [],
            "visited": [],
            "trajectory": [achieved],
            "leg_steps": [],
            "steps": 0,
            "current_cell": spec.base_spec.reset_cell,
            "observation": observation,
            "np": np,
        }
        self.sessions[session_id] = session
        return self._public(session_id, session, success=False, done=False)

    def _drive_cells(
        self,
        session: dict[str, Any],
        cells,
        *,
        settle_at_end: bool,
    ) -> tuple[bool, bool, bool]:
        """Drive one planned leg; return (arrived, environment_success, blocked)."""

        np = session["np"]
        spec = session["spec"]
        env = session["env"]
        budget = spec.tour_step_budget
        leg_start = session["steps"]
        arrived = False
        environment_success = False
        for index, cell in enumerate(cells):
            final_cell = index == len(cells) - 1
            target = env.unwrapped.maze.cell_rowcol_to_xy(np.asarray(cell))
            arrived = False
            while True:
                if session["steps"] >= budget:
                    return False, environment_success, True
                if session["steps"] - leg_start >= spec.max_steps_per_leg:
                    return False, environment_success, True
                state = session["observation"]["observation"]
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
                error = float(np.linalg.norm(observation["achieved_goal"] - target))
                speed = float(np.linalg.norm(observation["observation"][2:4]))
                environment_success = bool(info.get("success", False))
                if final_cell and settle_at_end:
                    arrived = (
                        error <= spec.position_tolerance
                        and speed <= spec.velocity_tolerance
                    )
                elif final_cell:
                    arrived = error <= spec.position_tolerance
                else:
                    arrived = error <= spec.intermediate_waypoint_radius
                if arrived or environment_success or terminated or truncated:
                    break
            if not arrived and not environment_success:
                return False, environment_success, True
        return True, environment_success, False

    def step_leg(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        session_id = str(raw["session_id"])
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("unknown Point tour session")
        spec = session["spec"]
        landmark_id = str(raw["landmark"])
        if landmark_id in session["chosen"]:
            raise ValueError("Point tour landmark was already chosen")
        if landmark_id not in spec.landmark_ids:
            raise ValueError("Point tour landmark is unknown")
        if len(session["chosen"]) >= spec.landmark_count:
            raise ValueError("Point tour has already named every landmark")

        session["chosen"].append(landmark_id)
        failure = None
        # A leg is attempted only while the tour is still on track and has
        # budget left; otherwise the decision is still recorded and scored, so
        # every episode contributes exactly K decisions no matter when it fails.
        on_track = len(session["visited"]) == len(session["chosen"]) - 1
        arrived = False
        if on_track:
            target_cell = spec.landmark_cell(landmark_id)
            cells = plan_leg_cells(spec, session["current_cell"], target_cell)
            arrived, _environment_success, blocked = self._drive_cells(
                session, cells, settle_at_end=True
            )
            if arrived:
                session["current_cell"] = target_cell
                session["visited"].append(landmark_id)
            else:
                failure = "budget_exhausted" if blocked else "leg_failed"
        else:
            failure = "tour_already_failed"
        session["leg_steps"].append(session["steps"])

        success = False
        done = len(session["chosen"]) == spec.landmark_count
        if done and len(session["visited"]) == spec.landmark_count:
            closing = plan_leg_cells(
                spec, session["current_cell"], spec.base_spec.goal_cell
            )
            reached, environment_success, closing_blocked = self._drive_cells(
                session, closing, settle_at_end=False
            )
            if reached or environment_success:
                session["current_cell"] = spec.base_spec.goal_cell
            success = bool(environment_success)
            if not success:
                failure = "budget_exhausted" if closing_blocked else "goal_leg_failed"
        canonical_key = None
        landmark_order: list[str] = []
        validation_error = None
        if done and success:
            execution = {
                "environment_sha256": spec.base_spec.environment_sha256,
                "controller_sha256": spec.controller_sha256,
                "spec_sha256": spec.spec_sha256,
                "reset_seed": spec.base_spec.reset_seed,
                "landmark_order": list(session["visited"]),
                "success": True,
                "trajectory_xy": session["trajectory"],
                "leg_steps": list(session["leg_steps"]),
                "simulator_steps": session["steps"],
            }
            try:
                validation = validate_point_tour_execution(
                    session["raw_spec"], execution
                )
                canonical_key = validation.canonical_key
                landmark_order = list(validation.action_tokens)
            except Exception as error:
                validation_error = f"{type(error).__name__}: {error}"

        response = {
            **self._public(session_id, session, success=success, done=done),
            "landmark": landmark_id,
            "arrived": bool(arrived),
            "failure": failure,
            "leg_count": len(session["chosen"]),
            "canonical_key": canonical_key,
            "landmark_order": landmark_order,
            "validation_error": validation_error,
        }
        if done:
            session["env"].close()
            self.sessions.pop(session_id)
        return response


def main() -> None:
    signal.signal(signal.SIGALRM, _timeout)
    worker = InteractivePointTourWorker()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                signal.setitimer(signal.ITIMER_REAL, 300.0)
                command = request.get("command")
                if command == "reset_batch":
                    results = [worker.reset(row) for row in request["sessions"]]
                    payload = {"ok": True, "results": results}
                elif command == "step_batch":
                    results = [worker.step_leg(row) for row in request["steps"]]
                    payload = {"ok": True, "results": results}
                elif command == "abort_batch":
                    for session_id in request["session_ids"]:
                        worker.abort(str(session_id))
                    payload = {"ok": True}
                elif command == "close":
                    worker.close()
                    payload = {"ok": True}
                else:
                    raise ValueError("unsupported Point tour worker command")
            except Exception as error:
                payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
            sys.stdout.flush()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
