#!/usr/bin/env python3
"""Repair sequential Ant control under goal-independent task semantics."""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np

import train_ant_stable_handoff_controller_v18 as v18


controller = v18.controller
ROOT = controller.ROOT
WAYPOINT_DISTANCE = 4.0
SUCCESS_THRESHOLD = 0.45
STABLE_PLANAR_SPEED = 1.0
SEGMENT_STEPS = 400
HEADINGS = tuple(range(8))
GRID_DELTAS = v18.GRID_DELTAS


def _maze(size: int, walls: tuple[tuple[int, int], ...]) -> list[list[int]]:
    maze = [[1] * size]
    maze.extend([[1] + [0] * (size - 2) + [1] for _ in range(size - 2)])
    maze.append([1] * size)
    for row, column in walls:
        maze[row][column] = 1
    return maze


# Anchors retain v11's reset-state competence. Every ordered handoff is seen
# both directly and after a repeated heading. The final family exposes the
# stable-arrival policy to two consecutive handoffs without using a gate row.
ANCHOR_PATTERNS = tuple((heading,) for heading in HEADINGS) * 32
PAIR_PATTERNS = tuple(
    (first, second) for first in HEADINGS for second in HEADINGS
) * 2
REPEATED_HANDOFF_PATTERNS = tuple(
    (first, first, second) for first in HEADINGS for second in HEADINGS
)
TWO_HANDOFF_PATTERNS = tuple(
    (first, second, second, (2 * second - first) % 8)
    for first in HEADINGS
    for second in HEADINGS
)
TRAINING_PATTERNS = (
    ANCHOR_PATTERNS
    + PAIR_PATTERNS
    + REPEATED_HANDOFF_PATTERNS
    + TWO_HANDOFF_PATTERNS
)

TRAIN_MAPS = (
    _maze(17, ((1, 1), (1, 15), (15, 1), (2, 15))),
    _maze(17, ((1, 1), (1, 15), (15, 15), (15, 2))),
    _maze(17, ((1, 1), (15, 1), (15, 15), (1, 14))),
    _maze(17, ((1, 15), (15, 1), (15, 15), (14, 1))),
)

# The length-eight slate and 21x21 maps are frozen before v19 optimization.
# None of these patterns appears in the length-one-to-four curriculum or in
# the v17/v18 development slates.
EVALUATION_PATTERNS = tuple(
    pattern
    for first in HEADINGS
    for pattern in (
        (
            first,
            first,
            (first + 1) % 8,
            (first + 2) % 8,
            (first + 2) % 8,
            (first + 3) % 8,
            (first + 4) % 8,
            (first + 5) % 8,
        ),
        (
            first,
            (first - 1) % 8,
            (first - 2) % 8,
            (first - 2) % 8,
            (first - 3) % 8,
            (first - 4) % 8,
            (first - 4) % 8,
            (first - 5) % 8,
        ),
        (
            first,
            (first + 3) % 8,
            (first + 3) % 8,
            (first + 1) % 8,
            (first + 1) % 8,
            (first - 2) % 8,
            (first - 2) % 8,
            (first + 4) % 8,
        ),
    )
)
DEVELOPMENT_MAPS = (
    _maze(21, ((1, 1), (1, 19), (19, 1), (2, 19), (18, 2))),
    _maze(21, ((1, 1), (1, 19), (19, 19), (19, 2), (2, 18))),
    _maze(21, ((1, 1), (19, 1), (19, 19), (1, 18), (18, 19))),
    _maze(21, ((1, 19), (19, 1), (19, 19), (18, 1), (2, 2))),
)


def _goal_cell(
    start: tuple[int, int], pattern: tuple[int, ...]
) -> tuple[int, int]:
    row, column = start
    for heading in pattern:
        delta_row, delta_column = GRID_DELTAS[heading]
        row += delta_row
        column += delta_column
    return row, column


# The inherited reset and target bookkeeping resolve these globals in v17.
v18.v17.TRAINING_PATTERNS = TRAINING_PATTERNS
v18.v17.TRAIN_MAPS = TRAIN_MAPS


def _ant_is_healthy(env: gym.Env) -> bool:
    """Read health explicitly because AntMaze-v5 discards Ant termination."""

    return bool(env.unwrapped.ant_env.is_healthy)


class AntContinuingStableHandoffEnv(v18.AntStableHandoffEnv):
    """Keep the maze goal from terminating controller-training episodes."""

    def __init__(self, *, rank: int, episode_steps: int = 1600) -> None:
        super().__init__(rank=rank, episode_steps=episode_steps)
        self.env.unwrapped.continuing_task = True
        self.env.unwrapped.reset_target = False

    def step(self, action):
        observation, _reward, task_terminated, truncated, info = self.env.step(
            action
        )
        self.segment_steps += 1
        current_xy = self._current_xy()
        distance = float(np.linalg.norm(self.target_xy - current_xy))
        progress = self.previous_distance - distance
        self.previous_distance = distance
        planar_speed = float(
            np.linalg.norm(np.asarray(self.env.unwrapped.data.qvel[:2]))
        )
        healthy = _ant_is_healthy(self.env)
        within_radius = distance <= SUCCESS_THRESHOLD
        reached = (
            within_radius and planar_speed <= STABLE_PLANAR_SPEED and healthy
        )
        final_waypoint = self.waypoint_index == len(self.pattern) - 1
        timed_out = self.segment_steps >= SEGMENT_STEPS and not reached
        braking_weight = max(0.0, 1.25 - distance)
        reward = (
            20.0 * progress
            - 0.02 * distance
            - 0.01
            + 0.10 * float(info.get("reward_survive", 1.0))
            + 0.10 * float(info.get("reward_ctrl", 0.0))
            + 0.05 * float(info.get("reward_contact", 0.0))
            - 0.20 * braking_weight * planar_speed * planar_speed
            + (50.0 if reached else 0.0)
            - (25.0 if (not healthy or timed_out) and not reached else 0.0)
        )
        completed = reached and final_waypoint
        if reached and not final_waypoint:
            self.waypoint_index += 1
            self.segment_steps = 0
            self._advance_target()
        info["ant_healthy"] = healthy
        info["maze_task_terminated"] = bool(task_terminated)
        info["waypoint_distance"] = distance
        info["waypoint_planar_speed"] = planar_speed
        info["waypoint_within_radius"] = within_radius
        info["waypoint_success"] = reached
        info["sequential_waypoints_completed"] = (
            self.waypoint_index + int(completed)
        )
        return (
            self._observation(observation),
            reward,
            bool(completed or timed_out or not healthy),
            truncated,
            info,
        )


def _factory(seed: int, rank: int, episode_steps: int):
    def make() -> gym.Env:
        env = AntContinuingStableHandoffEnv(
            rank=rank, episode_steps=episode_steps
        )
        env.reset(seed=seed + rank)
        return env

    return make


def _evaluate_pattern(
    model,
    *,
    maze_map: list[list[int]],
    map_index: int,
    pattern: tuple[int, ...],
    pattern_index: int,
    seed: int,
) -> dict:
    start = (10, 10)
    goal = _goal_cell(start, pattern)
    env = gym.make(
        "AntMaze_UMaze-v5",
        maze_map=[list(row) for row in maze_map],
        reward_type="sparse",
        continuing_task=True,
        reset_target=False,
        max_episode_steps=len(pattern) * SEGMENT_STEPS,
    )
    try:
        env.unwrapped.position_noise_range = 0.1
        observation, _ = env.reset(
            seed=seed,
            options={"reset_cell": list(start), "goal_cell": list(goal)},
        )
        initial = np.asarray(
            observation["achieved_goal"], dtype=np.float32
        ).copy()
        cumulative = np.zeros(2, dtype=np.float32)
        segment_steps: list[int] = []
        segment_arrival_speeds: list[float] = []
        success = True
        health_failure = False
        task_termination_count = 0
        final_distance = math.inf
        final_speed = math.inf
        for heading in pattern:
            cumulative += controller.HEADINGS[heading]
            target = initial + WAYPOINT_DISTANCE * cumulative
            reached = False
            truncated = False
            steps = 0
            while steps < SEGMENT_STEPS and not (
                reached or health_failure or truncated
            ):
                relative = np.clip(
                    (
                        target
                        - np.asarray(
                            observation["achieved_goal"], dtype=np.float32
                        )
                    )
                    / WAYPOINT_DISTANCE,
                    -1.0,
                    1.0,
                )
                augmented = np.concatenate(
                    [
                        np.asarray(
                            observation["observation"], dtype=np.float32
                        ),
                        relative,
                    ]
                )
                action, _ = model.predict(augmented, deterministic=True)
                observation, _reward, task_terminated, truncated, _info = (
                    env.step(action)
                )
                task_termination_count += int(bool(task_terminated))
                steps += 1
                final_distance = float(
                    np.linalg.norm(
                        target
                        - np.asarray(
                            observation["achieved_goal"], dtype=np.float32
                        )
                    )
                )
                final_speed = float(
                    np.linalg.norm(np.asarray(env.unwrapped.data.qvel[:2]))
                )
                health_failure = not _ant_is_healthy(env)
                reached = (
                    final_distance <= SUCCESS_THRESHOLD
                    and final_speed <= STABLE_PLANAR_SPEED
                    and not health_failure
                )
            segment_steps.append(steps)
            if reached:
                segment_arrival_speeds.append(final_speed)
            else:
                success = False
                break
        return {
            "map_index": map_index,
            "pattern_index": pattern_index,
            "pattern": list(pattern),
            "success": success,
            "segment_steps": segment_steps,
            "segment_arrival_speeds": segment_arrival_speeds,
            "steps": sum(segment_steps),
            "final_distance": final_distance,
            "final_planar_speed": final_speed,
            "health_failure": health_failure,
            # Retained for the inherited receipt/check schema. Unlike v18,
            # this is explicit Ant health rather than maze-goal termination.
            "unhealthy_termination": health_failure,
            "maze_task_termination_count": task_termination_count,
        }
    finally:
        env.close()


def _evaluate(
    model, *, seed: int, episodes_per_heading: int, episode_steps: int
) -> dict:
    del episodes_per_heading, episode_steps
    episodes = []
    for map_index, maze_map in enumerate(DEVELOPMENT_MAPS):
        for pattern_index, pattern in enumerate(EVALUATION_PATTERNS):
            episodes.append(
                _evaluate_pattern(
                    model,
                    maze_map=maze_map,
                    map_index=map_index,
                    pattern=pattern,
                    pattern_index=pattern_index,
                    seed=seed + 10_000 * map_index + pattern_index,
                )
            )
    pattern_success_rates = {
        str(index): float(
            np.mean(
                [
                    row["success"]
                    for row in episodes
                    if row["pattern_index"] == index
                ]
            )
        )
        for index in range(len(EVALUATION_PATTERNS))
    }
    map_success_rates = {
        str(index): float(
            np.mean(
                [row["success"] for row in episodes if row["map_index"] == index]
            )
        )
        for index in range(len(DEVELOPMENT_MAPS))
    }
    successful_segments = [
        step
        for row in episodes
        if row["success"]
        for step in row["segment_steps"]
    ]
    arrival_speeds = [
        speed for row in episodes for speed in row["segment_arrival_speeds"]
    ]
    return {
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "success_rate": float(np.mean([row["success"] for row in episodes])),
            "minimum_heading_success_rate": min(
                pattern_success_rates.values()
            ),
            "heading_success_rates": pattern_success_rates,
            "minimum_map_success_rate": min(map_success_rates.values()),
            "map_success_rates": map_success_rates,
            "health_failure_rate": float(
                np.mean([row["health_failure"] for row in episodes])
            ),
            "unhealthy_termination_rate": float(
                np.mean([row["health_failure"] for row in episodes])
            ),
            "maze_task_termination_count": sum(
                row["maze_task_termination_count"] for row in episodes
            ),
            "median_success_steps": (
                float(np.median(successful_segments))
                if successful_segments
                else None
            ),
            "maximum_arrival_speed": (
                max(arrival_speeds) if arrival_speeds else None
            ),
            "all_metrics_finite": all(
                math.isfinite(float(row["final_distance"]))
                and math.isfinite(float(row["final_planar_speed"]))
                for row in episodes
            ),
        },
    }


def _extra_checks(summary: dict) -> dict[str, bool]:
    return {
        "minimum_map_success_rate_at_least_0p75": (
            summary["minimum_map_success_rate"] >= 0.75
        ),
        "all_recorded_arrivals_stable": (
            summary["maximum_arrival_speed"] is not None
            and summary["maximum_arrival_speed"]
            <= STABLE_PLANAR_SPEED + 1e-6
        ),
        "maze_task_never_terminated_controller_gate": (
            summary["maze_task_termination_count"] == 0
        ),
    }


controller.CONTROLLER_VERSION = "v19"
controller.DEFAULT_OUTPUT = (
    ROOT / "var/maze_runtime/controllers/ant_continuing_waypoint_v19"
)
controller.INITIAL_MODEL = (
    ROOT / "var/maze_runtime/controllers/ant_waypoint_v11.zip"
)
controller.INITIAL_MODEL_SHA256 = (
    "e6d202bd525be5469135b35b63b2bf0884459cc630b71e76f8c49e86dcb8f913"
)
controller.DEFAULT_TIMESTEPS = 6_000_000
controller.DEFAULT_SEED = 73019
controller.DEFAULT_LEARNING_RATE = 5e-7
controller.EVALUATION_SEED_OFFSET = 13_000_000
controller.TRAINING_SOURCE_PATH = Path(__file__).resolve()
controller._factory = _factory
controller._evaluate = _evaluate
controller.EXTRA_EVALUATION_CHECKS = _extra_checks
controller.INFORMATION_FIREWALL = (
    "The sealed v18 receipt and the disclosed task-termination diagnosis are "
    "antecedents. Optimization restarts from exact admitted v11 and uses "
    "only four generic 17x17 maps under continuing-task semantics. The "
    "fresh v19 21x21 maps and length-eight gate trajectories, the v15 route "
    "slate, language samples, MaxEnt outcomes, and Dr.GRPO outcomes are not "
    "loaded by training."
)
controller.INITIALIZATION_DESCRIPTION = (
    "Exact admitted-v11 policy weights with a reset PPO optimizer. AntMaze "
    "task-goal termination is disabled during controller training and gate "
    "execution; Ant health is measured explicitly. Stable-arrival anchors, "
    "all ordered handoffs, and two-handoff patterns are trained."
)


if __name__ == "__main__":
    controller.main()
