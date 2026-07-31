#!/usr/bin/env python3
"""Train a velocity-stabilized Ant controller on all ordered handoffs."""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np

import train_ant_sequential_waypoint_controller_v17 as v17


controller = v17.controller
ROOT = controller.ROOT
WAYPOINT_DISTANCE = 4.0
SUCCESS_THRESHOLD = 0.45
STABLE_PLANAR_SPEED = 1.0
SEGMENT_STEPS = 400
HEADINGS = tuple(range(8))
GRID_DELTAS = v17.GRID_DELTAS


def _maze(size: int, walls: tuple[tuple[int, int], ...]) -> list[list[int]]:
    maze = [[1] * size]
    maze.extend([[1] + [0] * (size - 2) + [1] for _ in range(size - 2)])
    maze.append([1] * size)
    for row, column in walls:
        maze[row][column] = 1
    return maze


# Training is deliberately local and generic.  Single-waypoint anchors retain
# reset-state competence; every ordered heading pair appears twice; and every
# repeated-heading-to-new-heading triple appears once.
ANCHOR_PATTERNS = tuple((heading,) for heading in HEADINGS) * 32
PAIR_PATTERNS = tuple(
    (first, second) for first in HEADINGS for second in HEADINGS
) * 2
TRIPLE_PATTERNS = tuple(
    (first, first, second) for first in HEADINGS for second in HEADINGS
)
TRAINING_PATTERNS = ANCHOR_PATTERNS + PAIR_PATTERNS + TRIPLE_PATTERNS

TRAIN_MAPS = (
    _maze(15, ((1, 1), (1, 13), (13, 1))),
    _maze(15, ((1, 1), (1, 13), (13, 13))),
    _maze(15, ((1, 1), (13, 1), (13, 13))),
    _maze(15, ((1, 13), (13, 1), (13, 13))),
)

# These length-eight sequences are disjoint from the length-one-to-three
# curriculum and from the v17 development patterns.  They are defined before
# v18 optimization and evaluated on four new 19x19 maps.
EVALUATION_PATTERNS = tuple(
    pattern
    for first in HEADINGS
    for pattern in (
        (
            first,
            first,
            (first + 2) % 8,
            (first + 2) % 8,
            (first + 4) % 8,
            (first + 4) % 8,
            (first + 6) % 8,
            (first + 6) % 8,
        ),
        (
            first,
            (first + 1) % 8,
            (first + 2) % 8,
            (first + 3) % 8,
            (first + 4) % 8,
            (first + 5) % 8,
            (first + 6) % 8,
            (first + 7) % 8,
        ),
        (
            first,
            first,
            (first + 1) % 8,
            (first + 1) % 8,
            (first - 1) % 8,
            (first - 1) % 8,
            (first - 2) % 8,
            (first - 2) % 8,
        ),
    )
)
DEVELOPMENT_MAPS = (
    _maze(19, ((1, 1), (1, 17), (17, 1), (2, 17))),
    _maze(19, ((1, 1), (1, 17), (17, 17), (17, 2))),
    _maze(19, ((1, 1), (17, 1), (17, 17), (1, 16))),
    _maze(19, ((1, 17), (17, 1), (17, 17), (16, 1))),
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


# The inherited reset implementation resolves these globals in the v17 module.
# Rebinding changes only the generic training curriculum and maps.
v17.TRAINING_PATTERNS = TRAINING_PATTERNS
v17.TRAIN_MAPS = TRAIN_MAPS


class AntStableHandoffEnv(v17.AntRetentionSequentialWaypointEnv):
    """Require low-speed arrival before switching to the next target."""

    def step(self, action):
        observation, _reward, terminated, truncated, info = self.env.step(action)
        self.segment_steps += 1
        current_xy = self._current_xy()
        distance = float(np.linalg.norm(self.target_xy - current_xy))
        progress = self.previous_distance - distance
        self.previous_distance = distance
        planar_speed = float(
            np.linalg.norm(np.asarray(self.env.unwrapped.data.qvel[:2]))
        )
        within_radius = distance <= SUCCESS_THRESHOLD
        reached = within_radius and planar_speed <= STABLE_PLANAR_SPEED
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
            - (25.0 if (terminated or timed_out) and not reached else 0.0)
        )
        completed = reached and final_waypoint
        if reached and not final_waypoint:
            self.waypoint_index += 1
            self.segment_steps = 0
            self._advance_target()
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
            bool(completed or timed_out or (terminated and not reached)),
            truncated,
            info,
        )


def _factory(seed: int, rank: int, episode_steps: int):
    def make() -> gym.Env:
        env = AntStableHandoffEnv(rank=rank, episode_steps=episode_steps)
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
    start = (9, 9)
    goal = _goal_cell(start, pattern)
    env = gym.make(
        "AntMaze_UMaze-v5",
        maze_map=[list(row) for row in maze_map],
        reward_type="sparse",
        continuing_task=False,
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
        unhealthy = False
        final_distance = math.inf
        final_speed = math.inf
        for heading in pattern:
            cumulative += controller.HEADINGS[heading]
            target = initial + WAYPOINT_DISTANCE * cumulative
            reached = False
            terminated = truncated = False
            steps = 0
            while steps < SEGMENT_STEPS and not (
                reached or terminated or truncated
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
                observation, _reward, terminated, truncated, _info = env.step(
                    action
                )
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
                reached = (
                    final_distance <= SUCCESS_THRESHOLD
                    and final_speed <= STABLE_PLANAR_SPEED
                )
            segment_steps.append(steps)
            if reached:
                segment_arrival_speeds.append(final_speed)
            else:
                success = False
                unhealthy = bool(terminated)
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
            "unhealthy_termination": unhealthy,
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
            "unhealthy_termination_rate": float(
                np.mean([row["unhealthy_termination"] for row in episodes])
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
    }


controller.CONTROLLER_VERSION = "v18"
controller.DEFAULT_OUTPUT = (
    ROOT / "var/maze_runtime/controllers/ant_stable_handoff_v18"
)
controller.INITIAL_MODEL = (
    ROOT / "var/maze_runtime/controllers/ant_sequential_waypoint_v17.zip"
)
controller.INITIAL_MODEL_SHA256 = (
    "7a964daa7ebc02d52e4717e62ec7d02eed70d5b3155d4910728e24a5e10bbf0d"
)
controller.DEFAULT_TIMESTEPS = 6_000_000
controller.DEFAULT_SEED = 73018
controller.DEFAULT_LEARNING_RATE = 2e-7
controller.EVALUATION_SEED_OFFSET = 11_000_000
controller.TRAINING_SOURCE_PATH = Path(__file__).resolve()
controller._factory = _factory
controller._evaluate = _evaluate
controller.EXTRA_EVALUATION_CHECKS = _extra_checks
controller.INFORMATION_FIREWALL = (
    "The sealed v17 checkpoint and aggregate/episode failure diagnosis are "
    "antecedents. Training uses only generic 15x15 maps, reset-state anchors, "
    "and the complete ordered heading-pair curriculum. The fresh v18 19x19 "
    "development trajectories are not accessed by the training environment "
    "or optimizer. The v15 map and route slate, language samples, MaxEnt "
    "outcomes, and Dr.GRPO outcomes are not loaded."
)
controller.INITIALIZATION_DESCRIPTION = (
    "Exact sealed failed-v17 weights with a reset PPO optimizer. A waypoint "
    "now switches only after position and planar velocity are jointly stable; "
    "all ordered heading handoffs are trained while reset-state anchors remain."
)


if __name__ == "__main__":
    controller.main()
