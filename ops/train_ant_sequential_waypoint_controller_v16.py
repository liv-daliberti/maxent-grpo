#!/usr/bin/env python3
"""Continue v11 on balanced multi-waypoint turn sequences."""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np

import train_ant_waypoint_controller_v7 as controller


gym.register_envs(gymnasium_robotics)
ROOT = controller.ROOT
WAYPOINT_DISTANCE = 4.0
SUCCESS_THRESHOLD = 0.45
SEGMENT_STEPS = 400
CARDINALS = (0, 2, 4, 6)
PERPENDICULAR = {
    0: (2, 6),
    2: (0, 4),
    4: (2, 6),
    6: (0, 4),
}
TRAINING_PATTERNS = tuple(
    (first, first, second, second)
    for first in CARDINALS
    for second in (*PERPENDICULAR[first], first)
)
EVALUATION_PATTERNS = tuple(
    (first, first, second, second, second, third, third)
    for first in CARDINALS
    for second in PERPENDICULAR[first]
    for third in PERPENDICULAR[second]
)
GRID_DELTAS = {
    0: (-1, 0),
    2: (0, 1),
    4: (1, 0),
    6: (0, -1),
}


def _maze(size: int, walls: tuple[tuple[int, int], ...]) -> list[list[int]]:
    maze = [[1] * size]
    maze.extend([[1] + [0] * (size - 2) + [1] for _ in range(size - 2)])
    maze.append([1] * size)
    for row, column in walls:
        maze[row][column] = 1
    return maze


TRAIN_MAPS = (
    _maze(11, ((1, 1), (1, 9))),
    _maze(11, ((1, 1), (9, 1))),
    _maze(11, ((1, 9), (9, 9))),
    _maze(11, ((9, 1), (9, 9))),
)
DEVELOPMENT_MAPS = (
    _maze(15, ((1, 1), (1, 13))),
    _maze(15, ((1, 1), (13, 1))),
    _maze(15, ((1, 13), (13, 13))),
    _maze(15, ((13, 1), (13, 13))),
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


class AntSequentialWaypointEnv(gym.Wrapper):
    """Train cardinal turns while retaining the controller observation API."""

    def __init__(self, *, rank: int, episode_steps: int = 1600) -> None:
        self.rank = int(rank)
        self.map_index = self.rank % len(TRAIN_MAPS)
        env = gym.make(
            "AntMaze_UMaze-v5",
            maze_map=[list(row) for row in TRAIN_MAPS[self.map_index]],
            reward_type="sparse",
            continuing_task=False,
            reset_target=False,
            max_episode_steps=episode_steps,
        )
        super().__init__(env)
        self.env.unwrapped.position_noise_range = 0.1
        self.episode_index = 0
        self.pattern: tuple[int, ...] = ()
        self.waypoint_index = 0
        self.segment_steps = 0
        self.initial_xy = np.zeros(2, dtype=np.float32)
        self.target_xy = np.zeros(2, dtype=np.float32)
        self.cumulative_heading = np.zeros(2, dtype=np.float32)
        self.previous_distance = WAYPOINT_DISTANCE
        base = env.observation_space["observation"]
        assert isinstance(base, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([base.low, np.full(2, -1.0, dtype=np.float32)]),
            high=np.concatenate([base.high, np.full(2, 1.0, dtype=np.float32)]),
            dtype=np.float32,
        )

    def _observation(self, observation) -> np.ndarray:
        relative = np.clip(
            (
                self.target_xy
                - np.asarray(observation["achieved_goal"], dtype=np.float32)
            )
            / WAYPOINT_DISTANCE,
            -1.0,
            1.0,
        )
        return np.concatenate(
            [np.asarray(observation["observation"], dtype=np.float32), relative]
        ).astype(np.float32, copy=False)

    def _advance_target(self) -> None:
        heading = self.pattern[self.waypoint_index]
        self.cumulative_heading += controller.HEADINGS[heading]
        self.target_xy = (
            self.initial_xy + WAYPOINT_DISTANCE * self.cumulative_heading
        ).astype(np.float32)
        self.previous_distance = float(
            np.linalg.norm(self.target_xy - self._current_xy())
        )

    def _current_xy(self) -> np.ndarray:
        return np.asarray(self.env.unwrapped.data.qpos[:2], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        pattern_index = (
            self.rank + self.episode_index
        ) % len(TRAINING_PATTERNS)
        self.pattern = TRAINING_PATTERNS[pattern_index]
        self.episode_index += 1
        start = (5, 5)
        goal = _goal_cell(start, self.pattern)
        observation, info = self.env.reset(
            seed=seed,
            options={"reset_cell": list(start), "goal_cell": list(goal)},
        )
        self.initial_xy = np.asarray(
            observation["achieved_goal"], dtype=np.float32
        ).copy()
        self.cumulative_heading = np.zeros(2, dtype=np.float32)
        self.waypoint_index = 0
        self.segment_steps = 0
        self._advance_target()
        info["sequential_pattern_index"] = pattern_index
        info["sequential_training_map_index"] = self.map_index
        return self._observation(observation), info

    def step(self, action):
        observation, _reward, terminated, truncated, info = self.env.step(action)
        self.segment_steps += 1
        distance = float(np.linalg.norm(self.target_xy - self._current_xy()))
        progress = self.previous_distance - distance
        self.previous_distance = distance
        reached = distance <= SUCCESS_THRESHOLD
        final_waypoint = self.waypoint_index == len(self.pattern) - 1
        timed_out = self.segment_steps >= SEGMENT_STEPS and not reached
        reward = (
            20.0 * progress
            - 0.02 * distance
            - 0.01
            + 0.10 * float(info.get("reward_survive", 1.0))
            + 0.10 * float(info.get("reward_ctrl", 0.0))
            + 0.05 * float(info.get("reward_contact", 0.0))
            + (50.0 if reached else 0.0)
            - (25.0 if (terminated or timed_out) and not reached else 0.0)
        )
        completed = reached and final_waypoint
        if reached and not final_waypoint:
            self.waypoint_index += 1
            self.segment_steps = 0
            self._advance_target()
        info["waypoint_distance"] = distance
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


def _factory(seed: int, rank: int, episode_steps: int) -> Callable[[], gym.Env]:
    def make() -> gym.Env:
        env = AntSequentialWaypointEnv(rank=rank, episode_steps=episode_steps)
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
    start = (7, 7)
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
        env.unwrapped.position_noise_range = 0.0
        observation, _ = env.reset(
            seed=seed,
            options={"reset_cell": list(start), "goal_cell": list(goal)},
        )
        initial = np.asarray(
            observation["achieved_goal"], dtype=np.float32
        ).copy()
        cumulative = np.zeros(2, dtype=np.float32)
        segment_steps = []
        success = True
        unhealthy = False
        final_distance = math.inf
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
                reached = final_distance <= SUCCESS_THRESHOLD
            segment_steps.append(steps)
            if not reached:
                success = False
                unhealthy = bool(terminated)
                break
        return {
            "map_index": map_index,
            "pattern_index": pattern_index,
            "pattern": list(pattern),
            "success": success,
            "segment_steps": segment_steps,
            "steps": sum(segment_steps),
            "final_distance": final_distance,
            "unhealthy_termination": unhealthy,
        }
    finally:
        env.close()


def _evaluate(
    model,
    *,
    seed: int,
    episodes_per_heading: int,
    episode_steps: int,
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
    successful_steps = [row["steps"] for row in episodes if row["success"]]
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
                float(np.median(successful_steps))
                if successful_steps
                else None
            ),
            "all_metrics_finite": all(
                math.isfinite(float(row["final_distance"])) for row in episodes
            ),
        },
    }


def _extra_checks(summary: dict) -> dict[str, bool]:
    return {
        "minimum_map_success_rate_at_least_0p75": (
            summary["minimum_map_success_rate"] >= 0.75
        )
    }


controller.CONTROLLER_VERSION = "v16"
controller.DEFAULT_OUTPUT = (
    ROOT / "var/maze_runtime/controllers/ant_sequential_waypoint_v16"
)
controller.INITIAL_MODEL = (
    ROOT / "var/maze_runtime/controllers/ant_waypoint_v11.zip"
)
controller.INITIAL_MODEL_SHA256 = (
    "e6d202bd525be5469135b35b63b2bf0884459cc630b71e76f8c49e86dcb8f913"
)
controller.DEFAULT_TIMESTEPS = 2_000_000
controller.DEFAULT_SEED = 73016
controller.DEFAULT_LEARNING_RATE = 1e-6
controller.EVALUATION_SEED_OFFSET = 7_000_000
controller.TRAINING_SOURCE_PATH = Path(__file__).resolve()
controller._factory = _factory
controller._evaluate = _evaluate
controller.EXTRA_EVALUATION_CHECKS = _extra_checks
controller.INFORMATION_FIREWALL = (
    "Only four preregistered 11x11 training maps and the complete balanced "
    "cardinal straight/perpendicular two-segment curriculum are loaded. The "
    "v15 map, v15 trajectories, language prompts, MaxEnt outcomes, and "
    "Dr.GRPO outcomes do not enter training."
)
controller.INITIALIZATION_DESCRIPTION = (
    "Exact admitted-v11 weights with a reset PPO optimizer; observation, "
    "action, relative-target, reward, architecture, four-unit waypoint, and "
    "0.45 success radius are retained while training exposes the policy to "
    "balanced four-command straight and left/right turn sequences."
)


if __name__ == "__main__":
    controller.main()
