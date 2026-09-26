#!/usr/bin/env python3
"""Continue v8 on balanced local waypoints from training-only maze maps."""

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
GRID_DELTAS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
TRAINING_HEADING_SCHEDULE = tuple(range(len(GRID_DELTAS)))


def _evaluation_edge_index(edges: tuple, replicate: int) -> int:
    """Preserve v9's frozen first-three-edge evaluation policy."""

    return replicate % len(edges)


def _maze(size: int, walls: tuple[tuple[int, int], ...]) -> list[list[int]]:
    maze = [[1] * size]
    maze.extend([[1] + [0] * (size - 2) + [1] for _ in range(size - 2)])
    maze.append([1] * size)
    for row, column in walls:
        maze[row][column] = 1
    return maze


# These are exactly the four training-map identities declared before v8 route
# execution. No v8 development/evaluation map or trajectory is loaded.
TRAIN_MAPS = tuple(
    _maze(7, ((3, 3), extra))
    for extra in ((1, 1), (2, 1), (3, 1), (4, 1))
)

# Different dimensions make these local-move evaluation maps fingerprint-
# disjoint from every 7x7 v5/v8 map. They never enter PPO training.
DEVELOPMENT_MAPS = (
    _maze(8, ((3, 3), (1, 6))),
    _maze(8, ((3, 4), (6, 1))),
    _maze(8, ((4, 3), (1, 2))),
    _maze(8, ((4, 4), (6, 5))),
)


def _edges_by_heading(
    maze_map: list[list[int]],
) -> tuple[tuple[tuple[tuple[int, int], tuple[int, int]], ...], ...]:
    height = len(maze_map)
    width = len(maze_map[0])
    by_heading = []
    for delta_row, delta_column in GRID_DELTAS:
        edges = []
        for row in range(1, height - 1):
            for column in range(1, width - 1):
                target = row + delta_row, column + delta_column
                if maze_map[row][column] or maze_map[target[0]][target[1]]:
                    continue
                if delta_row and delta_column:
                    if (
                        maze_map[row + delta_row][column]
                        or maze_map[row][column + delta_column]
                    ):
                        continue
                edges.append(((row, column), target))
        if not edges:
            raise ValueError("maze-local heading has no valid training edge")
        by_heading.append(tuple(edges))
    return tuple(by_heading)


TRAIN_EDGES = tuple(_edges_by_heading(maze_map) for maze_map in TRAIN_MAPS)
DEVELOPMENT_EDGES = tuple(
    _edges_by_heading(maze_map) for maze_map in DEVELOPMENT_MAPS
)


class AntMazeLocalWaypointEnv(gym.Wrapper):
    """One collision-aware local waypoint; map geometry is never observed."""

    def __init__(self, *, rank: int, episode_steps: int = 400) -> None:
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
        self.target_xy = np.zeros(2, dtype=np.float32)
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

    def reset(self, *, seed=None, options=None):
        heading_index = TRAINING_HEADING_SCHEDULE[
            (self.rank + self.episode_index) % len(TRAINING_HEADING_SCHEDULE)
        ]
        heading_edges = TRAIN_EDGES[self.map_index][heading_index]
        edge_index = (
            self.episode_index // len(TRAINING_HEADING_SCHEDULE)
        ) % len(heading_edges)
        reset_cell, goal_cell = heading_edges[edge_index]
        self.episode_index += 1
        observation, info = self.env.reset(
            seed=seed,
            options={
                "reset_cell": list(reset_cell),
                "goal_cell": list(goal_cell),
            },
        )
        self.target_xy = np.asarray(
            observation["desired_goal"], dtype=np.float32
        ).copy()
        self.previous_distance = float(
            np.linalg.norm(
                self.target_xy
                - np.asarray(observation["achieved_goal"], dtype=np.float32)
            )
        )
        info["waypoint_heading_index"] = heading_index
        info["waypoint_training_map_index"] = self.map_index
        return self._observation(observation), info

    def step(self, action):
        observation, _reward, terminated, truncated, info = self.env.step(action)
        distance = float(
            np.linalg.norm(
                self.target_xy
                - np.asarray(observation["achieved_goal"], dtype=np.float32)
            )
        )
        progress = self.previous_distance - distance
        self.previous_distance = distance
        success = distance <= SUCCESS_THRESHOLD
        reward = (
            20.0 * progress
            - 0.02 * distance
            - 0.01
            + 0.10 * float(info.get("reward_survive", 1.0))
            + 0.10 * float(info.get("reward_ctrl", 0.0))
            + 0.05 * float(info.get("reward_contact", 0.0))
            + (50.0 if success else 0.0)
            - (25.0 if terminated and not success else 0.0)
        )
        info["waypoint_distance"] = distance
        info["waypoint_success"] = success
        return (
            self._observation(observation),
            reward,
            bool(terminated or success),
            truncated,
            info,
        )


def _factory(seed: int, rank: int, episode_steps: int) -> Callable[[], gym.Env]:
    def make() -> gym.Env:
        env = AntMazeLocalWaypointEnv(rank=rank, episode_steps=episode_steps)
        env.reset(seed=seed + rank)
        return env

    return make


def _evaluate(
    model,
    *,
    seed: int,
    episodes_per_heading: int,
    episode_steps: int,
) -> dict:
    episodes = []
    for map_index, maze_map in enumerate(DEVELOPMENT_MAPS):
        for heading_index in range(len(GRID_DELTAS)):
            edges = DEVELOPMENT_EDGES[map_index][heading_index]
            for replicate in range(episodes_per_heading // len(DEVELOPMENT_MAPS)):
                reset_cell, goal_cell = edges[
                    _evaluation_edge_index(edges, replicate)
                ]
                env = gym.make(
                    "AntMaze_UMaze-v5",
                    maze_map=[list(row) for row in maze_map],
                    reward_type="sparse",
                    continuing_task=False,
                    reset_target=False,
                    max_episode_steps=episode_steps,
                )
                try:
                    env.unwrapped.position_noise_range = 0.0
                    observation, _ = env.reset(
                        seed=(
                            seed
                            + 10_000 * map_index
                            + 100 * heading_index
                            + replicate
                        ),
                        options={
                            "reset_cell": list(reset_cell),
                            "goal_cell": list(goal_cell),
                        },
                    )
                    target = np.asarray(
                        observation["desired_goal"], dtype=np.float32
                    )
                    success = False
                    terminated = truncated = False
                    steps = 0
                    distance = math.inf
                    while not (terminated or truncated or success):
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
                        distance = float(
                            np.linalg.norm(
                                target
                                - np.asarray(
                                    observation["achieved_goal"], dtype=np.float32
                                )
                            )
                        )
                        success = distance <= SUCCESS_THRESHOLD
                    episodes.append(
                        {
                            "map_index": map_index,
                            "heading_index": heading_index,
                            "replicate": replicate,
                            "success": success,
                            "steps": steps,
                            "final_distance": distance,
                            "unhealthy_termination": bool(
                                terminated and not success
                            ),
                        }
                    )
                finally:
                    env.close()
    heading_success_rates = {
        str(index): float(
            np.mean(
                [row["success"] for row in episodes if row["heading_index"] == index]
            )
        )
        for index in range(len(GRID_DELTAS))
    }
    map_success_rates = {
        str(index): float(
            np.mean([row["success"] for row in episodes if row["map_index"] == index])
        )
        for index in range(len(DEVELOPMENT_MAPS))
    }
    successful_steps = [row["steps"] for row in episodes if row["success"]]
    return {
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "success_rate": float(np.mean([row["success"] for row in episodes])),
            "minimum_heading_success_rate": min(heading_success_rates.values()),
            "heading_success_rates": heading_success_rates,
            "minimum_map_success_rate": min(map_success_rates.values()),
            "map_success_rates": map_success_rates,
            "unhealthy_termination_rate": float(
                np.mean([row["unhealthy_termination"] for row in episodes])
            ),
            "median_success_steps": (
                float(np.median(successful_steps)) if successful_steps else None
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


controller.CONTROLLER_VERSION = "v9"
controller.DEFAULT_OUTPUT = ROOT / "var/maze_runtime/controllers/ant_waypoint_v9"
controller.INITIAL_MODEL = ROOT / "var/maze_runtime/controllers/ant_waypoint_v8.zip"
controller.INITIAL_MODEL_SHA256 = (
    "77e780dfff1147bc2f542c1761f6efeaa2305fd5ddb625244ebf8e82b3fa871d"
)
controller.DEFAULT_TIMESTEPS = 5_000_000
controller.DEFAULT_SEED = 73009
controller.DEFAULT_LEARNING_RATE = 5e-6
controller.EVALUATION_SEED_OFFSET = 4_000_000
controller.TRAINING_SOURCE_PATH = Path(__file__).resolve()
controller._factory = _factory
controller._evaluate = _evaluate
controller.EXTRA_EVALUATION_CHECKS = _extra_checks
controller.INFORMATION_FIREWALL = (
    "Only four preregistered 7x7 training maps and local one-cell waypoint "
    "episodes are loaded. No v8 route trajectory, v8 development/evaluation "
    "map, v9 development-map outcome, future route map, language prompt, "
    "MaxEnt outcome, or Dr.GRPO outcome enters training."
)
controller.INITIALIZATION_DESCRIPTION = (
    "Exact admitted-v8 weights with a reset PPO optimizer; the observation, "
    "four-unit relative target, reward, architecture, eight-heading balance, "
    "success radius, and episode horizon are retained while the environment "
    "changes from open plane to training-only maze-local free-cell edges."
)


if __name__ == "__main__":
    controller.main()
