#!/usr/bin/env python3
"""Continue failed v9 with a frozen northeast-heavy local-edge schedule."""

from __future__ import annotations

from pathlib import Path

import train_ant_waypoint_controller_v9 as v9


ROOT = v9.ROOT


DEVELOPMENT_MAPS = (
    v9._maze(10, ((4, 4), (1, 8))),
    v9._maze(10, ((4, 5), (8, 1))),
    v9._maze(10, ((5, 4), (1, 2))),
    v9._maze(10, ((5, 5), (8, 7))),
)
DEVELOPMENT_EDGES = tuple(
    v9._edges_by_heading(maze_map) for maze_map in DEVELOPMENT_MAPS
)
TRAINING_HEADING_SCHEDULE = (1, 0, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7)


def _stratified_edge_index(edges: tuple, replicate: int) -> int:
    indices = (0, (len(edges) - 1) // 2, len(edges) - 1)
    return indices[replicate]


v9.DEVELOPMENT_MAPS = DEVELOPMENT_MAPS
v9.DEVELOPMENT_EDGES = DEVELOPMENT_EDGES
v9.TRAINING_HEADING_SCHEDULE = TRAINING_HEADING_SCHEDULE
v9._evaluation_edge_index = _stratified_edge_index
v9.controller.CONTROLLER_VERSION = "v10"
v9.controller.DEFAULT_OUTPUT = ROOT / "var/maze_runtime/controllers/ant_waypoint_v10"
v9.controller.INITIAL_MODEL = ROOT / "var/maze_runtime/controllers/ant_waypoint_v9.zip"
v9.controller.INITIAL_MODEL_SHA256 = (
    "c49cdb1b8c1bf0d89027766d062492e84538c827d6aa88a2079067a6867b19b7"
)
v9.controller.DEFAULT_TIMESTEPS = 2_000_000
v9.controller.DEFAULT_SEED = 73010
v9.controller.DEFAULT_LEARNING_RATE = 2e-6
v9.controller.EVALUATION_SEED_OFFSET = 5_000_000
v9.controller.TRAINING_SOURCE_PATH = Path(__file__).resolve()
v9.controller.INFORMATION_FIREWALL = (
    "Only the four preregistered 7x7 training maps and local one-cell waypoint "
    "episodes are loaded. The exact failed-v9 weights and aggregate terminal "
    "receipt are antecedents. No v9 development trajectory, route map, "
    "language prompt, MaxEnt outcome, or Dr.GRPO outcome enters training."
)
v9.controller.INITIALIZATION_DESCRIPTION = (
    "Exact failed-v9 weights with a reset PPO optimizer; northeast occupies "
    "half the fixed local-edge schedule, every other heading one fourteenth, "
    "with unchanged architecture, reward, observation, and success radius."
)


if __name__ == "__main__":
    v9.controller.main()
