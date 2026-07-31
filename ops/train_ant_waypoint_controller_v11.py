#!/usr/bin/env python3
"""Continue failed v10 with a frozen southeast/northeast remediation schedule."""

from __future__ import annotations

from pathlib import Path

import train_ant_waypoint_controller_v10 as v10


ROOT = v10.ROOT


DEVELOPMENT_MAPS = (
    v10.v9._maze(12, ((5, 5), (1, 10))),
    v10.v9._maze(12, ((5, 6), (10, 1))),
    v10.v9._maze(12, ((6, 5), (1, 2))),
    v10.v9._maze(12, ((6, 6), (10, 9))),
)
DEVELOPMENT_EDGES = tuple(
    v10.v9._edges_by_heading(maze_map) for maze_map in DEVELOPMENT_MAPS
)
TRAINING_HEADING_SCHEDULE = (3, 1, 3, 0, 3, 1, 3, 2, 3, 1, 3, 4, 1, 5, 6, 7)


def _stratified_edge_index(edges: tuple, replicate: int) -> int:
    indices = (0, (len(edges) - 1) // 2, len(edges) - 1)
    return indices[replicate]


v10.v9.DEVELOPMENT_MAPS = DEVELOPMENT_MAPS
v10.v9.DEVELOPMENT_EDGES = DEVELOPMENT_EDGES
v10.v9.TRAINING_HEADING_SCHEDULE = TRAINING_HEADING_SCHEDULE
v10.v9._evaluation_edge_index = _stratified_edge_index
v10.v9.controller.CONTROLLER_VERSION = "v11"
v10.v9.controller.DEFAULT_OUTPUT = ROOT / "var/maze_runtime/controllers/ant_waypoint_v11"
v10.v9.controller.INITIAL_MODEL = ROOT / "var/maze_runtime/controllers/ant_waypoint_v10.zip"
v10.v9.controller.INITIAL_MODEL_SHA256 = (
    "fffb66696aa4435c83b8bed854834ce82ccb01a9a1ce9dfe22e813c7e94a7b94"
)
v10.v9.controller.DEFAULT_TIMESTEPS = 2_000_000
v10.v9.controller.DEFAULT_SEED = 73011
v10.v9.controller.DEFAULT_LEARNING_RATE = 1e-6
v10.v9.controller.EVALUATION_SEED_OFFSET = 6_000_000
v10.v9.controller.TRAINING_SOURCE_PATH = Path(__file__).resolve()
v10.v9.controller.INFORMATION_FIREWALL = (
    "Only the four preregistered 7x7 training maps and local one-cell waypoint "
    "episodes are loaded. The exact failed-v10 weights and aggregate terminal "
    "and per-heading counts are antecedents. No v10 development trajectory, "
    "route map, language prompt, MaxEnt outcome, or Dr.GRPO outcome enters training."
)
v10.v9.controller.INITIALIZATION_DESCRIPTION = (
    "Exact failed-v10 weights with a reset PPO optimizer; southeast occupies "
    "six sixteenths and northeast four sixteenths of the fixed local-edge "
    "schedule, with every other heading represented once and all other "
    "controller mechanics unchanged."
)


if __name__ == "__main__":
    v10.v9.controller.main()
