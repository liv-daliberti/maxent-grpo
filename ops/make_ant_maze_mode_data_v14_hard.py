#!/usr/bin/env python3
"""Materialize the prospective harder AntMaze v14 route slate."""

from __future__ import annotations

import make_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v12 import controller_receipt_sha256


class _HardVerifierProcess(base.MazeVerifierProcess):
    def __init__(self, *, timeout_seconds=180.0, **kwargs):
        super().__init__(timeout_seconds=180.0, **kwargs)


base.MazeVerifierProcess = _HardVerifierProcess

# The v12 benchmark crossed one blocked cell with four high-level decisions.
# V14 crosses an eight-cell-wide 5x5 obstacle with fourteen decisions and two
# prospectively fixed route classes.
base.UPPER = ("N",) * 3 + ("E",) * 8 + ("S",) * 3
base.LOWER = ("S",) * 3 + ("E",) * 8 + ("N",) * 3
base.PERIPHERAL_CELLS = (
    (1, 1),
    (2, 1),
    (3, 1),
    (11, 1),
    (12, 1),
    (13, 1),
    (1, 13),
    (13, 13),
)
base.MAP_SIZE = 15
base.STATIC_WALLS = tuple(
    (row, column)
    for row in range(5, 10)
    for column in range(5, 10)
)
base.RESET_CELL = (7, 3)
base.GOAL_CELL = (7, 11)
base.BOUNDS_XY = ((-32.0, 32.0), (-32.0, 32.0))
base.CONTROLLER_RECEIPT_SHA256 = controller_receipt_sha256()
base.ANT_WORKER_SOURCE = "ant_maze_worker_v12.py"
base.DEFAULT_OUTPUT = base.ROOT / "var/data/ant_maze_modebench_v14_hard"
base.RESET_SEED_BASE = 108_400
base.MAP_ID_PREFIX = "ant_v14_hard"
base.MIN_ACTIONS = 14
base.MAX_ACTIONS = 24
base.ACTION_REPEAT = 400
base.DATA_SCHEMA = "ant-maze-modebench-data-v14-hard"
base.VERSION_LABEL = "ant-maze-v14-hard-data"


if __name__ == "__main__":
    base.main()

