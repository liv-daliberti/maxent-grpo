#!/usr/bin/env python3
"""Materialize the frozen 9x9 fresh-map AntMaze route slate for v9."""

from __future__ import annotations

import make_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v9 import controller_receipt_sha256


base.UPPER = ("N", "E", "E", "S")
base.LOWER = ("S", "E", "E", "N")
base.PERIPHERAL_CELLS = (
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
    (1, 2),
)
base.MAP_SIZE = 9
base.STATIC_WALLS = ((4, 4),)
base.RESET_CELL = (4, 3)
base.GOAL_CELL = (4, 5)
base.BOUNDS_XY = ((-16.0, 16.0), (-16.0, 16.0))
base.CONTROLLER_RECEIPT_SHA256 = controller_receipt_sha256()
base.ANT_WORKER_SOURCE = "ant_maze_worker_v9.py"
base.DEFAULT_OUTPUT = base.ROOT / "var/data/ant_maze_modebench_v9"
base.RESET_SEED_BASE = 97_300
base.MAP_ID_PREFIX = "ant_v9_admission"
base.MIN_ACTIONS = 4
base.MAX_ACTIONS = 16
base.ACTION_REPEAT = 400
base.DATA_SCHEMA = "ant-maze-modebench-data-v9"
base.VERSION_LABEL = "ant-maze-v9-data"


if __name__ == "__main__":
    base.main()
