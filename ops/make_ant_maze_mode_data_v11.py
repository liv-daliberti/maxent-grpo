#!/usr/bin/env python3
"""Materialize the unchanged, previously unexecuted 11x11 AntMaze route slate for v11."""

from __future__ import annotations

import make_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v11 import controller_receipt_sha256


base.UPPER = ("N", "E", "E", "S")
base.LOWER = ("S", "E", "E", "N")
base.PERIPHERAL_CELLS = ((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1))
base.MAP_SIZE = 11
base.STATIC_WALLS = ((5, 5),)
base.RESET_CELL = (5, 4)
base.GOAL_CELL = (5, 6)
base.BOUNDS_XY = ((-20.0, 20.0), (-20.0, 20.0))
base.CONTROLLER_RECEIPT_SHA256 = controller_receipt_sha256()
base.ANT_WORKER_SOURCE = "ant_maze_worker_v11.py"
base.DEFAULT_OUTPUT = base.ROOT / "var/data/ant_maze_modebench_v11"
base.RESET_SEED_BASE = 107_300
base.MAP_ID_PREFIX = "ant_v11_admission"
base.MIN_ACTIONS = 4
base.MAX_ACTIONS = 16
base.ACTION_REPEAT = 400
base.DATA_SCHEMA = "ant-maze-modebench-data-v11"
base.VERSION_LABEL = "ant-maze-v11-data"


if __name__ == "__main__":
    base.main()
