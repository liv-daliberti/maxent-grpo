#!/usr/bin/env python3
"""Materialize the frozen fresh-map AntMaze route slate for controller v8."""

from __future__ import annotations

import make_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v8 import RECEIPT_SHA256


base.UPPER = ("N", "E", "E", "S")
base.LOWER = ("S", "E", "E", "N")
base.PERIPHERAL_CELLS = (
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (1, 2),
    (1, 3),
    (1, 4),
)
base.CONTROLLER_RECEIPT_SHA256 = RECEIPT_SHA256
base.ANT_WORKER_SOURCE = "ant_maze_worker_v8.py"
base.DEFAULT_OUTPUT = base.ROOT / "var/data/ant_maze_modebench_v8"
base.RESET_SEED_BASE = 87_300
base.MAP_ID_PREFIX = "ant_v8_admission"
base.MIN_ACTIONS = 4
base.MAX_ACTIONS = 16
base.ACTION_REPEAT = 400
base.DATA_SCHEMA = "ant-maze-modebench-data-v8"
base.VERSION_LABEL = "ant-maze-v8-data"


if __name__ == "__main__":
    base.main()
