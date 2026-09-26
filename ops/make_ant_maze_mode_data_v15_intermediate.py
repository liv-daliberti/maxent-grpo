#!/usr/bin/env python3
"""Materialize the prospective intermediate-difficulty AntMaze v15 slate."""

from __future__ import annotations

import os

import make_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v12 import controller_receipt_sha256


def _controller_receipt() -> str:
    identity = os.environ.pop("OAT_ZERO_PROTOCOL_IDENTITY", None)
    try:
        return controller_receipt_sha256()
    finally:
        if identity is not None:
            os.environ["OAT_ZERO_PROTOCOL_IDENTITY"] = identity


class _IntermediateVerifierProcess(base.MazeVerifierProcess):
    def __init__(self, *, timeout_seconds=180.0, **kwargs):
        super().__init__(timeout_seconds=180.0, **kwargs)


base.MazeVerifierProcess = _IntermediateVerifierProcess

# V12 used four decisions around one blocked cell and saturated before
# learning. V14 used fourteen decisions around a 5x5 block and exceeded the
# controller's reliable envelope. V15 prospectively brackets those outcomes:
# eight decisions around a 3x3 block, using the unchanged v12 controller.
base.UPPER = ("N",) * 2 + ("E",) * 4 + ("S",) * 2
base.LOWER = ("S",) * 2 + ("E",) * 4 + ("N",) * 2
base.PERIPHERAL_CELLS = (
    (1, 1),
    (2, 1),
    (3, 1),
    (9, 1),
    (10, 1),
    (11, 1),
    (1, 11),
    (11, 11),
)
base.MAP_SIZE = 13
base.STATIC_WALLS = tuple(
    (row, column)
    for row in range(5, 8)
    for column in range(5, 8)
)
base.RESET_CELL = (6, 4)
base.GOAL_CELL = (6, 8)
base.BOUNDS_XY = ((-24.0, 24.0), (-24.0, 24.0))
base.CONTROLLER_RECEIPT_SHA256 = _controller_receipt()
base.ANT_WORKER_SOURCE = "ant_maze_worker_v12.py"
base.DEFAULT_OUTPUT = base.ROOT / "var/data/ant_maze_modebench_v15_intermediate"
base.RESET_SEED_BASE = 108_500
base.MAP_ID_PREFIX = "ant_v15_intermediate"
base.MIN_ACTIONS = 8
base.MAX_ACTIONS = 20
base.ACTION_REPEAT = 400
base.DATA_SCHEMA = "ant-maze-modebench-data-v15-intermediate"
base.VERSION_LABEL = "ant-maze-v15-intermediate-data"


if __name__ == "__main__":
    base.main()
