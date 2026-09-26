#!/usr/bin/env python3
"""Materialize frozen Ant v15 with v17 and the long-route timeout."""

from __future__ import annotations

import os
from pathlib import Path

import make_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v17_r1 import controller_receipt_sha256
from oat_drgrpo.maze_modebench_process_v17_r1 import MazeVerifierProcessV17R1


def _controller_receipt() -> str:
    identity = os.environ.pop("OAT_ZERO_PROTOCOL_IDENTITY", None)
    try:
        return controller_receipt_sha256()
    finally:
        if identity is not None:
            os.environ["OAT_ZERO_PROTOCOL_IDENTITY"] = identity


_base_sha256_file = base._sha256_file


def _sha256_file(path: Path) -> str:
    if path.name == "maze_modebench_worker.py":
        path = path.with_name("maze_modebench_worker_v17_r1.py")
    return _base_sha256_file(path)


class _V17VerifierProcess(MazeVerifierProcessV17R1):
    def __init__(self, *, timeout_seconds=180.0, **kwargs):
        super().__init__(timeout_seconds=180.0, **kwargs)


base._sha256_file = _sha256_file
base.MazeVerifierProcess = _V17VerifierProcess
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
base.ANT_WORKER_SOURCE = "ant_maze_worker_v17_r1.py"
base.DEFAULT_OUTPUT = (
    base.ROOT / "var/data/ant_maze_modebench_v15_controller_v17_r1"
)
base.RESET_SEED_BASE = 108_500
base.MAP_ID_PREFIX = "ant_v15_controller_v17_r1"
base.MIN_ACTIONS = 8
base.MAX_ACTIONS = 20
base.ACTION_REPEAT = 400
base.DATA_SCHEMA = "ant-maze-modebench-data-v15-controller-v17-r1"
base.VERSION_LABEL = "ant-maze-v15-controller-v17-r3-data"


if __name__ == "__main__":
    base.main()
