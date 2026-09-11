#!/usr/bin/env python3
"""Materialize the unchanged Ant v15 route slate under stable v18."""

from __future__ import annotations

import os
from pathlib import Path

import make_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v18 import controller_receipt_sha256
from oat_drgrpo.maze_modebench_process_v18 import MazeVerifierProcessV18


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
        path = path.with_name("maze_modebench_worker_v18.py")
    return _base_sha256_file(path)


class _V18VerifierProcess(MazeVerifierProcessV18):
    def __init__(self, *, timeout_seconds=180.0, **kwargs):
        super().__init__(timeout_seconds=180.0, **kwargs)


base._sha256_file = _sha256_file
base.MazeVerifierProcess = _V18VerifierProcess

# Exact controller-only rebind of the prospectively frozen v15 route slate.
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
base.ANT_WORKER_SOURCE = "ant_maze_worker_v18.py"
base.DEFAULT_OUTPUT = (
    base.ROOT / "var/data/ant_maze_modebench_v15_controller_v18"
)
base.RESET_SEED_BASE = 108_500
base.MAP_ID_PREFIX = "ant_v15_controller_v18"
base.MIN_ACTIONS = 8
base.MAX_ACTIONS = 20
base.ACTION_REPEAT = 400
base.DATA_SCHEMA = "ant-maze-modebench-data-v15-controller-v18"
base.VERSION_LABEL = "ant-maze-v15-controller-v18-data"


if __name__ == "__main__":
    base.main()
