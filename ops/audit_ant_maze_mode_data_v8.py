#!/usr/bin/env python3
"""Audit the frozen fresh-map AntMaze v8 route slate."""

from __future__ import annotations

import audit_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v8 import RECEIPT_SHA256


base.CONTROLLER_RECEIPT_SHA256 = RECEIPT_SHA256
base.DATA_SCHEMA = "ant-maze-modebench-data-v8"
base.AUDIT_SCHEMA = "ant-maze-modebench-admission-audit-v8"
base.DEFAULT_DATA_ROOT = base.ROOT / "var/data/ant_maze_modebench_v8"
base.DEFAULT_OUTPUT = (
    base.ROOT / "var/artifacts/ant_maze_modebench_v8_admission_audit.json"
)
base.ANT_WORKER_SOURCE = "ant_maze_worker_v8.py"
base.DECISION = "admitted_to_v8_cross_node_route_determinism_gate"
base.VERSION_LABEL = "ant-maze-v8-audit"


if __name__ == "__main__":
    base.main()
