#!/usr/bin/env python3
"""Audit the frozen 9x9 fresh-map AntMaze v9 route slate."""

from __future__ import annotations

import audit_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v9 import controller_receipt_sha256


base.CONTROLLER_RECEIPT_SHA256 = controller_receipt_sha256()
base.DATA_SCHEMA = "ant-maze-modebench-data-v9"
base.AUDIT_SCHEMA = "ant-maze-modebench-admission-audit-v9"
base.DEFAULT_DATA_ROOT = base.ROOT / "var/data/ant_maze_modebench_v9"
base.DEFAULT_OUTPUT = (
    base.ROOT / "var/artifacts/ant_maze_modebench_v9_admission_audit.json"
)
base.ANT_WORKER_SOURCE = "ant_maze_worker_v9.py"
base.DECISION = "admitted_to_v9_cross_node_route_determinism_gate"
base.VERSION_LABEL = "ant-maze-v9-audit"


if __name__ == "__main__":
    base.main()
