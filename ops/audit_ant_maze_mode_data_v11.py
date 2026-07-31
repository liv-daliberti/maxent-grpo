#!/usr/bin/env python3
"""Audit the unchanged, previously unexecuted 11x11 AntMaze v11 route slate."""

from __future__ import annotations

import audit_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v11 import controller_receipt_sha256


base.CONTROLLER_RECEIPT_SHA256 = controller_receipt_sha256()
base.DATA_SCHEMA = "ant-maze-modebench-data-v11"
base.AUDIT_SCHEMA = "ant-maze-modebench-admission-audit-v11"
base.DEFAULT_DATA_ROOT = base.ROOT / "var/data/ant_maze_modebench_v11"
base.DEFAULT_OUTPUT = base.ROOT / "var/artifacts/ant_maze_modebench_v11_admission_audit.json"
base.ANT_WORKER_SOURCE = "ant_maze_worker_v11.py"
base.DECISION = "admitted_to_v11_cross_node_route_determinism_gate"
base.VERSION_LABEL = "ant-maze-v11-audit"


if __name__ == "__main__":
    base.main()
