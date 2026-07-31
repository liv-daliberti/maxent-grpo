#!/usr/bin/env python3
"""Audit the anchored v12 executor on the unchanged 11x11 route slate."""

from __future__ import annotations

import audit_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v12 import controller_receipt_sha256


base.CONTROLLER_RECEIPT_SHA256 = controller_receipt_sha256()
base.DATA_SCHEMA = "ant-maze-modebench-data-v12"
base.AUDIT_SCHEMA = "ant-maze-modebench-admission-audit-v12"
base.DEFAULT_DATA_ROOT = base.ROOT / "var/data/ant_maze_modebench_v12"
base.DEFAULT_OUTPUT = base.ROOT / "var/artifacts/ant_maze_modebench_v12_admission_audit.json"
base.ANT_WORKER_SOURCE = "ant_maze_worker_v12.py"
base.DECISION = "admitted_to_v12_cross_node_route_determinism_gate"
base.VERSION_LABEL = "ant-maze-v12-audit"


if __name__ == "__main__":
    base.main()
