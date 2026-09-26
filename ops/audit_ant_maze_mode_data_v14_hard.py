#!/usr/bin/env python3
"""Audit the prospective harder AntMaze v14 route slate."""

from __future__ import annotations

import audit_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v12 import controller_receipt_sha256


class _HardVerifierProcess(base.MazeVerifierProcess):
    def __init__(self, *, timeout_seconds=180.0, **kwargs):
        super().__init__(timeout_seconds=180.0, **kwargs)


base.MazeVerifierProcess = _HardVerifierProcess
base.CONTROLLER_RECEIPT_SHA256 = controller_receipt_sha256()
base.DATA_SCHEMA = "ant-maze-modebench-data-v14-hard"
base.AUDIT_SCHEMA = "ant-maze-modebench-admission-audit-v14-hard"
base.DEFAULT_DATA_ROOT = base.ROOT / "var/data/ant_maze_modebench_v14_hard"
base.DEFAULT_OUTPUT = (
    base.ROOT / "var/artifacts/ant_maze_modebench_v14_hard_admission_audit.json"
)
base.ANT_WORKER_SOURCE = "ant_maze_worker_v12.py"
base.DECISION = "admitted_to_harder_antmaze_frozen_model_viability_gate"
base.VERSION_LABEL = "ant-maze-v14-hard-audit"


if __name__ == "__main__":
    base.main()

