#!/usr/bin/env python3
"""Audit AntMaze v14-hard without leaking protocol identity to controller load."""

from __future__ import annotations

import os

import audit_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v12 import controller_receipt_sha256


def _controller_receipt() -> str:
    identity = os.environ.pop("OAT_ZERO_PROTOCOL_IDENTITY", None)
    try:
        return controller_receipt_sha256()
    finally:
        if identity is not None:
            os.environ["OAT_ZERO_PROTOCOL_IDENTITY"] = identity


class _HardVerifierProcess(base.MazeVerifierProcess):
    def __init__(self, *, timeout_seconds=180.0, **kwargs):
        super().__init__(timeout_seconds=180.0, **kwargs)


base.MazeVerifierProcess = _HardVerifierProcess
base.CONTROLLER_RECEIPT_SHA256 = _controller_receipt()
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

