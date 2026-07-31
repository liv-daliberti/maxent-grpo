#!/usr/bin/env python3
"""Audit the frozen Ant v15 slate under the admitted v17 controller."""

from __future__ import annotations

import os
from pathlib import Path

import audit_ant_maze_mode_data as base
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


base._sha256_file = _sha256_file
base.MazeVerifierProcess = MazeVerifierProcessV17R1
base.CONTROLLER_RECEIPT_SHA256 = _controller_receipt()
base.DATA_SCHEMA = "ant-maze-modebench-data-v15-controller-v17-r1"
base.AUDIT_SCHEMA = (
    "ant-maze-modebench-admission-audit-v15-controller-v17-r1"
)
base.DEFAULT_DATA_ROOT = (
    base.ROOT / "var/data/ant_maze_modebench_v15_controller_v17_r1"
)
base.DEFAULT_OUTPUT = (
    base.ROOT
    / "var/artifacts/"
    "ant_maze_modebench_v15_controller_v17_r1_admission_audit.json"
)
base.ANT_WORKER_SOURCE = "ant_maze_worker_v17_r1.py"
base.DECISION = "admitted_to_ant_v15_v17_frozen_model_viability_gate"
base.VERSION_LABEL = "ant-maze-v15-controller-v17-r1-audit"


if __name__ == "__main__":
    base.main()
