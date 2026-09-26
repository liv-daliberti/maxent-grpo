#!/usr/bin/env python3
"""Audit the unchanged Ant v15 route slate under controller v19."""

from __future__ import annotations

import os
from pathlib import Path

import audit_ant_maze_mode_data as base
from oat_drgrpo.ant_maze_worker_v19 import controller_receipt_sha256
from oat_drgrpo.maze_modebench_process_v19 import MazeVerifierProcessV19


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
        path = path.with_name("maze_modebench_worker_v19.py")
    return _base_sha256_file(path)


class _V19VerifierProcess(MazeVerifierProcessV19):
    def __init__(self, *, timeout_seconds=180.0, **kwargs):
        super().__init__(timeout_seconds=180.0, **kwargs)


base._sha256_file = _sha256_file
base.MazeVerifierProcess = _V19VerifierProcess
base.CONTROLLER_RECEIPT_SHA256 = _controller_receipt()
base.DATA_SCHEMA = "ant-maze-modebench-data-v15-controller-v19"
base.AUDIT_SCHEMA = "ant-maze-modebench-admission-audit-v15-controller-v19"
base.DEFAULT_DATA_ROOT = (
    base.ROOT / "var/data/ant_maze_modebench_v15_controller_v19"
)
base.DEFAULT_OUTPUT = (
    base.ROOT
    / "var/artifacts/"
    "ant_maze_modebench_v15_controller_v19_admission_audit.json"
)
base.ANT_WORKER_SOURCE = "ant_maze_worker_v19.py"
base.DECISION = "admitted_to_ant_v15_v19_frozen_model_viability_gate"
base.VERSION_LABEL = "ant-maze-v15-controller-v19-audit"


if __name__ == "__main__":
    base.main()
