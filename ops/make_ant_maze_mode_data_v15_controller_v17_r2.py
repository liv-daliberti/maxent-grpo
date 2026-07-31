#!/usr/bin/env python3
"""Timeout-corrected materializer for frozen Ant v15 under controller v17."""

from __future__ import annotations

import make_ant_maze_mode_data_v15_controller_v17_r1 as v1
from oat_drgrpo.maze_modebench_process_v17_r1 import MazeVerifierProcessV17R1


class _V17VerifierProcess(MazeVerifierProcessV17R1):
    def __init__(self, *, timeout_seconds=180.0, **kwargs):
        super().__init__(timeout_seconds=180.0, **kwargs)


v1.base.MazeVerifierProcess = _V17VerifierProcess
v1.base.VERSION_LABEL = "ant-maze-v15-controller-v17-r2-data"


if __name__ == "__main__":
    v1.base.main()
