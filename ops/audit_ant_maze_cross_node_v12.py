#!/usr/bin/env python3
"""Bind the proven one-node exact-slate replay to AntMaze v12."""

import audit_ant_maze_cross_node_v10 as base
from oat_drgrpo.ant_maze_worker_v12 import controller_receipt_sha256, execute_ant_v12_raw

base.VERSION = "v12"
base.EXPORT_SCHEMA = "ant-maze-v12-cross-node-spec-export-v1"
base.REPLICA_SCHEMA = "ant-maze-v12-cross-node-replica-v1"
base.controller_receipt_sha256 = controller_receipt_sha256
base.execute_ant_v10_raw = execute_ant_v12_raw

if __name__ == "__main__":
    base.main()
