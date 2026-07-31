#!/usr/bin/env python3
"""Bind the proven cross-node spec exporter to admitted AntMaze v12."""

import export_ant_maze_v10_cross_node_specs as base

base.VERSION = "v12"
base.DATA_SCHEMA = "ant-maze-modebench-data-v12"
base.AUDIT_DECISION = "admitted_to_v12_cross_node_route_determinism_gate"
base.ROUTE_SCHEMA = "ant-maze-v12-route-generation-identity-v1"
base.EXPORT_SCHEMA = "ant-maze-v12-cross-node-spec-export-v1"
base.CONTROLLER_IDENTITY_FIELD = "executor_identity_sha256"

if __name__ == "__main__":
    base.main()
