#!/usr/bin/env python3
"""Bind the proven three-node aggregate to AntMaze v12."""

import aggregate_ant_maze_cross_node_v10 as base

base.VERSION = "v12"
base.REPLICA_SCHEMA = "ant-maze-v12-cross-node-replica-v1"
base.AUDIT_SCHEMA = "ant-maze-v12-cross-node-audit-v1"
base.PASS_DECISION = "eligible_for_frozen_05b_viability_gate_v12"
base.FAIL_DECISION = "ant_maze_v12_stopped"

if __name__ == "__main__":
    base.main()
