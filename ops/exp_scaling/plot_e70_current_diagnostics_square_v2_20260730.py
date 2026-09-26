#!/usr/bin/env python3
"""Bind the diagnostic companion page to the live Ant v17 gate."""

from __future__ import annotations

import plot_e70_current_diagnostics_square_20260730 as diagnostics


ROOT = diagnostics.ROOT
diagnostics.primary.repair.REPAIR_INPUTS["ant_protocol"] = (
    ROOT
    / "paper/preregistration/"
    "ant_sequential_waypoint_controller_v17_20260730.md"
)
diagnostics.primary.repair.REPAIR_INPUTS["ant_identity"] = (
    ROOT
    / "var/artifacts/"
    "ant_sequential_waypoint_controller_v17_identity.json"
)
diagnostics.primary.repair.REPAIR_INPUTS["ant_controller_gate"] = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_sequential_waypoint_v17.evaluation.json"
)
diagnostics.primary.repair.REPAIR_INPUTS["ant_v16_failed_gate"] = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_sequential_waypoint_v16.evaluation.json"
)


if __name__ == "__main__":
    diagnostics.render()
