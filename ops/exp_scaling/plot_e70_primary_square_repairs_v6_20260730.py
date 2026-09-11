#!/usr/bin/env python3
"""Prevent the base renderer from resetting the live Ant v18 binding."""

from __future__ import annotations

import plot_e70_primary_square_repairs_v5_20260730 as v5


primary = v5.primary
repair = v5.repair
ROOT = v5.ROOT
PROTECTED = {
    "ant_protocol",
    "ant_identity",
    "ant_controller_gate",
    "ant_v17_failed_gate",
}


class RepairBindings(dict):
    def __setitem__(self, key, value):
        if key in PROTECTED and key in self:
            return
        super().__setitem__(key, value)


# Restore v18 once, then ignore the older renderer's in-function v17 writes.
bindings = RepairBindings(repair.REPAIR_INPUTS)
bindings["ant_protocol"] = (
    ROOT
    / "paper/preregistration/"
    "ant_stable_handoff_controller_v18_20260730.md"
)
bindings["ant_identity"] = (
    ROOT / "var/artifacts/ant_stable_handoff_controller_v18_identity.json"
)
bindings["ant_controller_gate"] = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_stable_handoff_v18.evaluation.json"
)
bindings["ant_v17_failed_gate"] = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_sequential_waypoint_v17.evaluation.json"
)
repair.REPAIR_INPUTS = bindings

_summary = repair._repair_summary


def summary():
    result = _summary()
    identity = repair._read(bindings["ant_identity"])
    gate = repair._read(bindings["ant_controller_gate"])
    result["ant_controller_job_id"] = (
        f"30205033 → v18 {identity.get('job_id', 'unknown')}"
    )
    result["ant_controller_gate_status"] = (
        f"failed → v18 {gate.get('status') or 'running'}"
    )
    return result


repair._repair_summary = summary


if __name__ == "__main__":
    primary.render()
