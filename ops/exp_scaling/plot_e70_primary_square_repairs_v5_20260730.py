#!/usr/bin/env python3
"""Show sealed Ant v17 failure and the live stable-handoff v18 repair."""

from __future__ import annotations

import plot_e70_primary_square_repairs_v4_20260730 as v4


primary = v4.v3.primary
repair = primary.repair
ROOT = primary.ROOT
repair.REPAIR_INPUTS["ant_protocol"] = (
    ROOT
    / "paper/preregistration/"
    "ant_stable_handoff_controller_v18_20260730.md"
)
repair.REPAIR_INPUTS["ant_identity"] = (
    ROOT / "var/artifacts/ant_stable_handoff_controller_v18_identity.json"
)
repair.REPAIR_INPUTS["ant_controller_gate"] = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_stable_handoff_v18.evaluation.json"
)
repair.REPAIR_INPUTS["ant_v17_failed_gate"] = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_sequential_waypoint_v17.evaluation.json"
)

_summary = repair._repair_summary


def summary():
    result = _summary()
    identity = repair._read(repair.REPAIR_INPUTS["ant_identity"])
    gate = repair._read(repair.REPAIR_INPUTS["ant_controller_gate"])
    result["ant_controller_job_id"] = (
        f"30205033 failed → v18 {identity.get('job_id', 'unknown')}"
    )
    result["ant_controller_gate_status"] = (
        gate.get("status") or "running"
    )
    return result


repair._repair_summary = summary
_status = primary._status


def status(domain: str):
    if domain != "ant_maze_harder_repair":
        return _status(domain)
    identity = repair._read(repair.REPAIR_INPUTS["ant_identity"])
    receipt = repair._read(repair.REPAIR_INPUTS["ant_controller_gate"])
    if receipt.get("status") == "pass":
        return (
            "STABLE-HANDOFF V18 GATE PASSED",
            "the unchanged harder v15 route slate is eligible for a fresh binding",
            "#008A5A",
        )
    if receipt.get("status") == "fail":
        return (
            "STABLE-HANDOFF V18 GATE FAILED",
            "v17 and v18 are retained as negative controller results",
            "#B91C1C",
        )
    return (
        "STABLE-HANDOFF V18 CONTROLLER TRAINING",
        f"v17 failed 21/96 sequences; identity-bound v18 job "
        f"{identity.get('job_id', 'unknown')} requires stable arrivals and "
        "a new disjoint 96-sequence gate",
        "#B36B00",
    )


primary._status = status


if __name__ == "__main__":
    primary.render()
