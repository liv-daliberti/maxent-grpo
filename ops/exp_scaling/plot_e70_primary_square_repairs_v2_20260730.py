#!/usr/bin/env python3
"""Label and live-gate correction for the square primary E70 surface."""

from __future__ import annotations

import plot_e70_primary_square_repairs_20260730 as primary


LABELS = {
    "point_maze_geometry_shift": "PointMaze\ngeometry shift",
    "pantry_support_retention_repair": "PantryPlan\nsupport retention",
    "point_maze_algorithm_repair": "PointMaze\nalgorithm repair",
    "ant_maze_harder_repair": "AntMaze\nharder task",
}
primary.ROWS = tuple(
    (
        LABELS.get(domain, label),
        note,
        domain,
        prefix,
        steps,
        seeds,
    )
    for label, note, domain, prefix, steps, seeds in primary.ROWS
)

_status = primary._status


def status(domain: str):
    if domain == "ant_maze_harder_repair":
        identity = primary.repair._read(
            primary.ROOT
            / "var/artifacts/"
            "ant_sequential_waypoint_controller_v17_identity.json"
        )
        receipt = primary.repair._read(
            primary.ROOT
            / "var/maze_runtime/controllers/"
            "ant_sequential_waypoint_v17.evaluation.json"
        )
        if receipt.get("status") == "pass":
            return (
                "V17 CONTROLLER GATE PASSED",
                "the unchanged harder v15 route slate is eligible for fresh executable admission",
                "#008A5A",
            )
        if receipt.get("status") == "fail":
            return (
                "V17 CONTROLLER GATE FAILED",
                "the harder AntMaze repair remains fail-closed",
                "#B91C1C",
            )
        return (
            "RETENTION-ANCHORED V17 CONTROLLER TRAINING",
            f"identity-bound job {identity.get('job_id', 'unknown')}; "
            "v16 failed at 25%; a new 96-episode gate controls route admission",
            "#B36B00",
        )
    return _status(domain)


primary._status = status


if __name__ == "__main__":
    primary.render()
