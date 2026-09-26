#!/usr/bin/env python3
"""Audit executable admission for PointMaze v4 horizontal repair."""

from __future__ import annotations

import json
import os
from pathlib import Path

import audit_point_maze_mode_data as base


ROOT = Path(
    os.environ.get(
        "OAT_ZERO_REPO_ROOT",
        Path(__file__).resolve().parents[1],
    )
)
SOURCE_ROOT = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
base.ROOT = ROOT
base.SRC = SOURCE_ROOT
DATA_ROOT = ROOT / "var/data/point_maze_algorithm_repair_v4"
OUTPUT = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v4_admission_audit.json"
)


def main() -> None:
    identity = json.loads(
        (DATA_ROOT / "identity.json").read_text(encoding="utf-8")
    )
    expected_counts = {
        "train": {"0": 2, "1": 2, "2": 2, "3": 2},
        "dev": {"0": 1, "1": 1, "2": 1, "3": 1},
        "eval": {"0": 1, "1": 1, "2": 1, "3": 1},
    }
    if (
        identity.get("schema_version")
        != "point-maze-algorithm-repair-data-v4"
        or identity.get("orientation_counts") != expected_counts
        or identity.get("orientation_balanced_within_each_split") is not True
        or identity.get("executable_task_overlap_count") != 0
        or identity.get("horizontal_geometry_repair") is not True
        or identity.get("horizontal_family_size") != 13
        or identity.get("horizontal_route_counts") != [13, 32, 16]
    ):
        raise RuntimeError("PointMaze v4 data identity drift")
    base.DEFAULT_DATA = DATA_ROOT
    base.DEFAULT_OUTPUT = OUTPUT
    base.main()
    receipt = json.loads(OUTPUT.read_text(encoding="utf-8"))
    receipt.update(
        schema_version=(
            "point-maze-algorithm-repair-v4-admission-audit-v1"
        ),
        decision="admitted_to_point_maze_v4_horizontal_viability_gate",
        orientation_counts=expected_counts,
        orientation_balanced_within_each_split=True,
        executable_task_overlap_count=0,
        horizontal_geometry_repair=True,
        horizontal_family_size=13,
        horizontal_route_counts=[13, 32, 16],
    )
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)


if __name__ == "__main__":
    main()
