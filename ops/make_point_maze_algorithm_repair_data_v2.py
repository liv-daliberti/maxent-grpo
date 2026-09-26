#!/usr/bin/env python3
"""Materialize the balanced medium/hard PointMaze algorithm-repair slate."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import make_point_maze_mode_data as base
from make_point_maze_algorithm_repair_data_v1 import repair_families
from make_point_maze_geometry_shift_data import geometry_shift_families


ROOT = Path(__file__).resolve().parents[1]


def balanced_families() -> list[dict]:
    medium = {
        family["family"]: family for family in geometry_shift_families()
    }
    hard = {family["family"]: family for family in repair_families()}
    selected = (
        ("cross9_balanced", medium["cross9_shift"]),
        ("upper_offset9_balanced", medium["upper_offset9_shift"]),
        ("block11_balanced", hard["block11_repair"]),
        ("bar11_balanced", hard["bar11_repair"]),
    )
    families = []
    for new_name, source in selected:
        family = deepcopy(source)
        family["family"] = new_name
        families.append(family)
    return families


def main() -> None:
    base.DEFAULT_OUTPUT = ROOT / "var/data/point_maze_algorithm_repair_v2"
    base._base_families = balanced_families
    base.main()


if __name__ == "__main__":
    main()
