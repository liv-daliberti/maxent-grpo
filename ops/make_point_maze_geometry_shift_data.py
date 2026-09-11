#!/usr/bin/env python3
"""Materialize the prospective PointMaze geometry-shift replacement row."""

from __future__ import annotations

from pathlib import Path

import make_point_maze_mode_data as base


ROOT = Path(__file__).resolve().parents[1]


def geometry_shift_families() -> list[dict]:
    def map_with(obstacles: set[tuple[int, int]]) -> list[list[int]]:
        return [
            [
                1 if row in {0, 8} or column in {0, 8} or (row, column) in obstacles else 0
                for column in range(9)
            ]
            for row in range(9)
        ]

    definitions = (
        (
            "wide_block9_shift",
            {(row, column) for row in range(3, 6) for column in range(2, 7)},
        ),
        (
            "cross9_shift",
            {(4, column) for column in range(2, 7)}
            | {(row, 4) for row in range(2, 7)},
        ),
        (
            "upper_offset9_shift",
            {(row, column) for row in range(2, 5) for column in range(2, 6)},
        ),
        (
            "lower_offset9_shift",
            {(row, column) for row in range(4, 7) for column in range(3, 7)},
        ),
    )
    families = []
    for family, obstacles in definitions:
        maze_map = map_with(obstacles)
        assert maze_map[4][1] == 0 and maze_map[4][7] == 0
        assert maze_map[0] == [1] * 9 and maze_map[8] == [1] * 9
        assert maze_map[1][1:-1] == [0] * 7 and maze_map[7][1:-1] == [0] * 7
        families.append(
            {
                "family": family,
                "maze_map": maze_map,
                "reset": (4, 1),
                "goal": (4, 7),
                "bounds": [[-4.5, 4.5], [-4.5, 4.5]],
                "spans": ((1.2, 3.4), (-3.4, -1.2)),
                "counts": (8, 24, 11),
            }
        )
    return families


def main() -> None:
    base.DEFAULT_OUTPUT = ROOT / "var/data/point_maze_geometry_shift_v1"
    base._base_families = geometry_shift_families
    base.main()


if __name__ == "__main__":
    main()
