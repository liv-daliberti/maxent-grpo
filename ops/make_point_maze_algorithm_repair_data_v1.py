#!/usr/bin/env python3
"""Materialize unseen 11x11 PointMaze geometries for algorithm repair."""

from __future__ import annotations

from pathlib import Path

import make_point_maze_mode_data as base


ROOT = Path(__file__).resolve().parents[1]


def repair_families() -> list[dict]:
    def map_with(obstacles: set[tuple[int, int]]) -> list[list[int]]:
        return [
            [
                1
                if row in {0, 10}
                or column in {0, 10}
                or (row, column) in obstacles
                else 0
                for column in range(11)
            ]
            for row in range(11)
        ]

    definitions = (
        (
            "block11_repair",
            {(row, column) for row in range(3, 8) for column in range(3, 8)},
        ),
        (
            "wide_block11_repair",
            {(row, column) for row in range(4, 7) for column in range(2, 9)},
        ),
        (
            "bar11_repair",
            {(5, column) for column in range(2, 9)},
        ),
        (
            "diamond11_repair",
            {
                (row, column)
                for row in range(2, 9)
                for column in range(2, 9)
                if abs(row - 5) + abs(column - 5) <= 3
            },
        ),
    )
    families = []
    for family, obstacles in definitions:
        maze_map = map_with(obstacles)
        assert maze_map[5][1] == 0 and maze_map[5][9] == 0
        families.append(
            {
                "family": family,
                "maze_map": maze_map,
                "reset": (5, 1),
                "goal": (5, 9),
                "bounds": [[-5.5, 5.5], [-5.5, 5.5]],
                "spans": ((2.0, 5.0), (-5.0, -2.0)),
                "counts": (12, 32, 15),
            }
        )
    return families


def main() -> None:
    base.DEFAULT_OUTPUT = ROOT / "var/data/point_maze_algorithm_repair_v1"
    base._base_families = repair_families
    base.main()


if __name__ == "__main__":
    main()

