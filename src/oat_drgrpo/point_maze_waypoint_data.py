"""Deterministic multi-route task generation for PointMaze waypoint pilots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Mapping, Sequence

from .point_maze_waypoint import (
    POINT_WAYPOINT_VERIFIER,
    find_point_waypoint_route_programs,
    make_point_waypoint_spec,
    render_point_waypoint_problem,
)


DEFAULT_POINT_WAYPOINT_SPLIT_COUNTS = {
    "train": 64,
    "dev": 32,
    "eval": 64,
}
POINT_WAYPOINT_DATA_SEED = 88_100


@dataclass(frozen=True)
class PointWaypointTask:
    split: str
    family: str
    instance_fingerprint: str
    spec: dict[str, Any]
    problem: str
    route_programs: dict[str, str]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _rotate_map_cw(maze_map: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*[list(row) for row in maze_map][::-1])]


def _rotate_cell_cw(cell: tuple[int, int], size: int) -> tuple[int, int]:
    return cell[1], size - 1 - cell[0]


def _rotate_point_cw(point: tuple[float, float]) -> tuple[float, float]:
    return point[1], -point[0]


def _rotate_gate_cw(gate: Mapping[str, Any]) -> dict[str, Any]:
    axis = str(gate["axis"])
    coordinate = float(gate["coordinate"])
    low, high = (float(value) for value in gate["span"])
    endpoints = (
        ((coordinate, low), (coordinate, high))
        if axis == "x"
        else ((low, coordinate), (high, coordinate))
    )
    first, second = (_rotate_point_cw(point) for point in endpoints)
    if abs(first[0] - second[0]) < 1e-12:
        new_axis = "x"
        new_coordinate = first[0]
        new_span = sorted((first[1], second[1]))
    else:
        new_axis = "y"
        new_coordinate = first[1]
        new_span = sorted((first[0], second[0]))
    return {
        "id": str(gate["id"]),
        "axis": new_axis,
        "coordinate": new_coordinate,
        "span": new_span,
        "hysteresis": float(gate["hysteresis"]),
    }


def _rotated_geometry(
    *,
    maze_map: list[list[int]],
    reset_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    gates: list[dict[str, Any]],
    rotations: int,
) -> tuple[
    list[list[int]],
    tuple[int, int],
    tuple[int, int],
    list[dict[str, Any]],
]:
    size = len(maze_map)
    for _ in range(int(rotations) % 4):
        maze_map = _rotate_map_cw(maze_map)
        reset_cell = _rotate_cell_cw(reset_cell, size)
        goal_cell = _rotate_cell_cw(goal_cell, size)
        gates = [_rotate_gate_cw(gate) for gate in gates]
    return maze_map, reset_cell, goal_cell, gates


def _barrier_candidate(
    rng: random.Random,
    *,
    rotation: int,
) -> tuple[
    list[list[int]],
    tuple[int, int],
    tuple[int, int],
    list[dict[str, Any]],
    int,
    int,
]:
    size = rng.choice((9, 11, 13))
    possible_openings = list(range(1, size - 1, 2))
    maximum_modes = min(5, len(possible_openings))
    mode_count = rng.randint(3, maximum_modes)
    openings = sorted(rng.sample(possible_openings, mode_count))
    non_opening_rows = [row for row in range(1, size - 1) if row not in set(openings)]
    reset_row = rng.choice(non_opening_rows)
    goal_row = rng.choice(non_opening_rows)
    barrier_column = size // 2
    maze_map = [
        [
            int(
                row in {0, size - 1}
                or column in {0, size - 1}
                or (column == barrier_column and row not in set(openings))
            )
            for column in range(size)
        ]
        for row in range(size)
    ]
    center = (size - 1) / 2.0
    gates = [
        {
            "id": f"corridor_{index}",
            "axis": "x",
            "coordinate": 0.0,
            "span": [center - row - 0.42, center - row + 0.42],
            "hysteresis": 0.1,
        }
        for index, row in enumerate(openings)
    ]
    rotation = int(rotation) % 4
    maze_map, reset_cell, goal_cell, gates = _rotated_geometry(
        maze_map=maze_map,
        reset_cell=(reset_row, 1),
        goal_cell=(goal_row, size - 2),
        gates=gates,
        rotations=rotation,
    )
    return maze_map, reset_cell, goal_cell, gates, mode_count, rotation


def _instance_fingerprint(spec: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "maze_map": spec["maze_map"],
            "reset_cell": spec["reset_cell"],
            "goal_cell": spec["goal_cell"],
            "route_gates": spec["route_gates"],
        }
    )


def generate_point_waypoint_tasks(
    *,
    environment_sha256: str,
    split_counts: Mapping[str, int] = DEFAULT_POINT_WAYPOINT_SPLIT_COUNTS,
    seed: int = POINT_WAYPOINT_DATA_SEED,
    excluded_fingerprints: Sequence[str] = (),
) -> dict[str, list[PointWaypointTask]]:
    """Generate nonoverlapping tasks with three to five simple route modes."""

    counts = {str(split): int(count) for split, count in split_counts.items()}
    if set(counts) != {"train", "dev", "eval"} or any(
        count <= 0 for count in counts.values()
    ):
        raise ValueError("waypoint splits must contain positive train/dev/eval counts")
    rng = random.Random(int(seed))
    observed = {str(fingerprint) for fingerprint in excluded_fingerprints}
    result: dict[str, list[PointWaypointTask]] = {
        split: [] for split in ("train", "dev", "eval")
    }
    split_seed_base = {"train": 881_000, "dev": 882_000, "eval": 883_000}
    for split in ("train", "dev", "eval"):
        attempts = 0
        while len(result[split]) < counts[split]:
            attempts += 1
            if attempts > counts[split] * 10_000:
                raise RuntimeError("could not generate enough unique waypoint tasks")
            index = len(result[split])
            (
                maze_map,
                reset_cell,
                goal_cell,
                gates,
                expected_modes,
                rotation,
            ) = _barrier_candidate(rng, rotation=index % 4)
            spec = make_point_waypoint_spec(
                environment_sha256=environment_sha256,
                map_id=f"barrier_{expected_modes}mode_{split}_{index:03d}_r{rotation}",
                maze_map=maze_map,
                reset_cell=reset_cell,
                goal_cell=goal_cell,
                reset_seed=split_seed_base[split] + index,
                route_gates=gates,
                max_actions=min(64, 4 * (len(maze_map) - 2)),
            )
            fingerprint = _instance_fingerprint(spec)
            if fingerprint in observed:
                continue
            programs = find_point_waypoint_route_programs(spec)
            if len(programs) != expected_modes or any(
                "," in route for route in programs
            ):
                continue
            observed.add(fingerprint)
            result[split].append(
                PointWaypointTask(
                    split=split,
                    family=f"barrier_{expected_modes}mode",
                    instance_fingerprint=fingerprint,
                    spec=spec,
                    problem=render_point_waypoint_problem(spec),
                    route_programs=programs,
                )
            )
    return result


def point_waypoint_task_rows(
    tasks: Mapping[str, Sequence[PointWaypointTask]],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, split_tasks in tasks.items():
        rows[split] = [
            {
                "problem": task.problem,
                "answer": json.dumps(
                    task.spec,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "modebench_task": POINT_WAYPOINT_VERIFIER,
                "answer_mode_family": task.family,
                "answer_mode_split": task.split,
                "certified_simple_route_count": len(task.route_programs),
                "instance_fingerprint": task.instance_fingerprint,
            }
            for task in split_tasks
        ]
    return rows
