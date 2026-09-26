"""Deterministic comb-layout map generation for PointMaze Tour.

Each map is a straight corridor from S to G with ``K`` two-cell side rooms
hanging off it.  Visiting a room costs the detour in and out plus whatever
corridor backtracking the chosen order forces, so orders differ in cost and a
total step budget makes only some of them feasible.  Which ones are feasible is
settled by execution, not by this module: the generator emits geometry and a
placeholder budget, and ``ops/make_point_maze_tour_data.py`` calibrates the
budget from measured tours.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import random
from typing import Any, Mapping, Sequence

from .point_maze_tour import (
    MAX_TOUR_STEP_BUDGET,
    POINT_TOUR_VERIFIER,
    make_point_tour_spec,
)


DEFAULT_TOUR_SPLIT_COUNTS = {"train": 384, "dev": 64, "eval": 128}
POINT_TOUR_DATA_SEED = 91_204
TOUR_SIZES = (15,)
TOUR_LANDMARKS = 5
GATE_HALF_SPAN = 0.42
GATE_HYSTERESIS = 0.1


@dataclass(frozen=True)
class PointTourTask:
    split: str
    family: str
    instance_fingerprint: str
    spec: dict[str, Any]


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
        "span": [new_span[0], new_span[1]],
        "hysteresis": float(gate["hysteresis"]),
    }


def _spaced_column_choices(size: int, count: int) -> list[tuple[int, ...]]:
    """All doorway column sets inside the corridor with a minimum gap of two."""

    columns = range(2, size - 2)
    return [
        combination
        for combination in itertools.combinations(columns, count)
        if all(
            second - first >= 2
            for first, second in zip(combination, combination[1:])
        )
    ]


def _comb_geometry(
    *,
    size: int,
    doorway_columns: Sequence[int],
    sides: Sequence[int],
    rotations: int,
) -> tuple[list[list[int]], tuple[int, int], tuple[int, int], list[dict[str, Any]]]:
    """Build one unrotated comb map, then apply the requested rotation."""

    corridor = size // 2
    maze_map = [[1] * size for _ in range(size)]
    for column in range(1, size - 1):
        maze_map[corridor][column] = 0
    landmarks: list[dict[str, Any]] = []
    for index, (column, side) in enumerate(zip(doorway_columns, sides)):
        direction = -1 if side else 1
        doorway_row = corridor + direction
        interior_row = corridor + 2 * direction
        maze_map[doorway_row][column] = 0
        maze_map[interior_row][column] = 0
        # The gate sits between the doorway cell and the interior cell, so a
        # trajectory can only cross it by actually entering the room.
        boundary_row = min(doorway_row, interior_row)
        coordinate = size / 2.0 - boundary_row - 1.0
        center_x = column + 0.5 - size / 2.0
        landmarks.append(
            {
                "id": f"lm{index + 1}",
                "cell": [interior_row, column],
                "gate": {
                    "id": f"lm{index + 1}",
                    "axis": "y",
                    "coordinate": coordinate,
                    "span": [center_x - GATE_HALF_SPAN, center_x + GATE_HALF_SPAN],
                    "hysteresis": GATE_HYSTERESIS,
                },
            }
        )
    reset_cell = (corridor, 1)
    goal_cell = (corridor, size - 2)
    for _ in range(int(rotations) % 4):
        maze_map = _rotate_map_cw(maze_map)
        reset_cell = _rotate_cell_cw(reset_cell, size)
        goal_cell = _rotate_cell_cw(goal_cell, size)
        landmarks = [
            {
                "id": landmark["id"],
                "cell": list(
                    _rotate_cell_cw(
                        (int(landmark["cell"][0]), int(landmark["cell"][1])), size
                    )
                ),
                "gate": _rotate_gate_cw(landmark["gate"]),
            }
            for landmark in landmarks
        ]
    return maze_map, reset_cell, goal_cell, landmarks


def _instance_fingerprint(spec: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "maze_map": spec["maze_map"],
            "reset_cell": spec["reset_cell"],
            "goal_cell": spec["goal_cell"],
            "landmarks": [
                {"id": landmark["id"], "cell": landmark["cell"]}
                for landmark in spec["landmarks"]
            ],
        }
    )


SPLIT_SEED_BASE = {"train": 912_000, "dev": 913_000, "eval": 914_000}


def point_tour_candidate(
    *,
    environment_sha256: str,
    split: str,
    attempt: int,
    seed: int = POINT_TOUR_DATA_SEED,
    landmark_count: int = TOUR_LANDMARKS,
) -> PointTourTask:
    """Build attempt ``attempt`` of ``split`` with a placeholder step budget.

    Every attempt is a pure function of its split and ordinal, so candidates
    can be measured in any order or in parallel and admitted afterwards by
    their executed tour counts.
    """

    if split not in SPLIT_SEED_BASE:
        raise ValueError(f"unknown tour split {split!r}")
    choices = {size: _spaced_column_choices(size, landmark_count) for size in TOUR_SIZES}
    for size, options in choices.items():
        if not options:
            raise ValueError(f"size {size} admits no spaced doorway layout")
    rng = random.Random(
        _canonical_sha256({"seed": int(seed), "split": split, "attempt": int(attempt)})
    )
    # Size advances only every fourth attempt so each size sees all four
    # rotations; tying both to ``attempt`` directly would pin size 13 to
    # rotations 0 and 2 and size 15 to rotations 1 and 3.
    rotation = attempt % 4
    size = TOUR_SIZES[(attempt // 4) % len(TOUR_SIZES)]
    doorway_columns = rng.choice(choices[size])
    sides = tuple(rng.randint(0, 1) for _ in range(landmark_count))
    maze_map, reset_cell, goal_cell, landmarks = _comb_geometry(
        size=size,
        doorway_columns=doorway_columns,
        sides=sides,
        rotations=rotation,
    )
    spec = make_point_tour_spec(
        environment_sha256=environment_sha256,
        map_id=f"comb_{landmark_count}lm_{size}_{split}_a{attempt:05d}_r{rotation}",
        maze_map=maze_map,
        reset_cell=reset_cell,
        goal_cell=goal_cell,
        reset_seed=SPLIT_SEED_BASE[split] + attempt,
        landmarks=landmarks,
        tour_step_budget=MAX_TOUR_STEP_BUDGET,
    )
    return PointTourTask(
        split=split,
        family=f"comb_{landmark_count}lm_{size}",
        instance_fingerprint=_instance_fingerprint(spec),
        spec=spec,
    )


def choose_tour_budget(
    costs: Sequence[int],
    *,
    minimum_modes: int,
    maximum_modes: int,
) -> tuple[int, int] | None:
    """Pick the step budget admitting a tour count inside the target band.

    ``costs`` are the executed step costs of every order that completed.  A
    budget is admissible only at an observed cost, so the admitted count is
    exactly the number of orders at or under it.  Ties are kept together: a
    budget can never split two orders that cost the same.
    """

    ordered = sorted(int(cost) for cost in costs)
    best: tuple[int, int] | None = None
    target = (minimum_modes + maximum_modes) / 2.0
    for candidate in sorted(set(ordered)):
        admitted = sum(1 for cost in ordered if cost <= candidate)
        if not minimum_modes <= admitted <= maximum_modes:
            continue
        if best is None or abs(admitted - target) < abs(best[1] - target):
            best = candidate, admitted
    return best


def point_tour_task_row(
    *,
    problem: str,
    spec: Mapping[str, Any],
    family: str,
    split: str,
    certified_tour_count: int,
    instance_fingerprint: str,
) -> dict[str, Any]:
    return {
        "problem": problem,
        "answer": json.dumps(
            spec,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "modebench_task": POINT_TOUR_VERIFIER,
        "answer_mode_family": family,
        "answer_mode_split": split,
        "certified_tour_count": int(certified_tour_count),
        "instance_fingerprint": instance_fingerprint,
    }
