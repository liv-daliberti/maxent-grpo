"""Option-level landmark tours for the sixth ModeBench domain.

The language policy chooses which unvisited landmark to travel to next.  A
pinned, deterministic grid planner and a frozen PD adapter execute the leg;
MuJoCo remains authoritative for collisions, arrival, budget consumption, and
the continuous trajectory.

Mode identity is the order in which the executed trajectory first crosses each
landmark's doorway gate, so different intra-leg paths that realise the same
order merge while different orders stay distinct.  A tour succeeds only when
every landmark and then the goal are reached inside the map's total simulator
step budget.  The adapter carries momentum through intermediate cells and
settles only at each leg's final cell, so a tour's cost depends on how much
turning and braking it actually incurs, not on its grid length alone: the
certified mode set of a map is therefore established by execution.

This module is additive.  ``point_maze_waypoint`` and its frozen v1 identity
rule are untouched so the E78pm/E79pm receipts stay reproducible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Iterator, Mapping, Sequence

from .maze_modebench import (
    MAZE_ACTION_VERSION,
    POINT_ACTIONS,
    POINT_MAZE_VERIFIER,
    MazeActionSpec,
    MazeModeBenchError,
    MazeValidation,
    parse_maze_action_spec,
)

# The tour extractor below reuses the audited crossing geometry verbatim and
# differs from ``extract_directed_gate_route`` in exactly one rule: a two-cell
# room must be left again, so a gate may be crossed more than once.
from .maze_modebench import _gate_crossing, _outside_side  # noqa: PLC2701


POINT_TOUR_VERIFIER = "point_maze_tour"
POINT_TOUR_ACTION_VERSION = "point-tour-v1"
POINT_TOUR_IDENTITY_RULE = "ordered-landmark-gate-sequence"
POINT_TOUR_CELL_ACTIONS = ("N", "E", "S", "W")
POINT_TOUR_CELL_DELTAS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
MAX_LANDMARKS = 6
MAX_TOUR_STEP_BUDGET = 4000
# ``action_repeat`` is capped at 100 upstream, so the base trajectory bound is
# expressed as blocks of steps rather than as legs.
BASE_TRAJECTORY_BLOCK_STEPS = 100
BASE_TRAJECTORY_BLOCKS = MAX_TOUR_STEP_BUDGET // BASE_TRAJECTORY_BLOCK_STEPS + 1

# The continuous adapter keeps the v1 gains and tolerances.  It differs only in
# carrying momentum through intermediate cells, which is what makes a tour's
# step cost a physical fact rather than a restatement of its grid length.
POINT_TOUR_CONTROLLER_CONFIG = {
    "controller": "point-maze-tour-leg-pd",
    "version": 1,
    "kp": 2.0,
    "kd": 0.4,
    "position_tolerance": 0.08,
    "velocity_tolerance": 0.5,
    "intermediate_waypoint_radius": 0.35,
    "max_steps_per_leg": 400,
    "leg_planner": "grid-bfs-shortest-path-nesw",
    "leg_settling": "momentum-through-intermediate-cells",
    "legal_action_rule": "unvisited-landmark",
}
POINT_TOUR_CONTROLLER_SHA256 = hashlib.sha256(
    json.dumps(
        POINT_TOUR_CONTROLLER_CONFIG,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()


@dataclass(frozen=True)
class PointTourSpec:
    base_spec: MazeActionSpec
    controller_sha256: str
    landmark_ids: tuple[str, ...]
    landmark_cells: tuple[tuple[int, int], ...]
    tour_step_budget: int
    max_steps_per_leg: int
    intermediate_waypoint_radius: float
    position_tolerance: float
    velocity_tolerance: float
    kp: float
    kd: float
    spec_sha256: str

    @property
    def landmark_count(self) -> int:
        return len(self.landmark_ids)

    def landmark_cell(self, landmark_id: str) -> tuple[int, int]:
        try:
            index = self.landmark_ids.index(str(landmark_id))
        except ValueError as error:
            raise MazeModeBenchError(f"unknown landmark {landmark_id!r}") from error
        return self.landmark_cells[index]


def _canonical_sha256(value: Mapping[str, Any], *, remove_claim: bool) -> str:
    payload = dict(value)
    if remove_claim:
        payload.pop("spec_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def make_point_tour_spec(
    *,
    environment_sha256: str,
    map_id: str,
    maze_map: Sequence[Sequence[int]],
    reset_cell: Sequence[int],
    goal_cell: Sequence[int],
    reset_seed: int,
    landmarks: Sequence[Mapping[str, Any]],
    tour_step_budget: int,
    environment_id: str = "PointMaze_UMaze-v3",
    success_threshold: float = 0.45,
    max_segment_length: float = 0.6,
) -> dict[str, Any]:
    """Build and self-validate one immutable landmark-tour specification."""

    rows = [[int(value) for value in row] for row in maze_map]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise MazeModeBenchError("tour maze_map must be rectangular")
    height = len(rows)
    width = len(rows[0])
    normalized_landmarks = []
    for landmark in landmarks:
        gate = landmark["gate"]
        normalized_landmarks.append(
            {
                "id": str(landmark["id"]),
                "cell": [int(value) for value in landmark["cell"]],
                "gate": {
                    "id": str(landmark["id"]),
                    "axis": str(gate["axis"]),
                    "coordinate": float(gate["coordinate"]),
                    "span": [float(gate["span"][0]), float(gate["span"][1])],
                    "hysteresis": float(gate["hysteresis"]),
                },
            }
        )
    raw = {
        "verifier": POINT_TOUR_VERIFIER,
        "tour_action_version": POINT_TOUR_ACTION_VERSION,
        "environment_id": str(environment_id),
        "environment_sha256": str(environment_sha256),
        "controller_sha256": POINT_TOUR_CONTROLLER_SHA256,
        "map_id": str(map_id),
        "maze_map": rows,
        "reset_cell": [int(value) for value in reset_cell],
        "goal_cell": [int(value) for value in goal_cell],
        "reset_seed": int(reset_seed),
        "landmarks": normalized_landmarks,
        "tour_step_budget": int(tour_step_budget),
        "max_steps_per_leg": int(POINT_TOUR_CONTROLLER_CONFIG["max_steps_per_leg"]),
        "cell_action_tokens": list(POINT_TOUR_CELL_ACTIONS),
        "success_threshold": float(success_threshold),
        "max_segment_length": float(max_segment_length),
        "bounds_xy": [
            [-width / 2.0, width / 2.0],
            [-height / 2.0, height / 2.0],
        ],
        "route_identity_rule": POINT_TOUR_IDENTITY_RULE,
        "intermediate_waypoint_radius": float(
            POINT_TOUR_CONTROLLER_CONFIG["intermediate_waypoint_radius"]
        ),
        "position_tolerance": float(POINT_TOUR_CONTROLLER_CONFIG["position_tolerance"]),
        "velocity_tolerance": float(POINT_TOUR_CONTROLLER_CONFIG["velocity_tolerance"]),
        "kp": float(POINT_TOUR_CONTROLLER_CONFIG["kp"]),
        "kd": float(POINT_TOUR_CONTROLLER_CONFIG["kd"]),
    }
    raw["spec_sha256"] = _canonical_sha256(raw, remove_claim=False)
    parse_point_tour_spec(raw)
    return raw


def with_tour_step_budget(
    raw_spec: Mapping[str, Any],
    tour_step_budget: int,
) -> dict[str, Any]:
    """Return the same geometry with a recalibrated budget and fresh hash."""

    raw = dict(raw_spec)
    raw.pop("spec_sha256", None)
    raw["tour_step_budget"] = int(tour_step_budget)
    raw["spec_sha256"] = _canonical_sha256(raw, remove_claim=False)
    parse_point_tour_spec(raw)
    return raw


def parse_point_tour_spec(raw_spec: Mapping[str, Any]) -> PointTourSpec:
    if not isinstance(raw_spec, Mapping):
        raise MazeModeBenchError("Point tour spec must be an object")
    if raw_spec.get("verifier") != POINT_TOUR_VERIFIER:
        raise MazeModeBenchError("wrong Point tour verifier")
    if raw_spec.get("tour_action_version") != POINT_TOUR_ACTION_VERSION:
        raise MazeModeBenchError("unsupported Point tour action version")
    if raw_spec.get("cell_action_tokens") != list(POINT_TOUR_CELL_ACTIONS):
        raise MazeModeBenchError("Point tour cell alphabet changed")
    if raw_spec.get("controller_sha256") != POINT_TOUR_CONTROLLER_SHA256:
        raise MazeModeBenchError("Point tour controller hash changed")
    if raw_spec.get("route_identity_rule") != POINT_TOUR_IDENTITY_RULE:
        raise MazeModeBenchError("Point tour route identity rule changed")
    observed_hash = _canonical_sha256(raw_spec, remove_claim=True)
    if raw_spec.get("spec_sha256") != observed_hash:
        raise MazeModeBenchError("Point tour spec hash is invalid")

    fixed = {
        "max_steps_per_leg": int(POINT_TOUR_CONTROLLER_CONFIG["max_steps_per_leg"]),
        "intermediate_waypoint_radius": float(
            POINT_TOUR_CONTROLLER_CONFIG["intermediate_waypoint_radius"]
        ),
        "position_tolerance": float(POINT_TOUR_CONTROLLER_CONFIG["position_tolerance"]),
        "velocity_tolerance": float(POINT_TOUR_CONTROLLER_CONFIG["velocity_tolerance"]),
        "kp": float(POINT_TOUR_CONTROLLER_CONFIG["kp"]),
        "kd": float(POINT_TOUR_CONTROLLER_CONFIG["kd"]),
    }
    for name, expected in fixed.items():
        value = raw_spec.get(name)
        if isinstance(value, bool) or value != expected:
            raise MazeModeBenchError(f"Point tour controller field {name} changed")

    budget = raw_spec.get("tour_step_budget")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or not 1 <= budget <= MAX_TOUR_STEP_BUDGET
    ):
        raise MazeModeBenchError("Point tour step budget is invalid")

    raw_landmarks = raw_spec.get("landmarks")
    if (
        not isinstance(raw_landmarks, list)
        or not 2 <= len(raw_landmarks) <= MAX_LANDMARKS
    ):
        raise MazeModeBenchError("Point tour needs two to six landmarks")
    rows = raw_spec.get("maze_map")
    if not isinstance(rows, list) or not rows:
        raise MazeModeBenchError("Point tour maze_map is missing")

    landmark_ids: list[str] = []
    landmark_cells: list[tuple[int, int]] = []
    gates: list[dict[str, Any]] = []
    for landmark in raw_landmarks:
        if not isinstance(landmark, Mapping):
            raise MazeModeBenchError("Point tour landmark must be an object")
        identifier = landmark.get("id")
        cell = landmark.get("cell")
        gate = landmark.get("gate")
        if not isinstance(identifier, str) or not identifier:
            raise MazeModeBenchError("Point tour landmark id is invalid")
        if (
            not isinstance(cell, list)
            or len(cell) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in cell
            )
        ):
            raise MazeModeBenchError("Point tour landmark cell is invalid")
        if not (
            0 <= cell[0] < len(rows)
            and 0 <= cell[1] < len(rows[0])
            and rows[cell[0]][cell[1]] == 0
        ):
            raise MazeModeBenchError("Point tour landmark is not a free cell")
        if not isinstance(gate, Mapping) or gate.get("id") != identifier:
            raise MazeModeBenchError("Point tour landmark gate is mislabelled")
        landmark_ids.append(identifier)
        landmark_cells.append((int(cell[0]), int(cell[1])))
        gates.append(dict(gate))
    if len(set(landmark_ids)) != len(landmark_ids):
        raise MazeModeBenchError("Point tour landmark ids must be unique")
    if len(set(landmark_cells)) != len(landmark_cells):
        raise MazeModeBenchError("Point tour landmark cells must be distinct")

    translated = {
        "verifier": POINT_MAZE_VERIFIER,
        "maze_action_version": MAZE_ACTION_VERSION,
        "environment_id": raw_spec.get("environment_id"),
        "environment_sha256": raw_spec.get("environment_sha256"),
        "controller_sha256": None,
        "map_id": raw_spec.get("map_id"),
        "maze_map": rows,
        "reset_cell": raw_spec.get("reset_cell"),
        "goal_cell": raw_spec.get("goal_cell"),
        "reset_seed": raw_spec.get("reset_seed"),
        "min_actions": 1,
        # The base spec only bounds how long a trajectory may be.  Fix that
        # bound above every admissible budget so recalibrating a map's budget
        # never moves its executable horizon.
        "max_actions": BASE_TRAJECTORY_BLOCKS,
        "action_repeat": BASE_TRAJECTORY_BLOCK_STEPS,
        "action_tokens": list(POINT_ACTIONS),
        "success_threshold": raw_spec.get("success_threshold"),
        "max_segment_length": raw_spec.get("max_segment_length"),
        "bounds_xy": raw_spec.get("bounds_xy"),
        "route_gates": gates,
    }
    translated["spec_sha256"] = _canonical_sha256(translated, remove_claim=False)
    base_spec = parse_maze_action_spec(translated)
    if budget > base_spec.max_actions * base_spec.action_repeat:
        raise MazeModeBenchError("Point tour budget exceeds the executable horizon")
    return PointTourSpec(
        base_spec=base_spec,
        controller_sha256=POINT_TOUR_CONTROLLER_SHA256,
        landmark_ids=tuple(landmark_ids),
        landmark_cells=tuple(landmark_cells),
        tour_step_budget=int(budget),
        max_steps_per_leg=fixed["max_steps_per_leg"],
        intermediate_waypoint_radius=fixed["intermediate_waypoint_radius"],
        position_tolerance=fixed["position_tolerance"],
        velocity_tolerance=fixed["velocity_tolerance"],
        kp=fixed["kp"],
        kd=fixed["kd"],
        spec_sha256=observed_hash,
    )


def advance_point_tour_cell(
    spec: PointTourSpec,
    cell: tuple[int, int],
    action: str,
) -> tuple[int, int]:
    token = str(action).upper()
    if token not in POINT_TOUR_CELL_DELTAS:
        raise MazeModeBenchError(f"unknown Point tour cell action {token!r}")
    delta = POINT_TOUR_CELL_DELTAS[token]
    target = cell[0] + delta[0], cell[1] + delta[1]
    rows = spec.base_spec.maze_map
    if not (
        0 <= target[0] < len(rows)
        and 0 <= target[1] < len(rows[0])
        and rows[target[0]][target[1]] == 0
    ):
        raise MazeModeBenchError("Point tour cell action enters a wall")
    return target


def plan_leg_cells(
    spec: PointTourSpec,
    start_cell: tuple[int, int],
    target_cell: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Pinned deterministic shortest grid path, NESW expansion order.

    The planner chooses no mode: which landmark the leg walks to is the
    policy's decision, and ties break by a fixed token order, so the leg is a
    function of the map and its two endpoints alone.
    """

    if start_cell == target_cell:
        return ()
    queue = deque([start_cell])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    while queue:
        cell = queue.popleft()
        if cell == target_cell:
            break
        for token in POINT_TOUR_CELL_ACTIONS:
            try:
                nxt = advance_point_tour_cell(spec, cell, token)
            except MazeModeBenchError:
                continue
            if nxt in parent:
                continue
            parent[nxt] = cell
            queue.append(nxt)
    if target_cell not in parent:
        raise MazeModeBenchError("Point tour leg target is unreachable")
    cells: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = target_cell
    while cursor is not None and parent[cursor] is not None:
        cells.append(cursor)
        cursor = parent[cursor]
    cells.reverse()
    return tuple(cells)


def tour_grid_cost(spec: PointTourSpec, order: Sequence[str]) -> int:
    """Total planned cell moves for one landmark order, including the goal leg."""

    cell = spec.base_spec.reset_cell
    total = 0
    for landmark_id in order:
        target = spec.landmark_cell(landmark_id)
        total += len(plan_leg_cells(spec, cell, target))
        cell = target
    total += len(plan_leg_cells(spec, cell, spec.base_spec.goal_cell))
    return total


def iter_tour_orders(spec: PointTourSpec) -> Iterator[tuple[str, ...]]:
    yield from itertools.permutations(spec.landmark_ids)


def cell_center_xy(spec: PointTourSpec, cell: Sequence[int]) -> tuple[float, float]:
    height = len(spec.base_spec.maze_map)
    width = len(spec.base_spec.maze_map[0])
    return cell[1] + 0.5 - width / 2.0, height / 2.0 - (cell[0] + 0.5)


def extract_tour_gate_route(
    trajectory_xy: Sequence[tuple[float, float]],
    spec: MazeActionSpec,
) -> tuple[str, ...]:
    """Ordered directed gate crossings for a tour trajectory.

    Bounds, teleport, hysteresis, and crossing geometry are the audited v1
    rules, applied through the same helpers.  The single difference is that a
    gate may be crossed more than once, because a two-cell room has to be left
    again and may legally be passed later in the tour.
    """

    if len(trajectory_xy) < 2:
        raise MazeModeBenchError("trajectory is too short")
    if len(trajectory_xy) > spec.max_actions * spec.action_repeat + 2:
        raise MazeModeBenchError("trajectory exceeds the frozen execution horizon")
    for index, point in enumerate(trajectory_xy):
        if not (
            spec.bounds[0][0] <= point[0] <= spec.bounds[0][1]
            and spec.bounds[1][0] <= point[1] <= spec.bounds[1][1]
        ):
            raise MazeModeBenchError(f"trajectory point {index} is outside maze bounds")
        if index:
            previous = trajectory_xy[index - 1]
            distance = math.hypot(point[0] - previous[0], point[1] - previous[1])
            if distance > spec.max_segment_length:
                raise MazeModeBenchError("trajectory contains a teleport-sized segment")

    route: list[str] = []
    state: dict[str, tuple[int, tuple[float, float]] | None] = {
        gate.gate_id: None for gate in spec.gates
    }
    for point in trajectory_xy:
        step_crossings: list[str] = []
        for gate in spec.gates:
            coordinate = point[0] if gate.axis == "x" else point[1]
            side = _outside_side(coordinate, gate)
            previous_state = state[gate.gate_id]
            if side == 0:
                continue
            if previous_state is None:
                state[gate.gate_id] = (side, point)
                continue
            previous_side, previous_point = previous_state
            if side != previous_side:
                crossing = _gate_crossing(previous_point, point, gate)
                if crossing is not None:
                    step_crossings.append(crossing)
            state[gate.gate_id] = (side, point)
        if len(step_crossings) > 1:
            raise MazeModeBenchError(
                "one trajectory segment crosses multiple route gates"
            )
        if step_crossings and (not route or route[-1] != step_crossings[0]):
            route.append(step_crossings[0])
    if not route:
        raise MazeModeBenchError("successful trajectory has no certified route gate")
    return tuple(route)


def tour_canonical_order(directed_route: Sequence[str]) -> tuple[str, ...]:
    """First-crossing order of landmark gates.

    A landmark's doorway gate sits between its doorway cell and its interior
    cell, so the first crossing of that gate is necessarily inbound; later
    re-crossings only record the exit and any revisit.
    """

    order: list[str] = []
    for crossing in directed_route:
        landmark_id = str(crossing)[:-1]
        if landmark_id not in order:
            order.append(landmark_id)
    return tuple(order)


def point_tour_canonical_key(spec: PointTourSpec, order: Sequence[str]) -> str:
    return (
        f"{POINT_TOUR_VERIFIER}:{POINT_TOUR_ACTION_VERSION}:"
        f"{spec.base_spec.map_id}:" + ">".join(str(item) for item in order)
    )


def render_point_tour_problem(raw_spec: Mapping[str, Any]) -> str:
    spec = parse_point_tour_spec(raw_spec)
    marks = {cell: identifier[-1] for identifier, cell in zip(spec.landmark_ids, spec.landmark_cells)}
    rows = []
    for row_index, row in enumerate(spec.base_spec.maze_map):
        chars = []
        for column_index, value in enumerate(row):
            cell = row_index, column_index
            if cell == spec.base_spec.reset_cell:
                chars.append("S")
            elif cell == spec.base_spec.goal_cell:
                chars.append("G")
            elif cell in marks:
                chars.append(marks[cell])
            else:
                chars.append("#" if value else ".")
        rows.append("".join(chars))
    listing = ", ".join(
        f"{identifier} at ({cell[0]},{cell[1]})"
        for identifier, cell in zip(spec.landmark_ids, spec.landmark_cells)
    )
    return "\n".join(
        [
            "Visit every landmark, then reach G, starting from S.",
            *rows,
            f"Landmarks: {listing}.",
            f"Total drive budget for the whole tour: {spec.tour_step_budget} steps.",
            "Each leg is driven for you; you choose only which landmark comes next.",
        ]
    )


def validate_point_tour_execution(
    raw_spec: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> MazeValidation:
    """Accept a completed tour and key it by the executed landmark order."""

    spec = parse_point_tour_spec(raw_spec)
    if execution.get("environment_sha256") != spec.base_spec.environment_sha256:
        raise MazeModeBenchError("Point tour environment hash mismatch")
    if execution.get("controller_sha256") != spec.controller_sha256:
        raise MazeModeBenchError("Point tour controller hash mismatch")
    if execution.get("spec_sha256") != spec.spec_sha256:
        raise MazeModeBenchError("Point tour spec hash mismatch")
    if execution.get("reset_seed") != spec.base_spec.reset_seed:
        raise MazeModeBenchError("Point tour reset seed mismatch")
    if not execution.get("success"):
        raise MazeModeBenchError("Point tour execution did not succeed")

    simulator_steps = execution.get("simulator_steps")
    if (
        isinstance(simulator_steps, bool)
        or not isinstance(simulator_steps, int)
        or not 1 <= simulator_steps <= spec.tour_step_budget
    ):
        raise MazeModeBenchError("Point tour step count is outside the budget")

    raw_trajectory = execution.get("trajectory_xy")
    if not isinstance(raw_trajectory, list):
        raise MazeModeBenchError("Point tour trajectory is missing")
    trajectory: list[tuple[float, float]] = []
    for point in raw_trajectory:
        if not isinstance(point, list) or len(point) != 2:
            raise MazeModeBenchError("Point tour trajectory is malformed")
        xy = float(point[0]), float(point[1])
        if not all(math.isfinite(value) for value in xy):
            raise MazeModeBenchError("Point tour trajectory is nonfinite")
        trajectory.append(xy)

    directed_route = extract_tour_gate_route(trajectory, spec.base_spec)
    order = tour_canonical_order(directed_route)
    if set(order) != set(spec.landmark_ids) or len(order) != spec.landmark_count:
        raise MazeModeBenchError("Point tour did not visit every landmark exactly once")
    return MazeValidation(
        canonical_key=point_tour_canonical_key(spec, order),
        directed_gates=tuple(directed_route),
        action_tokens=tuple(order),
        simulator_steps=int(simulator_steps),
    )
