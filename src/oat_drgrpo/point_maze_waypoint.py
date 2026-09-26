"""Sequential adjacent-cell actions for the PointMaze domain.

The language policy chooses one prompt-visible neighboring free cell at a
time.  A small, deterministic, hash-bound PD controller handles only the
continuous actuation required to settle the point mass at that selected cell.
Goal success and semantic route identity remain properties of the pinned
MuJoCo execution, not of the logical grid plan.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .maze_modebench import (
    MAZE_ACTION_VERSION,
    POINT_ACTIONS,
    POINT_MAZE_VERIFIER,
    DirectedGate,
    MazeActionSpec,
    MazeModeBenchError,
    MazeValidation,
    extract_directed_gate_route,
    parse_maze_action_spec,
)


POINT_WAYPOINT_VERIFIER = "point_maze_waypoint"
POINT_WAYPOINT_ACTION_VERSION = "point-waypoint-v1"
POINT_WAYPOINT_ACTIONS = ("N", "E", "S", "W")
POINT_WAYPOINT_DELTAS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}

POINT_WAYPOINT_CONTROLLER_CONFIG = {
    "controller": "point-maze-adjacent-cell-pd",
    "version": 1,
    "kp": 2.0,
    "kd": 0.4,
    "position_tolerance": 0.08,
    "velocity_tolerance": 0.5,
    "max_steps_per_action": 100,
    "legal_action_rule": "adjacent-prompt-visible-free-cell",
    "revisits": "allowed",
}
POINT_WAYPOINT_CONTROLLER_SHA256 = hashlib.sha256(
    json.dumps(
        POINT_WAYPOINT_CONTROLLER_CONFIG,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()

_MAX_PROGRAM_CHARS = 1024


@dataclass(frozen=True)
class PointWaypointSpec:
    base_spec: MazeActionSpec
    controller_sha256: str
    min_actions: int
    max_actions: int
    max_steps_per_action: int
    position_tolerance: float
    velocity_tolerance: float
    kp: float
    kd: float
    spec_sha256: str


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


def make_point_waypoint_spec(
    *,
    environment_sha256: str,
    map_id: str,
    maze_map: Sequence[Sequence[int]],
    reset_cell: Sequence[int],
    goal_cell: Sequence[int],
    reset_seed: int,
    route_gates: Sequence[Mapping[str, Any]],
    max_actions: int = 64,
    min_actions: int = 1,
    environment_id: str = "PointMaze_UMaze-v3",
    success_threshold: float = 0.45,
    max_segment_length: float = 0.2,
) -> dict[str, Any]:
    """Build and self-validate one immutable waypoint specification."""

    rows = [list(row) for row in maze_map]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise MazeModeBenchError("waypoint maze_map must be rectangular")
    height = len(rows)
    width = len(rows[0])
    raw = {
        "verifier": POINT_WAYPOINT_VERIFIER,
        "waypoint_action_version": POINT_WAYPOINT_ACTION_VERSION,
        "environment_id": str(environment_id),
        "environment_sha256": str(environment_sha256),
        "controller_sha256": POINT_WAYPOINT_CONTROLLER_SHA256,
        "map_id": str(map_id),
        "maze_map": rows,
        "reset_cell": [int(value) for value in reset_cell],
        "goal_cell": [int(value) for value in goal_cell],
        "reset_seed": int(reset_seed),
        "min_actions": int(min_actions),
        "max_actions": int(max_actions),
        "max_steps_per_action": int(
            POINT_WAYPOINT_CONTROLLER_CONFIG["max_steps_per_action"]
        ),
        "action_tokens": list(POINT_WAYPOINT_ACTIONS),
        "success_threshold": float(success_threshold),
        "max_segment_length": float(max_segment_length),
        "bounds_xy": [
            [-width / 2.0, width / 2.0],
            [-height / 2.0, height / 2.0],
        ],
        "route_gates": [dict(gate) for gate in route_gates],
        "route_identity_rule": "exactly-one-directed-gate",
        "position_tolerance": float(
            POINT_WAYPOINT_CONTROLLER_CONFIG["position_tolerance"]
        ),
        "velocity_tolerance": float(
            POINT_WAYPOINT_CONTROLLER_CONFIG["velocity_tolerance"]
        ),
        "kp": float(POINT_WAYPOINT_CONTROLLER_CONFIG["kp"]),
        "kd": float(POINT_WAYPOINT_CONTROLLER_CONFIG["kd"]),
    }
    raw["spec_sha256"] = _canonical_sha256(raw, remove_claim=False)
    parse_point_waypoint_spec(raw)
    return raw


def adapt_point_maze_to_waypoints(
    raw_legacy_spec: Mapping[str, Any],
    *,
    map_id: str | None = None,
    max_actions: int | None = None,
) -> dict[str, Any]:
    """Adapt an admitted force-program map without changing its geometry."""

    legacy = parse_maze_action_spec(raw_legacy_spec)
    if legacy.verifier != POINT_MAZE_VERIFIER:
        raise MazeModeBenchError("waypoint adapter requires a PointMaze spec")
    free_cells = sum(cell == 0 for row in legacy.maze_map for cell in row)
    gates = [
        {
            "id": gate.gate_id,
            "axis": gate.axis,
            "coordinate": gate.coordinate,
            "span": [gate.span_min, gate.span_max],
            "hysteresis": gate.hysteresis,
        }
        for gate in legacy.gates
    ]
    return make_point_waypoint_spec(
        environment_id=legacy.environment_id,
        environment_sha256=legacy.environment_sha256,
        map_id=map_id or f"{legacy.map_id}_waypoint_v1",
        maze_map=legacy.maze_map,
        reset_cell=legacy.reset_cell,
        goal_cell=legacy.goal_cell,
        reset_seed=legacy.reset_seed,
        route_gates=gates,
        min_actions=1,
        max_actions=(
            min(64, max(1, free_cells - 1)) if max_actions is None else int(max_actions)
        ),
        success_threshold=legacy.success_threshold,
        max_segment_length=legacy.max_segment_length,
    )


def parse_point_waypoint_spec(raw_spec: Mapping[str, Any]) -> PointWaypointSpec:
    if not isinstance(raw_spec, Mapping):
        raise MazeModeBenchError("Point waypoint spec must be an object")
    if raw_spec.get("verifier") != POINT_WAYPOINT_VERIFIER:
        raise MazeModeBenchError("wrong Point waypoint verifier")
    if raw_spec.get("waypoint_action_version") != POINT_WAYPOINT_ACTION_VERSION:
        raise MazeModeBenchError("unsupported Point waypoint action version")
    if raw_spec.get("action_tokens") != list(POINT_WAYPOINT_ACTIONS):
        raise MazeModeBenchError("Point waypoint action alphabet changed")
    if raw_spec.get("controller_sha256") != POINT_WAYPOINT_CONTROLLER_SHA256:
        raise MazeModeBenchError("Point waypoint controller hash changed")
    if raw_spec.get("route_identity_rule") != "exactly-one-directed-gate":
        raise MazeModeBenchError("Point waypoint route identity rule changed")
    observed_hash = _canonical_sha256(raw_spec, remove_claim=True)
    if raw_spec.get("spec_sha256") != observed_hash:
        raise MazeModeBenchError("Point waypoint spec hash is invalid")

    fixed = {
        "max_steps_per_action": int(
            POINT_WAYPOINT_CONTROLLER_CONFIG["max_steps_per_action"]
        ),
        "position_tolerance": float(
            POINT_WAYPOINT_CONTROLLER_CONFIG["position_tolerance"]
        ),
        "velocity_tolerance": float(
            POINT_WAYPOINT_CONTROLLER_CONFIG["velocity_tolerance"]
        ),
        "kp": float(POINT_WAYPOINT_CONTROLLER_CONFIG["kp"]),
        "kd": float(POINT_WAYPOINT_CONTROLLER_CONFIG["kd"]),
    }
    for name, expected in fixed.items():
        value = raw_spec.get(name)
        if isinstance(value, bool) or value != expected:
            raise MazeModeBenchError(f"Point waypoint controller field {name} changed")
    min_actions = raw_spec.get("min_actions")
    max_actions = raw_spec.get("max_actions")
    if (
        isinstance(min_actions, bool)
        or not isinstance(min_actions, int)
        or isinstance(max_actions, bool)
        or not isinstance(max_actions, int)
        or not 1 <= min_actions <= max_actions <= 64
    ):
        raise MazeModeBenchError("Point waypoint action bounds are invalid")

    translated = {
        "verifier": POINT_MAZE_VERIFIER,
        "maze_action_version": MAZE_ACTION_VERSION,
        "environment_id": raw_spec.get("environment_id"),
        "environment_sha256": raw_spec.get("environment_sha256"),
        "controller_sha256": None,
        "map_id": raw_spec.get("map_id"),
        "maze_map": raw_spec.get("maze_map"),
        "reset_cell": raw_spec.get("reset_cell"),
        "goal_cell": raw_spec.get("goal_cell"),
        "reset_seed": raw_spec.get("reset_seed"),
        "min_actions": min_actions,
        "max_actions": max_actions,
        "action_repeat": fixed["max_steps_per_action"],
        "action_tokens": list(POINT_ACTIONS),
        "success_threshold": raw_spec.get("success_threshold"),
        "max_segment_length": raw_spec.get("max_segment_length"),
        "bounds_xy": raw_spec.get("bounds_xy"),
        "route_gates": raw_spec.get("route_gates"),
    }
    translated["spec_sha256"] = _canonical_sha256(translated, remove_claim=False)
    base_spec = parse_maze_action_spec(translated)
    return PointWaypointSpec(
        base_spec=base_spec,
        controller_sha256=POINT_WAYPOINT_CONTROLLER_SHA256,
        min_actions=int(min_actions),
        max_actions=int(max_actions),
        max_steps_per_action=fixed["max_steps_per_action"],
        position_tolerance=fixed["position_tolerance"],
        velocity_tolerance=fixed["velocity_tolerance"],
        kp=fixed["kp"],
        kd=fixed["kd"],
        spec_sha256=observed_hash,
    )


def advance_point_waypoint_cell(
    spec: PointWaypointSpec,
    cell: tuple[int, int],
    action: str,
) -> tuple[int, int]:
    token = str(action).upper()
    if token not in POINT_WAYPOINT_DELTAS:
        raise MazeModeBenchError(f"unknown Point waypoint action {token!r}")
    delta = POINT_WAYPOINT_DELTAS[token]
    target = cell[0] + delta[0], cell[1] + delta[1]
    rows = spec.base_spec.maze_map
    if not (
        0 <= target[0] < len(rows)
        and 0 <= target[1] < len(rows[0])
        and rows[target[0]][target[1]] == 0
    ):
        raise MazeModeBenchError("Point waypoint action enters a wall")
    return target


def legal_point_waypoint_actions(
    spec: PointWaypointSpec,
    cell: tuple[int, int],
) -> tuple[str, ...]:
    result = []
    for action in POINT_WAYPOINT_ACTIONS:
        try:
            advance_point_waypoint_cell(spec, cell, action)
        except MazeModeBenchError:
            continue
        result.append(action)
    if not result:
        raise MazeModeBenchError("Point waypoint state has no legal action")
    return tuple(result)


def parse_point_waypoint_program(
    candidate: str,
    spec: PointWaypointSpec,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    text = str(candidate).strip()
    if not text or len(text) > _MAX_PROGRAM_CHARS:
        raise MazeModeBenchError("Point waypoint program is empty or too long")
    tokens = tuple(token for token in re.split(r"[\s,]+", text.upper()) if token)
    if not spec.min_actions <= len(tokens) <= spec.max_actions:
        raise MazeModeBenchError("Point waypoint program length is outside its bounds")
    current = spec.base_spec.reset_cell
    cells: list[tuple[int, int]] = []
    for token in tokens:
        if token not in legal_point_waypoint_actions(spec, current):
            raise MazeModeBenchError("Point waypoint program selects an illegal action")
        current = advance_point_waypoint_cell(spec, current, token)
        cells.append(current)
    if current != spec.base_spec.goal_cell:
        raise MazeModeBenchError("Point waypoint program does not end at the goal")
    return tokens, tuple(cells)


def render_point_waypoint_problem(raw_spec: Mapping[str, Any]) -> str:
    spec = parse_point_waypoint_spec(raw_spec)
    rows = []
    for row_index, row in enumerate(spec.base_spec.maze_map):
        chars = []
        for column_index, value in enumerate(row):
            cell = row_index, column_index
            if cell == spec.base_spec.reset_cell:
                chars.append("S")
            elif cell == spec.base_spec.goal_cell:
                chars.append("G")
            else:
                chars.append("#" if value else ".")
        rows.append("".join(chars))
    return "\n".join(
        [
            "Navigate the point mass from S to G through the public maze.",
            *rows,
            "Choose one adjacent free cell per decision.",
            "The physical controller may execute only the selected local edge.",
        ]
    )


def _cell_xy(spec: PointWaypointSpec, cell: tuple[int, int]) -> tuple[float, float]:
    height = len(spec.base_spec.maze_map)
    width = len(spec.base_spec.maze_map[0])
    return cell[1] + 0.5 - width / 2.0, height / 2.0 - (cell[0] + 0.5)


def _outside_side(point: tuple[float, float], gate: DirectedGate) -> int:
    coordinate = point[0] if gate.axis == "x" else point[1]
    if coordinate < gate.coordinate - gate.hysteresis:
        return -1
    if coordinate > gate.coordinate + gate.hysteresis:
        return 1
    return 0


def _edge_crossing(
    first: tuple[float, float],
    second: tuple[float, float],
    gate: DirectedGate,
) -> str | None:
    axis = 0 if gate.axis == "x" else 1
    span_axis = 1 - axis
    delta = second[axis] - first[axis]
    if delta == 0.0:
        return None
    fraction = (gate.coordinate - first[axis]) / delta
    if not 0.0 <= fraction <= 1.0:
        return None
    span = first[span_axis] + fraction * (second[span_axis] - first[span_axis])
    if not gate.span_min <= span <= gate.span_max:
        return None
    return f"{gate.gate_id}{'+' if delta > 0 else '-'}"


def find_point_waypoint_route_programs(raw_spec: Mapping[str, Any]) -> dict[str, str]:
    """Find a shortest legal logical program for each simple gate route."""

    spec = parse_point_waypoint_spec(raw_spec)
    start_cell = spec.base_spec.reset_cell
    start_xy = _cell_xy(spec, start_cell)
    start_memory = tuple(
        (side, start_cell[0], start_cell[1]) if side else None
        for side in (_outside_side(start_xy, gate) for gate in spec.base_spec.gates)
    )
    start = start_cell, (), start_memory
    queue = deque([start])
    parent: dict[Any, tuple[Any, str] | None] = {start: None}
    depth = {start: 0}
    found: dict[tuple[str, ...], Any] = {}
    while queue:
        state = queue.popleft()
        cell, route, memory = state
        if cell == spec.base_spec.goal_cell:
            if route and route not in found:
                found[route] = state
            continue
        if depth[state] >= spec.max_actions:
            continue
        for token in legal_point_waypoint_actions(spec, cell):
            nxt = advance_point_waypoint_cell(spec, cell, token)
            next_xy = _cell_xy(spec, nxt)
            next_memory = list(memory)
            crossings: list[str] = []
            for index, gate in enumerate(spec.base_spec.gates):
                side = _outside_side(next_xy, gate)
                if side == 0:
                    continue
                previous = memory[index]
                if previous is not None and side != previous[0]:
                    crossing = _edge_crossing(
                        _cell_xy(spec, (previous[1], previous[2])),
                        next_xy,
                        gate,
                    )
                    if crossing is not None:
                        crossings.append(crossing)
                next_memory[index] = (side, nxt[0], nxt[1])
            if len(crossings) > 1:
                continue
            next_route = route
            if crossings:
                crossing = crossings[0]
                gate_id = crossing[:-1]
                if any(existing[:-1] == gate_id for existing in route):
                    continue
                next_route = route + (crossing,)
            next_state = nxt, next_route, tuple(next_memory)
            if next_state in parent:
                continue
            parent[next_state] = state, token
            depth[next_state] = depth[state] + 1
            queue.append(next_state)

    programs: dict[str, str] = {}
    for route, state in sorted(found.items()):
        tokens = []
        cursor = state
        while parent[cursor] is not None:
            previous, token = parent[cursor]
            tokens.append(token)
            cursor = previous
        tokens.reverse()
        if len(route) == 1 and spec.min_actions <= len(tokens) <= spec.max_actions:
            programs[",".join(route)] = " ".join(tokens)
    return programs


def validate_point_waypoint_execution(
    candidate: str,
    raw_spec: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> MazeValidation:
    spec = parse_point_waypoint_spec(raw_spec)
    tokens, cells = parse_point_waypoint_program(candidate, spec)
    checks = {
        "environment_sha256": spec.base_spec.environment_sha256,
        "controller_sha256": spec.controller_sha256,
        "spec_sha256": spec.spec_sha256,
        "reset_seed": spec.base_spec.reset_seed,
    }
    for name, expected in checks.items():
        if execution.get(name) != expected:
            raise MazeModeBenchError(f"Point waypoint execution {name} differs")
    if execution.get("action_tokens") != list(tokens):
        raise MazeModeBenchError("Point waypoint execution actions differ")
    if execution.get("target_cells") != [list(cell) for cell in cells]:
        raise MazeModeBenchError("Point waypoint execution targets differ")
    if execution.get("logical_final_cell") != list(spec.base_spec.goal_cell):
        raise MazeModeBenchError("Point waypoint logical final cell is not the goal")
    if execution.get("success") is not True:
        raise MazeModeBenchError("unsuccessful Point waypoint execution")
    if execution.get("stable_waypoint_failure") is not False:
        raise MazeModeBenchError("Point waypoint controller failed a local edge")
    distance = float(execution.get("final_goal_distance", math.inf))
    if not math.isfinite(distance) or not (
        0.0 <= distance <= spec.base_spec.success_threshold
    ):
        raise MazeModeBenchError("Point waypoint execution misses the goal")
    simulator_steps = execution.get("simulator_steps")
    if (
        isinstance(simulator_steps, bool)
        or not isinstance(simulator_steps, int)
        or not 1 <= simulator_steps <= len(tokens) * spec.max_steps_per_action
    ):
        raise MazeModeBenchError("Point waypoint execution exceeds its step budget")
    segment_steps = execution.get("segment_steps")
    if (
        not isinstance(segment_steps, list)
        or len(segment_steps) != len(tokens)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= spec.max_steps_per_action
            for value in segment_steps
        )
        or sum(segment_steps) != simulator_steps
    ):
        raise MazeModeBenchError("Point waypoint segment-step record is invalid")
    raw_trajectory = execution.get("trajectory_xy")
    if not isinstance(raw_trajectory, list):
        raise MazeModeBenchError("Point waypoint trajectory is missing")
    trajectory: list[tuple[float, float]] = []
    for point in raw_trajectory:
        if not isinstance(point, list) or len(point) != 2:
            raise MazeModeBenchError("Point waypoint trajectory is malformed")
        xy = float(point[0]), float(point[1])
        if not all(math.isfinite(value) for value in xy):
            raise MazeModeBenchError("Point waypoint trajectory is nonfinite")
        trajectory.append(xy)
    route = extract_directed_gate_route(trajectory, spec.base_spec)
    if len(route) != 1:
        raise MazeModeBenchError(
            "Point waypoint success must cross exactly one route gate"
        )
    return MazeValidation(
        canonical_key=(
            f"{POINT_WAYPOINT_VERIFIER}:{POINT_WAYPOINT_ACTION_VERSION}:"
            f"{spec.base_spec.map_id}:" + ",".join(route)
        ),
        directed_gates=route,
        action_tokens=tokens,
        simulator_steps=simulator_steps,
    )
