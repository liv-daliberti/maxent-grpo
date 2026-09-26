"""Verified language action programs for PointMaze and AntMaze.

This module is deliberately simulator-independent. It validates the trusted
episode specification, parses the LM's finite action alphabet, and derives a
topological route key from a hash-bound successful execution record. MuJoCo
remains outside the trainer process; its worker must return the trajectory and
environment hashes consumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


POINT_MAZE_VERIFIER = "point_maze_action_program"
ANT_MAZE_VERIFIER = "ant_maze_action_program"
MAZE_ACTION_VERSION = "maze-action-v1"
ANT_MAZE_ACTION_VERSION = "maze-action-ant-v5"

POINT_ACTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "COAST")
ANT_ACTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_GATE_ID = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_MAX_PROGRAM_CHARS = 4096


class MazeModeBenchError(ValueError):
    """Raised when a trusted maze spec or execution record is invalid."""


@dataclass(frozen=True)
class DirectedGate:
    gate_id: str
    axis: str
    coordinate: float
    span_min: float
    span_max: float
    hysteresis: float


@dataclass(frozen=True)
class MazeActionSpec:
    verifier: str
    environment_id: str
    environment_sha256: str
    controller_sha256: str | None
    map_id: str
    maze_map: tuple[tuple[int, ...], ...]
    reset_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    reset_seed: int
    min_actions: int
    max_actions: int
    action_repeat: int
    action_tokens: tuple[str, ...]
    success_threshold: float
    max_segment_length: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    gates: tuple[DirectedGate, ...]
    spec_sha256: str


@dataclass(frozen=True)
class MazeExecutionRecord:
    environment_sha256: str
    controller_sha256: str | None
    spec_sha256: str
    reset_seed: int
    action_tokens: tuple[str, ...]
    success: bool
    final_goal_distance: float
    trajectory_xy: tuple[tuple[float, float], ...]
    simulator_steps: int


@dataclass(frozen=True)
class MazeValidation:
    canonical_key: str
    directed_gates: tuple[str, ...]
    action_tokens: tuple[str, ...]
    simulator_steps: int


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MazeModeBenchError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MazeModeBenchError(f"{label} must be a finite number")
    return result


def _bounded_int(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MazeModeBenchError(f"{label} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise MazeModeBenchError(f"{label} is outside [{minimum}, {maximum}]")
    return result


def _sha(value: Any, *, label: str) -> str:
    result = str(value)
    if _SHA256.fullmatch(result) is None:
        raise MazeModeBenchError(f"{label} must be a lowercase SHA-256")
    return result


def _canonical_spec_sha256(spec: Mapping[str, Any]) -> str:
    payload = dict(spec)
    claimed = payload.pop("spec_sha256", None)
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    observed = hashlib.sha256(canonical).hexdigest()
    if claimed is not None and claimed != observed:
        raise MazeModeBenchError("spec_sha256 does not match the canonical spec")
    return observed


def _parse_bounds(value: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    if not isinstance(value, list) or len(value) != 2:
        raise MazeModeBenchError("bounds_xy must contain x and y bounds")
    parsed: list[tuple[float, float]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) != 2:
            raise MazeModeBenchError(f"bounds_xy[{index}] must contain two values")
        low = _finite_float(raw[0], label=f"bounds_xy[{index}][0]")
        high = _finite_float(raw[1], label=f"bounds_xy[{index}][1]")
        if low >= high:
            raise MazeModeBenchError("each coordinate bound must be increasing")
        parsed.append((low, high))
    return parsed[0], parsed[1]


def _parse_gates(value: Any) -> tuple[DirectedGate, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise MazeModeBenchError("route_gates must contain between 1 and 32 gates")
    gates: list[DirectedGate] = []
    observed: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise MazeModeBenchError(f"route gate {index} must be an object")
        gate_id = str(raw.get("id", ""))
        if _GATE_ID.fullmatch(gate_id) is None:
            raise MazeModeBenchError(f"route gate {index} has an invalid id")
        if gate_id in observed:
            raise MazeModeBenchError(f"duplicate route gate id {gate_id!r}")
        observed.add(gate_id)
        axis = str(raw.get("axis", ""))
        if axis not in {"x", "y"}:
            raise MazeModeBenchError(f"route gate {gate_id!r} has invalid axis")
        coordinate = _finite_float(raw.get("coordinate"), label=f"{gate_id}.coordinate")
        raw_span = raw.get("span")
        if not isinstance(raw_span, list) or len(raw_span) != 2:
            raise MazeModeBenchError(f"route gate {gate_id!r} needs a span")
        span_min = _finite_float(raw_span[0], label=f"{gate_id}.span[0]")
        span_max = _finite_float(raw_span[1], label=f"{gate_id}.span[1]")
        if span_min >= span_max:
            raise MazeModeBenchError(f"route gate {gate_id!r} span is not increasing")
        hysteresis = _finite_float(raw.get("hysteresis"), label=f"{gate_id}.hysteresis")
        if not 0.0 < hysteresis <= 0.25:
            raise MazeModeBenchError(f"route gate {gate_id!r} hysteresis is invalid")
        gates.append(
            DirectedGate(
                gate_id=gate_id,
                axis=axis,
                coordinate=coordinate,
                span_min=span_min,
                span_max=span_max,
                hysteresis=hysteresis,
            )
        )
    return tuple(gates)


def _parse_maze_map(
    value: Any,
    reset_cell_value: Any,
    goal_cell_value: Any,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, int], tuple[int, int]]:
    if not isinstance(value, list) or not 5 <= len(value) <= 16:
        raise MazeModeBenchError("maze_map must contain between 5 and 16 rows")
    width = len(value[0]) if isinstance(value[0], list) else 0
    if not 5 <= width <= 16:
        raise MazeModeBenchError("maze_map must contain between 5 and 16 columns")
    rows: list[tuple[int, ...]] = []
    for index, raw_row in enumerate(value):
        if (
            not isinstance(raw_row, list)
            or len(raw_row) != width
            or any(isinstance(cell, bool) or cell not in {0, 1} for cell in raw_row)
        ):
            raise MazeModeBenchError(f"maze_map row {index} is malformed")
        rows.append(tuple(int(cell) for cell in raw_row))
    if any(cell != 1 for cell in rows[0] + rows[-1]):
        raise MazeModeBenchError("maze_map top and bottom borders must be walls")
    if any(row[0] != 1 or row[-1] != 1 for row in rows):
        raise MazeModeBenchError("maze_map left and right borders must be walls")

    def parse_cell(raw: Any, label: str) -> tuple[int, int]:
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in raw)
        ):
            raise MazeModeBenchError(f"{label} must be a [row, column] integer pair")
        cell = int(raw[0]), int(raw[1])
        if not (0 <= cell[0] < len(rows) and 0 <= cell[1] < width):
            raise MazeModeBenchError(f"{label} is outside maze_map")
        if rows[cell[0]][cell[1]] != 0:
            raise MazeModeBenchError(f"{label} must select a free maze cell")
        return cell

    reset_cell = parse_cell(reset_cell_value, "reset_cell")
    goal_cell = parse_cell(goal_cell_value, "goal_cell")
    if reset_cell == goal_cell:
        raise MazeModeBenchError("reset_cell and goal_cell must differ")
    return tuple(rows), reset_cell, goal_cell


def parse_maze_action_spec(spec: Mapping[str, Any]) -> MazeActionSpec:
    """Parse one trusted PointMaze or AntMaze action-program specification."""

    if not isinstance(spec, Mapping):
        raise MazeModeBenchError("maze spec must be an object")
    verifier = str(spec.get("verifier", ""))
    if verifier not in {POINT_MAZE_VERIFIER, ANT_MAZE_VERIFIER}:
        raise MazeModeBenchError("wrong maze action-program verifier")
    expected_version = (
        MAZE_ACTION_VERSION
        if verifier == POINT_MAZE_VERIFIER
        else ANT_MAZE_ACTION_VERSION
    )
    if spec.get("maze_action_version") != expected_version:
        raise MazeModeBenchError("unsupported maze action-program version")
    environment_id = str(spec.get("environment_id", ""))
    if _IDENTIFIER.fullmatch(environment_id) is None:
        raise MazeModeBenchError("environment_id is invalid")
    expected_prefix = "PointMaze" if verifier == POINT_MAZE_VERIFIER else "AntMaze"
    if not environment_id.startswith(expected_prefix):
        raise MazeModeBenchError("environment_id does not match the verifier")
    map_id = str(spec.get("map_id", ""))
    if _IDENTIFIER.fullmatch(map_id) is None:
        raise MazeModeBenchError("map_id is invalid")

    expected_actions = POINT_ACTIONS if verifier == POINT_MAZE_VERIFIER else ANT_ACTIONS
    raw_actions = spec.get("action_tokens")
    if not isinstance(raw_actions, list) or tuple(raw_actions) != expected_actions:
        raise MazeModeBenchError("action_tokens differ from the frozen verifier alphabet")

    environment_sha256 = _sha(spec.get("environment_sha256"), label="environment_sha256")
    raw_controller = spec.get("controller_sha256")
    if verifier == ANT_MAZE_VERIFIER:
        controller_sha256 = _sha(raw_controller, label="controller_sha256")
    else:
        if raw_controller is not None:
            raise MazeModeBenchError("PointMaze cannot declare a locomotion controller")
        controller_sha256 = None

    min_actions = _bounded_int(
        spec.get("min_actions"), label="min_actions", minimum=1, maximum=512
    )
    max_actions = _bounded_int(
        spec.get("max_actions"), label="max_actions", minimum=min_actions, maximum=512
    )
    action_repeat = _bounded_int(
        spec.get("action_repeat"),
        label="action_repeat",
        minimum=1,
        maximum=100 if verifier == POINT_MAZE_VERIFIER else 400,
    )
    success_threshold = _finite_float(
        spec.get("success_threshold"), label="success_threshold"
    )
    if not 0.0 < success_threshold <= 1.0:
        raise MazeModeBenchError("success_threshold is outside (0, 1]")
    max_segment_length = _finite_float(
        spec.get("max_segment_length"), label="max_segment_length"
    )
    if not 0.0 < max_segment_length <= 10.0:
        raise MazeModeBenchError("max_segment_length is outside (0, 10]")

    maze_map, reset_cell, goal_cell = _parse_maze_map(
        spec.get("maze_map"),
        spec.get("reset_cell"),
        spec.get("goal_cell"),
    )

    return MazeActionSpec(
        verifier=verifier,
        environment_id=environment_id,
        environment_sha256=environment_sha256,
        controller_sha256=controller_sha256,
        map_id=map_id,
        maze_map=maze_map,
        reset_cell=reset_cell,
        goal_cell=goal_cell,
        reset_seed=_bounded_int(
            spec.get("reset_seed"), label="reset_seed", minimum=0, maximum=2**31 - 1
        ),
        min_actions=min_actions,
        max_actions=max_actions,
        action_repeat=action_repeat,
        action_tokens=expected_actions,
        success_threshold=success_threshold,
        max_segment_length=max_segment_length,
        bounds=_parse_bounds(spec.get("bounds_xy")),
        gates=_parse_gates(spec.get("route_gates")),
        spec_sha256=_canonical_spec_sha256(spec),
    )


def parse_maze_action_program(candidate: str, spec: MazeActionSpec) -> tuple[str, ...]:
    """Parse a bounded whitespace/comma-separated action program."""

    text = str(candidate).strip()
    if not text or len(text) > _MAX_PROGRAM_CHARS:
        raise MazeModeBenchError("action program is empty or too long")
    tokens = tuple(token for token in re.split(r"[\s,]+", text.upper()) if token)
    if not spec.min_actions <= len(tokens) <= spec.max_actions:
        raise MazeModeBenchError("action program length is outside the frozen bounds")
    allowed = set(spec.action_tokens)
    unknown = [token for token in tokens if token not in allowed]
    if unknown:
        raise MazeModeBenchError(f"unknown action token {unknown[0]!r}")
    return tokens


def _parse_execution_record(value: Mapping[str, Any]) -> MazeExecutionRecord:
    if not isinstance(value, Mapping):
        raise MazeModeBenchError("execution record must be an object")
    raw_trajectory = value.get("trajectory_xy")
    if not isinstance(raw_trajectory, list) or len(raw_trajectory) < 2:
        raise MazeModeBenchError("execution trajectory must contain at least two points")
    trajectory: list[tuple[float, float]] = []
    for index, raw_point in enumerate(raw_trajectory):
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise MazeModeBenchError(f"trajectory point {index} is malformed")
        trajectory.append(
            (
                _finite_float(raw_point[0], label=f"trajectory[{index}].x"),
                _finite_float(raw_point[1], label=f"trajectory[{index}].y"),
            )
        )
    raw_success = value.get("success")
    if not isinstance(raw_success, bool):
        raise MazeModeBenchError("execution success must be boolean")
    raw_tokens = value.get("action_tokens")
    if not isinstance(raw_tokens, list) or not all(isinstance(x, str) for x in raw_tokens):
        raise MazeModeBenchError("execution action_tokens must be a string list")
    raw_controller = value.get("controller_sha256")
    controller_sha256 = None if raw_controller is None else _sha(
        raw_controller, label="execution.controller_sha256"
    )
    return MazeExecutionRecord(
        environment_sha256=_sha(
            value.get("environment_sha256"), label="execution.environment_sha256"
        ),
        controller_sha256=controller_sha256,
        spec_sha256=_sha(value.get("spec_sha256"), label="execution.spec_sha256"),
        reset_seed=_bounded_int(
            value.get("reset_seed"),
            label="execution.reset_seed",
            minimum=0,
            maximum=2**31 - 1,
        ),
        action_tokens=tuple(raw_tokens),
        success=raw_success,
        final_goal_distance=_finite_float(
            value.get("final_goal_distance"), label="execution.final_goal_distance"
        ),
        trajectory_xy=tuple(trajectory),
        simulator_steps=_bounded_int(
            value.get("simulator_steps"),
            label="execution.simulator_steps",
            minimum=1,
            maximum=1_000_000,
        ),
    )


def _outside_side(value: float, gate: DirectedGate) -> int:
    if value < gate.coordinate - gate.hysteresis:
        return -1
    if value > gate.coordinate + gate.hysteresis:
        return 1
    return 0


def _gate_crossing(
    first: tuple[float, float],
    second: tuple[float, float],
    gate: DirectedGate,
) -> str | None:
    axis_index = 0 if gate.axis == "x" else 1
    span_index = 1 - axis_index
    denominator = second[axis_index] - first[axis_index]
    if denominator == 0.0:
        return None
    fraction = (gate.coordinate - first[axis_index]) / denominator
    if not 0.0 <= fraction <= 1.0:
        return None
    span_value = first[span_index] + fraction * (
        second[span_index] - first[span_index]
    )
    if not gate.span_min <= span_value <= gate.span_max:
        return None
    direction = "+" if second[axis_index] > first[axis_index] else "-"
    return f"{gate.gate_id}{direction}"


def extract_directed_gate_route(
    trajectory_xy: Sequence[tuple[float, float]],
    spec: MazeActionSpec,
) -> tuple[str, ...]:
    """Extract a hysteresis-stable ordered directed bottleneck sequence."""

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
            raise MazeModeBenchError("one trajectory segment crosses multiple route gates")
        if step_crossings and (not route or route[-1] != step_crossings[0]):
            route.append(step_crossings[0])
    if not route:
        raise MazeModeBenchError("successful trajectory has no certified route gate")
    gate_ids = [crossing[:-1] for crossing in route]
    if len(gate_ids) != len(set(gate_ids)):
        raise MazeModeBenchError("trajectory recrosses a route gate")
    return tuple(route)


def validate_maze_execution(
    candidate: str,
    raw_spec: Mapping[str, Any],
    raw_execution: Mapping[str, Any],
) -> MazeValidation:
    """Bind one successful simulator execution to its semantic route key."""

    spec = parse_maze_action_spec(raw_spec)
    tokens = parse_maze_action_program(candidate, spec)
    execution = _parse_execution_record(raw_execution)
    if execution.environment_sha256 != spec.environment_sha256:
        raise MazeModeBenchError("execution environment hash does not match the spec")
    if execution.controller_sha256 != spec.controller_sha256:
        raise MazeModeBenchError("execution controller hash does not match the spec")
    if execution.spec_sha256 != spec.spec_sha256:
        raise MazeModeBenchError("execution is bound to a different spec")
    if execution.reset_seed != spec.reset_seed:
        raise MazeModeBenchError("execution reset seed does not match the spec")
    if execution.action_tokens != tokens:
        raise MazeModeBenchError("execution action program does not match the candidate")
    if not execution.success:
        raise MazeModeBenchError("unsuccessful execution has no ModeBench identity")
    if not 0.0 <= execution.final_goal_distance <= spec.success_threshold:
        raise MazeModeBenchError("successful execution fails the frozen goal threshold")
    if execution.simulator_steps > len(tokens) * spec.action_repeat:
        raise MazeModeBenchError("execution exceeds the action-program step budget")
    route = extract_directed_gate_route(execution.trajectory_xy, spec)
    action_version = (
        MAZE_ACTION_VERSION
        if spec.verifier == POINT_MAZE_VERIFIER
        else ANT_MAZE_ACTION_VERSION
    )
    key = f"{spec.verifier}:{action_version}:{spec.map_id}:" + ",".join(route)
    return MazeValidation(
        canonical_key=key,
        directed_gates=route,
        action_tokens=tokens,
        simulator_steps=execution.simulator_steps,
    )
