"""Development-only grid actions with trusted PointMaze feedback control.

The language policy chooses a cardinal path through the prompt-visible maze.
Each token advances one legal grid cell. A hash-bound PD controller handles
only the continuous point-mass dynamics needed to reach that selected cell.
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
from .maze_runtime_identity import maze_runtime_identity


POINT_GRID_VERIFIER = "point_maze_grid_program"
POINT_GRID_ACTION_VERSION = "point-grid-action-development-v1"
POINT_GRID_ACTIONS = ("N", "E", "S", "W")

_CONTROLLER = {
    "controller": "point-grid-pd",
    "development_version": 1,
    "kp": 2.0,
    "kd": 0.4,
    "position_tolerance": 0.08,
    "velocity_tolerance": 0.5,
    "max_steps_per_action": 100,
}
POINT_GRID_CONTROLLER_SHA256 = hashlib.sha256(
    json.dumps(
        _CONTROLLER,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()

_DELTAS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
_MAX_PROGRAM_CHARS = 1024


@dataclass(frozen=True)
class PointGridSpec:
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


def adapt_point_maze_spec(raw_legacy_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Create a separated development spec from one admitted v1 Point map."""

    legacy = parse_maze_action_spec(raw_legacy_spec)
    if legacy.verifier != POINT_MAZE_VERIFIER:
        raise MazeModeBenchError("grid adapter requires a PointMaze v1 spec")
    free_cells = sum(cell == 0 for row in legacy.maze_map for cell in row)
    adapted = {
        "verifier": POINT_GRID_VERIFIER,
        "grid_action_version": POINT_GRID_ACTION_VERSION,
        "environment_id": legacy.environment_id,
        "environment_sha256": legacy.environment_sha256,
        "controller_sha256": POINT_GRID_CONTROLLER_SHA256,
        "map_id": f"{legacy.map_id}_grid_dev_v1",
        "maze_map": [list(row) for row in legacy.maze_map],
        "reset_cell": list(legacy.reset_cell),
        "goal_cell": list(legacy.goal_cell),
        "reset_seed": legacy.reset_seed,
        "min_actions": 2,
        "max_actions": min(64, free_cells - 1),
        "max_steps_per_action": _CONTROLLER["max_steps_per_action"],
        "action_tokens": list(POINT_GRID_ACTIONS),
        "success_threshold": legacy.success_threshold,
        "max_segment_length": legacy.max_segment_length,
        "bounds_xy": [list(bound) for bound in legacy.bounds],
        "route_gates": [
            {
                "id": gate.gate_id,
                "axis": gate.axis,
                "coordinate": gate.coordinate,
                "span": [gate.span_min, gate.span_max],
                "hysteresis": gate.hysteresis,
            }
            for gate in legacy.gates
        ],
        "position_tolerance": _CONTROLLER["position_tolerance"],
        "velocity_tolerance": _CONTROLLER["velocity_tolerance"],
        "kp": _CONTROLLER["kp"],
        "kd": _CONTROLLER["kd"],
    }
    adapted["spec_sha256"] = _canonical_sha256(adapted, remove_claim=False)
    return adapted


def parse_point_grid_spec(raw_spec: Mapping[str, Any]) -> PointGridSpec:
    if not isinstance(raw_spec, Mapping):
        raise MazeModeBenchError("Point grid spec must be an object")
    if raw_spec.get("verifier") != POINT_GRID_VERIFIER:
        raise MazeModeBenchError("wrong Point grid verifier")
    if raw_spec.get("grid_action_version") != POINT_GRID_ACTION_VERSION:
        raise MazeModeBenchError("unsupported Point grid action version")
    if raw_spec.get("action_tokens") != list(POINT_GRID_ACTIONS):
        raise MazeModeBenchError("Point grid action alphabet differs from the contract")
    if raw_spec.get("controller_sha256") != POINT_GRID_CONTROLLER_SHA256:
        raise MazeModeBenchError("Point grid controller hash differs from the runtime")
    observed_hash = _canonical_sha256(raw_spec, remove_claim=True)
    if raw_spec.get("spec_sha256") != observed_hash:
        raise MazeModeBenchError("Point grid spec hash is invalid")

    fixed = {
        "max_steps_per_action": int(_CONTROLLER["max_steps_per_action"]),
        "position_tolerance": float(_CONTROLLER["position_tolerance"]),
        "velocity_tolerance": float(_CONTROLLER["velocity_tolerance"]),
        "kp": float(_CONTROLLER["kp"]),
        "kd": float(_CONTROLLER["kd"]),
    }
    for name, expected in fixed.items():
        value = raw_spec.get(name)
        if isinstance(value, bool) or value != expected:
            raise MazeModeBenchError(f"Point grid controller field {name} changed")
    min_actions = raw_spec.get("min_actions")
    max_actions = raw_spec.get("max_actions")
    if (
        isinstance(min_actions, bool)
        or not isinstance(min_actions, int)
        or isinstance(max_actions, bool)
        or not isinstance(max_actions, int)
        or not 1 <= min_actions <= max_actions <= 64
    ):
        raise MazeModeBenchError("Point grid action bounds are invalid")

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
    translated["spec_sha256"] = _canonical_sha256(
        translated, remove_claim=False
    )
    base_spec = parse_maze_action_spec(translated)
    return PointGridSpec(
        base_spec=base_spec,
        controller_sha256=POINT_GRID_CONTROLLER_SHA256,
        min_actions=min_actions,
        max_actions=max_actions,
        max_steps_per_action=fixed["max_steps_per_action"],
        position_tolerance=fixed["position_tolerance"],
        velocity_tolerance=fixed["velocity_tolerance"],
        kp=fixed["kp"],
        kd=fixed["kd"],
        spec_sha256=observed_hash,
    )


def parse_point_grid_program(
    candidate: str,
    spec: PointGridSpec,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    text = str(candidate).strip()
    if not text or len(text) > _MAX_PROGRAM_CHARS:
        raise MazeModeBenchError("Point grid program is empty or too long")
    tokens = tuple(token for token in re.split(r"[\s,]+", text.upper()) if token)
    if not spec.min_actions <= len(tokens) <= spec.max_actions:
        raise MazeModeBenchError("Point grid program length is outside its bounds")
    unknown = [token for token in tokens if token not in _DELTAS]
    if unknown:
        raise MazeModeBenchError(f"unknown Point grid action {unknown[0]!r}")

    current = spec.base_spec.reset_cell
    cells: list[tuple[int, int]] = []
    for token in tokens:
        delta = _DELTAS[token]
        current = current[0] + delta[0], current[1] + delta[1]
        if not (
            0 <= current[0] < len(spec.base_spec.maze_map)
            and 0 <= current[1] < len(spec.base_spec.maze_map[0])
            and spec.base_spec.maze_map[current[0]][current[1]] == 0
        ):
            raise MazeModeBenchError("Point grid program enters a wall")
        cells.append(current)
    if current != spec.base_spec.goal_cell:
        raise MazeModeBenchError("Point grid program does not end at the goal cell")
    return tokens, tuple(cells)


def _cell_xy(spec: PointGridSpec, cell: tuple[int, int]) -> tuple[float, float]:
    height = len(spec.base_spec.maze_map)
    width = len(spec.base_spec.maze_map[0])
    return cell[1] + 0.5 - width / 2.0, height / 2.0 - (cell[0] + 0.5)


def _edge_crossings(
    first: tuple[float, float],
    second: tuple[float, float],
    gates: Sequence[DirectedGate],
) -> tuple[str, ...]:
    crossings: list[str] = []
    for gate in gates:
        axis = 0 if gate.axis == "x" else 1
        span_axis = 1 - axis
        delta = second[axis] - first[axis]
        if delta == 0.0:
            continue
        fraction = (gate.coordinate - first[axis]) / delta
        if not 0.0 <= fraction <= 1.0:
            continue
        span = first[span_axis] + fraction * (second[span_axis] - first[span_axis])
        if gate.span_min <= span <= gate.span_max:
            crossings.append(f"{gate.gate_id}{'+' if delta > 0 else '-'}")
    return tuple(crossings)


def find_point_grid_route_programs(raw_spec: Mapping[str, Any]) -> dict[str, str]:
    """Find shortest graph fixtures for every reachable single-gate route."""

    spec = parse_point_grid_spec(raw_spec)
    def outside_side(point: tuple[float, float], gate: DirectedGate) -> int:
        coordinate = point[0] if gate.axis == "x" else point[1]
        if coordinate < gate.coordinate - gate.hysteresis:
            return -1
        if coordinate > gate.coordinate + gate.hysteresis:
            return 1
        return 0

    start_cell = spec.base_spec.reset_cell
    start_xy = _cell_xy(spec, start_cell)
    start_memory = tuple(
        (side, start_cell[0], start_cell[1]) if side else None
        for side in (outside_side(start_xy, gate) for gate in spec.base_spec.gates)
    )
    start = (start_cell, (), start_memory)
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
        for token in POINT_GRID_ACTIONS:
            delta = _DELTAS[token]
            nxt = cell[0] + delta[0], cell[1] + delta[1]
            if not (
                0 <= nxt[0] < len(spec.base_spec.maze_map)
                and 0 <= nxt[1] < len(spec.base_spec.maze_map[0])
                and spec.base_spec.maze_map[nxt[0]][nxt[1]] == 0
            ):
                continue
            next_memory = list(memory)
            crossings: list[str] = []
            next_xy = _cell_xy(spec, nxt)
            for gate_index, gate in enumerate(spec.base_spec.gates):
                side = outside_side(next_xy, gate)
                if side == 0:
                    continue
                previous = memory[gate_index]
                if previous is not None and side != previous[0]:
                    previous_cell = previous[1], previous[2]
                    crossings.extend(
                        _edge_crossings(
                            _cell_xy(spec, previous_cell), next_xy, (gate,)
                        )
                    )
                next_memory[gate_index] = (side, nxt[0], nxt[1])
            if len(crossings) > 1:
                continue
            next_route = route
            if crossings:
                crossing = crossings[0]
                gate_id = crossing[:-1]
                if any(existing[:-1] == gate_id for existing in route):
                    continue
                next_route = route + (crossing,)
            next_state = (nxt, next_route, tuple(next_memory))
            if next_state in parent:
                continue
            parent[next_state] = state, token
            depth[next_state] = depth[state] + 1
            queue.append(next_state)

    programs: dict[str, str] = {}
    for route, state in sorted(found.items()):
        tokens: list[str] = []
        cursor = state
        while parent[cursor] is not None:
            previous, token = parent[cursor]
            tokens.append(token)
            cursor = previous
        tokens.reverse()
        if spec.min_actions <= len(tokens) <= spec.max_actions:
            programs[",".join(route)] = " ".join(tokens)
    return programs


def validate_point_grid_execution(
    candidate: str,
    raw_spec: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> MazeValidation:
    spec = parse_point_grid_spec(raw_spec)
    tokens, _cells = parse_point_grid_program(candidate, spec)
    if execution.get("environment_sha256") != spec.base_spec.environment_sha256:
        raise MazeModeBenchError("Point grid execution environment hash differs")
    if execution.get("controller_sha256") != spec.controller_sha256:
        raise MazeModeBenchError("Point grid execution controller hash differs")
    if execution.get("spec_sha256") != spec.spec_sha256:
        raise MazeModeBenchError("Point grid execution spec hash differs")
    if execution.get("reset_seed") != spec.base_spec.reset_seed:
        raise MazeModeBenchError("Point grid execution reset seed differs")
    if execution.get("action_tokens") != list(tokens):
        raise MazeModeBenchError("Point grid execution actions differ")
    if execution.get("success") is not True:
        raise MazeModeBenchError("unsuccessful Point grid execution")
    distance = float(execution.get("final_goal_distance", math.inf))
    if not math.isfinite(distance) or not 0.0 <= distance <= spec.base_spec.success_threshold:
        raise MazeModeBenchError("Point grid execution misses the goal threshold")
    simulator_steps = execution.get("simulator_steps")
    if (
        isinstance(simulator_steps, bool)
        or not isinstance(simulator_steps, int)
        or not 1 <= simulator_steps <= len(tokens) * spec.max_steps_per_action
    ):
        raise MazeModeBenchError("Point grid execution exceeds its step budget")
    raw_trajectory = execution.get("trajectory_xy")
    if not isinstance(raw_trajectory, list):
        raise MazeModeBenchError("Point grid execution trajectory is missing")
    trajectory: list[tuple[float, float]] = []
    for point in raw_trajectory:
        if not isinstance(point, list) or len(point) != 2:
            raise MazeModeBenchError("Point grid execution trajectory is malformed")
        xy = float(point[0]), float(point[1])
        if not all(math.isfinite(value) for value in xy):
            raise MazeModeBenchError("Point grid execution trajectory is nonfinite")
        trajectory.append(xy)
    route = extract_directed_gate_route(trajectory, spec.base_spec)
    return MazeValidation(
        canonical_key=(
            f"{POINT_GRID_VERIFIER}:{POINT_GRID_ACTION_VERSION}:"
            f"{spec.base_spec.map_id}:" + ",".join(route)
        ),
        directed_gates=route,
        action_tokens=tokens,
        simulator_steps=simulator_steps,
    )


def execute_point_grid(candidate: str, raw_spec: Mapping[str, Any]):
    """Execute one grid program in the pinned networkless PointMaze runtime."""

    import gymnasium as gym
    import gymnasium_robotics
    import numpy as np

    gym.register_envs(gymnasium_robotics)
    spec = parse_point_grid_spec(raw_spec)
    runtime = maze_runtime_identity()
    if spec.base_spec.environment_sha256 != runtime["point_environment_sha256"]:
        raise MazeModeBenchError("Point grid environment hash differs from runtime")
    tokens, cells = parse_point_grid_program(candidate, spec)
    env = gym.make(
        spec.base_spec.environment_id,
        maze_map=[list(row) for row in spec.base_spec.maze_map],
        reward_type="sparse",
        continuing_task=False,
        reset_target=False,
        max_episode_steps=len(tokens) * spec.max_steps_per_action,
    )
    try:
        env.unwrapped.position_noise_range = 0.0
        observation, info = env.reset(
            seed=spec.base_spec.reset_seed,
            options={
                "reset_cell": list(spec.base_spec.reset_cell),
                "goal_cell": list(spec.base_spec.goal_cell),
            },
        )
        trajectory = [observation["achieved_goal"].astype(float).tolist()]
        simulator_steps = 0
        success = bool(info.get("success", False))
        for cell in cells:
            target = env.unwrapped.maze.cell_rowcol_to_xy(np.asarray(cell))
            reached = False
            for _ in range(spec.max_steps_per_action):
                state = observation["observation"]
                position = state[:2]
                velocity = state[2:4]
                action = np.clip(
                    spec.kp * (target - position) - spec.kd * velocity,
                    -1.0,
                    1.0,
                )
                observation, _reward, terminated, truncated, info = env.step(action)
                simulator_steps += 1
                trajectory.append(
                    observation["achieved_goal"].astype(float).tolist()
                )
                position_error = float(
                    np.linalg.norm(observation["achieved_goal"] - target)
                )
                speed = float(np.linalg.norm(observation["observation"][2:4]))
                success = bool(info.get("success", False))
                if (
                    position_error <= spec.position_tolerance
                    and speed <= spec.velocity_tolerance
                ):
                    reached = True
                    break
                if terminated or truncated:
                    break
            if success:
                break
            if not reached:
                break
        distance = float(
            np.linalg.norm(
                observation["achieved_goal"] - observation["desired_goal"]
            )
        )
    finally:
        env.close()
    execution = {
        "environment_sha256": spec.base_spec.environment_sha256,
        "controller_sha256": spec.controller_sha256,
        "spec_sha256": spec.spec_sha256,
        "reset_seed": spec.base_spec.reset_seed,
        "action_tokens": list(tokens),
        "success": success,
        "final_goal_distance": distance,
        "trajectory_xy": trajectory,
        "simulator_steps": simulator_steps,
    }
    return validate_point_grid_execution(candidate, raw_spec, execution), execution
