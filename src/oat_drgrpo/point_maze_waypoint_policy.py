"""Public prompt and transition contracts for sequential PointMaze waypoints."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .point_maze_waypoint import POINT_WAYPOINT_ACTIONS


POINT_WAYPOINT_LABELS = tuple("ABCD")
POINT_WAYPOINT_PROMPT_FORMATS = ("qwen_chatml", "falcon3")
POINT_WAYPOINT_ACTION_TO_LABEL = dict(
    zip(POINT_WAYPOINT_ACTIONS, POINT_WAYPOINT_LABELS)
)
POINT_WAYPOINT_LABEL_TO_ACTION = dict(
    zip(POINT_WAYPOINT_LABELS, POINT_WAYPOINT_ACTIONS)
)
POINT_WAYPOINT_SYSTEM = (
    "Plan a route through PointMaze one adjacent free cell at a time. "
    "Return exactly one listed capital option letter."
)
POINT_WAYPOINT_TERMINAL_SYSTEM = (
    "PointMaze waypoint terminal padding. Return one public action label."
)
POINT_WAYPOINT_TERMINAL_USER = (
    "The episode is terminal; this fixed slot has zero decision mask. "
    "Global labels: A B C D."
)


def _serialize_point_waypoint_chat(
    system: str,
    user: str,
    *,
    prompt_format: str,
) -> str:
    if prompt_format == "qwen_chatml":
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if prompt_format == "falcon3":
        # Exact no-tools serialization of Falcon3-1B-Instruct's published
        # tokenizer chat template, ending at the assistant generation boundary.
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{user}\n"
            "<|assistant|>\n"
        )
    raise ValueError(f"unsupported PointMaze prompt format: {prompt_format!r}")


def convert_point_waypoint_prompt(prompt: str, *, prompt_format: str) -> str:
    """Convert a frozen Qwen-ChatML waypoint example to a model surface."""

    system_prefix = "<|im_start|>system\n"
    user_separator = "<|im_end|>\n<|im_start|>user\n"
    assistant_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if not prompt.startswith(system_prefix) or not prompt.endswith(assistant_suffix):
        raise ValueError("frozen PointMaze prompt does not match the audited contract")
    body = prompt[len(system_prefix) : -len(assistant_suffix)]
    if body.count(user_separator) != 1:
        raise ValueError("frozen PointMaze prompt has an ambiguous message boundary")
    system, user = body.split(user_separator)
    return _serialize_point_waypoint_chat(
        system,
        user,
        prompt_format=prompt_format,
    )


def render_point_waypoint_terminal_padding_prompt(
    *, prompt_format: str = "qwen_chatml"
) -> str:
    return _serialize_point_waypoint_chat(
        POINT_WAYPOINT_TERMINAL_SYSTEM,
        POINT_WAYPOINT_TERMINAL_USER,
        prompt_format=prompt_format,
    )


POINT_WAYPOINT_TERMINAL_PADDING_PROMPT = (
    render_point_waypoint_terminal_padding_prompt()
)


def _compact_coordinate(values: Sequence[Any]) -> str:
    numbers = []
    for raw in values:
        value = float(raw)
        if abs(value) < 0.0005:
            value = 0.0
        numbers.append(f"{value:.3f}")
    if len(numbers) != 2:
        raise ValueError("PointMaze coordinates must have two components")
    return "(" + ",".join(numbers) + ")"


def _cell(values: Sequence[Any] | None, *, allow_none: bool = False) -> str:
    if values is None and allow_none:
        return "START"
    if values is None or len(values) != 2:
        raise ValueError("PointMaze cell must have two components")
    return f"({int(values[0])},{int(values[1])})"


def point_waypoint_allowed_labels(actions: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(action).upper() for action in actions)
    if (
        not normalized
        or len(normalized) != len(set(normalized))
        or any(action not in POINT_WAYPOINT_ACTION_TO_LABEL for action in normalized)
    ):
        raise ValueError("PointMaze waypoint actions must be a nonempty unique subset")
    expected = tuple(
        action for action in POINT_WAYPOINT_ACTIONS if action in set(normalized)
    )
    if normalized != expected:
        raise ValueError("PointMaze waypoint actions must use canonical order")
    return tuple(POINT_WAYPOINT_ACTION_TO_LABEL[action] for action in normalized)


def render_point_waypoint_prompt(
    problem: str,
    observation: Mapping[str, Any],
    *,
    prompt_format: str = "qwen_chatml",
) -> str:
    """Render one fully observed route-planning decision without route hints."""

    allowed = tuple(str(action) for action in observation["allowed_actions"])
    labels = point_waypoint_allowed_labels(allowed)
    options = "\n".join(f"{label}: {action}" for label, action in zip(labels, allowed))
    user = (
        "PUBLIC MAZE\n"
        + str(problem).strip()
        + "\n\nPUBLIC DECISION STATE\n"
        + f"current_cell={_cell(observation['current_cell'])}\n"
        + f"previous_cell={_cell(observation.get('previous_cell'), allow_none=True)}\n"
        + f"goal_cell={_cell(observation['goal_cell'])}\n"
        + f"position_xy={_compact_coordinate(observation['achieved_goal'])}\n"
        + f"velocity_xy={_compact_coordinate(observation['velocity_xy'])}\n"
        + f"remaining_waypoints={int(observation['remaining_actions'])}\n"
        + "Rows increase downward; N decreases row and E increases column.\n"
        + "LEGAL ADJACENT MOVES\n"
        + options
        + "\nChoose the next cell."
    )
    return _serialize_point_waypoint_chat(
        POINT_WAYPOINT_SYSTEM,
        user,
        prompt_format=prompt_format,
    )


def point_waypoint_transition_sha256(
    *,
    before: Mapping[str, Any],
    action: str,
    after: Mapping[str, Any],
) -> str:
    """Hash every public field that can condition a waypoint decision."""

    def state(value: Mapping[str, Any]) -> dict[str, Any]:
        required = (
            "achieved_goal",
            "desired_goal",
            "velocity_xy",
            "current_cell",
            "goal_cell",
            "remaining_actions",
            "allowed_actions",
        )
        if any(name not in value for name in required):
            raise ValueError("Point waypoint transition lacks a public state field")
        previous = value.get("previous_cell")
        return {
            "achieved_goal": [float(item) for item in value["achieved_goal"]],
            "desired_goal": [float(item) for item in value["desired_goal"]],
            "velocity_xy": [float(item) for item in value["velocity_xy"]],
            "current_cell": [int(item) for item in value["current_cell"]],
            "previous_cell": (
                None if previous is None else [int(item) for item in previous]
            ),
            "goal_cell": [int(item) for item in value["goal_cell"]],
            "remaining_actions": int(value["remaining_actions"]),
            "allowed_actions": [str(item) for item in value["allowed_actions"]],
            "done": bool(value.get("done", False)),
            "success": bool(value.get("success", False)),
        }

    token = str(action).upper()
    if token not in POINT_WAYPOINT_ACTIONS:
        raise ValueError("Point waypoint transition action is invalid")
    encoded = json.dumps(
        {
            "schema": "public-point-waypoint-transition-v1",
            "before": state(before),
            "action": token,
            "after": state(after),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
