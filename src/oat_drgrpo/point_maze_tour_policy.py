"""Public prompt and transition contracts for option-level PointMaze Tours.

Labels are global: ``A`` always names ``lm1``, ``B`` names ``lm2``, and so on
for the map's landmarks.  Only the labels of landmarks that are still unvisited
enter sampling, SFT normalization, on-policy scoring, or replay scoring, so the
mask removes formatting as a nuisance variable without ever hinting which order
is cheap.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


POINT_TOUR_LABELS = tuple("ABCDEF")
POINT_TOUR_PROMPT_FORMATS = ("qwen_chatml", "falcon3")
POINT_TOUR_SYSTEM = (
    "Plan a tour of a PointMaze one landmark at a time. "
    "Return exactly one listed capital option letter."
)
POINT_TOUR_TERMINAL_SYSTEM = (
    "PointMaze tour terminal padding. Return one public action label."
)
POINT_TOUR_TERMINAL_USER = (
    "The tour is over; this fixed slot has zero decision mask. "
    "Global labels: A B C D E F."
)


def _serialize_point_tour_chat(
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
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    raise ValueError(f"unsupported PointMaze tour prompt format: {prompt_format!r}")


def convert_point_tour_prompt(prompt: str, *, prompt_format: str) -> str:
    """Convert a frozen Qwen-ChatML tour example to another model surface."""

    system_prefix = "<|im_start|>system\n"
    user_separator = "<|im_end|>\n<|im_start|>user\n"
    assistant_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if not prompt.startswith(system_prefix) or not prompt.endswith(assistant_suffix):
        raise ValueError("frozen PointMaze tour prompt does not match the contract")
    body = prompt[len(system_prefix) : -len(assistant_suffix)]
    if body.count(user_separator) != 1:
        raise ValueError("frozen PointMaze tour prompt has an ambiguous boundary")
    system, user = body.split(user_separator)
    return _serialize_point_tour_chat(system, user, prompt_format=prompt_format)


def render_point_tour_terminal_padding_prompt(
    *, prompt_format: str = "qwen_chatml"
) -> str:
    return _serialize_point_tour_chat(
        POINT_TOUR_TERMINAL_SYSTEM,
        POINT_TOUR_TERMINAL_USER,
        prompt_format=prompt_format,
    )


POINT_TOUR_TERMINAL_PADDING_PROMPT = render_point_tour_terminal_padding_prompt()


def _compact_coordinate(values: Sequence[Any]) -> str:
    numbers = []
    for raw in values:
        value = float(raw)
        if abs(value) < 0.0005:
            value = 0.0
        numbers.append(f"{value:.3f}")
    if len(numbers) != 2:
        raise ValueError("PointMaze tour coordinates must have two components")
    return "(" + ",".join(numbers) + ")"


def _cell(values: Sequence[Any]) -> str:
    if values is None or len(values) != 2:
        raise ValueError("PointMaze tour cell must have two components")
    return f"({int(values[0])},{int(values[1])})"


def point_tour_label_map(landmark_ids: Sequence[str]) -> dict[str, str]:
    """Global, order-independent label assignment for a map's landmarks."""

    identifiers = [str(item) for item in landmark_ids]
    if not identifiers or len(identifiers) > len(POINT_TOUR_LABELS):
        raise ValueError("PointMaze tour landmark count is unsupported")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("PointMaze tour landmark ids must be unique")
    return dict(zip(POINT_TOUR_LABELS, identifiers))


def point_tour_allowed_labels(
    landmark_ids: Sequence[str],
    allowed_landmarks: Sequence[str],
) -> tuple[str, ...]:
    """Labels of the still-unvisited landmarks, in global label order."""

    labels = point_tour_label_map(landmark_ids)
    allowed = {str(item) for item in allowed_landmarks}
    if not allowed or not allowed.issubset(set(labels.values())):
        raise ValueError("PointMaze tour allowed landmarks are not a valid subset")
    return tuple(
        label for label in POINT_TOUR_LABELS if labels.get(label) in allowed
    )


def render_point_tour_prompt(
    problem: str,
    observation: Mapping[str, Any],
    *,
    prompt_format: str = "qwen_chatml",
) -> str:
    """Render one fully observed tour decision without any order hint."""

    landmark_ids = [str(item) for item in observation["landmark_ids"]]
    labels = point_tour_label_map(landmark_ids)
    cells = {
        identifier: cell
        for identifier, cell in zip(landmark_ids, observation["landmark_cells"])
    }
    allowed = [str(item) for item in observation["allowed_landmarks"]]
    option_labels = point_tour_allowed_labels(landmark_ids, allowed)
    options = "\n".join(
        f"{label}: go to {labels[label]} at {_cell(cells[labels[label]])}"
        for label in option_labels
    )
    visited = [str(item) for item in observation["visited"]]
    user = (
        "PUBLIC MAP\n"
        + str(problem).strip()
        + "\n\nPUBLIC DECISION STATE\n"
        + f"current_cell={_cell(observation['current_cell'])}\n"
        + f"goal_cell={_cell(observation['goal_cell'])}\n"
        + f"position_xy={_compact_coordinate(observation['achieved_goal'])}\n"
        + f"velocity_xy={_compact_coordinate(observation['velocity_xy'])}\n"
        + f"visited={'>'.join(visited) if visited else 'NONE'}\n"
        + f"remaining_steps={int(observation['remaining_steps'])}\n"
        + "Every landmark must be visited before G. Legs are driven for you.\n"
        + "UNVISITED LANDMARKS\n"
        + options
        + "\nChoose the next landmark."
    )
    return _serialize_point_tour_chat(
        POINT_TOUR_SYSTEM,
        user,
        prompt_format=prompt_format,
    )


def point_tour_transition_sha256(
    *,
    before: Mapping[str, Any],
    landmark: str,
    after: Mapping[str, Any],
) -> str:
    """Hash every public field that can condition a tour decision."""

    def state(value: Mapping[str, Any]) -> dict[str, Any]:
        required = (
            "achieved_goal",
            "desired_goal",
            "velocity_xy",
            "current_cell",
            "goal_cell",
            "landmark_ids",
            "landmark_cells",
            "visited",
            "allowed_landmarks",
            "remaining_steps",
        )
        if any(name not in value for name in required):
            raise ValueError("Point tour transition lacks a public state field")
        return {
            "achieved_goal": [float(item) for item in value["achieved_goal"]],
            "desired_goal": [float(item) for item in value["desired_goal"]],
            "velocity_xy": [float(item) for item in value["velocity_xy"]],
            "current_cell": [int(item) for item in value["current_cell"]],
            "goal_cell": [int(item) for item in value["goal_cell"]],
            "landmark_ids": [str(item) for item in value["landmark_ids"]],
            "landmark_cells": [
                [int(component) for component in cell]
                for cell in value["landmark_cells"]
            ],
            "visited": [str(item) for item in value["visited"]],
            "allowed_landmarks": [str(item) for item in value["allowed_landmarks"]],
            "remaining_steps": int(value["remaining_steps"]),
            "done": bool(value.get("done", False)),
            "success": bool(value.get("success", False)),
        }

    encoded = json.dumps(
        {
            "schema": "public-point-tour-transition-v1",
            "before": state(before),
            "landmark": str(landmark),
            "after": state(after),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
