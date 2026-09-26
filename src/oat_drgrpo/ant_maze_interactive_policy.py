"""Public one-token policy contract for the constrained AntMaze interface."""

from __future__ import annotations

from typing import Any, Sequence


ANT_POLICY_ACTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
ANT_TERMINAL_PADDING_PROMPT = (
    "<|im_start|>system\nAntMaze terminal padding. Return one compass token."
    "<|im_end|>\n<|im_start|>user\nThe episode is terminal; this fixed slot "
    "has zero decision mask. Options: N NE E SE S SW W NW."
    "<|im_end|>\n<|im_start|>assistant\n"
)


def _coordinate(raw: Any) -> str:
    values = [float(value) for value in raw]
    if len(values) != 2:
        raise ValueError("AntMaze public coordinates must have two components")
    normalized = [0.0 if abs(value) < 0.0005 else value for value in values]
    return "(" + ",".join(f"{value:.3f}" for value in normalized) + ")"


def _public_maze(problem: str) -> str:
    marker = "\nReturn 4 to 16 commands"
    return str(problem).split(marker, 1)[0].strip()


def render_ant_policy_prompt(
    problem: str,
    observation: dict[str, Any],
    actions: Sequence[str],
    _history: Sequence[str],
) -> str:
    action_tuple = tuple(str(action) for action in actions)
    if action_tuple != ANT_POLICY_ACTIONS:
        raise ValueError("AntMaze action alphabet differs from the public contract")
    return (
        "<|im_start|>system\n"
        "Control the frozen Ant one adjacent-cell target at a time. Choose "
        "exactly one compass action and return only that single token."
        "<|im_end|>\n<|im_start|>user\n"
        "PUBLIC MAZE\n"
        + _public_maze(problem)
        + "\n\nPUBLIC MARKOV STATE\n"
        + f"position_xy={_coordinate(observation['achieved_goal'])}\n"
        + f"velocity_xy={_coordinate(observation['velocity_xy'])}\n"
        + f"goal_xy={_coordinate(observation['desired_goal'])}\n"
        + f"remaining_actions={int(observation['remaining_actions'])}\n"
        + "Coordinates use +x east/right and +y north/up. Each action sets "
        + "the next adjacent grid-cell target for the fixed low-level controller.\n"
        + "OPTIONS: N NE E SE S SW W NW\n"
        + "Choose the next compass action."
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
