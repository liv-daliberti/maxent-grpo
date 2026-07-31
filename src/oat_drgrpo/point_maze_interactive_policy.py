"""Public prompt contract for a stepwise PointMaze language policy."""

from __future__ import annotations

from typing import Any, Sequence


POINT_POLICY_LABELS = tuple("ABCDEFGHI")
POINT_TERMINAL_PADDING_PROMPT = (
    "<|im_start|>system\nPointMaze terminal padding. "
    "Return one public action label.<|im_end|>\n"
    "<|im_start|>user\nThe episode is terminal; this fixed slot has zero "
    "decision mask. Options: A B C D E F G H I."
    "<|im_end|>\n<|im_start|>assistant\n"
)


def _compact_coordinate(values: Any) -> str:
    numbers = []
    for raw in values:
        value = float(raw)
        if abs(value) < 0.0005:
            value = 0.0
        numbers.append(f"{value:.3f}")
    if len(numbers) != 2:
        raise ValueError("PointMaze coordinates must have two components")
    return "(" + ",".join(numbers) + ")"


def _public_maze_only(problem: str) -> str:
    """Drop the endpoint-program instruction from the stepwise policy view."""

    marker = "\n\nOutput only a whitespace-separated action program"
    return str(problem).split(marker, 1)[0].strip()


def render_point_policy_prompt(
    problem: str,
    observation: dict[str, Any],
    actions: Sequence[str],
    history: Sequence[str],
) -> str:
    """Render one public state with no planner or verifier feedback."""

    action_tuple = tuple(str(action) for action in actions)
    if len(action_tuple) != len(POINT_POLICY_LABELS):
        raise ValueError("PointMaze alphabet differs from the label contract")
    options = "\n".join(
        f"{POINT_POLICY_LABELS[index]}: {action}"
        for index, action in enumerate(action_tuple)
    )
    recent = " ".join(str(action) for action in history[-32:]) if history else "(none)"
    return (
        "<|im_start|>system\n"
        "Act as a closed-loop PointMaze policy. Choose exactly one listed "
        "force pulse. Return only its single capital option letter."
        "<|im_end|>\n<|im_start|>user\n"
        + str(problem)
        + "\n\nCURRENT SIMULATOR OBSERVATION\n"
        + f"position_xy={observation['achieved_goal']}\n"
        + f"goal_xy={observation['desired_goal']}\n"
        + f"remaining_actions={observation['remaining_actions']}\n"
        + f"recent_actions={recent}\n"
        + "N is upward in the printed map; E is rightward. "
        + "Choose the next short force pulse from:\n"
        + options
        + "\nChoose one option letter."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def render_point_policy_prompt_v2(
    problem: str,
    observation: dict[str, Any],
    actions: Sequence[str],
    _history: Sequence[str],
) -> str:
    """Render the compact public-state v2 prompt without action-history shortcuts."""

    action_tuple = tuple(str(action) for action in actions)
    if len(action_tuple) != len(POINT_POLICY_LABELS):
        raise ValueError("PointMaze alphabet differs from the label contract")
    options = "\n".join(
        f"{POINT_POLICY_LABELS[index]}: {action}"
        for index, action in enumerate(action_tuple)
    )
    position = _compact_coordinate(observation["achieved_goal"])
    goal = _compact_coordinate(observation["desired_goal"])
    return (
        "<|im_start|>system\n"
        "Control a PointMaze point mass one force pulse at a time. "
        "Return exactly one capital option letter."
        "<|im_end|>\n<|im_start|>user\n"
        "PUBLIC MAZE\n"
        + _public_maze_only(problem)
        + "\n\nPUBLIC STATE\n"
        + f"position_xy={position}\n"
        + f"goal_xy={goal}\n"
        + "Coordinates use +x east/right and +y north/up.\n"
        + "OPTIONS\n"
        + options
        + "\nChoose the next force pulse."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def render_point_policy_prompt_v3(
    problem: str,
    observation: dict[str, Any],
    actions: Sequence[str],
    _history: Sequence[str],
) -> str:
    """Render the Markov public-state v3 prompt including point velocity."""

    action_tuple = tuple(str(action) for action in actions)
    if len(action_tuple) != len(POINT_POLICY_LABELS):
        raise ValueError("PointMaze alphabet differs from the label contract")
    options = "\n".join(
        f"{POINT_POLICY_LABELS[index]}: {action}"
        for index, action in enumerate(action_tuple)
    )
    position = _compact_coordinate(observation["achieved_goal"])
    velocity = _compact_coordinate(observation["velocity_xy"])
    goal = _compact_coordinate(observation["desired_goal"])
    return (
        "<|im_start|>system\n"
        "Control a PointMaze point mass one force pulse at a time. "
        "Return exactly one capital option letter."
        "<|im_end|>\n<|im_start|>user\n"
        "PUBLIC MAZE\n"
        + _public_maze_only(problem)
        + "\n\nPUBLIC MARKOV STATE\n"
        + f"position_xy={position}\n"
        + f"velocity_xy={velocity}\n"
        + f"goal_xy={goal}\n"
        + "Coordinates use +x east/right and +y north/up.\n"
        + "OPTIONS\n"
        + options
        + "\nChoose the next force pulse."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
