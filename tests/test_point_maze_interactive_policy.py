import importlib.util
from pathlib import Path

from oat_drgrpo.point_maze_interactive_policy import (
    POINT_POLICY_LABELS,
    POINT_TERMINAL_PADDING_PROMPT,
    render_point_policy_prompt,
    render_point_policy_prompt_v2,
    render_point_policy_prompt_v3,
)


def test_terminal_padding_prompt_is_public_fixed_and_zero_decision_labeled():
    assert "terminal padding" in POINT_TERMINAL_PADDING_PROMPT
    assert "zero decision mask" in POINT_TERMINAL_PADDING_PROMPT
    assert "A B C D E F G H I" in POINT_TERMINAL_PADDING_PROMPT


def test_point_policy_prompt_exposes_public_state_and_all_actions():
    actions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "COAST")
    observation = {
        "achieved_goal": [1.0, 2.0],
        "desired_goal": [3.0, 4.0],
        "remaining_actions": 17,
    }
    prompt = render_point_policy_prompt(
        "printed public map",
        observation,
        actions,
        ["N", "E"],
    )
    assert len(POINT_POLICY_LABELS) == len(actions)
    assert "position_xy=[1.0, 2.0]" in prompt
    assert "goal_xy=[3.0, 4.0]" in prompt
    assert "recent_actions=N E" in prompt
    for label, action in zip(POINT_POLICY_LABELS, actions):
        assert f"{label}: {action}" in prompt


def test_shared_prompt_renderer_matches_frozen_viability_renderer():
    source = (
        Path(__file__).resolve().parents[1]
        / "ops/evaluate_point_maze_interactive_viability.py"
    )
    spec = importlib.util.spec_from_file_location("point_viability", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "COAST")
    observation = {
        "achieved_goal": [0.25, -0.5],
        "desired_goal": [2.0, 3.0],
        "remaining_actions": 41,
    }
    expected = module._prompt("map", observation, actions, ["S", "NE"])
    observed = render_point_policy_prompt(
        "map", observation, actions, ["S", "NE"]
    )
    assert observed == expected


def test_compact_v2_prompt_removes_program_and_history_shortcuts():
    actions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "COAST")
    observation = {
        "achieved_goal": [-0.0001, 2.1236],
        "desired_goal": [3.0, 4.0],
        "remaining_actions": 17,
    }
    problem = "Navigate:\n###\n#G#\n###\n\nOutput only a whitespace-separated action program inside \\boxed{}."
    prompt = render_point_policy_prompt_v2(
        problem, observation, actions, ["N", "E", "E"]
    )
    assert "position_xy=(0.000,2.124)" in prompt
    assert "goal_xy=(3.000,4.000)" in prompt
    assert "Output only a whitespace-separated" not in prompt
    assert "recent_actions" not in prompt
    assert "remaining_actions" not in prompt
    for label, action in zip(POINT_POLICY_LABELS, actions):
        assert f"{label}: {action}" in prompt


def test_velocity_v3_prompt_exposes_public_markov_state():
    actions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "COAST")
    observation = {
        "achieved_goal": [1.0, -2.0],
        "velocity_xy": [0.1254, -0.75],
        "desired_goal": [3.0, 4.0],
        "remaining_actions": 17,
    }
    prompt = render_point_policy_prompt_v3(
        "Maze:\n###\n#G#\n###\n\nOutput only a whitespace-separated action program",
        observation,
        actions,
        ["N", "E"],
    )
    assert "PUBLIC MARKOV STATE" in prompt
    assert "position_xy=(1.000,-2.000)" in prompt
    assert "velocity_xy=(0.125,-0.750)" in prompt
    assert "goal_xy=(3.000,4.000)" in prompt
    assert "recent_actions" not in prompt
    assert "Output only a whitespace-separated" not in prompt
