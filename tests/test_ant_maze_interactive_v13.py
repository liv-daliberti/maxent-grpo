from __future__ import annotations

import json
from pathlib import Path

import pytest

from oat_drgrpo.ant_maze_interactive_policy import (
    ANT_POLICY_ACTIONS,
    ANT_TERMINAL_PADDING_PROMPT,
    render_ant_policy_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


def observation():
    return {
        "achieved_goal": [0.0, -4.0],
        "desired_goal": [0.0, 4.0],
        "velocity_xy": [0.25, -0.125],
        "remaining_actions": 12,
    }


def test_prompt_is_public_markov_one_token_interface() -> None:
    problem = (
        "Navigate the Ant from S to G in the frozen maze.\n"
        "The map rows are:\n###\n#.#\n###\n"
        "S=[1, 1]; G=[1, 2].\nUse only these commands: N, NE, E, SE, S, SW, W, NW.\n"
        "Return 4 to 16 commands inside \\boxed{}, separated by spaces. Do not explain."
    )
    prompt = render_ant_policy_prompt(problem, observation(), ANT_POLICY_ACTIONS, [])
    assert "PUBLIC MARKOV STATE" in prompt
    assert "position_xy=(0.000,-4.000)" in prompt
    assert "velocity_xy=(0.250,-0.125)" in prompt
    assert "OPTIONS: N NE E SE S SW W NW" in prompt
    assert "Return 4 to 16" not in prompt
    for forbidden in ("canonical_key", "route_key", "reward", "answer"):
        assert forbidden not in prompt.lower()


def test_action_alphabet_and_padding_are_exact() -> None:
    assert ANT_POLICY_ACTIONS == ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    assert all(action in ANT_TERMINAL_PADDING_PROMPT for action in ANT_POLICY_ACTIONS)
    with pytest.raises(ValueError, match="alphabet"):
        render_ant_policy_prompt("map", observation(), ANT_POLICY_ACTIONS[:-1], [])


def test_materializer_is_train_only_and_fail_closed() -> None:
    source = (ROOT / "ops/materialize_ant_maze_train_warmstart_v13.py").read_text()
    for required in (
        'if set(dataset) != {"train"}',
        'if len(rows) != 4',
        'if record.get("split") == "train"',
        '"dev_dataset_loaded": False',
        '"eval_dataset_loaded": False',
        '"model_sampled": False',
        "canonical key changed",
    ):
        assert required in source
