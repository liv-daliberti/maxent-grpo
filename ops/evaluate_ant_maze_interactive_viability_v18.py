#!/usr/bin/env python3
"""Run the frozen AntMaze v15/v18 stable-handoff model viability gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import evaluate_point_maze_interactive_viability as base

from oat_drgrpo.ant_maze_interactive_policy import (
    ANT_POLICY_ACTIONS,
    render_ant_policy_prompt,
)
from oat_drgrpo.ant_maze_interactive_process_v18 import (
    AntMazeInteractiveProcessV18,
)
from oat_drgrpo.maze_modebench import (
    ANT_MAZE_VERIFIER,
    parse_maze_action_spec,
)


def _spec(row):
    value = row["answer"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("AntMaze v18 answer must contain a JSON specification")
    parsed = parse_maze_action_spec(value)
    if parsed.verifier != ANT_MAZE_VERIFIER:
        raise ValueError("v18 viability received a non-AntMaze row")
    if tuple(parsed.action_tokens) != ANT_POLICY_ACTIONS:
        raise ValueError("v18 AntMaze action support changed")
    return value


def _output_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("AntMaze v18 evaluator requires --output") from error


def main() -> None:
    base.LABELS = ANT_POLICY_ACTIONS
    base._spec = _spec
    base.PointMazeInteractiveProcess = AntMazeInteractiveProcessV18
    base.render_point_policy_prompt_v3 = render_ant_policy_prompt
    output = _output_path()
    base.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["schema_version"] = "ant-maze-interactive-viability-v18"
    payload["domain"] = "ant_maze"
    payload["decision"] = (
        "eligible_for_ant_v18_stable_handoff_qualification"
        if payload.get("status") == "pass"
        else "ant_maze_v18_model_viability_stopped"
    )
    payload["sampling"]["policy_interface"] = (
        "closed_loop_public_markov_compass_token_v18"
    )
    payload["sampling"]["action_repeat"] = 400
    payload["information_boundary"].update(
        constrained_language_action_interface=True,
        frozen_low_level_controller=True,
        stable_waypoint_handoff=True,
        controller_feedback_in_prompt=False,
    )
    base._atomic_json(output, payload)


if __name__ == "__main__":
    main()
