#!/usr/bin/env python3
"""Run the frozen AntMaze v15/v19 model viability gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import evaluate_point_maze_interactive_viability as base

from oat_drgrpo.ant_maze_interactive_policy import (
    ANT_POLICY_ACTIONS,
    render_ant_policy_prompt,
)
from oat_drgrpo.ant_maze_interactive_process_v19 import (
    AntMazeInteractiveProcessV19,
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
        raise ValueError("AntMaze v19 answer must contain a JSON specification")
    parsed = parse_maze_action_spec(value)
    if parsed.verifier != ANT_MAZE_VERIFIER:
        raise ValueError("v19 viability received a non-AntMaze row")
    if tuple(parsed.action_tokens) != ANT_POLICY_ACTIONS:
        raise ValueError("v19 AntMaze action support changed")
    return value


def _output_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("AntMaze v19 evaluator requires --output") from error


def main() -> None:
    base.LABELS = ANT_POLICY_ACTIONS
    base._spec = _spec
    base.PointMazeInteractiveProcess = AntMazeInteractiveProcessV19
    base.render_point_policy_prompt_v3 = render_ant_policy_prompt
    output = _output_path()
    base.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["schema_version"] = "ant-maze-interactive-viability-v19"
    payload["domain"] = "ant_maze"
    payload["decision"] = (
        "eligible_for_ant_v19_continuing_handoff_qualification"
        if payload.get("status") == "pass"
        else "ant_maze_v19_model_viability_stopped"
    )
    payload["sampling"]["policy_interface"] = (
        "closed_loop_public_markov_compass_token_v19"
    )
    payload["sampling"]["action_repeat"] = 400
    payload["information_boundary"].update(
        constrained_language_action_interface=True,
        frozen_low_level_controller=True,
        stable_waypoint_handoff=True,
        continuing_task_controller_gate=True,
        explicit_ant_health_controller_gate=True,
        controller_feedback_in_prompt=False,
    )
    base._atomic_json(output, payload)


if __name__ == "__main__":
    main()
