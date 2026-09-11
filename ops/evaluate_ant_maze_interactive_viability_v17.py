#!/usr/bin/env python3
"""Evaluate the frozen 0.5B closed-loop policy on Ant v15/v17 dev maps."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import evaluate_ant_maze_interactive_viability_v13 as v13
from oat_drgrpo.ant_maze_interactive_policy import render_ant_policy_prompt
from oat_drgrpo.ant_maze_interactive_process_v17_r1 import (
    AntMazeInteractiveProcessV17R1,
)


def _render(problem, observation, actions, history):
    public_map = str(problem).split("\nReturn ", 1)[0].strip()
    return render_ant_policy_prompt(
        public_map, observation, actions, history
    )


def _output_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("Ant v17 evaluator requires --output") from error


def main() -> None:
    v13.AntMazeInteractiveProcess = AntMazeInteractiveProcessV17R1
    v13.render_ant_policy_prompt = _render
    output = _output_path()
    v13.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["schema_version"] = "ant-maze-interactive-viability-v17"
    payload["decision"] = (
        "eligible_for_ant_v17_paired_online_smoke"
        if payload.get("status") == "pass"
        else "ant_maze_v17_stage_b_stopped"
    )
    payload["sampling"]["policy_interface"] = (
        "closed_loop_public_markov_compass_token_v17"
    )
    payload["information_boundary"].update(
        transferred_v13_train_only_warmstart=True,
        v15_development_only=True,
        v15_evaluation_prompts_loaded=False,
        v15_certified_routes_in_context=False,
    )
    v13.base._atomic_json(output, payload)
    print(
        "[ant-v17-interactive] "
        f"status={payload['status']} "
        f"verified={payload['summary']['verified_completions']} "
        f"prefix_success={payload['summary']['prefix_success_prompts']} "
        f"multimode={payload['summary']['multimode_prompts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
