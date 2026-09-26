#!/usr/bin/env python3
"""Continue the maze-blind Ant controller from v7 on fresh seeds."""

from __future__ import annotations

from pathlib import Path

import train_ant_waypoint_controller_v7 as controller


ROOT = controller.ROOT
controller.CONTROLLER_VERSION = "v8"
controller.DEFAULT_OUTPUT = ROOT / "var/maze_runtime/controllers/ant_waypoint_v8"
controller.INITIAL_MODEL = ROOT / "var/maze_runtime/controllers/ant_waypoint_v7.zip"
controller.INITIAL_MODEL_SHA256 = (
    "2bb5d192698639a294ac8266ae9be68484c9d493f4f7ba4317ee06d2d4b04442"
)
controller.DEFAULT_TIMESTEPS = 3_000_000
controller.DEFAULT_SEED = 73008
controller.DEFAULT_LEARNING_RATE = 5e-6
controller.EVALUATION_SEED_OFFSET = 3_000_000
controller.INFORMATION_FIREWALL = (
    "Open-plane Ant-v5 only; no maze map, route-gate observation, language "
    "prompt, MaxEnt outcome, Dr.GRPO outcome, or maze trajectory loaded. "
    "Only the preregistered aggregate v7 receipt and exact v7 model are "
    "antecedents; v7 evaluation episodes are not training inputs."
)
controller.INITIALIZATION_DESCRIPTION = (
    "Exact failed-v7 waypoint weights with a reset optimizer; all rewards, "
    "four-unit targets, cyclic eight-heading balance, observation coordinates, "
    "success radius, and episode horizon are unchanged."
)


if __name__ == "__main__":
    controller.main()
