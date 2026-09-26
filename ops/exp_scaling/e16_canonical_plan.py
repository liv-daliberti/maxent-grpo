#!/usr/bin/env python3
"""Single machine-readable source of truth for E16 campaign configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


GRAPH_TARGET = 2.7974740052946436
COUNTDOWN_TARGET = 3.9741470618167156
ARMS = ("maxent", "maxent_control", "maxent_dual")
SMOKE_SEEDS = (9006,)
FULL_SEEDS = (43, 44, 45)
EXPECTED_SOURCE_HASH = "0e9929f09bac51ddcd85a6d5a506bf0613279a6fd983e0147da2f39a7aa320ec"
EXPECTED_DATASET_IDENTITY_SHA256 = (
    "efc23631bd96c631e04e5a8161533cfdfc88e8b737f4e77d9d78a547b66fefce"
)
EXPECTED_RUNTIME_IDENTITY_SHA256 = (
    "40bafc191219c2612ad6b74aa07aa878eef06644323aec5cfb6a6dd18d7a3e66"
)
EXPECTED_CONFIG_DIGESTS = {
    "smoke": "7c91976613e1f23ff965256c7ff4abd49ce73b7f1f32122b6f9e60820a72b761",
    "full": "10f9183ae4155b28e46addafb49f38800ab6fd39c3f054a81a828326c64d43aa",
}


def _budget(updates: int) -> int:
    return (int(updates) - 1) * 16


def campaign_plan(stage: str) -> dict[str, Any]:
    if stage not in {"smoke", "full"}:
        raise ValueError("stage must be smoke or full")
    smoke = stage == "smoke"
    seeds = SMOKE_SEEDS if smoke else FULL_SEEDS
    tasks = {
        "graph_coloring": {
            "canonical_action_task": "graph_coloring",
            "canonical_graph_actions": True,
            "data_rows": {"eval": 96, "train": 192},
            "eval_prompt_interval": 32 if smoke else 48,
            "max_action_entropy": math.log(27),
            "prefix": (
                "gce16_canonical_maxent_joint_smoke_v3"
                if smoke
                else "gce16_canonical_maxent_05b_v2"
            ),
            "prompt_template": "qwen_graph_digits",
            "sequence_count": 27,
            "target_entropy": GRAPH_TARGET,
            "target_optimizer_updates": 32 if smoke else 960,
        },
        "countdown": {
            "canonical_action_task": "countdown",
            "canonical_graph_actions": False,
            "data_rows": {"eval": 128, "train": 384},
            "eval_prompt_interval": 32 if smoke else 96,
            "max_action_entropy": math.log(108),
            "prefix": (
                "cde16_canonical_maxent_joint_smoke_v3"
                if smoke
                else "cde16_canonical_maxent_05b_v2"
            ),
            "prompt_template": "qwen_countdown_digits",
            "sequence_count": 108,
            "target_entropy": COUNTDOWN_TARGET,
            "target_optimizer_updates": 32 if smoke else 1920,
        },
    }
    for task in tasks.values():
        task["trajectory_query_budget"] = _budget(
            int(task["target_optimizer_updates"])
        )
    plan = {
        "arms": {
            "maxent": {
                "alpha": 0.10,
                "controller": "none",
            },
            "maxent_control": {
                "alpha": 0.075,
                "controller": "proportional",
                "ema_decay": 0.9,
                "gain": 4.0,
                "max_alpha": 0.10,
                "target_ratio_interface": 1.0,
                "warmup_steps_interface": 1,
            },
            "maxent_dual": {
                "adam_betas": [0.9, 0.999],
                "alpha": 0.075,
                "alpha_lr": 0.005,
                "controller": "haarnoja_log_alpha_adam",
                "max_alpha": 0.10,
                "min_alpha": 0.05,
                "target_ratio_interface": 1.0,
                "warmup_steps_interface": 1,
            },
        },
        "common": {
            "auto_resume": False,
            "beta": 0.0,
            "canonical_action_count": 3,
            "canonical_fixed_shape_sampling": True,
            "canonical_learner_sampling": True,
            "eval_mode_coverage_k": 8,
            "generate_max_length": 192,
            "group_size": 16,
            "learning_rate": 2e-7,
            "maxent_length_target": 0.0,
            "model": "qwen2.5-0.5b-instruct",
            "num_ppo_epochs": 1,
            "prompt_epochs": 1 if smoke else 5,
            "temperature": 1.0,
            "top_p": 1.0,
            "watchdog_requeue": False,
        },
        "protocol": "E16",
        "schema": "e16_canonical_campaign_plan_v1",
        "seeds": list(seeds),
        "stage": stage,
        "tasks": tasks,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    if tuple(plan["arms"]) != ARMS:
        raise ValueError("E16 plan must contain exactly the three MaxEnt arms")
    common = plan["common"]
    if common["maxent_length_target"] != 0 or common["prompt_epochs"] not in {1, 5}:
        raise ValueError("E16 cannot enable length control or exceed five epochs")
    if common["group_size"] != 16 or common["num_ppo_epochs"] != 1:
        raise ValueError("E16 group/PPO settings drifted")
    for task_name, task in plan["tasks"].items():
        if not 0 < float(task["target_entropy"]) <= float(task["max_action_entropy"]):
            raise ValueError(f"{task_name} entropy target is outside finite support")
        expected_budget = _budget(int(task["target_optimizer_updates"]))
        if task["trajectory_query_budget"] != expected_budget:
            raise ValueError(f"{task_name} trajectory budget drifted")
    if plan["arms"]["maxent"]["alpha"] != 0.10:
        raise ValueError("fixed E16 alpha must be E15 M10")
    for adaptive in ("maxent_control", "maxent_dual"):
        arm = plan["arms"][adaptive]
        if arm["alpha"] != 0.075 or arm["max_alpha"] != 0.10:
            raise ValueError(f"{adaptive} bounds/base drifted")


def plan_digest(plan: dict[str, Any]) -> str:
    encoded = json.dumps(
        plan, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plan = campaign_plan(args.stage)
    digest = plan_digest(plan)
    if digest != EXPECTED_CONFIG_DIGESTS[args.stage]:
        raise SystemExit(
            "E16 plan drifted from its prospective config digest: "
            f"stage={args.stage} expected={EXPECTED_CONFIG_DIGESTS[args.stage]} "
            f"observed={digest}"
        )
    payload = {
        "config_digest": digest,
        "frozen_identity": {
            "config_digest": EXPECTED_CONFIG_DIGESTS[args.stage],
            "dataset_identity_sha256": EXPECTED_DATASET_IDENTITY_SHA256,
            "runtime_identity_sha256": EXPECTED_RUNTIME_IDENTITY_SHA256,
            "source_hash": EXPECTED_SOURCE_HASH,
        },
        "plan": plan,
    }
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
