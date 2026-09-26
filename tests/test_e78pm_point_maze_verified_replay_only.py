from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from oat_drgrpo.interactive_episode_replay import (
    InteractiveDecisionRecord,
    InteractiveEpisodeRecord,
    InteractiveReplayGroup,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ops/train_point_maze_verified_replay_only.py"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e78pm_point_maze_verified_replay_only_05b.py"
PROTOCOL = ROOT / "paper/preregistration/e78pm_point_maze_verified_replay_only_05b_20260804.md"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode(key: str, actions: tuple[int, ...]) -> InteractiveEpisodeRecord:
    decisions = tuple(
        InteractiveDecisionRecord(
            prompt_token_ids=(10, index),
            allowed_token_ids=(1, 2, 3, 4),
            selected_token_id=action,
            behavior_logprobs=(-math.log(4),) * 4,
            transition_sha256=(str(index % 10) * 64),
        )
        for index, action in enumerate(actions)
    )
    return InteractiveEpisodeRecord(
        group_prompt_token_ids=(7, 8),
        outcome_key=key,
        task_reward=1.0,
        decisions=decisions,
    )


def test_e78pm_is_five_paired_seeds_eight_passes_and_half_passes():
    launch = load(LAUNCHER, "e78pm_launcher")
    assert launch.ARMS == ("control", "replay")
    assert launch.SEEDS == (43, 44, 45, 46, 47)
    assert launch.PASSES == 8
    assert launch.TRAIN_ROWS == 384
    assert launch.DEV_ROWS == 64
    assert launch.EVAL_ROWS == 128
    assert launch.TARGET_STEPS == 3072
    assert launch.CHECKPOINT_INTERVAL == 192
    assert launch.DATA_SEED == 88104


def test_uniform_replay_matches_e78_per_rollout_scale_and_control_is_zero():
    runner = load(RUNNER, "e78pm_runner")
    group = InteractiveReplayGroup(
        group_prompt_token_ids=(7, 8),
        outcome_keys=("north", "south"),
        episodes=(episode("north", (1, 2)), episode("south", (3, 4, 1, 2))),
    )
    replay_slots, replay = runner.uniform_verified_replay_slots(
        group=group,
        padding_token_ids=(9,),
        action_token_ids=(1, 2, 3, 4),
        compute_only=False,
        replay_weight=0.10,
    )
    control_slots, control = runner.uniform_verified_replay_slots(
        group=group,
        padding_token_ids=(9,),
        action_token_ids=(1, 2, 3, 4),
        compute_only=True,
        replay_weight=0.10,
    )
    scale = (15 / 16) / 16
    active_replay = [slot for slot in replay_slots if slot["active"]]
    assert math.isclose(sum(slot["weight"] for slot in active_replay[:2]), -0.05 * scale)
    assert math.isclose(sum(slot["weight"] for slot in active_replay[2:]), -0.05 * scale)
    assert all(slot["weight"] == 0.0 for slot in control_slots)
    assert replay["replay_applied_score_gradient_l2"] > 0
    assert control["replay_applied_score_gradient_l2"] == 0
    assert replay["replay_balance_eligible_groups"] == 0


def test_protocol_is_separate_fresh_and_replay_only():
    text = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    for literal in (
        "separately identified sixth-domain extension",
        "not a retroactive change",
        "Data seed: `88104`",
        "exactly eight ordered passes",
        "every half pass",
        "There is no outcome-dependent viability gate",
        "verified replay is the only auxiliary derivative",
        "does not emit MuJoCo forces",
        "`PointMaze_UMaze-v3`",
        "384 train, 64 development, and 128 evaluation maps",
        "3,072 optimizer updates",
    ):
        assert literal in text
    assert "PointMaze_UMaze-v5" not in text


def test_slurm_contract_uses_fresh_data_and_replay_only_runner():
    prepare = (ROOT / "ops/slurm/e78pm_point_maze_prepare.slurm").read_text()
    train = (ROOT / "ops/slurm/e78pm_point_maze_train.slurm").read_text()
    assert "--seed 88104" in prepare
    assert prepare.count("--exclude-identity") == 4
    assert "train_point_maze_verified_replay_only.py" in train
    assert "--train-count 384 --dev-count 64 --eval-count 128" in prepare
    assert "--passes 8 --evaluation-interval 192 --evaluation-prompts 128" in train
    assert "--replay-weight 0.10" in train
    assert "--auto-resume" in train
