from __future__ import annotations

import importlib.util
from pathlib import Path

from oat_drgrpo.point_maze_waypoint_policy import (
    convert_point_waypoint_prompt,
    render_point_waypoint_prompt,
    render_point_waypoint_terminal_padding_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = (
    ROOT / "ops/exp_scaling/launch_e79pm_falcon_point_maze_verified_replay.py"
)


def _launcher():
    spec = importlib.util.spec_from_file_location("e79pm_launcher", LAUNCHER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation() -> dict[str, object]:
    return {
        "allowed_actions": ["N", "E", "S"],
        "current_cell": [4, 1],
        "previous_cell": None,
        "goal_cell": [3, 7],
        "achieved_goal": [-3.0, 0.0],
        "velocity_xy": [0.0, 0.0],
        "remaining_actions": 28,
    }


def test_falcon_surface_preserves_semantic_prompt() -> None:
    qwen = render_point_waypoint_prompt("PUBLIC GRID", _observation())
    falcon = render_point_waypoint_prompt(
        "PUBLIC GRID", _observation(), prompt_format="falcon3"
    )
    converted = convert_point_waypoint_prompt(qwen, prompt_format="falcon3")
    assert falcon == converted
    assert falcon.startswith("<|system|>\n")
    assert falcon.endswith("<|assistant|>\n")
    assert "<|im_start|>" not in falcon
    for text in ("PUBLIC GRID", "current_cell=(4,1)", "A: N", "B: E", "C: S"):
        assert text in qwen and text in falcon


def test_falcon_terminal_padding_uses_same_surface() -> None:
    prompt = render_point_waypoint_terminal_padding_prompt(prompt_format="falcon3")
    assert prompt.startswith("<|system|>\n")
    assert prompt.endswith("<|assistant|>\n")
    assert "Global labels: A B C D." in prompt


def test_launcher_freezes_paired_eight_pass_design() -> None:
    launcher = _launcher()
    assert launcher.ARMS == ("control", "replay")
    assert launcher.SEEDS == (55, 56, 57, 58, 59)
    assert launcher.PASSES == 8
    assert launcher.TRAIN_ROWS == 384
    assert launcher.DEV_ROWS == 64
    assert launcher.EVAL_ROWS == 128
    assert launcher.CHECKPOINT_INTERVAL == 192
    assert launcher.TARGET_STEPS == 3072
    assert launcher.REPLAY_WEIGHT == 0.10
    assert launcher.MODEL_REVISION == "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"


def test_slurm_contract_has_native_surface_and_no_extra_controller() -> None:
    sft = (ROOT / "ops/slurm/e79pm_falcon_point_maze_sft.slurm").read_text()
    train = (ROOT / "ops/slurm/e79pm_falcon_point_maze_train.slurm").read_text()
    assert "--max-updates 72" in sft
    assert "--prompt-format falcon3" in sft
    assert "--passes 8" in train
    assert "--evaluation-interval 192" in train
    assert "--evaluation-prompts 128" in train
    assert "--prompt-format falcon3" in train
    assert "--replay-weight 0.10" in train
    for forbidden in ("semantic", "maxent", "balance", "adaptive"):
        assert forbidden not in train.lower()


def test_online_runner_uses_aligned_falcon_optimizer() -> None:
    runner = (ROOT / "ops/train_point_maze_verified_replay_only.py").read_text()
    assert "betas=(0.9, 0.999)" in runner
    assert "eps=1e-8" in runner
    assert "weight_decay=0.0" in runner
    assert 'choices=("e78pm", "e79pm")' in runner
