from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper/preregistration/ant_maze_interactive_warmstart_v13_20260730.md"
TRAINER = ROOT / "ops/train_ant_maze_interactive_warmstart_v13.py"
EVALUATOR = ROOT / "ops/evaluate_ant_maze_interactive_viability_v13.py"
BATCH = ROOT / "ops/slurm/train_ant_maze_interactive_warmstart_v13.slurm"


def test_protocol_freezes_separate_05b_interface_and_gate() -> None:
    text = PROTOCOL.read_text()
    for required in (
        "v12 free-form viability remains failed at 0/256",
        "32 v13-r1 examples",
        "64 epochs",
        "128 optimizer updates",
        "seed 75313",
        "base seed 107313",
        "at least two of four maps",
        "both canonical route\nkeys",
        "task-specific 0.5B initialization\nstratum",
    ):
        assert required in text


def test_trainer_and_batch_bind_exact_schedule_and_firewall() -> None:
    trainer = TRAINER.read_text(); batch = BATCH.read_text()
    for required in (
        'identity.get("example_count") != 32',
        'identity.get("episode_count") != 8',
        'total_steps != 128',
        '"dev_dataset_loaded": False',
        '"eval_dataset_loaded": False',
        "restricted_action_cross_entropy",
    ):
        assert required in trainer
    for required in (
        "OMP_NUM_THREADS=1",
        "--epochs 64",
        "--sample-count 64",
        "--prefix-count 16",
        "--seed 107313",
        "ant_maze_modebench_v12/dev",
    ):
        assert required in batch
    assert "ant_maze_modebench_v12/eval" not in batch


def test_evaluator_relabels_generic_engine_as_ant_v13() -> None:
    source = EVALUATOR.read_text()
    for required in (
        'base.LABELS = ANT_POLICY_ACTIONS',
        'base.PointMazeInteractiveProcess = AntMazeInteractiveProcess',
        '"ant-maze-interactive-viability-v13"',
        '"eligible_for_ant_v13_paired_online_smoke"',
        '"ant_maze_v13_stage_b_stopped"',
        'payload["sampling"]["action_repeat"] = 400',
    ):
        assert required in source
