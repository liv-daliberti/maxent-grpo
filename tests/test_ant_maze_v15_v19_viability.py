from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v19_viability_is_fresh_development_only_and_pass_gated():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "ant_maze_v15_controller_v19_05b_viability_20260804.md"
    ).read_text(encoding="utf-8")
    batch = (
        ROOT
        / "ops/slurm/evaluate_ant_maze_v15_controller_v19_viability.slurm"
    ).read_text(encoding="utf-8")
    assert "BEFORE THE V19 CONTROLLER OUTCOME" in protocol
    assert "fresh seed" in protocol and "76702" in protocol
    assert "four v15 development rows" in protocol
    assert "64 trajectories per prompt" in protocol
    assert "inclusive interval [0.02, 0.50]" in protocol
    assert "authorizes only a" in protocol
    assert "separately frozen five-seed online comparison" in protocol
    assert "ant_maze_modebench_v15_controller_v19/dev" in batch
    assert "--seed 76702" in batch
    assert "--sample-count 64 --prefix-count 16" in batch


def test_v19_evaluator_uses_v19_interactive_process_and_schema():
    evaluator = (
        ROOT / "ops/evaluate_ant_maze_interactive_viability_v19.py"
    ).read_text(encoding="utf-8")
    process = (
        ROOT / "src/oat_drgrpo/ant_maze_interactive_process_v19.py"
    ).read_text(encoding="utf-8")
    worker = (
        ROOT / "src/oat_drgrpo/ant_maze_interactive_worker_v19.py"
    ).read_text(encoding="utf-8")
    assert "AntMazeInteractiveProcessV19" in evaluator
    assert "ant-maze-interactive-viability-v19" in evaluator
    assert "continuing_task_controller_gate=True" in evaluator
    assert "ant_maze_interactive_worker_v19" in process
    assert "ant_maze_worker_v19" in worker


def test_v19_qualifier_and_launcher_bind_fresh_contract():
    qualifier = (
        ROOT / "ops/qualify_ant_maze_v15_controller_v19_viability.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT
        / "ops/exp_scaling/"
        "launch_ant_maze_v15_controller_v19_viability.py"
    ).read_text(encoding="utf-8")
    assert "admitted_to_ant_v15_v19_frozen_model_viability_gate" in qualifier
    assert "ant-maze-interactive-viability-v19" in qualifier
    assert "eligible_for_ant_v15_v19_five_seed_pair" in qualifier
    assert '"seed": 76702' in launcher
    assert '"controller_outcome_loaded_before_freeze": False' in launcher
    assert '"admission_outcome_loaded_before_freeze": False' in launcher
