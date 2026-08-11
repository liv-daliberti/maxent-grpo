from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))


def test_v19_protocol_discloses_v18_termination_confound_and_fresh_gate():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "ant_continuing_waypoint_controller_v19_20260804.md"
    ).read_text(encoding="utf-8")
    assert "FROZEN BEFORE V19 OPTIMIZATION OR V19 GATE EXECUTION" in protocol
    assert "defines `terminated` solely" in protocol
    assert "does not initialize from failed v17 or v18 weights" in protocol
    assert "four new 21x21" in protocol
    assert "zero maze-task termination events" in protocol
    assert "It does not by" in protocol
    assert "itself authorize a language-model cohort" in protocol


def test_v19_curriculum_and_gate_are_disjoint_and_goal_safe():
    pytest.importorskip("gymnasium")
    import train_ant_continuing_waypoint_controller_v19 as v19

    assert len(v19.TRAINING_PATTERNS) == 512
    assert len(v19.EVALUATION_PATTERNS) == 24
    assert len(set(v19.EVALUATION_PATTERNS)) == 24
    assert not set(v19.TRAINING_PATTERNS) & set(v19.EVALUATION_PATTERNS)
    assert not set(v19.v18.EVALUATION_PATTERNS) & set(
        v19.EVALUATION_PATTERNS
    )
    assert not set(v19.v18.v17.EVALUATION_PATTERNS) & set(
        v19.EVALUATION_PATTERNS
    )
    assert all(len(pattern) <= 4 for pattern in v19.TRAINING_PATTERNS)
    assert all(len(pattern) == 8 for pattern in v19.EVALUATION_PATTERNS)
    assert len(v19.TRAIN_MAPS) == 4
    assert len(v19.DEVELOPMENT_MAPS) == 4
    assert all(len(maze) == 17 for maze in v19.TRAIN_MAPS)
    assert all(len(maze) == 21 for maze in v19.DEVELOPMENT_MAPS)
    for pattern in v19.TRAINING_PATTERNS:
        row, column = v19._goal_cell((8, 8), pattern)
        assert 0 < row < 16 and 0 < column < 16
    for pattern in v19.EVALUATION_PATTERNS:
        row, column = v19._goal_cell((10, 10), pattern)
        assert 0 < row < 20 and 0 < column < 20


def test_v19_uses_continuing_task_and_explicit_inner_ant_health():
    trainer = (
        ROOT / "ops/train_ant_continuing_waypoint_controller_v19.py"
    ).read_text(encoding="utf-8")
    assert 'continuing_task=True' in trainer
    assert "self.env.unwrapped.continuing_task = True" in trainer
    assert "env.unwrapped.ant_env.is_healthy" in trainer
    assert '"maze_task_termination_count"' in trainer
    assert '"maze_task_never_terminated_controller_gate"' in trainer
    assert "ant_waypoint_v11.zip" in trainer
    assert "ant_sequential_waypoint_v17.zip" not in trainer


def test_v19_launcher_holds_binds_and_forbids_language_sampling():
    launcher = (
        ROOT
        / "ops/exp_scaling/"
        "launch_ant_continuing_waypoint_controller_v19.py"
    ).read_text(encoding="utf-8")
    assert '"--hold"' in launcher
    assert '"language_model_sampled": False' in launcher
    assert '"continuing_task": True' in launcher
    assert '"explicit_ant_health": True' in launcher
    assert "EXPECTED_V18_RECEIPT" in launcher
    assert "EXPECTED_SIMULATOR_SOURCE" in launcher
    assert "EXPECTED_TERMINATION_SOURCE" in launcher
    assert 'run(["scontrol", "release", str(job_id)])' in launcher


def test_pinned_antmaze_source_explains_v18_termination_confound():
    simulator_source = (
        ROOT
        / "var/maze_runtime/venv/lib/python3.11/site-packages/"
        "gymnasium_robotics/envs/maze/ant_maze_v5.py"
    ).read_text(encoding="utf-8")
    termination_source = (
        ROOT
        / "var/maze_runtime/venv/lib/python3.11/site-packages/"
        "gymnasium_robotics/envs/maze/maze_v4.py"
    ).read_text(encoding="utf-8")
    assert (
        "ant_obs, _, _, _, info = self.ant_env.step(action)"
        in simulator_source
    )
    assert "terminated = self.compute_terminated" in simulator_source
    assert "if not self.continuing_task:" in termination_source
