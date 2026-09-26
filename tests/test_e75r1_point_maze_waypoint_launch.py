from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e75r1_point_maze_waypoint_weak_sft_05b.py"
PROTOCOL = (
    ROOT
    / "paper/preregistration/e75r1_point_maze_waypoint_weak_sft_05b_20260804.md"
)


def test_e75r1_is_a_separate_prospective_successor():
    launcher = LAUNCHER.read_text()
    protocol = PROTOCOL.read_text()
    normalized_protocol = " ".join(protocol.split())
    assert '"experiment": "E75R1"' in launcher
    assert '"data_seed": 88101' in launcher
    assert '"sft_optimizer_updates": 32' in launcher
    assert '"evaluation_job_submitted": False' in launcher
    assert "E75 remains immutable" in normalized_protocol
    assert "There is no adaptive checkpoint search" in normalized_protocol
    assert "A failed gate is the terminal E75R1 result" in normalized_protocol


def test_e75r1_stages_use_disjoint_paths_and_fixed_budget():
    prepare = (ROOT / "ops/slurm/e75r1_point_maze_waypoint_prepare.slurm").read_text()
    sft = (ROOT / "ops/slurm/e75r1_point_maze_waypoint_sft.slurm").read_text()
    dev = (ROOT / "ops/slurm/e75r1_point_maze_waypoint_dev.slurm").read_text()
    train = (ROOT / "ops/slurm/e75r1_point_maze_waypoint_train.slurm").read_text()
    assert "--seed 88101" in prepare
    assert "--exclude-identity" in prepare
    assert "point_maze_waypoint_pilot_e75r1" in prepare
    assert "--max-updates 32" in sft
    assert "--seed 88402" in sft
    assert "point_maze_waypoint_warmstart_e75r1" in sft
    assert "--seed 88502" in dev
    assert "--seed 88502" in train
    assert "e75r1_point_maze_waypoint" in dev
    assert "e75r1_point_maze_waypoint" in train


def test_e75r1_launcher_forces_resolved_partition_and_monitor_entry():
    launcher = LAUNCHER.read_text()
    monitor = (ROOT / "ops/exp_scaling/status_e72.py").read_text()
    assert 'name="e75r1pmw-sft"' in launcher
    assert '"Partition=all", "Requeue=0"' in launcher
    assert '"Partition=all",' in launcher
    assert "E75R1 PointMaze weak warm start" not in monitor
