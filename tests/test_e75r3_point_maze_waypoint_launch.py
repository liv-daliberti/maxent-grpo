from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e75r3_point_maze_waypoint_weak_sft_05b.py"
PROTOCOL = (
    ROOT
    / "paper/preregistration/e75r3_point_maze_waypoint_weak_sft_05b_20260804.md"
)


def test_e75r3_is_a_separate_prospective_successor():
    launcher = LAUNCHER.read_text()
    protocol = PROTOCOL.read_text()
    normalized_protocol = " ".join(protocol.split())
    assert '"experiment": "E75R3"' in launcher
    assert '"data_seed": 88103' in launcher
    assert '"sft_optimizer_updates": 72' in launcher
    assert '"evaluation_job_submitted": False' in launcher
    assert '"predecessors": [' in launcher
    assert "E75 remains immutable" in normalized_protocol
    assert "E75R1 remains immutable" in normalized_protocol
    assert "E75R2 remains immutable" in normalized_protocol
    assert "There is no adaptive checkpoint search" in normalized_protocol
    assert "A failed gate is the terminal E75R3 result" in normalized_protocol


def test_e75r3_stages_use_disjoint_paths_and_fixed_budget():
    prepare = (ROOT / "ops/slurm/e75r3_point_maze_waypoint_prepare.slurm").read_text()
    sft = (ROOT / "ops/slurm/e75r3_point_maze_waypoint_sft.slurm").read_text()
    dev = (ROOT / "ops/slurm/e75r3_point_maze_waypoint_dev.slurm").read_text()
    train = (ROOT / "ops/slurm/e75r3_point_maze_waypoint_train.slurm").read_text()
    assert "--seed 88103" in prepare
    assert prepare.count("--exclude-identity") == 3
    assert "point_maze_waypoint_pilot_e75r3" in prepare
    assert "--max-updates 72" in sft
    assert "--seed 88404" in sft
    assert "point_maze_waypoint_warmstart_e75r3" in sft
    assert "--seed 88504" in dev
    assert "--seed 88504" in train
    assert "e75r3_point_maze_waypoint" in dev
    assert "e75r3_point_maze_waypoint" in train


def test_e75r3_launcher_forces_resolved_partition_and_monitor_entry():
    launcher = LAUNCHER.read_text()
    monitor = (ROOT / "ops/exp_scaling/status_e72.py").read_text()
    assert 'name="e75r3pmw-sft"' in launcher
    assert '"Partition=all", "Requeue=0"' in launcher
    assert '"Partition=all",' in launcher
    assert "E75R3 PointMaze calibrated warm start" in monitor
    assert "E75R3 untouched final eval" in monitor
