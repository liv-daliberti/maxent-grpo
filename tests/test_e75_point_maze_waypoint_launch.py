from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from qualify_point_maze_waypoint_dev_v1 import qualify  # noqa: E402


def _gate_inputs():
    families = ("barrier_3mode", "barrier_4mode", "barrier_5mode")
    per_map = []
    for index in range(32):
        success = index < 24
        per_map.append(
            {
                "map_id": f"dev-{index}",
                "family": families[index % len(families)],
                "mean8": 0.25 if success else 0.0,
                "pass8": float(success),
                "distinct8": 2.0 if index < 16 else float(success),
                "modes_per_success": 0.5 if success else 0.0,
            }
        )
    receipt = {
        "status": "complete",
        "evaluation_only": True,
        "optimizer_updates": 0,
        "evaluation_split": "dev",
        "evaluation_prompt_count": 32,
        "evaluation_coordinates": 1,
        "data_identity_sha256": "data-hash",
        "metrics_sha256": "metrics-hash",
    }
    evaluation = {
        "schema": "point-maze-waypoint-pilot-evaluation-v1",
        "split": "dev",
        "evaluation_prompt_count": 32,
        "evaluation_trajectory_count": 256,
        "mean8": 0.25,
        "per_map": per_map,
    }
    return receipt, evaluation


def test_e75_development_gate_passes_only_the_frozen_band():
    receipt, evaluation = _gate_inputs()
    result = qualify(
        receipt=receipt,
        evaluation=evaluation,
        data_identity_sha256="data-hash",
        metrics_sha256="metrics-hash",
    )
    assert result["status"] == "pass"
    assert result["eligible_for_online_pilot"] is True
    assert result["observed"]["maps_with_pass8"] == 24
    assert result["observed"]["maps_with_two_routes"] == 16


def test_e75_development_gate_fails_closed_on_dead_geometry():
    receipt, evaluation = _gate_inputs()
    for row in evaluation["per_map"]:
        if row["family"] == "barrier_5mode":
            row["pass8"] = 0.0
    result = qualify(
        receipt=receipt,
        evaluation=evaluation,
        data_identity_sha256="data-hash",
        metrics_sha256="metrics-hash",
    )
    assert result["status"] == "fail"
    assert "barrier_5mode" in result["observed"]["dead_families"]
    assert "one or more geometry strata are entirely dead" in result["reasons"]


def test_e75_launcher_freezes_dependencies_and_holds_eval():
    launcher = (
        ROOT / "ops/exp_scaling/launch_e75_point_maze_waypoint_05b.py"
    ).read_text()
    protocol = (
        ROOT / "paper/preregistration/e75_point_maze_waypoint_05b_20260803.md"
    ).read_text()
    process = (ROOT / "src/oat_drgrpo/point_maze_waypoint_process.py").read_text()
    monitor = (ROOT / "ops/exp_scaling/status_e72.py").read_text()
    assert "E75 PointMaze waypoint 0.5B" in monitor
    assert 'name.startswith("e75pmw")' in monitor
    assert 'GPU_NODE = "node208"' in launcher
    assert '"--hold"' in launcher
    assert 'f"--dependency=afterok:{dependency}"' in launcher
    assert '"Partition=all", "Requeue=0"' in launcher
    assert '"Partition=all",' in launcher
    assert '"evaluation_job_submitted": False' in launcher
    assert "No final-evaluation job is submitted" in protocol
    assert 'os.environ.get("OAT_ZERO_SOURCE_ROOT"' in process
