from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v11_route_gate_frozen_during_controller_and_reuses_only_unexecuted_slate():
    text = (ROOT / "paper/preregistration/ant_maze_route_admission_v11_20260730.md").read_text()
    assert "FROZEN DURING V11 TRAINING" in text
    assert "wholly unexecuted slate" in text
    assert "107300..107311" in text
    assert "2,400 deterministic" in text
    assert "0.15 routes/second" in text


def test_v11_wrappers_keep_exact_v10_route_geometry_with_new_binding():
    generator = (ROOT / "ops/make_ant_maze_mode_data_v11.py").read_text()
    audit = (ROOT / "ops/audit_ant_maze_mode_data_v11.py").read_text()
    assert 'base.UPPER = ("N", "E", "E", "S")' in generator
    assert 'base.LOWER = ("S", "E", "E", "N")' in generator
    assert "base.MAP_SIZE = 11" in generator
    assert "base.RESET_SEED_BASE = 107_300" in generator
    assert 'base.ANT_WORKER_SOURCE = "ant_maze_worker_v11.py"' in generator
    assert "controller_receipt_sha256()" in generator and "controller_receipt_sha256()" in audit


def test_v11_worker_and_launcher_bind_exact_controller_and_hold_job():
    worker = (ROOT / "src/oat_drgrpo/ant_maze_worker_v11.py").read_text()
    launcher = (ROOT / "ops/exp_scaling/launch_ant_maze_route_admission_v11.sh").read_text()
    assert "admitted_to_fresh_maze_route_gate_v11" in worker
    assert 'training_identity.get("job_id") != 30198291' in worker
    assert "ant-maze-v11-route-generation-identity-v1" in worker
    assert "V11 route freeze requires the v10 route slate to remain wholly unexecuted" in launcher
    assert "sbatch --parsable --hold" in launcher
    assert '"reused_unexecuted_v10_slate":True' in launcher
    assert '"frozen_before_controller_outcome":True' in launcher
