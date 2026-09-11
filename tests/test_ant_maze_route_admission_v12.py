from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v12_protocol_freezes_only_grid_anchor_after_v11_failure():
    text = (
        ROOT
        / "paper/preregistration/ant_maze_route_admission_v12_anchored_waypoint_repair_20260730.md"
    ).read_text()
    assert "FROZEN AFTER THE TERMINAL V11 ROUTE FAILURE" in text
    assert "30199417" in text
    assert "0.5586808" in text
    assert "initial_xy + 4 * cumulative_sum" in text
    assert "There is no map, seed, route, threshold, or outcome substitution" in text
    assert "other 23 simple route fixtures" in text


def test_v12_worker_anchors_targets_and_keeps_controller_contract():
    worker = (ROOT / "src/oat_drgrpo/ant_maze_worker_v12.py").read_text()
    assert 'TARGETING_VERSION = "initial-grid-cumulative-v12"' in worker
    assert "cumulative_heading += _heading(token)" in worker
    assert "target = initial + WAYPOINT_DISTANCE * cumulative_heading" in worker
    assert "current + WAYPOINT_DISTANCE" not in worker
    assert 'MODEL_PATH = ROOT / "var/maze_runtime/controllers/ant_waypoint_v11.zip"' in worker
    assert 'training_identity.get("job_id") != 30198291' in worker


def test_v12_wrappers_reuse_exact_v11_geometry_and_new_worker_identity():
    generator = (ROOT / "ops/make_ant_maze_mode_data_v12.py").read_text()
    v11 = (ROOT / "ops/make_ant_maze_mode_data_v11.py").read_text()
    audit = (ROOT / "ops/audit_ant_maze_mode_data_v12.py").read_text()
    for frozen in (
        'base.UPPER = ("N", "E", "E", "S")',
        'base.LOWER = ("S", "E", "E", "N")',
        "base.MAP_SIZE = 11",
        "base.RESET_SEED_BASE = 107_300",
        "base.ACTION_REPEAT = 400",
    ):
        assert frozen in v11
    assert "import make_ant_maze_mode_data as base" in generator
    assert "import make_ant_maze_mode_data_v11" not in generator
    for frozen in (
        'base.UPPER = ("N", "E", "E", "S")',
        'base.LOWER = ("S", "E", "E", "N")',
        "base.MAP_SIZE = 11",
        "base.RESET_SEED_BASE = 107_300",
        "base.ACTION_REPEAT = 400",
    ):
        assert frozen in generator
    assert 'base.ANT_WORKER_SOURCE = "ant_maze_worker_v12.py"' in generator
    assert 'base.ANT_WORKER_SOURCE = "ant_maze_worker_v12.py"' in audit


def test_v12_cold_start_timeout_is_operational_only():
    generator = (ROOT / "ops/make_ant_maze_mode_data_v12.py").read_text()
    dispatcher = (ROOT / "src/oat_drgrpo/maze_modebench_worker.py").read_text()
    process = (ROOT / "src/oat_drgrpo/maze_modebench_process.py").read_text()
    assert "class _V12VerifierProcess(base.MazeVerifierProcess)" in generator
    assert "super().__init__(timeout_seconds=90.0" in generator
    assert "OAT_ZERO_MAZE_WORKER_TIMEOUT_SECONDS" in dispatcher
    assert "WORKER_TIMEOUT_SECONDS" in dispatcher
    assert '[str(self.worker_python), "-B", "-I", "-c", entrypoint]' in process
    assert "spec.action_repeat" in (
        ROOT / "src/oat_drgrpo/ant_maze_worker_v12.py"
    ).read_text()


def test_dispatcher_selects_unique_v12_executor_before_all_older_bound_receipts():
    dispatcher = (ROOT / "src/oat_drgrpo/maze_modebench_worker.py").read_text()
    v12 = dispatcher.index("spec.controller_sha256 == ant_v12_receipt_sha256()")
    v9 = dispatcher.index("spec.controller_sha256 == ant_v9_receipt_sha256()")
    v10 = dispatcher.index("spec.controller_sha256 == ant_v10_receipt_sha256()")
    v11 = dispatcher.index("spec.controller_sha256 == ant_v11_receipt_sha256()")
    assert v12 < v9 < v10 < v11
