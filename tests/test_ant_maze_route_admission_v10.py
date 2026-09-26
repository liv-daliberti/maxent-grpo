from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_v10_route_gate_was_frozen_before_controller_outcome():
    protocol = (
        ROOT / "paper/preregistration/ant_maze_route_admission_v10_20260729.md"
    ).read_text()
    assert "FROZEN DURING V10 TRAINING" in protocol
    assert "107300..107311" in protocol
    assert "eleven-by-eleven" in protocol
    assert "upper witness `N E E S`" in protocol
    assert "lower witness `S E E N`" in protocol
    assert "2,400 deterministic trajectory" in protocol
    assert "0.15 routes/second" in protocol


def test_v10_route_wrapper_freezes_disjoint_maps_and_programs():
    generator = (ROOT / "ops/make_ant_maze_mode_data_v10.py").read_text()
    audit = (ROOT / "ops/audit_ant_maze_mode_data_v10.py").read_text()
    assert 'base.UPPER = ("N", "E", "E", "S")' in generator
    assert 'base.LOWER = ("S", "E", "E", "N")' in generator
    assert "base.MAP_SIZE = 11" in generator
    assert "base.STATIC_WALLS = ((5, 5),)" in generator
    assert "base.RESET_SEED_BASE = 107_300" in generator
    assert "base.ACTION_REPEAT = 400" in generator
    assert 'base.ANT_WORKER_SOURCE = "ant_maze_worker_v10.py"' in generator
    assert "controller_receipt_sha256()" in generator
    assert "controller_receipt_sha256()" in audit


def test_v10_worker_uses_terminal_receipt_identity_not_unknown_hardcoded_hash():
    worker = (ROOT / "src/oat_drgrpo/ant_maze_worker_v10.py").read_text()
    dispatch = (ROOT / "src/oat_drgrpo/maze_modebench_worker.py").read_text()
    assert "admitted_to_fresh_maze_route_gate_v10" in worker
    assert 'receipt.get("seed") != 73010' in worker
    assert 'receipt.get("timesteps") != 2_000_000' in worker
    assert "controller_receipt_sha256" in worker
    assert "controller_training_identity_sha256" in worker
    assert 'os.environ.get("OAT_ZERO_PROTOCOL_IDENTITY")' in worker
    assert "ant_v10_receipt_sha256()" in dispatch


def test_v10_route_maps_are_unique_11x11_and_both_frozen_corridors_are_free():
    sys.path.insert(0, str(ROOT / "ops"))
    import make_ant_maze_mode_data as base

    original = {
        name: getattr(base, name)
        for name in ("PERIPHERAL_CELLS", "MAP_SIZE", "STATIC_WALLS")
    }
    try:
        base.PERIPHERAL_CELLS = (
            (1, 1), (2, 1), (3, 1), (4, 1),
            (5, 1), (6, 1), (7, 1), (8, 1),
        )
        base.MAP_SIZE = 11
        base.STATIC_WALLS = ((5, 5),)
        maps = base._maps()
    finally:
        for name, value in original.items():
            setattr(base, name, value)
    assert len(maps) == 12
    assert len({base._canonical_sha256(maze) for maze in maps}) == 12
    upper = ((5, 4), (4, 4), (4, 5), (4, 6), (5, 6))
    lower = ((5, 4), (6, 4), (6, 5), (6, 6), (5, 6))
    for maze in maps:
        assert len(maze) == 11 and all(len(row) == 11 for row in maze)
        assert maze[5][5] == 1
        assert all(maze[row][column] == 0 for row, column in (*upper, *lower))
