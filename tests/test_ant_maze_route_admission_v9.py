from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_v9_route_gate_was_frozen_before_controller_outcome():
    protocol = (
        ROOT / "paper/preregistration/ant_maze_route_admission_v9_20260729.md"
    ).read_text()
    assert "FROZEN DURING V9 TRAINING" in protocol
    assert "97300..97311" in protocol
    assert "nine-by-nine" in protocol
    assert "upper fixture `N E E S`" in protocol
    assert "lower fixture `S E E N`" in protocol
    assert "2,400 replays" in protocol
    assert "0.15 real routes per second" in protocol


def test_v9_route_wrapper_freezes_disjoint_maps_and_programs():
    generator = (ROOT / "ops/make_ant_maze_mode_data_v9.py").read_text()
    audit = (ROOT / "ops/audit_ant_maze_mode_data_v9.py").read_text()
    assert 'base.UPPER = ("N", "E", "E", "S")' in generator
    assert 'base.LOWER = ("S", "E", "E", "N")' in generator
    assert "base.MAP_SIZE = 9" in generator
    assert "base.STATIC_WALLS = ((4, 4),)" in generator
    assert "base.RESET_SEED_BASE = 97_300" in generator
    assert "base.ACTION_REPEAT = 400" in generator
    assert 'base.ANT_WORKER_SOURCE = "ant_maze_worker_v9.py"' in generator
    assert "controller_receipt_sha256()" in generator
    assert "controller_receipt_sha256()" in audit


def test_v9_worker_uses_terminal_receipt_identity_not_unknown_hardcoded_hash():
    worker = (ROOT / "src/oat_drgrpo/ant_maze_worker_v9.py").read_text()
    dispatch = (ROOT / "src/oat_drgrpo/maze_modebench_worker.py").read_text()
    assert "admitted_to_fresh_maze_route_gate_v9" in worker
    assert 'receipt.get("seed") != 73009' in worker
    assert 'receipt.get("timesteps") != 5_000_000' in worker
    assert "controller_receipt_sha256" in worker
    assert "controller_training_identity_sha256" in worker
    assert 'os.environ.get("OAT_ZERO_PROTOCOL_IDENTITY")' in worker
    assert "ant_v9_receipt_sha256()" in dispatch


def test_v9_route_maps_are_unique_9x9_and_both_frozen_corridors_are_free():
    sys.path.insert(0, str(ROOT / "ops"))
    import make_ant_maze_mode_data as base

    original = {
        name: getattr(base, name)
        for name in ("PERIPHERAL_CELLS", "MAP_SIZE", "STATIC_WALLS")
    }
    try:
        base.PERIPHERAL_CELLS = (
            (1, 1), (2, 1), (3, 1), (4, 1),
            (5, 1), (6, 1), (7, 1), (1, 2),
        )
        base.MAP_SIZE = 9
        base.STATIC_WALLS = ((4, 4),)
        maps = base._maps()
    finally:
        for name, value in original.items():
            setattr(base, name, value)
    assert len(maps) == 12
    assert len({base._canonical_sha256(maze) for maze in maps}) == 12
    upper = ((4, 3), (3, 3), (3, 4), (3, 5), (4, 5))
    lower = ((4, 3), (5, 3), (5, 4), (5, 5), (4, 5))
    for maze in maps:
        assert len(maze) == 9 and all(len(row) == 9 for row in maze)
        assert maze[4][4] == 1
        assert all(maze[row][column] == 0 for row, column in (*upper, *lower))
