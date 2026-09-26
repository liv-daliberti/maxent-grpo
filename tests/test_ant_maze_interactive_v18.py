from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v18_interactive_worker_binds_stable_handoff_controller():
    text = (
        ROOT / "src/oat_drgrpo/ant_maze_interactive_worker_v18.py"
    ).read_text()
    assert "ant_maze_worker_v18 as v18" in text
    assert "final_planar_speed <= v18.STABLE_PLANAR_SPEED" in text
    assert "stable_route_failure = not waypoint_reached" in text
    assert '"segment_arrival_speeds"' in text


def test_v18_interactive_process_uses_isolated_snapshot_entrypoint():
    text = (
        ROOT / "src/oat_drgrpo/ant_maze_interactive_process_v18.py"
    ).read_text()
    assert '"-B", "-I", "-c", entrypoint' in text
    assert "ant_maze_interactive_worker_v18" in text
