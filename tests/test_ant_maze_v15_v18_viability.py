from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ant_v18_viability_is_stable_handoff_and_development_only():
    protocol = (
        ROOT
        / "paper/preregistration/ant_maze_v15_controller_v18_05b_viability_20260730.md"
    ).read_text()
    batch = (
        ROOT / "ops/slurm/evaluate_ant_maze_v15_controller_v18_viability.slurm"
    ).read_text()
    assert "BEFORE THE V18 CONTROLLER OUTCOME" in protocol
    assert "inclusive interval [0.02, 0.50]" in protocol
    assert "ant_maze_modebench_v15_controller_v18/dev" in batch
    assert "ant_maze_interactive_warmstart_v13" in batch
    assert "--seed 76701" in batch


def test_ant_v18_evaluator_uses_new_interactive_process():
    text = (
        ROOT / "ops/evaluate_ant_maze_interactive_viability_v18.py"
    ).read_text()
    assert "AntMazeInteractiveProcessV18" in text
    assert "stable_waypoint_handoff=True" in text
