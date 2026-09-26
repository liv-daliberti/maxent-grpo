from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ant_route_gate_is_prospective_and_model_free():
    protocol = (
        ROOT / "paper/preregistration/ant_maze_route_admission_v1_20260729.md"
    ).read_text()
    launcher = (
        ROOT / "ops/exp_scaling/launch_ant_maze_route_admission_v1.sh"
    ).read_text()
    slurm = (ROOT / "ops/slurm/admit_ant_maze_modebench_v1.slurm").read_text()

    assert "FROZEN BEFORE THE FIRST ADMISSION-MAP EXECUTION" in protocol
    assert "0.15 real Ant" in protocol
    assert "2,400 perturbation" in protocol
    assert "sbatch --parsable --hold" in launcher
    assert '"language_model_sampling": False' in launcher
    assert "#SBATCH --gres=gpu" not in slurm


def test_ant_data_freezes_v5_alphabet_and_two_routes():
    generator = (ROOT / "ops/make_ant_maze_mode_data.py").read_text()
    audit = (ROOT / "ops/audit_ant_maze_mode_data.py").read_text()

    assert 'UPPER = tuple(["N"] * 5' in generator
    assert 'LOWER = tuple(["S"] * 6' in generator
    assert "ACTION_REPEAT = 75" in generator
    assert '"action_repeat": ACTION_REPEAT' in generator
    assert '"STOP"' not in generator
    assert "PERTURBATIONS_PER_ROUTE = 100" in audit
    assert "THROUGHPUT_FLOOR = 0.15" in audit
    assert "validate_with_execution" in generator
    assert "validate_with_execution" in audit
