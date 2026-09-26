from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ant_v10_viability_is_frozen_conditional_and_development_only():
    protocol = (
        ROOT / "paper/preregistration/ant_maze_05b_viability_v10_20260729.md"
    ).read_text()
    launcher = (
        ROOT / "ops/exp_scaling/launch_ant_maze_05b_viability_v10.sh"
    ).read_text()
    slurm = (ROOT / "ops/slurm/evaluate_ant_maze_05b_viability_v10.slurm").read_text()
    assert "FROZEN DURING V10 CONTROLLER TRAINING" in protocol
    assert "first 16" in protocol
    assert "at least two of four" in protocol
    assert "at least one of" in protocol
    assert 'audit.get("status") != "pass"' in launcher
    assert "sbatch --parsable --hold" in launcher
    assert '--data-split-root "$ROOT_DIR/var/data/ant_maze_modebench_v10/dev"' in slurm
    assert "ant_maze_modebench_v10/eval" not in slurm
    assert "--minimum-prefix-success-prompts 2" in slurm
    assert "--minimum-multimode-prompts 1" in slurm
    assert "--sample-count 64" in slurm
    assert "--prefix-count 16" in slurm
    assert "--prompt-repair ant_maze_v10" in slurm
    assert "--seed 107310" in slurm


def test_ant_v10_prompt_repair_discloses_interface_not_witnesses():
    evaluator = (ROOT / "ops/evaluate_modebench_base_viability.py").read_text()
    assert '"ant_maze_v10"' in evaluator
    assert "Each token targets one adjacent grid cell" in evaluator
    assert "N E E S" not in evaluator
    assert "S E E N" not in evaluator
