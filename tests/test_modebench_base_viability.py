from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_point_maze_viability_is_development_only_and_prospective():
    protocol = (
        ROOT
        / "paper/preregistration/point_maze_05b_viability_v1_20260729.md"
    ).read_text()
    slurm = (
        ROOT / "ops/slurm/evaluate_point_maze_05b_viability.slurm"
    ).read_text()
    launcher = (
        ROOT / "ops/exp_scaling/launch_point_maze_05b_viability.sh"
    ).read_text()

    assert "FROZEN BEFORE THE FIRST MODEL COMPLETION" in protocol
    assert "--data-split-root \"$ROOT_DIR/var/data/point_maze_modebench_v1/dev\"" in slurm
    assert "--minimum-prefix-success-prompts 2" in slurm
    assert "--minimum-multimode-prompts 1" in slurm
    assert "--sample-count 64" in slurm
    assert "--prefix-count 16" in slurm
    assert "point_maze_modebench_v1/eval" not in slurm
    assert "sbatch --parsable --hold" in launcher
    assert "scontrol release" in launcher


def test_base_viability_evaluator_uses_executable_identity():
    evaluator = (ROOT / "ops/evaluate_modebench_base_viability.py").read_text()
    assert "validated_modebench_outcome_key" in evaluator
    assert "certified_route_programs_in_context" in evaluator
    assert "evaluation_prompts_loaded" in evaluator
    assert "llm.generate" in evaluator
