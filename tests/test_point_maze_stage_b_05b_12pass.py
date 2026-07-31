from pathlib import Path

import train_point_maze_stage_b_05b_12pass as trainer


ROOT = Path(__file__).resolve().parents[1]


def test_point_stage_b_protocol_is_frozen_before_smoke_outcome() -> None:
    text = (ROOT / "paper/preregistration/point_maze_stage_b_05b_12pass_20260730.md").read_text()
    for required in (
        "FROZEN BEFORE THE PAIRED-SMOKE OUTCOME",
        "30201821",
        "30201822",
        "Seeds: 43, 44, 45, 46, and 47",
        "96 optimizer updates",
        "all 49 quarter-pass coordinates",
        "All 132 trajectories",
        "four deterministic temperature-one replicates of K=8",
    ):
        assert required in text


def test_point_stage_b_exact_schedule_constants() -> None:
    assert trainer.SEEDS == (43, 44, 45, 46, 47)
    assert trainer.TRAIN_PROMPTS == 8
    assert trainer.PASSES == 12
    assert trainer.UPDATES == 96
    assert trainer.EVAL_INTERVAL == 2
    assert trainer.EVAL_DRAWS == 4
    assert trainer.EVAL_K == 8
    assert trainer.EVAL_TRAJECTORIES == 132
    assert len(range(0, trainer.UPDATES + 1, trainer.EVAL_INTERVAL)) == 49


def test_point_stage_b_evaluation_is_feedback_free_and_route_key_based() -> None:
    source = (ROOT / "ops/train_point_maze_stage_b_05b_12pass.py").read_text()
    assert 'key = transition.get("canonical_key")' in source
    assert 'result["greedy"]' in source
    assert 'result["mean8"]' in source
    assert 'result["pass8"]' in source
    assert 'result["distinct8"]' in source
    assert '"evaluation_feedback_to_training": False' in source
    assert "evaluation_metrics.append(evaluation)" in source
    assert "canonical.score_and_update" in source


def test_point_stage_b_launcher_freezes_exact_ten_cells_and_gate() -> None:
    launcher = (ROOT / "ops/exp_scaling/launch_point_maze_stage_b_05b_12pass.py").read_text()
    batch = (ROOT / "ops/slurm/train_point_maze_stage_b_05b_12pass.slurm").read_text()
    assert "qualification_passes()" in launcher
    assert 'SEEDS = (43, 44, 45, 46, 47)' in launcher
    assert '"optimizer_updates": 96' in launcher
    assert '"evaluation_trajectories_per_coordinate": 132' in launcher
    assert '"final_seed_cohort": True' in launcher
    assert "--time=3-00:00:00" in batch
    assert "--evaluation-root" in batch
    assert "--microbatch-size 16" in batch
    assert "paired_smoke_v3_audit.json" in launcher
    assert '"policy_microbatch_size": 16' in launcher


def test_point_stage_b_audit_reexecutes_every_stored_transition() -> None:
    source = (ROOT / "ops/audit_point_maze_stage_b_05b_12pass.py").read_text()
    assert "interactive_transition_sha256" in source
    assert "PointMazeInteractiveProcess" in source
    assert 'range(1, UPDATES + 1)' in source
    assert 'range(0, UPDATES + 1, 2)' in source
    assert 'decision.transition_sha256' in source
    assert '"compute_traversal_match_by_seed"' in source
    assert '"policy_microbatch_size": 16' in source
    assert 'behavior/live drift' in source


def test_geometry_shift_stage_b_is_prospectively_separate_and_gate_bound() -> None:
    protocol = (
        ROOT
        / "paper/preregistration/point_maze_geometry_shift_stage_b_05b_12pass_v1_20260730.md"
    ).read_text()
    launcher = (
        ROOT / "ops/exp_scaling/launch_point_maze_stage_b_05b_12pass.py"
    ).read_text()
    batch = (ROOT / "ops/slurm/train_point_maze_stage_b_05b_12pass.slurm").read_text()
    auditor = (ROOT / "ops/audit_point_maze_stage_b_05b_12pass.py").read_text()
    assert "before its outcome and before any final replacement cell" in protocol
    assert "configuration-level replacement" in protocol
    assert 'choices=("standard", "geometry_shift")' in launcher
    assert "point_maze_geometry_shift_paired_smoke_v1_audit.json" in launcher
    assert "OAT_ZERO_POINT_STAGE_B_VARIANT" in batch
    assert "GEOMETRY_SHIFT_FAMILIES" in auditor
