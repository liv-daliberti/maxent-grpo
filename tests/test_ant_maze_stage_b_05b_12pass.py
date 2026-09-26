from pathlib import Path

import train_ant_maze_stage_b_05b_12pass as trainer


ROOT = Path(__file__).resolve().parents[1]


def test_ant_stage_b_protocol_is_frozen_before_gate_and_smoke_outcomes() -> None:
    text = (
        ROOT / "paper/preregistration/ant_maze_stage_b_05b_12pass_20260730.md"
    ).read_text()
    for required in (
        "FROZEN BEFORE THE V13 VIABILITY AND PAIRED-SMOKE OUTCOMES",
        "30202183",
        "30202318",
        "Seeds: 43, 44, 45, 46, and 47",
        "48 optimizer updates",
        "all 49 quarter-pass coordinates",
        "All 132 trajectories",
        "four deterministic temperature-one replicates of K=8",
        "fixed v11 low-level controller",
    ):
        assert required in text


def test_ant_stage_b_exact_schedule_constants() -> None:
    assert trainer.SEEDS == (43, 44, 45, 46, 47)
    assert trainer.TRAIN_PROMPTS == 4
    assert trainer.PASSES == 12
    assert trainer.UPDATES == 48
    assert trainer.EVAL_INTERVAL == 1
    assert trainer.EVAL_DRAWS == 4
    assert trainer.EVAL_K == 8
    assert trainer.EVAL_TRAJECTORIES == 132
    assert trainer.ant_base.HORIZON == 16
    assert len(range(0, trainer.UPDATES + 1, trainer.EVAL_INTERVAL)) == 49


def test_ant_stage_b_evaluation_is_feedback_free_and_route_key_based() -> None:
    source = (ROOT / "ops/train_ant_maze_stage_b_05b_12pass.py").read_text()
    assert 'key = transition.get("canonical_key")' in source
    assert 'result["greedy"]' in source
    assert 'result["mean8"]' in source
    assert 'result["pass8"]' in source
    assert 'result["distinct8"]' in source
    assert '"evaluation_feedback_to_training": False' in source
    assert 'spec.action_repeat != 400' in source
    assert "ant_base.rollout_group" in source


def test_ant_stage_b_launcher_freezes_exact_ten_cells_and_gate() -> None:
    launcher = (
        ROOT / "ops/exp_scaling/launch_ant_maze_stage_b_05b_12pass.py"
    ).read_text()
    batch = (ROOT / "ops/slurm/train_ant_maze_stage_b_05b_12pass.slurm").read_text()
    assert "qualification_passes()" in launcher
    assert 'SEEDS = (43, 44, 45, 46, 47)' in launcher
    assert '"optimizer_updates": 48' in launcher
    assert '"decision_horizon": 16' in launcher
    assert '"action_repeat": 400' in launcher
    assert '"evaluation_trajectories_per_coordinate": 132' in launcher
    assert '"final_seed_cohort": True' in launcher
    assert "--time=3-00:00:00" in batch
    assert "ant_maze_interactive_warmstart_v13" in batch
    assert "ant_maze_modebench_v12/eval" in batch
    assert "--microbatch-size 16" in batch
    assert "paired_smoke_v13r2_audit.json" in launcher
    assert '"policy_microbatch_size": 16' in launcher


def test_ant_stage_b_audit_reexecutes_every_controller_transition() -> None:
    source = (ROOT / "ops/audit_ant_maze_stage_b_05b_12pass.py").read_text()
    assert "interactive_transition_sha256" in source
    assert "AntMazeInteractiveProcess" in source
    assert 'range(1, UPDATES + 1)' in source
    assert 'range(0, UPDATES + 1)' in source
    assert 'spec.action_repeat != 400' in source
    assert 'decision.transition_sha256' in source
    assert '"compute_traversal_match_by_seed"' in source
    assert '"policy_microbatch_size": 16' in source
    assert 'behavior/live drift' in source


def test_ant_stage_b_nested_dependency_is_hash_bound_and_fail_closed() -> None:
    dependency = (
        ROOT / "ops/exp_scaling/launch_ant_maze_stage_b_dependency.sh"
    ).read_text()
    final_batch = (
        ROOT / "ops/slurm/launch_ant_maze_stage_b_after_smoke.slurm"
    ).read_text()
    preparer = (
        ROOT / "ops/exp_scaling/launch_ant_maze_stage_b_dependency_preparer.sh"
    ).read_text()
    preparer_batch = (
        ROOT
        / "ops/slurm/prepare_ant_maze_stage_b_dependency_after_smoke_launch.slurm"
    ).read_text()
    assert '--dependency="afterok:$audit_job_id"' in dependency
    assert '"paper_jobs_before_smoke_pass":False' in dependency
    assert "OAT_ZERO_EXPECTED_LAUNCHER_SHA256" in final_batch
    assert 'audit.get("status") != "pass"' in final_batch
    assert "eligible_for_ten_ant_maze_stage_b_jobs" in final_batch
    assert '--dependency="afterok:$smoke_launch_job_id"' in preparer
    assert '[[ "$smoke_launch_job_id" == 30202318 ]]' in preparer
    assert "OAT_ZERO_EXPECTED_DEPENDENCY_LAUNCHER_SHA256" in preparer_batch
    assert 'exec bash "$LAUNCHER" run' in preparer_batch
