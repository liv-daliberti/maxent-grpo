from pathlib import Path

import train_ant_maze_interactive_paired_smoke_v13 as trainer


ROOT = Path(__file__).resolve().parents[1]


def test_ant_v13_pair_is_frozen_before_viability_outcome() -> None:
    text = (ROOT / "paper/preregistration/ant_maze_interactive_paired_smoke_v13_20260730.md").read_text()
    for required in (
        "FROZEN BEFORE THE V13 VIABILITY OUTCOME",
        "30202183",
        "eligible_for_ant_v13_paired_online_smoke",
        "seed: 76313",
        "16 × 16 policy-forward budget",
        "fixed v11 low-level controller",
    ):
        assert required in text


def test_ant_v13_pair_exact_training_contract() -> None:
    assert trainer.SEED == 76313
    assert trainer.SAMPLES == 16
    assert trainer.HORIZON == 16
    assert trainer.REPLAY_CAPACITY == 16
    source = (ROOT / "ops/train_ant_maze_interactive_paired_smoke_v13.py").read_text()
    assert "spec.action_repeat != 400" in source
    assert "ANT_POLICY_ACTIONS" in source
    assert "AntMazeInteractiveProcess" in source
    assert '"compute_only_control": args.arm == CONTROL' in source


def test_ant_v13_pair_audit_replays_controller_transitions() -> None:
    source = (ROOT / "ops/audit_ant_maze_interactive_paired_smoke_v13.py").read_text()
    assert "interactive_transition_sha256" in source
    assert "render_ant_policy_prompt" in source
    assert 'metric.get("fixed_policy_slots") != 256' in source
    assert 'metric.get("replay_decision_forward_slots") != 256.0' in source
    assert 'metric.get("policy_microbatch_size") != expected_microbatch' in source
    assert '"eligible_for_ten_ant_maze_stage_b_jobs"' in source


def test_ant_v13_pair_launcher_is_conditional_and_development_only() -> None:
    launcher = (ROOT / "ops/exp_scaling/launch_ant_maze_interactive_paired_smoke_v13.py").read_text()
    batch = (ROOT / "ops/slurm/train_ant_maze_interactive_paired_smoke_v13.slurm").read_text()
    assert "viability_passes()" in launcher
    assert '"optimizer_updates_per_arm": 4' in launcher
    assert '"fixed_policy_slots_per_arm": 1024' in launcher
    assert '"development_only": True' in launcher
    assert '"policy_microbatch_size": 16 if COHORT == "v13r2" else 4' in launcher
    assert 'choices=("v13", "v13r1", "v13r2")' in launcher
    assert '[[ "$COHORT" == "v13r2" ]] && MICROBATCH=16' in batch
    assert "--model \"$ROOT_DIR/var/models/ant_maze_interactive_warmstart_v13\"" in batch


def test_ant_v13r2_binds_failed_pair_and_exact_batch16_diagnostic() -> None:
    launcher = (ROOT / "ops/exp_scaling/launch_ant_maze_interactive_paired_smoke_v13.py").read_text()
    protocol = (
        ROOT / "paper/preregistration/ant_maze_interactive_paired_smoke_v13r2_20260730.md"
    ).read_text()
    assert '"failed_predecessor_jobs": [30202916, 30202917, 30202918]' in launcher
    assert '"v13r1_batch_diagnostic_sha256": sha(V13R1_DIAGNOSTIC)' in launcher
    assert "Only policy/replay microbatch changes from 4 to 16" in protocol


def test_ant_v13_pair_dependency_is_hash_bound_and_fail_closed() -> None:
    submission = (
        ROOT
        / "ops/exp_scaling/launch_ant_maze_paired_smoke_v13_dependency.sh"
    ).read_text()
    batch = (
        ROOT / "ops/slurm/launch_ant_maze_paired_smoke_after_v13.slurm"
    ).read_text()
    assert '--dependency="afterok:$v13_job_id"' in submission
    assert '[[ "$v13_job_id" == 30202183 ]]' in submission
    assert '"online_training_before_v13_pass": False' in submission
    assert '"paper_jobs_before_paired_smoke_pass": False' in submission
    assert "OAT_ZERO_EXPECTED_LAUNCHER_SHA256" in batch
    assert 'viability.get("status") != "pass"' in batch
    assert "eligible_for_ant_v13_paired_online_smoke" in batch
    assert 'exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"' in batch
