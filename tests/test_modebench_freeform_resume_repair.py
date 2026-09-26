"""Contracts for the stale-actor ModeBench free-form remediation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_modebench_freeform_3b_resume_repair.sh"
PROTOCOL = (
    ROOT / "paper/preregistration/modebench_freeform_resume_repair_20260722.md"
)
RUN_EXPERIMENT = ROOT / "ops/run_experiment.sh"
SUBMITTER = ROOT / "ops/submit_countdown_comparative.sh"


def test_protocol_certifies_clean_05b_and_enumerates_every_3b_repair():
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "Status: FROZEN REMEDIATION ADDENDUM" in text
    assert "certified clean and are not rerun" in text
    for boundary in ("576", "480", "96", "864", "144", "48"):
        assert boundary in text
    for prefix in (
        "gce25_freeform_conditional_dual_3b_repair_v2",
        "cde25_freeform_conditional_dual_3b_repair_v2",
        "gce28_freeform_drgrpo_3b_repair_v2",
        "cde28_freeform_drgrpo_3b_repair_v2",
    ):
        assert prefix in text


def test_launcher_preserves_frozen_methods_and_starts_from_clean_state():
    text = LAUNCHER.read_text(encoding="utf-8")

    for setting in (
        "OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_NUM_PROMPT_EPOCH=5",
        "OAT_ZERO_LEARNING_RATE=0.0000002",
        "OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean",
        "OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.000075",
        "OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00060",
        "OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.010",
        "OAT_ZERO_CANONICAL_ACTION_TASK=none",
        "OAT_ZERO_AUTO_RESUME=1",
        "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0",
        "OAT_ZERO_SBATCH_HOLD=1",
    ):
        assert f"export {setting}" in text

    assert "submit_fresh_cohort graph_coloring \"$E25_GRAPH_PREFIX\" maxent_dual" in text
    assert "submit_fresh_cohort countdown \"$E25_COUNTDOWN_PREFIX\" maxent_dual" in text
    assert "submit_fresh_cohort graph_coloring \"$E28_GRAPH_PREFIX\" grpo" in text
    assert "submit_resumed_countdown_seed 43 'debug_0721T18:03:35' step_00096" in text
    assert "submit_resumed_countdown_seed 44 'debug_0721T18:03:34' step_00096" in text
    assert "submit_resumed_countdown_seed 45 'debug_0721T23:20:55' step_00864" in text
    assert "Repair cohort incomplete (${#job_ids[@]}/12)" in text
    assert 'scontrol release "${job_ids[@]}"' in text


def test_external_bootstrap_is_one_time_and_requeues_follow_clean_branch():
    text = RUN_EXPERIMENT.read_text(encoding="utf-8")

    assert "OAT_ZERO_INITIAL_RESUME_DIR" in text
    assert "SLURM_RESTART_COUNT" in text
    assert "initial_resume=skipped on restart" in text
    assert "initial_resume=fallback" in text
    assert text.index("initial_resume=skipped on restart") < text.index(
        "find \"$SAVE_PATH\""
    )
    assert text.index("find \"$SAVE_PATH\"") < text.index(
        "initial_resume=fallback"
    )


def test_recovery_provenance_is_explicit_in_slurm_submit_line():
    text = SUBMITTER.read_text(encoding="utf-8")

    for name in (
        "OAT_ZERO_AUTO_RESUME",
        "OAT_ZERO_RESUME_STEPS",
        "OAT_ZERO_RESUME_FROM",
        "OAT_ZERO_MAX_RESUME_NUM",
        "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS",
        "OAT_ZERO_INITIAL_RESUME_DIR",
        "OAT_ZERO_INITIAL_RESUME_TAG",
    ):
        assert f'export_vars+=",{name}=' in text
