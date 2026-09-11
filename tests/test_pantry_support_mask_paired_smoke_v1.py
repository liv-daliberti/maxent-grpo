from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_freezes_exact_e70_pair_and_authorization_boundary():
    text = (ROOT / "paper/preregistration/pantry_support_mask_paired_mechanism_smoke_v1_20260730.md").read_text()
    for required in ("grpo_compute_matched", "verified_first_global_replay_canonical", "76201", "32 optimizer updates", "capacity 16", "exact-zero applied replay-gradient", "ten Pantry Stage-B jobs"):
        assert required in text


def test_launcher_selects_both_arms_and_e70_coefficients():
    text = (ROOT / "ops/exp_scaling/launch_pantry_support_mask_paired_smoke_v1.py").read_text()
    for required in ("grpo,verified_first_global_replay_canonical", 'OAT_ZERO_DRGRPO_VARIANT="grpo_compute_matched"', 'OAT_ZERO_SEMANTIC_SHANNON_COEF="0.10"', 'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA="0.50"', 'OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA="0.10"'):
        assert required in text


def test_audit_checks_nonzero_treatment_and_zero_control_derivatives():
    text = (ROOT / "ops/audit_pantry_support_mask_paired_smoke_v1.py").read_text()
    for required in ("canonical_replay_applied_score_gradient_l2", "canonical_replay_mass_score_gradient_l2", "semantic_shannon_separate_semantic_advantage_rms", "online_canonical_novelty_advantage_rms", "compute_traversal_match", "eligible_for_ten_stage_b_jobs"):
        assert required in text
