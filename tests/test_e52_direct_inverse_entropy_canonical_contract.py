from pathlib import Path


ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e52_direct_inverse_entropy_canonical_05b.md"
)
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/"
    "launch_e52_direct_inverse_entropy_canonical_05b.sh"
)
STABILITY_AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e52_scale_free_stability_gate_amendment_20260726.md"
)


def test_e52_freezes_label_free_unbounded_direct_entropy_controller():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "**Status: FROZEN BEFORE SENTINEL SUBMISSION",
        "`q_theta(a | s, continue) = pi_theta(a | s) / (1 - pi_theta(EOS | s))`",
        "`L_direct = -lambda_t * H_t`",
        "`lambda_(t+1) = 0.000075 * H_ref / M_t`",
        "no lower projection, upper projection",
        "gold list or count of valid outcomes",
        "canonical-bank size, entropy, or support",
        "graph coloring",
        "Countdown easy3",
        "executable Python factors",
        "exactly 50 complete prompt-pool passes",
    ):
        assert required in protocol

    for required in (
        "EXPECTED_JOBS_PER_DOMAIN=3",
        "EXPECTED_JOBS=9",
        "PROMPT_EPOCHS=50",
        "OAT_ZERO_ONLY_ARMS=grpo,maxent_inverse,maxent_inverse_canonical",
        "OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean",
        "OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA=0.000075",
        "OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS=64",
        "OAT_ZERO_MAXENT_INVERSE_EMA_DECAY=0.90",
        "OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0",
        '"evaluation_feedback": False',
        '"canonical_bank_feedback": False',
        '"alpha_projection": None',
        "scontrol release",
    ):
        assert required in launcher


def test_e52_held_audit_separates_control_direct_and_hybrid_actuators():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "'OAT_ZERO_MAXENT_ALPHA=0.0'",
        "'OAT_ZERO_MAXENT_INVERSE_ADAPTATION=0'",
        "'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0'",
        "'OAT_ZERO_VARIANT=maxent_inverse'",
        "'OAT_ZERO_MAXENT_ALPHA=0.000075'",
        "'OAT_ZERO_MAXENT_INVERSE_ADAPTATION=1'",
        "'OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS=64'",
        "'OAT_ZERO_MAXENT_INVERSE_EMA_DECAY=0.90'",
        'OAT_ZERO_VARIANT=maxent_inverse_canonical',
        "'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10'",
        "'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50'",
    ):
        assert required in launcher


def test_e52_sentinel_has_fresh_three_domain_prefixes_and_one_seed():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "gce52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2",
        "cde52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2_allcs",
        "pye52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2_allcs",
        "OAT_ZERO_TRAIN_SEEDS=9009",
        "submit_domain graph_coloring",
        "submit_domain countdown",
        "submit_domain python_factor",
    ):
        assert required in launcher


def test_e52_stability_gate_is_scale_free_and_terminal_only():
    amendment = STABILITY_AMENDMENT.read_text(encoding="utf-8")
    audit = (
        ROOT / "ops/exp_scaling/audit_e52_sentinel.py"
    ).read_text(encoding="utf-8")

    for required in (
        "FROZEN DURING SENTINEL PASS 2, BEFORE ANY TERMINAL WINDOW",
        "`X_t = distinct-correct@8_t - pass@8_t`",
        "valid-answer catalogue",
        "gold mode count",
        "at least six of eight times",
        "retain at least half",
        "exact pass-50 boundary",
        "cannot pass or fail Stage S",
    ):
        assert required in amendment

    for required in (
        '"higher_mean_distinct_excess_over_pass"',
        '"excess_wins_at_least_six"',
        '"positive_multiplicity_at_least_six"',
        '"retains_half_of_own_best_rolling_eight"',
        'control.get("status") != "complete"',
        'hybrid.get("status") != "complete"',
    ):
        assert required in audit
