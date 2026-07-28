from pathlib import Path


ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e51_policy_entropy_adaptive_canonical_05b_50ep.md"
)
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/"
    "launch_e51_policy_entropy_adaptive_canonical_05b_50ep.sh"
)


def test_e51_freezes_three_domain_unprojected_policy_entropy_cohort():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "**Status: FROZEN BEFORE CORRECTED RELAUNCH",
        "policy-token-uncertainty controller",
        "`d_t = h_ref / m_t`",
        "`alpha_(t+1) = 0.10 * d_t`",
        "no explicit lower or upper projection",
        "no Haarnoja/SAC loss",
        "graph coloring",
        "Countdown easy3",
        "executable Python factors",
        "exactly 50 complete prompt-pool passes",
    ):
        assert required in protocol

    for required in (
        "EXPECTED_JOBS_PER_DOMAIN=6",
        "EXPECTED_JOBS=18",
        "OAT_ZERO_ONLY_ARMS=grpo,online_canonical_policy_entropy",
        "OAT_ZERO_INCLUDE_ONLINE_CANONICAL_POLICY_ENTROPY_ARM=1",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=1",
        "OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_WARMUP_STEPS=64",
        "OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_EMA_DECAY=0.90",
        "gce51_policy_entropy_adaptive_canonical_05b_50ep_v2",
        "cde51_policy_entropy_adaptive_canonical_05b_50ep_v2_allcs",
        "pye51_policy_entropy_adaptive_canonical_05b_50ep_v2_allcs",
        "scontrol release",
    ):
        assert required in launcher


def test_e51_control_and_treatment_are_held_audited_separately():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0'",
        "'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.0'",
        "'OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0'",
        "'OAT_ZERO_VARIANT=online_canonical_policy_entropy'",
        "'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10'",
        "'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50'",
        "'OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=1'",
        '"reference_alpha": 0.10',
        '"alpha_projection": None',
        '"haarnoja_dual": False',
    ):
        assert required in launcher
