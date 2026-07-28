from pathlib import Path


ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50_uncapped_normalized_canonical_haarnoja_05b_50ep.md"
)
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/"
    "launch_e50_uncapped_normalized_canonical_haarnoja_05b_50ep.sh"
)


def test_e50_freezes_fresh_matched_uncapped_fifty_pass_cohort():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "**Status: FROZEN BEFORE LAUNCH",
        "from-scratch replacement",
        "exactly 50 complete prompt-pool passes",
        "`grpo`: ordinary matched Dr.GRPO",
        "`online_canonical_haarnoja`",
        "`alpha_max = +inf`",
        "no configured upper bound",
        "identically zero",
    ):
        assert required in protocol
    for required in (
        "EXPECTED_JOBS_PER_TASK=6",
        "EXPECTED_JOBS=12",
        "PROMPT_EPOCHS=50",
        "export OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja",
        "export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1",
        "export OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10",
        "export OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=inf",
        'export OAT_ZERO_MAX_PROMPT_EPOCHS="$PROMPT_EPOCHS"',
        'export OAT_ZERO_NUM_PROMPT_EPOCH="$PROMPT_EPOCHS"',
        "gce50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1",
        "cde50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1",
        '"alpha_upper_bound": None',
        '"upper_projection": False',
        "scontrol release",
    ):
        assert required in launcher


def test_e50_held_audit_attests_uncapped_treatment_and_zero_control():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0'",
        "'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.0'",
        "'OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0'",
        "'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10'",
        "'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50'",
        "'OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80'",
        "'OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10'",
        "'OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=inf'",
        "'OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003'",
        "'OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90'",
        '"TimeLimit=7-00:00:00"',
    ):
        assert required in launcher
