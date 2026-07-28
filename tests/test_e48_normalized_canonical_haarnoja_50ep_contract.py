from pathlib import Path


ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT
    / "paper/preregistration/e48_normalized_canonical_haarnoja_05b_50ep.md"
)
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/launch_e48_normalized_canonical_haarnoja_05b_50ep.sh"
)


def test_e48_freezes_fresh_matched_fifty_pass_cohort():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "**Status: FROZEN BEFORE LAUNCH",
        "fresh contemporaneous",
        "exactly 50 complete prompt-pool passes",
        "`grpo`: ordinary matched Dr.GRPO",
        "`online_canonical_haarnoja`",
        "cumulative verified discoveries",
        "mean verified support per tracked prompt",
        "identically zero",
    ):
        assert required in protocol
    for required in (
        "EXPECTED_JOBS_PER_TASK=6",
        "EXPECTED_JOBS=12",
        "PROMPT_EPOCHS=50",
        "export OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja",
        "export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1",
        'export OAT_ZERO_MAX_PROMPT_EPOCHS="$PROMPT_EPOCHS"',
        'export OAT_ZERO_NUM_PROMPT_EPOCH="$PROMPT_EPOCHS"',
        'OAT_ZERO_E48_TRAIN_TIME_LIMIT:-7-00:00:00',
        "gce48_normalized_canonical_haarnoja_05b_50ep_v1",
        "cde48_normalized_canonical_haarnoja_05b_50ep_v1",
        '("grpo", "online_canonical_haarnoja")',
        "scontrol release",
    ):
        assert required in launcher


def test_e48_held_audit_distinguishes_zero_influence_control():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "'OAT_ZERO_MAX_PROMPT_EPOCHS=50'",
        "'OAT_ZERO_NUM_PROMPT_EPOCH=50'",
        "'OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1'",
        "'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0'",
        "'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.0'",
        "'OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0'",
        "'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10'",
        "'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50'",
        "'OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80'",
        '"TimeLimit=7-00:00:00"',
    ):
        assert required in launcher


def test_current_live_figure_shows_discovery_metrics_for_drgrpo():
    plot = (
        ROOT / "ops/exp_scaling/plot_divergence.py"
    ).read_text(encoding="utf-8")

    assert "passive_control_metrics" in plot
    assert '"online_canonical_mean_support_per_prompt"' in plot
    assert '"online_canonical_tracked_outcomes"' in plot
    assert "e51_current_canonical_05b_live" in plot
