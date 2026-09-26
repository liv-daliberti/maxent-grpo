from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_e46_protocol_and_launcher_freeze_normalized_dual_contract():
    protocol = (
        ROOT
        / "paper/preregistration/e46_normalized_online_canonical_haarnoja_05b.md"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT
        / "ops/exp_scaling/launch_e46_normalized_canonical_haarnoja_05b.sh"
    ).read_text(encoding="utf-8")

    for required in (
        "**Status: FROZEN BEFORE LAUNCH",
        "`rho_x = H(q_x) / log |B_x^+|`",
        "`K_x >= 2`",
        "`rho*=0.80`",
        "`alpha in [0.10, 0.50]`",
        "next round",
        "not a contemporaneously launched three-arm",
    ):
        assert required in protocol
    for required in (
        "export OAT_ZERO_ONLY_ARMS=online_canonical_haarnoja",
        "export OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80",
        "export OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10",
        "export OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=0.50",
        "export OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003",
        "export OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90",
        "export OAT_ZERO_TRAIN_NODELIST=",
        "export OAT_ZERO_TRAIN_GRES=",
    ):
        assert required in launcher


def test_e46_runtime_checkpoints_bank_and_controller_separately():
    run = (ROOT / "src/oat_drgrpo/learner/run.py").read_text(
        encoding="utf-8"
    )
    controller = (
        ROOT / "src/oat_drgrpo/online_canonical_controller.py"
    ).read_text(encoding="utf-8")
    bank = (ROOT / "src/oat_drgrpo/online_canonical_bank.py").read_text(
        encoding="utf-8"
    )

    assert "online_canonical_bank_state" in run
    assert "online_canonical_alpha_controller_state" in run
    assert "verified_bank_entropy_over_log_support_v1" in controller
    assert "idle_diagnostics" in controller
    assert "normalized_entropy_ratio_eligible_fraction" in bank
    assert "entropy_alpha_override" in bank
