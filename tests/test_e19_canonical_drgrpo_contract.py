"""Frozen contract for the post-hoc E19 canonical Dr.GRPO controls."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "paper/preregistration/e19_canonical_drgrpo_05b_control.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e19_canonical_drgrpo_05b_control.sh"


def test_e19_protocol_is_a_post_hoc_shared_canonical_control():
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "Status: FROZEN POST-HOC MATCHED-CONTROL EXTENSION" in text
    assert "canonical Dr.GRPO (`alpha=0`) only" in text
    assert "seeds: 43, 44, and 45" in text
    assert "six jobs total" in text
    assert "post-hoc and exploratory" in text
    assert "Historical free-text Dr.GRPO" in text


def test_e19_launcher_changes_only_the_entropy_treatment():
    text = LAUNCHER.read_text(encoding="utf-8")

    frozen_values = (
        "OAT_ZERO_ONLY_ARMS=grpo",
        "OAT_ZERO_XDR_TAU=inf",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0",
        "OAT_ZERO_MAXENT_ALPHA=0",
        "OAT_ZERO_MAXENT_CONTROL_RATIO=0",
        "OAT_ZERO_MAXENT_DUAL_RATIO=0",
        "OAT_ZERO_BETA=0",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_NUM_PROMPT_EPOCH=5",
        "OAT_ZERO_LEARNING_RATE=0.0000002",
        "OAT_ZERO_NUM_PPO_EPOCHS=1",
        "OAT_ZERO_TRAIN_BATCH_SIZE=16",
        "OAT_ZERO_ROLLOUT_BATCH_SIZE=1",
        "OAT_ZERO_AUTO_RESUME=0",
        "OAT_ZERO_WATCHDOG_REQUEUE=0",
    )
    for value in frozen_values:
        assert f"export {value}" in text

    assert "15344 960 48" in text
    assert "30704 1920 96" in text
    assert "gce19_canonical_drgrpo_05b_v1" in text
    assert "cde19_canonical_drgrpo_05b_v1" in text
    assert "scontrol release" in text
    assert "manifest already exists; refusing a duplicate cohort" in text
    assert "config|full|release" in text
    assert 'grep -Fqx \'  --beta "${OAT_ZERO_BETA:-0}"\'' in text
    assert "Reason=JobHeldUser" in text
    assert "OAT_ZERO_SEED_ENTROPY_ALPHA=0.0" in text
    assert "OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO=0.0" in text
