"""Contract for E28's post-hoc matched 3B free-form Dr.GRPO controls."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e28_modebench_freeform_drgrpo_3b_control.sh"
PROTOCOL = ROOT / "paper/preregistration/e28_modebench_freeform_drgrpo_3b_control.md"


def test_e28_is_a_post_hoc_six_job_matched_control():
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "Status: FROZEN POST-HOC MATCHED-CONTROL EXTENSION" in protocol
    assert "E28 changes exactly the entropy treatment relative to E25-v2" in protocol
    assert "seeds 43, 44, and 45" in protocol
    assert "six jobs total" in protocol
    assert "post-hoc and exploratory" in protocol


def test_e28_launcher_disables_only_the_entropy_treatment():
    text = LAUNCHER.read_text(encoding="utf-8")

    for value in (
        "OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct",
        "OAT_ZERO_ONLY_ARMS=grpo",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_NUM_PROMPT_EPOCH=5",
        "OAT_ZERO_LEARNING_RATE=0.0000002",
        "OAT_ZERO_NUM_PPO_EPOCHS=1",
        "OAT_ZERO_TRAIN_BATCH_SIZE=16",
        "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1",
        "OAT_ZERO_ROLLOUT_BATCH_SIZE=1",
        "OAT_ZERO_MAXENT_ALPHA=0",
        "OAT_ZERO_MAXENT_DUAL_RATIO=0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0",
        "OAT_ZERO_XDR_TAU=inf",
        "OAT_ZERO_CANONICAL_ACTION_TASK=none",
        "OAT_ZERO_ADAM_OFFLOAD=1",
        "OAT_ZERO_ACTIVATION_OFFLOADING=1",
    ):
        assert f"export {value}" in text

    assert "gce28_freeform_drgrpo_3b_v1" in text
    assert "cde28_freeform_drgrpo_3b_v1" in text
    assert "OAT_ZERO_SBATCH_HOLD=1" in text
    assert "E28 cohort incomplete (${#job_ids[@]}/6)" in text
    assert 'scontrol release "${job_ids[@]}"' in text
    assert "Reason=JobHeldUser" in text


def test_e28_launcher_has_audited_same_stamp_countdown_recovery():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "countdown-retry|countdown-release-staged" in text
    assert "export OAT_ZERO_APPEND_MANIFEST=1" in text
    assert "E28 Countdown recovery incomplete (${#job_ids[@]}/3)" in text
    assert "OAT_ZERO_AUTO_RESUME=1" in text
    assert "no duplicate jobs submitted" in text
    assert "released three same-stamp Countdown recovery jobs" in text
