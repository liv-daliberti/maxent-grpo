"""Contract for E26's treatment-only high-entropy MATH dual."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e26_math_freeform_dual_high_entropy.sh"
PROTOCOL = ROOT / "paper/preregistration/e26_math_freeform_dual_high_entropy.md"
CALIBRATION = ROOT / "paper/results/e26_math_freeform_dual_high_entropy_calibration.json"


def test_e26_launches_only_the_base_preserving_dual():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_ONLY_ARMS=maxent_dual" in text
    assert "OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1" in text
    assert "OAT_ZERO_INCLUDE_MAXENT_ARM=0" in text
    assert "OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0" in text
    assert "OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean" in text
    assert "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=\"$ALPHA_BASE\"" in text


def test_e26_freezes_the_125_percent_target_and_aggressive_controller():
    text = LAUNCHER.read_text(encoding="utf-8")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))

    assert "TARGET=0.4567" in text
    assert "ALPHA_MAX=0.00060" in text
    assert "ALPHA_LR=0.010" in text
    assert calibration["new_fixed_target_entropy"] == 0.4567
    assert calibration["alpha"]["min"] == 0.000075
    assert calibration["alpha"]["max"] == 0.0006
    assert calibration["alpha_lr"] == 0.01
    assert calibration["new_fixed_target_entropy"] > calibration["pooled_warmup_mean_entropy"]
    assert calibration["new_fixed_target_entropy"] > calibration["legacy_pooled_target_entropy"]


def test_e26_is_three_seed_one_pass_math_and_held_audited():
    text = LAUNCHER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "OAT_ZERO_TRAIN_SEEDS=43,44,45" in text
    assert "OAT_ZERO_MAX_TRAIN=8523" in text
    assert "OAT_ZERO_NUM_PROMPT_EPOCH=1" in text
    assert "OAT_ZERO_SBATCH_HOLD=1" in text
    assert "E26 cohort incomplete (${#job_ids[@]}/3)" in text
    assert "scontrol release \"${job_ids[@]}\"" in text
    assert "does not restart E21's cancelled" in protocol
