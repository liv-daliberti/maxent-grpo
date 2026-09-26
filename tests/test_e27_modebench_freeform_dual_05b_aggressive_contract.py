"""Contract for E27's aggressive treatment-only 0.5B ModeBench dual."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "ops/exp_scaling/launch_e27_modebench_freeform_dual_05b_aggressive.sh"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e22_modebench_freeform_token_maxent_v2.sh"
PROTOCOL = ROOT / "paper/preregistration/e27_modebench_freeform_dual_05b_aggressive.md"
CALIBRATION = ROOT / "paper/results/e27_modebench_freeform_dual_05b_aggressive_calibration.json"


def test_e27_selects_fresh_treatment_only_prefixes():
    wrapper = WRAPPER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_E27_AGGRESSIVE_05B=1" in wrapper
    assert "gce27_freeform_conditional_dual_05b_v1" in launcher
    assert "cde27_freeform_conditional_dual_05b_v1" in launcher
    assert "ONLY_ARMS=maxent_dual" in launcher
    assert "EXPECTED_JOBS_PER_TASK=3" in launcher


def test_e27_freezes_task_scaled_targets_and_aggressive_controller():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))

    assert "GRAPH_TARGET=1.622718550885717" in launcher
    assert "COUNTDOWN_TARGET=1.347109432487438" in launcher
    assert "DUAL_MAX_ALPHA=0.00060" in launcher
    assert "DUAL_ALPHA_LR=0.010" in launcher
    assert calibration["alpha"] == {"base": 0.000075, "min": 0.000075, "max": 0.0006}
    assert calibration["alpha_lr"] == 0.01


def test_e27_is_frozen_held_and_five_passes():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "**Status: FROZEN FOR LAUNCH" in protocol
    assert "OAT_ZERO_SBATCH_HOLD=1" in launcher
    assert "OAT_ZERO_NUM_PROMPT_EPOCH=5" in launcher
    assert "OAT_ZERO_VARIANT=maxent_dual" in launcher
    assert "scontrol release \"${job_ids[@]}\"" in launcher
