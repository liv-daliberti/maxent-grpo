"""Contract for E22-v2's matched, base-preserving free-form comparison."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper/preregistration/e22_modebench_freeform_token_maxent_v2.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e22_modebench_freeform_token_maxent_v2.sh"
CALIBRATION = ROOT / "paper/results/e22_freeform_dual_v1_calibration.json"


def test_v2_preserves_base_dose_and_uses_fixed_calibrated_targets():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))

    assert "OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.000075" in launcher
    assert "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.000075" in launcher
    assert "DUAL_MAX_ALPHA=0.00015" in launcher
    assert "OAT_ZERO_MAXENT_DUAL_RATIO=1.0" in launcher
    assert "GRAPH_TARGET=1.2981748407085736" in launcher
    assert "COUNTDOWN_TARGET=1.0776875459899504" in launcher
    assert calibration["warmup_steps"] == 64
    assert calibration["calibration"]["graph_coloring"]["fixed_target_entropy"] == 1.2981748407085736
    assert calibration["calibration"]["countdown"]["fixed_target_entropy"] == 1.0776875459899504


def test_v2_has_a_matched_free_form_control_and_fresh_prefixes():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "ONLY_ARMS=grpo,maxent_dual" in launcher
    assert "gce22_freeform_conditional_dual_05b_v2" in launcher
    assert "cde22_freeform_conditional_dual_05b_v2" in launcher
    assert "EXPECTED_JOBS_PER_TASK=6" in launcher
    assert "free-form Dr.GRPO (`alpha=0`" in protocol
    assert "canonical-action Dr.GRPO is not a matched control" in protocol


def test_v2_remains_unrestricted_and_frozen_before_launch():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "**Status: FROZEN FOR LAUNCH" in protocol
    assert "OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean" in launcher
    assert "OAT_ZERO_CANONICAL_ACTION_TASK=none" in launcher
    assert "OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed" in launcher
    assert "OAT_ZERO_MAXENT_LENGTH_TARGET=0" in launcher
