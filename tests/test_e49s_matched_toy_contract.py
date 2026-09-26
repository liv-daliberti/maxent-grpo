from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/"
    "launch_e49s_deterministic_mathir_toy_05b.sh"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49s_deterministic_mathir_repairs_20260724.md"
)


def test_e49s_is_a_matched_three_epoch_e46_run():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja" in source
    assert "OAT_ZERO_MAX_PROMPT_EPOCHS=3" in source
    assert "OAT_ZERO_NUM_PROMPT_EPOCH=3" in source
    assert "OAT_ZERO_TRAIN_SEEDS=45" in source
    assert "OAT_ZERO_NUM_SAMPLES=16" in source
    assert "OAT_ZERO_LEARNING_RATE=0.0000002" in source
    assert "OAT_ZERO_COMPARATIVE_DATA_ROOT=\"$DATA_ROOT\"" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=0.50" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0" in source
    assert "OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD=1" in source


def test_e49s_uses_one_a100_per_arm_and_frozen_repair_gate():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "OAT_ZERO_TRAIN_NODELIST=node302" in source
    assert "OAT_ZERO_TRAIN_GRES=gpu:a100:1" in source
    assert "advance_to_matched_toy_training" in source
    assert "deterministic_certificates_sha256" in source
    assert "math_strategy_canonicalizer_menu_bound_v18_trace" in source
    assert "launcher_sha256" in source


def test_e49s_protocol_is_frozen_before_training():
    source = PROTOCOL.read_text(encoding="utf-8")
    assert "FROZEN BEFORE REPAIR EXECUTION OR TRAINING" in source
    assert "at least ten train and ten evaluation rows" in source
    assert "regular execution-gated Dr.GRPO arm" in source
