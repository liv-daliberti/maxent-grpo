from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/launch_e49t_natural_menu_math_toy_05b.sh"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/e49t_menu_inferred_math_haarnoja_05b.md"
)


def test_e49t_is_matched_current_e46_for_exactly_three_epochs():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja" in source
    assert "OAT_ZERO_MAX_PROMPT_EPOCHS=3" in source
    assert "OAT_ZERO_NUM_PROMPT_EPOCH=3" in source
    assert "OAT_ZERO_TRAIN_SEEDS=45" in source
    assert "OAT_ZERO_NUM_SAMPLES=16" in source
    assert "OAT_ZERO_LEARNING_RATE=0.0000002" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=0.50" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0" in source


def test_e49t_requires_full_calibration_and_shared_execution_gate():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "e49t_route_confusion_calibration_result_v1" in source
    assert "all(calibration.get(\"checks\", {}).values())" in source
    assert "calibration endpoint identity mismatch" in source
    assert "e49t_declaration_mismatch_result_v1" in source
    assert "declaration endpoint identity mismatch" in source
    assert "calibration canonicalizer source identity mismatch" in source
    assert "declaration canonicalizer source identity mismatch" in source
    assert "OAT_ZERO_MATH_STRATEGY_ALLOW_UNSTRUCTURED_INFERENCE=1" in source
    assert "OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD=1" in source
    assert "answer_validator_plus_unanimous_finite_menu_route_inference" in source


def test_e49t_uses_one_a100_per_arm_and_frozen_natural_menu_data():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "e49t_natural_menu_math_toy" in source
    assert "e49t_natural_menu_materialization_v1" in source
    assert "OAT_ZERO_TRAIN_NODELIST=node302" in source
    assert "OAT_ZERO_TRAIN_GRES=gpu:a100:1" in source
    assert "OAT_ZERO_TRAIN_MEMORY=48G" in source
    assert "source snapshot mismatch" in source
    assert "launcher_sha256" in source


def test_e49t_protocol_is_frozen_before_training():
    source = PROTOCOL.read_text(encoding="utf-8")
    assert "PREREGISTERED BEFORE TRAINING" in source
    assert "current E46 normalized canonical-bank" in source
    assert "exactly three prompt epochs" in source
    assert "384-train/MATH-500-eval" in source
