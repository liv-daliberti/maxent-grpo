"""Contract for E22's 0.5B free-form ModeBench extension."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper/preregistration/e22_modebench_freeform_token_maxent.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e22_modebench_freeform_token_maxent.sh"


def test_protocol_separates_free_form_from_canonical_policy():
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "free-form conditional-token MaxEnt (Haarnoja dual)" in text
    assert "graph coloring and Countdown" in text
    assert "No canonical action restriction" in text
    assert "Seeds are 43, 44, and 45" in text


def test_launcher_uses_e21_objective_and_haarnoja_dual_only():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_ONLY_ARMS=maxent_dual" in text
    assert "OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean" in text
    assert "OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.000075" in text
    assert "OAT_ZERO_MAXENT_DUAL_RATIO=0.8" in text
    assert "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.00005" in text
    assert "OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00015" in text
    assert "OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.005" in text
    assert "OAT_ZERO_MAXENT_LENGTH_TARGET=0" in text


def test_launcher_is_unrestricted_and_covers_both_05b_tasks():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_CANONICAL_ACTION_TASK=none" in text
    assert "OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed" in text
    assert "OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct" in text
    assert "OAT_ZERO_TRAIN_SEEDS=43,44,45" in text
    assert "OAT_ZERO_MAX_TRAIN=192" in text
    assert "OAT_ZERO_MAX_TRAIN=384" in text
    assert "submit_task graph_coloring" in text
    assert "submit_task countdown" in text
