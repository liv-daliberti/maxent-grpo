"""Contract for E25's 3B free-form conditional-token dual treatment."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e25_modebench_freeform_dual_3b.sh"
PROTOCOL = ROOT / "paper/preregistration/e25_modebench_freeform_dual_3b.md"


def test_e25_is_the_requested_treatment_only_3b_scale_extension():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-3b-instruct" in launcher
    assert "OAT_ZERO_ONLY_ARMS=maxent_dual" in launcher
    assert "OAT_ZERO_TRAIN_SEEDS=43,44,45" in launcher
    assert "gce25_freeform_conditional_dual_3b_v2" in launcher
    assert "cde25_freeform_conditional_dual_3b_v2" in launcher
    assert "treatment-only exploratory scale extension" in protocol


def test_e25_preserves_e22_v2_objective_controller_and_tasks():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean" in text
    assert "OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.000075" in text
    assert "OAT_ZERO_CANONICAL_ACTION_TASK=none" in text
    assert "OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed" in text
    assert "OAT_ZERO_NUM_SAMPLES=16" in text
    assert "OAT_ZERO_NUM_PROMPT_EPOCH=5" in text


def test_e25_v2_uses_125_percent_targets_and_aggressive_controller():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "GRAPH_TARGET=1.622718550885717" in text
    assert "COUNTDOWN_TARGET=1.347109432487438" in text
    assert "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.000075" in text
    assert "OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00060" in text
    assert "OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.010" in text


def test_e25_uses_held_audited_one_a100_jobs():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_SBATCH_HOLD=1" in text
    assert "OAT_ZERO_E25_TRAIN_NODELIST:-node302" in text
    assert "OAT_ZERO_E25_TRAIN_GRES:-gpu:a100:1" in text
    assert "OAT_ZERO_E25_TRAIN_MEMORY:-96G" in text
    assert "OAT_ZERO_ADAM_OFFLOAD=1" in text
    assert "OAT_ZERO_ACTIVATION_OFFLOADING=1" in text
    assert "E25 cohort incomplete (${#job_ids[@]}/6)" in text
    assert "release-staged" in text
    assert "TresPerNode=gres/gpu:a100:1" in text
    assert 'scontrol release "${job_ids[@]}"' in text
