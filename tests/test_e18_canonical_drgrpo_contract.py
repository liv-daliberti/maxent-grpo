"""Frozen contract for the post-hoc E18 canonical Dr.GRPO controls."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "paper/preregistration/e18_canonical_drgrpo_3b_control.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e18_canonical_drgrpo_3b_control.sh"


def test_e18_protocol_is_a_post_hoc_matched_canonical_control():
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "Status: FROZEN POST-HOC MATCHED-CONTROL EXTENSION" in text
    assert "matched canonical Dr.GRPO only" in text
    assert "Seeds: 43, 44, and 45" in text
    assert "six jobs total" in text
    assert "post-hoc/exploratory" in text
    assert "Qwen2.5-3B-Instruct revision" in text


def test_e18_launcher_is_reward_only_on_the_frozen_canonical_surface():
    text = LAUNCHER.read_text(encoding="utf-8")

    frozen_values = (
        "OAT_ZERO_ONLY_ARMS=grpo",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0",
        "OAT_ZERO_MAXENT_ALPHA=0",
        "OAT_ZERO_MAXENT_LENGTH_TARGET=0",
        "OAT_ZERO_BETA=0",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_NUM_PROMPT_EPOCH=5",
        "OAT_ZERO_LEARNING_RATE=0.0000002",
        "OAT_ZERO_NUM_PPO_EPOCHS=1",
        "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=1",
        "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1",
    )
    for value in frozen_values:
        assert f"export {value}" in text

    assert "gce18_canonical_drgrpo_3b_v1" in text
    assert "cde18_canonical_drgrpo_3b_v1" in text
    assert "exact_answer_mode_probe" in text
    assert "exact_countdown_easy3_probe" in text
    assert "scontrol release" in text


def test_e18_launcher_has_countdown_only_same_stamp_recovery():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "countdown-config" in text
    assert "countdown-retry" in text
    assert "export OAT_ZERO_APPEND_MANIFEST=1" in text
    assert "manifest_lines_before" in text
    assert "released three Countdown recovery jobs" in text


def test_e18_launcher_has_fail_closed_single_seed_priority_recoveries():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "graph-seed43-config" in text
    assert "graph-seed43-retry" in text
    assert "graph-seed44-config" in text
    assert "graph-seed44-retry" in text
    assert "graph-seed45-config" in text
    assert "graph-seed45-retry" in text
    assert "countdown-seed44-config" in text
    assert "countdown-seed44-retry" in text
    assert 'export OAT_ZERO_TRAIN_SEEDS="$seed"' in text
    assert 'OAT_ZERO_E18_VLLM_GPU_RATIO:-0.25' in text
    assert "held-job audit failed" in text
    assert 'scontrol release "$job_id"' in text


def test_e18_countdown_seed43_recovery_retires_loop_and_starts_fresh():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "countdown-seed43-config" in text
    assert "countdown-seed43-retry" in text
    assert "export OAT_ZERO_AUTO_RESUME=0" in text
    assert "E18-fresh-start-no-auto-resume" in text
    assert 'scontrol update JobId="$prior_job_id" Requeue=0' in text
    assert 'scancel "$prior_job_id"' in text
    assert "replacement remains held" in text
    assert '"--nodelist=${OAT_ZERO_TRAIN_NODELIST}"' in text
