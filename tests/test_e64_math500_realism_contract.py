from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "paper/preregistration/e64_math500_realism_transfer_05b.md"
)
LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e64_math500_realism_smoke.sh"
)
AUDITOR = (
    ROOT / "ops/exp_scaling/audit_e64_math500_realism_smoke.py"
)
MATCHED = (
    ROOT / "ops/exp_scaling/launch_e64_math500_realism_matched.sh"
)
MATCHED_AUDITOR = (
    ROOT / "ops/exp_scaling/audit_e64_math500_realism_matched.py"
)
CHECKPOINT_AUDITOR = (
    ROOT / "ops/exp_scaling/audit_e64_math500_realism_checkpoint.py"
)
PLOT = ROOT / "ops/exp_scaling/plot_e64_math500_realism.py"
WATCHER = ROOT / "ops/exp_scaling/watch_e64_math500_realism.sh"
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e64_math500_audit_classification_amendment_20260728.md"
)


def _load_matched_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_e64_math500_realism_matched",
        MATCHED_AUDITOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_a_held_out_realism_track_not_a_fake_mode_domain():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "external-validity and generalization track" in text
    assert "not a fifth multi-mode ModeBench" in text
    assert "MATH-500 is evaluation-only" in text
    assert "problem overlap: exactly zero" in text
    assert "distinct@8" in text
    assert "not a reasoning-route endpoint" in text


def test_launcher_freezes_data_boundary_and_verified_answer_contract():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "math12k_384_math500" in text
    assert "normalized_problem_overlap" in text
    assert "OAT_ZERO_MAX_TRAIN=96" in text
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_K=0" in text
    assert "OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=math_verified_answer" in text
    assert "OAT_ZERO_PROMPT_TEMPLATE=qwen_math" in text
    assert "OAT_ZERO_VERIFIER_VERSION=math_verify" in text
    assert "OAT_ZERO_TEST_SPLIT=math" in text
    assert '"math500_is_training_input": False' in text
    assert '"reasoning_strategy_claim": False' in text
    assert '"auditor_sha256"' in text


def test_auditor_requires_singleton_balance_inactivity_and_mass_activity():
    text = AUDITOR.read_text(encoding="utf-8")
    for required in (
        "online_canonical_support_at_least_two_prompt_fraction",
        "canonical_replay_balance_loss",
        "canonical_replay_balance_score_gradient_sum",
        "canonical_replay_balance_score_gradient_l2",
        "canonical_replay_observations",
        "canonical_replay_mass_observations",
        "canonical_replay_mass_score_gradient_sum",
        "canonical_replay_mass_score_gradient_l2",
        "canonical_replay_projection_active",
        "canonical_replay_mass_projection_active",
        "math_verified_answer:correct",
    ):
        assert required in text


def test_matched_launcher_is_smoke_gated_paired_and_held_out():
    text = MATCHED.read_text(encoding="utf-8")
    assert "e64_math500_realism_smoke_audit_latest.json" in text
    assert 'audit.get("status") != "pass"' in text
    assert "checkpoint_gate" in text
    assert "SMOKE_CHECKPOINT_AUDIT" in text
    assert "nonfinite_model_tensors" in text
    assert "OAT_ZERO_TRAIN_SEEDS=43,44,45" in text
    assert (
        "OAT_ZERO_ONLY_ARMS=grpo,verified_first_global_replay_canonical"
        in text
    )
    assert "OAT_ZERO_MAX_PROMPT_EPOCHS=12" in text
    assert "OAT_ZERO_EVAL_PROMPT_INTERVAL=768" in text
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_K=8" in text
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1" in text
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=640100" in text
    assert '"math500_is_training_input": False' in text
    assert '"auditor_sha256"' in text
    assert "expected six" in text


def test_matched_auditor_enforces_realism_and_singleton_contracts():
    text = MATCHED_AUDITOR.read_text(encoding="utf-8")
    assert "EXPECTED_STEP = 384 * 12" in text
    assert "eval/math/eval_count" in text
    assert "MATH-500 eval count is not 500" in text
    assert "expected_pairs" in text
    assert "maximum_verified_support" in text
    assert "canonical_replay_balance_loss" in text
    assert "canonical_replay_balance_score_gradient_l2" in text
    assert "canonical_replay_mass_observations" in text
    assert "canonical_replay_mass_projection_active" in text
    assert "open_set_projection_active" in text


def test_matched_auditor_only_downgrades_exact_caught_verifier_trace(
    tmp_path,
):
    auditor = _load_matched_auditor()
    log = tmp_path / "caught.log"
    log.write_text(
        "[actor_0_0/0] Error during comparison\n"
        "[actor_0_0/0] Traceback (most recent call last):\n"
        "[actor_0_0/0]   File \"/env/math_verify/grader.py\", line 809, "
        "in compare_single_extraction_wrapper\n"
        "[actor_0_0/0]     return compare_single_extraction(g, t)\n",
        encoding="utf-8",
    )

    assert auditor._scan_log(log) == (None, 1)

    log.write_text(
        "[rank0]: Traceback (most recent call last):\n",
        encoding="utf-8",
    )
    crash, caught = auditor._scan_log(log)
    assert crash == "Traceback (most recent call last)"
    assert caught == 0


def test_e64_audit_amendment_freezes_a_classification_only_repair():
    text = AMENDMENT.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "post-run process-audit repair only" in text
    assert "compare_single_extraction_wrapper" in text
    assert "Every other traceback remains fatal" in normalized
    assert "does not change training" in text


def test_checkpoint_gate_scans_all_saved_model_tensors():
    text = CHECKPOINT_AUDITOR.read_text(encoding="utf-8")
    assert "step_00096/mp_rank_00_model_states.pt" in text
    assert 'state.get("module")' in text
    assert "torch.isfinite(tensor).all()" in text
    assert "model_tensor_count" in text
    assert "model_parameter_count" in text
    assert "math_verified_answer:correct" in text


def test_realism_plot_omits_fake_reasoning_diversity_endpoint():
    text = PLOT.read_text(encoding="utf-8")
    assert "Held-out greedy pass@1" in text
    assert "Held-out sampled mean@8" in text
    assert "Held-out sampled pass@8" in text
    assert "Sampled answer extraction@8" in text
    assert "Verified-mass coefficient μ" in text
    assert "Open-set coefficient β" in text
    assert "Balance gradient norm (must be 0)" in text
    assert "not reasoning-mode coverage" in text
    assert "sampled_distinct_correct_at_8" not in text


def test_watcher_refreshes_full_plot_and_stops_fail_closed():
    text = WATCHER.read_text(encoding="utf-8")
    assert "audit_e64_math500_realism_matched.py" in text
    assert "plot_e64_math500_realism.py" in text
    assert "INTERVAL_SECONDS" in text
    assert '"$status" == pass' in text
    assert '"$status" == fail' in text
