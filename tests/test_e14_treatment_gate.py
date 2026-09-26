from __future__ import annotations

import math

import pytest

from exp_scaling.check_e14_treatments import (
    ENTROPY_GAIN_MIN,
    GateError,
    _inspect_update,
    choose_arm,
    classify_treatment,
)


def _row(step: int, *, alpha: float = 0.01) -> dict:
    entropy = math.log(27.0) - 0.1
    row = {
        "actor/canonical_action_count": 3.0,
        "actor/canonical_action_support_size": 3.0,
        "actor/canonical_behavior_q_norm_error_max": 1e-8,
        "actor/canonical_behavior_q_row_count": 48.0,
        "actor/canonical_behavior_q_support_max": 3.0,
        "actor/canonical_behavior_q_support_min": 3.0,
        "actor/canonical_finish_length_count": 16.0,
        "actor/canonical_finish_unexpected_count": 0.0,
        "actor/canonical_graph_actions": 1.0,
        "actor/canonical_invalid_count": 0.0,
        "actor/canonical_sampler_fixed_shape": 1.0,
        "actor/canonical_sampler_learner": 1.0,
        "actor/formatted": 1.0,
        "actor/generate_avg_str_len": 3.0,
        "actor/no_eos_count": 0.0,
        "actor/num_data": 16.0,
        "actor/response_tok_len": 3.0,
        "actor/rewards": 0.5,
        "actor/sampling_max_tokens": 3.0,
        "actor/sampling_temperature": 1.0,
        "misc/lr": 2e-7,
        "misc/prompt_consumed": float(step * 16),
        "misc/query_step": float(step * 16),
        "train/canonical_action_count": 3.0,
        "train/canonical_action_vocab_size": 3.0,
        "train/canonical_behavior_denominator_actor": 1.0,
        "train/canonical_behavior_kl_actor_learner_max": 0.0,
        "train/canonical_behavior_kl_actor_learner_mean": 0.0,
        "train/canonical_behavior_kl_learner_actor_max": 0.0,
        "train/canonical_behavior_kl_learner_actor_mean": 0.0,
        "train/canonical_behavior_prefix_ess_fraction_min": 1.0,
        "train/canonical_behavior_q_norm_error_max": 0.0,
        "train/canonical_behavior_q_row_count": 48.0,
        "train/canonical_behavior_q_support_max": 3.0,
        "train/canonical_behavior_q_support_min": 3.0,
        "train/canonical_behavior_ratio_max": 1.0,
        "train/canonical_behavior_ratio_min": 1.0,
        "train/canonical_behavior_sequence_ess_fraction": 1.0,
        "train/canonical_behavior_tv_max": 0.0,
        "train/canonical_behavior_tv_mean": 0.0,
        "train/canonical_sampled_prefix_entropy_ratio": entropy / math.log(27.0),
        "train/canonical_sampled_prefix_entropy_sum": entropy,
        "train/canonical_token_entropy_mean": entropy / 3.0,
        "train/entropy": entropy / 3.0,
        "train/learning_round": float(step),
        "train/pg_loss": 0.0,
        "train/policy_grad_norm": 1.0,
        "train/maxent_alpha_used": alpha,
        "train/maxent_sequence_entropy": entropy,
        "train/maxent_sequence_entropy_per_tmax": entropy / 192.0,
        "train/maxent_entropy_surrogate": entropy,
        "train/maxent_sampled_prefix_entropy": entropy,
        "train/maxent_sampled_prefix_entropy_per_tmax": entropy / 192.0,
        "train/maxent_prefix_ratio_mean": 1.0,
        "train/maxent_prefix_ratio_max": 1.0,
        "train/maxent_prefix_ratio_clipfrac": 0.0,
        "train/maxent_entropy_loss": -alpha * (15.0 / 16.0) * entropy / 192.0,
        "train/maxent_reward_estimator_scale": 15.0 / 16.0,
        "train/maxent_valid_row_fraction": 1.0,
    }
    return row


def test_treatment_update_requires_fixed_alpha_and_single_outer_scale():
    summary = _inspect_update(_row(17), step=17, alpha=0.01)
    assert summary["entropy"] == pytest.approx(math.log(27.0) - 0.1)
    assert summary["entropy_loss"] < 0

    row = _row(17)
    row["train/maxent_entropy_loss"] /= 192.0
    with pytest.raises(GateError, match="entropy loss"):
        _inspect_update(row, step=17, alpha=0.01)


def test_treatment_update_rejects_alpha_drift_and_adaptive_telemetry():
    with pytest.raises(GateError, match="alpha"):
        _inspect_update(_row(9, alpha=0.05), step=9, alpha=0.01)

    row = _row(9)
    row["train/maxent_dual_alpha_next"] = 0.02
    with pytest.raises(GateError, match="prohibited controller"):
        _inspect_update(row, step=9, alpha=0.01)


def test_treatment_update_accepts_theoretical_kl_zero_roundoff_only():
    row = _row(105)
    row["train/canonical_behavior_kl_actor_learner_mean"] = -8.511e-19
    _inspect_update(row, step=105, alpha=0.01)

    row["train/canonical_behavior_kl_actor_learner_mean"] = -1.01e-12
    with pytest.raises(GateError, match="roundoff-aware range"):
        _inspect_update(row, step=105, alpha=0.01)


def test_preregistered_scientific_classification_uses_exact_endpoint_metrics():
    c0 = {
        "p_valid_mean": 0.30,
        "exact_action_entropy_mean": 1.0,
        "n_eff_valid_mean": 2.0,
    }
    passing = classify_treatment(
        arm="M01",
        c0_endpoint=c0,
        endpoint={
            "p_valid_mean": 0.25,
            "exact_action_entropy_mean": 1.0 + ENTROPY_GAIN_MIN,
            "n_eff_valid_mean": 2.5,
        },
        final_reward=0.1,
    )
    assert passing["behaviorally_safe"] is True
    assert passing["diversity_effective"] is True
    assert passing["viable"] is True

    allocation_failure = classify_treatment(
        arm="M05",
        c0_endpoint=c0,
        endpoint={
            "p_valid_mean": 0.20,
            "exact_action_entropy_mean": 1.5,
            "n_eff_valid_mean": 2.2,
        },
        final_reward=0.1,
    )
    assert allocation_failure["behaviorally_safe"] is False
    assert allocation_failure["diversity_effective"] is False
    assert allocation_failure["viable"] is False


def test_selection_uses_valid_support_and_small_alpha_tie_break():
    base = {"viable": True}
    selected, _ = choose_arm(
        [
            {**base, "arm": "M01", "n_eff_valid_mean": 2.00},
            {**base, "arm": "M05", "n_eff_valid_mean": 2.09},
        ]
    )
    assert selected == "M01"

    selected, _ = choose_arm(
        [
            {**base, "arm": "M01", "n_eff_valid_mean": 2.00},
            {**base, "arm": "M05", "n_eff_valid_mean": 2.20},
        ]
    )
    assert selected == "M05"

    assert choose_arm([{**base, "arm": "M01", "n_eff_valid_mean": 2.0, "viable": False}])[0] is None
