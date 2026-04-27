"""Logging helpers for the public Dr.GRPO vs Dr.X comparison."""

from __future__ import annotations

import os
from typing import Any

WANDB_DEBUG_METRIC_ENV = "OAT_ZERO_WANDB_LOG_DEBUG_METRICS"

_WANDB_DROP_EXACT_METRICS = {
    "train/get_grad_norm_time",
    "train/logprobs_diff_max",
    "train/logprobs_diff_min",
    "train/zero_pg_loss_count",
    "train/listwise_grad_probe_enabled",
    "train/listwise_grad_probe_update_index",
    "train/listwise_grad_probe_valid",
    "train/listwise_semantic_answer_key_extracted_frac",
    "train/listwise_semantic_trace_extracted_frac",
    "train/listwise_semantic_signature_extracted_frac",
    "train/listwise_semantic_cluster_valid_frac",
    "train/listwise_semantic_trace_truncated_frac",
    "train/listwise_semantic_cluster_ref_available",
    "train/listwise_semantic_exploration_prompt_count",
    "train/listwise_semantic_exploration_entropy_std",
    "train/listwise_semantic_exploration_gain_any_correct_std",
    "train/listwise_semantic_exploration_gain_drgrpo_std",
}

_WANDB_DROP_METRIC_PREFIXES = (
    "train/listwise_grad_",
    "train/listwise_semantic_competitive_",
    "train/listwise_semantic_correctness_schedule_",
    "train/listwise_semantic_expected_",
    "train/listwise_semantic_exploration_gain_any_correct_",
    "train/listwise_semantic_exploration_gain_drgrpo_",
    "train/listwise_semantic_prompt_",
    "train/listwise_tau_adaptation_metric_is_",
    "train/listwise_tau_signal_",
)


def _as_finite_float(value: Any) -> float | None:
    """Return a plain finite float for scalar-ish log values."""

    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if scalar != scalar or scalar in {float("inf"), float("-inf")}:
        return None
    return scalar


def _copy_metric(
    target: dict[str, Any],
    source: dict[str, Any],
    output_key: str,
    source_key: str,
) -> float | None:
    value = _as_finite_float(source.get(source_key))
    if value is not None:
        target[output_key] = value
    return value


def add_public_drx_training_metrics(logs_dict: dict[str, Any]) -> dict[str, Any]:
    """Add readable Dr.X dashboard metrics derived from raw learner logs.

    Raw listwise metrics are useful for debugging but hard to scan during a
    long run. These summaries keep the important questions visible:
    how expensive was the Dr.X update, how often did it have useful signal,
    and whether semantic extraction is healthy.
    """

    enriched = dict(logs_dict)
    learn_time = _copy_metric(
        enriched,
        logs_dict,
        "drx/perf/learn_seconds",
        "train/learn_batch_time",
    )
    total_train_time = _copy_metric(
        enriched,
        logs_dict,
        "drx/perf/train_total_seconds",
        "train/total_time",
    )
    actor_time = _copy_metric(
        enriched,
        logs_dict,
        "drx/perf/actor_seconds",
        "actor/total_time",
    )
    if actor_time is None:
        actor_time = _copy_metric(
            enriched,
            logs_dict,
            "drx/perf/actor_seconds",
            "actor/generate_time",
        )
    actor_num_data = _as_finite_float(logs_dict.get("actor/num_data"))
    if actor_num_data is not None and learn_time and learn_time > 0:
        enriched["drx/perf/train_samples_per_second"] = actor_num_data / learn_time
    if actor_time and actor_time > 0 and learn_time is not None:
        enriched["drx/perf/learn_to_actor_time_ratio"] = learn_time / actor_time
    if total_train_time and total_train_time > 0 and learn_time is not None:
        enriched["drx/perf/learn_time_frac"] = learn_time / total_train_time

    skip_rate = _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/zero_signal_skip_rate",
        "train/listwise_zero_signal_skip",
    )
    if skip_rate is not None:
        enriched["drx/signal/useful_update_rate"] = max(0.0, min(1.0, 1.0 - skip_rate))
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/active_group_frac",
        "train/listwise_active_group_frac",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/active_group_count",
        "train/listwise_active_group_count_global",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/token_active_row_frac",
        "train/listwise_drgrpo_token_active_row_frac",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/token_active_row_count",
        "train/listwise_drgrpo_token_active_row_count_global",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/neutral_group_frac",
        "train/listwise_neutral_group_frac",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/raw_neutral_group_frac",
        "train/listwise_raw_neutral_group_frac",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/all_wrong_group_frac",
        "train/listwise_all_wrong_group_frac",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/mixed_correctness_group_frac",
        "train/listwise_mixed_correctness_group_frac",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/all_correct_group_frac",
        "train/listwise_all_correct_group_frac",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/signal/any_correct_group_frac",
        "train/listwise_any_correct_group_frac",
    )

    semantic_health_keys = (
        "train/listwise_semantic_answer_key_extracted_frac",
        "train/listwise_semantic_trace_extracted_frac",
        "train/listwise_semantic_signature_extracted_frac",
        "train/listwise_semantic_cluster_valid_frac",
    )
    semantic_health_values = [
        value
        for key in semantic_health_keys
        if (value := _as_finite_float(logs_dict.get(key))) is not None
    ]
    if semantic_health_values:
        enriched["drx/semantic/extraction_health"] = sum(semantic_health_values) / len(
            semantic_health_values
        )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/semantic/cluster_count",
        "train/listwise_semantic_cluster_count",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/semantic/cluster_entropy",
        "train/listwise_semantic_cluster_entropy",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/semantic/exploration_gain",
        "train/listwise_semantic_exploration_gain_drgrpo",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/semantic/correctness_gate",
        "train/listwise_exact_semantic_gate_mean",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/listwise_adv_abs_mean",
        "train/listwise_adv_abs_mean",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/semantic_piece_mean",
        "train/listwise_exact_semantic_piece_mean",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/correctness_adv_abs_mean",
        "train/listwise_exact_correctness_adv_abs_mean",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/semantic_adv_abs_mean",
        "train/listwise_exact_semantic_adv_abs_mean",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/semantic_to_correctness_adv_ratio",
        "train/listwise_exact_semantic_to_correctness_adv_ratio",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/semantic_adv_fraction",
        "train/listwise_exact_semantic_adv_fraction",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/semantic_adv_abs_all_wrong",
        "train/listwise_exact_semantic_adv_abs_mean_all_wrong",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/semantic_adv_abs_mixed",
        "train/listwise_exact_semantic_adv_abs_mean_mixed",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/semantic_adv_abs_all_correct",
        "train/listwise_exact_semantic_adv_abs_mean_all_correct",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/semantic_effective_group_frac",
        "train/listwise_exact_semantic_effective_group_frac",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/semantic_correctness_adv_cosine",
        "train/listwise_exact_semantic_correctness_adv_cosine",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/drgrpo_adv_abs_mean",
        "train/drgrpo_adv_abs_mean",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/post_scale_ratio",
        "train/listwise_post_scale_ratio",
    )
    _copy_metric(
        enriched,
        logs_dict,
        "drx/objective/helpfulness_proxy",
        "train/listwise_helpfulness_proxy",
    )
    return enriched


def filter_wandb_logs_for_public_comparison(
    logs_dict: dict[str, Any],
) -> dict[str, Any]:
    """Keep W&B focused on the public Dr.GRPO vs Dr.X comparison surface."""

    debug_value = os.environ.get(WANDB_DEBUG_METRIC_ENV, "")
    if str(debug_value).strip().lower() in {"1", "true", "yes", "on"}:
        return logs_dict

    filtered_logs: dict[str, Any] = {}
    for key, value in logs_dict.items():
        if key in _WANDB_DROP_EXACT_METRICS:
            continue
        if any(key.startswith(prefix) for prefix in _WANDB_DROP_METRIC_PREFIXES):
            continue
        filtered_logs[key] = value
    return filtered_logs
