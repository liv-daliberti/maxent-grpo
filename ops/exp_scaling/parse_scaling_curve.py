#!/usr/bin/env python
"""Assemble the E1 compute-scaling divergence curve from inline training metrics.

Each training run writes `<run_dir>/debug_<ts>/train_metrics.jsonl`, one record
per optimizer step. Records at eval steps carry the inline sampled-coverage eval:
    eval/{split}/sampled_any_correct_at_8    -> pass@8
    eval/{split}/sampled_mean_at_8           -> mean@8
    eval/{split}/sampled_mode_coverage_at_8  -> coverage@8   (E1 primary)
    eval/{split}/sampled_distinct_correct_at_8 -> distinct@8
    eval/{split}/accuracy                    -> greedy pass@1
No checkpoints or GPU re-eval needed: the whole divergence curve is in the logs.

Usage:
    parse_scaling_curve.py --stamp-prefix gce1_05b [--run-data-root var/data]
                           [--out var/artifacts/gce1_05b_scaling_curve.json]
Emits a tidy long-format JSON: one row per (arm, seed, step, split, metric).
Also prints the treatment divergence coverage@8(xdr) - coverage@8(grpo) vs step.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

SPLITS = ("multi_answer", "unique_answer")
METRICS = {
    "pass8": "sampled_any_correct_at_8",
    "mean8": "sampled_mean_at_8",
    "coverage8": "sampled_mode_coverage_at_8",
    "distinct8": "sampled_distinct_correct_at_8",
    # E45 MathIR deliberately evaluates K=16 and distinguishes any validated
    # program from a validated strategy different from the public seed path.
    # Keep these as separate fields instead of silently presenting them as the
    # older K=8 ModeBench metrics.
    "pass16": "sampled_any_correct_at_16",
    "mean16": "sampled_mean_at_16",
    "distinct16": "sampled_distinct_correct_at_16",
    "nonseed_pass16": "sampled_any_nonseed_correct_at_16",
    "nonseed_distinct16": "sampled_distinct_nonseed_correct_at_16",
    "latent_pass8": "sampled_latent_any_correct_at_8",
    "latent_mean8": "sampled_latent_mean_at_8",
    "latent_coverage8": "sampled_latent_mode_coverage_at_8",
    "latent_distinct8": "sampled_latent_distinct_correct_at_8",
    "option_mi": "sampled_option_answer_mi_lower_bound_nats_at_8",
    "option_classifier": "sampled_option_classifier_accuracy_at_8",
    "option_eligible": "sampled_option_eligible_fraction_at_8",
    "option_correct_range": "sampled_option_correct_rate_range_at_8",
    "greedy": "accuracy",
}
# E37's mechanism telemetry is emitted once per optimizer step rather than
# under an evaluation split. Copy the value present on each evaluation record
# into the tidy curve so the live diagnostic can align it with neutral quality.
# The aliases keep the plotting layer tolerant to the logger's usual `train/`
# namespace as well as actor-side or already-flat records.
OUTCOME_COLLISION_METRICS = {
    "outcome_collision_task_reward_mean": (
        "train/outcome_collision_task_reward_mean",
        "actor/outcome_collision_task_reward_mean",
        "outcome_collision_task_reward_mean",
    ),
    "outcome_collision_augmented_reward_mean": (
        "train/outcome_collision_augmented_reward_mean",
        "actor/outcome_collision_augmented_reward_mean",
        "outcome_collision_augmented_reward_mean",
    ),
    "outcome_collision_bonus_mean": (
        "train/outcome_collision_bonus_mean",
        "actor/outcome_collision_bonus_mean",
        "outcome_collision_bonus_mean",
    ),
    "outcome_collision_bonus_min": (
        "train/outcome_collision_bonus_min",
        "actor/outcome_collision_bonus_min",
        "outcome_collision_bonus_min",
    ),
    "outcome_collision_bonus_max": (
        "train/outcome_collision_bonus_max",
        "actor/outcome_collision_bonus_max",
        "outcome_collision_bonus_max",
    ),
    "outcome_collision_rate": (
        "train/outcome_collision_rate",
        "actor/outcome_collision_rate",
        "outcome_collision_rate",
    ),
    "outcome_collision_distinct_fraction": (
        "train/outcome_collision_distinct_fraction",
        "actor/outcome_collision_distinct_fraction",
        "outcome_collision_distinct_fraction",
    ),
    "outcome_collision_distinct_outcomes_mean": (
        "train/outcome_collision_distinct_outcomes_mean",
        "actor/outcome_collision_distinct_outcomes_mean",
        "outcome_collision_distinct_outcomes_mean",
    ),
    "outcome_collision_invalid_fraction": (
        "train/outcome_collision_invalid_fraction",
        "actor/outcome_collision_invalid_fraction",
        "outcome_collision_invalid_fraction",
    ),
    "outcome_collision_parseable_fraction": (
        "train/outcome_collision_parseable_fraction",
        "actor/outcome_collision_parseable_fraction",
        "outcome_collision_parseable_fraction",
    ),
}
OUTCOME_COLLISION_METRICS.update(
    {
        short: (
            f"train/{short}",
            f"actor/{short}",
            short,
        )
        for short in (
            "outcome_collision_bonus_zero_spread_group_fraction",
            "outcome_collision_centered_bonus_abs_mean",
            "outcome_collision_centered_bonus_rms",
        )
    }
)
SEMANTIC_SHANNON_METRICS = {
    short: (
        f"train/{short}",
        f"actor/{short}",
        short,
    )
    for short in (
        "semantic_shannon_task_reward_mean",
        "semantic_shannon_augmented_reward_mean",
        "semantic_shannon_bonus_mean",
        "semantic_shannon_bonus_min",
        "semantic_shannon_bonus_max",
        "semantic_shannon_surprisal_mean",
        "semantic_shannon_normalized_surprisal_mean",
        "semantic_shannon_entropy_mean",
        "semantic_shannon_clipped_surprisal_mean",
        "semantic_shannon_clip_fraction",
        "semantic_shannon_predictive_probability_mean",
        "semantic_shannon_predictive_probability_min",
        "semantic_shannon_predictive_probability_max",
        "semantic_shannon_normalization_error_max",
        "semantic_shannon_unseen_fraction",
        "semantic_shannon_history_total_mean",
        "semantic_shannon_distinct_outcomes_mean",
        "semantic_shannon_distinct_fraction",
        "semantic_shannon_invalid_fraction",
        "semantic_shannon_parseable_fraction",
        "semantic_shannon_tracked_prompts",
        "semantic_shannon_tracked_outcomes",
    )
}
OUTCOME_COLLISION_OUTSIDE_CENTERING_METRICS = {
    short: (
        f"train/{short}",
        f"actor/{short}",
        short,
    )
    for short in (
        "outcome_collision_reward_sent_to_centering_mean",
        "outcome_collision_outside_centering_active",
        "outcome_collision_outside_base_advantage_mean",
        "outcome_collision_outside_base_advantage_abs_mean",
        "outcome_collision_outside_base_advantage_rms",
        "outcome_collision_outside_base_advantage_nonzero_fraction",
        "outcome_collision_outside_semantic_advantage_mean",
        "outcome_collision_outside_semantic_advantage_min",
        "outcome_collision_outside_semantic_advantage_max",
        "outcome_collision_outside_semantic_advantage_abs_mean",
        "outcome_collision_outside_semantic_advantage_rms",
        "outcome_collision_outside_semantic_advantage_nonzero_fraction",
        "outcome_collision_outside_combined_advantage_mean",
        "outcome_collision_outside_combined_advantage_abs_mean",
        "outcome_collision_outside_combined_advantage_rms",
        "outcome_collision_outside_combined_advantage_nonzero_fraction",
    )
}
SEMANTIC_SHANNON_SEPARATE_ADVANTAGE_METRICS = {
    short: (
        f"train/{short}",
        f"actor/{short}",
        short,
    )
    for short in (
        "semantic_shannon_separate_advantage_active",
        "semantic_shannon_reward_sent_to_centering_mean",
        "semantic_shannon_separate_predictive_baseline_mean",
        "semantic_shannon_separate_predictive_baseline_min",
        "semantic_shannon_separate_predictive_baseline_max",
        "semantic_shannon_separate_predictive_baseline_normalized_mean",
        "semantic_shannon_separate_predictive_centering_error_max",
        "semantic_shannon_separate_advantage_scale",
        "semantic_shannon_separate_base_advantage_mean",
        "semantic_shannon_separate_base_advantage_abs_mean",
        "semantic_shannon_separate_base_advantage_rms",
        "semantic_shannon_separate_base_advantage_nonzero_fraction",
        "semantic_shannon_separate_semantic_advantage_mean",
        "semantic_shannon_separate_semantic_advantage_min",
        "semantic_shannon_separate_semantic_advantage_max",
        "semantic_shannon_separate_semantic_advantage_abs_mean",
        "semantic_shannon_separate_semantic_advantage_rms",
        "semantic_shannon_separate_semantic_advantage_positive_fraction",
        "semantic_shannon_separate_semantic_advantage_negative_fraction",
        "semantic_shannon_separate_semantic_advantage_zero_fraction",
        "semantic_shannon_separate_semantic_advantage_nonzero_fraction",
        "semantic_shannon_separate_combined_advantage_mean",
        "semantic_shannon_separate_combined_advantage_abs_mean",
        "semantic_shannon_separate_combined_advantage_rms",
        "semantic_shannon_separate_combined_advantage_nonzero_fraction",
    )
}
SEMANTIC_SHANNON_QUALITY_GATED_METRICS = {
    short: (
        f"train/{short}",
        f"actor/{short}",
        short,
    )
    for short in (
        "semantic_shannon_quality_gated_advantage_active",
        "semantic_shannon_quality_gated_raw_all_row_advantage_mean",
        "semantic_shannon_quality_gated_raw_all_row_advantage_min",
        "semantic_shannon_quality_gated_raw_all_row_advantage_max",
        "semantic_shannon_quality_gated_raw_all_row_advantage_abs_mean",
        "semantic_shannon_quality_gated_raw_all_row_advantage_rms",
        "semantic_shannon_quality_gated_raw_all_row_advantage_positive_fraction",
        "semantic_shannon_quality_gated_raw_all_row_advantage_negative_fraction",
        "semantic_shannon_quality_gated_raw_all_row_advantage_zero_fraction",
        "semantic_shannon_quality_gated_effective_advantage_mean",
        "semantic_shannon_quality_gated_effective_advantage_min",
        "semantic_shannon_quality_gated_effective_advantage_max",
        "semantic_shannon_quality_gated_effective_advantage_abs_mean",
        "semantic_shannon_quality_gated_effective_advantage_rms",
        "semantic_shannon_quality_gated_effective_advantage_positive_fraction",
        "semantic_shannon_quality_gated_effective_advantage_zero_fraction",
        "semantic_shannon_quality_gated_eligible_fraction",
        "semantic_shannon_quality_gated_gated_fraction",
        "semantic_shannon_quality_gated_active_fraction",
        "semantic_shannon_quality_gated_reward_positive_fraction",
        "semantic_shannon_quality_gated_parseable_fraction",
        "semantic_shannon_quality_gated_positive_only_zeroed_fraction",
        "semantic_shannon_quality_gated_cap_fraction",
        "semantic_shannon_quality_gated_advantage_cap",
        "semantic_shannon_quality_gated_predictive_baseline_mean",
        "semantic_shannon_quality_gated_predictive_centering_error_max",
        "semantic_shannon_quality_gated_predictive_probability_mean",
        "semantic_shannon_quality_gated_predictive_probability_min",
        "semantic_shannon_quality_gated_predictive_probability_max",
        "semantic_shannon_quality_gated_normalization_error_max",
        "semantic_shannon_quality_gated_history_total_before_mean",
        "semantic_shannon_quality_gated_history_rows_added",
        "semantic_shannon_quality_gated_history_groups_updated",
        "semantic_shannon_quality_gated_history_groups_skipped",
        "semantic_shannon_quality_gated_tracked_prompts",
        "semantic_shannon_quality_gated_tracked_outcomes",
    )
}
SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_METRICS = {
    short: (
        f"train/{short}",
        f"actor/{short}",
        short,
    )
    for short in (
        "semantic_shannon_success_conditioned_signed_advantage_active",
        "semantic_shannon_success_conditioned_signed_raw_eligible_advantage_mean",
        "semantic_shannon_success_conditioned_signed_raw_eligible_advantage_min",
        "semantic_shannon_success_conditioned_signed_raw_eligible_advantage_max",
        "semantic_shannon_success_conditioned_signed_raw_eligible_advantage_abs_mean",
        "semantic_shannon_success_conditioned_signed_raw_eligible_advantage_rms",
        "semantic_shannon_success_conditioned_signed_effective_advantage_mean",
        "semantic_shannon_success_conditioned_signed_effective_advantage_min",
        "semantic_shannon_success_conditioned_signed_effective_advantage_max",
        "semantic_shannon_success_conditioned_signed_effective_advantage_abs_mean",
        "semantic_shannon_success_conditioned_signed_effective_advantage_rms",
        "semantic_shannon_success_conditioned_signed_effective_advantage_positive_fraction",
        "semantic_shannon_success_conditioned_signed_effective_advantage_negative_fraction",
        "semantic_shannon_success_conditioned_signed_effective_advantage_zero_fraction",
        "semantic_shannon_success_conditioned_signed_eligible_fraction",
        "semantic_shannon_success_conditioned_signed_gated_fraction",
        "semantic_shannon_success_conditioned_signed_active_fraction",
        "semantic_shannon_success_conditioned_signed_reward_positive_fraction",
        "semantic_shannon_success_conditioned_signed_parseable_fraction",
        "semantic_shannon_success_conditioned_signed_positive_cap_fraction",
        "semantic_shannon_success_conditioned_signed_negative_cap_fraction",
        "semantic_shannon_success_conditioned_signed_advantage_cap",
        "semantic_shannon_success_conditioned_signed_predictive_baseline_mean",
        "semantic_shannon_success_conditioned_signed_predictive_centering_error_max",
        "semantic_shannon_success_conditioned_signed_predictive_probability_mean",
        "semantic_shannon_success_conditioned_signed_predictive_probability_min",
        "semantic_shannon_success_conditioned_signed_predictive_probability_max",
        "semantic_shannon_success_conditioned_signed_normalization_error_max",
        "semantic_shannon_success_conditioned_signed_history_total_before_mean",
        "semantic_shannon_success_conditioned_signed_history_rows_added",
        "semantic_shannon_success_conditioned_signed_history_groups_updated",
        "semantic_shannon_success_conditioned_signed_history_groups_skipped",
        "semantic_shannon_success_conditioned_signed_tracked_prompts",
        "semantic_shannon_success_conditioned_signed_tracked_outcomes",
        "semantic_shannon_success_conditioned_signed_open_set_inverse_adaptation_active",
        "semantic_shannon_success_conditioned_signed_open_set_coefficient_used",
        "semantic_shannon_success_conditioned_signed_open_set_observed_normalized_entropy",
        "semantic_shannon_success_conditioned_signed_open_set_entropy_ema",
        "semantic_shannon_success_conditioned_signed_open_set_reference_entropy",
        "semantic_shannon_success_conditioned_signed_open_set_inverse_multiplier",
        "semantic_shannon_success_conditioned_signed_open_set_next_coefficient",
        "semantic_shannon_success_conditioned_signed_open_set_observations",
        "semantic_shannon_success_conditioned_signed_open_set_warmup_complete",
        "semantic_shannon_success_conditioned_signed_open_set_observation_skipped",
        "semantic_shannon_success_conditioned_signed_open_set_projection_active",
    )
}
ONLINE_CANONICAL_METRICS = {
    short: (
        f"train/{short}",
        f"actor/{short}",
        short,
    )
    for short in (
        "online_canonical_entropy_estimate_mean",
        "online_canonical_normalized_entropy_mean",
        "online_canonical_normalized_entropy_ratio_mean",
        "online_canonical_normalized_entropy_ratio_eligible_fraction",
        "online_canonical_log_support_mean",
        "online_canonical_entropy_alpha_used",
        "online_canonical_entropy_advantage_mean",
        "online_canonical_entropy_advantage_rms",
        "online_canonical_novelty_advantage_mean",
        "online_canonical_novelty_advantage_rms",
        "online_canonical_combined_advantage_mean",
        "online_canonical_combined_advantage_rms",
        "online_canonical_eligible_fraction",
        "online_canonical_reward_positive_fraction",
        "online_canonical_canonicalizable_correct_fraction",
        "online_canonical_new_outcome_count",
        "online_canonical_new_outcome_row_fraction",
        "online_canonical_bank_size_before_mean",
        "online_canonical_bank_size_after_mean",
        "online_canonical_tracked_prompts",
        "online_canonical_tracked_outcomes",
        "online_canonical_support_at_least_two_prompt_fraction",
        "online_canonical_task_reward_mean",
        "online_canonical_reward_sent_to_centering_mean",
        "online_canonical_separate_base_advantage_mean",
        "online_canonical_separate_base_advantage_rms",
        "online_canonical_separate_combined_advantage_mean",
        "online_canonical_separate_combined_advantage_rms",
        "online_canonical_advantage_applied_after_task_centering",
        "online_canonical_dual_observed_normalized_entropy",
        "online_canonical_dual_normalized_entropy_ema",
        "online_canonical_dual_target_ratio",
        "online_canonical_dual_entropy_error",
        "online_canonical_dual_alpha_loss",
        "online_canonical_dual_alpha_gradient",
        "online_canonical_dual_alpha_before",
        "online_canonical_dual_next_alpha",
        "online_canonical_dual_log_alpha",
        "online_canonical_dual_observations",
        "online_canonical_dual_optimizer_steps",
        "online_canonical_dual_observation_skipped",
        "online_canonical_dual_global_eligibility_weight",
        "online_canonical_policy_entropy_observed",
        "online_canonical_policy_entropy_ema",
        "online_canonical_policy_entropy_reference",
        "online_canonical_policy_entropy_normalized_score",
        "online_canonical_policy_entropy_alpha_before",
        "online_canonical_policy_entropy_next_alpha",
        "online_canonical_policy_entropy_observations",
        "online_canonical_policy_entropy_warmup_complete",
        "canonical_replay_available_groups",
        "canonical_replay_available_modes",
        "canonical_replay_capacity",
        "canonical_replay_global_scheduler_active",
        "canonical_replay_global_groups_per_step",
        "canonical_replay_global_bootstrap_steps",
        "canonical_replay_global_bootstrap_updates",
        "canonical_replay_global_bootstrap_active",
        "canonical_replay_prompt_local_phase_active",
        "canonical_replay_schedule_used_global",
        "canonical_replay_schedule_used_prompt_local",
        "canonical_replay_balance_loss",
        "canonical_replay_actuator_loss",
        "canonical_replay_weighted_loss",
        "canonical_replay_backward_scale",
        "canonical_replay_chunk_size",
        "canonical_replay_score_passes",
        "canonical_replay_normalized_model_entropy",
        "canonical_replay_cross_entropy_excess",
        "canonical_replay_alpha_used",
        "canonical_replay_eligible_groups",
        "canonical_replay_retained_modes",
        "canonical_replay_actuator_groups",
        "canonical_replay_actuator_modes",
        "canonical_replay_reward_estimator_scale",
        "canonical_replay_observed_normalized_entropy",
        "canonical_replay_entropy_ema",
        "canonical_replay_reference_entropy",
        "canonical_replay_inverse_multiplier",
        "canonical_replay_alpha_before",
        "canonical_replay_next_alpha",
        "canonical_replay_observations",
        "canonical_replay_warmup_complete",
        "canonical_replay_projection_active",
        "canonical_replay_observation_skipped",
        "canonical_replay_global_eligibility_weight",
        "canonical_replay_gold_support_feedback",
        "canonical_replay_alpha_projection_active",
        "canonical_replay_score_gradient_sum",
        "canonical_replay_objective_scale",
        "canonical_replay_applied_score_gradient_sum",
        "canonical_replay_verified_likelihood_active",
        "canonical_replay_mass_alpha_before",
        "canonical_replay_mass_alpha_used",
        "canonical_replay_mass_observed_surprisal",
        "canonical_replay_mass_surprisal_ema",
        "canonical_replay_mass_surprisal_reference",
        "canonical_replay_mass_inverse_multiplier",
        "canonical_replay_mass_next_alpha",
        "canonical_replay_mass_observations",
        "canonical_replay_mass_warmup_complete",
        "canonical_replay_mass_observation_skipped",
        "canonical_replay_mass_projection_active",
        "canonical_replay_mass_score_gradient_sum",
        "canonical_replay_balance_score_gradient_sum",
        "canonical_replay_mass_score_gradient_l2",
        "canonical_replay_balance_score_gradient_l2",
        "canonical_replay_applied_score_gradient_l2",
        "canonical_replay_balance_alpha_used",
        "maxent_conditional_token_entropy",
        "maxent_entropy_loss",
        "maxent_alpha_used",
        "maxent_inverse_observed_entropy",
        "maxent_inverse_entropy_ema",
        "maxent_inverse_multiplier",
        "maxent_inverse_alpha_before",
        "maxent_inverse_next_alpha",
        "maxent_inverse_observations",
        "maxent_inverse_warmup_complete",
        "maxent_inverse_projection_active",
        "maxent_inverse_reference_entropy",
        "math_strategy_judge_calls",
        "math_strategy_validator_positive_rows",
        "math_strategy_accepted_rows",
        "math_strategy_rejected_integrity_rows",
        "math_strategy_rejected_ambiguous_rows",
        "math_strategy_rejected_disagreement_rows",
        "math_strategy_matched_existing_rows",
        "math_strategy_new_strategy_rows",
        "math_strategy_new_strategy_count",
        "math_strategy_judge_format_failure_rows",
        "math_strategy_rejected_contract_rows",
        "math_strategy_inferred_unstructured_rows",
        "math_strategy_rejected_strategy_inference_rows",
        "math_strategy_raw_task_reward_mean",
        "math_strategy_gated_task_reward_mean",
        "math_strategy_task_reward_gate_active",
    )
}
VERIFIED_DISCOVERY_METRICS = {
    short: (
        f"train/{short}",
        f"actor/{short}",
        short,
    )
    for short in (
        "verified_discovery_cumulative_outcomes",
        "verified_discovery_tracked_prompts",
        "verified_discovery_mean_support_per_prompt",
    )
}
MECHANISM_METRICS = {
    **OUTCOME_COLLISION_METRICS,
    **SEMANTIC_SHANNON_METRICS,
    **OUTCOME_COLLISION_OUTSIDE_CENTERING_METRICS,
    **SEMANTIC_SHANNON_SEPARATE_ADVANTAGE_METRICS,
    **SEMANTIC_SHANNON_QUALITY_GATED_METRICS,
    **SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_METRICS,
    **ONLINE_CANONICAL_METRICS,
    **VERIFIED_DISCOVERY_METRICS,
}


def _first_present(record, aliases):
    for key in aliases:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _add_online_canonical_derived_metrics(row):
    tracked_outcomes = row.get("verified_discovery_cumulative_outcomes")
    if tracked_outcomes is None:
        tracked_outcomes = row.get("online_canonical_tracked_outcomes")
    tracked_prompts = row.get("verified_discovery_tracked_prompts")
    if tracked_prompts is None:
        tracked_prompts = row.get("online_canonical_tracked_prompts")
    row["verified_discovery_cumulative_outcomes"] = tracked_outcomes
    row["verified_discovery_tracked_prompts"] = tracked_prompts
    if row.get("online_canonical_tracked_outcomes") is None:
        row["online_canonical_tracked_outcomes"] = tracked_outcomes
    if row.get("online_canonical_tracked_prompts") is None:
        row["online_canonical_tracked_prompts"] = tracked_prompts

    mean_support = row.get("verified_discovery_mean_support_per_prompt")
    if (
        mean_support is None
        and tracked_outcomes is not None
        and tracked_prompts is not None
        and float(tracked_prompts) > 0.0
    ):
        mean_support = (
            float(tracked_outcomes) / float(tracked_prompts)
        )
    row["verified_discovery_mean_support_per_prompt"] = mean_support
    row["online_canonical_mean_support_per_prompt"] = mean_support

    exploration_rms = row.get("online_canonical_combined_advantage_rms")
    task_rms = row.get("online_canonical_separate_base_advantage_rms")
    if (
        exploration_rms is not None
        and task_rms is not None
        and float(task_rms) > 0.0
    ):
        row["online_canonical_exploration_to_task_rms_ratio"] = (
            float(exploration_rms) / float(task_rms)
        )
    else:
        row["online_canonical_exploration_to_task_rms_ratio"] = None


def _parse_jsonl(
    path,
    prompt_pool_size=None,
    num_samples=None,
    eval_splits=SPLITS,
):
    rows = []
    seen = set()
    signed_records = []
    online_canonical_records = []
    for record_index, line in enumerate(open(path)):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if (
            _first_present(
                rec,
                SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_METRICS[
                    "semantic_shannon_success_conditioned_signed_advantage_active"
                ],
            )
            is not None
        ):
            signed_step = int(
                rec.get(
                    "trainer/global_step",
                    rec.get("misc/global_step", -1),
                )
            )
            if signed_step >= 0:
                signed_records.append((signed_step, record_index, rec))
        if (
            _first_present(
                rec,
                VERIFIED_DISCOVERY_METRICS[
                    "verified_discovery_cumulative_outcomes"
                ],
            )
            is not None
            or _first_present(
                rec,
                ONLINE_CANONICAL_METRICS[
                    "online_canonical_advantage_applied_after_task_centering"
                ],
            )
            is not None
        ):
            online_canonical_step = int(
                rec.get(
                    "trainer/global_step",
                    rec.get("misc/global_step", -1),
                )
            )
            if online_canonical_step >= 0:
                online_canonical_records.append(
                    (online_canonical_step, record_index, rec)
                )
        present_splits = tuple(
            split
            for split in eval_splits
            if any(key.startswith(f"eval/{split}/") for key in rec)
        )
        if not present_splits:
            continue
        step = int(rec.get("trainer/global_step", rec.get("misc/global_step", -1)))
        epoch = rec.get("misc/prompt_epoch")
        for split in present_splits:
            key = (step, split)
            # The training loop can emit a scheduled checkpoint evaluation and
            # then a second terminal evaluation at the same final step. Keep
            # the scheduled draw so every curve point has the same one-draw
            # interpretation and the terminal policy is not double-weighted.
            if key in seen:
                continue
            seen.add(key)
            prompt_consumed = rec.get("misc/prompt_consumed")
            training_passes = None
            if (
                prompt_consumed is not None
                and prompt_pool_size is not None
                and num_samples is not None
            ):
                training_passes = float(prompt_consumed) / (
                    prompt_pool_size * num_samples
                )
            row = {
                "step": step,
                "epoch": epoch,
                "prompt_consumed": prompt_consumed,
                "training_passes": training_passes,
                "split": split,
            }
            for short, key in METRICS.items():
                metric_key = f"eval/{split}/{key}"
                row[short] = rec.get(metric_key)
                raw_draws = []
                draw_index = 0
                while f"{metric_key}_draw_{draw_index}" in rec:
                    raw_draws.append(rec[f"{metric_key}_draw_{draw_index}"])
                    draw_index += 1
                row[f"{short}_draws"] = raw_draws
                for suffix in ("draw_std", "draw_se", "draw_min", "draw_max"):
                    row[f"{short}_{suffix}"] = rec.get(f"{metric_key}_{suffix}")
            for short, aliases in MECHANISM_METRICS.items():
                row[short] = _first_present(rec, aliases)
            rows.append(row)

    # Optimizer records and evaluation records are not guaranteed to be the
    # same JSONL entry. Bind every persisted evaluation to the newest E43
    # telemetry at or before its step so historical mechanism points remain
    # visible as training advances.
    for row in rows:
        eligible_records = [
            signed_record
            for signed_record in signed_records
            if signed_record[0] <= row["step"]
        ]
        if not eligible_records:
            continue
        _, _, telemetry_record = max(
            eligible_records,
            key=lambda signed_record: signed_record[:2],
        )
        for short, aliases in (
            SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_METRICS.items()
        ):
            value = _first_present(telemetry_record, aliases)
            if value is not None:
                row[short] = value

    # Verified-bank telemetry is emitted on optimizer records by both passive
    # Dr.GRPO tracking and active online-canonical treatments. Bind each
    # persisted evaluation to the newest bank snapshot at or before that step.
    for row in rows:
        eligible_records = [
            online_record
            for online_record in online_canonical_records
            if online_record[0] <= row["step"]
        ]
        if not eligible_records:
            continue
        _, _, telemetry_record = max(
            eligible_records,
            key=lambda online_record: online_record[:2],
        )
        for short, aliases in ONLINE_CANONICAL_METRICS.items():
            value = _first_present(telemetry_record, aliases)
            if value is not None:
                row[short] = value
        for short, aliases in VERIFIED_DISCOVERY_METRICS.items():
            value = _first_present(telemetry_record, aliases)
            if value is not None:
                row[short] = value

    if signed_records:
        step, _, rec = max(
            signed_records,
            key=lambda signed_record: signed_record[:2],
        )
        prompt_consumed = rec.get("misc/prompt_consumed")
        training_passes = None
        if (
            prompt_consumed is not None
            and prompt_pool_size is not None
            and num_samples is not None
        ):
                training_passes = float(prompt_consumed) / (
                    prompt_pool_size * num_samples
                )
        for split in eval_splits:
            latest_eval_step = max(
                (
                    row["step"]
                    for row in rows
                    if row["split"] == split
                ),
                default=None,
            )
            if latest_eval_step is not None and step <= latest_eval_step:
                continue
            row = {
                "step": step,
                "epoch": rec.get("misc/prompt_epoch"),
                "prompt_consumed": prompt_consumed,
                "training_passes": training_passes,
                "split": split,
                "mechanism_only": True,
            }
            for short in METRICS:
                row[short] = None
                row[f"{short}_draws"] = []
                for suffix in (
                    "draw_std",
                    "draw_se",
                    "draw_min",
                    "draw_max",
                ):
                    row[f"{short}_{suffix}"] = None
            for short, aliases in MECHANISM_METRICS.items():
                row[short] = _first_present(rec, aliases)
            rows.append(row)
    if online_canonical_records:
        step, _, rec = max(
            online_canonical_records,
            key=lambda online_record: online_record[:2],
        )
        prompt_consumed = rec.get("misc/prompt_consumed")
        training_passes = None
        if (
            prompt_consumed is not None
            and prompt_pool_size is not None
            and num_samples is not None
        ):
            training_passes = float(prompt_consumed) / (
                prompt_pool_size * num_samples
            )
        for split in eval_splits:
            latest_eval_step = max(
                (
                    row["step"]
                    for row in rows
                    if row["split"] == split
                    and not row.get("mechanism_only", False)
                ),
                default=None,
            )
            if latest_eval_step is not None and step <= latest_eval_step:
                continue
            row = {
                "step": step,
                "epoch": rec.get("misc/prompt_epoch"),
                "prompt_consumed": prompt_consumed,
                "training_passes": training_passes,
                "split": split,
                "mechanism_only": True,
            }
            for short in METRICS:
                row[short] = None
                row[f"{short}_draws"] = []
                for suffix in (
                    "draw_std",
                    "draw_se",
                    "draw_min",
                    "draw_max",
                ):
                    row[f"{short}_{suffix}"] = None
            for short, aliases in MECHANISM_METRICS.items():
                row[short] = _first_present(rec, aliases)
            rows.append(row)
    for row in rows:
        _add_online_canonical_derived_metrics(row)
    split_order = {split: index for index, split in enumerate(eval_splits)}
    rows.sort(key=lambda row: (row["step"], split_order[row["split"]]))
    return rows


def parse_run(
    run_dir,
    prompt_pool_size=None,
    num_samples=None,
    max_training_passes=None,
    eval_splits=SPLITS,
    allowed_debug_dir=None,
):
    """Per-eval-step rows plus E43's latest live mechanism row.

    A run dir can hold several debug_<ts> attempts (e.g. a short mis-budgeted
    attempt and the corrected long rerun). Fresh attempts (first eval at step
    zero) remain separate trajectories. A checkpoint-resumed attempt (first
    eval above zero) is stitched onto a predecessor through its resume
    boundary. When both attempts evaluated that boundary, retain the
    predecessor's evaluation and use the resumed attempt only after it. Among
    the resulting coherent chains, use the one that got furthest (largest max
    step; ties -> latest timestamp). When a plotted
    training-pass horizon is supplied, prefer the chain with the furthest
    evaluation inside that horizon. This prevents an obsolete over-budget
    attempt from hiding a later, valid terminal evaluation.

    This preserves the original pre-checkpoint curve while avoiding a false
    splice through the abandoned post-checkpoint branch. Until a resume chain
    overtakes the crashed attempt, the longer crashed attempt remains visible.
    E43 emits signed-mechanism telemetry on optimizer records rather than
    evaluation records. Each evaluation receives the newest signed telemetry
    at or before its step; when training has advanced beyond the last
    evaluation, the latest telemetry is also retained as a ``mechanism_only``
    row. Quality metrics remain evaluation-only.
    """
    eval_splits = tuple(dict.fromkeys(eval_splits))
    if not eval_splits or any(not split for split in eval_splits):
        raise ValueError("eval_splits must contain at least one non-empty split")
    chains = []
    metric_paths = glob.glob(
        os.path.join(run_dir, "debug_*", "train_metrics.jsonl")
    )
    if allowed_debug_dir is not None:
        metric_paths = [
            path
            for path in metric_paths
            if os.path.basename(os.path.dirname(path)) == allowed_debug_dir
        ]
    for path in sorted(metric_paths):
        rows = _parse_jsonl(
            path,
            prompt_pool_size,
            num_samples,
            eval_splits,
        )
        if not rows:
            continue
        first_step = min(r["step"] for r in rows)
        if first_step <= 0:
            chain_rows = rows
        else:
            predecessors = [
                chain
                for chain in chains
                if max(r["step"] for r in chain["rows"]) >= first_step
            ]
            if predecessors:
                predecessor = max(
                    predecessors,
                    key=lambda chain: (
                        max(r["step"] for r in chain["rows"]),
                        chain["path"],
                    ),
                )
                predecessor_prefix = [
                    r for r in predecessor["rows"] if r["step"] <= first_step
                ]
                predecessor_boundary_keys = {
                    (r["step"], r["split"])
                    for r in predecessor_prefix
                    if r["step"] == first_step
                }
                resumed_suffix = [
                    r
                    for r in rows
                    if (r["step"], r["split"])
                    not in predecessor_boundary_keys
                ]
                chain_rows = predecessor_prefix + resumed_suffix
            else:
                chain_rows = rows
        chains.append({"path": path, "rows": chain_rows})

    if not chains:
        return []

    def chain_score(chain):
        if max_training_passes is not None:
            plotted_passes = [
                r["training_passes"]
                for r in chain["rows"]
                if r["training_passes"] is not None
                and r["training_passes"] <= max_training_passes + 1e-9
            ]
            if plotted_passes:
                return (max(plotted_passes), chain["path"])
        return (max(r["step"] for r in chain["rows"]), chain["path"])

    best = max(chains, key=chain_score)
    return best["rows"]


def arm_seed_from_dir(run_dir, stamp):
    # run_dir basename ends with ..._<stamp>_<arm>_s<seed>
    tail = os.path.basename(run_dir).split(f"{stamp}_", 1)[-1]
    m = re.match(r"(?P<arm>.+)_s(?P<seed>\d+)$", tail)
    return (m.group("arm"), int(m.group("seed"))) if m else (tail, -1)


def attempts_from_manifest(path):
    attempts = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            run_stamp = row.get("run_stamp", "")
            job_id = row.get("job_id", "")
            if not run_stamp or not job_id.isdigit():
                raise ValueError(f"invalid manifest row in {path}: {row!r}")
            attempt = f"debug_job{job_id}"
            if run_stamp in attempts and attempts[run_stamp] != attempt:
                raise ValueError(f"ambiguous manifest run stamp {run_stamp}")
            attempts[run_stamp] = attempt
    return attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp-prefix", required=True)
    ap.add_argument("--run-data-root", default="var/data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompt-pool-size", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--max-training-passes", type=float, default=None)
    ap.add_argument(
        "--jobs-manifest",
        default=None,
        help=(
            "optional comparative_jobs.tsv; when supplied, read only the "
            "debug_job<ID> attempt bound to each manifest row"
        ),
    )
    ap.add_argument(
        "--eval-splits",
        nargs="+",
        default=list(SPLITS),
        help=(
            "evaluation split names to extract "
            "(default: multi_answer unique_answer)"
        ),
    )
    args = ap.parse_args()
    eval_splits = tuple(dict.fromkeys(args.eval_splits))
    manifest_attempts = (
        attempts_from_manifest(args.jobs_manifest)
        if args.jobs_manifest is not None
        else None
    )

    # run dirs use the oat_zero_tiny_* prefix before the 2026-07-16 ops refactor
    # and the xdr_* prefix after it; match both so old and new stamps parse.
    run_dirs = sorted(
        set(
            glob.glob(
                os.path.join(
                    args.run_data_root,
                    f"oat_zero_tiny_*_{args.stamp_prefix}_*",
                )
            )
        )
        | set(
            glob.glob(
                os.path.join(
                    args.run_data_root,
                    f"xdr_*_{args.stamp_prefix}_*",
                )
            )
        )
    )
    tidy = []
    per_arm_cov = defaultdict(dict)  # (arm) -> {step: [coverage over seeds]}
    for rd in run_dirs:
        arm, seed = arm_seed_from_dir(rd, args.stamp_prefix)
        run_stamp = f"{args.stamp_prefix}_{arm}_s{seed}"
        allowed_debug_dir = None
        if manifest_attempts is not None:
            if run_stamp not in manifest_attempts:
                raise ValueError(
                    f"{run_stamp} is absent from {args.jobs_manifest}"
                )
            allowed_debug_dir = manifest_attempts[run_stamp]
        rows = parse_run(
            rd,
            args.prompt_pool_size,
            args.num_samples,
            args.max_training_passes,
            eval_splits,
            allowed_debug_dir,
        )
        for r in rows:
            r2 = {"arm": arm, "seed": seed, **r}
            tidy.append(r2)
            if r["split"] == "multi_answer" and r["coverage8"] is not None:
                per_arm_cov[arm].setdefault(r["step"], []).append(r["coverage8"])
        eval_points = len(
            {
                row["step"]
                for row in rows
                if not row.get("mechanism_only", False)
            }
        )
        mechanism_live_steps = [
            row["step"] for row in rows if row.get("mechanism_only", False)
        ]
        mechanism_note = (
            f"; mechanism live step {max(mechanism_live_steps)}"
            if mechanism_live_steps
            else ""
        )
        print(
            f"[parse] {arm} s{seed}: {eval_points} eval points "
            f"(last step {rows[-1]['step'] if rows else 'NA'}"
            f"{mechanism_note})"
        )

    out = args.out or os.path.join(
        "var/artifacts",
        f"{args.stamp_prefix}_scaling_curve.json",
    )
    json.dump(tidy, open(out, "w"), indent=2)
    print(f"[parse] wrote {len(tidy)} rows -> {out}")

    # Divergence preview: mean multi-answer coverage@8 and each xDr arm's own
    # Dr.GRPO contrast. Multiple xDr treatments (for example fixed and
    # entropy-feedback tau) must not silently overwrite one another's gap.
    arms = sorted(per_arm_cov)
    if {"grpo"}.issubset(set(arms)):
        xdr_arms = [a for a in arms if a.startswith("xdr")]
        steps = sorted(set(s for a in arms for s in per_arm_cov[a]))
        print(
            "\n[divergence] multi_answer coverage@8 (mean over seeds), "
            "and each xDr-Dr.GRPO gap:"
        )
        value_header = " ".join(f"{a:>16}" for a in xdr_arms)
        gap_header = " ".join(f"{'gap:' + a:>16}" for a in xdr_arms)
        print(f"{'step':>6} {'grpo':>8} {value_header} {gap_header}")
        for s in steps:
            g = per_arm_cov.get("grpo", {}).get(s)
            gm = sum(g) / len(g) if g else None
            cells = []
            gaps = []
            for a in xdr_arms:
                v = per_arm_cov[a].get(s)
                vm = sum(v) / len(v) if v else None
                cells.append(f"{vm:>16.3f}" if vm is not None else f"{'-':>16}")
                gap = vm - gm if vm is not None and gm is not None else None
                gaps.append(f"{gap:>+16.3f}" if gap is not None else f"{'-':>16}")
            gstr = f"{gm:>8.3f}" if gm is not None else f"{'-':>8}"
            print(f"{s:>6} {gstr} " + " ".join(cells + gaps))


if __name__ == "__main__":
    main()
