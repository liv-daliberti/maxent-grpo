"""Small tensor-stat helpers shared by learner code."""

from __future__ import annotations

import torch
import torch.distributed as dist


def compute_correctness_group_rate_infos(
    *,
    correctness_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
    threshold: float = 0.999,
) -> dict[str, torch.Tensor]:
    """Return correctness composition rates over contributing prompt groups."""

    if correctness_grouped.shape != valid_row_mask_grouped.shape:
        raise ValueError("correctness_grouped must match valid_row_mask_grouped.")

    valid_mask = valid_row_mask_grouped.to(torch.bool)
    contributing_group_mask = valid_mask.any(dim=1)
    valid_count = valid_mask.to(torch.int64).sum(dim=1)
    correct_mask = (
        correctness_grouped.to(device=valid_mask.device) >= float(threshold)
    ) & valid_mask
    correct_count = correct_mask.to(torch.int64).sum(dim=1)
    incorrect_count = valid_count - correct_count

    all_wrong_group_mask = contributing_group_mask & (correct_count == 0)
    all_correct_group_mask = contributing_group_mask & (correct_count == valid_count)
    mixed_group_mask = (
        contributing_group_mask & (correct_count > 0) & (incorrect_count > 0)
    )
    any_correct_group_mask = contributing_group_mask & (correct_count > 0)

    denom = contributing_group_mask.to(torch.float32).sum().clamp(min=1.0)
    return {
        "listwise_all_wrong_group_frac": (
            all_wrong_group_mask.to(torch.float32).sum() / denom
        ),
        "listwise_mixed_correctness_group_frac": (
            mixed_group_mask.to(torch.float32).sum() / denom
        ),
        "listwise_all_correct_group_frac": (
            all_correct_group_mask.to(torch.float32).sum() / denom
        ),
        "listwise_any_correct_group_frac": (
            any_correct_group_mask.to(torch.float32).sum() / denom
        ),
    }


LISTWISE_OPTIONAL_LOGGING_METRIC_DEFAULTS: dict[str, float] = {
    "listwise_all_wrong_group_frac": 0.0,
    "listwise_mixed_correctness_group_frac": 0.0,
    "listwise_all_correct_group_frac": 0.0,
    "listwise_any_correct_group_frac": 0.0,
    "listwise_clip_reward_mass_mean": 0.0,
    "listwise_grad_probe_enabled": 0.0,
    "listwise_grad_probe_update_index": 0.0,
    "listwise_grad_probe_valid": 0.0,
    "listwise_grad_token_norm": 0.0,
    "listwise_grad_sequence_norm": 0.0,
    "listwise_grad_combined_norm": 0.0,
    "listwise_grad_ratio_unscaled": 0.0,
    "listwise_grad_ratio_scaled": 0.0,
    "listwise_grad_cosine": 0.0,
    "listwise_projection_ce_loss_effective": 0.0,
    "listwise_drgrpo_token_active_row_count_global": 0.0,
    "listwise_drgrpo_token_active_row_frac": 0.0,
    "listwise_drgrpo_token_advantage_source_utility_centered": 0.0,
    "listwise_drgrpo_token_advantage_source_maxent_centered": 0.0,
    "listwise_drgrpo_token_length_normalizer_response": 0.0,
    "listwise_semantic_cluster_entropy": 0.0,
    "listwise_semantic_cluster_entropy_norm": 0.0,
    "listwise_semantic_cluster_count": 0.0,
    "listwise_exact_quality_mean": 0.0,
    "listwise_exact_semantic_surprisal_mean": 0.0,
    "listwise_exact_semantic_gate_mean": 0.0,
    "listwise_exact_semantic_piece_mean": 0.0,
    "listwise_exact_correctness_adv_abs_mean": 0.0,
    "listwise_exact_semantic_adv_abs_mean": 0.0,
    "listwise_exact_semantic_to_correctness_adv_ratio": 0.0,
    "listwise_exact_semantic_adv_fraction": 0.0,
    "listwise_exact_semantic_adv_abs_mean_all_wrong": 0.0,
    "listwise_exact_semantic_adv_abs_mean_mixed": 0.0,
    "listwise_exact_semantic_adv_abs_mean_all_correct": 0.0,
    "listwise_exact_semantic_effective_group_frac": 0.0,
    "listwise_exact_semantic_correctness_adv_cosine": 0.0,
    "listwise_exact_utility_mean": 0.0,
    "listwise_all_zero_rewards_count": 0.0,
    "listwise_all_one_rewards_count": 0.0,
    "listwise_semantic_competitive_mode_count": 0.0,
    "listwise_semantic_competitive_mode_eligible_count": 0.0,
    "listwise_semantic_competitive_mode_eligible_frac": 0.0,
    "listwise_semantic_distinct_correct_mode_count": 0.0,
    "listwise_semantic_distinct_correct_mode_frac": 0.0,
    "listwise_semantic_distinct_correct_answer_count": 0.0,
    "listwise_semantic_correctness_min_answer_rejected_frac": 0.0,
    "listwise_semantic_competitive_mode_best_score": 0.0,
    "listwise_semantic_competitive_mode_second_score": 0.0,
    "listwise_semantic_competitive_mode_gap": 0.0,
    "listwise_semantic_answer_key_extracted_frac": 0.0,
    "listwise_semantic_trace_extracted_frac": 0.0,
    "listwise_semantic_signature_extracted_frac": 0.0,
    "listwise_semantic_cluster_valid_frac": 0.0,
    "listwise_semantic_trace_truncated_frac": 0.0,
    "listwise_semantic_cluster_entropy_ref": 0.0,
    "listwise_semantic_cluster_entropy_gain": 0.0,
    "listwise_semantic_exploration_gain_any_correct": 0.0,
    "listwise_semantic_exploration_gain_drgrpo": 0.0,
    "listwise_semantic_exploration_gain_any_correct_corr": 0.0,
    "listwise_semantic_exploration_gain_drgrpo_corr": 0.0,
    "listwise_semantic_exploration_prompt_count": 0.0,
    "listwise_semantic_exploration_entropy_std": 0.0,
    "listwise_semantic_exploration_gain_any_correct_std": 0.0,
    "listwise_semantic_exploration_gain_drgrpo_std": 0.0,
    "listwise_semantic_exploration_gain_any_correct_low_entropy": 0.0,
    "listwise_semantic_exploration_gain_any_correct_high_entropy": 0.0,
    "listwise_semantic_exploration_gain_any_correct_high_minus_low": 0.0,
    "listwise_semantic_exploration_gain_drgrpo_low_entropy": 0.0,
    "listwise_semantic_exploration_gain_drgrpo_high_entropy": 0.0,
    "listwise_semantic_exploration_gain_drgrpo_high_minus_low": 0.0,
    "listwise_semantic_cluster_ref_available": 0.0,
    "listwise_semantic_explore_budget_mean": 0.0,
    "listwise_semantic_explore_budget_saturated_frac": 0.0,
    "listwise_semantic_explore_applied_group_frac": 0.0,
    "listwise_semantic_prompt_selected_frac": 0.0,
    "listwise_semantic_prompt_rejected_low_opp_frac": 0.0,
    "listwise_semantic_prompt_rejected_nonpositive_frac": 0.0,
    "listwise_semantic_prompt_rejected_len_guard_frac": 0.0,
    "listwise_semantic_prompt_rejected_format_guard_frac": 0.0,
    "listwise_semantic_moved_mass_l1": 0.0,
    "listwise_semantic_alpha_raw_mean": 0.0,
    "listwise_semantic_alpha_applied_mean": 0.0,
    "listwise_semantic_expected_utility_q": 0.0,
    "listwise_semantic_expected_utility_explore_target": 0.0,
    "listwise_semantic_expected_utility_final_w": 0.0,
    "listwise_semantic_expected_len_q": 0.0,
    "listwise_semantic_expected_len_explore_target": 0.0,
    "listwise_semantic_expected_len_final_w": 0.0,
    "listwise_semantic_expected_format_q": 0.0,
    "listwise_semantic_expected_format_explore_target": 0.0,
    "listwise_semantic_expected_format_final_w": 0.0,
    "listwise_sequence_aux_eligible_group_frac": 0.0,
    "listwise_sequence_aux_kept_group_frac": 0.0,
    "listwise_sequence_aux_kept_frac_of_eligible": 0.0,
    "listwise_sequence_aux_rejected_group_filter_frac": 0.0,
    "listwise_sequence_aux_rejected_len_guard_frac": 0.0,
    "listwise_sequence_aux_rejected_len_gain_guard_frac": 0.0,
    "listwise_sequence_aux_rejected_format_guard_frac": 0.0,
    "listwise_sequence_aux_rejected_correctness_guard_frac": 0.0,
    "listwise_sequence_aux_all_wrong_group_frac": 0.0,
    "listwise_sequence_aux_mixed_group_frac": 0.0,
    "listwise_sequence_aux_all_correct_group_frac": 0.0,
    "listwise_sequence_aux_target_mass_all_wrong_frac": 0.0,
    "listwise_sequence_aux_target_mass_mixed_frac": 0.0,
    "listwise_sequence_aux_target_mass_all_correct_frac": 0.0,
    "listwise_sequence_aux_target_expected_len": 0.0,
    "listwise_sequence_aux_behavior_expected_len": 0.0,
    "listwise_sequence_aux_expected_len_delta": 0.0,
    "listwise_sequence_aux_target_expected_format": 0.0,
    "listwise_sequence_aux_behavior_expected_format": 0.0,
    "listwise_sequence_aux_expected_format_delta": 0.0,
    "listwise_sequence_aux_target_expected_correctness": 0.0,
    "listwise_sequence_aux_behavior_expected_correctness": 0.0,
    "listwise_sequence_aux_expected_correctness_delta": 0.0,
    "listwise_sequence_aux_correct_mass_delta": 0.0,
    "listwise_sequence_aux_wrong_mass_delta": 0.0,
    "listwise_sequence_aux_moved_mass_l1": 0.0,
    "listwise_sequence_aux_moved_mass_to_correct": 0.0,
    "listwise_sequence_aux_moved_mass_from_correct": 0.0,
    "listwise_sequence_aux_moved_mass_to_wrong": 0.0,
    "listwise_sequence_aux_moved_mass_from_wrong": 0.0,
    "listwise_projection_to_pg_loss_ratio": 0.0,
    "listwise_tau_target_metric": 0.0,
    "listwise_tau_metric_ema": 0.0,
    "listwise_tau_adaptation_metric_value": 0.0,
    "listwise_tau_adaptation_metric_is_semantic_entropy_mu": 0.0,
    "listwise_tau_adaptation_metric_is_exploration_gain_any_correct": 0.0,
    "listwise_tau_adaptation_metric_is_exploration_gain_drgrpo": 0.0,
    "listwise_tau_signal_semantic_entropy_mu": 0.0,
    "listwise_tau_signal_exploration_gain_any_correct": 0.0,
    "listwise_tau_signal_exploration_gain_drgrpo": 0.0,
    "listwise_semantic_correctness_schedule_any_correct_batch_mean": 0.0,
    "listwise_semantic_correctness_schedule_any_correct_ema": 0.0,
    "listwise_semantic_correctness_schedule_exploration_level": 0.0,
    "listwise_semantic_correctness_schedule_consolidation_level": 0.0,
    "listwise_semantic_correctness_schedule_budget_max": 0.0,
    "listwise_semantic_correctness_schedule_prompt_select_min_alpha_frac": 0.0,
    "listwise_semantic_correctness_schedule_mode_tau": 0.0,
    "listwise_semantic_correctness_schedule_intra_tau": 0.0,
}


LISTWISE_MEAN_INFO_STAT_KEYS: tuple[str, ...] = (
    "listwise_semantic_cluster_count",
    "listwise_exact_quality_mean",
    "listwise_exact_semantic_surprisal_mean",
    "listwise_exact_semantic_gate_mean",
    "listwise_exact_semantic_piece_mean",
    "listwise_exact_correctness_adv_abs_mean",
    "listwise_exact_semantic_adv_abs_mean",
    "listwise_exact_semantic_to_correctness_adv_ratio",
    "listwise_exact_semantic_adv_fraction",
    "listwise_exact_semantic_adv_abs_mean_all_wrong",
    "listwise_exact_semantic_adv_abs_mean_mixed",
    "listwise_exact_semantic_adv_abs_mean_all_correct",
    "listwise_exact_semantic_effective_group_frac",
    "listwise_exact_semantic_correctness_adv_cosine",
    "listwise_exact_utility_mean",
    "listwise_semantic_competitive_mode_count",
    "listwise_semantic_competitive_mode_eligible_count",
    "listwise_semantic_competitive_mode_eligible_frac",
    "listwise_semantic_distinct_correct_mode_count",
    "listwise_semantic_distinct_correct_mode_frac",
    "listwise_semantic_distinct_correct_answer_count",
    "listwise_semantic_correctness_min_answer_rejected_frac",
    "listwise_semantic_competitive_mode_best_score",
    "listwise_semantic_competitive_mode_second_score",
    "listwise_semantic_competitive_mode_gap",
    "listwise_semantic_explore_budget_mean",
    "listwise_semantic_explore_budget_saturated_frac",
    "listwise_semantic_explore_applied_group_frac",
    "listwise_semantic_prompt_selected_frac",
    "listwise_semantic_prompt_rejected_low_opp_frac",
    "listwise_semantic_prompt_rejected_nonpositive_frac",
    "listwise_semantic_prompt_rejected_len_guard_frac",
    "listwise_semantic_prompt_rejected_format_guard_frac",
    "listwise_semantic_moved_mass_l1",
    "listwise_semantic_alpha_raw_mean",
    "listwise_semantic_alpha_applied_mean",
    "listwise_semantic_correctness_schedule_any_correct_batch_mean",
    "listwise_semantic_correctness_schedule_any_correct_ema",
    "listwise_semantic_correctness_schedule_exploration_level",
    "listwise_semantic_correctness_schedule_consolidation_level",
    "listwise_semantic_correctness_schedule_budget_max",
    "listwise_semantic_correctness_schedule_prompt_select_min_alpha_frac",
    "listwise_semantic_correctness_schedule_mode_tau",
    "listwise_semantic_correctness_schedule_intra_tau",
    "listwise_semantic_expected_utility_q",
    "listwise_semantic_expected_utility_explore_target",
    "listwise_semantic_expected_utility_final_w",
    "listwise_semantic_expected_len_q",
    "listwise_semantic_expected_len_explore_target",
    "listwise_semantic_expected_len_final_w",
    "listwise_semantic_expected_format_q",
    "listwise_semantic_expected_format_explore_target",
    "listwise_semantic_expected_format_final_w",
    "listwise_sequence_aux_eligible_group_frac",
    "listwise_sequence_aux_kept_group_frac",
    "listwise_sequence_aux_kept_frac_of_eligible",
    "listwise_sequence_aux_rejected_group_filter_frac",
    "listwise_sequence_aux_rejected_len_guard_frac",
    "listwise_sequence_aux_rejected_len_gain_guard_frac",
    "listwise_sequence_aux_rejected_format_guard_frac",
    "listwise_sequence_aux_rejected_correctness_guard_frac",
    "listwise_sequence_aux_all_wrong_group_frac",
    "listwise_sequence_aux_mixed_group_frac",
    "listwise_sequence_aux_all_correct_group_frac",
    "listwise_sequence_aux_target_mass_all_wrong_frac",
    "listwise_sequence_aux_target_mass_mixed_frac",
    "listwise_sequence_aux_target_mass_all_correct_frac",
    "listwise_sequence_aux_target_expected_len",
    "listwise_sequence_aux_behavior_expected_len",
    "listwise_sequence_aux_expected_len_delta",
    "listwise_sequence_aux_target_expected_format",
    "listwise_sequence_aux_behavior_expected_format",
    "listwise_sequence_aux_expected_format_delta",
    "listwise_sequence_aux_target_expected_correctness",
    "listwise_sequence_aux_behavior_expected_correctness",
    "listwise_sequence_aux_expected_correctness_delta",
    "listwise_sequence_aux_correct_mass_delta",
    "listwise_sequence_aux_wrong_mass_delta",
    "listwise_sequence_aux_moved_mass_l1",
    "listwise_sequence_aux_moved_mass_to_correct",
    "listwise_sequence_aux_moved_mass_from_correct",
    "listwise_sequence_aux_moved_mass_to_wrong",
    "listwise_sequence_aux_moved_mass_from_wrong",
    "listwise_semantic_answer_key_extracted_frac",
    "listwise_semantic_trace_extracted_frac",
    "listwise_semantic_signature_extracted_frac",
    "listwise_semantic_cluster_valid_frac",
    "listwise_semantic_trace_truncated_frac",
    "listwise_semantic_cluster_entropy_ref",
    "listwise_semantic_cluster_entropy_gain",
    "listwise_semantic_exploration_gain_any_correct",
    "listwise_semantic_exploration_gain_drgrpo",
    "listwise_semantic_cluster_ref_available",
)


def stack_scalar_stats(
    values: list[float | torch.Tensor],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Stack scalar stats once at the end of a learning step."""

    target_device = device
    if target_device is None:
        for value in values:
            if isinstance(value, torch.Tensor):
                target_device = value.device
                break
    if target_device is None:
        target_device = torch.device("cpu")
    if not values:
        return torch.zeros(1, dtype=torch.float32, device=target_device)
    stacked: list[torch.Tensor] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            stacked.append(
                value.detach().to(device=target_device, dtype=torch.float32).reshape(1)
            )
        else:
            stacked.append(
                torch.tensor([float(value)], device=target_device, dtype=torch.float32)
            )
    return torch.cat(stacked)


def resolve_stat_device(
    stats: dict[str, list[float | torch.Tensor]],
    *,
    fallback: torch.Tensor | torch.device | None = None,
) -> torch.device:
    """Resolve a device for scalar summaries from available tensor stats."""

    for values in stats.values():
        for value in values:
            if isinstance(value, torch.Tensor):
                return value.device
    if isinstance(fallback, torch.Tensor):
        return fallback.device
    if isinstance(fallback, torch.device):
        return fallback
    return torch.device("cpu")


def mean_scalar_stats_by_key(
    stats: dict[str, list[float | torch.Tensor]],
    keys: tuple[str, ...],
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate present scalar stat lists by mean."""

    infos = {}
    for key in keys:
        values = stats.get(key, [])
        if values:
            infos[key] = stack_scalar_stats(values, device=device).mean()
    return infos


def fill_missing_scalar_info_defaults(
    infos: dict[str, torch.Tensor],
    defaults: dict[str, float],
    *,
    device: torch.device,
) -> None:
    """Fill missing scalar info keys in-place with float32 tensors."""

    for key, default in defaults.items():
        infos.setdefault(
            key,
            torch.tensor(
                default,
                device=device,
                dtype=torch.float32,
            ),
        )


def finalize_row_sharded_info_stats(
    infos: dict[str, torch.Tensor],
    stats: dict[str, list[float | torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Finalize row-sharded exact Dr.X scalar info metrics."""

    max_stat_keys = {"policy_grad_norm", "logprobs_diff_max"}
    min_stat_keys = {"logprobs_diff_min"}
    for key, values in stats.items():
        stacked = stack_scalar_stats(values, device=device)
        if key in max_stat_keys:
            infos[key] = stacked.max()
        elif key in min_stat_keys:
            infos[key] = stacked.min()
        else:
            infos[key] = stacked.mean()
    fill_missing_scalar_info_defaults(
        infos,
        {
            "policy_grad_norm": 0.0,
            "get_grad_norm_time": 0.0,
            "listwise_semantic_exploration_gain_any_correct": 0.0,
            "listwise_semantic_exploration_gain_drgrpo": 0.0,
        },
        device=device,
    )
    return {key: infos[key] for key in sorted(infos)}


def summarize_listwise_weight_entropy_stats(
    stats: dict[str, list[float | torch.Tensor]],
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize listwise weight-entropy stats and compatibility aliases."""

    infos: dict[str, torch.Tensor] = {}
    if stats.get("listwise_weight_entropy", []):
        infos["listwise_weight_entropy"] = stack_scalar_stats(
            stats["listwise_weight_entropy"],
            device=device,
        ).mean()
        infos["weight_entropy"] = infos["listwise_weight_entropy"]
        infos["listwise_weight_entropy_active"] = infos["listwise_weight_entropy"]
        infos["listwise_weight_entropy_min"] = stack_scalar_stats(
            stats["listwise_weight_entropy_min"],
            device=device,
        ).min()
        infos["weight_entropy_min"] = infos["listwise_weight_entropy_min"]
        infos["listwise_weight_entropy_max"] = stack_scalar_stats(
            stats["listwise_weight_entropy_max"],
            device=device,
        ).max()
        infos["weight_entropy_max"] = infos["listwise_weight_entropy_max"]
    if stats.get("listwise_weight_entropy_all", []):
        infos["listwise_weight_entropy_all"] = stack_scalar_stats(
            stats["listwise_weight_entropy_all"],
            device=device,
        ).mean()
        infos["listwise_weight_entropy_all_min"] = stack_scalar_stats(
            stats["listwise_weight_entropy_all_min"],
            device=device,
        ).min()
        infos["listwise_weight_entropy_all_max"] = stack_scalar_stats(
            stats["listwise_weight_entropy_all_max"],
            device=device,
        ).max()
    return infos


def summarize_listwise_core_scalar_stats(
    stats: dict[str, list[float | torch.Tensor]],
    *,
    advantages: torch.Tensor,
    grouped_reward_values: torch.Tensor,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize core listwise scalar diagnostics."""

    target_device = device or advantages.device
    infos = {
        "logprobs_diff_max": stack_scalar_stats(
            stats.get("logprobs_diff_max", []),
            device=target_device,
        ).max(),
        "logprobs_diff_min": stack_scalar_stats(
            stats.get("logprobs_diff_min", []),
            device=target_device,
        ).min(),
        "zero_pg_loss_count": stack_scalar_stats(
            stats.get("zero_pg_loss_count", []),
            device=target_device,
        ).mean(),
        "adv_mean": advantages.mean().to(device=target_device, dtype=torch.float32),
        "adv_min": advantages.min().to(device=target_device, dtype=torch.float32),
        "adv_max": advantages.max().to(device=target_device, dtype=torch.float32),
    }
    if stats.get("clip_ratio_low", []):
        infos["clip_ratio_low"] = stack_scalar_stats(
            stats["clip_ratio_low"],
            device=target_device,
        ).mean()
        infos["clip_ratio_high"] = stack_scalar_stats(
            stats["clip_ratio_high"],
            device=target_device,
        ).mean()
        infos["clip_ratio_region"] = stack_scalar_stats(
            stats["clip_ratio_region"],
            device=target_device,
        ).mean()
    if stats.get("pg_clipfrac", []):
        infos["pg_clipfrac"] = stack_scalar_stats(
            stats["pg_clipfrac"],
            device=target_device,
        ).mean()

    all_zero_rewards_count = (
        (grouped_reward_values.mean(-1) == 0)
        .sum()
        .to(
            device=target_device,
            dtype=torch.float32,
        )
    )
    all_one_rewards_count = (
        (grouped_reward_values.mean(-1) == 1)
        .sum()
        .to(
            device=target_device,
            dtype=torch.float32,
        )
    )
    infos["all_zero_rewards_count"] = all_zero_rewards_count
    infos["all_one_rewards_count"] = all_one_rewards_count
    infos["listwise_all_zero_rewards_count"] = all_zero_rewards_count
    infos["listwise_all_one_rewards_count"] = all_one_rewards_count
    return infos


def concat_vector_stats(
    values: list[float | torch.Tensor],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Concatenate vector-valued stats once at the end of a learning step."""

    target_device = device
    if target_device is None:
        for value in values:
            if isinstance(value, torch.Tensor):
                target_device = value.device
                break
    if target_device is None:
        target_device = torch.device("cpu")
    if not values:
        return torch.empty(0, dtype=torch.float32, device=target_device)
    pieces: list[torch.Tensor] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            pieces.append(
                value.detach().to(device=target_device, dtype=torch.float32).reshape(-1)
            )
        else:
            pieces.append(
                torch.tensor([float(value)], device=target_device, dtype=torch.float32)
            )
    return (
        torch.cat(pieces, dim=0)
        if pieces
        else torch.empty(
            0,
            dtype=torch.float32,
            device=target_device,
        )
    )


def tensor_correlation_or_zero(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Return a centered vector correlation, or zero when it is undefined."""

    if lhs.numel() < 2 or rhs.numel() < 2:
        return lhs.new_zeros(())
    lhs_centered = lhs - lhs.mean()
    rhs_centered = rhs - rhs.mean()
    denom = lhs_centered.norm() * rhs_centered.norm()
    if float(denom.item()) <= 1e-12:
        return lhs.new_zeros(())
    return (lhs_centered * rhs_centered).sum() / denom


def entropy_split_means(
    entropy: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return value means for the low-entropy and high-entropy halves."""

    if entropy.numel() < 2 or values.numel() < 2:
        mean_value = values.mean() if values.numel() > 0 else values.new_zeros(())
        return mean_value, mean_value
    order = torch.argsort(entropy)
    low_count = max(int(order.numel() // 2), 1)
    high_count = max(int(order.numel() - low_count), 1)
    low_idx = order[:low_count]
    high_idx = order[-high_count:]
    return values[low_idx].mean(), values[high_idx].mean()


def summarize_semantic_prompt_diagnostics(
    prompt_diag_stats: dict[str, list[float | torch.Tensor]],
    *,
    device: torch.device | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
    """Summarize prompt-level semantic exploration diagnostics."""

    semantic_prompt_entropy = concat_vector_stats(
        prompt_diag_stats["listwise_semantic_entropy_prompt"],
        device=device,
    )
    semantic_prompt_gain_any = concat_vector_stats(
        prompt_diag_stats["listwise_semantic_exploration_gain_any_correct_prompt"],
        device=device,
    )
    semantic_prompt_gain_drgrpo = concat_vector_stats(
        prompt_diag_stats["listwise_semantic_exploration_gain_drgrpo_prompt"],
        device=device,
    )
    if semantic_prompt_entropy.numel() <= 0:
        return {}, None, None

    semantic_prompt_entropy = all_gather_variable_length_1d_tensor(
        semantic_prompt_entropy
    )
    semantic_prompt_gain_any = all_gather_variable_length_1d_tensor(
        semantic_prompt_gain_any
    )
    semantic_prompt_gain_drgrpo = all_gather_variable_length_1d_tensor(
        semantic_prompt_gain_drgrpo
    )

    target_device = device or semantic_prompt_entropy.device
    low_any, high_any = entropy_split_means(
        semantic_prompt_entropy,
        semantic_prompt_gain_any,
    )
    low_drgrpo, high_drgrpo = entropy_split_means(
        semantic_prompt_entropy,
        semantic_prompt_gain_drgrpo,
    )
    return (
        {
            "listwise_semantic_exploration_prompt_count": torch.tensor(
                float(semantic_prompt_entropy.numel()),
                device=target_device,
                dtype=torch.float32,
            ),
            "listwise_semantic_exploration_entropy_std": (
                semantic_prompt_entropy.std(unbiased=False)
            ),
            "listwise_semantic_exploration_gain_any_correct_std": (
                semantic_prompt_gain_any.std(unbiased=False)
            ),
            "listwise_semantic_exploration_gain_drgrpo_std": (
                semantic_prompt_gain_drgrpo.std(unbiased=False)
            ),
            "listwise_semantic_exploration_gain_any_correct_corr": (
                tensor_correlation_or_zero(
                    semantic_prompt_entropy,
                    semantic_prompt_gain_any,
                )
            ),
            "listwise_semantic_exploration_gain_drgrpo_corr": (
                tensor_correlation_or_zero(
                    semantic_prompt_entropy,
                    semantic_prompt_gain_drgrpo,
                )
            ),
            "listwise_semantic_exploration_gain_any_correct_low_entropy": low_any,
            "listwise_semantic_exploration_gain_any_correct_high_entropy": high_any,
            "listwise_semantic_exploration_gain_any_correct_high_minus_low": (
                high_any - low_any
            ),
            "listwise_semantic_exploration_gain_drgrpo_low_entropy": low_drgrpo,
            "listwise_semantic_exploration_gain_drgrpo_high_entropy": high_drgrpo,
            "listwise_semantic_exploration_gain_drgrpo_high_minus_low": (
                high_drgrpo - low_drgrpo
            ),
        },
        semantic_prompt_gain_any.mean(),
        semantic_prompt_gain_drgrpo.mean(),
    )


def finalize_listwise_info_stats(
    infos: dict[str, torch.Tensor],
    stats: dict[str, list[float | torch.Tensor]],
    prompt_diag_stats: dict[str, list[float | torch.Tensor]],
    *,
    advantages: torch.Tensor,
    grouped_reward_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Finalize listwise learner info metrics and fill stable default keys."""

    scalar_stat_device = resolve_stat_device(stats, fallback=advantages)
    infos["policy_grad_norm"] = stack_scalar_stats(
        stats.get("policy_grad_norm", []),
        device=scalar_stat_device,
    ).max()
    infos.update(
        summarize_listwise_weight_entropy_stats(
            stats,
            device=scalar_stat_device,
        )
    )
    if stats.get("listwise_semantic_cluster_entropy", []):
        infos["listwise_semantic_cluster_entropy"] = stack_scalar_stats(
            stats["listwise_semantic_cluster_entropy"],
            device=scalar_stat_device,
        ).mean()
        infos["listwise_semantic_cluster_entropy_norm"] = infos[
            "listwise_semantic_cluster_entropy"
        ]
    infos.update(
        mean_scalar_stats_by_key(
            stats,
            LISTWISE_MEAN_INFO_STAT_KEYS,
            device=scalar_stat_device,
        )
    )
    (
        prompt_diag_infos,
        semantic_prompt_gain_any_mean,
        semantic_prompt_gain_drgrpo_mean,
    ) = summarize_semantic_prompt_diagnostics(
        prompt_diag_stats,
        device=scalar_stat_device,
    )
    infos.update(prompt_diag_infos)
    if semantic_prompt_gain_any_mean is not None:
        infos["listwise_semantic_exploration_gain_any_correct"] = (
            semantic_prompt_gain_any_mean
        )
        infos["listwise_semantic_exploration_gain_drgrpo"] = (
            semantic_prompt_gain_drgrpo_mean
        )
    infos.update(
        summarize_listwise_core_scalar_stats(
            stats,
            advantages=advantages,
            grouped_reward_values=grouped_reward_values,
            device=scalar_stat_device,
        )
    )
    fill_missing_scalar_info_defaults(
        infos,
        LISTWISE_OPTIONAL_LOGGING_METRIC_DEFAULTS,
        device=scalar_stat_device,
    )
    return {key: infos[key] for key in sorted(infos)}


def all_gather_variable_length_1d_tensor(values: torch.Tensor) -> torch.Tensor:
    """Gather a 1D tensor with variable local length across distributed ranks."""

    flat = values.reshape(-1)
    if not dist.is_available() or not dist.is_initialized():
        return flat
    length = torch.tensor([flat.numel()], device=flat.device, dtype=torch.int64)
    world_size = dist.get_world_size()
    gathered_lengths = [torch.zeros_like(length) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, length)
    max_length = int(torch.stack(gathered_lengths).max().item())
    if max_length <= 0:
        return flat.new_zeros((0,), dtype=flat.dtype)
    padded = flat.new_zeros((max_length,), dtype=flat.dtype)
    if flat.numel() > 0:
        padded[: flat.numel()] = flat
    gathered = [
        flat.new_zeros((max_length,), dtype=flat.dtype) for _ in range(world_size)
    ]
    dist.all_gather(gathered, padded)
    chunks = []
    for shard, shard_length in zip(gathered, gathered_lengths):
        shard_count = int(shard_length.item())
        if shard_count > 0:
            chunks.append(shard[:shard_count])
    return (
        torch.cat(chunks, dim=0) if chunks else flat.new_zeros((0,), dtype=flat.dtype)
    )
