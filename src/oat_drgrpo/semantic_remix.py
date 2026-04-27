"""Semantic remix and anchor-mass helpers for Dr.X/listwise learning."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class SemanticWeightDiagnostics:
    """Grouped diagnostics for residual competitive-mode semantic weighting."""

    mode_count_grouped: torch.Tensor
    eligible_mode_count_grouped: torch.Tensor
    eligible_mode_frac_grouped: torch.Tensor
    distinct_correct_mode_count_grouped: torch.Tensor
    distinct_correct_mode_frac_grouped: torch.Tensor
    best_score_grouped: torch.Tensor
    second_score_grouped: torch.Tensor
    competitive_gap_grouped: torch.Tensor
    explore_budget_grouped: torch.Tensor
    explore_budget_saturated_grouped: torch.Tensor
    explore_applied_group_mask: torch.Tensor
    verified_bonus_applied_group_mask: torch.Tensor
    prompt_selected_group_mask: torch.Tensor
    prompt_rejected_low_opp_group_mask: torch.Tensor
    prompt_rejected_nonpositive_group_mask: torch.Tensor
    prompt_rejected_len_guard_group_mask: torch.Tensor
    prompt_rejected_format_guard_group_mask: torch.Tensor
    prompt_rejected_verified_bonus_len_guard_group_mask: torch.Tensor
    prompt_rejected_verified_bonus_format_guard_group_mask: torch.Tensor
    moved_mass_l1_grouped: torch.Tensor
    alpha_raw_grouped: torch.Tensor
    alpha_applied_grouped: torch.Tensor
    verified_bonus_grouped: torch.Tensor
    expected_utility_q_grouped: torch.Tensor
    expected_utility_explore_target_grouped: torch.Tensor
    expected_utility_final_w_grouped: torch.Tensor
    expected_len_q_grouped: torch.Tensor
    expected_len_explore_target_grouped: torch.Tensor
    expected_len_final_w_grouped: torch.Tensor
    expected_format_q_grouped: torch.Tensor
    expected_format_explore_target_grouped: torch.Tensor
    expected_format_final_w_grouped: torch.Tensor


def compute_anchor_relative_sequence_utilities(
    *,
    anchor_seq_logps_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return prompt-group anchor-relative utilities ``u_i = -log pi_anc(y_i|x)``."""

    if valid_row_mask_grouped is None:
        valid_mask = torch.ones_like(anchor_seq_logps_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != anchor_seq_logps_grouped.shape:
            raise ValueError(
                "valid_row_mask_grouped must match anchor_seq_logps_grouped."
            )
        valid_mask = valid_row_mask_grouped.to(torch.bool)
    utilities = -anchor_seq_logps_grouped.to(torch.float32)
    return torch.where(valid_mask, utilities, torch.zeros_like(utilities))


def compute_anchor_relative_weights(
    *,
    anchor_seq_logps_grouped: torch.Tensor,
    tau: float,
    candidate_kl_coef: float,
    valid_row_mask_grouped: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return anchor-relative utilities plus their closed-form listwise weights."""

    from .listwise import compute_listwise_weights_from_utilities

    utility_grouped = compute_anchor_relative_sequence_utilities(
        anchor_seq_logps_grouped=anchor_seq_logps_grouped,
        valid_row_mask_grouped=valid_row_mask_grouped,
    )
    weights_grouped = compute_listwise_weights_from_utilities(
        utility_grouped=utility_grouped,
        ref_seq_logps_grouped=anchor_seq_logps_grouped.to(torch.float32),
        tau=tau,
        candidate_kl_coef=candidate_kl_coef,
        valid_row_mask_grouped=valid_row_mask_grouped,
    )
    return utility_grouped, weights_grouped


def compute_semantic_cluster_weights_from_utilities(
    *,
    utility_grouped: torch.Tensor,
    ref_seq_logps_grouped: torch.Tensor,
    cluster_ids_grouped: torch.Tensor,
    semantic_mass_weights_grouped: torch.Tensor | None = None,
    candidate_correctness_grouped: torch.Tensor | None = None,
    candidate_lengths_grouped: torch.Tensor | None = None,
    candidate_formatted_grouped: torch.Tensor | None = None,
    tau: float,
    mode_tau: float,
    mode_gap: float,
    mode_top_k: int,
    budget_grouped: torch.Tensor | None = None,
    budget_max: float = 0.0,
    intra_tau: float = 1e-2,
    candidate_kl_coef: float,
    prompt_select_min_alpha_frac: float = 0.0,
    positive_only: bool = False,
    verified_distinct_bonus_coef: float = 0.0,
    verified_distinct_min_modes: int = 2,
    verified_distinct_reward_threshold: float = 0.999,
    max_expected_len_delta: float = float("inf"),
    max_expected_format_drop: float = 0.0,
    valid_row_mask_grouped: torch.Tensor | None = None,
    semantic_remix_mode: str = "competitive",
) -> tuple[torch.Tensor, SemanticWeightDiagnostics]:
    """Return baseline-plus-residual competitive semantic weights."""

    from .listwise import (
        coerce_non_negative_float,
        compute_listwise_weights_from_utilities,
        normalize_semantic_remix_mode,
    )

    if utility_grouped.shape != ref_seq_logps_grouped.shape:
        raise ValueError(
            "utility_grouped and ref_seq_logps_grouped must have matching shapes."
        )
    if cluster_ids_grouped.shape != utility_grouped.shape:
        raise ValueError("cluster_ids_grouped must match grouped utilities.")
    if (
        semantic_mass_weights_grouped is not None
        and semantic_mass_weights_grouped.shape != utility_grouped.shape
    ):
        raise ValueError("semantic_mass_weights_grouped must match grouped utilities.")
    if (
        candidate_correctness_grouped is not None
        and candidate_correctness_grouped.shape != utility_grouped.shape
    ):
        raise ValueError("candidate_correctness_grouped must match grouped utilities.")
    if (
        candidate_lengths_grouped is not None
        and candidate_lengths_grouped.shape != utility_grouped.shape
    ):
        raise ValueError("candidate_lengths_grouped must match grouped utilities.")
    if (
        candidate_formatted_grouped is not None
        and candidate_formatted_grouped.shape != utility_grouped.shape
    ):
        raise ValueError("candidate_formatted_grouped must match grouped utilities.")
    if valid_row_mask_grouped is None:
        valid_mask = torch.ones_like(utility_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != utility_grouped.shape:
            raise ValueError(
                "valid_row_mask_grouped must match the grouped utility shape."
            )
        valid_mask = valid_row_mask_grouped.to(torch.bool)
    if candidate_correctness_grouped is None:
        candidate_correctness = torch.zeros_like(utility_grouped)
    else:
        candidate_correctness = candidate_correctness_grouped.to(
            device=utility_grouped.device,
            dtype=utility_grouped.dtype,
        ).clamp(min=0.0, max=1.0)
    if candidate_lengths_grouped is None:
        candidate_lengths = torch.zeros_like(utility_grouped)
    else:
        candidate_lengths = candidate_lengths_grouped.to(
            device=utility_grouped.device,
            dtype=utility_grouped.dtype,
        )
    if candidate_formatted_grouped is None:
        candidate_formatted = torch.zeros_like(utility_grouped)
    else:
        candidate_formatted = candidate_formatted_grouped.to(
            device=utility_grouped.device,
            dtype=utility_grouped.dtype,
        )
    semantic_valid_mask = valid_mask & (cluster_ids_grouped >= 0)
    if budget_grouped is None:
        budget_values = torch.zeros(
            int(utility_grouped.size(0)),
            device=utility_grouped.device,
            dtype=utility_grouped.dtype,
        )
    else:
        if budget_grouped.dim() != 1 or int(budget_grouped.numel()) != int(
            utility_grouped.size(0)
        ):
            raise ValueError("budget_grouped must provide one value per prompt group.")
        budget_values = budget_grouped.to(
            device=utility_grouped.device,
            dtype=utility_grouped.dtype,
        ).clamp(min=0.0)
    safe_mode_tau = max(coerce_non_negative_float(mode_tau, default=0.0), 1e-8)
    safe_mode_gap = coerce_non_negative_float(mode_gap, default=0.0)
    safe_mode_top_k = max(int(mode_top_k), 1)
    safe_budget_max = coerce_non_negative_float(budget_max, default=0.0)
    safe_intra_tau = max(coerce_non_negative_float(intra_tau, default=0.0), 1e-8)
    safe_candidate_kl = coerce_non_negative_float(candidate_kl_coef, default=0.0)
    safe_select_min_alpha_frac = min(
        max(coerce_non_negative_float(prompt_select_min_alpha_frac, default=0.0), 0.0),
        1.0,
    )
    positive_only = bool(positive_only)
    safe_verified_bonus_coef = coerce_non_negative_float(
        verified_distinct_bonus_coef,
        default=0.0,
    )
    safe_verified_distinct_min_modes = max(int(verified_distinct_min_modes), 2)
    safe_verified_distinct_reward_threshold = min(
        max(
            coerce_non_negative_float(
                verified_distinct_reward_threshold,
                default=0.999,
            ),
            0.0,
        ),
        1.0,
    )
    normalized_semantic_remix_mode = normalize_semantic_remix_mode(semantic_remix_mode)
    safe_max_expected_len_delta = float(max_expected_len_delta)
    if not math.isfinite(safe_max_expected_len_delta):
        safe_max_expected_len_delta = float("inf")
    safe_max_expected_format_drop = coerce_non_negative_float(
        max_expected_format_drop,
        default=0.0,
    )
    num_prompts = int(utility_grouped.size(0))
    base_weights = compute_listwise_weights_from_utilities(
        utility_grouped=utility_grouped,
        ref_seq_logps_grouped=ref_seq_logps_grouped,
        tau=tau,
        candidate_kl_coef=candidate_kl_coef,
        valid_row_mask_grouped=valid_mask,
    )
    semantic_mass_weights = None
    if semantic_mass_weights_grouped is not None:
        semantic_mass_weights = torch.where(
            valid_mask,
            semantic_mass_weights_grouped.to(
                device=utility_grouped.device,
                dtype=utility_grouped.dtype,
            ),
            torch.zeros_like(utility_grouped),
        )
        semantic_mass_row_sums = semantic_mass_weights.sum(dim=1, keepdim=True)
        uniform_semantic_mass = torch.where(
            valid_mask,
            1.0
            / valid_mask.to(dtype=utility_grouped.dtype)
            .sum(dim=1, keepdim=True)
            .clamp(min=1.0),
            torch.zeros_like(base_weights),
        )
        semantic_mass_weights = torch.where(
            semantic_mass_row_sums > 0,
            semantic_mass_weights / semantic_mass_row_sums.clamp(min=1e-12),
            uniform_semantic_mass,
        )
    weights = base_weights.clone()
    mode_count_grouped = torch.zeros(
        (num_prompts,), device=utility_grouped.device, dtype=utility_grouped.dtype
    )
    eligible_mode_count_grouped = torch.zeros_like(mode_count_grouped)
    eligible_mode_frac_grouped = torch.zeros_like(mode_count_grouped)
    distinct_correct_mode_count_grouped = torch.zeros_like(mode_count_grouped)
    distinct_correct_mode_frac_grouped = torch.zeros_like(mode_count_grouped)
    best_score_grouped = torch.zeros_like(mode_count_grouped)
    second_score_grouped = torch.zeros_like(mode_count_grouped)
    competitive_gap_grouped = torch.zeros_like(mode_count_grouped)
    explore_budget_grouped = torch.zeros_like(mode_count_grouped)
    explore_budget_saturated_grouped = torch.zeros_like(mode_count_grouped)
    explore_applied_group_mask = torch.zeros(
        (num_prompts,), device=utility_grouped.device, dtype=torch.bool
    )
    verified_bonus_applied_group_mask = torch.zeros_like(explore_applied_group_mask)
    prompt_selected_group_mask = torch.zeros_like(explore_applied_group_mask)
    prompt_rejected_low_opp_group_mask = torch.zeros_like(explore_applied_group_mask)
    prompt_rejected_nonpositive_group_mask = torch.zeros_like(
        explore_applied_group_mask
    )
    prompt_rejected_len_guard_group_mask = torch.zeros_like(explore_applied_group_mask)
    prompt_rejected_format_guard_group_mask = torch.zeros_like(
        explore_applied_group_mask
    )
    prompt_rejected_verified_bonus_len_guard_group_mask = torch.zeros_like(
        explore_applied_group_mask
    )
    prompt_rejected_verified_bonus_format_guard_group_mask = torch.zeros_like(
        explore_applied_group_mask
    )
    moved_mass_l1_grouped = torch.zeros_like(mode_count_grouped)
    alpha_raw_grouped = torch.zeros_like(mode_count_grouped)
    alpha_applied_grouped = torch.zeros_like(mode_count_grouped)
    verified_bonus_grouped = torch.zeros_like(mode_count_grouped)
    expected_utility_q_grouped = (
        base_weights
        * torch.where(valid_mask, utility_grouped, torch.zeros_like(utility_grouped))
    ).sum(dim=1)
    expected_utility_explore_target_grouped = expected_utility_q_grouped.clone()
    expected_utility_final_w_grouped = expected_utility_q_grouped.clone()
    expected_len_q_grouped = (
        base_weights
        * torch.where(
            valid_mask, candidate_lengths, torch.zeros_like(candidate_lengths)
        )
    ).sum(dim=1)
    expected_len_explore_target_grouped = expected_len_q_grouped.clone()
    expected_len_final_w_grouped = expected_len_q_grouped.clone()
    expected_format_q_grouped = (
        base_weights
        * torch.where(
            valid_mask, candidate_formatted, torch.zeros_like(candidate_formatted)
        )
    ).sum(dim=1)
    expected_format_explore_target_grouped = expected_format_q_grouped.clone()
    expected_format_final_w_grouped = expected_format_q_grouped.clone()

    for p in range(num_prompts):
        semantic_idx = torch.where(semantic_valid_mask[p])[0]
        if semantic_idx.numel() <= 0:
            continue

        row_clusters = cluster_ids_grouped[p, semantic_idx]
        unique_clusters = torch.unique(row_clusters[row_clusters >= 0], sorted=True)
        num_clusters = int(unique_clusters.numel())
        mode_count_grouped[p] = float(num_clusters)
        if num_clusters <= 1:
            continue

        member_mode_logits_all = utility_grouped[p, semantic_idx]
        if safe_candidate_kl > 0.0:
            member_mode_logits_all = member_mode_logits_all + (
                safe_candidate_kl * ref_seq_logps_grouped[p, semantic_idx]
            )

        cluster_scores = []
        cluster_correctness = []
        cluster_member_masks = []
        for cid in unique_clusters.tolist():
            mask = row_clusters == cid
            cluster_member_masks.append(mask)
            member_mode_logits = member_mode_logits_all[mask]
            cluster_scores.append(member_mode_logits.max().to(utility_grouped.dtype))
            cluster_correctness.append(
                candidate_correctness[p, semantic_idx][mask]
                .max()
                .to(utility_grouped.dtype)
            )

        cluster_scores_t = torch.stack(cluster_scores).to(dtype=utility_grouped.dtype)
        cluster_correctness_t = torch.stack(cluster_correctness).to(
            dtype=utility_grouped.dtype
        )
        sorted_scores, sorted_idx = torch.sort(cluster_scores_t, descending=True)
        best_score_grouped[p] = sorted_scores[0]
        second_score_grouped[p] = (
            sorted_scores[1] if num_clusters > 1 else sorted_scores[0]
        )
        competitive_gap_grouped[p] = (
            best_score_grouped[p] - second_score_grouped[p]
            if num_clusters > 1
            else torch.zeros_like(best_score_grouped[p])
        )

        alpha_p = torch.clamp(budget_values[p], min=0.0, max=safe_budget_max).to(
            dtype=utility_grouped.dtype
        )
        alpha_raw = (
            alpha_p / safe_budget_max
            if safe_budget_max > 0.0
            else torch.zeros_like(alpha_p)
        )
        alpha_raw_grouped[p] = alpha_raw
        if safe_budget_max > 0.0:
            explore_budget_saturated_grouped[p] = alpha_raw >= (1.0 - 1e-8)
        if float(alpha_raw.item()) < safe_select_min_alpha_frac:
            prompt_rejected_low_opp_group_mask[p] = True
            continue
        if positive_only and float(best_score_grouped[p].item()) <= 0.0:
            prompt_rejected_nonpositive_group_mask[p] = True
            continue

        if normalized_semantic_remix_mode != "anchor_rare":
            eligible_mask = cluster_scores_t >= (cluster_scores_t.max() - safe_mode_gap)
            if positive_only:
                eligible_mask = eligible_mask & (cluster_scores_t > 0.0)
            if safe_mode_top_k < num_clusters:
                topk_mask = torch.zeros_like(eligible_mask, dtype=torch.bool)
                topk_mask[sorted_idx[:safe_mode_top_k]] = True
                eligible_mask = eligible_mask & topk_mask
            eligible_count = int(eligible_mask.to(torch.int64).sum().item())
            eligible_mode_count_grouped[p] = float(eligible_count)
            eligible_mode_frac_grouped[p] = float(eligible_count) / float(
                max(num_clusters, 1)
            )
            if eligible_count < 2:
                if (
                    positive_only
                    and int((cluster_scores_t > 0.0).to(torch.int64).sum().item()) < 2
                ):
                    prompt_rejected_nonpositive_group_mask[p] = True
                continue

        if float(alpha_p.item()) <= 0.0:
            prompt_rejected_low_opp_group_mask[p] = True
            continue

        semantic_q = base_weights[p, semantic_idx]
        semantic_base_mass = semantic_q.sum()
        if float(semantic_base_mass.item()) <= 1e-12:
            continue

        if normalized_semantic_remix_mode == "anchor_rare":
            semantic_mass_q = (
                semantic_q
                if semantic_mass_weights is None
                else semantic_mass_weights[p, semantic_idx]
            )
            semantic_mass_q = semantic_mass_q / semantic_mass_q.sum().clamp(min=1e-12)
            if positive_only:
                prompt_valid_utilities = utility_grouped[p, valid_mask[p]]
                if (
                    prompt_valid_utilities.numel() <= 0
                    or float(prompt_valid_utilities.max().item()) <= 0.0
                ):
                    prompt_rejected_nonpositive_group_mask[p] = True
                    continue

            cluster_mass = torch.zeros(
                (num_clusters,),
                device=utility_grouped.device,
                dtype=utility_grouped.dtype,
            )
            for cluster_pos, mask in enumerate(cluster_member_masks):
                cluster_mass[cluster_pos] = semantic_mass_q[mask].sum()
            cluster_entropy = -(
                cluster_mass * torch.log(cluster_mass.clamp(min=1e-12))
            ).sum()
            cluster_bonus_t = torch.clamp(
                -torch.log(cluster_mass.clamp(min=1e-12)) - cluster_entropy,
                min=0.0,
            )
            eligible_mask = cluster_bonus_t > 1e-8
            eligible_count = int(eligible_mask.to(torch.int64).sum().item())
            eligible_mode_count_grouped[p] = float(eligible_count)
            eligible_mode_frac_grouped[p] = float(eligible_count) / float(
                max(num_clusters, 1)
            )
            sorted_bonus, _ = torch.sort(cluster_bonus_t, descending=True)
            best_score_grouped[p] = sorted_bonus[0]
            second_score_grouped[p] = (
                sorted_bonus[1] if num_clusters > 1 else sorted_bonus[0]
            )
            competitive_gap_grouped[p] = (
                best_score_grouped[p] - second_score_grouped[p]
                if num_clusters > 1
                else torch.zeros_like(best_score_grouped[p])
            )
            if eligible_count <= 0:
                continue

            semantic_surprisal_rows = torch.zeros(
                (int(semantic_idx.numel()),),
                device=utility_grouped.device,
                dtype=utility_grouped.dtype,
            )
            for cluster_pos, mask in enumerate(cluster_member_masks):
                bonus_scale = cluster_bonus_t[cluster_pos]
                if float(bonus_scale.item()) <= 1e-8:
                    continue
                semantic_surprisal_rows[mask] = semantic_mass_q[mask] * bonus_scale
            semantic_surprisal_mass = semantic_surprisal_rows.sum()
            if float(semantic_surprisal_mass.item()) <= 1e-12:
                continue

            semantic_explore_target = semantic_base_mass * (
                semantic_surprisal_rows / semantic_surprisal_mass.clamp(min=1e-12)
            )
            semantic_explore_full = base_weights[p].clone()
            semantic_explore_full[semantic_idx] = semantic_explore_target
            expected_utility_explore_target_grouped[p] = (
                semantic_explore_full
                * torch.where(
                    valid_mask[p],
                    utility_grouped[p],
                    torch.zeros_like(utility_grouped[p]),
                )
            ).sum()
            expected_len_explore_target_grouped[p] = (
                semantic_explore_full
                * torch.where(
                    valid_mask[p],
                    candidate_lengths[p],
                    torch.zeros_like(candidate_lengths[p]),
                )
            ).sum()
            expected_format_explore_target_grouped[p] = (
                semantic_explore_full
                * torch.where(
                    valid_mask[p],
                    candidate_formatted[p],
                    torch.zeros_like(candidate_formatted[p]),
                )
            ).sum()
            if (
                float(expected_len_explore_target_grouped[p].item())
                > float(expected_len_q_grouped[p].item())
                + safe_max_expected_len_delta
                + 1e-6
            ):
                prompt_rejected_len_guard_group_mask[p] = True
                continue
            if float(
                expected_format_explore_target_grouped[p].item()
            ) + safe_max_expected_format_drop + 1e-6 < float(
                expected_format_q_grouped[p].item()
            ):
                prompt_rejected_format_guard_group_mask[p] = True
                continue

            prompt_selected_group_mask[p] = True
            alpha_applied_grouped[p] = alpha_p
            explore_budget_grouped[p] = alpha_p
            weights[p, semantic_idx] = (
                1.0 - alpha_p
            ) * semantic_q + alpha_p * semantic_explore_target
            explore_applied_group_mask[p] = True
            moved_mass_l1_grouped[p] = (
                0.5 * torch.abs(weights[p] - base_weights[p]).sum()
            )
            expected_utility_final_w_grouped[p] = (
                weights[p]
                * torch.where(
                    valid_mask[p],
                    utility_grouped[p],
                    torch.zeros_like(utility_grouped[p]),
                )
            ).sum()
            expected_len_final_w_grouped[p] = (
                weights[p]
                * torch.where(
                    valid_mask[p],
                    candidate_lengths[p],
                    torch.zeros_like(candidate_lengths[p]),
                )
            ).sum()
            expected_format_final_w_grouped[p] = (
                weights[p]
                * torch.where(
                    valid_mask[p],
                    candidate_formatted[p],
                    torch.zeros_like(candidate_formatted[p]),
                )
            ).sum()
            continue

        eligible_scores = cluster_scores_t[eligible_mask]
        cluster_mass = torch.softmax(eligible_scores / safe_mode_tau, dim=0)
        semantic_group_weights = torch.zeros(
            (int(semantic_idx.numel()),),
            device=utility_grouped.device,
            dtype=utility_grouped.dtype,
        )
        eligible_positions = torch.where(eligible_mask)[0]
        for cluster_mass_idx, cluster_pos in enumerate(eligible_positions.tolist()):
            mask = cluster_member_masks[cluster_pos]
            member_mode_logits = member_mode_logits_all[mask].to(utility_grouped.dtype)
            within = torch.softmax(member_mode_logits / safe_intra_tau, dim=0)
            semantic_group_weights[mask] = (
                cluster_mass[cluster_mass_idx].to(utility_grouped.dtype) * within
            )

        semantic_explore_target = semantic_base_mass * semantic_group_weights
        semantic_q = base_weights[p, semantic_idx]
        semantic_explore_full = base_weights[p].clone()
        semantic_explore_full[semantic_idx] = semantic_explore_target
        expected_utility_explore_target_grouped[p] = (
            semantic_explore_full
            * torch.where(
                valid_mask[p], utility_grouped[p], torch.zeros_like(utility_grouped[p])
            )
        ).sum()
        expected_len_explore_target_grouped[p] = (
            semantic_explore_full
            * torch.where(
                valid_mask[p],
                candidate_lengths[p],
                torch.zeros_like(candidate_lengths[p]),
            )
        ).sum()
        expected_format_explore_target_grouped[p] = (
            semantic_explore_full
            * torch.where(
                valid_mask[p],
                candidate_formatted[p],
                torch.zeros_like(candidate_formatted[p]),
            )
        ).sum()
        if (
            float(expected_len_explore_target_grouped[p].item())
            > float(expected_len_q_grouped[p].item())
            + safe_max_expected_len_delta
            + 1e-6
        ):
            prompt_rejected_len_guard_group_mask[p] = True
            continue
        if float(
            expected_format_explore_target_grouped[p].item()
        ) + safe_max_expected_format_drop + 1e-6 < float(
            expected_format_q_grouped[p].item()
        ):
            prompt_rejected_format_guard_group_mask[p] = True
            continue

        right_mode_mask = eligible_mask & (
            cluster_correctness_t >= safe_verified_distinct_reward_threshold
        )
        right_count = int(right_mode_mask.to(torch.int64).sum().item())
        distinct_correct_mode_count_grouped[p] = float(right_count)
        distinct_correct_mode_frac_grouped[p] = float(right_count) / float(
            max(num_clusters, 1)
        )

        prompt_selected_group_mask[p] = True
        alpha_applied_grouped[p] = alpha_p
        beta_p = torch.zeros_like(alpha_p)
        semantic_surprisal_target = None
        if (
            safe_verified_bonus_coef > 0.0
            and right_count >= safe_verified_distinct_min_modes
            and float(alpha_p.item()) > 0.0
        ):
            right_scores = cluster_scores_t[right_mode_mask]
            right_cluster_mass = torch.softmax(right_scores / safe_mode_tau, dim=0)
            semantic_surprisal_group_weights = torch.zeros(
                (int(semantic_idx.numel()),),
                device=utility_grouped.device,
                dtype=utility_grouped.dtype,
            )
            right_positions = torch.where(right_mode_mask)[0]
            for right_mass_idx, cluster_pos in enumerate(right_positions.tolist()):
                mask = cluster_member_masks[cluster_pos]
                member_mode_logits = member_mode_logits_all[mask].to(
                    utility_grouped.dtype
                )
                within = torch.softmax(member_mode_logits / safe_intra_tau, dim=0)
                semantic_surprisal_group_weights[mask] = (
                    right_cluster_mass[right_mass_idx].to(utility_grouped.dtype)
                    * within
                )
            semantic_surprisal_target = (
                semantic_base_mass * semantic_surprisal_group_weights
            )
            beta_p = torch.clamp(
                alpha_p * safe_verified_bonus_coef,
                min=0.0,
                max=max(1.0 - float(alpha_p.item()), 0.0),
            )
            if float(beta_p.item()) > 0.0:
                candidate_semantic_weights = (
                    (1.0 - alpha_p - beta_p) * semantic_q
                    + alpha_p * semantic_explore_target
                    + beta_p * semantic_surprisal_target
                )
                candidate_bonus_full = base_weights[p].clone()
                candidate_bonus_full[semantic_idx] = candidate_semantic_weights
                candidate_expected_len = (
                    candidate_bonus_full
                    * torch.where(
                        valid_mask[p],
                        candidate_lengths[p],
                        torch.zeros_like(candidate_lengths[p]),
                    )
                ).sum()
                candidate_expected_format = (
                    candidate_bonus_full
                    * torch.where(
                        valid_mask[p],
                        candidate_formatted[p],
                        torch.zeros_like(candidate_formatted[p]),
                    )
                ).sum()
                if (
                    float(candidate_expected_len.item())
                    > float(expected_len_q_grouped[p].item())
                    + safe_max_expected_len_delta
                    + 1e-6
                ):
                    prompt_rejected_verified_bonus_len_guard_group_mask[p] = True
                    beta_p = torch.zeros_like(alpha_p)
                elif float(
                    candidate_expected_format.item()
                ) + safe_max_expected_format_drop + 1e-6 < float(
                    expected_format_q_grouped[p].item()
                ):
                    prompt_rejected_verified_bonus_format_guard_group_mask[p] = True
                    beta_p = torch.zeros_like(alpha_p)
                else:
                    verified_bonus_applied_group_mask[p] = True

        verified_bonus_grouped[p] = beta_p
        explore_budget_grouped[p] = alpha_p + beta_p
        if semantic_surprisal_target is not None and float(beta_p.item()) > 0.0:
            weights[p, semantic_idx] = (
                (1.0 - alpha_p - beta_p) * semantic_q
                + alpha_p * semantic_explore_target
                + beta_p * semantic_surprisal_target
            )
        else:
            weights[p, semantic_idx] = (
                1.0 - alpha_p
            ) * semantic_q + alpha_p * semantic_explore_target
        explore_applied_group_mask[p] = True
        moved_mass_l1_grouped[p] = 0.5 * torch.abs(weights[p] - base_weights[p]).sum()
        expected_utility_final_w_grouped[p] = (
            weights[p]
            * torch.where(
                valid_mask[p], utility_grouped[p], torch.zeros_like(utility_grouped[p])
            )
        ).sum()
        expected_len_final_w_grouped[p] = (
            weights[p]
            * torch.where(
                valid_mask[p],
                candidate_lengths[p],
                torch.zeros_like(candidate_lengths[p]),
            )
        ).sum()
        expected_format_final_w_grouped[p] = (
            weights[p]
            * torch.where(
                valid_mask[p],
                candidate_formatted[p],
                torch.zeros_like(candidate_formatted[p]),
            )
        ).sum()

    row_sums = weights.sum(dim=1, keepdim=True)
    uniform_weights = torch.where(
        valid_mask,
        1.0
        / valid_mask.to(dtype=utility_grouped.dtype)
        .sum(dim=1, keepdim=True)
        .clamp(min=1.0),
        torch.zeros_like(weights),
    )
    weights = torch.where(
        row_sums > 0, weights / row_sums.clamp(min=1e-12), uniform_weights
    )
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    diagnostics = SemanticWeightDiagnostics(
        mode_count_grouped=mode_count_grouped,
        eligible_mode_count_grouped=eligible_mode_count_grouped,
        eligible_mode_frac_grouped=eligible_mode_frac_grouped,
        distinct_correct_mode_count_grouped=distinct_correct_mode_count_grouped,
        distinct_correct_mode_frac_grouped=distinct_correct_mode_frac_grouped,
        best_score_grouped=best_score_grouped,
        second_score_grouped=second_score_grouped,
        competitive_gap_grouped=competitive_gap_grouped,
        explore_budget_grouped=explore_budget_grouped,
        explore_budget_saturated_grouped=explore_budget_saturated_grouped,
        explore_applied_group_mask=explore_applied_group_mask,
        verified_bonus_applied_group_mask=verified_bonus_applied_group_mask,
        prompt_selected_group_mask=prompt_selected_group_mask,
        prompt_rejected_low_opp_group_mask=prompt_rejected_low_opp_group_mask,
        prompt_rejected_nonpositive_group_mask=prompt_rejected_nonpositive_group_mask,
        prompt_rejected_len_guard_group_mask=prompt_rejected_len_guard_group_mask,
        prompt_rejected_format_guard_group_mask=prompt_rejected_format_guard_group_mask,
        prompt_rejected_verified_bonus_len_guard_group_mask=prompt_rejected_verified_bonus_len_guard_group_mask,
        prompt_rejected_verified_bonus_format_guard_group_mask=prompt_rejected_verified_bonus_format_guard_group_mask,
        moved_mass_l1_grouped=moved_mass_l1_grouped,
        alpha_raw_grouped=alpha_raw_grouped,
        alpha_applied_grouped=alpha_applied_grouped,
        verified_bonus_grouped=verified_bonus_grouped,
        expected_utility_q_grouped=expected_utility_q_grouped,
        expected_utility_explore_target_grouped=expected_utility_explore_target_grouped,
        expected_utility_final_w_grouped=expected_utility_final_w_grouped,
        expected_len_q_grouped=expected_len_q_grouped,
        expected_len_explore_target_grouped=expected_len_explore_target_grouped,
        expected_len_final_w_grouped=expected_len_final_w_grouped,
        expected_format_q_grouped=expected_format_q_grouped,
        expected_format_explore_target_grouped=expected_format_explore_target_grouped,
        expected_format_final_w_grouped=expected_format_final_w_grouped,
    )
    return weights, diagnostics
