"""Dr.X target bundle and projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class DrXTargetBundle:
    """Grouped DrX objects: utility, posterior target, and optimization gates."""

    utility_grouped: torch.Tensor
    w_star_grouped: torch.Tensor
    token_target_grouped: torch.Tensor
    projection_target_grouped: torch.Tensor
    informative_group_mask: torch.Tensor
    neutral_group_mask: torch.Tensor
    contributing_group_mask: torch.Tensor
    projection_group_scale: torch.Tensor
    semantic_diagnostics: Any | None = None


def _masked_group_log_softmax(
    values_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
) -> torch.Tensor:
    if values_grouped.shape != valid_row_mask_grouped.shape:
        raise ValueError(
            "values_grouped and valid_row_mask_grouped must have matching shapes."
        )
    valid_mask = valid_row_mask_grouped.to(torch.bool)
    if values_grouped.dim() != 2:
        raise ValueError("masked_group_log_softmax requires rank-2 grouped values.")

    neg_inf = torch.full_like(values_grouped, torch.finfo(values_grouped.dtype).min)
    masked_values = torch.where(valid_mask, values_grouped, neg_inf)
    has_valid = valid_mask.any(dim=1, keepdim=True)
    max_vals = masked_values.max(dim=1, keepdim=True).values
    max_vals = torch.where(has_valid, max_vals, torch.zeros_like(max_vals))
    shifted = torch.where(
        valid_mask,
        masked_values - max_vals,
        torch.zeros_like(masked_values),
    )
    exp_shifted = torch.where(valid_mask, torch.exp(shifted), torch.zeros_like(shifted))
    log_denom = torch.log(exp_shifted.sum(dim=1, keepdim=True).clamp(min=1e-12))
    log_probs = shifted - log_denom
    return torch.where(valid_mask, log_probs, torch.zeros_like(log_probs))


def compute_drx_group_masks(
    *,
    utility_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
    neutral_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return informative/neutral/contributing masks for grouped DrX utilities."""

    if utility_grouped.shape != valid_row_mask_grouped.shape:
        raise ValueError(
            "utility_grouped and valid_row_mask_grouped must have matching shapes."
        )

    valid_mask = valid_row_mask_grouped.to(torch.bool)
    contributing_group_mask = valid_mask.any(dim=1)

    dtype_info = torch.finfo(utility_grouped.dtype)
    valid_max = torch.where(
        valid_mask,
        utility_grouped,
        torch.full_like(utility_grouped, dtype_info.min),
    ).amax(dim=1)
    valid_min = torch.where(
        valid_mask,
        utility_grouped,
        torch.full_like(utility_grouped, dtype_info.max),
    ).amin(dim=1)
    valid_count = valid_mask.to(torch.int64).sum(dim=1)

    neutral_group_mask = contributing_group_mask & (
        (valid_count <= 1) | ((valid_max - valid_min) <= float(neutral_eps))
    )
    informative_group_mask = contributing_group_mask & (~neutral_group_mask)
    return informative_group_mask, neutral_group_mask, contributing_group_mask


def build_drgrpo_token_active_row_mask(
    *,
    advantage_source: str,
    informative_group_mask: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
    utility_centered_advantages_grouped: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return rows in the token-level Dr.GRPO loss denominator."""

    if informative_group_mask.dim() != 1:
        raise ValueError("informative_group_mask must be one-dimensional.")
    if valid_row_mask_grouped.shape != utility_centered_advantages_grouped.shape:
        raise ValueError(
            "valid_row_mask_grouped and utility_centered_advantages_grouped must match."
        )
    if int(informative_group_mask.numel()) != int(valid_row_mask_grouped.size(0)):
        raise ValueError("informative_group_mask must have one entry per prompt group.")

    valid_row_mask = valid_row_mask_grouped.to(torch.bool)
    if str(advantage_source) == "utility_centered":
        centered_advantages = utility_centered_advantages_grouped.to(
            device=valid_row_mask.device
        )
        del eps
        return valid_row_mask & torch.isfinite(centered_advantages)
    return (
        informative_group_mask.to(device=valid_row_mask.device, dtype=torch.bool)[
            :, None
        ]
        & valid_row_mask
    )


def apply_neutral_tiebreak_to_advantages(
    *,
    row_advantages_grouped: torch.Tensor,
    utility_grouped: torch.Tensor,
    tiebreak_values_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
    enabled: bool,
    neutral_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inject a centered semantic tie-break only for raw-neutral prompt groups."""

    if row_advantages_grouped.shape != utility_grouped.shape:
        raise ValueError(
            "row_advantages_grouped and utility_grouped must have matching shapes."
        )
    if row_advantages_grouped.shape != tiebreak_values_grouped.shape:
        raise ValueError(
            "tiebreak_values_grouped must match the grouped advantage shape."
        )
    if row_advantages_grouped.shape != valid_row_mask_grouped.shape:
        raise ValueError(
            "valid_row_mask_grouped must match the grouped advantage shape."
        )

    _, raw_neutral_group_mask, _ = compute_drx_group_masks(
        utility_grouped=utility_grouped,
        valid_row_mask_grouped=valid_row_mask_grouped,
        neutral_eps=neutral_eps,
    )
    if not enabled:
        return (
            row_advantages_grouped,
            raw_neutral_group_mask,
            torch.zeros_like(raw_neutral_group_mask),
        )

    valid_mask = valid_row_mask_grouped.to(torch.bool)
    safe_tiebreak = torch.where(
        valid_mask,
        tiebreak_values_grouped.to(dtype=row_advantages_grouped.dtype),
        torch.zeros_like(row_advantages_grouped),
    )
    tiebreak_group_mean = safe_tiebreak.sum(dim=1, keepdim=True) / valid_mask.to(
        dtype=row_advantages_grouped.dtype
    ).sum(dim=1, keepdim=True).clamp(min=1.0)
    centered_tiebreak = torch.where(
        valid_mask,
        safe_tiebreak - tiebreak_group_mean,
        torch.zeros_like(safe_tiebreak),
    )
    applied_group_mask = raw_neutral_group_mask & (
        centered_tiebreak.abs().amax(dim=1) > 1e-8
    )
    adjusted_advantages = torch.where(
        applied_group_mask[:, None] & valid_mask,
        row_advantages_grouped + centered_tiebreak,
        row_advantages_grouped,
    )
    adjusted_advantages = torch.where(
        valid_mask,
        adjusted_advantages,
        torch.zeros_like(adjusted_advantages),
    )
    return adjusted_advantages, raw_neutral_group_mask, applied_group_mask


def compute_drx_projection_sequence_coefficients(
    *,
    policy_seq_logps_grouped: torch.Tensor,
    projection_target_grouped: torch.Tensor,
    projection_group_scale: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
    normalizer_total_group_weight: Optional[float] = None,
) -> torch.Tensor:
    """Return exact d(loss)/d(seq_logp) coeffs for KL(w* || p^pi)."""

    if policy_seq_logps_grouped.shape != projection_target_grouped.shape:
        raise ValueError(
            "policy_seq_logps_grouped and projection_target_grouped must match."
        )
    if policy_seq_logps_grouped.shape != valid_row_mask_grouped.shape:
        raise ValueError(
            "valid_row_mask_grouped must match the grouped sequence shape."
        )
    if projection_group_scale.dim() != 1 or int(projection_group_scale.numel()) != int(
        policy_seq_logps_grouped.size(0)
    ):
        raise ValueError(
            "projection_group_scale must match the prompt-group dimension."
        )

    valid_mask = valid_row_mask_grouped.to(torch.bool)
    policy_log_probs_grouped = _masked_group_log_softmax(
        policy_seq_logps_grouped,
        valid_mask,
    )
    policy_probs_grouped = torch.where(
        valid_mask,
        torch.exp(policy_log_probs_grouped),
        torch.zeros_like(policy_log_probs_grouped),
    )

    target_mass_grouped = (
        projection_target_grouped * valid_mask.to(projection_target_grouped.dtype)
    ).sum(dim=1, keepdim=True)

    coeffs = target_mass_grouped * policy_probs_grouped - projection_target_grouped
    coeffs = torch.where(valid_mask, coeffs, torch.zeros_like(coeffs))

    scale = projection_group_scale.to(
        device=policy_seq_logps_grouped.device,
        dtype=policy_seq_logps_grouped.dtype,
    )
    if normalizer_total_group_weight is None:
        total_weight = float(scale.sum().item())
    else:
        total_weight = float(normalizer_total_group_weight)
    if total_weight <= 0.0:
        return torch.zeros_like(coeffs)

    coeffs = coeffs * (scale[:, None] / float(total_weight))
    return coeffs


def build_drx_target_bundle(
    *,
    utility_grouped: torch.Tensor,
    ref_seq_logps_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
    tau: float,
    competitive_mode_tau: float = 0.05,
    competitive_mode_gap: float = 0.10,
    competitive_mode_top_k: int = 3,
    competitive_mode_budget_grouped: torch.Tensor | None = None,
    competitive_mode_budget_max: float = 0.10,
    competitive_mode_intra_tau: float = 0.01,
    candidate_correctness_grouped: torch.Tensor | None = None,
    candidate_lengths_grouped: torch.Tensor | None = None,
    candidate_formatted_grouped: torch.Tensor | None = None,
    prompt_select_min_alpha_frac: float = 0.0,
    competitive_mode_positive_only: bool = False,
    verified_distinct_bonus_coef: float = 0.0,
    verified_distinct_min_modes: int = 2,
    verified_distinct_reward_threshold: float = 0.999,
    semantic_guard_max_expected_len_delta: float = float("inf"),
    semantic_guard_max_expected_format_drop: float = 0.0,
    candidate_kl_coef: float,
    cluster_ids_grouped: torch.Tensor | None = None,
    semantic_mass_weights_grouped: torch.Tensor | None = None,
    neutral_eps: float = 1e-8,
    neutral_projection_coef: float = 0.0,
    semantic_remix_mode: str = "competitive",
) -> DrXTargetBundle:
    """Build grouped DrX objects with optional competitive-mode semantic remixing.

    When ``cluster_ids_grouped`` is provided, the baseline DrX candidate posterior
    remains primary and semantic structure only reallocates a bounded residual
    budget over competitive semantic modes.
    """

    from .listwise import compute_listwise_weights_from_utilities
    from .semantic_remix import compute_semantic_cluster_weights_from_utilities

    if utility_grouped.shape != ref_seq_logps_grouped.shape:
        raise ValueError(
            "utility_grouped and ref_seq_logps_grouped must have matching shapes."
        )
    if utility_grouped.shape != valid_row_mask_grouped.shape:
        raise ValueError(
            "utility_grouped and valid_row_mask_grouped must have matching shapes."
        )

    informative_group_mask, neutral_group_mask, contributing_group_mask = (
        compute_drx_group_masks(
            utility_grouped=utility_grouped,
            valid_row_mask_grouped=valid_row_mask_grouped,
            neutral_eps=neutral_eps,
        )
    )

    semantic_diagnostics = None
    if cluster_ids_grouped is None:
        w_star_grouped = compute_listwise_weights_from_utilities(
            utility_grouped=utility_grouped,
            ref_seq_logps_grouped=ref_seq_logps_grouped,
            tau=tau,
            candidate_kl_coef=candidate_kl_coef,
            valid_row_mask_grouped=valid_row_mask_grouped,
        )
    else:
        w_star_grouped, semantic_diagnostics = (
            compute_semantic_cluster_weights_from_utilities(
                utility_grouped=utility_grouped,
                ref_seq_logps_grouped=ref_seq_logps_grouped,
                cluster_ids_grouped=cluster_ids_grouped,
                semantic_mass_weights_grouped=semantic_mass_weights_grouped,
                candidate_correctness_grouped=candidate_correctness_grouped,
                tau=tau,
                mode_tau=competitive_mode_tau,
                mode_gap=competitive_mode_gap,
                mode_top_k=competitive_mode_top_k,
                budget_grouped=competitive_mode_budget_grouped,
                budget_max=competitive_mode_budget_max,
                intra_tau=competitive_mode_intra_tau,
                candidate_lengths_grouped=candidate_lengths_grouped,
                candidate_formatted_grouped=candidate_formatted_grouped,
                candidate_kl_coef=candidate_kl_coef,
                prompt_select_min_alpha_frac=prompt_select_min_alpha_frac,
                positive_only=competitive_mode_positive_only,
                verified_distinct_bonus_coef=verified_distinct_bonus_coef,
                verified_distinct_min_modes=verified_distinct_min_modes,
                verified_distinct_reward_threshold=verified_distinct_reward_threshold,
                max_expected_len_delta=semantic_guard_max_expected_len_delta,
                max_expected_format_drop=semantic_guard_max_expected_format_drop,
                valid_row_mask_grouped=valid_row_mask_grouped,
                semantic_remix_mode=semantic_remix_mode,
            )
        )

    target_mass_grouped = (
        w_star_grouped * valid_row_mask_grouped.to(dtype=w_star_grouped.dtype)
    ).sum(dim=1)
    has_target_mass_mask = target_mass_grouped > 1e-8
    contributing_target_group_mask = contributing_group_mask & has_target_mass_mask
    effective_neutral_group_mask = neutral_group_mask & has_target_mass_mask
    token_group_mask = informative_group_mask & has_target_mass_mask
    # The projection sidecar is an optional weak neutral-group regularizer. When
    # disabled, neutral groups remain true no-op groups instead of contributing
    # zero-weighted targets that make diagnostics look active.
    projection_enabled = float(max(neutral_projection_coef, 0.0)) > 0.0
    projection_group_mask = effective_neutral_group_mask
    if not projection_enabled:
        projection_group_mask = torch.zeros_like(projection_group_mask)

    token_target_grouped = torch.where(
        token_group_mask[:, None],
        w_star_grouped,
        torch.zeros_like(w_star_grouped),
    )
    projection_target_grouped = torch.where(
        projection_group_mask[:, None],
        w_star_grouped,
        torch.zeros_like(w_star_grouped),
    )

    proj_scale = torch.full(
        (utility_grouped.size(0),),
        float(max(neutral_projection_coef, 0.0)),
        device=utility_grouped.device,
        dtype=utility_grouped.dtype,
    ) * projection_group_mask.to(dtype=utility_grouped.dtype)

    return DrXTargetBundle(
        utility_grouped=utility_grouped,
        w_star_grouped=w_star_grouped,
        token_target_grouped=token_target_grouped,
        projection_target_grouped=projection_target_grouped,
        informative_group_mask=token_group_mask,
        neutral_group_mask=effective_neutral_group_mask,
        contributing_group_mask=contributing_target_group_mask,
        projection_group_scale=proj_scale,
        semantic_diagnostics=semantic_diagnostics,
    )
