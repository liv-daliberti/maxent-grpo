"""Dr.X target bundle and projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
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


@dataclass
class SequenceAuxProjectionDiagnostics:
    """Pre-backprop diagnostics for the sequence projection target."""

    eligible_group_mask: torch.Tensor
    kept_group_mask: torch.Tensor
    rejected_group_filter_group_mask: torch.Tensor
    rejected_len_guard_group_mask: torch.Tensor
    rejected_len_gain_guard_group_mask: torch.Tensor
    rejected_format_guard_group_mask: torch.Tensor
    rejected_correctness_guard_group_mask: torch.Tensor
    all_wrong_group_mask: torch.Tensor
    mixed_group_mask: torch.Tensor
    all_correct_group_mask: torch.Tensor
    target_mass_grouped: torch.Tensor
    behavior_mass_grouped: torch.Tensor
    target_expected_len_grouped: torch.Tensor
    behavior_expected_len_grouped: torch.Tensor
    target_expected_format_grouped: torch.Tensor
    behavior_expected_format_grouped: torch.Tensor
    target_expected_correctness_grouped: torch.Tensor
    behavior_expected_correctness_grouped: torch.Tensor
    target_correct_mass_grouped: torch.Tensor
    behavior_correct_mass_grouped: torch.Tensor
    target_wrong_mass_grouped: torch.Tensor
    behavior_wrong_mass_grouped: torch.Tensor
    moved_mass_l1_grouped: torch.Tensor
    moved_mass_to_correct_grouped: torch.Tensor
    moved_mass_from_correct_grouped: torch.Tensor
    moved_mass_to_wrong_grouped: torch.Tensor
    moved_mass_from_wrong_grouped: torch.Tensor


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


def _normalize_grouped_probs(
    probs_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probs_grouped.shape != valid_row_mask_grouped.shape:
        raise ValueError("probs_grouped must match valid_row_mask_grouped.")
    valid_mask = valid_row_mask_grouped.to(torch.bool)
    safe_probs = torch.where(
        valid_mask.to(device=probs_grouped.device),
        probs_grouped,
        torch.zeros_like(probs_grouped),
    )
    mass_grouped = safe_probs.sum(dim=1)
    normalized = torch.where(
        mass_grouped[:, None] > float(eps),
        safe_probs / mass_grouped[:, None].clamp(min=float(eps)),
        torch.zeros_like(safe_probs),
    )
    return normalized, mass_grouped


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
    if str(advantage_source) in {"utility_centered", "maxent_centered"}:
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


def build_token_primary_sequence_aux_projection(
    *,
    token_target_grouped: torch.Tensor,
    projection_target_grouped: torch.Tensor,
    informative_group_mask: torch.Tensor,
    projection_group_scale: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact MaxEnt sequence-aux target for token-primary Dr.X."""

    if token_target_grouped.shape != projection_target_grouped.shape:
        raise ValueError(
            "token_target_grouped and projection_target_grouped must match."
        )
    if token_target_grouped.shape != valid_row_mask_grouped.shape:
        raise ValueError("valid_row_mask_grouped must match grouped targets.")
    if informative_group_mask.dim() != 1:
        raise ValueError("informative_group_mask must be one-dimensional.")
    if int(informative_group_mask.numel()) != int(token_target_grouped.size(0)):
        raise ValueError("informative_group_mask must have one entry per group.")
    if projection_group_scale.dim() != 1 or int(projection_group_scale.numel()) != int(
        token_target_grouped.size(0)
    ):
        raise ValueError("projection_group_scale must have one entry per group.")

    valid_mask = valid_row_mask_grouped.to(torch.bool)
    informative_mask = informative_group_mask.to(
        device=token_target_grouped.device,
        dtype=torch.bool,
    )
    target_grouped = torch.where(
        informative_mask[:, None],
        token_target_grouped,
        projection_target_grouped,
    )
    target_grouped = torch.where(
        valid_mask.to(device=target_grouped.device),
        target_grouped,
        torch.zeros_like(target_grouped),
    )

    projection_scale = projection_group_scale.to(
        device=target_grouped.device,
        dtype=target_grouped.dtype,
    )
    informative_scale = torch.ones_like(projection_scale)
    group_scale = torch.where(
        informative_mask.to(device=projection_scale.device),
        informative_scale,
        projection_scale,
    )
    return target_grouped, group_scale


def apply_sequence_aux_projection_gates(
    *,
    sequence_aux_target_grouped: torch.Tensor,
    sequence_aux_group_scale: torch.Tensor,
    behavior_probs_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
    candidate_correctness_grouped: torch.Tensor | None = None,
    candidate_lengths_grouped: torch.Tensor | None = None,
    candidate_formatted_grouped: torch.Tensor | None = None,
    group_filter: str = "all",
    max_expected_len_drop: float = float("inf"),
    max_expected_len_gain: float = float("inf"),
    max_expected_format_drop: float = 1.0,
    min_expected_correctness_delta: float = -1.0,
    correctness_threshold: float = 0.999,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, SequenceAuxProjectionDiagnostics]:
    """Gate sequence-aux projection targets and report target-vs-behavior stats.

    The helper is intentionally conservative: rejected groups have their target
    mass and CE scale zeroed instead of being renormalized into the remaining
    groups. This keeps the aux coefficient as a real strength knob.
    """

    if sequence_aux_target_grouped.shape != valid_row_mask_grouped.shape:
        raise ValueError("sequence_aux_target_grouped must match valid rows.")
    if behavior_probs_grouped.shape != sequence_aux_target_grouped.shape:
        raise ValueError("behavior_probs_grouped must match sequence aux target.")
    if sequence_aux_group_scale.dim() != 1 or int(
        sequence_aux_group_scale.numel()
    ) != int(sequence_aux_target_grouped.size(0)):
        raise ValueError("sequence_aux_group_scale must have one entry per group.")
    if (
        candidate_correctness_grouped is not None
        and candidate_correctness_grouped.shape != sequence_aux_target_grouped.shape
    ):
        raise ValueError("candidate_correctness_grouped must match grouped target.")
    if (
        candidate_lengths_grouped is not None
        and candidate_lengths_grouped.shape != sequence_aux_target_grouped.shape
    ):
        raise ValueError("candidate_lengths_grouped must match grouped target.")
    if (
        candidate_formatted_grouped is not None
        and candidate_formatted_grouped.shape != sequence_aux_target_grouped.shape
    ):
        raise ValueError("candidate_formatted_grouped must match grouped target.")

    group_filter_normalized = str(group_filter).strip().lower()
    if group_filter_normalized not in {"all", "mixed", "has_correct"}:
        raise ValueError(
            "group_filter must be one of: all, mixed, has_correct"
        )

    valid_mask = valid_row_mask_grouped.to(torch.bool)
    target_grouped = torch.where(
        valid_mask.to(device=sequence_aux_target_grouped.device),
        sequence_aux_target_grouped,
        torch.zeros_like(sequence_aux_target_grouped),
    )
    target_probs_grouped, target_mass_grouped = _normalize_grouped_probs(
        target_grouped,
        valid_mask,
        eps=eps,
    )
    behavior_probs_grouped, behavior_mass_grouped = _normalize_grouped_probs(
        behavior_probs_grouped.to(
            device=sequence_aux_target_grouped.device,
            dtype=sequence_aux_target_grouped.dtype,
        ),
        valid_mask,
        eps=eps,
    )

    if candidate_correctness_grouped is None:
        candidate_correctness = torch.zeros_like(target_probs_grouped)
    else:
        candidate_correctness = candidate_correctness_grouped.to(
            device=target_probs_grouped.device,
            dtype=target_probs_grouped.dtype,
        )
    if candidate_lengths_grouped is None:
        candidate_lengths = torch.zeros_like(target_probs_grouped)
    else:
        candidate_lengths = candidate_lengths_grouped.to(
            device=target_probs_grouped.device,
            dtype=target_probs_grouped.dtype,
        )
    if candidate_formatted_grouped is None:
        candidate_formatted = torch.zeros_like(target_probs_grouped)
    else:
        candidate_formatted = candidate_formatted_grouped.to(
            device=target_probs_grouped.device,
            dtype=target_probs_grouped.dtype,
        )

    correct_mask = (
        candidate_correctness >= float(correctness_threshold)
    ) & valid_mask.to(device=target_probs_grouped.device)
    valid_count_grouped = valid_mask.to(device=target_probs_grouped.device).sum(dim=1)
    correct_count_grouped = correct_mask.to(torch.int64).sum(dim=1)
    has_valid_group_mask = valid_count_grouped > 0
    all_wrong_group_mask = has_valid_group_mask & (correct_count_grouped == 0)
    all_correct_group_mask = has_valid_group_mask & (
        correct_count_grouped == valid_count_grouped
    )
    mixed_group_mask = (
        has_valid_group_mask
        & (correct_count_grouped > 0)
        & (correct_count_grouped < valid_count_grouped)
    )
    has_correct_group_mask = has_valid_group_mask & (correct_count_grouped > 0)

    target_expected_len_grouped = (target_probs_grouped * candidate_lengths).sum(dim=1)
    behavior_expected_len_grouped = (
        behavior_probs_grouped * candidate_lengths
    ).sum(dim=1)
    target_expected_format_grouped = (
        target_probs_grouped * candidate_formatted
    ).sum(dim=1)
    behavior_expected_format_grouped = (
        behavior_probs_grouped * candidate_formatted
    ).sum(dim=1)
    target_expected_correctness_grouped = (
        target_probs_grouped * candidate_correctness
    ).sum(dim=1)
    behavior_expected_correctness_grouped = (
        behavior_probs_grouped * candidate_correctness
    ).sum(dim=1)

    correct_mask_f = correct_mask.to(target_probs_grouped.dtype)
    wrong_mask_f = (
        valid_mask.to(device=target_probs_grouped.device) & (~correct_mask)
    ).to(target_probs_grouped.dtype)
    target_correct_mass_grouped = (target_probs_grouped * correct_mask_f).sum(dim=1)
    behavior_correct_mass_grouped = (behavior_probs_grouped * correct_mask_f).sum(
        dim=1
    )
    target_wrong_mass_grouped = (target_probs_grouped * wrong_mask_f).sum(dim=1)
    behavior_wrong_mass_grouped = (behavior_probs_grouped * wrong_mask_f).sum(dim=1)

    mass_delta_grouped = target_probs_grouped - behavior_probs_grouped
    moved_mass_l1_grouped = 0.5 * mass_delta_grouped.abs().sum(dim=1)
    moved_mass_to_correct_grouped = (
        mass_delta_grouped.clamp(min=0.0) * correct_mask_f
    ).sum(dim=1)
    moved_mass_from_correct_grouped = (
        (-mass_delta_grouped).clamp(min=0.0) * correct_mask_f
    ).sum(dim=1)
    moved_mass_to_wrong_grouped = (
        mass_delta_grouped.clamp(min=0.0) * wrong_mask_f
    ).sum(dim=1)
    moved_mass_from_wrong_grouped = (
        (-mass_delta_grouped).clamp(min=0.0) * wrong_mask_f
    ).sum(dim=1)

    group_scale = sequence_aux_group_scale.to(
        device=target_probs_grouped.device,
        dtype=target_probs_grouped.dtype,
    )
    eligible_group_mask = (
        has_valid_group_mask
        & (target_mass_grouped > float(eps))
        & (group_scale > 0.0)
    )
    if group_filter_normalized == "mixed":
        filter_group_mask = mixed_group_mask
    elif group_filter_normalized == "has_correct":
        filter_group_mask = has_correct_group_mask
    else:
        filter_group_mask = has_valid_group_mask

    safe_max_expected_len_drop = float(max_expected_len_drop)
    if math.isfinite(safe_max_expected_len_drop):
        if safe_max_expected_len_drop < 0:
            raise ValueError("max_expected_len_drop must be non-negative.")
        len_guard_group_mask = (
            target_expected_len_grouped + 1e-6
        ) >= behavior_expected_len_grouped - safe_max_expected_len_drop
    else:
        len_guard_group_mask = torch.ones_like(eligible_group_mask)

    safe_max_expected_len_gain = float(max_expected_len_gain)
    if math.isfinite(safe_max_expected_len_gain):
        if safe_max_expected_len_gain < 0:
            raise ValueError("max_expected_len_gain must be non-negative.")
        len_gain_guard_group_mask = (
            target_expected_len_grouped
        ) <= behavior_expected_len_grouped + safe_max_expected_len_gain + 1e-6
    else:
        len_gain_guard_group_mask = torch.ones_like(eligible_group_mask)

    safe_max_expected_format_drop = float(max_expected_format_drop)
    if math.isfinite(safe_max_expected_format_drop):
        if safe_max_expected_format_drop < 0:
            raise ValueError("max_expected_format_drop must be non-negative.")
        format_guard_group_mask = (
            target_expected_format_grouped + safe_max_expected_format_drop + 1e-6
        ) >= behavior_expected_format_grouped
    else:
        format_guard_group_mask = torch.ones_like(eligible_group_mask)

    safe_min_expected_correctness_delta = float(min_expected_correctness_delta)
    if (
        safe_min_expected_correctness_delta < -1.0
        or safe_min_expected_correctness_delta > 1.0
    ):
        raise ValueError("min_expected_correctness_delta must be in [-1, 1].")
    correctness_guard_group_mask = (
        target_expected_correctness_grouped + 1e-6
    ) >= behavior_expected_correctness_grouped + safe_min_expected_correctness_delta

    kept_group_mask = (
        eligible_group_mask
        & filter_group_mask
        & len_guard_group_mask
        & len_gain_guard_group_mask
        & format_guard_group_mask
        & correctness_guard_group_mask
    )
    rejected_group_filter_group_mask = eligible_group_mask & (~filter_group_mask)
    rejected_len_guard_group_mask = (
        eligible_group_mask & filter_group_mask & (~len_guard_group_mask)
    )
    rejected_len_gain_guard_group_mask = (
        eligible_group_mask
        & filter_group_mask
        & len_guard_group_mask
        & (~len_gain_guard_group_mask)
    )
    rejected_format_guard_group_mask = (
        eligible_group_mask
        & filter_group_mask
        & len_guard_group_mask
        & len_gain_guard_group_mask
        & (~format_guard_group_mask)
    )
    rejected_correctness_guard_group_mask = (
        eligible_group_mask
        & filter_group_mask
        & len_guard_group_mask
        & len_gain_guard_group_mask
        & format_guard_group_mask
        & (~correctness_guard_group_mask)
    )

    gated_target_grouped = torch.where(
        kept_group_mask[:, None],
        target_grouped,
        torch.zeros_like(target_grouped),
    )
    gated_group_scale = torch.where(
        kept_group_mask,
        group_scale,
        torch.zeros_like(group_scale),
    )
    diagnostics = SequenceAuxProjectionDiagnostics(
        eligible_group_mask=eligible_group_mask,
        kept_group_mask=kept_group_mask,
        rejected_group_filter_group_mask=rejected_group_filter_group_mask,
        rejected_len_guard_group_mask=rejected_len_guard_group_mask,
        rejected_len_gain_guard_group_mask=rejected_len_gain_guard_group_mask,
        rejected_format_guard_group_mask=rejected_format_guard_group_mask,
        rejected_correctness_guard_group_mask=rejected_correctness_guard_group_mask,
        all_wrong_group_mask=all_wrong_group_mask,
        mixed_group_mask=mixed_group_mask,
        all_correct_group_mask=all_correct_group_mask,
        target_mass_grouped=target_mass_grouped,
        behavior_mass_grouped=behavior_mass_grouped,
        target_expected_len_grouped=target_expected_len_grouped,
        behavior_expected_len_grouped=behavior_expected_len_grouped,
        target_expected_format_grouped=target_expected_format_grouped,
        behavior_expected_format_grouped=behavior_expected_format_grouped,
        target_expected_correctness_grouped=target_expected_correctness_grouped,
        behavior_expected_correctness_grouped=behavior_expected_correctness_grouped,
        target_correct_mass_grouped=target_correct_mass_grouped,
        behavior_correct_mass_grouped=behavior_correct_mass_grouped,
        target_wrong_mass_grouped=target_wrong_mass_grouped,
        behavior_wrong_mass_grouped=behavior_wrong_mass_grouped,
        moved_mass_l1_grouped=moved_mass_l1_grouped,
        moved_mass_to_correct_grouped=moved_mass_to_correct_grouped,
        moved_mass_from_correct_grouped=moved_mass_from_correct_grouped,
        moved_mass_to_wrong_grouped=moved_mass_to_wrong_grouped,
        moved_mass_from_wrong_grouped=moved_mass_from_wrong_grouped,
    )
    return gated_target_grouped, gated_group_scale, diagnostics


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
    answer_ids_grouped: torch.Tensor | None = None,
    neutral_eps: float = 1e-8,
    neutral_projection_coef: float = 0.0,
    semantic_remix_mode: str = "competitive",
    semantic_correctness_answer_level: bool = False,
    semantic_correctness_min_answer_count: int = 1,
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
                answer_ids_grouped=answer_ids_grouped,
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
                correctness_answer_level=semantic_correctness_answer_level,
                correctness_min_answer_count=semantic_correctness_min_answer_count,
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
