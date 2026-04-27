"""PPO clipping helpers for listwise and token-level Dr.X updates."""

from __future__ import annotations

import math

import torch


def _coerce_non_negative_float(value: object, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return max(numeric, 0.0)


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


def compute_listwise_clip_advantages(
    *,
    weights_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor | None = None,
    baseline_value: float | None = None,
    baseline_grouped: torch.Tensor | None = None,
    reward_mass_grouped: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return prompt-local clip advantages, optionally preserving reward mass."""

    if valid_row_mask_grouped is None:
        valid_group_mask = torch.ones_like(weights_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != weights_grouped.shape:
            raise ValueError(
                "valid_row_mask_grouped must match the grouped weight shape."
            )
        valid_group_mask = valid_row_mask_grouped.to(torch.bool)

    if baseline_grouped is not None:
        if baseline_grouped.shape != weights_grouped.shape:
            raise ValueError("baseline_grouped must match the grouped weight shape.")
        baseline = baseline_grouped.to(
            device=weights_grouped.device,
            dtype=weights_grouped.dtype,
        )
    else:
        baseline = torch.full_like(
            weights_grouped,
            1.0 / float(max(weights_grouped.size(1), 1))
            if baseline_value is None
            else float(baseline_value),
        )

    clip_adv = torch.where(
        valid_group_mask,
        weights_grouped - baseline,
        torch.zeros_like(weights_grouped),
    )

    if reward_mass_grouped is not None:
        if reward_mass_grouped.ndim != 2 or int(reward_mass_grouped.size(0)) != int(
            weights_grouped.size(0)
        ):
            raise ValueError(
                "reward_mass_grouped must have shape [num_groups, 1] or "
                "[num_groups, group_size]."
            )
        if int(reward_mass_grouped.size(1)) == 1:
            reward_mass_grouped = reward_mass_grouped.expand(
                -1,
                int(weights_grouped.size(1)),
            )
        elif reward_mass_grouped.shape != weights_grouped.shape:
            raise ValueError(
                "reward_mass_grouped must have shape [num_groups, 1] or "
                "[num_groups, group_size]."
            )
        reward_mass_grouped = reward_mass_grouped.to(
            device=weights_grouped.device,
            dtype=weights_grouped.dtype,
        )
        clip_adv = clip_adv * torch.where(
            valid_group_mask,
            reward_mass_grouped,
            torch.zeros_like(reward_mass_grouped),
        )

    return clip_adv


def compute_sequence_clip_coefficients(
    *,
    policy_seq_logps_grouped: torch.Tensor,
    behavior_seq_logps_grouped: torch.Tensor,
    row_advantages_grouped: torch.Tensor,
    active_group_mask: torch.Tensor,
    normalizer_active_group_count: int | None = None,
    valid_row_mask_grouped: torch.Tensor | None = None,
    clip_low: float = 0.0,
    clip_high: float = 0.0,
) -> torch.Tensor:
    """Return exact d(loss)/d(seq_logp) coefficients for sequence-level PPO clip."""

    if policy_seq_logps_grouped.shape != behavior_seq_logps_grouped.shape:
        raise ValueError(
            "policy_seq_logps_grouped and behavior_seq_logps_grouped must have matching shapes."
        )
    if policy_seq_logps_grouped.shape != row_advantages_grouped.shape:
        raise ValueError(
            "policy_seq_logps_grouped and row_advantages_grouped must have matching shapes."
        )
    if active_group_mask.dim() != 1 or int(active_group_mask.numel()) != int(
        policy_seq_logps_grouped.size(0)
    ):
        raise ValueError("active_group_mask must match the prompt-group dimension.")
    if valid_row_mask_grouped is None:
        valid_group_mask = torch.ones_like(row_advantages_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != row_advantages_grouped.shape:
            raise ValueError(
                "valid_row_mask_grouped must match the grouped advantage shape."
            )
        valid_group_mask = valid_row_mask_grouped.to(torch.bool)

    local_active_count = int(active_group_mask.to(torch.int64).sum().item())
    if local_active_count <= 0:
        return torch.zeros_like(policy_seq_logps_grouped)
    if normalizer_active_group_count is None:
        active_count = local_active_count
    else:
        active_count = max(int(normalizer_active_group_count), 0)
    if active_count <= 0:
        raise ValueError(
            "normalizer_active_group_count must be positive when active groups exist."
        )

    safe_clip_low = _coerce_non_negative_float(clip_low, default=0.0)
    safe_clip_high = _coerce_non_negative_float(clip_high, default=0.0)
    active_scale = active_group_mask.to(
        device=policy_seq_logps_grouped.device,
        dtype=policy_seq_logps_grouped.dtype,
    ).unsqueeze(1) / float(active_count)
    log_seq_ratio = (policy_seq_logps_grouped - behavior_seq_logps_grouped).clamp(
        -40.0, 40.0
    )
    seq_ratio = torch.exp(log_seq_ratio).to(policy_seq_logps_grouped.dtype)
    row_advantages_grouped = row_advantages_grouped.to(
        device=policy_seq_logps_grouped.device,
        dtype=policy_seq_logps_grouped.dtype,
    )
    clipped_region = (
        (seq_ratio > 1.0 + safe_clip_high) & (row_advantages_grouped > 0.0)
    ) | ((seq_ratio < 1.0 - safe_clip_low) & (row_advantages_grouped < 0.0))
    coeffs = -seq_ratio * row_advantages_grouped
    coeffs = torch.where(clipped_region, torch.zeros_like(coeffs), coeffs)
    coeffs = coeffs * active_scale
    return torch.where(valid_group_mask, coeffs, torch.zeros_like(coeffs))


def compute_listwise_sequence_coefficients(
    *,
    policy_seq_logps_grouped: torch.Tensor,
    weights_grouped: torch.Tensor,
    active_group_mask: torch.Tensor,
    normalizer_active_group_count: int | None = None,
    valid_row_mask_grouped: torch.Tensor | None = None,
    behavior_seq_logps_grouped: torch.Tensor | None = None,
    clip_row_mask_grouped: torch.Tensor | None = None,
    clip_low: float = 0.0,
    clip_high: float = 0.0,
    clip_coef: float = 0.0,
    baseline_value: float | None = None,
    baseline_grouped: torch.Tensor | None = None,
    reward_mass_grouped: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact d(loss)/d(seq_logp) coefficients for listwise MaxEnt."""

    if policy_seq_logps_grouped.shape != weights_grouped.shape:
        raise ValueError(
            "policy_seq_logps_grouped and weights_grouped must have matching shapes."
        )
    if active_group_mask.dim() != 1 or int(active_group_mask.numel()) != int(
        policy_seq_logps_grouped.size(0)
    ):
        raise ValueError("active_group_mask must match the prompt-group dimension.")
    if (
        valid_row_mask_grouped is not None
        and valid_row_mask_grouped.shape != weights_grouped.shape
    ):
        raise ValueError("valid_row_mask_grouped must match the grouped weight shape.")
    if (
        clip_row_mask_grouped is not None
        and clip_row_mask_grouped.shape != weights_grouped.shape
    ):
        raise ValueError("clip_row_mask_grouped must match the grouped weight shape.")
    if baseline_grouped is not None and baseline_grouped.shape != weights_grouped.shape:
        raise ValueError("baseline_grouped must match the grouped weight shape.")
    if clip_coef > 0.0:
        if behavior_seq_logps_grouped is None:
            raise ValueError(
                "behavior_seq_logps_grouped is required when clip_coef is positive."
            )
        if behavior_seq_logps_grouped.shape != policy_seq_logps_grouped.shape:
            raise ValueError(
                "behavior_seq_logps_grouped and policy_seq_logps_grouped must "
                "have matching shapes."
            )

    if valid_row_mask_grouped is None:
        valid_group_mask = torch.ones_like(weights_grouped, dtype=torch.bool)
    else:
        valid_group_mask = valid_row_mask_grouped.to(torch.bool)
    local_active_count = int(active_group_mask.to(torch.int64).sum().item())
    if local_active_count <= 0:
        return torch.zeros_like(policy_seq_logps_grouped)
    if normalizer_active_group_count is None:
        active_count = local_active_count
    else:
        active_count = max(int(normalizer_active_group_count), 0)
    if active_count <= 0:
        raise ValueError(
            "normalizer_active_group_count must be positive when active groups exist."
        )

    active_scale = active_group_mask.to(
        device=policy_seq_logps_grouped.device,
        dtype=policy_seq_logps_grouped.dtype,
    ).unsqueeze(1) / float(active_count)
    policy_log_probs_grouped = _masked_group_log_softmax(
        policy_seq_logps_grouped,
        valid_group_mask,
    )
    policy_probs_grouped = torch.where(
        valid_group_mask,
        torch.exp(policy_log_probs_grouped),
        torch.zeros_like(policy_log_probs_grouped),
    )
    target_mass_grouped = (
        (weights_grouped * valid_group_mask.to(weights_grouped.dtype))
        .sum(dim=1, keepdim=True)
        .to(policy_seq_logps_grouped.dtype)
    )
    coeffs = (
        target_mass_grouped * policy_probs_grouped - weights_grouped
    ) * active_scale
    coeffs = torch.where(valid_group_mask, coeffs, torch.zeros_like(coeffs))

    safe_clip_coef = _coerce_non_negative_float(clip_coef, default=0.0)
    if safe_clip_coef > 0.0:
        clip_adv = compute_listwise_clip_advantages(
            weights_grouped=weights_grouped,
            valid_row_mask_grouped=clip_row_mask_grouped,
            baseline_value=baseline_value,
            baseline_grouped=baseline_grouped,
            reward_mass_grouped=reward_mass_grouped,
        ).to(
            device=policy_seq_logps_grouped.device,
            dtype=policy_seq_logps_grouped.dtype,
        )
        log_seq_ratio = (policy_seq_logps_grouped - behavior_seq_logps_grouped).clamp(
            -40.0, 40.0
        )
        seq_ratio = torch.exp(log_seq_ratio).to(policy_seq_logps_grouped.dtype)
        clipped_region = ((seq_ratio > 1.0 + float(clip_high)) & (clip_adv > 0.0)) | (
            (seq_ratio < 1.0 - float(clip_low)) & (clip_adv < 0.0)
        )
        clip_grad = -seq_ratio * clip_adv
        clip_grad = torch.where(clipped_region, torch.zeros_like(clip_grad), clip_grad)
        coeffs = coeffs + (safe_clip_coef * clip_grad * active_scale)

    return coeffs


def compute_token_level_clip_loss(
    *,
    new_logps: torch.Tensor,
    behavior_logps: torch.Tensor,
    response_masks: torch.Tensor,
    row_advantages: torch.Tensor,
    clip_low: float,
    clip_high: float,
    constant_normalizer: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-row PPO-style token clip losses plus clipping masks."""

    if new_logps.shape != behavior_logps.shape:
        raise ValueError("new_logps and behavior_logps must have matching shapes.")
    if new_logps.shape != response_masks.shape:
        raise ValueError("response_masks must match the log-prob tensor shape.")
    if row_advantages.dim() == 2 and int(row_advantages.size(1)) == 1:
        row_advantages = row_advantages.squeeze(1)
    if row_advantages.dim() != 1 or int(row_advantages.numel()) != int(
        new_logps.size(0)
    ):
        raise ValueError("row_advantages must provide one value per row.")

    safe_clip_low = _coerce_non_negative_float(clip_low, default=0.0)
    safe_clip_high = _coerce_non_negative_float(clip_high, default=0.0)

    log_ratio = (new_logps - behavior_logps).clamp(-40.0, 40.0)
    ratio = torch.exp(log_ratio).to(new_logps.dtype)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - safe_clip_low,
        1.0 + safe_clip_high,
    )
    row_advantages = row_advantages.to(device=new_logps.device, dtype=new_logps.dtype)
    token_advantages = row_advantages.unsqueeze(1)
    clip_objective = torch.min(
        ratio * token_advantages,
        clipped_ratio * token_advantages,
    )
    per_token_loss = -clip_objective

    response_mask_float = response_masks.to(dtype=new_logps.dtype)
    if (
        isinstance(constant_normalizer, (int, float))
        and math.isfinite(float(constant_normalizer))
        and float(constant_normalizer) > 0.0
    ):
        per_row_loss = (per_token_loss * response_mask_float).sum(dim=1) / float(
            constant_normalizer
        )
    else:
        per_row_loss = (per_token_loss * response_mask_float).sum(dim=1) / (
            response_mask_float.sum(dim=1).clamp(min=1.0)
        )

    token_advantages_mask = token_advantages.expand_as(ratio)
    is_low_clipped = (
        (ratio < 1.0 - safe_clip_low)
        & (token_advantages_mask < 0.0)
        & response_masks.to(torch.bool)
    )
    is_high_clipped = (
        (ratio > 1.0 + safe_clip_high)
        & (token_advantages_mask > 0.0)
        & response_masks.to(torch.bool)
    )
    return per_row_loss, ratio, is_low_clipped, is_high_clipped
