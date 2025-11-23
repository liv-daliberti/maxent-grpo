"""Loss computation helpers for the MaxEnt-GRPO training loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from .run_helpers import require_torch
from .run_training_types import (
    BatchDiagnostics,
    ClipSettings,
    LossOutputs,
    LossScalarBundle,
    ReferenceLogprobs,
    WeightStats,
    WeightingSettings,
)

torch = require_torch("training_loss")
Tensor = torch.Tensor


@dataclass
class SequenceScores:
    """Bundle sequence-level log-prob statistics."""

    cur_logp_sum: Tensor
    behavior_logp_sum: Tensor
    log_ratio_train: Tensor
    denom_tok_tensor: Tensor


@dataclass
class GroupLossData:
    """Per-group tensors required for the policy loss."""

    group_sizes: List[int]
    weight_tensor: Tensor
    logp_sums: Tensor
    token_counts: Tensor


@dataclass
class RatioContext:
    """Inputs needed to compute KL and diagnostics."""

    log_ratio_train: Tensor
    denom_tok_tensor: Tensor
    clip_cfg: ClipSettings
    weighting_cfg: WeightingSettings
    ref_stats: ReferenceLogprobs
    cur_logp_sum: Tensor
    behavior_logp_sum: Tensor


@dataclass
class LossInputConfig:
    """Configurations shared when constructing loss inputs."""

    clip_cfg: ClipSettings
    weighting_cfg: WeightingSettings
    ref_stats: ReferenceLogprobs


@dataclass
class _LossScalarInputs:
    """Helper container for scalar KL/clip contributions."""

    clip_loss_scalar: Optional[float]
    kl_loss_scalar: float
    weighted_kl_loss_scalar: float


def build_loss_inputs(
    grouped_completions: List[List[str]],
    weight_stats: WeightStats,
    scores: SequenceScores,
    config: LossInputConfig,
) -> Tuple[GroupLossData, RatioContext]:
    """Return helper dataclasses that aggregate loss inputs."""
    group_sizes = [len(group) for group in grouped_completions]
    total_completions = sum(group_sizes)
    weight_tensor = torch.tensor(
        weight_stats.flat_weights,
        device=scores.cur_logp_sum.device,
        dtype=scores.cur_logp_sum.dtype,
    ).view(-1)
    if (
        total_completions != scores.cur_logp_sum.numel()
        or weight_tensor.numel() != total_completions
    ):
        raise ValueError("Mismatch between weights and log-prob dimensions.")
    group_data = GroupLossData(
        group_sizes=group_sizes,
        weight_tensor=weight_tensor,
        logp_sums=scores.cur_logp_sum,
        token_counts=scores.denom_tok_tensor.to(scores.cur_logp_sum.device),
    )
    ratio_context = RatioContext(
        log_ratio_train=scores.log_ratio_train,
        denom_tok_tensor=scores.denom_tok_tensor,
        clip_cfg=config.clip_cfg,
        weighting_cfg=config.weighting_cfg,
        ref_stats=config.ref_stats,
        cur_logp_sum=scores.cur_logp_sum,
        behavior_logp_sum=scores.behavior_logp_sum,
    )
    return group_data, ratio_context


def _policy_loss_from_groups(group_data: GroupLossData) -> torch.Tensor:
    """Return the mean policy loss aggregated over prompt groups."""
    policy_group_losses: List[torch.Tensor] = []
    offset = 0
    for size in group_data.group_sizes:
        if size <= 0:
            continue
        end = offset + size
        w_slice = group_data.weight_tensor[offset:end]
        logp_slice = group_data.logp_sums[offset:end]
        token_slice = group_data.token_counts[offset:end].clamp(min=1.0)
        normalized_logp = logp_slice / token_slice
        policy_group_losses.append((-normalized_logp * w_slice).sum())
        offset = end
    if not policy_group_losses:
        raise ValueError("No completions available for loss computation.")
    return torch.stack(policy_group_losses).mean()


def _iter_group_offsets(group_sizes: List[int]) -> Iterator[Tuple[int, int]]:
    """Yield (offset, size) pairs for each completion group."""
    offset = 0
    for size in group_sizes:
        yield offset, size
        offset += size


def _clip_loss_for_slice(
    weight_tensor: Tensor,
    ratio_slice: Tensor,
    clipped_slice: Tensor,
    adv_base_val: float,
) -> torch.Tensor:
    """Return the PPO-style clip loss for a contiguous slice."""
    adv_tensor = weight_tensor - adv_base_val
    obj_unclipped = ratio_slice * adv_tensor
    obj_clipped = clipped_slice * adv_tensor
    obj = torch.where(
        adv_tensor >= 0,
        torch.minimum(obj_unclipped, obj_clipped),
        torch.maximum(obj_unclipped, obj_clipped),
    )
    return -(obj.sum())


def _apply_clip_objective(
    ratio_ctx: RatioContext,
    group_data: GroupLossData,
    policy_loss: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[float]]:
    """Apply the optional clip objective and return updated loss/scalar."""
    clip_cfg = ratio_ctx.clip_cfg
    if not (clip_cfg.use_clip_objective and clip_cfg.clip_range > 0.0):
        return policy_loss, None
    ratio_for_loss, clipped_ratio_vals = _compute_clip_ratios(ratio_ctx, clip_cfg)
    clip_losses: List[torch.Tensor] = []
    for offset, size in _iter_group_offsets(group_data.group_sizes):
        if size <= 0:
            continue
        end = offset + size
        adv_base_val = (
            clip_cfg.clip_adv_baseline
            if clip_cfg.clip_adv_baseline is not None
            else (1.0 / float(max(size, 1)))
        )
        clip_losses.append(
            _clip_loss_for_slice(
                group_data.weight_tensor[offset:end],
                ratio_for_loss[offset:end],
                clipped_ratio_vals[offset:end],
                adv_base_val,
            )
        )
    if not clip_losses:
        return policy_loss, None
    clip_loss_tensor = torch.stack(clip_losses).mean()
    updated_loss = policy_loss + clip_cfg.clip_objective_coef * clip_loss_tensor
    clip_loss_scalar = float(clip_loss_tensor.detach().float().cpu())
    return updated_loss, clip_loss_scalar


def _kl_terms(ratio_ctx: RatioContext) -> Tuple[torch.Tensor, float, float]:
    """Return KL tensor, scalar, and weighted scalar contributions.

    We mirror TRL's GRPOTrainer by using the always-non-negative scalar
    ``exp(ref_logp - cur_logp) - (ref_logp - cur_logp) - 1`` computed on
    *length-normalized* per-token log-probabilities, then aggregate with a
    token-weighted mean to avoid overweighting short completions.
    """
    denom = ratio_ctx.denom_tok_tensor.clamp(min=1.0)
    cur_logp_per_tok = ratio_ctx.cur_logp_sum / denom
    if ratio_ctx.weighting_cfg.len_norm_ref:
        ref_logp_per_tok = ratio_ctx.ref_stats.ref_logp_sum
    else:
        ref_logp_per_tok = ratio_ctx.ref_stats.ref_logp_sum_raw / denom
    delta = (ref_logp_per_tok - cur_logp_per_tok).clamp(min=-60.0, max=60.0)
    per_seq_kl = delta.exp() - delta - 1.0
    # Weight by reference token counts so short completions do not dominate.
    ref_tok_weights = ratio_ctx.ref_stats.ref_tok_counts.detach().clamp(min=1.0).to(per_seq_kl)
    denom_weight = ref_tok_weights.sum().clamp(min=1.0)
    kl_loss_tensor = (per_seq_kl * ref_tok_weights).sum() / denom_weight
    kl_loss_scalar = float(kl_loss_tensor.detach().float().cpu())
    weighted_kl_loss_scalar = ratio_ctx.weighting_cfg.beta * kl_loss_scalar
    return kl_loss_tensor, kl_loss_scalar, weighted_kl_loss_scalar


def _compute_clip_ratios(
    ratio_ctx: RatioContext,
    clip_cfg: ClipSettings,
) -> Tuple[Tensor, Tensor]:
    """Return raw and clipped PPO ratios."""
    behavior_log_ratio = ratio_ctx.cur_logp_sum - ratio_ctx.behavior_logp_sum
    ratio_for_loss = behavior_log_ratio.clamp(min=-60.0, max=60.0).exp()
    clipped_ratio_vals = torch.clamp(
        ratio_for_loss,
        1.0 - clip_cfg.clip_range,
        1.0 + clip_cfg.clip_range,
    )
    return ratio_for_loss, clipped_ratio_vals


def _build_loss_outputs(
    ratio_ctx: RatioContext,
    total_loss: torch.Tensor,
    policy_loss_tensor: torch.Tensor,
    scalar_inputs: _LossScalarInputs,
) -> LossOutputs:
    """Pack scalar values into a LossOutputs dataclass."""
    loss_scalar = float(total_loss.detach().float().cpu())
    policy_scalar = float(policy_loss_tensor.detach().float().cpu())
    scalar_bundle = LossScalarBundle(
        total_loss=loss_scalar,
        policy_loss=policy_scalar,
        clip_loss=scalar_inputs.clip_loss_scalar,
        kl_loss=scalar_inputs.kl_loss_scalar,
        weighted_kl_loss=scalar_inputs.weighted_kl_loss_scalar,
    )
    return LossOutputs(
        loss=total_loss,
        scalars=scalar_bundle,
        log_ratio_train=ratio_ctx.log_ratio_train,
        denom_tok_tensor=ratio_ctx.denom_tok_tensor,
    )


def _clip_bounds(clip_cfg: ClipSettings) -> Tuple[float, float]:
    """Return natural-log clip bounds for PPO ratios."""
    lower = math.log(max(1.0 - clip_cfg.clip_range, 1e-6))
    upper = math.log(1.0 + clip_cfg.clip_range)
    return lower, upper


def _tensor_mean_std(tensor: torch.Tensor) -> Tuple[float, float]:
    """Return mean and std of a tensor detached to CPU."""
    if tensor.numel() == 0:
        return 0.0, 0.0
    mean = float(tensor.mean().detach().cpu().item())
    if tensor.numel() > 1:
        std = float(tensor.std(unbiased=False).detach().cpu().item())
    else:
        std = 0.0
    return mean, std


def _clip_region_metrics(
    log_ratio: Tensor,
    clip_cfg: ClipSettings,
) -> Tuple[float, float, float, float, float, float]:
    """Return statistics describing the two-sided clipping regions."""
    log_clip_low, log_clip_high = _clip_bounds(clip_cfg)
    low_mask = (log_ratio < log_clip_low).float()
    high_mask = (log_ratio > log_clip_high).float()
    combined_mask = (low_mask + high_mask).clamp(max=1.0)
    region_mean = (
        float(combined_mask.mean().detach().cpu().item())
        if combined_mask.numel() > 0
        else 0.0
    )
    clip_ratio = region_mean
    low_mean = float(low_mask.mean().detach().cpu().item()) if low_mask.numel() > 0 else 0.0
    low_min = float(low_mask.min().detach().cpu().item()) if low_mask.numel() > 0 else 0.0
    high_mean = float(high_mask.mean().detach().cpu().item()) if high_mask.numel() > 0 else 0.0
    high_max = float(high_mask.max().detach().cpu().item()) if high_mask.numel() > 0 else 0.0
    return (
        clip_ratio,
        low_mean,
        low_min,
        high_mean,
        high_max,
        region_mean,
    )


def _ratio_stats_with_ref(
    ratio_ctx: RatioContext,
) -> Tuple[float, float, float, float, float, float, float]:
    """Return diagnostics when reference stats are available."""
    denom_tok_tensor = ratio_ctx.denom_tok_tensor.clamp(min=1.0)
    cur_logp_per_token = ratio_ctx.cur_logp_sum.detach() / denom_tok_tensor
    if ratio_ctx.weighting_cfg.len_norm_ref:
        ref_logp_per_token = ratio_ctx.ref_stats.ref_logp_sum.detach()
    else:
        ref_logp_per_token = ratio_ctx.ref_stats.ref_logp_sum_raw.detach() / denom_tok_tensor
    delta = (ref_logp_per_token - cur_logp_per_token).clamp(min=-60.0, max=60.0)
    ratio = delta.exp()
    kl_value = float((ratio - delta - 1.0).mean().detach().cpu().item())
    log_ratio = (cur_logp_per_token - ref_logp_per_token).clamp(min=-60.0, max=60.0)
    clip_ratio, low_mean, low_min, high_mean, high_max, region_mean = _clip_region_metrics(
        log_ratio, ratio_ctx.clip_cfg
    )
    return (
        kl_value,
        clip_ratio,
        low_mean,
        low_min,
        high_mean,
        high_max,
        region_mean,
    )


def _ratio_stats_without_ref() -> Tuple[float, float, float, float, float, float, float]:
    """Default diagnostics when no reference stats exist."""
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def _ratio_diagnostics(ratio_ctx: RatioContext) -> BatchDiagnostics:
    """Summarize ratio statistics for logging."""
    ref_stats = ratio_ctx.ref_stats
    if ref_stats.ref_tok_counts.numel() > 0:
        stats = _ratio_stats_with_ref(ratio_ctx)
    else:
        stats = _ratio_stats_without_ref()
    kl_value = stats[0] if ratio_ctx.weighting_cfg.beta > 0.0 else None
    return BatchDiagnostics(
        kl_value=kl_value,
        clip_ratio=stats[1],
        clip_ratio_low_mean=stats[2],
        clip_ratio_low_min=stats[3],
        clip_ratio_high_mean=stats[4],
        clip_ratio_high_max=stats[5],
        clip_ratio_region_mean=stats[6],
    )


def evaluate_losses(
    group_data: GroupLossData,
    ratio_ctx: RatioContext,
) -> Tuple[LossOutputs, BatchDiagnostics]:
    """Compute policy loss, clipping objectives, and diagnostics."""
    policy_loss = _policy_loss_from_groups(group_data)
    policy_loss, clip_loss_scalar = _apply_clip_objective(ratio_ctx, group_data, policy_loss)
    kl_loss_tensor, kl_loss_scalar, weighted_kl_loss_scalar = _kl_terms(ratio_ctx)
    total_loss = policy_loss + ratio_ctx.weighting_cfg.beta * kl_loss_tensor
    scalar_inputs = _LossScalarInputs(
        clip_loss_scalar=clip_loss_scalar,
        kl_loss_scalar=kl_loss_scalar,
        weighted_kl_loss_scalar=weighted_kl_loss_scalar,
    )
    loss_outputs = _build_loss_outputs(
        ratio_ctx,
        total_loss,
        policy_loss,
        scalar_inputs,
    )
    diagnostics = _ratio_diagnostics(ratio_ctx)
    return loss_outputs, diagnostics


__all__ = [
    "GroupLossData",
    "RatioContext",
    "SequenceScores",
    "build_loss_inputs",
    "evaluate_losses",
    "LossInputConfig",
]
