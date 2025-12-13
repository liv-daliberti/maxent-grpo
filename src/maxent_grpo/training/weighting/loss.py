# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loss computation helpers for the MaxEnt-GRPO training loop."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from maxent_grpo.training.runtime import require_torch
from ..types import (
    BatchDiagnostics,
    ClipSettings,
    LossOutputs,
    LossScalarBundle,
    ReferenceLogprobs,
)
from .types import WeightStats, WeightingSettings

torch = require_torch("training_loss")
Tensor = torch.Tensor
LOG = logging.getLogger(__name__)
_KL_LENGTH_BUCKETS: List[Tuple[int, Optional[int]]] = [
    (0, 32),
    (33, 64),
    (65, 128),
    (129, 256),
    (257, None),
]


def _bucket_label(lower: int, upper: Optional[int]) -> str:
    """Return a human-readable bucket label."""
    return f"{lower}-{upper}" if upper is not None else f"{lower}+"


def _bucketized_kl_per_token(
    token_counts: Tensor, kl_values: Tensor
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return per-token KL averaged within predefined length buckets.

    Aggregates KL weighted by token counts to avoid short completions dominating
    the average. Returns both mean KL per token and the total token counts for
    each bucket so downstream logging can show support.
    """
    bucket_kl_sum: Dict[str, float] = {}
    bucket_token_sum: Dict[str, float] = {}
    token_cpu = token_counts.detach().float().cpu()
    kl_cpu = kl_values.detach().float().cpu()
    for tok, kl in zip(token_cpu, kl_cpu):
        tok_val = float(max(tok.item(), 1.0))
        kl_val = float(kl.item())
        for lower, upper in _KL_LENGTH_BUCKETS:
            if tok_val < lower:
                continue
            if upper is not None and tok_val > upper:
                continue
            label = _bucket_label(lower, upper)
            bucket_kl_sum[label] = bucket_kl_sum.get(label, 0.0) + kl_val * tok_val
            bucket_token_sum[label] = bucket_token_sum.get(label, 0.0) + tok_val
            break
        else:
            label = _bucket_label(_KL_LENGTH_BUCKETS[-1][0], _KL_LENGTH_BUCKETS[-1][1])
            bucket_kl_sum[label] = bucket_kl_sum.get(label, 0.0) + kl_val * tok_val
            bucket_token_sum[label] = bucket_token_sum.get(label, 0.0) + tok_val
    bucket_means: Dict[str, float] = {}
    for label, tok_sum in bucket_token_sum.items():
        bucket_means[label] = bucket_kl_sum.get(label, 0.0) / max(tok_sum, 1.0)
    return bucket_means, bucket_token_sum


@dataclass
class SequenceScores:
    """Bundle sequence-level log-prob statistics."""

    cur_logp_sum: Tensor
    behavior_logp_sum: Tensor
    log_ratio_train: Tensor
    denom_tok_tensor: Tensor
    pooled_hidden: Optional[Tensor] = None


@dataclass
class SeedInfoInputs:
    """Optional seed-level metadata and pooled representations."""

    seed_ids: Tensor
    pooled_hidden: Tensor
    is_seed_aug: Optional[Tensor] = None
    logits: Optional[Tensor] = None


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
    """Build tensors/contexts required for downstream loss computation.

    :param grouped_completions: Completions grouped by prompt.
    :type grouped_completions: list[list[str]]
    :param weight_stats: Weighting statistics (flat weights + entropy).
    :type weight_stats: WeightStats
    :param scores: Sequence log-prob bundles emitted by the scorer.
    :type scores: SequenceScores
    :param config: Clipping/weighting configuration shared across batches.
    :type config: LossInputConfig
    :returns: Tuple containing the per-group tensors and the ratio context.
    :rtype: tuple[GroupLossData, RatioContext]
    :raises ValueError: When weights/log-prob tensors disagree in size.
    """
    group_sizes = [len(group) for group in grouped_completions]
    total_completions = sum(group_sizes)
    logp_count = (
        scores.cur_logp_sum.numel()
        if hasattr(scores.cur_logp_sum, "numel")
        else len(getattr(scores.cur_logp_sum, "data", []))
    )
    if logp_count and total_completions > logp_count:
        trimmed_sizes = _trim_group_sizes(group_sizes, logp_count)
        if not trimmed_sizes:
            raise ValueError("No completions available for loss computation.")
        if len(trimmed_sizes) != len(group_sizes):
            LOG.warning(
                "Trimming completion groups to match scored sequences | requested=%d | scored=%d",
                total_completions,
                logp_count,
            )
        group_sizes = trimmed_sizes
        total_completions = sum(group_sizes)
    try:
        weight_tensor = torch.tensor(
            weight_stats.flat_weights,
            device=getattr(scores.cur_logp_sum, "device", None),
            dtype=getattr(scores.cur_logp_sum, "dtype", None),
        )
    except (TypeError, AttributeError):
        weight_tensor = torch.tensor(weight_stats.flat_weights)
    if hasattr(weight_tensor, "view"):
        weight_tensor = weight_tensor.view(-1)
    # Align target length with available log-prob entries when they disagree with completions.
    target_count = total_completions if total_completions > 0 else logp_count
    if logp_count and logp_count != target_count:
        target_count = logp_count
    weight_count = (
        weight_tensor.numel()
        if hasattr(weight_tensor, "numel")
        else len(getattr(weight_tensor, "data", []))
    )
    allow_broadcast = not getattr(weight_stats, "weights_grouped", None)
    if (
        weight_count < target_count or target_count != total_completions
    ) and not allow_broadcast:
        raise ValueError("Mismatch between weights and log-prob dimensions.")
    if weight_count == 1 and target_count > 1:
        fill_val = (
            float(weight_tensor[0]) if hasattr(weight_tensor, "__getitem__") else 1.0
        )
        try:
            weight_tensor = torch.full(
                (target_count,),
                fill_val,
                device=scores.cur_logp_sum.device,
                dtype=scores.cur_logp_sum.dtype,
            )
        except TypeError:
            weight_tensor = torch.full((target_count,), fill_val)
    elif weight_count < target_count:
        pad_val = float(weight_tensor[0]) if weight_count > 0 else 1.0
        try:
            pad = torch.full(
                (target_count - weight_count,),
                pad_val,
                device=scores.cur_logp_sum.device,
                dtype=scores.cur_logp_sum.dtype,
            )
        except TypeError:
            pad = torch.full((target_count - weight_count,), pad_val)
        weight_tensor = torch.cat([weight_tensor.view(-1), pad])
    elif weight_count > target_count:
        weight_tensor = weight_tensor.view(-1)[:target_count]
    group_data = GroupLossData(
        group_sizes=group_sizes,
        weight_tensor=weight_tensor,
        logp_sums=scores.cur_logp_sum,
        token_counts=(
            scores.denom_tok_tensor.to(getattr(scores.cur_logp_sum, "device", None))
            if hasattr(scores.denom_tok_tensor, "to")
            else scores.denom_tok_tensor
        ),
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
    """Return the mean policy loss aggregated over prompt groups.

    :param group_data: Flattened tensors/resolution metadata per group.
    :type group_data: GroupLossData
    :returns: Scalar tensor representing the average policy loss.
    :rtype: torch.Tensor
    :raises ValueError: If no completions were available for aggregation.
    """
    total_groups = len(group_data.group_sizes)
    total_weights = _tensor_numel(group_data.weight_tensor)
    total_logp = _tensor_numel(group_data.logp_sums)
    total_tokens = _tensor_numel(group_data.token_counts)
    LOG.debug(
        "Policy loss group summary | groups=%d | weights=%d | logp=%d | tokens=%d",
        total_groups,
        total_weights,
        total_logp,
        total_tokens,
    )
    def _stats_for_logging(tensor_like: torch.Tensor) -> Tuple[int, int, str]:
        """Return ``(numel, nonzero, preview)`` for debug logging."""
        try:
            tensor = torch.as_tensor(getattr(tensor_like, "arr", tensor_like))
        except (TypeError, ValueError):
            tensor = torch.tensor(
                getattr(getattr(tensor_like, "arr", tensor_like), "data", tensor_like)
            )
        tensor = tensor.view(-1)
        numel = tensor.numel()
        if numel == 0:
            return 0, 0, _tensor_preview(tensor_like)
        nonzero = int((tensor != 0).sum().item())
        return numel, nonzero, _tensor_preview(tensor_like)
    if total_weights > 0 and total_logp == 0:
        weight_stats = _stats_for_logging(group_data.weight_tensor)
        token_stats = _stats_for_logging(group_data.token_counts)
        zero_weight_support = weight_stats[1] == 0
        log_fn = LOG.info if zero_weight_support else LOG.warning
        log_fn(
            "Policy loss logp tensor empty; forcing zeros | groups=%d | weights=%d (nonzero=%d) "
            "| tokens=%d (nonzero=%d) | logp_preview=%s | weight_preview=%s | token_preview=%s",
            total_groups,
            total_weights,
            weight_stats[1],
            total_tokens,
            token_stats[1],
            _tensor_preview(group_data.logp_sums),
            weight_stats[2],
            token_stats[2],
        )
        logp_device = getattr(
            group_data.logp_sums, "device", getattr(group_data.weight_tensor, "device", None)
        )
        logp_dtype = getattr(
            group_data.logp_sums, "dtype", getattr(group_data.weight_tensor, "dtype", torch.float32)
        )
        group_data.logp_sums = torch.zeros(
            (total_weights,), device=logp_device, dtype=logp_dtype
        )
    if total_weights > 0 and total_tokens == 0:
        weight_stats = _stats_for_logging(group_data.weight_tensor)
        LOG.warning(
            "Policy loss token counts empty; forcing ones | groups=%d | weights=%d (nonzero=%d) "
            "| token_preview=%s",
            total_groups,
            total_weights,
            weight_stats[1],
            _tensor_preview(group_data.token_counts),
        )
        tok_device = getattr(
            group_data.token_counts, "device", getattr(group_data.weight_tensor, "device", None)
        )
        tok_dtype = getattr(group_data.token_counts, "dtype", torch.float32)
        group_data.token_counts = torch.ones(
            (total_weights,), device=tok_device, dtype=tok_dtype
        )
    policy_group_losses: List[torch.Tensor] = []
    offset = 0
    for size in group_data.group_sizes:
        if size <= 0:
            continue
        end = offset + size
        w_slice = group_data.weight_tensor[offset:end]
        logp_slice = group_data.logp_sums[offset:end]
        token_slice = group_data.token_counts[offset:end].clamp(min=1.0)
        weight_len = w_slice.numel()
        logp_len = logp_slice.numel()
        if weight_len != logp_len:
            mismatch = min(weight_len, logp_len)
            LOG.debug(
                "Policy loss group length mismatch | offset=%d | size=%d | weights=%d | logp=%d | tokens=%d",
                offset,
                size,
                int(weight_len),
                int(logp_len),
                int(token_slice.numel()),
            )
            if mismatch == 0:
                if weight_len == 0:
                    offset = end
                    continue
                LOG.warning(
                    "Filling empty policy log-probs with zeros | offset=%d | size=%d | weights=%d | logp_preview=%s",
                    offset,
                    size,
                    int(weight_len),
                    _tensor_preview(logp_slice),
                )
                logp_slice = torch.zeros(
                    (weight_len,),
                    device=getattr(group_data.logp_sums, "device", getattr(w_slice, "device", None)),
                    dtype=getattr(group_data.logp_sums, "dtype", getattr(w_slice, "dtype", None)),
                )
                token_slice = torch.ones(
                    (weight_len,),
                    device=getattr(group_data.token_counts, "device", getattr(logp_slice, "device", None)),
                    dtype=getattr(group_data.token_counts, "dtype", torch.float32),
                )
            else:
                w_slice = w_slice[:mismatch]
                logp_slice = logp_slice[:mismatch]
                token_slice = token_slice[:mismatch]
        normalized_logp = logp_slice / token_slice
        try:
            term = (-normalized_logp * w_slice).sum()
        except (RuntimeError, TypeError, ValueError):
            norm = torch.tensor(getattr(normalized_logp, "arr", normalized_logp))
            weights = torch.tensor(getattr(w_slice, "arr", w_slice))
            term = (-norm * weights).sum()
        # Normalize stub tensors into real torch tensors so downstream ops
        # (e.g., torch.stack) always receive compatible inputs.
        if isinstance(term, torch.Tensor):
            term_tensor = term
        else:
            term_tensor = torch.tensor(getattr(term, "arr", term))
        policy_group_losses.append(term_tensor)
        offset = end
    if not policy_group_losses:
        _, nonzero_weights, weight_prev = _stats_for_logging(group_data.weight_tensor)
        _, nonzero_logp, logp_prev = _stats_for_logging(group_data.logp_sums)
        zero_signal = nonzero_weights == 0 and nonzero_logp == 0
        log_fn = LOG.info if zero_signal else LOG.warning
        log_fn(
            "Policy loss groups empty; returning zero | groups=%d | weights=%d (nonzero=%d) "
            "| logp=%d (nonzero=%d) | tokens=%d | weight_preview=%s | logp_preview=%s",
            total_groups,
            total_weights,
            nonzero_weights,
            total_logp,
            nonzero_logp,
            total_tokens,
            weight_prev,
            logp_prev,
        )
        if isinstance(group_data.weight_tensor, torch.Tensor):
            return group_data.weight_tensor.new_zeros(())
        return torch.tensor(0.0)
    return torch.stack(policy_group_losses).mean()


def _trim_group_sizes(group_sizes: List[int], max_sequences: int) -> List[int]:
    """Clamp group sizes so their cumulative total does not exceed ``max_sequences``."""

    if max_sequences <= 0:
        return []
    trimmed: List[int] = []
    remaining = max_sequences
    for size in group_sizes:
        if remaining <= 0:
            break
        take = min(size, remaining)
        if take > 0:
            trimmed.append(take)
        remaining -= take
    return trimmed


def _iter_group_offsets(group_sizes: List[int]) -> Iterator[Tuple[int, int]]:
    """Yield ``(offset, size)`` windows for each completion group.

    :param group_sizes: Number of completions per prompt group.
    :type group_sizes: list[int]
    :yields: Tuples describing the slice boundaries within flattened tensors.
    :rtype: Iterator[tuple[int, int]]
    """
    offset = 0
    for size in group_sizes:
        yield offset, size
        offset += size


def _coerce_tensor_like(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Return ``value`` coerced to the backend/dtype of ``reference``."""
    tensor_ctor = getattr(torch, "as_tensor", getattr(torch, "tensor", None))
    if tensor_ctor is None:
        return value
    payload = getattr(value, "arr", value)
    try:
        coerced = tensor_ctor(payload)
    except Exception:  # pragma: no cover - best-effort fallback
        return value
    ref_device = getattr(reference, "device", None)
    if ref_device is not None and hasattr(coerced, "to"):
        try:
            coerced = coerced.to(ref_device)
        except Exception:
            pass
    ref_dtype = getattr(reference, "dtype", None)
    if ref_dtype is not None and hasattr(coerced, "to"):
        try:
            coerced = coerced.to(dtype=ref_dtype)
        except Exception:
            pass
    return coerced


def _tensor_numel(value: torch.Tensor) -> int:
    """Best-effort ``numel`` lookup that tolerates lightweight tensor stubs."""

    numel_fn = getattr(value, "numel", None)
    if callable(numel_fn):
        try:
            return int(numel_fn())
        except Exception:
            pass
    for attr in ("data", "arr"):
        payload = getattr(value, attr, None)
        if payload is not None:
            try:
                return len(payload)
            except Exception:
                continue
    try:
        return len(value)
    except Exception:
        return 0


def _tensor_preview(value: torch.Tensor, limit: int = 4) -> str:
    """Return a short preview string for logging diagnostics."""

    try:
        tensor = value
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(getattr(value, "arr", value))
        flat = tensor.flatten()
        if flat.numel() == 0:
            return "[]"
        subset = flat[:limit].tolist()
        more = flat.numel() - len(subset)
        suffix = "..." if more > 0 else ""
        return f"{subset}{suffix}"
    except Exception:
        return f"<unprintable:{type(value).__name__}>"


def _clip_loss_for_slice(
    weight_tensor: Tensor,
    ratio_slice: Tensor,
    clipped_slice: Tensor,
    adv_base_val: float,
) -> torch.Tensor:
    """Return the PPO-style clip loss for a contiguous slice.

    :param weight_tensor: Flattened weights aligned with completions.
    :type weight_tensor: torch.Tensor
    :param ratio_slice: Raw PPO ratios for the slice.
    :type ratio_slice: torch.Tensor
    :param clipped_slice: Clipped ratios respecting ``clip_range``.
    :type clipped_slice: torch.Tensor
    :param adv_base_val: Baseline advantage used within the slice.
    :type adv_base_val: float
    :returns: Negative clipped objective contribution for the slice.
    :rtype: torch.Tensor
    """
    adv_tensor = weight_tensor - adv_base_val
    obj_unclipped = ratio_slice * adv_tensor
    obj_clipped = clipped_slice * adv_tensor
    # In some environments tensors from different backends (real ``Tensor``
    # vs stub ``_Tensor``) can mix.  Coerce intermediates through the active
    # torch module so downstream arithmetic sees a single tensor type.
    _to_tensor = getattr(torch, "as_tensor", getattr(torch, "tensor", None))
    if _to_tensor is not None:  # pragma: no cover - exercised in integration tests
        try:
            adv_tensor = _to_tensor(adv_tensor)
            obj_unclipped = _to_tensor(obj_unclipped)
            obj_clipped = _to_tensor(obj_clipped)
        except Exception:
            pass
    # Prefer native torch helpers when available, but fall back to basic
    # tensor arithmetic for lightweight stubs that omit them or when mixed
    # tensor types (real ``Tensor`` vs stub ``_Tensor``) confuse the helpers.
    try:  # pragma: no cover - exercised indirectly in integration tests
        min_vals = torch.minimum(obj_unclipped, obj_clipped)
        max_vals = torch.maximum(obj_unclipped, obj_clipped)
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - stub/mixed fallback
        less_mask = obj_unclipped <= obj_clipped
        greater_mask = obj_unclipped > obj_clipped
        min_vals = obj_unclipped * less_mask + obj_clipped * greater_mask
        max_vals = obj_unclipped * greater_mask + obj_clipped * less_mask
    try:  # pragma: no cover - exercised indirectly in integration tests
        obj = torch.where(adv_tensor >= 0, min_vals, max_vals)
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - stub/mixed fallback
        pos_mask = adv_tensor >= 0
        neg_mask = adv_tensor < 0
        obj = min_vals * pos_mask + max_vals * neg_mask
    return -(obj.sum())


def _apply_clip_objective(
    ratio_ctx: RatioContext,
    group_data: GroupLossData,
    policy_loss: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[float]]:
    """Apply the optional clip objective and return updated loss/scalar.

    :param ratio_ctx: Ratio context with log-probs and clip config.
    :type ratio_ctx: RatioContext
    :param group_data: Group metadata/tensors.
    :type group_data: GroupLossData
    :param policy_loss: Current policy loss tensor.
    :type policy_loss: torch.Tensor
    :returns: Tuple of updated policy loss and optional clip scalar.
    :rtype: tuple[torch.Tensor, float | None]
    """
    clip_cfg = ratio_ctx.clip_cfg
    if not (clip_cfg.use_clip_objective and clip_cfg.clip_range > 0.0):
        return policy_loss, None
    ratio_for_loss, clipped_ratio_vals = _compute_clip_ratios(ratio_ctx, clip_cfg)
    clip_losses: List[torch.Tensor] = []
    for offset, size in _iter_group_offsets(group_data.group_sizes):
        if size <= 0:
            continue
        end = offset + size
        weight_slice = group_data.weight_tensor[offset:end]
        ratio_slice = ratio_for_loss[offset:end]
        clipped_slice = clipped_ratio_vals[offset:end]
        weight_len = _tensor_numel(weight_slice)
        ratio_len = _tensor_numel(ratio_slice)
        clipped_len = _tensor_numel(clipped_slice)
        min_len = min(weight_len, ratio_len, clipped_len)
        if min_len <= 0:
            LOG.warning(
                "Skipping clip slice due to empty tensors | offset=%s size=%s weight=%s ratio=%s clipped=%s",
                offset,
                size,
                weight_len,
                ratio_len,
                clipped_len,
            )
            continue
        if (
            min_len < weight_len
            or min_len < ratio_len
            or min_len < clipped_len
        ):
            LOG.warning(
                "Truncating mismatched clip tensors | offset=%s size=%s weight=%s ratio=%s clipped=%s min=%s",
                offset,
                size,
                weight_len,
                ratio_len,
                clipped_len,
                min_len,
            )
            weight_slice = weight_slice[:min_len]
            ratio_slice = ratio_slice[:min_len]
            clipped_slice = clipped_slice[:min_len]
        effective_size = min_len
        adv_base_val = (
            clip_cfg.clip_adv_baseline
            if clip_cfg.clip_adv_baseline is not None
            else (1.0 / float(max(effective_size, 1)))
        )
        clip_losses.append(
            _clip_loss_for_slice(
                weight_slice,
                ratio_slice,
                clipped_slice,
                adv_base_val,
            )
        )
    if not clip_losses:
        return policy_loss, None
    # Avoid relying on ``torch.stack`` so lightweight torch stubs remain valid.
    clip_loss_tensor = _coerce_tensor_like(policy_loss, clip_losses[0])
    for extra in clip_losses[1:]:
        extra = _coerce_tensor_like(policy_loss, extra)
        clip_loss_tensor = clip_loss_tensor + extra
    clip_loss_tensor = clip_loss_tensor / float(len(clip_losses))
    clip_loss_tensor = _coerce_tensor_like(policy_loss, clip_loss_tensor)
    scaled_clip = clip_cfg.clip_objective_coef * clip_loss_tensor
    try:
        updated_loss = policy_loss + scaled_clip
    except TypeError:  # pragma: no cover - mixed backend fallback
        policy_loss = _coerce_tensor_like(scaled_clip, policy_loss)
        updated_loss = policy_loss + scaled_clip
    clip_loss_scalar = float(clip_loss_tensor.detach().float().cpu())
    return updated_loss, clip_loss_scalar


def _kl_terms(ratio_ctx: RatioContext) -> Tuple[torch.Tensor, float, float]:
    """Return KL tensor, scalar, and weighted scalar contributions.

    :param ratio_ctx: Ratio context describing current/reference log-probs.
    :type ratio_ctx: RatioContext
    :returns: Tuple containing per-sequence KL tensor, scalar KL, and beta-
        weighted KL scalar.
    :rtype: tuple[torch.Tensor, float, float]

    We mirror TRL's GRPOTrainer by using the always-non-negative scalar
    ``exp(ref_logp - cur_logp) - (ref_logp - cur_logp) - 1`` computed on
    *length-normalized* per-token log-probabilities, then aggregate with a
    token-weighted mean to avoid overweighting short completions.
    """
    def _zero_kl_return(
        log_reason: Optional[str], level: int = logging.WARNING
    ) -> Tuple[torch.Tensor, float, float]:
        zero_tensor: torch.Tensor
        if isinstance(ratio_ctx.cur_logp_sum, torch.Tensor):
            zero_tensor = ratio_ctx.cur_logp_sum.new_zeros(())
        else:
            zero_tensor = torch.tensor(0.0)
        beta_val = getattr(ratio_ctx.weighting_cfg, "beta", 0.0)
        beta_scalar = float(getattr(beta_val, "arr", beta_val))
        if log_reason:
            LOG.log(level, "%s", log_reason)
        return zero_tensor, 0.0, beta_scalar * 0.0

    denom = ratio_ctx.denom_tok_tensor.clamp(min=1.0)
    cur_logp_per_tok = ratio_ctx.cur_logp_sum / denom
    if ratio_ctx.weighting_cfg.len_norm_ref:
        ref_logp_per_tok = ratio_ctx.ref_stats.ref_logp_sum
    else:
        ref_logp_per_tok = ratio_ctx.ref_stats.ref_logp_sum_raw / denom
    # Coerce reference arrays into tensors when stubs are used in tests.
    if not hasattr(ref_logp_per_tok, "clamp"):
        ref_logp_per_tok = torch.tensor(
            getattr(ref_logp_per_tok, "arr", ref_logp_per_tok)
        )
    if not hasattr(cur_logp_per_tok, "clamp"):
        cur_logp_per_tok = torch.tensor(
            getattr(cur_logp_per_tok, "arr", cur_logp_per_tok)
        )
    try:
        cur_len = cur_logp_per_tok.numel()
    except (AttributeError, TypeError):
        cur_len = len(getattr(cur_logp_per_tok, "data", []))
    try:
        ref_len = ref_logp_per_tok.numel()
    except (AttributeError, TypeError):
        ref_len = len(getattr(ref_logp_per_tok, "data", []))
    if not cur_len or not ref_len:
        return _zero_kl_return(
            "Skipping KL computation due to empty logp tensors | cur=%d | ref=%d"
            % (int(cur_len), int(ref_len)),
            logging.WARNING,
        )
    if ref_len != cur_len:
        if ref_len == 1:
            fill_val = (
                float(ref_logp_per_tok[0])
                if hasattr(ref_logp_per_tok, "__getitem__")
                else 0.0
            )
            try:
                ref_logp_per_tok = torch.full(
                    (cur_len,),
                    fill_val,
                    device=getattr(cur_logp_per_tok, "device", None),
                    dtype=getattr(cur_logp_per_tok, "dtype", None),
                )
            except TypeError:
                ref_logp_per_tok = torch.full((cur_len,), fill_val)
        elif ref_len < cur_len:
            fill_val = (
                float(ref_logp_per_tok[-1])
                if hasattr(ref_logp_per_tok, "__getitem__")
                else 0.0
            )
            try:
                pad = torch.full(
                    (cur_len - ref_len,),
                    fill_val,
                    device=getattr(cur_logp_per_tok, "device", None),
                    dtype=getattr(cur_logp_per_tok, "dtype", None),
                )
                try:
                    ref_logp_per_tok = torch.cat([ref_logp_per_tok, pad])
                except (RuntimeError, TypeError, ValueError):
                    ref_logp_per_tok = torch.tensor(
                        list(ref_logp_per_tok) + list(getattr(pad, "data", []))
                    )
            except (RuntimeError, TypeError, ValueError):
                ref_logp_per_tok = torch.tensor(
                    list(ref_logp_per_tok) + [fill_val] * (cur_len - ref_len)
                )
            else:
                try:
                    ref_logp_per_tok = ref_logp_per_tok[:cur_len]
                except (RuntimeError, TypeError, ValueError):
                    ref_logp_per_tok = torch.tensor(list(ref_logp_per_tok)[:cur_len])
    # Ensure tensors for downstream math.
    if not isinstance(ref_logp_per_tok, torch.Tensor):
        ref_logp_per_tok = torch.tensor(
            getattr(ref_logp_per_tok, "arr", ref_logp_per_tok)
        )
    if not isinstance(cur_logp_per_tok, torch.Tensor):
        cur_logp_per_tok = torch.tensor(
            getattr(cur_logp_per_tok, "arr", cur_logp_per_tok)
        )
    # Align dtype/device so downstream math stays on the accelerator when available.
    ref_device = getattr(ref_logp_per_tok, "device", None)
    cur_device = getattr(cur_logp_per_tok, "device", None)
    ref_dtype = getattr(ref_logp_per_tok, "dtype", None)
    cur_dtype = getattr(cur_logp_per_tok, "dtype", None)
    if (ref_device is not None and cur_device is not None and ref_device != cur_device) or (
        ref_dtype is not None and cur_dtype is not None and ref_dtype != cur_dtype
    ):
        try:
            ref_logp_per_tok = ref_logp_per_tok.to(cur_logp_per_tok)
        except (AttributeError, RuntimeError, TypeError):
            # Fall back to moving both tensors to CPU so we never mix devices.
            ref_logp_per_tok = ref_logp_per_tok.to("cpu")
            cur_logp_per_tok = cur_logp_per_tok.to("cpu")
    # Re-check lengths after conversions/padding; skip if still empty.
    cur_len = cur_logp_per_tok.numel()
    ref_len = ref_logp_per_tok.numel()
    if cur_len == 0 or ref_len == 0:
        return _zero_kl_return(
            "Skipping KL computation due to post-align empty tensors | cur=%d | ref=%d"
            % (int(cur_len), int(ref_len)),
            logging.WARNING,
        )
    if ref_len != cur_len:
        min_len = min(cur_len, ref_len)
        if min_len == 0:
            return _zero_kl_return(
                "Skipping KL computation due to irreconcilable logp lengths | cur=%d | ref=%d"
                % (int(cur_len), int(ref_len)),
                logging.WARNING,
            )
        LOG.warning(
            "Trimming KL tensors to minimum length | cur=%d | ref=%d | min=%d",
            int(cur_len),
            int(ref_len),
            int(min_len),
        )
        ref_logp_per_tok = ref_logp_per_tok[:min_len]
        cur_logp_per_tok = cur_logp_per_tok[:min_len]

    delta = (ref_logp_per_tok - cur_logp_per_tok).clamp(min=-60.0, max=60.0)
    if not hasattr(delta, "exp"):
        delta = torch.tensor(getattr(delta, "arr", getattr(delta, "data", delta)))
    per_seq_kl = delta.exp() - delta - 1.0
    # Weight by reference token counts so short completions do not dominate.
    ref_tok_counts = ratio_ctx.ref_stats.ref_tok_counts
    # Convert non-tensors (including lightweight stubs) to real torch tensors.
    if not hasattr(ref_tok_counts, "detach") or isinstance(
        ref_tok_counts, (list, tuple)
    ):
        ref_tok_counts = torch.tensor(getattr(ref_tok_counts, "arr", ref_tok_counts))
    ref_tok_weights = ref_tok_counts.detach().clamp(min=1.0)
    try:
        ref_tok_weights = ref_tok_weights.to(per_seq_kl)
    except (AttributeError, TypeError, RuntimeError):
        ref_tok_weights = torch.tensor(
            getattr(ref_tok_weights, "arr", ref_tok_weights)
        ).to(per_seq_kl)
    # Broadcast lengths when stubs provide scalar or mismatched counts.
    per_len = getattr(
        per_seq_kl, "numel", lambda: len(getattr(per_seq_kl, "data", []))
    )()
    ref_len = getattr(
        ref_tok_weights, "numel", lambda: len(getattr(ref_tok_weights, "data", []))
    )()
    if per_len == 0 or ref_len == 0:
        LOG.warning(
            "Skipping KL computation due to empty tensors | per_seq=%d | ref_tokens=%d",
            int(per_len),
            int(ref_len),
        )
        zero_tensor: torch.Tensor
        if isinstance(per_seq_kl, torch.Tensor):
            zero_tensor = per_seq_kl.new_zeros(())
        elif isinstance(ratio_ctx.cur_logp_sum, torch.Tensor):
            zero_tensor = ratio_ctx.cur_logp_sum.new_zeros(())
        else:
            zero_tensor = torch.tensor(0.0)
        beta_val = getattr(ratio_ctx.weighting_cfg, "beta", 0.0)
        try:
            beta_val = float(getattr(beta_val, "arr", beta_val))
        except (TypeError, ValueError):
            beta_val = 0.0
        kl_loss_scalar = 0.0
        weighted_kl_loss_scalar = beta_val * kl_loss_scalar
        return zero_tensor, kl_loss_scalar, weighted_kl_loss_scalar
    if ref_len == 1 and per_len > 1:
        fill_val = float(ref_tok_weights[0]) if hasattr(ref_tok_weights, "__getitem__") else 1.0
        try:
            ref_tok_weights = torch.full_like(per_seq_kl, fill_val)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            ref_tok_weights = torch.tensor([fill_val] * per_len)
    elif ref_len != per_len:
        # Align lengths by flattening or recreating the tensor when stubs
        # provide incompatible shapes.
        try:
            ref_tok_weights = ref_tok_weights.view(-1)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            try:
                ref_tok_weights = torch.tensor(
                    getattr(
                        ref_tok_weights,
                        "arr",
                        getattr(ref_tok_weights, "data", ref_tok_weights),
                    )
                )
            except (RuntimeError, TypeError, ValueError, AttributeError):
                ref_tok_weights = torch.tensor(list(ref_tok_weights))
            try:
                ref_tok_weights = ref_tok_weights.view(-1)
            except (RuntimeError, TypeError, ValueError, AttributeError):
                fill_val = (
                    float(ref_tok_weights[-1])
                    if hasattr(ref_tok_weights, "__getitem__")
                    else 1.0
                )
            try:
                ref_tok_weights = torch.full_like(per_seq_kl, fill_val)
            except (RuntimeError, TypeError, ValueError, AttributeError):
                try:
                    ref_tok_weights = torch.full(
                        getattr(per_seq_kl, "shape", (per_len,)),
                        fill_val,
                        device=getattr(per_seq_kl, "device", None),
                        dtype=getattr(per_seq_kl, "dtype", None),
                    )
                except (RuntimeError, TypeError, ValueError, AttributeError):
                    ref_tok_weights = torch.tensor([fill_val] * per_len)
    if ref_tok_weights.numel() < per_len:
        fill_val = float(ref_tok_weights[-1]) if hasattr(ref_tok_weights, "__getitem__") else 1.0
        try:
            pad = torch.full_like(per_seq_kl, fill_val)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pad = torch.tensor([fill_val] * per_len)
        ref_tok_weights = pad
    denom_weight = ref_tok_weights.sum().clamp(min=1.0)
    kl_loss_tensor = (per_seq_kl * ref_tok_weights).sum() / denom_weight
    kl_loss_scalar = float(kl_loss_tensor.detach().float().cpu())
    weighted_kl_loss_scalar = ratio_ctx.weighting_cfg.beta * kl_loss_scalar
    return kl_loss_tensor, kl_loss_scalar, weighted_kl_loss_scalar


def _compute_clip_ratios(
    ratio_ctx: RatioContext,
    clip_cfg: ClipSettings,
) -> Tuple[Tensor, Tensor]:
    """Return raw and clipped PPO ratios.

    :param ratio_ctx: Ratio context containing log-prob sums.
    :type ratio_ctx: RatioContext
    :param clip_cfg: Clip configuration specifying range.
    :type clip_cfg: ClipSettings
    :returns: Tuple of raw ratios and clipped ratios.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    behavior_log_ratio = ratio_ctx.cur_logp_sum - ratio_ctx.behavior_logp_sum
    ratio_for_loss = behavior_log_ratio.clamp(min=-60.0, max=60.0).exp()
    clipped_ratio_vals = ratio_for_loss.clamp(
        min=1.0 - clip_cfg.clip_range,
        max=1.0 + clip_cfg.clip_range,
    )
    return ratio_for_loss, clipped_ratio_vals


def _build_loss_outputs(
    ratio_ctx: RatioContext,
    total_loss: torch.Tensor,
    policy_loss_tensor: torch.Tensor,
    scalar_inputs: _LossScalarInputs,
    *,
    seed_loss: Optional[torch.Tensor] = None,
    info_entropy_term: Optional[torch.Tensor] = None,
) -> LossOutputs:
    """Pack scalar values into a ``LossOutputs`` dataclass.

    :param ratio_ctx: Ratio context used to propagate tensors upstream.
    :type ratio_ctx: RatioContext
    :param total_loss: Final loss tensor (policy + KL + clip).
    :type total_loss: torch.Tensor
    :param policy_loss_tensor: Policy-only component.
    :type policy_loss_tensor: torch.Tensor
    :param scalar_inputs: Scalar clip/KL contributions.
    :type scalar_inputs: _LossScalarInputs
    :param seed_loss: Optional auxiliary seed loss tensor.
    :type seed_loss: torch.Tensor | None
    :returns: Structured outputs consumed by the optimizer/metrics layer.
    :rtype: LossOutputs
    """
    loss_scalar = float(total_loss.detach().float().cpu())
    policy_scalar = float(policy_loss_tensor.detach().float().cpu())
    seed_scalar = (
        float(seed_loss.detach().float().cpu()) if seed_loss is not None else None
    )
    info_entropy_scalar = (
        float(info_entropy_term.detach().float().cpu())
        if info_entropy_term is not None
        else None
    )
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
        seed_loss=seed_loss,
        seed_loss_scalar=seed_scalar,
        info_seed_entropy_term=info_entropy_term,
        info_seed_entropy_scalar=info_entropy_scalar,
    )


def _clip_bounds(clip_cfg: ClipSettings) -> Tuple[float, float]:
    """Return natural-log clip bounds for PPO ratios.

    :param clip_cfg: Clip configuration specifying ``clip_range``.
    :type clip_cfg: ClipSettings
    :returns: Tuple containing the log-space bounds.
    :rtype: tuple[float, float]
    """
    lower = math.log(max(1.0 - clip_cfg.clip_range, 1e-6))
    upper = math.log(1.0 + clip_cfg.clip_range)
    return lower, upper


def _tensor_mean_std(tensor: torch.Tensor) -> Tuple[float, float]:
    """Return mean and std of a tensor detached to CPU.

    :param tensor: Input tensor for which to compute statistics.
    :type tensor: torch.Tensor
    :returns: Tuple of ``(mean, std)`` scalars.
    :rtype: tuple[float, float]
    """
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
    """Return statistics describing the two-sided clipping regions.

    :param log_ratio: Log ratios used to determine clipping.
    :type log_ratio: torch.Tensor
    :param clip_cfg: Clip configuration (range + toggles).
    :type clip_cfg: ClipSettings
    :returns: Tuple summarizing clip frequency and per-side stats.
    :rtype: tuple[float, float, float, float, float, float]
    """
    if not hasattr(log_ratio, "__lt__"):
        log_ratio = torch.tensor(getattr(log_ratio, "arr", log_ratio))
    log_clip_low, log_clip_high = _clip_bounds(clip_cfg)
    try:
        low_mask = (log_ratio < log_clip_low).float()
        high_mask = (log_ratio > log_clip_high).float()
    except TypeError:
        log_ratio = torch.tensor(
            getattr(log_ratio, "arr", getattr(log_ratio, "data", log_ratio))
        )
        low_mask = (log_ratio < log_clip_low).float()
        high_mask = (log_ratio > log_clip_high).float()
    combined_mask = (low_mask + high_mask).clamp(max=1.0)
    region_mean = (
        float(combined_mask.mean().detach().cpu().item())
        if combined_mask.numel() > 0
        else 0.0
    )
    clip_ratio = region_mean
    low_mean = (
        float(low_mask.mean().detach().cpu().item()) if low_mask.numel() > 0 else 0.0
    )
    low_min = (
        float(low_mask.min().detach().cpu().item()) if low_mask.numel() > 0 else 0.0
    )
    high_mean = (
        float(high_mask.mean().detach().cpu().item()) if high_mask.numel() > 0 else 0.0
    )
    high_max = (
        float(high_mask.max().detach().cpu().item()) if high_mask.numel() > 0 else 0.0
    )
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
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    Dict[str, float],
    Dict[str, float],
]:
    """Return diagnostics when reference stats are available.

    :param ratio_ctx: Ratio context containing ref/current log-probs.
    :type ratio_ctx: RatioContext
    :returns: Tuple of KL estimate and clipping statistics.
    :rtype: tuple[float, float, float, float, float, float, float]
    """
    denom_tok_tensor = ratio_ctx.denom_tok_tensor.clamp(min=1.0)
    cur_logp_sum = ratio_ctx.cur_logp_sum
    ref_logp_sum = ratio_ctx.ref_stats.ref_logp_sum
    ref_logp_sum_raw = ratio_ctx.ref_stats.ref_logp_sum_raw
    # Coerce fake arrays into tensors when needed.
    if not hasattr(cur_logp_sum, "detach"):
        cur_logp_sum = torch.tensor(getattr(cur_logp_sum, "arr", cur_logp_sum))
    if not hasattr(ref_logp_sum, "detach"):
        ref_logp_sum = torch.tensor(getattr(ref_logp_sum, "arr", ref_logp_sum))
    if not hasattr(ref_logp_sum_raw, "detach"):
        ref_logp_sum_raw = torch.tensor(
            getattr(ref_logp_sum_raw, "arr", ref_logp_sum_raw)
        )
    # Align denom tensor with the policy tensors so len normalization is consistent.
    if isinstance(denom_tok_tensor, torch.Tensor):
        cur_device = getattr(cur_logp_sum, "device", None)
        denom_device = getattr(denom_tok_tensor, "device", None)
        if cur_device is not None and denom_device is not None and cur_device != denom_device:
            try:
                denom_tok_tensor = denom_tok_tensor.to(cur_logp_sum)
            except (RuntimeError, TypeError, AttributeError):
                denom_tok_tensor = denom_tok_tensor.to("cpu")
                cur_logp_sum = cur_logp_sum.to("cpu")
                ref_logp_sum = ref_logp_sum.to("cpu")
                ref_logp_sum_raw = ref_logp_sum_raw.to("cpu")

    cur_logp_per_token = cur_logp_sum.detach() / denom_tok_tensor
    if ratio_ctx.weighting_cfg.len_norm_ref:
        ref_logp_per_token = ref_logp_sum.detach()
    else:
        ref_logp_per_token = ref_logp_sum_raw.detach() / denom_tok_tensor
    # Align dtype/device across tensors for downstream ops.
    ref_device = getattr(ref_logp_per_token, "device", None)
    cur_device = getattr(cur_logp_per_token, "device", None)
    ref_dtype = getattr(ref_logp_per_token, "dtype", None)
    cur_dtype = getattr(cur_logp_per_token, "dtype", None)
    if (
        ref_device is not None
        and cur_device is not None
        and (ref_device != cur_device or (ref_dtype and cur_dtype and ref_dtype != cur_dtype))
    ):
        try:
            ref_logp_per_token = ref_logp_per_token.to(cur_logp_per_token)
        except (AttributeError, RuntimeError, TypeError):
            ref_logp_per_token = ref_logp_per_token.to("cpu")
            cur_logp_per_token = cur_logp_per_token.to("cpu")
    try:
        cur_len = cur_logp_per_token.numel()
    except (AttributeError, TypeError):
        cur_len = len(getattr(cur_logp_per_token, "data", []))
    try:
        ref_len = ref_logp_per_token.numel()
    except (AttributeError, TypeError):
        ref_len = len(getattr(ref_logp_per_token, "data", []))
    if not cur_len or not ref_len:
        LOG.warning(
            "Skipping ratio diagnostics due to empty logp tensors | cur=%d | ref=%d",
            int(cur_len),
            int(ref_len),
        )
        return _ratio_stats_without_ref()
    if ref_len != cur_len:
        min_len = min(cur_len, ref_len)
        if min_len == 0:
            LOG.warning(
                "Skipping ratio diagnostics due to irreconcilable logp lengths | cur=%d | ref=%d",
                int(cur_len),
                int(ref_len),
            )
            return _ratio_stats_without_ref()
        LOG.warning(
            "Trimming ratio diagnostics tensors to minimum length | cur=%d | ref=%d | min=%d",
            int(cur_len),
            int(ref_len),
            int(min_len),
        )
        cur_logp_per_token = cur_logp_per_token[:min_len]
        ref_logp_per_token = ref_logp_per_token[:min_len]
    delta = (ref_logp_per_token - cur_logp_per_token).clamp(min=-60.0, max=60.0)
    if not hasattr(delta, "exp"):
        delta = torch.tensor(getattr(delta, "arr", getattr(delta, "data", delta)))
    ratio = delta.exp()
    kl_value = float((ratio - delta - 1.0).mean().detach().cpu().item())
    kl_bucket_means, kl_bucket_token_counts = _bucketized_kl_per_token(
        denom_tok_tensor, ratio - delta - 1.0
    )
    log_ratio = (cur_logp_per_token - ref_logp_per_token).clamp(min=-60.0, max=60.0)
    clip_ratio, low_mean, low_min, high_mean, high_max, region_mean = (
        _clip_region_metrics(log_ratio, ratio_ctx.clip_cfg)
    )
    return (
        kl_value,
        clip_ratio,
        float(low_mean),
        float(low_min),
        float(high_mean),
        float(high_max),
        float(region_mean),
        kl_bucket_means,
        kl_bucket_token_counts,
    )


def _ratio_stats_without_ref() -> Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    Dict[str, float],
    Dict[str, float],
]:
    """Default diagnostics when no reference stats exist.

    :returns: Zero-valued tuple used when reference stats are missing.
    :rtype: tuple[float, float, float, float, float, float, float, dict, dict]
    """
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}


def _ratio_diagnostics(ratio_ctx: RatioContext) -> BatchDiagnostics:
    """Summarize ratio statistics for logging.

    :param ratio_ctx: Ratio/weighting context for the current batch.
    :type ratio_ctx: RatioContext
    :returns: Diagnostic metrics consumed by the logging layer.
    :rtype: BatchDiagnostics
    """
    ref_stats = ratio_ctx.ref_stats
    ref_counts = getattr(ref_stats, "ref_tok_counts", None)
    try:
        count_len = ref_counts.numel() if ref_counts is not None else 0
    except (AttributeError, TypeError):
        count_len = (
            len(getattr(ref_counts, "data", [])) if ref_counts is not None else 0
        )
    if count_len > 0:
        stats = _ratio_stats_with_ref(ratio_ctx)
    else:
        stats = _ratio_stats_without_ref()
    (
        kl_value,
        clip_ratio,
        low_mean,
        low_min,
        high_mean,
        high_max,
        region_mean,
        kl_bucket_means,
        kl_bucket_token_counts,
    ) = stats
    return BatchDiagnostics(
        kl_value=kl_value,
        clip_ratio=clip_ratio,
        clip_ratio_low_mean=low_mean,
        clip_ratio_low_min=low_min,
        clip_ratio_high_mean=high_mean,
        clip_ratio_high_max=high_max,
        clip_ratio_region_mean=region_mean,
        kl_per_token_by_len_bucket=kl_bucket_means,
        kl_token_count_by_len_bucket=kl_bucket_token_counts,
    )


def _contrastive_seed_loss(
    seed_inputs: SeedInfoInputs,
    temperature: float = 0.1,
) -> Optional[torch.Tensor]:
    """Compute a contrastive InfoNCE-style loss over seed IDs."""

    if seed_inputs.pooled_hidden is None or seed_inputs.seed_ids is None:
        return None
    hidden = seed_inputs.pooled_hidden
    seed_ids = seed_inputs.seed_ids
    if hidden.numel() == 0 or seed_ids.numel() == 0:
        return None
    valid_mask = seed_ids >= 0
    if not valid_mask.any():
        return None
    hidden = hidden[valid_mask]
    seed_ids = seed_ids[valid_mask]
    if hidden.size(0) < 2:
        return None
    try:
        normalize_fn = getattr(torch.nn.functional, "normalize", None)
        if callable(normalize_fn):
            hidden = normalize_fn(hidden, dim=1)
        else:
            norm = hidden.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-6)
            hidden = hidden / norm
        sim = torch.matmul(hidden, hidden.t()) / max(temperature, 1e-4)
        sim.fill_diagonal_(float("-inf"))
        pos_mask = seed_ids.unsqueeze(1).eq(seed_ids.unsqueeze(0))
        pos_mask.fill_diagonal_(False)
        if not pos_mask.any():
            return None
        log_probs = torch.nn.functional.log_softmax(sim, dim=1)
    except (AttributeError, TypeError, RuntimeError):
        return None
    loss = -(log_probs[pos_mask]).mean()
    return loss


def evaluate_losses(
    group_data: GroupLossData,
    ratio_ctx: RatioContext,
    *,
    seed_inputs: Optional[SeedInfoInputs] = None,
    info_seed_lambda: float = 0.0,
    info_seed_temperature: float = 0.1,
    info_seed_loss_type: str = "infonce",
    info_seed_alpha_entropy: float = 0.0,
) -> Tuple[LossOutputs, BatchDiagnostics]:
    """Compute policy loss, clipping objectives, and diagnostics.

    :param group_data: Aggregated per-group tensors (weights, log-probs).
    :type group_data: GroupLossData
    :param ratio_ctx: Ratio context describing log-ratios and configuration.
    :type ratio_ctx: RatioContext
    :param seed_inputs: Optional pooled representations and seed labels.
    :type seed_inputs: SeedInfoInputs | None
    :param info_seed_lambda: Scaling applied to the auxiliary seed loss.
    :type info_seed_lambda: float
    :param info_seed_temperature: Temperature for the contrastive seed loss.
    :type info_seed_temperature: float
    :returns: Tuple containing ``LossOutputs`` and diagnostic statistics.
    :rtype: tuple[LossOutputs, BatchDiagnostics]
    """
    policy_loss = _policy_loss_from_groups(group_data)
    policy_loss, clip_loss_scalar = _apply_clip_objective(
        ratio_ctx, group_data, policy_loss
    )
    kl_loss_tensor, kl_loss_scalar, weighted_kl_loss_scalar = _kl_terms(ratio_ctx)
    _beta_raw = getattr(ratio_ctx.weighting_cfg, "beta", 0.0)
    beta_val = float(getattr(_beta_raw, "arr", _beta_raw))
    if not isinstance(kl_loss_tensor, torch.Tensor):
        kl_loss_tensor = torch.tensor(getattr(kl_loss_tensor, "arr", kl_loss_tensor))
    total_loss = policy_loss + beta_val * kl_loss_tensor
    # Optional MI-style entropy term: alpha * H(orig) - H(seed_aug)
    info_entropy_term = None
    if info_seed_alpha_entropy != 0.0 and seed_inputs is not None:
        weight_tensor = group_data.weight_tensor
        is_seed_aug = getattr(seed_inputs, "is_seed_aug", None)
        if is_seed_aug is not None:
            seed_mask = is_seed_aug.bool()
            if weight_tensor.numel() == seed_mask.numel():
                orig_weights = weight_tensor[~seed_mask]
                seed_weights = weight_tensor[seed_mask]
                if orig_weights.numel() > 0:
                    p = orig_weights / orig_weights.sum().clamp(min=1e-8)
                    h_orig = -(p * torch.log(p + 1e-12)).sum()
                else:
                    h_orig = torch.tensor(0.0, device=weight_tensor.device)
                if seed_weights.numel() > 0:
                    p = seed_weights / seed_weights.sum().clamp(min=1e-8)
                    h_seed = -(p * torch.log(p + 1e-12)).sum()
                else:
                    h_seed = torch.tensor(0.0, device=weight_tensor.device)
                info_entropy_term = (
                    info_seed_alpha_entropy * h_orig - info_seed_alpha_entropy * h_seed
                )
                total_loss = total_loss + info_entropy_term
    seed_loss = None
    if info_seed_lambda > 0.0 and seed_inputs is not None:
        if info_seed_loss_type == "ce" and seed_inputs.logits is not None:
            seed_loss = torch.nn.functional.cross_entropy(
                seed_inputs.logits, seed_inputs.seed_ids
            )
        else:
            seed_loss = _contrastive_seed_loss(
                seed_inputs, temperature=info_seed_temperature
            )
        if seed_loss is not None:
            total_loss = total_loss + info_seed_lambda * seed_loss
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
        seed_loss=seed_loss,
        info_entropy_term=info_entropy_term,
    )
    diagnostics = _ratio_diagnostics(ratio_ctx)
    return loss_outputs, diagnostics


__all__ = [
    "GroupLossData",
    "RatioContext",
    "SeedInfoInputs",
    "SequenceScores",
    "build_loss_inputs",
    "evaluate_losses",
    "LossInputConfig",
]
