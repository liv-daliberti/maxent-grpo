"""Prompt-group helpers for listwise MaxEnt on top of OAT PPO/Dr.GRPO."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Iterator, Optional, Sequence

import torch

_OBJECTIVE_ALIASES = {
    None: "grpo",
    "": "grpo",
    "baseline": "grpo",
    "grpo": "grpo",
    "listwise": "maxent_listwise",
    "maxent_listwise": "maxent_listwise",
}
_CLIP_MODE_ALIASES = {
    None: "sequence",
    "": "sequence",
    "default": "sequence",
    "seq": "sequence",
    "sequence": "sequence",
    "token": "token",
    "ppo_token": "token",
    "none": "none",
    "off": "none",
    "disabled": "none",
}
_TAU_METRIC_EMA_DECAY = 0.9


@dataclass
class ListwiseControllerState:
    """Mutable controller history used by adaptive tau updates."""

    tau_metric_ema: float | None = None
    tau_log: float | None = None


@dataclass
class SanitizedTokenIdsResult:
    """Best-effort sanitized token IDs plus replacement metadata."""

    token_ids: torch.Tensor
    invalid_count: int = 0
    replacement_id: int | None = None
    min_invalid: int | None = None
    max_invalid: int | None = None



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
    semantic_diagnostics: "SemanticWeightDiagnostics | None" = None


@dataclass
class SemanticClusterBundle:
    """Semantic cluster assignments and masses for one prompt-major minibatch."""

    cluster_ids_grouped: torch.Tensor
    num_clusters_per_group: torch.Tensor
    semantic_entropy_grouped: torch.Tensor
    semantic_valid_row_mask_grouped: torch.Tensor


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



def normalize_oat_objective(value: object) -> str:
    """Return the canonical OAT-side objective label."""

    if value is None:
        candidate = None
    else:
        candidate = str(value).strip().lower()
    normalized = _OBJECTIVE_ALIASES.get(candidate, candidate)
    if normalized not in {"grpo", "maxent_listwise"}:
        raise ValueError("objective must be one of: grpo, maxent_listwise")
    return normalized


def normalize_maxent_clip_mode(value: object) -> str:
    """Return the canonical listwise clip-mode label."""

    if value is None:
        candidate = None
    else:
        candidate = str(value).strip().lower()
    normalized = _CLIP_MODE_ALIASES.get(candidate, candidate)
    if normalized not in {"sequence", "token", "none"}:
        raise ValueError("maxent_clip_mode must be one of: sequence, token, none")
    return normalized


def coerce_non_negative_float(value: object, *, default: float = 0.0) -> float:
    """Return a finite non-negative float for config-like inputs."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return max(numeric, 0.0)


def coerce_optional_int(value: object | None) -> Optional[int]:
    """Return ``value`` as an int when possible."""

    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _get_config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from config-like objects without assuming a concrete type."""

    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _get_embedding_vocab_size(model: Any, config: Any = None) -> Optional[int]:
    """Return the model embedding vocab size when exposed."""

    get_embeddings = getattr(model, "get_input_embeddings", None)
    embedding_module = None
    if callable(get_embeddings):
        try:
            embedding_module = get_embeddings()
        except Exception:
            embedding_module = None
    num_embeddings = coerce_optional_int(getattr(embedding_module, "num_embeddings", None))
    if isinstance(num_embeddings, int) and num_embeddings > 0:
        return num_embeddings
    weight = getattr(embedding_module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 1:
        size_value = coerce_optional_int(shape[0])
        if isinstance(size_value, int) and size_value > 0:
            return size_value
    config_vocab_size = coerce_optional_int(_get_config_value(config, "vocab_size", None))
    if isinstance(config_vocab_size, int) and config_vocab_size > 0:
        return config_vocab_size
    return None


def resolve_model_vocab_limit(model: Any) -> Optional[int]:
    """Return the largest positive vocab-size limit exposed by the model."""

    config = getattr(model, "config", None)
    candidates = [
        value
        for value in (
            _get_embedding_vocab_size(model, config),
            coerce_optional_int(_get_config_value(config, "vocab_size", None)),
            coerce_optional_int(getattr(model, "vocab_size", None)),
        )
        if isinstance(value, int) and value > 0
    ]
    if not candidates:
        return None
    return max(candidates)


def resolve_tokenizer_vocab_limit(tokenizer: Any) -> Optional[int]:
    """Return the full addressable tokenizer range, including added tokens."""

    candidates = []
    vocab_size = coerce_optional_int(getattr(tokenizer, "vocab_size", None))
    if isinstance(vocab_size, int) and vocab_size > 0:
        candidates.append(vocab_size)
    try:
        tokenizer_len = coerce_optional_int(len(tokenizer))
    except Exception:
        tokenizer_len = None
    if isinstance(tokenizer_len, int) and tokenizer_len > 0:
        candidates.append(tokenizer_len)
    if not candidates:
        return None
    return max(candidates)


def resolve_token_id_upper_bound(model: Any, tokenizer: Any = None) -> Optional[int]:
    """Return a conservative upper bound for valid token IDs."""

    candidates = []
    model_limit = resolve_model_vocab_limit(model)
    if isinstance(model_limit, int) and model_limit > 0:
        candidates.append(model_limit)
    tokenizer_limit = resolve_tokenizer_vocab_limit(tokenizer)
    if isinstance(tokenizer_limit, int) and tokenizer_limit > 0:
        candidates.append(tokenizer_limit)
    if not candidates:
        return None
    return min(candidates)


def mask_invalid_logit_columns(
    logits: torch.Tensor,
    *,
    valid_vocab_size: Optional[int],
) -> torch.Tensor:
    """Mask logits that correspond to tokenizer-inaccessible token IDs."""

    if not isinstance(valid_vocab_size, int) or valid_vocab_size <= 0:
        return logits
    if logits.ndim < 1:
        return logits
    if int(logits.size(-1)) <= valid_vocab_size:
        return logits
    masked = logits.clone()
    masked[..., valid_vocab_size:] = torch.finfo(masked.dtype).min
    return masked


def sanitize_scoring_token_ids(
    token_ids: torch.Tensor,
    *,
    upper_bound: Optional[int],
    tokenizer: Any = None,
) -> SanitizedTokenIdsResult:
    """Clamp scorer token IDs into range before model/gather indexing."""

    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    if token_ids.dtype.is_floating_point or token_ids.dtype == torch.bool:
        return SanitizedTokenIdsResult(token_ids=token_ids)
    if not isinstance(upper_bound, int) or upper_bound <= 0:
        return SanitizedTokenIdsResult(token_ids=token_ids)

    replacement_id = coerce_optional_int(getattr(tokenizer, "pad_token_id", None))
    if replacement_id is None or replacement_id < 0 or replacement_id >= upper_bound:
        replacement_id = coerce_optional_int(getattr(tokenizer, "eos_token_id", None))
    if replacement_id is None or replacement_id < 0 or replacement_id >= upper_bound:
        replacement_id = max(upper_bound - 1, 0)

    invalid_mask = (token_ids < 0) | (token_ids >= upper_bound)
    invalid_count = int(invalid_mask.to(torch.long).sum().item())
    if invalid_count <= 0:
        return SanitizedTokenIdsResult(token_ids=token_ids)

    invalid_vals = token_ids[invalid_mask]
    min_invalid = int(invalid_vals.min().item()) if invalid_vals.numel() > 0 else None
    max_invalid = int(invalid_vals.max().item()) if invalid_vals.numel() > 0 else None
    sanitized = token_ids.clone()
    sanitized[invalid_mask] = int(replacement_id)
    return SanitizedTokenIdsResult(
        token_ids=sanitized,
        invalid_count=invalid_count,
        replacement_id=int(replacement_id),
        min_invalid=min_invalid,
        max_invalid=max_invalid,
    )


def collect_weight_entropy(
    weights_grouped: torch.Tensor | Sequence[Sequence[float]],
) -> tuple[float, float, float, list[float]]:
    """Summarize entropy statistics for grouped listwise weights."""

    if isinstance(weights_grouped, torch.Tensor):
        weight_groups = weights_grouped.detach().cpu().tolist()
    else:
        weight_groups = [list(group) for group in weights_grouped]
    entropy_vals: list[float] = []
    entropy_advantage_samples: list[float] = []
    for weight_group in weight_groups:
        if not weight_group:
            continue
        try:
            weight_tensor = torch.tensor(weight_group, dtype=torch.float32)
            clamped = weight_tensor.clamp(min=1e-12)
            entropy_vals.append(float((-(clamped.log()) * weight_tensor).sum().item()))
        except (TypeError, ValueError, RuntimeError):
            probs = [max(float(weight), 1e-12) for weight in weight_group]
            total = sum(probs)
            if total <= 0.0:
                continue
            probs = [value / total for value in probs]
            entropy_vals.append(float(-sum(value * math.log(value) for value in probs)))
        baseline = 1.0 / float(len(weight_group))
        entropy_advantage_samples.extend([float(val) - baseline for val in weight_group])
    if not entropy_vals:
        return 0.0, 0.0, 0.0, entropy_advantage_samples
    return (
        float(sum(entropy_vals) / len(entropy_vals)),
        float(min(entropy_vals)),
        float(max(entropy_vals)),
        entropy_advantage_samples,
    )


def collect_weight_entropy_stats(
    weights_grouped: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute grouped entropy statistics without copying weights to the host."""

    if not isinstance(weights_grouped, torch.Tensor):
        raise TypeError("weights_grouped must be a torch.Tensor")
    if weights_grouped.ndim == 1:
        weights_grouped = weights_grouped.unsqueeze(0)
    if weights_grouped.numel() == 0 or int(weights_grouped.size(0)) == 0:
        zeros = torch.zeros((), dtype=torch.float32, device=weights_grouped.device)
        return zeros, zeros, zeros
    probs = weights_grouped.to(dtype=torch.float32).clamp(min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
    entropy_vals = -(probs.clamp(min=1e-12).log() * probs).sum(dim=1)
    return entropy_vals.mean(), entropy_vals.min(), entropy_vals.max()


def resolve_listwise_target_entropy(
    *,
    target_entropy: float | None,
    target_entropy_start: float | None,
    target_entropy_peak: float | None,
    target_entropy_peak_step: int,
    target_entropy_final: float | None,
    target_entropy_horizon: int,
    global_step: int,
) -> Optional[float]:
    """Return the active target entropy, honoring optional annealing settings."""

    if (
        target_entropy is None
        and target_entropy_start is None
        and target_entropy_peak is None
        and target_entropy_final is None
    ):
        return None
    if (
        target_entropy_start is None
        and target_entropy_peak is None
        and target_entropy_final is None
    ):
        return float(target_entropy) if target_entropy is not None else None
    start = (
        float(target_entropy_start)
        if target_entropy_start is not None
        else float(target_entropy)
    )
    peak = (
        float(target_entropy_peak)
        if target_entropy_peak is not None
        else None
    )
    final = (
        float(target_entropy_final)
        if target_entropy_final is not None
        else float(target_entropy)
    )
    if not math.isfinite(start) or not math.isfinite(final):
        return None
    if peak is not None and not math.isfinite(peak):
        return None
    horizon = max(int(target_entropy_horizon), 0)
    peak_step = max(int(target_entropy_peak_step), 0)
    step = max(int(global_step), 0)
    if peak is None:
        if horizon <= 0:
            return final
        frac = min(step, horizon) / float(horizon)
        return start + (final - start) * frac
    if horizon <= 0:
        return final
    if peak_step <= 0:
        if step >= horizon:
            return final
        down_frac = min(step, horizon) / float(horizon)
        return peak + (final - peak) * down_frac
    if step <= peak_step:
        up_frac = min(step, peak_step) / float(peak_step)
        return start + (peak - start) * up_frac
    if horizon <= peak_step:
        return peak
    down_frac = min(step - peak_step, horizon - peak_step) / float(
        horizon - peak_step
    )
    return peak + (final - peak) * down_frac


def update_listwise_tau_metric_ema(
    state: ListwiseControllerState | None,
    *,
    measured_metric: float | None,
) -> float | None:
    """Update and return the smoothed controller statistic for tau adaptation."""

    if state is None:
        return None
    if not isinstance(measured_metric, (int, float)) or not math.isfinite(
        float(measured_metric)
    ):
        return None
    if (
        not isinstance(state.tau_metric_ema, (int, float))
        or not math.isfinite(state.tau_metric_ema)
    ):
        state.tau_metric_ema = float(measured_metric)
    else:
        state.tau_metric_ema = (
            _TAU_METRIC_EMA_DECAY * float(state.tau_metric_ema)
            + (1.0 - _TAU_METRIC_EMA_DECAY) * float(measured_metric)
        )
    return float(state.tau_metric_ema)


def update_listwise_tau_entropy_ema(
    state: ListwiseControllerState | None,
    *,
    measured_entropy: float | None,
) -> float | None:
    """Backward-compatible alias for the old tau-entropy EMA helper."""

    return update_listwise_tau_metric_ema(
        state,
        measured_metric=measured_entropy,
    )


def clamp_listwise_tau(
    current_tau: float,
    *,
    tau_min: float,
    tau_max: float,
) -> float:
    """Project tau into the configured positive range."""

    new_tau = max(float(current_tau), max(float(tau_min), 1e-8))
    safe_tau_max = float(tau_max)
    if math.isfinite(safe_tau_max) and safe_tau_max > 0.0:
        new_tau = min(new_tau, safe_tau_max)
    return float(new_tau)


def compute_learnable_tau_loss(
    tau_log: torch.Tensor,
    *,
    measured_metric: float | torch.Tensor | None,
    target_metric: float | torch.Tensor | None,
) -> torch.Tensor | None:
    """Return the SAC-style log-tau objective for scalar-metric matching."""

    if measured_metric is None or target_metric is None:
        return None
    if tau_log.numel() != 1:
        raise ValueError("tau_log must contain exactly one scalar value")

    measured = torch.as_tensor(
        measured_metric,
        device=tau_log.device,
        dtype=tau_log.dtype,
    )
    target = torch.as_tensor(
        target_metric,
        device=tau_log.device,
        dtype=tau_log.dtype,
    )
    if not bool(torch.isfinite(measured).all()) or not bool(torch.isfinite(target).all()):
        return None
    return tau_log.reshape(()) * (measured.detach() - target.detach())


def maybe_update_listwise_tau(
    current_tau: float,
    *,
    measured_metric: float | None,
    global_step: int,
    state: ListwiseControllerState | None,
    target_metric: float | None,
    target_metric_start: float | None,
    target_metric_peak: float | None,
    target_metric_peak_step: int,
    target_metric_final: float | None,
    target_metric_horizon: int,
    tau_lr: float,
    tau_min: float,
    tau_max: float,
    tau_warmup_steps: int,
) -> float:
    """Return the next tau under the simple scalar-target controller."""

    active_target = resolve_listwise_target_entropy(
        target_entropy=target_metric,
        target_entropy_start=target_metric_start,
        target_entropy_peak=target_metric_peak,
        target_entropy_peak_step=target_metric_peak_step,
        target_entropy_final=target_metric_final,
        target_entropy_horizon=target_metric_horizon,
        global_step=global_step,
    )
    if active_target is None:
        return float(current_tau)
    if global_step <= max(0, int(tau_warmup_steps)):
        return float(current_tau)
    if not isinstance(measured_metric, (int, float)) or not math.isfinite(
        float(measured_metric)
    ):
        return float(current_tau)
    safe_tau_lr = float(tau_lr)
    if not math.isfinite(safe_tau_lr) or safe_tau_lr <= 0.0:
        return float(current_tau)

    if state is None:
        state = ListwiseControllerState()
    if not isinstance(state.tau_log, (int, float)) or not math.isfinite(state.tau_log):
        state.tau_log = math.log(max(float(current_tau), 1e-8))
    ema_metric = update_listwise_tau_metric_ema(
        state,
        measured_metric=float(measured_metric),
    )
    if ema_metric is None:
        return float(current_tau)

    error = float(active_target) - float(ema_metric)
    if abs(error) < 1e-12:
        return float(current_tau)
    tau_log = float(state.tau_log) + safe_tau_lr * error
    new_tau = clamp_listwise_tau(
        math.exp(tau_log),
        tau_min=tau_min,
        tau_max=tau_max,
    )
    state.tau_log = math.log(max(new_tau, 1e-8))
    return float(new_tau)


def maybe_update_listwise_beta(
    current_beta: float,
    *,
    measured_kl: float | None,
    kl_target: float,
    kl_horizon: int,
    kl_ctl_step_size: float,
) -> float:
    """Return the next beta under the simple KL controller."""

    safe_target = float(kl_target)
    safe_horizon = int(kl_horizon)
    safe_step = float(kl_ctl_step_size)
    if safe_target <= 0.0 or safe_horizon <= 0 or safe_step <= 0.0:
        return float(current_beta)
    if not isinstance(measured_kl, (int, float)) or not math.isfinite(float(measured_kl)):
        return float(current_beta)

    ratio = float(measured_kl) / max(safe_target, 1e-8)
    error = ratio - 1.0
    if abs(error) < 1e-8:
        return float(current_beta)
    clipped_error = max(min(error, safe_step), -safe_step)
    scale = 1.0 + clipped_error / float(max(safe_horizon, 1))
    if scale <= 0.0:
        scale = 1e-6
    return max(0.0, float(current_beta) * scale)


def reshape_prompt_major_tensor(
    tensor: torch.Tensor,
    group_size: int,
) -> Optional[torch.Tensor]:
    """Reshape a flat prompt-major tensor into ``[prompts, generations, ...]``."""

    if group_size <= 0:
        return None
    total_rows = int(tensor.size(0))
    if total_rows <= 0 or total_rows % group_size != 0:
        return None
    num_prompts = total_rows // group_size
    if num_prompts <= 0:
        return None
    shape = (num_prompts, group_size) + tuple(tensor.shape[1:])
    return tensor.reshape(shape).contiguous()


def flatten_prompt_major_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten a prompt-major tensor back into ``[rows, ...]`` order."""

    if tensor.dim() < 2:
        return tensor.reshape(-1)
    shape = (-1,) + tuple(tensor.shape[2:])
    return tensor.reshape(shape).contiguous()


def iter_fixed_row_chunks(total_rows: int, *, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield fixed-size ``[start, stop)`` row chunks for synchronized distributed work."""

    safe_total_rows = max(int(total_rows), 0)
    safe_chunk_size = max(int(chunk_size), 1)
    for start in range(0, safe_total_rows, safe_chunk_size):
        yield start, min(start + safe_chunk_size, safe_total_rows)


def iter_budgeted_row_chunks(
    token_counts: Sequence[int],
    *,
    max_rows: int,
    token_budget: int,
) -> Iterator[tuple[int, int]]:
    """Yield prefix chunks whose padded token load stays within the requested budget."""

    safe_counts = [max(int(count), 1) for count in token_counts]
    if not safe_counts:
        return
    safe_max_rows = max(int(max_rows), 1)
    safe_budget = max(int(token_budget), 0)
    if safe_budget <= 0:
        yield from iter_fixed_row_chunks(len(safe_counts), chunk_size=safe_max_rows)
        return
    start = 0
    total_rows = len(safe_counts)
    while start < total_rows:
        remaining_rows = total_rows - start
        chunk_rows = choose_prefix_chunk_size_for_token_budget(
            safe_counts[start:],
            max_rows=min(safe_max_rows, remaining_rows),
            token_budget=safe_budget,
        )
        stop = min(start + max(chunk_rows, 1), total_rows)
        yield start, stop
        start = stop


def choose_prefix_chunk_size_for_token_budget(
    token_counts: Sequence[int],
    *,
    max_rows: int,
    token_budget: int,
) -> int:
    """Choose the largest prefix chunk whose padded token load fits the budget."""

    safe_max_rows = max(int(max_rows), 1)
    if token_budget <= 0:
        return safe_max_rows
    capped_counts = [max(int(count), 1) for count in token_counts[:safe_max_rows]]
    if not capped_counts:
        return 1
    max_seen = 0
    best = 1
    for index, count in enumerate(capped_counts, start=1):
        max_seen = max(max_seen, count)
        if max_seen * index <= token_budget:
            best = index
        else:
            break
    return max(best, 1)


def cap_last_valid_token_pos_for_zero_advantage(
    *,
    prompt_len: int,
    last_valid_token_pos: int,
    response_token_budget: int,
) -> int:
    """Trim zero-advantage single-row updates to a short prompt+response prefix."""

    safe_last = max(int(last_valid_token_pos), 0)
    if safe_last <= 0:
        return 0
    safe_prompt_len = max(int(prompt_len), 0)
    minimum_last = safe_prompt_len + 1
    if safe_last <= minimum_last:
        return safe_last
    safe_response_budget = max(int(response_token_budget), 1)
    return min(safe_last, safe_prompt_len + safe_response_budget)


def normalize_listwise_q_targets(
    q_grouped: torch.Tensor,
    *,
    num_prompts: int,
    group_size: int,
    context: str,
) -> torch.Tensor:
    """Validate and simplex-normalize per-prompt listwise q targets."""

    if q_grouped.dim() != 2:
        raise ValueError(f"{context} requires rank-2 listwise q targets.")
    expected_shape = (num_prompts, group_size)
    actual_shape = (int(q_grouped.size(0)), int(q_grouped.size(1)))
    if actual_shape != expected_shape:
        raise ValueError(
            f"{context} requires listwise q targets with shape {expected_shape}, "
            f"got {actual_shape}."
        )
    if not torch.isfinite(q_grouped).all():
        raise ValueError(f"{context} requires finite listwise q targets.")
    if (q_grouped < 0).any():
        raise ValueError(f"{context} requires non-negative listwise q targets.")
    row_sums = q_grouped.sum(dim=1, keepdim=True)
    if (row_sums <= 0).any():
        raise ValueError(f"{context} requires listwise q targets with positive mass.")
    return q_grouped / row_sums


def mask_and_normalize_listwise_q_targets(
    q_grouped: torch.Tensor,
    *,
    row_mask_grouped: torch.Tensor,
    context: str,
) -> torch.Tensor:
    """Restrict grouped q targets to valid rows and renormalize per prompt."""

    if row_mask_grouped.shape != q_grouped.shape:
        raise ValueError(
            f"{context} requires row_mask_grouped with shape {tuple(q_grouped.shape)}, "
            f"got {tuple(row_mask_grouped.shape)}."
        )
    normalized = normalize_listwise_q_targets(
        q_grouped,
        num_prompts=int(q_grouped.size(0)),
        group_size=int(q_grouped.size(1)),
        context=context,
    )
    valid_mask = row_mask_grouped.to(torch.bool)
    masked = torch.where(valid_mask, normalized, torch.zeros_like(normalized))
    row_sums = masked.sum(dim=1, keepdim=True)
    has_valid_mass = row_sums > 0
    renormalized = masked / row_sums.clamp(min=1e-12)
    return torch.where(has_valid_mass, renormalized, normalized)


def masked_group_log_softmax(
    values_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor,
) -> torch.Tensor:
    """Apply a per-group log-softmax over only the valid rows."""

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
    shifted = torch.where(valid_mask, masked_values - max_vals, torch.zeros_like(masked_values))
    exp_shifted = torch.where(valid_mask, torch.exp(shifted), torch.zeros_like(shifted))
    log_denom = torch.log(exp_shifted.sum(dim=1, keepdim=True).clamp(min=1e-12))
    log_probs = shifted - log_denom
    return torch.where(valid_mask, log_probs, torch.zeros_like(log_probs))


def build_listwise_q_targets(
    rewards: torch.Tensor,
    *,
    group_size: int,
    temperature: float,
    epsilon: float,
) -> torch.Tensor:
    """Convert flat prompt-major rewards into grouped listwise q targets."""

    grouped_rewards = reshape_prompt_major_tensor(rewards, group_size)
    if grouped_rewards is None:
        raise ValueError(
            "Listwise MaxEnt rewards must arrive as whole prompt groups with flat "
            f"batch size divisible by num_samples={group_size}."
        )
    safe_temperature = max(coerce_non_negative_float(temperature, default=1.0), 1e-8)
    q_grouped = torch.softmax(grouped_rewards / safe_temperature, dim=1)
    eps = coerce_non_negative_float(epsilon, default=1e-6)
    if eps > 0.0:
        max_eps = max((1.0 / float(max(q_grouped.size(1), 1))) - 1e-8, 0.0)
        eps = min(eps, max_eps)
        if eps > 0.0:
            q_grouped = q_grouped * (1.0 - eps * q_grouped.size(1)) + eps
            q_grouped = q_grouped / q_grouped.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return normalize_listwise_q_targets(
        q_grouped,
        num_prompts=int(q_grouped.size(0)),
        group_size=group_size,
        context="Listwise MaxEnt rollout targets",
    )


def aggregate_masked_row_values(
    values: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    constant_normalizer: float | None = None,
) -> torch.Tensor:
    """Aggregate tokenwise values into one scalar per row."""

    if values.shape != response_masks.shape:
        raise ValueError("values and response_masks must have matching shapes.")
    mask = response_masks.to(dtype=values.dtype)
    if (
        isinstance(constant_normalizer, (int, float))
        and math.isfinite(float(constant_normalizer))
        and float(constant_normalizer) > 0.0
    ):
        return (values * mask).sum(dim=1) / float(constant_normalizer)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def build_padded_action_logprobs(
    action_logprobs: Sequence[Sequence[float]],
    response_masks: torch.Tensor,
) -> torch.Tensor:
    """Pad actor-side per-token log-probs onto the learner response-mask grid."""

    padded = torch.zeros(
        response_masks.shape,
        device=response_masks.device,
        dtype=torch.float32,
    )
    if len(action_logprobs) != int(response_masks.size(0)):
        return padded
    for row_idx, row_logprobs in enumerate(action_logprobs):
        valid_positions = torch.where(response_masks[row_idx])[0]
        if valid_positions.numel() <= 0:
            continue
        if row_logprobs is None:
            continue
        width = min(len(row_logprobs), int(valid_positions.numel()))
        if width <= 0:
            continue
        padded[row_idx, valid_positions[:width]] = torch.tensor(
            list(row_logprobs[:width]),
            device=response_masks.device,
            dtype=torch.float32,
        )
    return padded


def iter_grouped_minibatch_indices(
    *,
    total_rows: int,
    group_size: int,
    flat_batch_size: int,
    device: Optional[torch.device] = None,
    prompt_permutation: Optional[Iterable[int]] = None,
) -> Iterator[torch.Tensor]:
    """Yield flat row indices while preserving whole prompt groups."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if flat_batch_size <= 0:
        raise ValueError("flat_batch_size must be positive")
    if flat_batch_size % group_size != 0:
        raise ValueError(
            "Listwise MaxEnt requires train_batch_size_per_device to be divisible "
            "by num_samples so each microbatch contains whole prompt groups."
        )
    if total_rows <= 0 or total_rows % group_size != 0:
        raise ValueError(
            "Listwise MaxEnt requires the flat rollout batch size to be divisible "
            "by num_samples."
        )

    num_prompts = total_rows // group_size
    prompts_per_batch = flat_batch_size // group_size
    grouped_indices = torch.arange(total_rows, device=device).reshape(num_prompts, group_size)
    if prompt_permutation is None:
        prompt_order = torch.randperm(num_prompts, device=device)
    else:
        prompt_order = torch.as_tensor(
            list(prompt_permutation),
            device=device,
            dtype=torch.long,
        )
        if prompt_order.numel() != num_prompts:
            raise ValueError("prompt_permutation must cover every prompt group exactly once.")
    for start in range(0, num_prompts, prompts_per_batch):
        stop = min(start + prompts_per_batch, num_prompts)
        yield grouped_indices[prompt_order[start:stop]].reshape(-1)


def compute_listwise_weights(
    *,
    q_grouped: torch.Tensor,
    ref_seq_logps_grouped: torch.Tensor,
    tau: float,
    beta: float,
) -> torch.Tensor:
    """Return paper-style listwise MaxEnt posterior weights for each prompt group."""

    if q_grouped.shape != ref_seq_logps_grouped.shape:
        raise ValueError("q_grouped and ref_seq_logps_grouped must have matching shapes.")
    safe_tau = max(coerce_non_negative_float(tau, default=0.0), 1e-8)
    safe_beta = coerce_non_negative_float(beta, default=0.0)
    safe_temperature = max(safe_tau + safe_beta, 1e-8)
    positive_q = q_grouped > 0
    neg_inf = torch.full_like(q_grouped, torch.finfo(q_grouped.dtype).min)
    log_terms = torch.where(
        positive_q,
        torch.log(q_grouped.clamp(min=1e-12)) / safe_temperature,
        neg_inf,
    )
    if safe_beta > 0.0:
        ref_term = (safe_beta * ref_seq_logps_grouped) / safe_temperature
        log_terms = torch.where(positive_q, log_terms + ref_term, neg_inf)
    return torch.softmax(log_terms, dim=1)




def _safe_logsumexp(values: torch.Tensor, *, dim: int, keepdim: bool = False) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim, keepdim=keepdim)


def build_semantic_cluster_bundle(
    *,
    final_answer_keys_grouped: Sequence[Sequence[str | None]],
    valid_row_mask_grouped: torch.Tensor,
    reasoning_signature_keys_grouped: Sequence[Sequence[str | None]] | None = None,
) -> SemanticClusterBundle:
    """Cluster candidates by final-answer key and optional reasoning signature.

    The returned semantic entropy is the *normalized empirical cluster entropy*
    induced by the observed sample counts inside each prompt group. The coarse
    cluster key is always the normalized final answer string. When a compressed
    structural reasoning signature is available, rows with the same answer key
    are further split by the tuple ``(answer_key, reasoning_signature)``.
    Rows without a usable signature fall back to an answer-only bucket so they
    stay eligible for the semantic path without receiving free diversity credit
    from verbose traces.

        H_sem_norm = H(counts / sum counts) / log K,

    with the convention H_sem_norm = 0 when K <= 1.
    """

    if valid_row_mask_grouped.dim() != 2:
        raise ValueError("valid_row_mask_grouped must have shape [prompts, group].")
    num_prompts, group_size = valid_row_mask_grouped.shape
    if len(final_answer_keys_grouped) != num_prompts:
        raise ValueError("final_answer_keys_grouped must match num_prompts.")
    if reasoning_signature_keys_grouped is not None and len(reasoning_signature_keys_grouped) != num_prompts:
        raise ValueError("reasoning_signature_keys_grouped must match num_prompts.")

    cluster_ids = torch.full(
        (num_prompts, group_size),
        -1,
        device=valid_row_mask_grouped.device,
        dtype=torch.long,
    )
    num_clusters = torch.zeros(
        (num_prompts,),
        device=valid_row_mask_grouped.device,
        dtype=torch.long,
    )
    semantic_entropy = torch.zeros(
        (num_prompts,),
        device=valid_row_mask_grouped.device,
        dtype=torch.float32,
    )
    valid_mask = valid_row_mask_grouped.to(torch.bool)
    semantic_valid_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

    for p in range(num_prompts):
        keys = list(final_answer_keys_grouped[p])
        if len(keys) != group_size:
            raise ValueError("Each prompt must provide one final-answer key per candidate.")
        if reasoning_signature_keys_grouped is None:
            signatures = [None] * group_size
        else:
            signatures = list(reasoning_signature_keys_grouped[p])
            if len(signatures) != group_size:
                raise ValueError(
                    "Each prompt must provide one reasoning signature per candidate."
                )
        answer_has_signature: dict[str, bool] = {}
        for key, signature in zip(keys, signatures):
            if key is None or signature is None:
                continue
            answer_has_signature[key] = True
        cluster_key_to_id: dict[str, int] = {}
        cluster_counts: list[int] = []
        next_cluster = 0
        for i in range(group_size):
            if not bool(valid_mask[p, i].item()):
                continue
            key = keys[i]
            if key is None:
                # Missing or malformed answers should not receive semantic
                # entropy credit as singleton clusters.
                continue
            signature = signatures[i]
            if signature is None and answer_has_signature.get(key, False):
                # If this answer bucket already has a structural signature from
                # another row, an unlabeled row should keep its base token mass
                # but should not create extra semantic entropy by pretending to
                # be a new reasoning category.
                continue
            if signature is None:
                cluster_key = f"answer::{key}"
            else:
                cluster_key = f"answer::{key}||sig::{signature}"
            assigned = cluster_key_to_id.get(cluster_key)
            if assigned is None:
                assigned = next_cluster
                next_cluster += 1
                cluster_key_to_id[cluster_key] = assigned
                cluster_counts.append(1)
            else:
                cluster_counts[assigned] = cluster_counts[assigned] + 1
            cluster_ids[p, i] = assigned
            semantic_valid_mask[p, i] = True
        num_clusters[p] = next_cluster
        if next_cluster > 1:
            counts = torch.tensor(
                cluster_counts,
                device=valid_row_mask_grouped.device,
                dtype=torch.float32,
            )
            probs = counts / counts.sum().clamp(min=1e-12)
            semantic_entropy[p] = -(
                probs * probs.clamp(min=1e-12).log()
            ).sum() / math.log(float(next_cluster))
        else:
            semantic_entropy[p] = 0.0
    return SemanticClusterBundle(
        cluster_ids_grouped=cluster_ids,
        num_clusters_per_group=num_clusters,
        semantic_entropy_grouped=semantic_entropy,
        semantic_valid_row_mask_grouped=semantic_valid_mask,
    )

def compute_normalized_semantic_cluster_entropy(
    *,
    candidate_probs_grouped: torch.Tensor,
    cluster_ids_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor | None = None,
    normalizer_group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized semantic entropy and observed cluster counts per prompt.

    The entropy is computed over cluster masses induced by ``candidate_probs_grouped``:

        M_k = sum_{i in C_k} p_i,
        H_sem_norm(M) = H(M) / log G,

    where ``G`` defaults to the observed cluster count ``K`` unless
    ``normalizer_group_size`` is provided explicitly. This allows callers to
    reproduce paper-style normalization by the fixed rollout count rather than
    the number of observed semantic clusters.
    """

    if candidate_probs_grouped.shape != cluster_ids_grouped.shape:
        raise ValueError("candidate_probs_grouped and cluster_ids_grouped must match.")
    if valid_row_mask_grouped is None:
        valid_mask = torch.ones_like(candidate_probs_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != candidate_probs_grouped.shape:
            raise ValueError("valid_row_mask_grouped must match candidate_probs_grouped.")
        valid_mask = valid_row_mask_grouped.to(torch.bool)

    probs_grouped = torch.where(
        valid_mask,
        candidate_probs_grouped.to(torch.float32),
        torch.zeros_like(candidate_probs_grouped, dtype=torch.float32),
    )
    probs_grouped = probs_grouped / probs_grouped.sum(dim=1, keepdim=True).clamp(min=1e-12)

    num_prompts = int(candidate_probs_grouped.size(0))
    entropy = torch.zeros((num_prompts,), device=candidate_probs_grouped.device, dtype=torch.float32)
    cluster_count = torch.zeros((num_prompts,), device=candidate_probs_grouped.device, dtype=torch.float32)
    safe_normalizer_group_size = (
        None
        if normalizer_group_size is None
        else max(int(normalizer_group_size), 0)
    )

    for p in range(num_prompts):
        valid_clusters = cluster_ids_grouped[p][valid_mask[p] & (cluster_ids_grouped[p] >= 0)]
        if valid_clusters.numel() <= 0:
            continue
        unique_clusters = torch.unique(valid_clusters, sorted=True)
        k = int(unique_clusters.numel())
        cluster_count[p] = float(k)
        normalizer_count = (
            k if safe_normalizer_group_size is None else safe_normalizer_group_size
        )
        if k <= 1 or normalizer_count <= 1:
            entropy[p] = 0.0
            continue
        masses = []
        for cid in unique_clusters.tolist():
            member_mask = valid_mask[p] & (cluster_ids_grouped[p] == cid)
            masses.append(probs_grouped[p][member_mask].sum())
        masses_t = torch.stack(masses).clamp(min=1e-12)
        masses_t = masses_t / masses_t.sum().clamp(min=1e-12)
        entropy[p] = -(masses_t * masses_t.log()).sum() / math.log(
            float(normalizer_count)
        )

    return entropy, cluster_count



def compute_semantic_cluster_weights_from_utilities(
    *,
    utility_grouped: torch.Tensor,
    ref_seq_logps_grouped: torch.Tensor,
    cluster_ids_grouped: torch.Tensor,
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
) -> tuple[torch.Tensor, SemanticWeightDiagnostics]:
    """Return baseline-plus-residual competitive semantic weights.

    The baseline DrX posterior ``q`` remains primary. Semantic structure only
    re-allocates a small prompt-local residual budget over *competitive* semantic
    modes: modes close enough in score to the best observed mode and limited by
    a top-k cap. Rows without a semantic label keep their exact baseline mass.
    """

    if utility_grouped.shape != ref_seq_logps_grouped.shape:
        raise ValueError("utility_grouped and ref_seq_logps_grouped must have matching shapes.")
    if cluster_ids_grouped.shape != utility_grouped.shape:
        raise ValueError("cluster_ids_grouped must match grouped utilities.")
    if candidate_correctness_grouped is not None and candidate_correctness_grouped.shape != utility_grouped.shape:
        raise ValueError("candidate_correctness_grouped must match grouped utilities.")
    if candidate_lengths_grouped is not None and candidate_lengths_grouped.shape != utility_grouped.shape:
        raise ValueError("candidate_lengths_grouped must match grouped utilities.")
    if candidate_formatted_grouped is not None and candidate_formatted_grouped.shape != utility_grouped.shape:
        raise ValueError("candidate_formatted_grouped must match grouped utilities.")
    if valid_row_mask_grouped is None:
        valid_mask = torch.ones_like(utility_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != utility_grouped.shape:
            raise ValueError("valid_row_mask_grouped must match the grouped utility shape.")
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
        if budget_grouped.dim() != 1 or int(
            budget_grouped.numel()
        ) != int(utility_grouped.size(0)):
            raise ValueError(
                "budget_grouped must provide one value per prompt group."
            )
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
    weights = base_weights.clone()
    mode_count_grouped = torch.zeros((num_prompts,), device=utility_grouped.device, dtype=utility_grouped.dtype)
    eligible_mode_count_grouped = torch.zeros_like(mode_count_grouped)
    eligible_mode_frac_grouped = torch.zeros_like(mode_count_grouped)
    distinct_correct_mode_count_grouped = torch.zeros_like(mode_count_grouped)
    distinct_correct_mode_frac_grouped = torch.zeros_like(mode_count_grouped)
    best_score_grouped = torch.zeros_like(mode_count_grouped)
    second_score_grouped = torch.zeros_like(mode_count_grouped)
    competitive_gap_grouped = torch.zeros_like(mode_count_grouped)
    explore_budget_grouped = torch.zeros_like(mode_count_grouped)
    explore_budget_saturated_grouped = torch.zeros_like(mode_count_grouped)
    explore_applied_group_mask = torch.zeros((num_prompts,), device=utility_grouped.device, dtype=torch.bool)
    verified_bonus_applied_group_mask = torch.zeros_like(explore_applied_group_mask)
    prompt_selected_group_mask = torch.zeros_like(explore_applied_group_mask)
    prompt_rejected_low_opp_group_mask = torch.zeros_like(explore_applied_group_mask)
    prompt_rejected_nonpositive_group_mask = torch.zeros_like(explore_applied_group_mask)
    prompt_rejected_len_guard_group_mask = torch.zeros_like(explore_applied_group_mask)
    prompt_rejected_format_guard_group_mask = torch.zeros_like(explore_applied_group_mask)
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
        base_weights * torch.where(valid_mask, utility_grouped, torch.zeros_like(utility_grouped))
    ).sum(dim=1)
    expected_utility_explore_target_grouped = expected_utility_q_grouped.clone()
    expected_utility_final_w_grouped = expected_utility_q_grouped.clone()
    expected_len_q_grouped = (
        base_weights * torch.where(valid_mask, candidate_lengths, torch.zeros_like(candidate_lengths))
    ).sum(dim=1)
    expected_len_explore_target_grouped = expected_len_q_grouped.clone()
    expected_len_final_w_grouped = expected_len_q_grouped.clone()
    expected_format_q_grouped = (
        base_weights
        * torch.where(valid_mask, candidate_formatted, torch.zeros_like(candidate_formatted))
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
                candidate_correctness[p, semantic_idx][mask].max().to(utility_grouped.dtype)
            )

        cluster_scores_t = torch.stack(cluster_scores).to(dtype=utility_grouped.dtype)
        cluster_correctness_t = torch.stack(cluster_correctness).to(
            dtype=utility_grouped.dtype
        )
        sorted_scores, sorted_idx = torch.sort(cluster_scores_t, descending=True)
        best_score_grouped[p] = sorted_scores[0]
        second_score_grouped[p] = sorted_scores[1] if num_clusters > 1 else sorted_scores[0]
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
            explore_budget_saturated_grouped[p] = (
                alpha_raw >= (1.0 - 1e-8)
            )
        if float(alpha_raw.item()) < safe_select_min_alpha_frac:
            prompt_rejected_low_opp_group_mask[p] = True
            continue
        if positive_only and float(best_score_grouped[p].item()) <= 0.0:
            prompt_rejected_nonpositive_group_mask[p] = True
            continue

        eligible_mask = cluster_scores_t >= (cluster_scores_t.max() - safe_mode_gap)
        if positive_only:
            eligible_mask = eligible_mask & (cluster_scores_t > 0.0)
        if safe_mode_top_k < num_clusters:
            topk_mask = torch.zeros_like(eligible_mask, dtype=torch.bool)
            topk_mask[sorted_idx[:safe_mode_top_k]] = True
            eligible_mask = eligible_mask & topk_mask
        eligible_count = int(eligible_mask.to(torch.int64).sum().item())
        eligible_mode_count_grouped[p] = float(eligible_count)
        eligible_mode_frac_grouped[p] = float(eligible_count) / float(max(num_clusters, 1))
        if eligible_count < 2:
            if positive_only and int((cluster_scores_t > 0.0).to(torch.int64).sum().item()) < 2:
                prompt_rejected_nonpositive_group_mask[p] = True
            continue

        if float(alpha_p.item()) <= 0.0:
            prompt_rejected_low_opp_group_mask[p] = True
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

        semantic_base_mass = base_weights[p, semantic_idx].sum()
        if float(semantic_base_mass.item()) <= 1e-12:
            continue
        semantic_explore_target = semantic_base_mass * semantic_group_weights
        semantic_q = base_weights[p, semantic_idx]
        semantic_explore_full = base_weights[p].clone()
        semantic_explore_full[semantic_idx] = semantic_explore_target
        expected_utility_explore_target_grouped[p] = (
            semantic_explore_full
            * torch.where(valid_mask[p], utility_grouped[p], torch.zeros_like(utility_grouped[p]))
        ).sum()
        expected_len_explore_target_grouped[p] = (
            semantic_explore_full
            * torch.where(valid_mask[p], candidate_lengths[p], torch.zeros_like(candidate_lengths[p]))
        ).sum()
        expected_format_explore_target_grouped[p] = (
            semantic_explore_full
            * torch.where(valid_mask[p], candidate_formatted[p], torch.zeros_like(candidate_formatted[p]))
        ).sum()
        if (
            float(expected_len_explore_target_grouped[p].item())
            > float(expected_len_q_grouped[p].item()) + safe_max_expected_len_delta + 1e-6
        ):
            prompt_rejected_len_guard_group_mask[p] = True
            continue
        if (
            float(expected_format_explore_target_grouped[p].item()) + safe_max_expected_format_drop + 1e-6
            < float(expected_format_q_grouped[p].item())
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
        semantic_bonus_target = None
        if (
            safe_verified_bonus_coef > 0.0
            and right_count >= safe_verified_distinct_min_modes
            and float(alpha_p.item()) > 0.0
        ):
            right_scores = cluster_scores_t[right_mode_mask]
            right_cluster_mass = torch.softmax(right_scores / safe_mode_tau, dim=0)
            semantic_bonus_group_weights = torch.zeros(
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
                semantic_bonus_group_weights[mask] = (
                    right_cluster_mass[right_mass_idx].to(utility_grouped.dtype) * within
                )
            semantic_bonus_target = semantic_base_mass * semantic_bonus_group_weights
            beta_p = torch.clamp(
                alpha_p * safe_verified_bonus_coef,
                min=0.0,
                max=max(1.0 - float(alpha_p.item()), 0.0),
            )
            if float(beta_p.item()) > 0.0:
                candidate_semantic_weights = (
                    (1.0 - alpha_p - beta_p) * semantic_q
                    + alpha_p * semantic_explore_target
                    + beta_p * semantic_bonus_target
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
                elif (
                    float(candidate_expected_format.item())
                    + safe_max_expected_format_drop
                    + 1e-6
                    < float(expected_format_q_grouped[p].item())
                ):
                    prompt_rejected_verified_bonus_format_guard_group_mask[p] = True
                    beta_p = torch.zeros_like(alpha_p)
                else:
                    verified_bonus_applied_group_mask[p] = True

        verified_bonus_grouped[p] = beta_p
        explore_budget_grouped[p] = alpha_p + beta_p
        if semantic_bonus_target is not None and float(beta_p.item()) > 0.0:
            weights[p, semantic_idx] = (
                (1.0 - alpha_p - beta_p) * semantic_q
                + alpha_p * semantic_explore_target
                + beta_p * semantic_bonus_target
            )
        else:
            weights[p, semantic_idx] = (
                (1.0 - alpha_p) * semantic_q + alpha_p * semantic_explore_target
            )
        explore_applied_group_mask[p] = True
        moved_mass_l1_grouped[p] = 0.5 * torch.abs(weights[p] - base_weights[p]).sum()
        expected_utility_final_w_grouped[p] = (
            weights[p]
            * torch.where(valid_mask[p], utility_grouped[p], torch.zeros_like(utility_grouped[p]))
        ).sum()
        expected_len_final_w_grouped[p] = (
            weights[p]
            * torch.where(valid_mask[p], candidate_lengths[p], torch.zeros_like(candidate_lengths[p]))
        ).sum()
        expected_format_final_w_grouped[p] = (
            weights[p]
            * torch.where(valid_mask[p], candidate_formatted[p], torch.zeros_like(candidate_formatted[p]))
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
    weights = torch.where(row_sums > 0, weights / row_sums.clamp(min=1e-12), uniform_weights)
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

def compute_listwise_weights_from_utilities(
    *,
    utility_grouped: torch.Tensor,
    ref_seq_logps_grouped: torch.Tensor,
    tau: float,
    candidate_kl_coef: float,
    valid_row_mask_grouped: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the closed-form candidate weights for a per-candidate utility lift."""

    if utility_grouped.shape != ref_seq_logps_grouped.shape:
        raise ValueError(
            "utility_grouped and ref_seq_logps_grouped must have matching shapes."
        )
    if valid_row_mask_grouped is None:
        valid_mask = torch.ones_like(utility_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != utility_grouped.shape:
            raise ValueError(
                "valid_row_mask_grouped must match the grouped utility shape."
            )
        valid_mask = valid_row_mask_grouped.to(torch.bool)

    safe_tau = max(coerce_non_negative_float(tau, default=0.0), 1e-8)
    safe_candidate_kl = coerce_non_negative_float(candidate_kl_coef, default=0.0)
    safe_temperature = max(safe_tau + safe_candidate_kl, 1e-8)
    neg_inf = torch.full_like(utility_grouped, torch.finfo(utility_grouped.dtype).min)
    log_terms = torch.where(
        valid_mask,
        utility_grouped / safe_temperature,
        neg_inf,
    )
    if safe_candidate_kl > 0.0:
        ref_term = (safe_candidate_kl * ref_seq_logps_grouped) / safe_temperature
        log_terms = torch.where(valid_mask, log_terms + ref_term, neg_inf)

    has_valid = valid_mask.any(dim=1, keepdim=True)
    weights_grouped = torch.softmax(log_terms, dim=1)
    weights_grouped = torch.where(valid_mask, weights_grouped, torch.zeros_like(weights_grouped))
    uniform_weights = torch.where(
        valid_mask,
        1.0
        / valid_mask.to(dtype=utility_grouped.dtype)
        .sum(dim=1, keepdim=True)
        .clamp(min=1.0),
        torch.zeros_like(weights_grouped),
    )
    return torch.where(has_valid, weights_grouped, uniform_weights)




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
    neutral_eps: float = 1e-8,
    neutral_projection_coef: float = 0.0,
) -> DrXTargetBundle:
    """Build grouped DrX objects with optional competitive-mode semantic remixing.

    When ``cluster_ids_grouped`` is provided, the baseline DrX candidate posterior
    remains primary and semantic structure only reallocates a bounded residual
    budget over competitive semantic modes.
    """

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
        w_star_grouped, semantic_diagnostics = compute_semantic_cluster_weights_from_utilities(
            utility_grouped=utility_grouped,
            ref_seq_logps_grouped=ref_seq_logps_grouped,
            cluster_ids_grouped=cluster_ids_grouped,
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

    proj_scale = (
        torch.full(
            (utility_grouped.size(0),),
            float(max(neutral_projection_coef, 0.0)),
            device=utility_grouped.device,
            dtype=utility_grouped.dtype,
        )
        * projection_group_mask.to(dtype=utility_grouped.dtype)
    )

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
    policy_log_probs_grouped = masked_group_log_softmax(
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


def compute_listwise_centered_advantages(
    *,
    weights_grouped: torch.Tensor,
    behavior_seq_logps_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return prompt-local centered listwise advantages against the behavior policy.

    The returned advantages are the frozen-bridge term ``w - p_mu`` over the sampled
    candidate set for each prompt group, where ``w`` is the closed-form paper target
    and ``p_mu`` is the behavior/rollout candidate distribution induced by the same
    sampled completions.
    """

    if weights_grouped.shape != behavior_seq_logps_grouped.shape:
        raise ValueError(
            "weights_grouped and behavior_seq_logps_grouped must have matching shapes."
        )
    if valid_row_mask_grouped is None:
        valid_group_mask = torch.ones_like(weights_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != weights_grouped.shape:
            raise ValueError(
                "valid_row_mask_grouped must match the grouped weight shape."
            )
        valid_group_mask = valid_row_mask_grouped.to(torch.bool)

    behavior_log_probs_grouped = masked_group_log_softmax(
        behavior_seq_logps_grouped,
        valid_group_mask,
    )
    behavior_probs_grouped = torch.where(
        valid_group_mask,
        torch.exp(behavior_log_probs_grouped),
        torch.zeros_like(behavior_log_probs_grouped),
    )
    target_mass_grouped = (
        weights_grouped * valid_group_mask.to(weights_grouped.dtype)
    ).sum(dim=1, keepdim=True).to(weights_grouped.dtype)
    advantages = weights_grouped - (target_mass_grouped * behavior_probs_grouped)
    return torch.where(valid_group_mask, advantages, torch.zeros_like(advantages))


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
            raise ValueError("valid_row_mask_grouped must match the grouped weight shape.")
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
                -1, int(weights_grouped.size(1))
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
    """Return exact d(loss)/d(seq_logp) coefficients for sequence-level PPO clip.

    This implements the per-prompt surrogate

        -sum_i min(r_i a_i, clip(r_i) a_i)

    where ``r_i = exp(s_i^pi - s_i^mu)`` and ``a_i`` is a frozen per-candidate
    scalar advantage, typically the centered listwise bridge term.
    """

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

    safe_clip_low = coerce_non_negative_float(clip_low, default=0.0)
    safe_clip_high = coerce_non_negative_float(clip_high, default=0.0)
    active_scale = (
        active_group_mask.to(
            device=policy_seq_logps_grouped.device,
            dtype=policy_seq_logps_grouped.dtype,
        ).unsqueeze(1)
        / float(active_count)
    )
    log_seq_ratio = (
        policy_seq_logps_grouped - behavior_seq_logps_grouped
    ).clamp(-40.0, 40.0)
    seq_ratio = torch.exp(log_seq_ratio).to(policy_seq_logps_grouped.dtype)
    row_advantages_grouped = row_advantages_grouped.to(
        device=policy_seq_logps_grouped.device,
        dtype=policy_seq_logps_grouped.dtype,
    )
    clipped_region = ((seq_ratio > 1.0 + safe_clip_high) & (row_advantages_grouped > 0.0)) | (
        (seq_ratio < 1.0 - safe_clip_low) & (row_advantages_grouped < 0.0)
    )
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
    if valid_row_mask_grouped is not None and valid_row_mask_grouped.shape != weights_grouped.shape:
        raise ValueError("valid_row_mask_grouped must match the grouped weight shape.")
    if clip_row_mask_grouped is not None and clip_row_mask_grouped.shape != weights_grouped.shape:
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
        raise ValueError("normalizer_active_group_count must be positive when active groups exist.")

    active_scale = (
        active_group_mask.to(
            device=policy_seq_logps_grouped.device,
            dtype=policy_seq_logps_grouped.dtype,
        ).unsqueeze(1)
        / float(active_count)
    )
    policy_log_probs_grouped = masked_group_log_softmax(
        policy_seq_logps_grouped,
        valid_group_mask,
    )
    policy_probs_grouped = torch.where(
        valid_group_mask,
        torch.exp(policy_log_probs_grouped),
        torch.zeros_like(policy_log_probs_grouped),
    )
    target_mass_grouped = (
        weights_grouped * valid_group_mask.to(weights_grouped.dtype)
    ).sum(dim=1, keepdim=True).to(
        policy_seq_logps_grouped.dtype
    )
    coeffs = (
        target_mass_grouped * policy_probs_grouped - weights_grouped
    ) * active_scale
    coeffs = torch.where(valid_group_mask, coeffs, torch.zeros_like(coeffs))

    safe_clip_coef = coerce_non_negative_float(clip_coef, default=0.0)
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
        log_seq_ratio = (
            policy_seq_logps_grouped - behavior_seq_logps_grouped
        ).clamp(-40.0, 40.0)
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
    if row_advantages.dim() != 1 or int(row_advantages.numel()) != int(new_logps.size(0)):
        raise ValueError("row_advantages must provide one value per row.")

    safe_clip_low = coerce_non_negative_float(clip_low, default=0.0)
    safe_clip_high = coerce_non_negative_float(clip_high, default=0.0)

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


def gather_selected_logps_chunked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """Return selected-token log-probs without materializing a full log-softmax."""

    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits and labels must agree on batch/sequence shape")

    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    safe_chunk = min(token_chunk_size, max(int(shifted_logits.size(1)), 1))
    selected_logps = []
    for start in range(0, int(shifted_logits.size(1)), safe_chunk):
        stop = min(start + safe_chunk, int(shifted_logits.size(1)))
        chunk_logits = shifted_logits[:, start:stop, :]
        chunk_labels = shifted_labels[:, start:stop]
        chunk_masks = response_masks[:, start:stop].to(torch.bool)
        chunk_labels = chunk_labels.masked_fill(~chunk_masks, 0)
        chunk_logits_fp32 = (
            chunk_logits if chunk_logits.dtype == torch.float32 else chunk_logits.float()
        )
        chunk_logsumexp = torch.logsumexp(chunk_logits_fp32, dim=-1)
        target_logits = torch.gather(
            chunk_logits_fp32,
            dim=2,
            index=chunk_labels.unsqueeze(2),
        ).squeeze(2)
        chunk_logps = target_logits - chunk_logsumexp
        selected_logps.append(
            torch.where(
                chunk_masks,
                chunk_logps,
                torch.zeros_like(chunk_logps),
            )
        )
    return torch.cat(selected_logps, dim=1)
