"""Prompt-group helpers for listwise MaxEnt on top of OAT PPO/Dr.GRPO."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Iterator, Optional, Sequence

import torch

from .controllers import (
    ListwiseControllerState as ListwiseControllerState,
    clamp_listwise_tau as clamp_listwise_tau,
    compute_learnable_tau_loss as compute_learnable_tau_loss,
    maybe_update_listwise_beta as maybe_update_listwise_beta,
    maybe_update_listwise_tau as maybe_update_listwise_tau,
    resolve_listwise_target_entropy as resolve_listwise_target_entropy,
    update_listwise_tau_entropy_ema as update_listwise_tau_entropy_ema,
    update_listwise_tau_metric_ema as update_listwise_tau_metric_ema,
)
from .drx_targets import (
    DrXTargetBundle as DrXTargetBundle,
    apply_neutral_tiebreak_to_advantages as apply_neutral_tiebreak_to_advantages,
    build_drx_target_bundle as build_drx_target_bundle,
    compute_drx_group_masks as compute_drx_group_masks,
    compute_drx_projection_sequence_coefficients as compute_drx_projection_sequence_coefficients,
)
from .ppo_clip import (
    compute_listwise_clip_advantages as compute_listwise_clip_advantages,
    compute_listwise_sequence_coefficients as compute_listwise_sequence_coefficients,
    compute_sequence_clip_coefficients as compute_sequence_clip_coefficients,
    compute_token_level_clip_loss as compute_token_level_clip_loss,
)
from .semantic_utility import (
    SemanticDrxUtilityDiagnostics as SemanticDrxUtilityDiagnostics,
    compute_quality_centered_semantic_drx_utilities as compute_quality_centered_semantic_drx_utilities,
)
from .semantic_remix import (
    SemanticWeightDiagnostics as SemanticWeightDiagnostics,
    compute_anchor_relative_sequence_utilities as compute_anchor_relative_sequence_utilities,
    compute_anchor_relative_weights as compute_anchor_relative_weights,
    compute_semantic_cluster_weights_from_utilities as compute_semantic_cluster_weights_from_utilities,
)
from .scoring import (
    SanitizedTokenIdsResult as SanitizedTokenIdsResult,
    coerce_optional_int as coerce_optional_int,
    mask_invalid_logit_columns as mask_invalid_logit_columns,
    resolve_model_vocab_limit as resolve_model_vocab_limit,
    resolve_token_id_upper_bound as resolve_token_id_upper_bound,
    resolve_tokenizer_vocab_limit as resolve_tokenizer_vocab_limit,
    sanitize_scoring_token_ids as sanitize_scoring_token_ids,
)

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
_TIEBREAK_ANCHOR_ALIASES = {
    None: "hybrid",
    "": "hybrid",
    "default": "hybrid",
    "auto": "hybrid",
    "hybrid": "hybrid",
    "mix": "hybrid",
    "behavior": "behavior",
    "beh": "behavior",
    "rollout": "behavior",
    "reference": "reference",
    "ref": "reference",
}
_SEMANTIC_REMIX_MODE_ALIASES = {
    None: "competitive",
    "": "competitive",
    "default": "competitive",
    "competitive": "competitive",
    "competitive_modes": "competitive",
    "anchor": "anchor_rare",
    "anchor_rare": "anchor_rare",
    "anchor_rare_full": "anchor_rare_full",
    "anchor_full": "anchor_rare_full",
    "anchor_native": "anchor_rare_full",
    "anchor_weighted": "anchor_rare",
    "listwise_native": "anchor_rare_full",
}
_SEMANTIC_CLUSTER_METHOD_ALIASES = {
    None: "default",
    "": "default",
    "default": "default",
    "auto": "default",
    "greedy": "greedy",
    "threshold": "greedy",
    "connected": "connected_components",
    "connected_component": "connected_components",
    "connected_components": "connected_components",
    "cc": "connected_components",
    "spectral": "spectral",
    "sse": "spectral",
}
_EXACT_DRX_UTILITY_MODE_ALIASES = {
    None: "drgrpo_probe",
    "": "drgrpo_probe",
    "default": "drgrpo_probe",
    "probe": "drgrpo_probe",
    "drgrpo_probe": "drgrpo_probe",
    "anchor": "anchor_relative",
    "anchor_relative": "anchor_relative",
    "anchor_ref": "anchor_relative",
    "anchor_reference": "anchor_relative",
    "anchor_reference_logprob": "anchor_relative",
}
_SIGNATURE_JACCARD_MERGE_THRESHOLD = 0.75
_TRACE_EMBEDDING_COSINE_MERGE_THRESHOLD = 0.9
_SEMANTIC_TIEBREAK_CORRECTNESS_THRESHOLD = 0.999
_SEMANTIC_SPECTRAL_EIGENGAP_MIN = 0.05


@dataclass
class SemanticClusterBundle:
    """Semantic cluster assignments and masses for one prompt-major minibatch."""

    cluster_ids_grouped: torch.Tensor
    num_clusters_per_group: torch.Tensor
    semantic_entropy_grouped: torch.Tensor
    semantic_valid_row_mask_grouped: torch.Tensor


@dataclass
class SemanticTiebreakDiagnostics:
    """Prompt-group diagnostics for semantic surprisal tiebreak shaping."""

    prompt_alpha_grouped: torch.Tensor
    correct_anchor_mass_grouped: torch.Tensor
    anchor_probs_grouped: torch.Tensor
    mode_surprisal_grouped: torch.Tensor
    semantic_valid_row_mask_grouped: torch.Tensor
    correct_row_mask_grouped: torch.Tensor
    distinct_correct_mode_count_grouped: torch.Tensor
    mode_count_grouped: torch.Tensor


def normalize_oat_objective(value: object) -> str:
    """Return the canonical OAT-side objective label."""

    if value is None:
        candidate = None
    else:
        candidate = str(value).strip().lower()
    normalized = _OBJECTIVE_ALIASES.get(candidate, candidate)
    if normalized not in {
        "grpo",
        "maxent_listwise",
    }:
        raise ValueError("objective must be one of: grpo, maxent_listwise")
    return normalized


def normalize_tiebreak_anchor(value: object) -> str:
    """Return the canonical semantic tiebreak anchor label."""

    if value is None:
        candidate = None
    else:
        candidate = str(value).strip().lower()
    normalized = _TIEBREAK_ANCHOR_ALIASES.get(candidate, candidate)
    if normalized not in {"hybrid", "behavior", "reference"}:
        raise ValueError(
            "maxent_tiebreak_anchor must be one of: hybrid, behavior, reference"
        )
    return normalized


def normalize_semantic_remix_mode(value: object) -> str:
    """Return the canonical listwise semantic-remix mode."""

    if value is None:
        candidate = None
    else:
        candidate = str(value).strip().lower()
    normalized = _SEMANTIC_REMIX_MODE_ALIASES.get(candidate, candidate)
    if normalized not in {"competitive", "anchor_rare", "anchor_rare_full"}:
        raise ValueError(
            "maxent_semantic_remix_mode must be one of: competitive, "
            "anchor_rare, anchor_rare_full"
        )
    return normalized


def normalize_semantic_cluster_method(value: object) -> str:
    """Return the canonical semantic clustering method label."""

    if value is None:
        candidate = None
    else:
        candidate = str(value).strip().lower()
    normalized = _SEMANTIC_CLUSTER_METHOD_ALIASES.get(candidate, candidate)
    if normalized not in {"default", "greedy", "connected_components", "spectral"}:
        raise ValueError(
            "maxent_semantic_cluster_method must be one of: default, greedy, "
            "connected_components, spectral"
        )
    return normalized


def normalize_exact_drx_utility_mode(value: object) -> str:
    """Return the canonical exact-DrX utility mode label."""

    if value is None:
        candidate = None
    else:
        candidate = str(value).strip().lower()
    normalized = _EXACT_DRX_UTILITY_MODE_ALIASES.get(candidate, candidate)
    if normalized not in {"drgrpo_probe", "anchor_relative"}:
        raise ValueError(
            "maxent_exact_drx_utility_mode must be one of: drgrpo_probe, "
            "anchor_relative"
        )
    return normalized


def select_semantic_tiebreak_anchor_logits(
    *,
    behavior_seq_logps_grouped: torch.Tensor,
    reference_seq_logps_grouped: torch.Tensor | None,
    anchor: str,
    beta: float,
    reference_available: bool,
) -> tuple[torch.Tensor, str]:
    """Select the prompt-group anchor logits used for semantic-mode surprisal."""

    normalized_anchor = normalize_tiebreak_anchor(anchor)
    if normalized_anchor == "behavior":
        return behavior_seq_logps_grouped, "behavior"
    if normalized_anchor == "reference":
        if not reference_available or reference_seq_logps_grouped is None:
            raise ValueError(
                "reference semantic tiebreak anchor requested but reference logits are unavailable."
            )
        return reference_seq_logps_grouped, "reference"
    if (
        reference_available
        and reference_seq_logps_grouped is not None
        and float(beta) > 0.0
    ):
        return reference_seq_logps_grouped, "reference"
    return behavior_seq_logps_grouped, "behavior"


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
        entropy_advantage_samples.extend(
            [float(val) - baseline for val in weight_group]
        )
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


def iter_fixed_row_chunks(
    total_rows: int, *, chunk_size: int
) -> Iterator[tuple[int, int]]:
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
    shifted = torch.where(
        valid_mask, masked_values - max_vals, torch.zeros_like(masked_values)
    )
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
    grouped_indices = torch.arange(total_rows, device=device).reshape(
        num_prompts, group_size
    )
    if prompt_permutation is None:
        prompt_order = torch.randperm(num_prompts, device=device)
    else:
        prompt_order = torch.as_tensor(
            list(prompt_permutation),
            device=device,
            dtype=torch.long,
        )
        if prompt_order.numel() != num_prompts:
            raise ValueError(
                "prompt_permutation must cover every prompt group exactly once."
            )
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
        raise ValueError(
            "q_grouped and ref_seq_logps_grouped must have matching shapes."
        )
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


def _safe_logsumexp(
    values: torch.Tensor, *, dim: int, keepdim: bool = False
) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim, keepdim=keepdim)


def _split_reasoning_signature_states(signature: str | None) -> frozenset[str]:
    """Return the normalized symbolic states encoded in ``signature``."""

    if signature is None:
        return frozenset()
    states = [
        piece.strip()
        for piece in str(signature).split("||")
        if piece is not None and str(piece).strip()
    ]
    return frozenset(states)


def _reasoning_signature_jaccard(
    left_states: frozenset[str],
    right_states: frozenset[str],
) -> float:
    """Return Jaccard overlap between two signature-state sets."""

    if not left_states or not right_states:
        return 0.0
    union_size = len(left_states | right_states)
    if union_size <= 0:
        return 0.0
    return float(len(left_states & right_states)) / float(union_size)


def _coerce_similarity_threshold(
    value: float,
    *,
    default: float,
    lower: float = 0.0,
    upper: float | None = 1.0,
) -> float:
    """Clamp a similarity threshold into a closed interval."""

    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = float(default)
    if not math.isfinite(threshold):
        threshold = float(default)
    threshold = max(threshold, lower)
    if upper is not None:
        threshold = min(threshold, upper)
    return threshold


def _reasoning_trace_embedding_cosine(
    left_embedding: torch.Tensor | None,
    right_embedding: torch.Tensor | None,
) -> float:
    """Return cosine similarity between two optional trace embeddings."""

    if left_embedding is None or right_embedding is None:
        return -1.0
    left = left_embedding.to(torch.float32)
    right = right_embedding.to(torch.float32)
    denom = left.norm() * right.norm()
    if float(denom.item()) <= 1e-12:
        return -1.0
    return float(torch.clamp((left * right).sum() / denom, -1.0, 1.0).item())


def _row_trace_embedding(
    *,
    prompt_idx: int,
    row_idx: int,
    reasoning_trace_embeddings_grouped: torch.Tensor | None,
    reasoning_trace_valid_row_mask_grouped: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return the row trace embedding when both tensors mark it valid."""

    if (
        reasoning_trace_embeddings_grouped is None
        or reasoning_trace_valid_row_mask_grouped is None
        or not bool(reasoning_trace_valid_row_mask_grouped[prompt_idx, row_idx].item())
    ):
        return None
    return reasoning_trace_embeddings_grouped[prompt_idx, row_idx]


def _rows_semantically_similar(
    *,
    left_signature_states: frozenset[str] | None,
    right_signature_states: frozenset[str] | None,
    signature_jaccard_merge_threshold: float,
    left_trace_embedding: torch.Tensor | None,
    right_trace_embedding: torch.Tensor | None,
    embedding_cosine_merge_threshold: float,
) -> tuple[bool, float]:
    """Return whether a same-answer row pair should share a semantic cluster.

    Rows merge when either the symbolic signature overlap or the dense trace
    embedding cosine clears its configured threshold.
    """

    best_similarity = -1.0
    if left_signature_states and right_signature_states:
        signature_similarity = _reasoning_signature_jaccard(
            left_signature_states,
            right_signature_states,
        )
        best_similarity = max(best_similarity, signature_similarity)
        if signature_similarity >= signature_jaccard_merge_threshold:
            return True, signature_similarity
    embedding_similarity = _reasoning_trace_embedding_cosine(
        left_trace_embedding,
        right_trace_embedding,
    )
    best_similarity = max(best_similarity, embedding_similarity)
    if embedding_similarity >= embedding_cosine_merge_threshold:
        return True, embedding_similarity
    return False, best_similarity


def _row_semantic_similarity(
    *,
    left_signature_states: frozenset[str] | None,
    right_signature_states: frozenset[str] | None,
    left_trace_embedding: torch.Tensor | None,
    right_trace_embedding: torch.Tensor | None,
) -> float:
    """Return a soft semantic affinity score in ``[0, 1]`` for a same-answer pair."""

    best_similarity = 0.0
    if left_signature_states and right_signature_states:
        best_similarity = max(
            best_similarity,
            _reasoning_signature_jaccard(left_signature_states, right_signature_states),
        )
    embedding_similarity = _reasoning_trace_embedding_cosine(
        left_trace_embedding,
        right_trace_embedding,
    )
    if embedding_similarity >= 0.0:
        best_similarity = max(best_similarity, embedding_similarity)
    return min(max(best_similarity, 0.0), 1.0)


def _normalize_cluster_labels(labels: list[int]) -> list[int]:
    """Map arbitrary integer labels to ``0..K-1`` in first-occurrence order."""

    label_map: dict[int, int] = {}
    normalized: list[int] = []
    next_label = 0
    for label in labels:
        mapped = label_map.get(label)
        if mapped is None:
            mapped = next_label
            label_map[label] = mapped
            next_label += 1
        normalized.append(mapped)
    return normalized


def _tiny_kmeans(
    points: torch.Tensor,
    *,
    num_clusters: int,
    max_iters: int = 16,
) -> torch.Tensor:
    """Cluster a small dense point set with deterministic k-means."""

    if points.dim() != 2:
        raise ValueError("points must have shape [rows, dim].")
    num_points = int(points.size(0))
    if num_points <= 0:
        raise ValueError("points must contain at least one row.")
    k = max(1, min(int(num_clusters), num_points))
    if k == 1:
        return torch.zeros((num_points,), device=points.device, dtype=torch.long)

    centroids = [points[0]]
    min_sq_dist = ((points - centroids[0]) ** 2).sum(dim=1)
    while len(centroids) < k:
        next_idx = int(torch.argmax(min_sq_dist).item())
        centroids.append(points[next_idx])
        candidate_sq_dist = ((points - points[next_idx]) ** 2).sum(dim=1)
        min_sq_dist = torch.minimum(min_sq_dist, candidate_sq_dist)
    centroid_t = torch.stack(centroids).to(points.dtype)

    labels = torch.zeros((num_points,), device=points.device, dtype=torch.long)
    for _ in range(max(int(max_iters), 1)):
        sq_dist = torch.cdist(points, centroid_t).pow(2)
        new_labels = torch.argmin(sq_dist, dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_idx in range(k):
            member_mask = labels == cluster_idx
            if bool(member_mask.any().item()):
                centroid_t[cluster_idx] = points[member_mask].mean(dim=0)
                continue
            farthest_idx = int(torch.argmax(sq_dist.min(dim=1).values).item())
            labels[farthest_idx] = cluster_idx
            member_mask = labels == cluster_idx
            centroid_t[cluster_idx] = points[member_mask].mean(dim=0)

    return labels


def _spectral_cluster_labels_from_affinity(
    affinity: torch.Tensor,
    *,
    max_num_clusters: int,
    eigengap_min: float,
) -> list[int]:
    """Return spectral-cluster labels for a small symmetric affinity matrix."""

    if affinity.dim() != 2 or affinity.size(0) != affinity.size(1):
        raise ValueError("affinity must have shape [rows, rows].")
    num_rows = int(affinity.size(0))
    if num_rows <= 1:
        return [0] * num_rows

    safe_max_num_clusters = max(1, min(int(max_num_clusters), num_rows))
    if safe_max_num_clusters <= 1:
        return [0] * num_rows

    safe_eigengap_min = coerce_non_negative_float(
        eigengap_min,
        default=_SEMANTIC_SPECTRAL_EIGENGAP_MIN,
    )
    affinity = 0.5 * (affinity + affinity.transpose(0, 1))
    affinity = affinity.clamp(min=0.0, max=1.0)
    affinity.fill_diagonal_(1.0)
    degree = affinity.sum(dim=1).clamp(min=1e-6)
    inv_sqrt_degree = degree.rsqrt()
    normalized_affinity = inv_sqrt_degree[:, None] * affinity * inv_sqrt_degree[None, :]
    eigvals, eigvecs = torch.linalg.eigh(normalized_affinity)
    eigvals_desc = torch.flip(eigvals, dims=[0])
    eigvecs_desc = torch.flip(eigvecs, dims=[1])
    candidate_count = min(safe_max_num_clusters, num_rows)
    if candidate_count <= 1:
        return [0] * num_rows
    gaps = eigvals_desc[: candidate_count - 1] - eigvals_desc[1:candidate_count]
    if gaps.numel() <= 0:
        return [0] * num_rows
    best_gap, best_idx = torch.max(gaps, dim=0)
    num_clusters = 1
    if float(best_gap.item()) >= safe_eigengap_min:
        num_clusters = int(best_idx.item()) + 1
    num_clusters = max(1, min(num_clusters, candidate_count))
    if num_clusters <= 1:
        return [0] * num_rows

    embedding = eigvecs_desc[:, :num_clusters].to(torch.float32)
    embedding = embedding / embedding.norm(dim=1, keepdim=True).clamp(min=1e-12)
    labels = _tiny_kmeans(embedding, num_clusters=num_clusters)
    return _normalize_cluster_labels([int(label) for label in labels.tolist()])


def build_semantic_cluster_bundle(
    *,
    final_answer_keys_grouped: Sequence[Sequence[str | None]],
    valid_row_mask_grouped: torch.Tensor,
    reasoning_signature_keys_grouped: Sequence[Sequence[str | None]] | None = None,
    reasoning_trace_embeddings_grouped: torch.Tensor | None = None,
    reasoning_trace_valid_row_mask_grouped: torch.Tensor | None = None,
    signature_jaccard_merge_threshold: float = _SIGNATURE_JACCARD_MERGE_THRESHOLD,
    embedding_cosine_merge_threshold: float = _TRACE_EMBEDDING_COSINE_MERGE_THRESHOLD,
) -> SemanticClusterBundle:
    """Cluster candidates by final answer plus symbolic or dense trace similarity.

    The returned semantic entropy is the *normalized empirical cluster entropy*
    induced by the observed sample counts inside each prompt group. The coarse
    cluster key is always the normalized final answer string. When a compressed
    structural reasoning signature is available, rows with the same answer key
    are further split only when both their symbolic signatures and their dense
    reasoning-trace embeddings fail to show high similarity. Same-answer rows
    merge when either signature-state Jaccard clears
    ``signature_jaccard_merge_threshold`` or trace-embedding cosine clears
    ``embedding_cosine_merge_threshold``. Rows without any usable semantic
    signal fall back to an answer-only bucket only when the entire answer
    family lacks semantic signals; otherwise they stay semantic-invalid so
    extraction failures do not create free diversity credit.
    Setting ``embedding_cosine_merge_threshold`` above 1 disables the dense
    embedding signal entirely.

        H_sem_norm = H(counts / sum counts) / log K,

    with the convention H_sem_norm = 0 when K <= 1.
    """

    if valid_row_mask_grouped.dim() != 2:
        raise ValueError("valid_row_mask_grouped must have shape [prompts, group].")
    num_prompts, group_size = valid_row_mask_grouped.shape
    if len(final_answer_keys_grouped) != num_prompts:
        raise ValueError("final_answer_keys_grouped must match num_prompts.")
    if (
        reasoning_signature_keys_grouped is not None
        and len(reasoning_signature_keys_grouped) != num_prompts
    ):
        raise ValueError("reasoning_signature_keys_grouped must match num_prompts.")
    if reasoning_trace_embeddings_grouped is not None:
        if reasoning_trace_valid_row_mask_grouped is None:
            raise ValueError(
                "reasoning_trace_valid_row_mask_grouped is required when embeddings are provided."
            )
        if reasoning_trace_embeddings_grouped.dim() != 3:
            raise ValueError(
                "reasoning_trace_embeddings_grouped must have shape [prompts, group, dim]."
            )
        if tuple(reasoning_trace_embeddings_grouped.shape[:2]) != (
            num_prompts,
            group_size,
        ):
            raise ValueError(
                "reasoning_trace_embeddings_grouped must match the prompt-major minibatch shape."
            )
        if tuple(reasoning_trace_valid_row_mask_grouped.shape) != (
            num_prompts,
            group_size,
        ):
            raise ValueError(
                "reasoning_trace_valid_row_mask_grouped must match valid_row_mask_grouped."
            )
    elif reasoning_trace_valid_row_mask_grouped is not None:
        raise ValueError(
            "reasoning_trace_valid_row_mask_grouped requires reasoning_trace_embeddings_grouped."
        )

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
    safe_signature_merge_threshold = _coerce_similarity_threshold(
        signature_jaccard_merge_threshold,
        default=_SIGNATURE_JACCARD_MERGE_THRESHOLD,
    )
    safe_embedding_merge_threshold = _coerce_similarity_threshold(
        embedding_cosine_merge_threshold,
        default=_TRACE_EMBEDDING_COSINE_MERGE_THRESHOLD,
        upper=None,
    )

    for p in range(num_prompts):
        keys = list(final_answer_keys_grouped[p])
        if len(keys) != group_size:
            raise ValueError(
                "Each prompt must provide one final-answer key per candidate."
            )
        if reasoning_signature_keys_grouped is None:
            signatures = [None] * group_size
        else:
            signatures = list(reasoning_signature_keys_grouped[p])
            if len(signatures) != group_size:
                raise ValueError(
                    "Each prompt must provide one reasoning signature per candidate."
                )
        answer_has_semantic_signal: dict[str, bool] = {}
        for row_idx, (key, signature) in enumerate(zip(keys, signatures)):
            if key is None:
                continue
            trace_embedding = _row_trace_embedding(
                prompt_idx=p,
                row_idx=row_idx,
                reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
                reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
            )
            if signature is None and trace_embedding is None:
                continue
            answer_has_semantic_signal[key] = True
        answer_only_cluster_to_id: dict[str, int] = {}
        cluster_answer_keys: list[str] = []
        cluster_signature_state_sets: list[list[frozenset[str]]] = []
        cluster_trace_embeddings: list[list[torch.Tensor]] = []
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
            signature_states = (
                _split_reasoning_signature_states(signature)
                if signature is not None
                else None
            )
            trace_embedding = _row_trace_embedding(
                prompt_idx=p,
                row_idx=i,
                reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
                reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
            )
            if (
                signature_states is None
                and trace_embedding is None
                and answer_has_semantic_signal.get(key, False)
            ):
                # If this answer bucket already has a structural signature from
                # another row or a dense trace embedding, an unlabeled row should
                # keep its base token mass but should not create extra semantic
                # entropy by pretending to be a new reasoning category.
                continue
            assigned: int | None = None
            if signature_states is None and trace_embedding is None:
                assigned = answer_only_cluster_to_id.get(key)
                if assigned is None:
                    assigned = next_cluster
                    next_cluster += 1
                    answer_only_cluster_to_id[key] = assigned
                    cluster_answer_keys.append(key)
                    cluster_signature_state_sets.append([])
                    cluster_trace_embeddings.append([])
                    cluster_counts.append(1)
                else:
                    cluster_counts[assigned] = cluster_counts[assigned] + 1
            else:
                best_similarity = -1.0
                for cluster_idx, answer_key in enumerate(cluster_answer_keys):
                    if answer_key != key:
                        continue
                    cluster_match = False
                    cluster_similarity = -1.0
                    for existing_states in cluster_signature_state_sets[cluster_idx]:
                        similar, similarity = _rows_semantically_similar(
                            left_signature_states=signature_states,
                            right_signature_states=existing_states,
                            signature_jaccard_merge_threshold=safe_signature_merge_threshold,
                            left_trace_embedding=trace_embedding,
                            right_trace_embedding=None,
                            embedding_cosine_merge_threshold=safe_embedding_merge_threshold,
                        )
                        if similar:
                            cluster_match = True
                        cluster_similarity = max(cluster_similarity, similarity)
                    for existing_embedding in cluster_trace_embeddings[cluster_idx]:
                        similar, similarity = _rows_semantically_similar(
                            left_signature_states=signature_states,
                            right_signature_states=None,
                            signature_jaccard_merge_threshold=safe_signature_merge_threshold,
                            left_trace_embedding=trace_embedding,
                            right_trace_embedding=existing_embedding,
                            embedding_cosine_merge_threshold=safe_embedding_merge_threshold,
                        )
                        if similar:
                            cluster_match = True
                        cluster_similarity = max(cluster_similarity, similarity)
                    if cluster_match and cluster_similarity > best_similarity:
                        assigned = cluster_idx
                        best_similarity = cluster_similarity
                if assigned is None:
                    assigned = next_cluster
                    next_cluster += 1
                    cluster_answer_keys.append(key)
                    cluster_signature_state_sets.append(
                        [signature_states] if signature_states is not None else []
                    )
                    cluster_trace_embeddings.append(
                        [trace_embedding] if trace_embedding is not None else []
                    )
                    cluster_counts.append(1)
                else:
                    cluster_counts[assigned] = cluster_counts[assigned] + 1
                    if signature_states is not None:
                        cluster_signature_state_sets[assigned].append(signature_states)
                    if trace_embedding is not None:
                        cluster_trace_embeddings[assigned].append(trace_embedding)
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


def build_connected_component_semantic_cluster_bundle(
    *,
    final_answer_keys_grouped: Sequence[Sequence[str | None]],
    valid_row_mask_grouped: torch.Tensor,
    reasoning_signature_keys_grouped: Sequence[Sequence[str | None]] | None = None,
    reasoning_trace_embeddings_grouped: torch.Tensor | None = None,
    reasoning_trace_valid_row_mask_grouped: torch.Tensor | None = None,
    signature_jaccard_merge_threshold: float = _SIGNATURE_JACCARD_MERGE_THRESHOLD,
    embedding_cosine_merge_threshold: float = _TRACE_EMBEDDING_COSINE_MERGE_THRESHOLD,
) -> SemanticClusterBundle:
    """Cluster candidates by connected components over same-answer similarity graphs.

    Two rows connect iff they share the same normalized final answer and either
    their reasoning-signature Jaccard overlap clears
    ``signature_jaccard_merge_threshold`` or their trace-embedding cosine
    clears ``embedding_cosine_merge_threshold``. Semantic modes are the
    connected components of that graph.

    Conservative missing-signature handling matches the baseline semantic path:
    rows missing a normalized final answer stay semantic-invalid, and answer
    buckets with at least one usable semantic signal do not let signal-missing
    rows create extra diversity credit. When an answer bucket has no usable
    semantic signals at all, its valid rows fall back to one answer-only
    component.
    Setting ``embedding_cosine_merge_threshold`` above 1 disables the dense
    embedding signal entirely.
    """

    if valid_row_mask_grouped.dim() != 2:
        raise ValueError("valid_row_mask_grouped must have shape [prompts, group].")
    num_prompts, group_size = valid_row_mask_grouped.shape
    if len(final_answer_keys_grouped) != num_prompts:
        raise ValueError("final_answer_keys_grouped must match num_prompts.")
    if (
        reasoning_signature_keys_grouped is not None
        and len(reasoning_signature_keys_grouped) != num_prompts
    ):
        raise ValueError("reasoning_signature_keys_grouped must match num_prompts.")
    if reasoning_trace_embeddings_grouped is not None:
        if reasoning_trace_valid_row_mask_grouped is None:
            raise ValueError(
                "reasoning_trace_valid_row_mask_grouped is required when embeddings are provided."
            )
        if reasoning_trace_embeddings_grouped.dim() != 3:
            raise ValueError(
                "reasoning_trace_embeddings_grouped must have shape [prompts, group, dim]."
            )
        if tuple(reasoning_trace_embeddings_grouped.shape[:2]) != (
            num_prompts,
            group_size,
        ):
            raise ValueError(
                "reasoning_trace_embeddings_grouped must match the prompt-major minibatch shape."
            )
        if tuple(reasoning_trace_valid_row_mask_grouped.shape) != (
            num_prompts,
            group_size,
        ):
            raise ValueError(
                "reasoning_trace_valid_row_mask_grouped must match valid_row_mask_grouped."
            )
    elif reasoning_trace_valid_row_mask_grouped is not None:
        raise ValueError(
            "reasoning_trace_valid_row_mask_grouped requires reasoning_trace_embeddings_grouped."
        )

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
    semantic_valid_mask = torch.zeros_like(valid_row_mask_grouped, dtype=torch.bool)
    valid_mask = valid_row_mask_grouped.to(torch.bool)
    safe_signature_merge_threshold = _coerce_similarity_threshold(
        signature_jaccard_merge_threshold,
        default=_SIGNATURE_JACCARD_MERGE_THRESHOLD,
    )
    safe_embedding_merge_threshold = _coerce_similarity_threshold(
        embedding_cosine_merge_threshold,
        default=_TRACE_EMBEDDING_COSINE_MERGE_THRESHOLD,
        upper=None,
    )

    for p in range(num_prompts):
        keys = list(final_answer_keys_grouped[p])
        if len(keys) != group_size:
            raise ValueError(
                "Each prompt must provide one final-answer key per candidate."
            )
        if reasoning_signature_keys_grouped is None:
            signatures = [None] * group_size
        else:
            signatures = list(reasoning_signature_keys_grouped[p])
            if len(signatures) != group_size:
                raise ValueError(
                    "Each prompt must provide one reasoning signature per candidate."
                )

        answer_to_rows: dict[str, list[int]] = {}
        for i in range(group_size):
            if not bool(valid_mask[p, i].item()):
                continue
            key = keys[i]
            if key is None:
                continue
            answer_to_rows.setdefault(key, []).append(i)

        next_cluster = 0
        cluster_counts: list[int] = []
        for key, row_indices in answer_to_rows.items():
            del key
            signal_rows = [
                row_idx
                for row_idx in row_indices
                if (
                    signatures[row_idx] is not None
                    or _row_trace_embedding(
                        prompt_idx=p,
                        row_idx=row_idx,
                        reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
                        reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
                    )
                    is not None
                )
            ]
            if not signal_rows:
                assigned = next_cluster
                next_cluster += 1
                count = 0
                for row_idx in row_indices:
                    cluster_ids[p, row_idx] = assigned
                    semantic_valid_mask[p, row_idx] = True
                    count += 1
                cluster_counts.append(count)
                continue

            signature_state_sets = {
                row_idx: _split_reasoning_signature_states(signatures[row_idx])
                for row_idx in signal_rows
                if signatures[row_idx] is not None
            }
            trace_embeddings = {
                row_idx: _row_trace_embedding(
                    prompt_idx=p,
                    row_idx=row_idx,
                    reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
                    reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
                )
                for row_idx in signal_rows
            }
            visited: set[int] = set()
            for start_row in signal_rows:
                if start_row in visited:
                    continue
                component_rows: list[int] = []
                stack = [start_row]
                visited.add(start_row)
                while stack:
                    row_idx = stack.pop()
                    component_rows.append(row_idx)
                    row_states = signature_state_sets.get(row_idx)
                    row_embedding = trace_embeddings.get(row_idx)
                    for other_idx in signal_rows:
                        if other_idx in visited:
                            continue
                        similar, _ = _rows_semantically_similar(
                            left_signature_states=row_states,
                            right_signature_states=signature_state_sets.get(other_idx),
                            signature_jaccard_merge_threshold=safe_signature_merge_threshold,
                            left_trace_embedding=row_embedding,
                            right_trace_embedding=trace_embeddings.get(other_idx),
                            embedding_cosine_merge_threshold=safe_embedding_merge_threshold,
                        )
                        if similar:
                            visited.add(other_idx)
                            stack.append(other_idx)
                assigned = next_cluster
                next_cluster += 1
                for row_idx in component_rows:
                    cluster_ids[p, row_idx] = assigned
                    semantic_valid_mask[p, row_idx] = True
                cluster_counts.append(len(component_rows))

        num_clusters[p] = next_cluster
        if next_cluster > 1 and cluster_counts:
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


def build_spectral_semantic_cluster_bundle(
    *,
    final_answer_keys_grouped: Sequence[Sequence[str | None]],
    valid_row_mask_grouped: torch.Tensor,
    reasoning_signature_keys_grouped: Sequence[Sequence[str | None]] | None = None,
    reasoning_trace_embeddings_grouped: torch.Tensor | None = None,
    reasoning_trace_valid_row_mask_grouped: torch.Tensor | None = None,
    spectral_max_num_clusters: int = 0,
    spectral_eigengap_min: float = _SEMANTIC_SPECTRAL_EIGENGAP_MIN,
) -> SemanticClusterBundle:
    """Cluster same-answer rows with a small torch-only spectral routine.

    The coarse semantic gate remains the normalized final answer string.
    Spectral clustering runs only inside each answer family that has at least
    one usable semantic signal. Rows missing all semantic signals stay
    conservative when their answer family already has signal; otherwise they
    fall back to a single answer-only cluster for that family.
    """

    if valid_row_mask_grouped.dim() != 2:
        raise ValueError("valid_row_mask_grouped must have shape [prompts, group].")
    num_prompts, group_size = valid_row_mask_grouped.shape
    if len(final_answer_keys_grouped) != num_prompts:
        raise ValueError("final_answer_keys_grouped must match num_prompts.")
    if (
        reasoning_signature_keys_grouped is not None
        and len(reasoning_signature_keys_grouped) != num_prompts
    ):
        raise ValueError("reasoning_signature_keys_grouped must match num_prompts.")
    if reasoning_trace_embeddings_grouped is not None:
        if reasoning_trace_valid_row_mask_grouped is None:
            raise ValueError(
                "reasoning_trace_valid_row_mask_grouped is required when embeddings are provided."
            )
        if reasoning_trace_embeddings_grouped.dim() != 3:
            raise ValueError(
                "reasoning_trace_embeddings_grouped must have shape [prompts, group, dim]."
            )
        if tuple(reasoning_trace_embeddings_grouped.shape[:2]) != (
            num_prompts,
            group_size,
        ):
            raise ValueError(
                "reasoning_trace_embeddings_grouped must match the prompt-major minibatch shape."
            )
        if tuple(reasoning_trace_valid_row_mask_grouped.shape) != (
            num_prompts,
            group_size,
        ):
            raise ValueError(
                "reasoning_trace_valid_row_mask_grouped must match valid_row_mask_grouped."
            )
    elif reasoning_trace_valid_row_mask_grouped is not None:
        raise ValueError(
            "reasoning_trace_valid_row_mask_grouped requires reasoning_trace_embeddings_grouped."
        )

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
    semantic_valid_mask = torch.zeros_like(valid_row_mask_grouped, dtype=torch.bool)
    valid_mask = valid_row_mask_grouped.to(torch.bool)
    safe_spectral_max_num_clusters = max(int(spectral_max_num_clusters), 0)
    safe_spectral_eigengap_min = coerce_non_negative_float(
        spectral_eigengap_min,
        default=_SEMANTIC_SPECTRAL_EIGENGAP_MIN,
    )

    for p in range(num_prompts):
        keys = list(final_answer_keys_grouped[p])
        if len(keys) != group_size:
            raise ValueError(
                "Each prompt must provide one final-answer key per candidate."
            )
        if reasoning_signature_keys_grouped is None:
            signatures = [None] * group_size
        else:
            signatures = list(reasoning_signature_keys_grouped[p])
            if len(signatures) != group_size:
                raise ValueError(
                    "Each prompt must provide one reasoning signature per candidate."
                )

        answer_to_rows: dict[str, list[int]] = {}
        for i in range(group_size):
            if not bool(valid_mask[p, i].item()):
                continue
            key = keys[i]
            if key is None:
                continue
            answer_to_rows.setdefault(key, []).append(i)

        next_cluster = 0
        cluster_counts: list[int] = []
        for row_indices in answer_to_rows.values():
            signal_rows = [
                row_idx
                for row_idx in row_indices
                if (
                    signatures[row_idx] is not None
                    or _row_trace_embedding(
                        prompt_idx=p,
                        row_idx=row_idx,
                        reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
                        reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
                    )
                    is not None
                )
            ]
            if not signal_rows:
                assigned = next_cluster
                next_cluster += 1
                count = 0
                for row_idx in row_indices:
                    cluster_ids[p, row_idx] = assigned
                    semantic_valid_mask[p, row_idx] = True
                    count += 1
                cluster_counts.append(count)
                continue

            if len(signal_rows) == 1:
                assigned = next_cluster
                next_cluster += 1
                row_idx = signal_rows[0]
                cluster_ids[p, row_idx] = assigned
                semantic_valid_mask[p, row_idx] = True
                cluster_counts.append(1)
                continue

            signature_state_sets = {
                row_idx: _split_reasoning_signature_states(signatures[row_idx])
                for row_idx in signal_rows
                if signatures[row_idx] is not None
            }
            trace_embeddings = {
                row_idx: _row_trace_embedding(
                    prompt_idx=p,
                    row_idx=row_idx,
                    reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
                    reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
                )
                for row_idx in signal_rows
            }
            num_signal_rows = len(signal_rows)
            affinity = torch.zeros(
                (num_signal_rows, num_signal_rows),
                device=valid_row_mask_grouped.device,
                dtype=torch.float32,
            )
            affinity.fill_diagonal_(1.0)
            for left_pos, left_row_idx in enumerate(signal_rows):
                left_states = signature_state_sets.get(left_row_idx)
                left_embedding = trace_embeddings.get(left_row_idx)
                for right_pos in range(left_pos + 1, num_signal_rows):
                    right_row_idx = signal_rows[right_pos]
                    similarity = _row_semantic_similarity(
                        left_signature_states=left_states,
                        right_signature_states=signature_state_sets.get(right_row_idx),
                        left_trace_embedding=left_embedding,
                        right_trace_embedding=trace_embeddings.get(right_row_idx),
                    )
                    affinity[left_pos, right_pos] = similarity
                    affinity[right_pos, left_pos] = similarity
            max_num_clusters = (
                num_signal_rows
                if safe_spectral_max_num_clusters <= 0
                else min(safe_spectral_max_num_clusters, num_signal_rows)
            )
            labels = _spectral_cluster_labels_from_affinity(
                affinity,
                max_num_clusters=max_num_clusters,
                eigengap_min=safe_spectral_eigengap_min,
            )
            cluster_id_map: dict[int, int] = {}
            local_cluster_counts: dict[int, int] = {}
            for row_idx, label in zip(signal_rows, labels):
                assigned = cluster_id_map.get(label)
                if assigned is None:
                    assigned = next_cluster
                    cluster_id_map[label] = assigned
                    next_cluster += 1
                    local_cluster_counts[assigned] = 0
                cluster_ids[p, row_idx] = assigned
                semantic_valid_mask[p, row_idx] = True
                local_cluster_counts[assigned] = local_cluster_counts[assigned] + 1
            cluster_counts.extend(
                local_cluster_counts[cluster_id]
                for cluster_id in sorted(local_cluster_counts)
            )

        num_clusters[p] = next_cluster
        if next_cluster > 1 and cluster_counts:
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


def build_answer_family_semantic_cluster_bundle(
    *,
    final_answer_keys_grouped: Sequence[Sequence[str | None]],
    valid_row_mask_grouped: torch.Tensor,
) -> SemanticClusterBundle:
    """Cluster candidates by normalized final-answer family only."""

    if valid_row_mask_grouped.dim() != 2:
        raise ValueError("valid_row_mask_grouped must have shape [prompts, group].")
    num_prompts, group_size = valid_row_mask_grouped.shape
    if len(final_answer_keys_grouped) != num_prompts:
        raise ValueError("final_answer_keys_grouped must match num_prompts.")

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
    semantic_valid_mask = torch.zeros_like(valid_row_mask_grouped, dtype=torch.bool)
    valid_mask = valid_row_mask_grouped.to(torch.bool)

    for p in range(num_prompts):
        keys = list(final_answer_keys_grouped[p])
        if len(keys) != group_size:
            raise ValueError(
                "Each prompt must provide one final-answer key per candidate."
            )

        answer_to_cluster: dict[str, int] = {}
        cluster_counts: list[int] = []
        next_cluster = 0
        for i in range(group_size):
            if not bool(valid_mask[p, i].item()):
                continue
            key = keys[i]
            if key is None:
                continue
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            assigned = answer_to_cluster.get(normalized_key)
            if assigned is None:
                assigned = next_cluster
                next_cluster += 1
                answer_to_cluster[normalized_key] = assigned
                cluster_counts.append(0)
            cluster_ids[p, i] = assigned
            semantic_valid_mask[p, i] = True
            cluster_counts[assigned] += 1

        num_clusters[p] = next_cluster
        if next_cluster > 1 and cluster_counts:
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
            raise ValueError(
                "valid_row_mask_grouped must match candidate_probs_grouped."
            )
        valid_mask = valid_row_mask_grouped.to(torch.bool)

    probs_grouped = torch.where(
        valid_mask,
        candidate_probs_grouped.to(torch.float32),
        torch.zeros_like(candidate_probs_grouped, dtype=torch.float32),
    )
    probs_grouped = probs_grouped / probs_grouped.sum(dim=1, keepdim=True).clamp(
        min=1e-12
    )

    num_prompts = int(candidate_probs_grouped.size(0))
    entropy = torch.zeros(
        (num_prompts,), device=candidate_probs_grouped.device, dtype=torch.float32
    )
    cluster_count = torch.zeros(
        (num_prompts,), device=candidate_probs_grouped.device, dtype=torch.float32
    )
    safe_normalizer_group_size = (
        None if normalizer_group_size is None else max(int(normalizer_group_size), 0)
    )

    for p in range(num_prompts):
        valid_clusters = cluster_ids_grouped[p][
            valid_mask[p] & (cluster_ids_grouped[p] >= 0)
        ]
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


def compute_semantic_tiebreak_from_anchor_logits(
    *,
    anchor_logits_grouped: torch.Tensor,
    candidate_correctness_grouped: torch.Tensor,
    cluster_ids_grouped: torch.Tensor,
    bonus_alpha: float,
    bonus_clip_max: float,
    valid_row_mask_grouped: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SemanticTiebreakDiagnostics]:
    """Return semantic surprisal tiebreak over correct rows plus prompt diagnostics."""

    if anchor_logits_grouped.shape != candidate_correctness_grouped.shape:
        raise ValueError(
            "anchor_logits_grouped and candidate_correctness_grouped must match."
        )
    if anchor_logits_grouped.shape != cluster_ids_grouped.shape:
        raise ValueError("cluster_ids_grouped must match the grouped anchor shape.")
    if valid_row_mask_grouped is None:
        valid_mask = torch.ones_like(anchor_logits_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != anchor_logits_grouped.shape:
            raise ValueError(
                "valid_row_mask_grouped must match the grouped anchor shape."
            )
        valid_mask = valid_row_mask_grouped.to(torch.bool)

    safe_bonus_alpha = coerce_non_negative_float(bonus_alpha, default=0.0)
    safe_bonus_clip_max = coerce_non_negative_float(bonus_clip_max, default=0.0)
    device = anchor_logits_grouped.device
    dtype = anchor_logits_grouped.dtype
    num_prompts = int(anchor_logits_grouped.size(0))

    semantic_valid_mask = valid_mask & (cluster_ids_grouped >= 0)
    correct_row_mask = semantic_valid_mask & (
        candidate_correctness_grouped >= _SEMANTIC_TIEBREAK_CORRECTNESS_THRESHOLD
    )
    mode_surprisal_grouped = torch.zeros_like(anchor_logits_grouped, dtype=dtype)
    prompt_alpha_grouped = torch.zeros((num_prompts,), device=device, dtype=dtype)
    correct_anchor_mass_grouped = torch.zeros(
        (num_prompts,), device=device, dtype=dtype
    )
    distinct_correct_mode_count_grouped = torch.zeros(
        (num_prompts,), device=device, dtype=dtype
    )
    mode_count_grouped = torch.zeros((num_prompts,), device=device, dtype=dtype)

    masked_anchor_logits = torch.where(
        valid_mask,
        anchor_logits_grouped,
        torch.full_like(anchor_logits_grouped, torch.finfo(dtype).min),
    )
    anchor_log_probs = masked_group_log_softmax(masked_anchor_logits, valid_mask)
    anchor_probs = torch.where(
        valid_mask,
        torch.exp(anchor_log_probs),
        torch.zeros_like(anchor_log_probs),
    )

    for p in range(num_prompts):
        prompt_semantic_valid = semantic_valid_mask[p]
        semantic_idx = torch.where(prompt_semantic_valid)[0]
        if semantic_idx.numel() <= 0:
            continue

        row_clusters = cluster_ids_grouped[p, semantic_idx]
        unique_clusters = torch.unique(row_clusters, sorted=True)
        num_clusters = int(unique_clusters.numel())
        mode_count_grouped[p] = float(num_clusters)
        prompt_correct_anchor_mass = anchor_probs[p][correct_row_mask[p]].sum()
        correct_anchor_mass_grouped[p] = prompt_correct_anchor_mass.to(dtype)
        prompt_alpha_grouped[p] = float(safe_bonus_alpha) * (
            1.0 - correct_anchor_mass_grouped[p]
        ).clamp(min=0.0, max=1.0)

        correct_clusters = cluster_ids_grouped[p][correct_row_mask[p]]
        if correct_clusters.numel() > 0:
            distinct_correct_mode_count_grouped[p] = float(
                torch.unique(correct_clusters, sorted=True).numel()
            )

        if num_clusters <= 1:
            continue

        cluster_mass_by_id: dict[int, torch.Tensor] = {}
        for cid in unique_clusters.tolist():
            member_mask = prompt_semantic_valid & (cluster_ids_grouped[p] == cid)
            cluster_mass_by_id[int(cid)] = anchor_probs[p][member_mask].sum()
        normalizer = math.log(float(num_clusters))
        if normalizer <= 0.0:
            continue
        for cid, cluster_mass in cluster_mass_by_id.items():
            member_mask = prompt_semantic_valid & (cluster_ids_grouped[p] == cid)
            mode_surprisal_grouped[p, member_mask] = (
                -torch.log(cluster_mass.clamp(min=1e-12)) / float(normalizer)
            ).to(dtype)

    bonus_grouped = (
        prompt_alpha_grouped[:, None]
        * torch.clamp(mode_surprisal_grouped, min=0.0, max=float(safe_bonus_clip_max))
        * correct_row_mask.to(dtype)
    )
    bonus_grouped = torch.where(
        semantic_valid_mask,
        bonus_grouped,
        torch.zeros_like(bonus_grouped),
    )
    diagnostics = SemanticTiebreakDiagnostics(
        prompt_alpha_grouped=prompt_alpha_grouped,
        correct_anchor_mass_grouped=correct_anchor_mass_grouped,
        anchor_probs_grouped=anchor_probs,
        mode_surprisal_grouped=mode_surprisal_grouped,
        semantic_valid_row_mask_grouped=semantic_valid_mask,
        correct_row_mask_grouped=correct_row_mask,
        distinct_correct_mode_count_grouped=distinct_correct_mode_count_grouped,
        mode_count_grouped=mode_count_grouped,
    )
    return bonus_grouped, diagnostics


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
    weights_grouped = torch.where(
        valid_mask, weights_grouped, torch.zeros_like(weights_grouped)
    )
    uniform_weights = torch.where(
        valid_mask,
        1.0
        / valid_mask.to(dtype=utility_grouped.dtype)
        .sum(dim=1, keepdim=True)
        .clamp(min=1.0),
        torch.zeros_like(weights_grouped),
    )
    return torch.where(has_valid, weights_grouped, uniform_weights)


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
        (weights_grouped * valid_group_mask.to(weights_grouped.dtype))
        .sum(dim=1, keepdim=True)
        .to(weights_grouped.dtype)
    )
    advantages = weights_grouped - (target_mass_grouped * behavior_probs_grouped)
    return torch.where(valid_group_mask, advantages, torch.zeros_like(advantages))


def compute_group_centered_advantages(
    *,
    reward_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-prompt ``R_i - mean_j R_j`` advantages over valid rows."""

    if valid_row_mask_grouped is None:
        valid_mask = torch.ones_like(reward_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != reward_grouped.shape:
            raise ValueError("valid_row_mask_grouped must match reward_grouped.")
        valid_mask = valid_row_mask_grouped.to(torch.bool)
    rewards = torch.where(
        valid_mask,
        reward_grouped,
        torch.zeros_like(reward_grouped),
    )
    denom = valid_mask.to(dtype=rewards.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
    baseline = rewards.sum(dim=1, keepdim=True) / denom
    advantages = rewards - baseline
    return torch.where(valid_mask, advantages, torch.zeros_like(advantages))


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
            chunk_logits
            if chunk_logits.dtype == torch.float32
            else chunk_logits.float()
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
