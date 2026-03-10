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

"""Helpers for preparing generation/scoring artifacts used by the training loop.

The training loop expects a consistent set of artifacts for every batch:

``PreparedBatch``
    Bundles grouped completions, reward statistics, reference log-probability
    tensors, weighting diagnostics, and derived scores.
``_collect_batch_stats``
    Bridges generation/reward outputs with the scoring stack by building
    :class:`~training.scoring.ScoreBatch` objects,
    gathering reference log-probs when necessary, and computing weighting and
    length summaries.
``prepare_training_batch``
    High-level orchestration that runs the generation function, computes
    rewards, fetches reference log-probs, scores the policy, and returns a
    :class:`PreparedBatch` instance to the optimizer.

The helpers raise the internal :class:`_SkipBatch` exception when any step
fails; :func:`prepare_training_batch` catches it and returns ``None`` so the
caller can skip the problematic batch gracefully.
"""

from __future__ import annotations
# pylint: disable=broad-exception-caught

import logging
import math
import os
import time
import sys
import traceback
from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar, TYPE_CHECKING, cast
from types import SimpleNamespace

from .rewards import (
    _group_q_distribution,
    compute_reward_statistics,
    group_advantages,
    prepare_generation_batch,
    reward_moments,
)
from .scoring import (
    build_score_batch,
    build_sequence_scores,
    gather_reference_logprobs,
    reference_stats_from_policy_logprobs,
    reference_from_vllm_meta,
    score_model_outputs,
    summarize_completion_lengths,
    token_counts_from_score_batch,
)
from .runtime import require_torch
from .types import (
    BatchingSettings,
    GenerationBatch,
    GenerationFn,
    GenerationSettings,
    LengthStats,
    PreTrainedTokenizer,
    PromptCacheEntry,
    ReferenceLogprobs,
    RewardComputation,
    RewardMoments,
    ScoreBatch,
    SequenceScores,
    Tensor,
    TrainingLoopContext,
    AdvantageStats,
    QDistribution,
)
from .weighting import WeightStats, WeightingSettings
from .weighting.logic import compute_weight_stats, build_uniform_weight_stats

if TYPE_CHECKING:
    import torch

LOG = logging.getLogger(__name__)


def _progress_log_enabled() -> bool:
    raw = os.getenv("MAXENT_PROGRESS_LOG")
    if raw is None or not str(raw).strip():
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


_REF_LOGPROB_TRACE_LIMIT = 3
torch = require_torch("training_pipeline")


class _TraceCounter:
    """Stateful helper to guard noisy tracebacks."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._count = 0

    def next_occurrence(self) -> Optional[int]:
        """Return the next occurrence number or None when exhausted."""
        if self._count >= self._limit:
            return None
        self._count += 1
        return self._count

    def reset(self) -> None:
        """Reset the counter so new traces can be emitted."""
        self._count = 0


_REF_LOGPROB_TRACE_LIMITER = _TraceCounter(_REF_LOGPROB_TRACE_LIMIT)


def _deepspeed_zero_stage(accelerator: Any) -> int:
    """Return DeepSpeed ZeRO stage from Accelerate plugin state when present."""
    state = getattr(accelerator, "state", None)
    ds_plugin = getattr(state, "deepspeed_plugin", None)
    try:
        return int(getattr(ds_plugin, "zero_stage", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _rank_tag(accelerator: Any = None) -> str:
    """Return best-effort rank string for logging."""

    rank = getattr(accelerator, "process_index", None)
    world = getattr(accelerator, "num_processes", None)
    if rank is None:
        try:
            dist = getattr(torch, "distributed", None)
            if dist is not None and dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world = dist.get_world_size()
        except Exception:
            rank = None
            world = None
    if rank is None:
        return "rank=na"
    if world is None:
        return f"rank={rank}"
    return f"rank={rank}/{world}"


def _mean(values: List[float]) -> float:
    """Return the arithmetic mean for a non-empty list, else 0.0."""
    return float(sum(values)) / float(len(values)) if values else 0.0


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    """Return the weighted mean or 0.0 when weights are empty."""
    if not values or not weights:
        return 0.0
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights))) / total_weight


def _tokenize_for_diversity(text: str, tokenizer: Any = None) -> List[Any]:
    """Tokenize a completion for diversity metrics.

    Prefers the configured tokenizer when available; falls back to whitespace.
    """
    if not text:
        return []
    if tokenizer is not None:
        try:
            encode = getattr(tokenizer, "encode", None)
            if callable(encode):
                return list(encode(text, add_special_tokens=False))
            if callable(tokenizer):
                tokenized = tokenizer(text, add_special_tokens=False)
                if isinstance(tokenized, dict) and "input_ids" in tokenized:
                    return list(tokenized["input_ids"])
                if isinstance(tokenized, (list, tuple)):
                    return list(tokenized)
        except Exception:
            pass
    return [tok for tok in text.strip().split() if tok]


def _completion_diversity_metrics(
    grouped_completions: List[List[str]],
    *,
    tokenizer: Any = None,
    accelerator: Any = None,
) -> Dict[str, float]:
    """Return coarse diversity metrics for grouped completions.

    Metrics are averaged across prompt groups so each prompt contributes equally.
    When running distributed, gathers group metrics across ranks.
    """
    if not grouped_completions:
        return {}

    def _distinct_n(tokens: List[Any], n: int) -> float:
        if n <= 0 or len(tokens) < n:
            return 0.0
        total = len(tokens) - n + 1
        if total <= 0:
            return 0.0
        ngrams = {tuple(tokens[i : i + n]) for i in range(total)}
        return float(len(ngrams)) / float(total)

    def _jaccard_distance(sets: List[set[Any]]) -> float:
        if len(sets) < 2:
            return 0.0
        total_dist = 0.0
        pairs = 0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a = sets[i]
                b = sets[j]
                union = a | b
                if not union:
                    dist = 0.0
                else:
                    dist = 1.0 - (len(a & b) / float(len(union)))
                total_dist += dist
                pairs += 1
        return total_dist / float(pairs) if pairs > 0 else 0.0

    group_metrics: List[Dict[str, float]] = []
    for group in grouped_completions:
        if not group:
            continue
        normalized = [comp.strip() for comp in group if comp is not None]
        group_size = len(normalized)
        if group_size <= 0:
            continue
        all_tokens: List[Any] = []
        token_sets: List[set[Any]] = []
        for comp in normalized:
            tokens = _tokenize_for_diversity(comp, tokenizer)
            if tokens:
                all_tokens.extend(tokens)
                token_sets.append(set(tokens))
            else:
                token_sets.append(set())
        group_metrics.append(
            {
                "group_size": float(group_size),
                "distinct_1": _distinct_n(all_tokens, 1),
                "distinct_2": _distinct_n(all_tokens, 2),
                "jaccard": _jaccard_distance(token_sets),
            }
        )

    if not group_metrics:
        return {}

    if accelerator is not None and getattr(accelerator, "num_processes", 1) > 1:
        gather_fn = getattr(accelerator, "gather_object", None)
        if callable(gather_fn):
            try:
                gather_fn_typed = cast(Callable[[Any], Any], gather_fn)
                gathered = gather_fn_typed(group_metrics)  # pylint: disable=not-callable
                if isinstance(gathered, list):
                    merged: List[Dict[str, float]] = []
                    for item in gathered:
                        if isinstance(item, list):
                            merged.extend([m for m in item if isinstance(m, dict)])
                        elif isinstance(item, dict):
                            merged.append(item)
                    if merged:
                        group_metrics = merged
            except Exception:
                pass
        else:
            dist = getattr(torch, "distributed", None)
            if (
                dist is not None
                and callable(getattr(dist, "is_available", None))
                and callable(getattr(dist, "is_initialized", None))
                and dist.is_available()
                and dist.is_initialized()
            ):
                try:
                    world = int(getattr(dist, "get_world_size")())
                except (TypeError, ValueError, RuntimeError):
                    world = 0
                if world > 1:
                    try:
                        gathered = [None for _ in range(world)]
                        gather_obj = getattr(dist, "all_gather_object", None)
                        if callable(gather_obj):
                            gather_obj(gathered, group_metrics)
                            merged: List[Dict[str, float]] = []
                            for item in gathered:
                                if isinstance(item, list):
                                    merged.extend(
                                        [m for m in item if isinstance(m, dict)]
                                    )
                                elif isinstance(item, dict):
                                    merged.append(item)
                            if merged:
                                group_metrics = merged
                    except (RuntimeError, ValueError, TypeError):
                        pass

    distinct1_vals = [m["distinct_1"] for m in group_metrics if "distinct_1" in m]
    distinct2_vals = [m["distinct_2"] for m in group_metrics if "distinct_2" in m]
    jaccard_vals = [m["jaccard"] for m in group_metrics if "jaccard" in m]
    weights = [m.get("group_size", 0.0) for m in group_metrics]
    return {
        "distinct_1": _mean(distinct1_vals),
        "distinct_2": _mean(distinct2_vals),
        "jaccard": _mean(jaccard_vals),
        "distinct_1_micro": _weighted_mean(distinct1_vals, weights),
        "distinct_2_micro": _weighted_mean(distinct2_vals, weights),
        "jaccard_micro": _weighted_mean(jaccard_vals, weights),
    }


def _dist_any_flag(accelerator: Any, flag: bool) -> bool:
    """Return True if flag is True on any rank (best-effort, object gather)."""
    if getattr(accelerator, "num_processes", 1) <= 1:
        return bool(flag)
    torch_mod = sys.modules.get("torch")
    dist = getattr(torch_mod, "distributed", None) if torch_mod is not None else None
    if (
        dist is None
        or not callable(getattr(dist, "is_available", None))
        or not callable(getattr(dist, "is_initialized", None))
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return bool(flag)
    get_world_size = getattr(dist, "get_world_size", None)
    if not callable(get_world_size):
        return bool(flag)
    try:
        world_size = int(cast(Any, get_world_size()))
    except (TypeError, ValueError, RuntimeError):
        return bool(flag)
    gathered = [None for _ in range(max(world_size, 1))]
    gather_fn = getattr(dist, "all_gather_object", None)
    if not callable(gather_fn):
        return bool(flag)
    try:
        gather_fn(gathered, bool(flag))
        return any(bool(x) for x in gathered)
    except (RuntimeError, ValueError, TypeError):
        return bool(flag)


def _resolve_weighting_value(
    ctx: TrainingLoopContext,
    attribute: str,
    default: Optional[float] = None,
) -> Optional[float]:
    """Return a weighting attribute with graceful fallbacks.

    Some lightweight test contexts construct ``ctx.scoring.weighting`` as a
    simple namespace that omits optional controller fields (e.g., ``beta`` and
    ``tau``).  When those values are absent we try ``ctx.settings.scoring`` and
    finally fall back to a default so :func:`compute_reward_statistics` always
    receives valid arguments.

    :param ctx: Training loop context supplying scoring configs.
    :param attribute: Weighting attribute name to resolve.
    :param default: Value returned when the attribute is missing everywhere.
    :returns: Attribute value or ``default`` when undefined.
    """
    scoring_cfg = getattr(ctx, "scoring", None)
    weighting = getattr(scoring_cfg, "weighting", None)
    if weighting is not None:
        value = getattr(weighting, attribute, None)
        if value is not None:
            return value
    settings = getattr(ctx, "settings", None)
    if settings is not None:
        settings_scoring = getattr(settings, "scoring", None)
        settings_weighting = getattr(settings_scoring, "weighting", None)
        if settings_weighting is not None:
            value = getattr(settings_weighting, attribute, None)
            if value is not None:
                return value
    return default


def _maybe_apply_entropy_bonus(
    ctx: TrainingLoopContext,
    gen_batch: GenerationBatch,
    reward_comp: RewardComputation,
    ref_stats: ReferenceLogprobs,
    policy_entropy_sum: Optional[Any],
) -> RewardComputation:
    """Optionally add a policy-entropy bonus to rewards and refresh stats."""

    scoring_cfg = getattr(ctx, "scoring", None)
    bonus_coef = getattr(scoring_cfg, "policy_entropy_bonus_coef", 0.0)
    try:
        bonus_coef = float(bonus_coef)
    except (TypeError, ValueError):
        LOG.warning(
            "Invalid policy_entropy_bonus_coef=%s; skipping entropy bonus.", bonus_coef
        )
        return reward_comp
    if bonus_coef == 0.0 or not math.isfinite(bonus_coef):
        return reward_comp
    if policy_entropy_sum is None:
        return reward_comp
    total_utils = list(getattr(reward_comp, "total_utils", []) or [])
    if not total_utils:
        return reward_comp
    device = getattr(policy_entropy_sum, "device", None)
    dtype = getattr(policy_entropy_sum, "dtype", None)
    try:
        if not isinstance(policy_entropy_sum, torch.Tensor):
            entropy_tensor = torch.tensor(
                getattr(policy_entropy_sum, "arr", policy_entropy_sum),
                device=device,
                dtype=dtype or getattr(torch, "float32", None),
            )
        else:
            entropy_tensor = policy_entropy_sum
    except (TypeError, ValueError, RuntimeError):
        return reward_comp
    tok_counts = getattr(ref_stats, "ref_tok_counts", None)
    try:
        if not isinstance(tok_counts, torch.Tensor):
            tok_tensor = torch.tensor(
                getattr(tok_counts, "arr", tok_counts),
                device=getattr(entropy_tensor, "device", None),
                dtype=getattr(entropy_tensor, "dtype", None)
                or getattr(torch, "float32", None),
            )
        else:
            tok_tensor = tok_counts
    except (TypeError, ValueError, RuntimeError):
        return reward_comp
    if getattr(entropy_tensor, "numel", lambda: 0)() == 0:
        return reward_comp
    if getattr(tok_tensor, "numel", lambda: 0)() == 0:
        return reward_comp
    entropy_tensor = entropy_tensor.view(-1).float()
    tok_tensor = tok_tensor.view(-1).float()
    target_len = len(total_utils)
    try:
        ent_len = int(entropy_tensor.numel())
    except (TypeError, ValueError, RuntimeError, AttributeError):
        ent_len = len(getattr(entropy_tensor, "data", []))
    try:
        tok_len = int(tok_tensor.numel())
    except (TypeError, ValueError, RuntimeError, AttributeError):
        tok_len = len(getattr(tok_tensor, "data", []))
    n = min(target_len, ent_len, tok_len)
    if n <= 0:
        return reward_comp
    if ent_len != target_len or tok_len != target_len:
        LOG.debug(
            "Entropy bonus length mismatch | rewards=%d entropy=%d tok_counts=%d; aligning to %d",
            target_len,
            ent_len,
            tok_len,
            n,
        )
    device = getattr(entropy_tensor, "device", None)
    if device is not None and hasattr(tok_tensor, "to"):
        try:
            tok_tensor = tok_tensor.to(device)
        except (TypeError, ValueError, RuntimeError):
            pass
    entropy_slice = entropy_tensor[:n]
    tok_slice = tok_tensor[:n].clamp(min=1.0)
    entropy_per_tok_raw = entropy_slice / tok_slice
    entropy_per_tok = entropy_per_tok_raw
    group_sizes = [
        len(group) for group in getattr(gen_batch, "grouped_completions", []) or []
    ]
    if group_sizes:
        zscored = entropy_per_tok.clone()
        offset = 0
        for group_size in group_sizes:
            if offset >= n:
                break
            take = min(group_size, n - offset)
            if take <= 0:
                offset += max(group_size, 0)
                continue
            slice_vals = entropy_per_tok[offset : offset + take]
            mean_val = slice_vals.mean()
            try:
                std_val = slice_vals.std(unbiased=False)
            except (TypeError, RuntimeError, ValueError, AttributeError):
                std_val = slice_vals.std()
            std_val = std_val.clamp(min=1e-6)
            zscored[offset : offset + take] = (slice_vals - mean_val) / std_val
            offset += group_size
        if offset < n:
            LOG.debug(
                "Entropy bonus group sizes shorter than rewards | groups_total=%d rewards=%d",
                sum(group_sizes),
                n,
            )
            zscored[offset:n] = entropy_per_tok[offset:n]
        entropy_per_tok = zscored
    try:
        entropy_per_tok = torch.nan_to_num(
            entropy_per_tok, nan=0.0, posinf=0.0, neginf=0.0
        )
    except (AttributeError, TypeError, RuntimeError):
        isfinite = getattr(torch, "isfinite", None)
        if callable(isfinite):
            entropy_per_tok = torch.where(
                isfinite(entropy_per_tok),
                entropy_per_tok,
                torch.zeros_like(entropy_per_tok),
            )
    reward_std = None
    moments = getattr(reward_comp, "moments", None)
    if moments is not None:
        reward_std = getattr(moments, "std", None)
    try:
        reward_std = float(reward_std)
    except (TypeError, ValueError):
        reward_std = None
    if reward_std is None or not math.isfinite(reward_std) or reward_std <= 0.0:
        reward_std = 1.0
    zscale = bonus_coef * reward_std
    bonus_tensor = entropy_per_tok * zscale
    try:
        bonus_vals = bonus_tensor.detach().float().cpu().tolist()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        bonus_vals = [float(x) for x in getattr(bonus_tensor, "arr", bonus_tensor)]
    if len(bonus_vals) < target_len:
        bonus_vals.extend([0.0] * (target_len - len(bonus_vals)))
    entropy_vals = [b / zscale if zscale != 0.0 else 0.0 for b in bonus_vals]
    try:
        entropy_raw_vals = entropy_per_tok_raw.detach().float().cpu().tolist()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        entropy_raw_vals = [
            float(x) for x in getattr(entropy_per_tok_raw, "arr", entropy_per_tok_raw)
        ]
    if len(entropy_raw_vals) < target_len:
        entropy_raw_vals.extend([0.0] * (target_len - len(entropy_raw_vals)))
    new_total_utils = [float(u) + b for u, b in zip(total_utils, bonus_vals)]
    per_reward_values = dict(getattr(reward_comp, "per_reward_values", {}) or {})
    per_reward_values["policy_entropy_group_zscore"] = entropy_vals
    per_reward_values["policy_entropy_per_token"] = entropy_raw_vals
    per_reward_values["entropy_bonus"] = bonus_vals
    moments = RewardMoments(*reward_moments(new_total_utils, ctx.runtime.device))
    training_args = getattr(ctx, "training_args", None)
    scale_rewards = True
    if training_args is not None:
        scale_rewards = bool(getattr(training_args, "scale_rewards", True))
    advantage_stats = AdvantageStats(
        *group_advantages(
            gen_batch.grouped_completions,
            new_total_utils,
            scale_rewards=scale_rewards,
        )
    )
    q_temperature = _resolve_weighting_value(ctx, "q_temperature", 1.0) or 1.0
    q_epsilon = _resolve_weighting_value(ctx, "q_epsilon", 1e-6) or 1e-6
    q_distribution = QDistribution(
        *_group_q_distribution(
            gen_batch.grouped_completions,
            new_total_utils,
            q_temperature,
            q_epsilon,
        )
    )
    try:
        reward_comp.total_utils = new_total_utils
        reward_comp.per_reward_values = per_reward_values
        reward_comp.advantage = advantage_stats
        reward_comp.moments = moments
        reward_comp.q_distribution = q_distribution
        reward_comp.entropy_bonus_scale = reward_std
    except (AttributeError, TypeError, ValueError):
        reward_comp = RewardComputation(
            total_utils=new_total_utils,
            per_reward_values=per_reward_values,
            advantage=advantage_stats,
            pairs=reward_comp.pairs,
            q_distribution=q_distribution,
            moments=moments,
            ref_logprob_meta=getattr(reward_comp, "ref_logprob_meta", None),
            completion_metadata=getattr(reward_comp, "completion_metadata", None),
            entropy_bonus_scale=reward_std,
        )
    return reward_comp


@dataclass
class _BatchStats:
    """Aggregated batch statistics before building losses."""

    score_batch: ScoreBatch
    ref_stats: ReferenceLogprobs
    weight_stats: WeightStats
    length_stats: LengthStats
    num_completion_tokens: float
    prompt_token_count: float


@dataclass
class PreparedBatch:
    """Artifacts required to run optimization for a training batch.

    :param grouped_completions: Nested list of completions per prompt.
    :type grouped_completions: list[list[str]]
    :param reward_comp: Reward statistics computed by
        :func:`training.rewards.compute_reward_statistics`.
    :type reward_comp: ~maxent_grpo.training.types.rewards.RewardComputation
    :param batch_stats: Auxiliary scoring/weighting artifacts built by
        :func:`_collect_batch_stats`.
    :type batch_stats: _BatchStats
    :param total_input_tokens: Prompt + completion token count used for
        throughput logging.
    :type total_input_tokens: float
    :param scores: Structure containing current-model log-probabilities aligned
        with the reference statistics.
    :type scores: ~maxent_grpo.training.types.rewards.SequenceScores
    """

    grouped_completions: List[List[str]]
    reward_comp: RewardComputation
    batch_stats: _BatchStats
    total_input_tokens: float
    scores: SequenceScores
    diversity_metrics: Optional[Dict[str, float]] = None

    @property
    def weight_stats(self) -> WeightStats:
        """Shortcut to the batch weighting statistics."""
        return self.batch_stats.weight_stats

    @property
    def ref_stats(self) -> ReferenceLogprobs:
        """Return reference log-probability statistics for the batch."""
        return self.batch_stats.ref_stats

    @property
    def length_stats(self) -> LengthStats:
        """Return sequence length statistics computed for the batch."""
        return self.batch_stats.length_stats

    @property
    def num_completion_tokens(self) -> float:
        """Return total completion token count used to build the batch."""
        return self.batch_stats.num_completion_tokens


class _SkipBatch(RuntimeError):
    """Internal control-flow exception to skip invalid batches."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage or "unknown"


_T = TypeVar("_T")


def _require_artifact(value: Optional[_T], stage: str) -> _T:
    """Return ``value`` or raise the internal ``_SkipBatch`` sentinel.

    :param value: Artifact produced by a preparation step.
    :type value: Any | None
    :raises _SkipBatch: When ``value`` is ``None`` indicating the step failed.
    :returns: The validated artifact.
    :rtype: Any
    """
    if value is None:
        raise _SkipBatch(stage)
    return value


def _reference_stats_from_meta(
    flat_meta: Optional[List[Optional[Any]]],
    total_sequences: int,
    device: "torch.device",
) -> Optional[ReferenceLogprobs]:
    """Return reference stats when metadata fully covers all sequences.

    :param flat_meta: Flattened list of reference metadata per sequence.
    :type flat_meta: list | None
    :param total_sequences: Number of sequences expected in the batch.
    :type total_sequences: int
    :param device: Target device used for the resulting tensors.
    :type device: torch.device
    :returns: Reference log-probability statistics or ``None`` if metadata is
        missing/partial.
    :rtype: ~maxent_grpo.training.types.rewards.ReferenceLogprobs | None
    """
    if not flat_meta:
        return None
    if total_sequences <= 0:
        total_sequences = len(flat_meta)
        if total_sequences <= 0:
            return None
    ref_fn = reference_from_vllm_meta
    try:
        return ref_fn(flat_meta, total_sequences, device)
    except (RuntimeError, TypeError, ValueError):
        return None


def _behavior_logp_tensor_from_meta(
    flat_meta: Optional[List[Optional[Any]]],
    total_sequences: int,
    template_tensor: Any,
) -> Optional[Tensor]:
    """Return a tensor of behavior log-prob sums derived from metadata.

    The metadata is expected to contain ``logprob_sum`` entries aligned with
    the flattened completions list.  When metadata is missing or incomplete
    ``None`` is returned so downstream callers can fall back to current-policy
    log-probs.

    :param flat_meta: Flattened metadata per sequence emitted by generation.
    :type flat_meta: list | None
    :param total_sequences: Expected number of completions in the batch.
    :type total_sequences: int
    :param template_tensor: Tensor used to infer device/dtype for the result.
    :type template_tensor: torch.Tensor
    :returns: Tensor of log-prob sums or ``None`` if unavailable.
    :rtype: torch.Tensor | None
    """
    if total_sequences <= 0:
        return None
    fallback_vals: Optional[List[float]] = None
    if template_tensor is not None:
        try:
            if isinstance(template_tensor, torch.Tensor):
                fallback_vals = template_tensor.detach().float().cpu().view(-1).tolist()
            elif hasattr(template_tensor, "tolist"):
                fallback_raw = template_tensor.tolist()
                if (
                    isinstance(fallback_raw, list)
                    and fallback_raw
                    and isinstance(fallback_raw[0], list)
                ):
                    fallback_raw = fallback_raw[0]
                fallback_vals = [float(val) for val in fallback_raw]
            else:
                fallback_vals = [float(val) for val in list(template_tensor)]
        except (TypeError, ValueError, RuntimeError):
            fallback_vals = None
    meta_len = len(flat_meta) if flat_meta else 0
    if (not flat_meta or meta_len < total_sequences) and not fallback_vals:
        if meta_len > 0:
            LOG.debug(
                "Behavior log-prob metadata too short | meta_len=%d | sequences=%d",
                meta_len,
                total_sequences,
            )
        return None
    logprob_vals: List[float] = []
    missing = 0
    for idx in range(total_sequences):
        entry = flat_meta[idx] if flat_meta and idx < meta_len else None
        if entry is None:
            if fallback_vals is not None and idx < len(fallback_vals):
                logprob_vals.append(float(fallback_vals[idx]))
                missing += 1
                continue
            LOG.debug("Behavior log-prob metadata missing entry at idx=%d", idx)
            return None
        logprob_sum = getattr(entry, "logprob_sum", None)
        if logprob_sum is None and isinstance(entry, dict):
            logprob_sum = entry.get("logprob_sum")
        if logprob_sum is None:
            if fallback_vals is not None and idx < len(fallback_vals):
                logprob_vals.append(float(fallback_vals[idx]))
                missing += 1
                continue
            LOG.debug("Behavior log-prob metadata missing logprob_sum at idx=%d", idx)
            return None
        try:
            logprob_vals.append(float(logprob_sum))
        except (TypeError, ValueError):
            if fallback_vals is not None and idx < len(fallback_vals):
                logprob_vals.append(float(fallback_vals[idx]))
                missing += 1
                continue
            LOG.debug(
                "Behavior log-prob metadata has non-castable value at idx=%d: %s",
                idx,
                logprob_sum,
            )
            return None
    if missing:
        if not getattr(_behavior_logp_tensor_from_meta, "_warned_partial", False):
            LOG.warning(
                "Behavior log-prob metadata missing entries | missing_entries=%d/%d | "
                "falling back to policy logprobs for missing entries.",
                missing,
                total_sequences,
            )
            setattr(_behavior_logp_tensor_from_meta, "_warned_partial", True)
        else:
            LOG.debug(
                "Behavior log-prob metadata missing entries | missing_entries=%d/%d",
                missing,
                total_sequences,
            )
    new_tensor = getattr(template_tensor, "new_tensor", None)
    tensor_obj: Optional[Tensor] = None
    if callable(new_tensor):
        try:
            tensor_obj = cast(
                Tensor,
                new_tensor(
                    logprob_vals,
                    dtype=getattr(template_tensor, "dtype", None),
                    device=getattr(template_tensor, "device", None),
                ),
            )
        except (TypeError, ValueError, RuntimeError):
            tensor_obj = None
    if tensor_obj is None:
        torch_mod = sys.modules.get("torch")
        tensor_fn = (
            getattr(torch_mod, "tensor", None) if torch_mod is not None else None
        )
        if callable(tensor_fn):
            try:
                tensor_obj = cast(
                    Tensor,
                    tensor_fn(
                        logprob_vals,
                        dtype=getattr(template_tensor, "dtype", None),
                        device=getattr(template_tensor, "device", None),
                    ),
                )
            except (TypeError, ValueError, RuntimeError):
                tensor_obj = None
    if tensor_obj is None:
        LOG.debug("Unable to convert behavior log-prob metadata into a tensor.")
        return None
    view_attr = getattr(tensor_obj, "view", None)
    if callable(view_attr):
        try:
            tensor_obj = tensor_obj.view(-1)
        except (TypeError, ValueError, RuntimeError):
            LOG.debug("Failed to reshape logprob tensor to 1D.")
    return tensor_obj


def _coerce_token_logprob_value(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        if "logprob" in value:
            return _coerce_token_logprob_value(value.get("logprob"))
        if "log_prob" in value:
            return _coerce_token_logprob_value(value.get("log_prob"))
        if len(value) == 1:
            return _coerce_token_logprob_value(next(iter(value.values())))
        return None
    attr_val = getattr(value, "logprob", None)
    if attr_val is not None:
        return _coerce_token_logprob_value(attr_val)
    return None


def _extract_token_logprob_seq(entry: Optional[Any]) -> Optional[List[float]]:
    if entry is None:
        return None
    token_logprobs = None
    if isinstance(entry, Mapping):
        token_logprobs = entry.get("token_logprobs") or entry.get("logprobs")
        if isinstance(token_logprobs, Mapping):
            token_logprobs = token_logprobs.get("token_logprobs") or token_logprobs.get(
                "logprobs"
            )
    else:
        token_logprobs = getattr(entry, "token_logprobs", None) or getattr(
            entry, "logprobs", None
        )
    if token_logprobs is None:
        return None
    if isinstance(token_logprobs, (str, bytes, bytearray)):
        return None
    if not isinstance(token_logprobs, Iterable):
        return None
    cleaned: List[float] = []
    for item in token_logprobs:
        val = _coerce_token_logprob_value(item)
        if val is None:
            continue
        cleaned.append(val)
    return cleaned if cleaned else None


def _token_logp_tensor_from_meta(
    flat_meta: Optional[List[Optional[Any]]],
    total_sequences: int,
    token_mask: Optional[Tensor],
    fallback_token_logp: Optional[Tensor],
) -> Optional[Tensor]:
    """Return per-token log-prob tensor derived from vLLM metadata when available."""
    if total_sequences <= 0 or not flat_meta:
        return None
    meta_len = len(flat_meta)
    if meta_len <= 0:
        return None
    target_len: Optional[int] = None
    if token_mask is not None:
        try:
            target_len = int(getattr(token_mask, "shape", [0, 0])[1])
        except (TypeError, ValueError, IndexError):
            target_len = None
    if target_len is None and fallback_token_logp is not None:
        try:
            target_len = int(getattr(fallback_token_logp, "shape", [0, 0])[1])
        except (TypeError, ValueError, IndexError):
            target_len = None
    sequences: List[List[float]] = []
    missing = 0
    for idx in range(total_sequences):
        entry = flat_meta[idx] if idx < meta_len else None
        seq_vals = _extract_token_logprob_seq(entry)
        if seq_vals is None:
            if fallback_token_logp is not None:
                try:
                    row = fallback_token_logp[idx].detach().float().cpu().tolist()
                    if isinstance(row, list):
                        seq_vals = [float(val) for val in row]
                    else:
                        seq_vals = None
                except (
                    TypeError,
                    ValueError,
                    RuntimeError,
                    AttributeError,
                    IndexError,
                ):
                    seq_vals = None
            if seq_vals is None:
                return None
            missing += 1
        if target_len is None:
            target_len = max(len(seq_vals), 0)
        sequences.append(seq_vals)
    if target_len is None or target_len <= 0:
        return None
    aligned: List[List[float]] = []
    for seq_vals in sequences:
        seq_len = len(seq_vals)
        if seq_len >= target_len:
            aligned.append(seq_vals[-target_len:])
        else:
            pad = [0.0] * (target_len - seq_len)
            aligned.append(pad + seq_vals)
    if missing:
        if not getattr(_token_logp_tensor_from_meta, "_warned_partial", False):
            LOG.warning(
                "Token logprob metadata missing entries | missing_entries=%d/%d | "
                "falling back to policy token logprobs for missing rows.",
                missing,
                total_sequences,
            )
            setattr(_token_logp_tensor_from_meta, "_warned_partial", True)
        else:
            LOG.debug(
                "Token logprob metadata missing entries | missing_entries=%d/%d",
                missing,
                total_sequences,
            )
    dtype = None
    device = None
    if isinstance(fallback_token_logp, torch.Tensor):
        dtype = fallback_token_logp.dtype
        device = fallback_token_logp.device
    elif isinstance(token_mask, torch.Tensor):
        device = token_mask.device
    if dtype is None:
        dtype = getattr(torch, "float32", None)
    try:
        return torch.tensor(aligned, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError):
        return None


def _collect_batch_stats(
    ctx: TrainingLoopContext,
    gen_batch: GenerationBatch,
    reward_comp: RewardComputation,
    *,
    score_batch: Optional[ScoreBatch] = None,
    cur_logp_sum: Optional[Any] = None,
    policy_entropy_sum: Optional[Any] = None,
) -> Optional[_BatchStats]:
    """Gather scoring, reference, and weighting artifacts for a batch.

    :param ctx: Training loop context supplying runtime/scoring handles.
    :type ctx: training.types.TrainingLoopContext
    :param gen_batch: Outputs from :func:`prepare_generation_batch`.
    :type gen_batch: training.types.GenerationBatch
    :param reward_comp: Reward statistics used to build weighting/reward logs.
    :type reward_comp: ~maxent_grpo.training.types.rewards.RewardComputation
    :returns: Aggregated structures required downstream, or ``None`` when any
        stage fails (e.g., reference log-prob gathering).
    :rtype: _BatchStats | None
    """
    ref_stats = None
    last_ref_stats = getattr(ctx, "_last_ref_stats", None)

    def _ref_stats_empty(candidate: Optional[ReferenceLogprobs]) -> bool:
        if candidate is not None and not getattr(
            _collect_batch_stats, "_ref_candidate_seen", False
        ):
            LOG.debug(
                "Reference stats candidate present | type=%s | ref_logp_sum_raw_shape=%s | ref_logp_sum_shape=%s | ref_tok_counts_shape=%s",
                type(candidate).__name__,
                getattr(getattr(candidate, "ref_logp_sum_raw", None), "shape", None),
                getattr(getattr(candidate, "ref_logp_sum", None), "shape", None),
                getattr(getattr(candidate, "ref_tok_counts", None), "shape", None),
            )
            setattr(_collect_batch_stats, "_ref_candidate_seen", True)
        if candidate is None:
            _log_ref_diag = not getattr(_collect_batch_stats, "_ref_diag_logged", False)
            if _log_ref_diag:
                LOG.warning(
                    "Reference stats deemed empty: candidate=None | ref_logp_sum_raw=None | ref_logp_sum=None | ref_tok_counts=None"
                )
                setattr(_collect_batch_stats, "_ref_diag_logged", True)
            return True
        tensor = getattr(candidate, "ref_logp_sum_raw", None)
        if tensor is None:
            tensor = getattr(candidate, "ref_logp_sum", None)
        if tensor is None:
            try:
                length_fn = getattr(candidate, "__len__", None)
                if callable(length_fn):
                    return length_fn() == 0
            except TypeError:
                return False
        numel = getattr(tensor, "numel", None)
        if callable(numel):
            try:
                return numel() == 0
            except (RuntimeError, TypeError, ValueError):
                LOG.debug("Unable to compute numel for reference stats tensor.")
        to_list = getattr(tensor, "tolist", None)
        data = tensor
        if callable(to_list):
            try:
                data = to_list()
            except (RuntimeError, TypeError, ValueError):
                data = tensor
        if not isinstance(data, Sized):
            return False
        length = len(data)
        is_empty = length == 0
        if is_empty and not getattr(_collect_batch_stats, "_ref_diag_logged", False):

            def _describe(obj: Any) -> str:
                if obj is None:
                    return "None"
                shape = getattr(obj, "shape", None)
                numel_fn = getattr(obj, "numel", None)
                numel_val = None
                if callable(numel_fn):
                    try:
                        numel_val = numel_fn()
                    except (RuntimeError, TypeError, ValueError):
                        numel_val = "error"
                return f"{type(obj).__name__}(shape={shape}, numel={numel_val})"

            LOG.warning(
                "Reference stats deemed empty: candidate=%s | ref_logp_sum_raw=%s | ref_logp_sum=%s | ref_tok_counts=%s",
                type(candidate).__name__,
                _describe(getattr(candidate, "ref_logp_sum_raw", None)),
                _describe(getattr(candidate, "ref_logp_sum", None)),
                _describe(getattr(candidate, "ref_tok_counts", None)),
            )
            setattr(_collect_batch_stats, "_ref_diag_logged", True)
        elif not is_empty and not getattr(
            _collect_batch_stats, "_ref_diag_logged_success", False
        ):
            LOG.debug(
                "Reference stats non-empty | ref_logp_sum_raw_shape=%s | ref_tok_counts_shape=%s",
                getattr(getattr(candidate, "ref_logp_sum_raw", None), "shape", None),
                getattr(getattr(candidate, "ref_tok_counts", None), "shape", None),
            )
            setattr(_collect_batch_stats, "_ref_diag_logged_success", True)
        # If logp sum tensors are length zero but token counts exist, treat as non-empty
        tok_counts = getattr(candidate, "ref_tok_counts", None)
        if tok_counts is not None:
            tok_numel = _safe_numel(tok_counts)
            logp_sum_raw_numel = _safe_numel(
                getattr(candidate, "ref_logp_sum_raw", None)
            )
            logp_sum_numel = _safe_numel(getattr(candidate, "ref_logp_sum", None))
            if (
                tok_numel
                and tok_numel > 0
                and (logp_sum_raw_numel == 0 or logp_sum_numel == 0)
            ):
                LOG.debug(
                    "Reference stats: allowing zero-length logp_sum because tok_counts exist | tok_numel=%s | logp_sum_raw_numel=%s | logp_sum_numel=%s",
                    tok_numel,
                    logp_sum_raw_numel,
                    logp_sum_numel,
                )
                return False
        return is_empty

    def _warn_fallback(reason: str) -> None:
        flag = getattr(_collect_batch_stats, "_fallback_warned", False)
        if flag:
            return
        LOG.error(
            "Reference scoring degraded (%s); configured to reuse last cached ReferenceLogprobs. "
            "Set maxent_allow_stale_reference_logprobs=false to skip batches instead.",
            reason,
        )
        setattr(_collect_batch_stats, "_fallback_warned", True)

    def _retry_reference_gather(
        score_batch_retry: ScoreBatch, batching_cfg_retry: BatchingSettings
    ) -> Optional[ReferenceLogprobs]:
        original_slice = getattr(score_batch_retry, "slice_size", None)
        original_chunk = getattr(batching_cfg_retry, "logprob_chunk_size", None)
        slice_val = int(original_slice or score_batch_retry.total_sequences or 1)
        reduced_slice = max(1, slice_val // 2)
        if reduced_slice == original_slice:
            if reduced_slice > 1:
                reduced_slice -= 1
            else:
                reduced_slice = 1
        if original_slice is not None and reduced_slice == original_slice:
            return None
        logprob_chunk = int(original_chunk or reduced_slice)
        reduced_chunk = max(1, logprob_chunk // 2) if logprob_chunk > 1 else 1
        try:
            score_batch_retry.slice_size = reduced_slice
            batching_cfg_retry.logprob_chunk_size = reduced_chunk
            try:
                result = gather_reference_logprobs(
                    score_batch_retry,
                    ctx.runtime,
                    batching_cfg_retry,
                    trl_reference_scoring=trl_reference_scoring,
                    temperature=ref_temperature,
                )
                LOG.debug(
                    "Retry reference gather result | slice_size=%s | chunk_size=%s | result=%s",
                    reduced_slice,
                    reduced_chunk,
                    _describe_ref(result),
                )
                return result
            except (
                RuntimeError,
                ValueError,
                TypeError,
                AssertionError,
            ) as exc:  # pragma: no cover - best-effort logging
                LOG.warning("Retry reference gather failed: %s", exc)
                return None
        finally:
            if original_slice is not None:
                score_batch_retry.slice_size = original_slice
            if original_chunk is not None:
                batching_cfg_retry.logprob_chunk_size = original_chunk

    scoring_cfg = getattr(ctx, "scoring", None)
    if scoring_cfg is None:
        scoring_cfg = getattr(
            getattr(ctx, "settings", SimpleNamespace()), "scoring", None
        )
    if scoring_cfg is None:
        scoring_cfg = SimpleNamespace()
    batching_cfg = getattr(scoring_cfg, "batching", SimpleNamespace())
    if not getattr(batching_cfg, "prompt_length_cache_get", None):
        runtime_cache = getattr(getattr(ctx, "runtime", None), "prompt_cache_get", None)
        if callable(runtime_cache):
            batching_cfg.prompt_length_cache_get = runtime_cache
        else:
            batching_cfg.prompt_length_cache_get = (
                lambda _p, _cls=PromptCacheEntry: _cls(input_ids=[], attention_mask=[])
            )
    batching_cfg = cast(BatchingSettings, batching_cfg)
    gen_cfg = getattr(ctx, "generation", None)
    if gen_cfg is None:
        gen_cfg = getattr(
            getattr(ctx, "settings", SimpleNamespace()), "generation", None
        )
    gen_cfg = cast(GenerationSettings, gen_cfg)
    trl_reference_scoring = bool(getattr(scoring_cfg, "trl_reference_scoring", False))
    ref_temperature = getattr(gen_cfg, "gen_temperature", None)
    if score_batch is None:
        score_batch = build_score_batch(
            reward_comp,
            ctx.runtime.tokenizer,
            gen_cfg,
            batching_cfg,
        )
    accelerator = getattr(ctx.runtime, "accelerator", None)
    # Under DeepSpeed ZeRO, reference scoring may invoke collective param gathers even
    # in no_grad forward passes. If ranks diverge (some build an empty score batch),
    # later collectives can hang. Make the skip decision consistent across ranks.
    if _deepspeed_zero_stage(accelerator) >= 2 and _dist_any_flag(
        accelerator, score_batch is None
    ):
        if score_batch is None:
            LOG.warning(
                "Score batch build failed; completions=%d | prompts=%d",
                len(getattr(reward_comp.pairs, "completions", []) or []),
                len(getattr(reward_comp.pairs, "prompts", []) or []),
            )
        LOG.warning(
            "Skipping batch because at least one rank could not build a ScoreBatch "
            "(DeepSpeed ZeRO safety guard)."
        )
        return None
    if score_batch is None:
        LOG.warning(
            "Score batch build failed; completions=%d | prompts=%d",
            len(getattr(reward_comp.pairs, "completions", []) or []),
            len(getattr(reward_comp.pairs, "prompts", []) or []),
        )
        return None
    completion_ids = getattr(score_batch, "completion_ids", None)
    completion_attention_mask = getattr(score_batch, "completion_attention_mask", None)
    LOG.debug(
        "Score batch built | total_sequences=%d | max_prompt_len=%s | slice_size=%s | comp_ids_shape=%s | comp_mask_shape=%s | pad_id=%s",
        getattr(score_batch, "total_sequences", 0),
        getattr(score_batch, "max_prompt_len", None),
        getattr(score_batch, "slice_size", None),
        completion_ids.shape if completion_ids is not None else None,
        (
            completion_attention_mask.shape
            if completion_attention_mask is not None
            else None
        ),
        getattr(score_batch, "pad_token_id", None),
    )
    weighting_cfg = getattr(scoring_cfg, "weighting", None)
    grpo_mode = bool(getattr(weighting_cfg, "train_grpo_objective", False))
    ref_meta = getattr(reward_comp, "ref_logprob_meta", None)
    ref_source = (
        str(getattr(scoring_cfg, "reference_logprobs_source", "auto") or "auto")
        .strip()
        .lower()
    )
    if grpo_mode:
        ref_source = "model"
        ref_meta = None
    force_reference_model = ref_source in {
        "model",
        "reference_model",
        "ref_model",
        "reference",
    }
    ref_stats_source = "unknown"
    ref_meta_len = len(ref_meta) if ref_meta else 0
    if not grpo_mode and ref_source in {"policy", "none"}:
        ref_meta_len = 0
        if cur_logp_sum is not None:
            try:
                tok_counts = token_counts_from_score_batch(
                    score_batch, ctx.runtime, batching_cfg
                )
                ref_stats = reference_stats_from_policy_logprobs(
                    cur_logp_sum, tok_counts
                )
                ref_stats_source = "policy_logprobs"
                if not getattr(
                    _collect_batch_stats, "_policy_ref_forced_warned", False
                ):
                    LOG.info(
                        "Reference logprobs source=%s; using policy logprobs as reference (no frozen reference model).",
                        ref_source,
                    )
                    setattr(_collect_batch_stats, "_policy_ref_forced_warned", True)
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                LOG.warning("Policy-logprob reference fallback failed: %s", exc)
        else:
            LOG.warning(
                "Reference logprobs source=%s but policy logprobs are unavailable; "
                "reference scoring may fall back to reference model.",
                ref_source,
            )
    elif not grpo_mode and ref_meta_len and not force_reference_model:
        # Prefer reconstructing from metadata; always make an initial attempt.
        ref_stats = _reference_stats_from_meta(
            ref_meta,
            score_batch.total_sequences,
            ctx.runtime.device,
        )
        if ref_meta_len != score_batch.total_sequences:
            # Mismatch path: retry metadata reconstruction using score batch length.
            ref_stats = _reference_stats_from_meta(
                ref_meta,
                score_batch.total_sequences,
                ctx.runtime.device,
            )
        if ref_stats is not None:
            ref_stats_source = "vllm_meta"
    if (
        not grpo_mode
        and ref_stats is None
        and not force_reference_model
        and cur_logp_sum is not None
    ):
        # Prefer policy logprobs over a reference-model pass when metadata is missing.
        try:
            tok_counts = token_counts_from_score_batch(
                score_batch, ctx.runtime, batching_cfg
            )
            ref_stats = reference_stats_from_policy_logprobs(cur_logp_sum, tok_counts)
            ref_stats_source = "policy_logprobs"
            if not getattr(_collect_batch_stats, "_policy_ref_warned", False):
                LOG.warning(
                    "vLLM did not provide reference logprob metadata; using policy logprobs as reference "
                    "(KL ~= 0 fallback; set vllm_return_logprobs=true or "
                    "maxent_reference_logprobs_source=model to force a reference-model pass)."
                )
                setattr(_collect_batch_stats, "_policy_ref_warned", True)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:  # pragma: no cover - defensive diagnostics
            LOG.warning("Policy-logprob reference fallback failed: %s", exc)

    needs_ref_model_local = bool(force_reference_model or ref_stats is None)
    needs_ref_model_any = needs_ref_model_local
    # Keep reference scoring branches aligned under ZeRO to avoid mismatched collectives.
    if _deepspeed_zero_stage(accelerator) >= 2:
        needs_ref_model_any = _dist_any_flag(accelerator, needs_ref_model_local)

    if needs_ref_model_any:
        use_ref_try = bool(force_reference_model or ref_stats is None)
        ref_try = None
        try:
            ref_try = gather_reference_logprobs(
                score_batch,
                ctx.runtime,
                batching_cfg,
                trl_reference_scoring=trl_reference_scoring,
                temperature=ref_temperature,
            )
        except (RuntimeError, AssertionError) as exc:
            LOG.warning("Failed to gather reference logprobs: %s", exc)
            occurrence = _REF_LOGPROB_TRACE_LIMITER.next_occurrence()
            if occurrence is not None:
                LOG.error(
                    "Reference logprob traceback (occurrence %d/%d):\n%s",
                    occurrence,
                    _REF_LOGPROB_TRACE_LIMIT,
                    traceback.format_exc(),
                )
        except (
            ValueError,
            TypeError,
            AttributeError,
        ):  # pragma: no cover - defensive diag
            LOG.error(
                "Unexpected exception during gather_reference_logprobs: %s",
                traceback.format_exc(),
            )
        # Always run the gather on every rank when any rank needs it, but only
        # overwrite precomputed metadata-derived stats when needed/forced.
        if use_ref_try:
            ref_stats = ref_try
            if ref_stats is None:
                LOG.warning(
                    "gather_reference_logprobs returned None | slice_size=%s chunk_size=%s device=%s | ref_meta_len=%d | total_sequences=%d",
                    getattr(score_batch, "slice_size", None),
                    getattr(batching_cfg, "logprob_chunk_size", None),
                    getattr(ctx.runtime, "device", None),
                    ref_meta_len,
                    getattr(score_batch, "total_sequences", 0),
                )
            else:
                ref_stats_source = "reference_model"
                LOG.debug(
                    "Reference stats gathered | type=%s | ref_logp_sum_shape=%s | ref_tok_counts_shape=%s",
                    type(ref_stats).__name__,
                    getattr(getattr(ref_stats, "ref_logp_sum", None), "shape", None),
                    getattr(getattr(ref_stats, "ref_tok_counts", None), "shape", None),
                )
        elif ref_try is None:
            LOG.debug(
                "Reference gather ran for ZeRO alignment but returned None; keeping metadata-derived stats."
            )

    def _safe_numel(tensor: Any) -> Any:
        numel_fn = getattr(tensor, "numel", None)
        if callable(numel_fn):
            try:
                return numel_fn()
            except (RuntimeError, ValueError, TypeError):
                return "error"
        return None

    def _describe_ref(obj: Any) -> str:
        if obj is None:
            return "None"
        return (
            f"{type(obj).__name__}(shape_logp_sum_raw="
            f"{getattr(getattr(obj, 'ref_logp_sum_raw', None), 'shape', None)}, "
            f"shape_logp_sum={getattr(getattr(obj, 'ref_logp_sum', None), 'shape', None)}, "
            f"shape_tok_counts={getattr(getattr(obj, 'ref_tok_counts', None), 'shape', None)}, "
            f"numel_logp_sum_raw={_safe_numel(getattr(obj, 'ref_logp_sum_raw', None))}, "
            f"numel_logp_sum={_safe_numel(getattr(obj, 'ref_logp_sum', None))}, "
            f"numel_tok_counts={_safe_numel(getattr(obj, 'ref_tok_counts', None))})"
        )

    LOG.debug(
        "Reference stats post gather | is_none=%s | type=%s | ref_logp_sum_raw_shape=%s | ref_logp_sum_shape=%s | ref_tok_counts_shape=%s | ref_logp_sum_raw_numel=%s | ref_logp_sum_numel=%s | ref_tok_counts_numel=%s",
        ref_stats is None,
        type(ref_stats).__name__ if ref_stats is not None else None,
        getattr(getattr(ref_stats, "ref_logp_sum_raw", None), "shape", None),
        getattr(getattr(ref_stats, "ref_logp_sum", None), "shape", None),
        getattr(getattr(ref_stats, "ref_tok_counts", None), "shape", None),
        (
            _safe_numel(getattr(ref_stats, "ref_logp_sum_raw", None))
            if ref_stats is not None
            else None
        ),
        (
            _safe_numel(getattr(ref_stats, "ref_logp_sum", None))
            if ref_stats is not None
            else None
        ),
        (
            _safe_numel(getattr(ref_stats, "ref_tok_counts", None))
            if ref_stats is not None
            else None
        ),
    )

    fallback_guard = getattr(_collect_batch_stats, "_fallback_warned", False)
    LOG.debug(
        "Reference stats emptiness check | ref_stats=%s | last_ref_stats=%s | ref_meta_len=%d | total_sequences=%d",
        _describe_ref(ref_stats),
        _describe_ref(last_ref_stats),
        ref_meta_len,
        getattr(score_batch, "total_sequences", 0),
    )
    if _ref_stats_empty(ref_stats):
        retry_stats = _retry_reference_gather(score_batch, batching_cfg)
        if not _ref_stats_empty(retry_stats):
            ref_stats = retry_stats
        elif last_ref_stats is None and not fallback_guard:
            # No cache yet and retry failed: force a minimal slice/chunk attempt.
            fallback_guard = True
            LOG.debug(
                "Attempting forced minimal reference gather | orig_slice=%s orig_chunk=%s",
                getattr(score_batch, "slice_size", None),
                getattr(batching_cfg, "logprob_chunk_size", None),
            )
            single_score = ScoreBatch(
                prompt_entries=score_batch.prompt_entries,
                completion_ids=score_batch.completion_ids,
                completion_attention_mask=score_batch.completion_attention_mask,
                pad_token_id=score_batch.pad_token_id,
                max_prompt_len=score_batch.max_prompt_len,
                slice_size=1,
                total_sequences=score_batch.total_sequences,
                score_tail_tokens=score_batch.score_tail_tokens,
            )
            single_batching = BatchingSettings(
                logprob_chunk_size=1,
                score_slice=1,
                prompt_length_cache_get=getattr(
                    batching_cfg,
                    "prompt_length_cache_get",
                    lambda _p, _cls=PromptCacheEntry: _cls(
                        input_ids=[], attention_mask=[]
                    ),
                ),
                score_tail_tokens=getattr(batching_cfg, "score_tail_tokens", None),
                slice_prefetch=0,
                prompt_cache_size=getattr(batching_cfg, "prompt_cache_size", 0),
            )
            try:
                forced = gather_reference_logprobs(
                    single_score,
                    ctx.runtime,
                    single_batching,
                    trl_reference_scoring=trl_reference_scoring,
                    temperature=ref_temperature,
                )
            except (RuntimeError, ValueError, TypeError, AssertionError):
                LOG.warning(
                    "Forced minimal reference gather raised an exception; skipping."
                )
                forced = None
            else:
                LOG.debug(
                    "Forced minimal reference gather result | ref_stats=%s",
                    _describe_ref(forced),
                )
            if not _ref_stats_empty(forced):
                ref_stats = forced
    if _ref_stats_empty(ref_stats) and last_ref_stats is not None:
        allow_stale = (
            False
            if grpo_mode
            else bool(getattr(scoring_cfg, "allow_stale_reference_logprobs", False))
        )
        if not allow_stale:
            LOG.warning(
                "Reference gather empty; skipping batch instead of reusing stale ref stats "
                "(enable maxent_allow_stale_reference_logprobs to override)."
            )
            try:
                setattr(ctx.runtime, "_last_skip_stage", "reference_logprobs")
            except (AttributeError, TypeError):
                LOG.debug("Failed to record reference_logprobs skip stage on runtime.")
            return None
        LOG.warning(
            "Reference gather empty; reusing last ref stats | last_ref_shapes=%s/%s",
            getattr(getattr(last_ref_stats, "ref_logp_sum", None), "shape", None),
            getattr(getattr(last_ref_stats, "ref_tok_counts", None), "shape", None),
        )
        _warn_fallback("reference gather returned empty tensors")
        ref_stats = last_ref_stats
        ref_stats_source = "stale_cached"
    if _ref_stats_empty(ref_stats):
        LOG.error(
            "Reference scoring returned empty tensors even after retries; meta_len=%d | sequences=%d",
            ref_meta_len,
            getattr(score_batch, "total_sequences", 0),
        )
        try:
            setattr(ctx.runtime, "_last_skip_stage", "reference_logprobs")
        except (AttributeError, TypeError):
            LOG.debug("Failed to record reference_logprobs skip stage on runtime.")
        return None
    prev_source = getattr(ctx, "_ref_logprobs_source", None)
    if prev_source != ref_stats_source and ref_stats_source != "unknown":
        LOG.info("Reference logprobs source=%s", ref_stats_source)
        try:
            setattr(ctx, "_ref_logprobs_source", ref_stats_source)
        except (AttributeError, TypeError):
            LOG.debug("Failed to update reference logprobs source on context.")
    setattr(ctx, "_last_ref_stats", ref_stats)
    LOG.debug(
        "Reference stats gathered | avg_completion_tokens=%.2f",
        getattr(ref_stats, "avg_completion_tokens", 0.0),
    )
    if ref_stats is None:
        return None
    ref_stats = cast(ReferenceLogprobs, ref_stats)

    prompt_token_count = 0.0
    prompt_entries = score_batch.prompt_entries
    if prompt_entries:
        max_prompt_len = score_batch.max_prompt_len
        prompt_token_count = float(
            sum(min(entry.length, max_prompt_len) for entry in prompt_entries)
        )
    reward_comp = _maybe_apply_entropy_bonus(
        ctx,
        gen_batch,
        reward_comp,
        ref_stats,
        policy_entropy_sum,
    )
    weighting_cfg = cast(
        WeightingSettings, getattr(scoring_cfg, "weighting", SimpleNamespace())
    )
    weight_stats = compute_weight_stats(
        gen_batch.grouped_completions,
        reward_comp,
        ref_stats,
        weighting_cfg,
    )
    grpo_mode = bool(getattr(weighting_cfg, "train_grpo_objective", False))
    fallback_enabled = (
        bool(getattr(weighting_cfg, "allow_empty_weight_fallback", False))
        and not grpo_mode
    )
    if weight_stats is None or not getattr(weight_stats, "flat_weights", None):
        fallback_weights = None
        if fallback_enabled:
            fallback_weights = build_uniform_weight_stats(gen_batch.grouped_completions)
            if fallback_weights is not None:
                LOG.warning(
                    "MaxEnt weighting returned no samples; falling back to uniform GRPO weights for this batch."
                )
        if fallback_weights is not None:
            weight_stats = fallback_weights
        else:
            if grpo_mode:
                LOG.error(
                    "GRPO weighting returned no samples; check reward outputs or scale_rewards."
                )
            else:
                LOG.error(
                    "MaxEnt weighting returned no samples; check reward outputs, `maxent_tau`, or `maxent_q_temperature`."
                )
            return None
    LOG.debug(
        "Weight stats ready | entropy=%.4f",
        getattr(weight_stats, "weight_entropy", 0.0),
    )
    _, length_stats, num_completion_tokens = summarize_completion_lengths(
        ref_stats,
        ctx.generation.max_completion_len,
    )
    return _BatchStats(
        score_batch=score_batch,
        ref_stats=ref_stats,
        weight_stats=weight_stats,
        length_stats=length_stats,
        num_completion_tokens=num_completion_tokens,
        prompt_token_count=prompt_token_count,
    )


def prepare_training_batch(
    ctx: TrainingLoopContext,
    generator: GenerationFn[Any],
    batch: Dict[str, List[str]],
) -> Optional[PreparedBatch]:
    """Return a :class:`PreparedBatch` or ``None`` when any stage fails.

    :param ctx: Full training context containing generation/scoring configs.
    :type ctx: training.types.TrainingLoopContext
    :param generator: Callable that produces grouped completions (typically
        from :class:`training.rollout.CompletionGenerator`).
    :type generator: training.types.GenerationFn
    :param batch: Mini-batch produced by the training dataloader.
    :type batch: dict[str, list[str]]
    :returns: Fully-populated batch artifacts or ``None`` if generation,
        reward computation, reference scoring, or policy scoring fails.
    :rtype: PreparedBatch | None
    """
    try:
        prompt_value = cast(Any, batch.get("prompt"))
        if isinstance(prompt_value, str):
            batch = dict(batch)
            batch["prompt"] = [prompt_value]
        answer_value = cast(Any, batch.get("answer"))
        if isinstance(answer_value, str):
            batch = dict(batch)
            batch["answer"] = [answer_value]
        retry_limit = ctx.generation.vllm_rounds_cfg
        if retry_limit <= 0:
            retry_limit = ctx.optimization.schedule.num_generations
        rank_tag = _rank_tag(getattr(ctx.runtime, "accelerator", None))
        accelerator = getattr(ctx.runtime, "accelerator", None)
        is_main = bool(getattr(accelerator, "is_main_process", True))
        progress_log = _progress_log_enabled()
        LOG.debug(
            "Preparing training batch | %s | prompts=%d | retry_limit=%d",
            rank_tag,
            len(batch.get("prompt", [])),
            retry_limit,
        )
        gen_start = time.monotonic()
        if progress_log and is_main:
            LOG.info(
                "Stage generation start | %s | prompts=%d | num_generations=%d | retry_limit=%d",
                rank_tag,
                len(batch.get("prompt", [])),
                ctx.optimization.schedule.num_generations,
                retry_limit,
            )
        try:
            gen_batch = _require_artifact(
                prepare_generation_batch(
                    batch,
                    generator,
                    ctx.generation.generation_stats,
                    ctx.optimization.schedule.num_generations,
                    max_retry_rounds=retry_limit,
                ),
                stage="generation",
            )
        except TypeError:
            gen_batch = _require_artifact(
                prepare_generation_batch(
                    batch,
                    generator,
                    ctx.generation.generation_stats,
                    ctx.optimization.schedule.num_generations,
                    max_retry_rounds=retry_limit,
                ),
                stage="generation",
            )
        grouped = getattr(gen_batch, "grouped_completions", []) or []
        group_count = len(grouped)
        total_comps = sum(len(group) for group in grouped) if grouped else 0
        empty_groups = sum(1 for group in grouped if not group) if grouped else 0
        min_group = min((len(group) for group in grouped), default=0)
        max_group = max((len(group) for group in grouped), default=0)
        avg_group = total_comps / max(group_count, 1)
        runtime_tokenizer = getattr(ctx.runtime, "tokenizer", None)
        diversity_metrics = _completion_diversity_metrics(
            grouped,
            tokenizer=runtime_tokenizer if callable(runtime_tokenizer) else None,
            accelerator=accelerator,
        )
        LOG.debug(
            "Generation complete | %s | grouped_prompts=%d | total_completions=%d | empty_groups=%d | min_group=%d | max_group=%d | avg_group_size=%.2f",
            rank_tag,
            group_count,
            total_comps,
            empty_groups,
            min_group,
            max_group,
            avg_group,
        )
        if progress_log and is_main:
            LOG.info(
                "Stage generation done | %s | grouped_prompts=%d | total_completions=%d | seconds=%.2f",
                rank_tag,
                group_count,
                total_comps,
                time.monotonic() - gen_start,
            )
        q_temperature = _resolve_weighting_value(ctx, "q_temperature", 1.0)
        if q_temperature is None:
            q_temperature = 1.0
        q_epsilon = _resolve_weighting_value(ctx, "q_epsilon", 1e-6)
        if q_epsilon is None:
            q_epsilon = 1e-6
        reward_start = time.monotonic()
        if progress_log and is_main:
            LOG.info(
                "Stage reward stats start | %s | completions=%d",
                rank_tag,
                total_comps,
            )
        training_args = getattr(ctx, "training_args", None)
        scale_rewards = True
        if training_args is not None:
            scale_rewards = bool(getattr(training_args, "scale_rewards", True))
        reward_comp = _require_artifact(
            compute_reward_statistics(
                gen_batch,
                ctx.reward,
                ctx.runtime.device,
                q_temperature,
                q_epsilon,
                _resolve_weighting_value(ctx, "beta"),
                _resolve_weighting_value(ctx, "tau"),
                scale_rewards=scale_rewards,
            ),
            stage="reward_stats",
        )
        reward_mean = float(getattr(getattr(reward_comp, "moments", None), "mean", 0.0))
        reward_std = float(getattr(getattr(reward_comp, "moments", None), "std", 0.0))
        LOG.debug(
            "Reward statistics ready | %s | completions=%d | reward_mean=%.4f | reward_std=%.4f",
            rank_tag,
            len(getattr(reward_comp.pairs, "completions", []) or []),
            reward_mean,
            reward_std,
        )
        if progress_log and is_main:
            LOG.info(
                "Stage reward stats done | %s | reward_mean=%.4f | reward_std=%.4f | seconds=%.2f",
                rank_tag,
                reward_mean,
                reward_std,
                time.monotonic() - reward_start,
            )
        if not callable(runtime_tokenizer):
            # Unit tests often stub `ctx.runtime.tokenizer`; preserve the previous
            # control-flow by letting `_collect_batch_stats` supply the ScoreBatch.
            stats = _require_artifact(
                _collect_batch_stats(ctx, gen_batch, reward_comp),
                stage="batch_stats",
            )
            score_batch = stats.score_batch
            LOG.debug(
                "Batch stats ready | %s | sequences=%d | prompt_tokens=%.0f | completion_tokens=%.0f",
                rank_tag,
                getattr(stats.score_batch, "total_sequences", 0),
                stats.prompt_token_count,
                stats.num_completion_tokens,
            )
        else:
            score_batch = build_score_batch(
                reward_comp,
                cast(PreTrainedTokenizer, runtime_tokenizer),
                ctx.generation,
                ctx.scoring.batching,
            )
            accelerator = getattr(ctx.runtime, "accelerator", None)
            if _deepspeed_zero_stage(accelerator) >= 2 and _dist_any_flag(
                accelerator, score_batch is None
            ):
                if score_batch is None:
                    LOG.warning(
                        "Score batch build failed | %s | completions=%d | prompts=%d",
                        rank_tag,
                        len(getattr(reward_comp.pairs, "completions", []) or []),
                        len(getattr(reward_comp.pairs, "prompts", []) or []),
                    )
                LOG.warning(
                    "Skipping batch because at least one rank could not build a ScoreBatch "
                    "(DeepSpeed ZeRO safety guard)."
                )
                return None
            score_batch = _require_artifact(score_batch, stage="score_batch")
            completion_ids = getattr(score_batch, "completion_ids", None)
            completion_attention_mask = getattr(
                score_batch, "completion_attention_mask", None
            )
            LOG.debug(
                "Score batch built | %s | total_sequences=%d | max_prompt_len=%s | slice_size=%s | comp_ids_shape=%s | comp_mask_shape=%s | pad_id=%s",
                rank_tag,
                getattr(score_batch, "total_sequences", 0),
                getattr(score_batch, "max_prompt_len", None),
                getattr(score_batch, "slice_size", None),
                completion_ids.shape if completion_ids is not None else None,
                (
                    completion_attention_mask.shape
                    if completion_attention_mask is not None
                    else None
                ),
                getattr(score_batch, "pad_token_id", None),
            )
        return_entropy = bool(getattr(ctx.scoring, "policy_entropy", False))
        entropy_mode = getattr(ctx.scoring, "policy_entropy_mode", "exact")
        return_token_logp = bool(
            getattr(
                getattr(ctx.scoring, "weighting", None), "train_grpo_objective", False
            )
        )
        score_start = time.monotonic()
        if progress_log and is_main:
            LOG.info(
                "Stage policy scoring start | %s | total_sequences=%d",
                rank_tag,
                getattr(score_batch, "total_sequences", 0),
            )
        try:
            cur_logp_result = _require_artifact(
                score_model_outputs(
                    ctx.runtime.model,
                    score_batch,
                    ctx.scoring.batching,
                    ctx.runtime,
                    return_hidden=False,
                    return_entropy=return_entropy,
                    entropy_mode=entropy_mode,
                    return_token_logp=return_token_logp,
                ),
                stage="policy_scoring",
            )
        except TypeError:
            cur_logp_result = _require_artifact(
                score_model_outputs(
                    ctx.runtime.model,
                    score_batch,
                    ctx.scoring.batching,
                    ctx.runtime,
                ),
                stage="policy_scoring",
            )
        logprob_tensor = (
            cur_logp_result[0]
            if isinstance(cur_logp_result, tuple)
            else cur_logp_result
        )
        LOG.debug(
            "Policy scoring complete | %s | logprob_shape=%s",
            rank_tag,
            getattr(logprob_tensor, "shape", None),
        )
        if progress_log and is_main:
            LOG.info(
                "Stage policy scoring done | %s | logprob_shape=%s | seconds=%.2f",
                rank_tag,
                getattr(logprob_tensor, "shape", None),
                time.monotonic() - score_start,
            )
        policy_entropy_sum = None
        token_logp = None
        token_mask = None
        if isinstance(cur_logp_result, tuple):
            if return_entropy:
                if return_token_logp and len(cur_logp_result) >= 5:
                    (
                        cur_logp_sum,
                        pooled_hidden,
                        policy_entropy_sum,
                        token_logp,
                        token_mask,
                    ) = cur_logp_result
                elif len(cur_logp_result) >= 3:
                    cur_logp_sum, pooled_hidden, policy_entropy_sum = cur_logp_result[
                        :3
                    ]
                else:
                    cur_logp_sum, pooled_hidden = cur_logp_result[:2]
            else:
                if return_token_logp and len(cur_logp_result) >= 4:
                    (
                        cur_logp_sum,
                        pooled_hidden,
                        token_logp,
                        token_mask,
                    ) = cur_logp_result
                else:
                    cur_logp_sum, pooled_hidden = cur_logp_result[:2]
        else:
            cur_logp_sum, pooled_hidden = cur_logp_result, None
        if callable(runtime_tokenizer):
            stats = _require_artifact(
                _collect_batch_stats(
                    ctx,
                    gen_batch,
                    reward_comp,
                    score_batch=score_batch,
                    cur_logp_sum=cur_logp_sum,
                    policy_entropy_sum=policy_entropy_sum,
                ),
                stage="batch_stats",
            )
            LOG.debug(
                "Batch stats ready | %s | sequences=%d | prompt_tokens=%.0f | completion_tokens=%.0f",
                rank_tag,
                getattr(getattr(stats, "score_batch", None), "total_sequences", 0),
                stats.prompt_token_count,
                stats.num_completion_tokens,
            )
        behavior_source = (
            str(getattr(ctx.scoring, "behavior_logprobs_source", "model") or "model")
            .strip()
            .lower()
        )
        use_vllm_behavior = behavior_source in {"vllm", "metadata", "meta"}
        behavior_tensor = None
        if use_vllm_behavior:
            behavior_tensor = _behavior_logp_tensor_from_meta(
                getattr(reward_comp, "ref_logprob_meta", None),
                stats.score_batch.total_sequences,
                cur_logp_sum,
            )
        old_token_logp = None
        if return_token_logp and use_vllm_behavior:
            old_token_logp = _token_logp_tensor_from_meta(
                getattr(reward_comp, "ref_logprob_meta", None),
                stats.score_batch.total_sequences,
                token_mask,
                token_logp,
            )
        try:
            scores = build_sequence_scores(
                cur_logp_sum,
                stats.ref_stats,
                pooled_hidden,
                behavior_logp_sum=behavior_tensor,
                policy_entropy_sum=policy_entropy_sum,
                token_logp=token_logp,
                token_mask=token_mask,
                old_token_logp=old_token_logp,
            )
        except TypeError:
            if behavior_tensor is not None:
                try:
                    scores = build_sequence_scores(
                        cur_logp_sum,
                        stats.ref_stats,
                        behavior_logp_sum=behavior_tensor,
                        policy_entropy_sum=policy_entropy_sum,
                    )
                except TypeError:
                    scores = build_sequence_scores(cur_logp_sum, stats.ref_stats)
            else:
                scores = build_sequence_scores(
                    cur_logp_sum, stats.ref_stats, policy_entropy_sum=policy_entropy_sum
                )
        return PreparedBatch(
            grouped_completions=gen_batch.grouped_completions,
            reward_comp=reward_comp,
            batch_stats=stats,
            total_input_tokens=stats.prompt_token_count + stats.num_completion_tokens,
            scores=scores,
            diversity_metrics=diversity_metrics or None,
        )
    except _SkipBatch as exc:
        skip_stage = getattr(exc, "stage", "unknown")
        try:
            setattr(ctx.runtime, "_last_skip_stage", skip_stage)
        except (AttributeError, TypeError):
            LOG.debug("Failed to record skip stage on runtime.")
        LOG.debug(
            "Skipping training batch: stage=%s returned None | %s",
            skip_stage,
            _rank_tag(getattr(ctx.runtime, "accelerator", None)),
        )
        return None


__all__ = ["PreparedBatch", "prepare_training_batch"]
