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

import logging
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeVar, TYPE_CHECKING
from types import SimpleNamespace

from .weighting.loss import SequenceScores
from .rewards import (
    compute_reward_statistics,
    prepare_generation_batch,
)
from .scoring import (
    build_score_batch,
    build_sequence_scores,
    gather_reference_logprobs,
    reference_from_vllm_meta,
    score_model_outputs,
    summarize_completion_lengths,
)
from .types import (
    GenerationBatch,
    GenerationFn,
    LengthStats,
    PromptCacheEntry,
    ReferenceLogprobs,
    RewardComputation,
    ScoreBatch,
    TrainingLoopContext,
)
from .weighting import WeightStats
from .weighting.logic import compute_weight_stats

if TYPE_CHECKING:
    import torch

LOG = logging.getLogger(__name__)
_REF_LOGPROB_TRACE_LIMIT = 3


class _TraceCounter:
    """Stateful helper to guard noisy tracebacks."""

    def __init__(self, limit: int):
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
    :type reward_comp: training.types.RewardComputation
    :param batch_stats: Auxiliary scoring/weighting artifacts built by
        :func:`_collect_batch_stats`.
    :type batch_stats: _BatchStats
    :param total_input_tokens: Prompt + completion token count used for
        throughput logging.
    :type total_input_tokens: float
    :param scores: Structure containing current-model log-probabilities aligned
        with the reference statistics.
    :type scores: training.weighting.loss.SequenceScores
    """

    grouped_completions: List[List[str]]
    reward_comp: RewardComputation
    batch_stats: _BatchStats
    total_input_tokens: float
    scores: SequenceScores
    seed_metrics: Optional[Dict[str, float]] = None
    seed_heatmap: Optional[Dict[str, Any]] = None

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


_T = TypeVar("_T")


def _require_artifact(value: Optional[_T]) -> _T:
    """Return ``value`` or raise the internal ``_SkipBatch`` sentinel.

    :param value: Artifact produced by a preparation step.
    :type value: Any | None
    :raises _SkipBatch: When ``value`` is ``None`` indicating the step failed.
    :returns: The validated artifact.
    :rtype: Any
    """
    if value is None:
        raise _SkipBatch()
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
    :rtype: training.types.ReferenceLogprobs | None
    """
    if not flat_meta or total_sequences <= 0:
        return None
    ref_fn = reference_from_vllm_meta
    try:
        return ref_fn(flat_meta, total_sequences, device)
    except (RuntimeError, TypeError, ValueError):
        return None


def _collect_batch_stats(
    ctx: TrainingLoopContext,
    gen_batch: GenerationBatch,
    reward_comp: RewardComputation,
) -> Optional[_BatchStats]:
    """Gather scoring, reference, and weighting artifacts for a batch.

    :param ctx: Training loop context supplying runtime/scoring handles.
    :type ctx: training.types.TrainingLoopContext
    :param gen_batch: Outputs from :func:`prepare_generation_batch`.
    :type gen_batch: training.types.GenerationBatch
    :param reward_comp: Reward statistics used to build weighting/reward logs.
    :type reward_comp: training.types.RewardComputation
    :returns: Aggregated structures required downstream, or ``None`` when any
        stage fails (e.g., reference log-prob gathering).
    :rtype: _BatchStats | None
    """
    ref_stats = None
    scoring_cfg = getattr(ctx, "scoring", None)
    if scoring_cfg is None:
        scoring_cfg = getattr(
            getattr(ctx, "settings", SimpleNamespace()), "scoring", None
        )
    batching_cfg = getattr(scoring_cfg, "batching", SimpleNamespace())
    if not getattr(batching_cfg, "prompt_length_cache_get", None):
        # Provide a no-op cache accessor for tests and lightweight callers.
        batching_cfg.prompt_length_cache_get = lambda _p, _cls=PromptCacheEntry: _cls(
            input_ids=[], attention_mask=[]
        )
    gen_cfg = getattr(ctx, "generation", None)
    if gen_cfg is None:
        gen_cfg = getattr(
            getattr(ctx, "settings", SimpleNamespace()), "generation", None
        )
    score_batch = build_score_batch(
        reward_comp,
        ctx.runtime.tokenizer,
        gen_cfg,
        batching_cfg,
    )
    if score_batch is None:
        return None
    ref_meta = getattr(reward_comp, "ref_logprob_meta", None)
    ref_meta_len = len(ref_meta) if ref_meta else 0
    if ref_meta_len:
        # Prefer reconstructing from metadata; always make an initial attempt.
        ref_stats = _reference_stats_from_meta(
            ref_meta,
            score_batch.total_sequences,
            ctx.runtime.device,
        )
        if ref_meta_len != score_batch.total_sequences:
            # Mismatch path: rebuild again and skip any remote gather.
            ref_stats = _reference_stats_from_meta(
                ref_meta,
                score_batch.total_sequences,
                ctx.runtime.device,
            )
        elif ref_stats is None:
            try:
                ref_stats = reference_from_vllm_meta(
                    ref_meta,
                    score_batch.total_sequences,
                    ctx.runtime.device,
                )
            except (RuntimeError, TypeError, ValueError):
                ref_stats = None
    if ref_stats is None:
        try:
            ref_stats = gather_reference_logprobs(
                score_batch,
                ctx.runtime,
                ctx.scoring.batching,
            )
        except RuntimeError as exc:
            LOG.warning("Failed to gather reference logprobs: %s", exc)
            occurrence = _REF_LOGPROB_TRACE_LIMITER.next_occurrence()
            if occurrence is not None:
                LOG.error(
                    "Reference logprob traceback (occurrence %d/%d):\n%s",
                    occurrence,
                    _REF_LOGPROB_TRACE_LIMIT,
                    traceback.format_exc(),
                )
            return None
    if ref_stats is None:
        return None
    prompt_token_count = 0.0
    prompt_entries = score_batch.prompt_entries
    if prompt_entries:
        max_prompt_len = score_batch.max_prompt_len
        prompt_token_count = float(
            sum(min(entry.length, max_prompt_len) for entry in prompt_entries)
        )
    weight_stats = compute_weight_stats(
        gen_batch.grouped_completions,
        reward_comp,
        ref_stats,
        ctx.scoring.weighting,
    )
    if weight_stats is None or not weight_stats.flat_weights:
        return None
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
    generator: GenerationFn,
    batch: Dict[str, List[str]],
) -> Optional[PreparedBatch]:
    """Return a :class:`PreparedBatch` or ``None`` when any stage fails.

    :param ctx: Full training context containing generation/scoring configs.
    :type ctx: training.types.TrainingLoopContext
    :param generator: Callable that produces grouped completions (typically
        from :class:`training.generation.CompletionGenerator`).
    :type generator: training.types.GenerationFn
    :param batch: Mini-batch produced by the training dataloader.
    :type batch: dict[str, list[str]]
    :returns: Fully-populated batch artifacts or ``None`` if generation,
        reward computation, reference scoring, or policy scoring fails.
    :rtype: PreparedBatch | None
    """
    try:
        if isinstance(batch.get("prompt"), str):
            batch = dict(batch)
            batch["prompt"] = [batch["prompt"]]
        if isinstance(batch.get("answer"), str):
            batch = dict(batch)
            batch["answer"] = [batch["answer"]]
        retry_limit = ctx.generation.vllm_rounds_cfg
        if retry_limit <= 0:
            retry_limit = ctx.optimization.schedule.num_generations
        try:
            gen_batch = _require_artifact(
                prepare_generation_batch(
                    batch,
                    generator,
                    ctx.generation.generation_stats,
                    ctx.optimization.schedule.num_generations,
                    max_retry_rounds=retry_limit,
                    seed_augmentation=getattr(
                        ctx.generation, "seed_augmentation", None
                    ),
                )
            )
        except TypeError:
            gen_batch = _require_artifact(
                prepare_generation_batch(
                    batch,
                    generator,
                    ctx.generation.generation_stats,
                    ctx.optimization.schedule.num_generations,
                    max_retry_rounds=retry_limit,
                )
            )
        reward_comp = _require_artifact(
            compute_reward_statistics(
                gen_batch,
                ctx.reward,
                ctx.runtime.device,
                ctx.scoring.weighting.q_temperature,
                ctx.scoring.weighting.q_epsilon,
            )
        )
        stats = _require_artifact(
            _collect_batch_stats(
                ctx,
                gen_batch,
                reward_comp,
            )
        )
        # Enable pooled hidden states when InfoSeed is active so seed loss can pool per-sequence reps.
        should_pool = any(
            [
                getattr(ctx.scoring, "info_seed_lambda", 0.0) > 0.0,
                getattr(ctx.scoring, "info_seed_alpha_entropy", 0.0) != 0.0,
                getattr(ctx.generation, "seed_augmentation", None) is not None,
            ]
        )
        try:
            cur_logp_result = _require_artifact(
                score_model_outputs(
                    ctx.runtime.model,
                    stats.score_batch,
                    ctx.scoring.batching,
                    ctx.runtime,
                    return_hidden=should_pool,
                    pooling=getattr(ctx.scoring, "info_seed_pooling", "mean"),
                )
            )
        except TypeError:
            cur_logp_result = _require_artifact(
                score_model_outputs(
                    ctx.runtime.model,
                    stats.score_batch,
                    ctx.scoring.batching,
                    ctx.runtime,
                )
            )
        if isinstance(cur_logp_result, tuple):
            cur_logp_sum, pooled_hidden = cur_logp_result
        else:
            cur_logp_sum, pooled_hidden = cur_logp_result, None
        try:
            scores = build_sequence_scores(cur_logp_sum, stats.ref_stats, pooled_hidden)
        except TypeError:
            scores = build_sequence_scores(cur_logp_sum, stats.ref_stats)
        # Optional seed metadata wiring for InfoSeed objectives.
        seed_inputs = None
        seed_metrics: Dict[str, float] = {}
        seed_heatmap = None
        completion_meta = getattr(reward_comp, "completion_metadata", None)
        if pooled_hidden is not None and completion_meta:
            import torch as torch_mod

            seed_ids = []
            is_seed_aug = []
            for meta in reward_comp.completion_metadata or []:
                seed_ids.append(int(meta.get("seed_id", -1)) if meta else -1)
                is_seed_aug.append(
                    bool(meta.get("is_seed_aug", False)) if meta else False
                )
            if seed_ids:
                seed_ids_t = torch_mod.tensor(
                    seed_ids, device=pooled_hidden.device, dtype=torch_mod.long
                )
                is_seed_aug_t = torch_mod.tensor(
                    is_seed_aug, device=pooled_hidden.device, dtype=torch_mod.bool
                )
                try:
                    from .weighting.loss import SeedInfoInputs
                except (
                    ImportError,
                    AttributeError,
                ):  # pragma: no cover - optional seed support
                    SeedInfoInputs = None
                if SeedInfoInputs is not None:
                    seed_inputs = SeedInfoInputs(
                        seed_ids=seed_ids_t,
                        pooled_hidden=pooled_hidden,
                        is_seed_aug=is_seed_aug_t,
                    )
                    valid_mask = seed_ids_t >= 0
                    if seed_inputs.is_seed_aug is not None and valid_mask.any():
                        aug_mask = valid_mask & seed_inputs.is_seed_aug
                        total = valid_mask.sum().item()
                        if total > 0:
                            seed_metrics["seed_aug_frac"] = float(
                                aug_mask.sum().item()
                            ) / float(total)
        if seed_inputs is not None:
            seed_head = getattr(ctx.runtime.model, "seed_head", None)
            if callable(seed_head):
                try:
                    seed_logits = seed_head(pooled_hidden)
                    seed_inputs.logits = seed_logits
                    # Seed prediction accuracy metrics
                    if seed_logits is not None:
                        valid_mask = seed_inputs.seed_ids >= 0
                        if valid_mask.any():
                            preds = seed_logits.argmax(dim=-1)
                            acc = (
                                (preds[valid_mask] == seed_inputs.seed_ids[valid_mask])
                                .float()
                                .mean()
                                .item()
                            )
                            seed_metrics["seed_pred_acc"] = acc
                            if seed_inputs.is_seed_aug is not None:
                                aug_mask = valid_mask & seed_inputs.is_seed_aug
                                if aug_mask.any():
                                    aug_acc = (
                                        (
                                            preds[aug_mask]
                                            == seed_inputs.seed_ids[aug_mask]
                                        )
                                        .float()
                                        .mean()
                                        .item()
                                    )
                                    seed_metrics["seed_pred_acc_aug"] = aug_acc
                        # Diversity: average pairwise distance between seed means
                        try:
                            unique_seeds = torch_mod.unique(
                                seed_inputs.seed_ids[valid_mask]
                            )
                            if unique_seeds.numel() > 1:
                                means = []
                                for sid in unique_seeds:
                                    sid_mask = valid_mask & (
                                        seed_inputs.seed_ids == sid
                                    )
                                    if sid_mask.any():
                                        means.append(
                                            seed_inputs.pooled_hidden[sid_mask].mean(
                                                dim=0
                                            )
                                        )
                                if len(means) > 1:
                                    stacked = torch_mod.stack(means)
                                    pdist = torch_mod.nn.functional.pdist(
                                        stacked, p=2
                                    ).mean()
                                    seed_metrics["seed_diversity_l2"] = float(
                                        pdist.item()
                                    )
                                    heatmap = torch_mod.nn.functional.cosine_similarity(
                                        stacked.unsqueeze(1),
                                        stacked.unsqueeze(0),
                                        dim=-1,
                                    )
                                    seed_heatmap = {
                                        "labels": [int(s.item()) for s in unique_seeds],
                                        "matrix": heatmap.detach().cpu().tolist(),
                                    }
                                else:
                                    seed_heatmap = None
                            else:
                                seed_heatmap = None
                        except (RuntimeError, ValueError, TypeError):
                            seed_heatmap = None
                except (RuntimeError, ValueError, TypeError):
                    seed_inputs.logits = None
            setattr(scores, "seed_aux", seed_inputs)
        else:
            seed_heatmap = None
        # Per-subset entropy (orig vs seed-aug) derived from weights + metadata.
        if getattr(reward_comp, "completion_metadata", None):
            weights = stats.weight_stats.flat_weights
            if len(weights) == len(reward_comp.completion_metadata):
                import math

                accelerator = ctx.runtime.accelerator

                def _gather_weights(weight_subset: list[float]) -> list[float]:
                    if getattr(accelerator, "num_processes", 1) <= 1:
                        return weight_subset
                    gather_fn = getattr(accelerator, "gather_object", None)
                    if callable(gather_fn):
                        gathered = gather_fn(weight_subset)
                        merged: list[float] = []
                        for chunk in gathered:
                            merged.extend(float(v) for v in chunk)
                        return merged
                    return weight_subset

                def _entropy(subset_weights: list[float]) -> float:
                    if not subset_weights:
                        return 0.0
                    total = sum(subset_weights)
                    if total <= 0:
                        return 0.0
                    ent = 0.0
                    for w in subset_weights:
                        p = w / total
                        if p > 0:
                            ent -= p * math.log(p + 1e-12)
                    return ent

                orig_weights: list[float] = []
                seed_weights: list[float] = []
                for w, meta in zip(weights, reward_comp.completion_metadata):
                    if meta and meta.get("is_seed_aug", False):
                        seed_weights.append(w)
                    else:
                        orig_weights.append(w)
                orig_weights = _gather_weights(orig_weights)
                seed_weights = _gather_weights(seed_weights)
                seed_metrics["entropy/orig"] = _entropy(orig_weights)
                seed_metrics["entropy/seed_aug"] = _entropy(seed_weights)
        return PreparedBatch(
            grouped_completions=gen_batch.grouped_completions,
            reward_comp=reward_comp,
            batch_stats=stats,
            total_input_tokens=stats.prompt_token_count + stats.num_completion_tokens,
            scores=scores,
            seed_metrics=seed_metrics or None,
            seed_heatmap=seed_heatmap,
        )
    except _SkipBatch:
        return None


__all__ = ["PreparedBatch", "prepare_training_batch"]

# Preserve a self-reference so monkeypatch paths like ``training.pipeline.pipeline``
# resolve even after test shuffling or aliasing.
pipeline = sys.modules[__name__]
# Expose the module under the legacy ``training.pipeline`` alias used in tests.
sys.modules["training.pipeline"] = pipeline
