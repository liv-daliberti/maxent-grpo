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
    :class:`~maxent_helpers.run_training_scoring.ScoreBatch` objects,
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
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeVar, TYPE_CHECKING

from .run_training_loss import SequenceScores
from .run_training_rewards import (
    compute_reward_statistics,
    prepare_generation_batch,
)
from .run_training_scoring import (
    build_score_batch,
    build_sequence_scores,
    gather_reference_logprobs,
    reference_from_vllm_meta,
    score_model_outputs,
    summarize_completion_lengths,
)
from .run_training_types import (
    GenerationBatch,
    GenerationFn,
    LengthStats,
    ReferenceLogprobs,
    RewardComputation,
    ScoreBatch,
    TrainingLoopContext,
    WeightStats,
)
from .run_training_weighting import compute_weight_stats

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
        :func:`maxent_helpers.run_training_rewards.compute_reward_statistics`.
    :type reward_comp: maxent_helpers.run_training_types.RewardComputation
    :param batch_stats: Auxiliary scoring/weighting artifacts built by
        :func:`_collect_batch_stats`.
    :type batch_stats: _BatchStats
    :param total_input_tokens: Prompt + completion token count used for
        throughput logging.
    :type total_input_tokens: float
    :param scores: Structure containing current-model log-probabilities aligned
        with the reference statistics.
    :type scores: maxent_helpers.run_training_loss.SequenceScores
    """

    grouped_completions: List[List[str]]
    reward_comp: RewardComputation
    batch_stats: _BatchStats
    total_input_tokens: float
    scores: SequenceScores

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
        raise _SkipBatch
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
    :rtype: maxent_helpers.run_training_types.ReferenceLogprobs | None
    """
    if not flat_meta or total_sequences <= 0:
        return None
    return reference_from_vllm_meta(flat_meta, total_sequences, device)


def _collect_batch_stats(
    ctx: TrainingLoopContext,
    gen_batch: GenerationBatch,
    reward_comp: RewardComputation,
) -> Optional[_BatchStats]:
    """Gather scoring, reference, and weighting artifacts for a batch.

    :param ctx: Training loop context supplying runtime/scoring handles.
    :type ctx: maxent_helpers.run_training_types.TrainingLoopContext
    :param gen_batch: Outputs from :func:`prepare_generation_batch`.
    :type gen_batch: maxent_helpers.run_training_types.GenerationBatch
    :param reward_comp: Reward statistics used to build weighting/reward logs.
    :type reward_comp: maxent_helpers.run_training_types.RewardComputation
    :returns: Aggregated structures required downstream, or ``None`` when any
        stage fails (e.g., reference log-prob gathering).
    :rtype: _BatchStats | None
    """
    total_sequences = len(getattr(reward_comp.pairs, "completions", []))
    ref_stats = _reference_stats_from_meta(
        reward_comp.ref_logprob_meta,
        total_sequences,
        ctx.runtime.device,
    )
    score_batch = build_score_batch(
        reward_comp,
        ctx.runtime.tokenizer,
        ctx.generation,
        ctx.scoring.batching,
    )
    if score_batch is None:
        return None
    if (
        ref_stats is None
        and reward_comp.ref_logprob_meta
        and score_batch.total_sequences != total_sequences
    ):
        ref_stats = _reference_stats_from_meta(
            reward_comp.ref_logprob_meta,
            score_batch.total_sequences,
            ctx.runtime.device,
        )
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
    :type ctx: maxent_helpers.run_training_types.TrainingLoopContext
    :param generator: Callable that produces grouped completions (typically
        from :class:`maxent_helpers.run_generation.CompletionGenerator`).
    :type generator: maxent_helpers.run_training_types.GenerationFn
    :param batch: Mini-batch produced by the training dataloader.
    :type batch: dict[str, list[str]]
    :returns: Fully-populated batch artifacts or ``None`` if generation,
        reward computation, reference scoring, or policy scoring fails.
    :rtype: PreparedBatch | None
    """
    try:
        retry_limit = ctx.generation.vllm_rounds_cfg
        if retry_limit <= 0:
            retry_limit = ctx.optimization.schedule.num_generations
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
        cur_logp_sum = _require_artifact(
            score_model_outputs(
                ctx.runtime.model,
                stats.score_batch,
                ctx.scoring.batching,
                ctx.runtime,
            )
        )
        scores = build_sequence_scores(cur_logp_sum, stats.ref_stats)
        return PreparedBatch(
            grouped_completions=gen_batch.grouped_completions,
            reward_comp=reward_comp,
            batch_stats=stats,
            total_input_tokens=stats.prompt_token_count + stats.num_completion_tokens,
            scores=scores,
        )
    except _SkipBatch:
        return None


__all__ = ["PreparedBatch", "prepare_training_batch"]
