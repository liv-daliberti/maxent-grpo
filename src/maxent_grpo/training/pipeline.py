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
    BatchingSettings,
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
from .weighting.logic import compute_weight_stats, build_uniform_weight_stats

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

    def __init__(self, stage: str):
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
    last_ref_stats = getattr(ctx, "_last_ref_stats", None)

    def _ref_stats_empty(candidate: Optional[ReferenceLogprobs]) -> bool:
        if candidate is None:
            return True
        tensor = getattr(candidate, "ref_logp_sum_raw", None)
        if tensor is None:
            tensor = getattr(candidate, "ref_logp_sum", None)
        if tensor is None:
            try:
                return len(candidate) == 0  # type: ignore[arg-type]
            except TypeError:
                return False
        numel = getattr(tensor, "numel", None)
        if callable(numel):
            try:
                return numel() == 0
            except Exception:
                pass
        to_list = getattr(tensor, "tolist", None)
        data = tensor
        if callable(to_list):
            try:
                data = to_list()
            except Exception:
                data = tensor
        try:
            length = len(data)
        except TypeError:
            return False
        return length == 0

    def _warn_fallback(reason: str) -> None:
        flag = getattr(_collect_batch_stats, "_fallback_warned", False)
        if flag:
            return
        LOG.error(
            "Reference scoring degraded (%s); reusing last cached ReferenceLogprobs.",
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
                return gather_reference_logprobs(
                    score_batch_retry,
                    ctx.runtime,
                    batching_cfg_retry,
                )
            except Exception as exc:  # pragma: no cover - best-effort logging
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
    LOG.debug(
        "Score batch built | total_sequences=%d | max_prompt_len=%s",
        getattr(score_batch, "total_sequences", 0),
        getattr(score_batch, "max_prompt_len", None),
    )
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
    first_ref_attempt = True
    if ref_stats is None:
        ref_try = None
        try:
            ref_try = gather_reference_logprobs(
                score_batch,
                ctx.runtime,
                batching_cfg,
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
        ref_stats = ref_try
    fallback_guard = getattr(_collect_batch_stats, "_fallback_warned", False)
    if _ref_stats_empty(ref_stats):
        retry_stats = _retry_reference_gather(score_batch, batching_cfg)
        if not _ref_stats_empty(retry_stats):
            ref_stats = retry_stats
        elif last_ref_stats is None and not fallback_guard:
            # No cache yet and retry failed: force a minimal slice/chunk attempt.
            fallback_guard = True
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
                )
            except Exception:
                forced = None
            if not _ref_stats_empty(forced):
                ref_stats = forced
    if _ref_stats_empty(ref_stats) and last_ref_stats is not None:
        _warn_fallback("reference gather returned empty tensors")
        ref_stats = last_ref_stats
    if _ref_stats_empty(ref_stats):
        return None
    setattr(ctx, "_last_ref_stats", ref_stats)
    LOG.debug(
        "Reference stats gathered | avg_completion_tokens=%.2f",
        getattr(ref_stats, "avg_completion_tokens", 0.0),
    )

    prompt_token_count = 0.0
    prompt_entries = score_batch.prompt_entries
    if prompt_entries:
        max_prompt_len = score_batch.max_prompt_len
        prompt_token_count = float(
            sum(min(entry.length, max_prompt_len) for entry in prompt_entries)
        )
    weighting_cfg = getattr(scoring_cfg, "weighting", SimpleNamespace())
    weight_stats = compute_weight_stats(
        gen_batch.grouped_completions,
        reward_comp,
        ref_stats,
        weighting_cfg,
    )
    fallback_enabled = bool(
        getattr(weighting_cfg, "allow_empty_weight_fallback", False)
    )
    if weight_stats is None or not getattr(weight_stats, "flat_weights", None):
        fallback_weights = None
        if fallback_enabled:
            fallback_weights = build_uniform_weight_stats(
                gen_batch.grouped_completions
            )
            if fallback_weights is not None:
                LOG.warning(
                    "MaxEnt weighting returned no samples; falling back to uniform GRPO weights for this batch."
                )
        if fallback_weights is not None:
            weight_stats = fallback_weights
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
        LOG.debug(
            "Preparing training batch | prompts=%d | retry_limit=%d",
            len(batch.get("prompt", [])),
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
                    seed_augmentation=getattr(
                        ctx.generation, "seed_augmentation", None
                    ),
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
        group_count = len(getattr(gen_batch, "grouped_completions", []) or [])
        avg_group = (
            sum(len(group) for group in getattr(gen_batch, "grouped_completions", []) or [])
            / max(group_count, 1)
        )
        LOG.debug(
            "Generation complete | grouped_prompts=%d | avg_group_size=%.2f",
            group_count,
            avg_group,
        )
        reward_comp = _require_artifact(
            compute_reward_statistics(
                gen_batch,
                ctx.reward,
                ctx.runtime.device,
                _resolve_weighting_value(ctx, "q_temperature", 1.0),
                _resolve_weighting_value(ctx, "q_epsilon", 1e-6),
                _resolve_weighting_value(ctx, "beta"),
                _resolve_weighting_value(ctx, "tau"),
            ),
            stage="reward_stats",
        )
        reward_mean = float(getattr(getattr(reward_comp, "moments", None), "mean", 0.0))
        reward_std = float(getattr(getattr(reward_comp, "moments", None), "std", 0.0))
        LOG.debug(
            "Reward statistics ready | completions=%d | reward_mean=%.4f | reward_std=%.4f",
            len(getattr(reward_comp.pairs, "completions", []) or []),
            reward_mean,
            reward_std,
        )
        stats = _require_artifact(
            _collect_batch_stats(
                ctx,
                gen_batch,
                reward_comp,
            ),
            stage="batch_stats",
        )
        LOG.debug(
            "Batch stats ready | sequences=%d | prompt_tokens=%.0f | completion_tokens=%.0f",
            getattr(getattr(stats, "score_batch", None), "total_sequences", 0),
            stats.prompt_token_count,
            stats.num_completion_tokens,
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
                ),
                stage="policy_scoring",
            )
        except TypeError:
            cur_logp_result = _require_artifact(
                score_model_outputs(
                    ctx.runtime.model,
                    stats.score_batch,
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
            "Policy scoring complete | logprob_shape=%s",
            getattr(logprob_tensor, "shape", None),
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
                if meta:
                    getter = getattr(meta, "get", None)
                    raw_seed = getter("seed_id", -1) if callable(getter) else -1
                    if raw_seed is None:
                        raw_seed = -1
                    try:
                        seed_ids.append(int(raw_seed))
                    except (TypeError, ValueError):
                        seed_ids.append(-1)
                    raw_aug = getter("is_seed_aug", False) if callable(getter) else False
                    is_seed_aug.append(bool(raw_aug))
                else:
                    seed_ids.append(-1)
                    is_seed_aug.append(False)
            if seed_ids:
                valid_total = sum(1 for sid in seed_ids if sid >= 0)
                aug_total = sum(
                    1 for sid, aug in zip(seed_ids, is_seed_aug) if sid >= 0 and aug
                )
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
                    if seed_inputs.is_seed_aug is not None and valid_total > 0:
                        seed_metrics["seed_aug_frac"] = float(aug_total) / float(
                            valid_total
                        )
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
                                    cosine = getattr(
                                        torch_mod.nn.functional,
                                        "cosine_similarity",
                                        None,
                                    )
                                    if callable(cosine):
                                        heatmap = cosine(
                                            stacked.unsqueeze(1),
                                            stacked.unsqueeze(0),
                                            dim=-1,
                                        )
                                        seed_heatmap = {
                                            "labels": [
                                                int(s.item()) for s in unique_seeds
                                            ],
                                            "matrix": heatmap.detach()
                                            .cpu()
                                            .tolist(),
                                        }
                                    else:
                                        seed_heatmap = None
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
    except _SkipBatch as exc:
        skip_stage = getattr(exc, "stage", "unknown")
        try:
            setattr(ctx.runtime, "_last_skip_stage", skip_stage)
        except Exception:
            pass
        LOG.debug("Skipping training batch: stage=%s returned None", skip_stage)
        return None


__all__ = ["PreparedBatch", "prepare_training_batch"]

# Preserve a self-reference so monkeypatch paths like ``training.pipeline.pipeline``
# resolve even after test shuffling or aliasing.
pipeline = sys.modules[__name__]
# Expose the module under the legacy ``training.pipeline`` alias used in tests.
sys.modules["training.pipeline"] = pipeline
