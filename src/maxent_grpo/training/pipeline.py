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
from collections.abc import Sized
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeVar, TYPE_CHECKING, cast
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
    reference_stats_from_policy_logprobs,
    reference_from_vllm_meta,
    score_model_outputs,
    summarize_completion_lengths,
    token_counts_from_score_batch,
)
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
    ScoreBatch,
    Tensor,
    TrainingLoopContext,
)
from .weighting import WeightStats, WeightingSettings
from .weighting.logic import compute_weight_stats, build_uniform_weight_stats

if TYPE_CHECKING:
    import torch
    from .weighting.loss import SeedInfoInputs

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


def _deepspeed_zero_stage(accelerator: Any) -> int:
    """Return DeepSpeed ZeRO stage from Accelerate plugin state when present."""
    state = getattr(accelerator, "state", None)
    ds_plugin = getattr(state, "deepspeed_plugin", None)
    try:
        return int(getattr(ds_plugin, "zero_stage", 0) or 0)
    except (TypeError, ValueError):
        return 0


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
    if not flat_meta or total_sequences <= 0:
        return None
    if len(flat_meta) < total_sequences:
        LOG.debug(
            "Behavior log-prob metadata too short | meta_len=%d | sequences=%d",
            len(flat_meta),
            total_sequences,
        )
        return None
    logprob_vals: List[float] = []
    for idx in range(total_sequences):
        entry = flat_meta[idx]
        if entry is None:
            LOG.debug("Behavior log-prob metadata missing entry at idx=%d", idx)
            return None
        logprob_sum = getattr(entry, "logprob_sum", None)
        if logprob_sum is None and isinstance(entry, dict):
            logprob_sum = entry.get("logprob_sum")
        if logprob_sum is None:
            LOG.debug("Behavior log-prob metadata missing logprob_sum at idx=%d", idx)
            return None
        try:
            logprob_vals.append(float(logprob_sum))
        except (TypeError, ValueError):
            LOG.debug(
                "Behavior log-prob metadata has non-castable value at idx=%d: %s",
                idx,
                logprob_sum,
            )
            return None
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
        tensor_fn = getattr(torch_mod, "tensor", None) if torch_mod is not None else None
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


def _collect_batch_stats(
    ctx: TrainingLoopContext,
    gen_batch: GenerationBatch,
    reward_comp: RewardComputation,
    *,
    score_batch: Optional[ScoreBatch] = None,
    cur_logp_sum: Optional[Any] = None,
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
                return len(candidate) == 0  # type: ignore[arg-type]
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
        elif not is_empty and not getattr(_collect_batch_stats, "_ref_diag_logged_success", False):
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
            logp_sum_raw_numel = _safe_numel(getattr(candidate, "ref_logp_sum_raw", None))
            logp_sum_numel = _safe_numel(getattr(candidate, "ref_logp_sum", None))
            if tok_numel and tok_numel > 0 and (
                logp_sum_raw_numel == 0 or logp_sum_numel == 0
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
                )
                LOG.debug(
                    "Retry reference gather result | slice_size=%s | chunk_size=%s | result=%s",
                    reduced_slice,
                    reduced_chunk,
                    _describe_ref(result),
                )
                return result
            except (RuntimeError, ValueError, TypeError, AssertionError) as exc:  # pragma: no cover - best-effort logging
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
    batching_cfg = cast(BatchingSettings, batching_cfg)
    gen_cfg = getattr(ctx, "generation", None)
    if gen_cfg is None:
        gen_cfg = getattr(
            getattr(ctx, "settings", SimpleNamespace()), "generation", None
        )
    gen_cfg = cast(GenerationSettings, gen_cfg)
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
        completion_attention_mask.shape if completion_attention_mask is not None else None,
        getattr(score_batch, "pad_token_id", None),
    )
    ref_meta = getattr(reward_comp, "ref_logprob_meta", None)
    ref_source = str(
        getattr(scoring_cfg, "reference_logprobs_source", "auto") or "auto"
    ).strip().lower()
    force_reference_model = ref_source in {"model", "reference_model", "ref_model", "reference"}
    ref_stats_source = "unknown"
    ref_meta_len = len(ref_meta) if ref_meta else 0
    if ref_meta_len and not force_reference_model:
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
    if ref_stats is None and not force_reference_model and cur_logp_sum is not None:
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
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:  # pragma: no cover - defensive diagnostics
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
        except (ValueError, TypeError, AttributeError) as exc:  # pragma: no cover - defensive diag
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
        _safe_numel(getattr(ref_stats, "ref_logp_sum_raw", None)) if ref_stats is not None else None,
        _safe_numel(getattr(ref_stats, "ref_logp_sum", None)) if ref_stats is not None else None,
        _safe_numel(getattr(ref_stats, "ref_tok_counts", None)) if ref_stats is not None else None,
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
                )
            except (RuntimeError, ValueError, TypeError, AssertionError):
                LOG.warning("Forced minimal reference gather raised an exception; skipping.")
                forced = None
            else:
                LOG.debug(
                    "Forced minimal reference gather result | ref_stats=%s",
                    _describe_ref(forced),
                )
            if not _ref_stats_empty(forced):
                ref_stats = forced
    if _ref_stats_empty(ref_stats) and last_ref_stats is not None:
        allow_stale = bool(
            getattr(scoring_cfg, "allow_stale_reference_logprobs", False)
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
    weighting_cfg = cast(
        WeightingSettings, getattr(scoring_cfg, "weighting", SimpleNamespace())
    )
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
        q_temperature = _resolve_weighting_value(ctx, "q_temperature", 1.0)
        if q_temperature is None:
            q_temperature = 1.0
        q_epsilon = _resolve_weighting_value(ctx, "q_epsilon", 1e-6)
        if q_epsilon is None:
            q_epsilon = 1e-6
        reward_comp = _require_artifact(
            compute_reward_statistics(
                gen_batch,
                ctx.reward,
                ctx.runtime.device,
                q_temperature,
                q_epsilon,
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
        runtime_tokenizer = getattr(ctx.runtime, "tokenizer", None)
        if not callable(runtime_tokenizer):
            # Unit tests often stub `ctx.runtime.tokenizer`; preserve the previous
            # control-flow by letting `_collect_batch_stats` supply the ScoreBatch.
            stats = _require_artifact(
                _collect_batch_stats(ctx, gen_batch, reward_comp),
                stage="batch_stats",
            )
            score_batch = stats.score_batch
            LOG.debug(
                "Batch stats ready | sequences=%d | prompt_tokens=%.0f | completion_tokens=%.0f",
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
                        "Score batch build failed; completions=%d | prompts=%d",
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
            completion_attention_mask = getattr(score_batch, "completion_attention_mask", None)
            LOG.debug(
                "Score batch built | total_sequences=%d | max_prompt_len=%s | slice_size=%s | comp_ids_shape=%s | comp_mask_shape=%s | pad_id=%s",
                getattr(score_batch, "total_sequences", 0),
                getattr(score_batch, "max_prompt_len", None),
                getattr(score_batch, "slice_size", None),
                completion_ids.shape if completion_ids is not None else None,
                completion_attention_mask.shape if completion_attention_mask is not None else None,
                getattr(score_batch, "pad_token_id", None),
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
                    score_batch,
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
            "Policy scoring complete | logprob_shape=%s",
            getattr(logprob_tensor, "shape", None),
        )
        if isinstance(cur_logp_result, tuple):
            cur_logp_sum, pooled_hidden = cur_logp_result
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
                ),
                stage="batch_stats",
            )
            LOG.debug(
                "Batch stats ready | sequences=%d | prompt_tokens=%.0f | completion_tokens=%.0f",
                getattr(getattr(stats, "score_batch", None), "total_sequences", 0),
                stats.prompt_token_count,
                stats.num_completion_tokens,
            )
        behavior_tensor = _behavior_logp_tensor_from_meta(
            getattr(reward_comp, "ref_logprob_meta", None),
            stats.score_batch.total_sequences,
            cur_logp_sum,
        )
        try:
            scores = build_sequence_scores(
                cur_logp_sum,
                stats.ref_stats,
                pooled_hidden,
                behavior_logp_sum=behavior_tensor,
            )
        except TypeError:
            if behavior_tensor is not None:
                try:
                    scores = build_sequence_scores(
                        cur_logp_sum,
                        stats.ref_stats,
                        behavior_logp_sum=behavior_tensor,
                    )
                except TypeError:
                    scores = build_sequence_scores(cur_logp_sum, stats.ref_stats)
            else:
                scores = build_sequence_scores(cur_logp_sum, stats.ref_stats)
        # Optional seed metadata wiring for InfoSeed objectives.
        seed_inputs: Optional["SeedInfoInputs"] = None
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
                    raw_seed = cast(
                        Any, getter("seed_id", -1) if callable(getter) else -1
                    )
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
                    seed_logits_t = cast(Optional["torch.Tensor"], seed_logits)
                    seed_inputs.logits = seed_logits_t
                    # Seed prediction accuracy metrics
                    if seed_logits_t is not None:
                        valid_mask = seed_inputs.seed_ids >= 0
                        if valid_mask.any():
                            preds = seed_logits_t.argmax(dim=-1)
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
                                        heatmap_t = cast("torch.Tensor", heatmap)
                                        seed_heatmap = {
                                            "labels": [
                                                int(s.item()) for s in unique_seeds
                                            ],
                                            "matrix": heatmap_t.detach()
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
        completion_meta = getattr(reward_comp, "completion_metadata", None)
        if isinstance(completion_meta, list):
            weight_stats = getattr(stats, "weight_stats", None)
            weights = (
                getattr(weight_stats, "flat_weights", None) if weight_stats is not None else None
            )
            if weights is not None and len(weights) == len(completion_meta):
                import math

                accelerator = getattr(ctx.runtime, "accelerator", SimpleNamespace(num_processes=1))

                def _gather_weights(weight_subset: list[float]) -> list[float]:
                    if getattr(accelerator, "num_processes", 1) <= 1:
                        return weight_subset
                    gather_fn = getattr(accelerator, "gather_object", None)
                    if callable(gather_fn):
                        gathered = gather_fn(weight_subset)
                        if not isinstance(gathered, list):
                            return weight_subset
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
                for w, meta in zip(weights, completion_meta):
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
        except (AttributeError, TypeError):
            LOG.debug("Failed to record skip stage on runtime.")
        LOG.debug("Skipping training batch: stage=%s returned None", skip_stage)
        return None


__all__ = ["PreparedBatch", "prepare_training_batch"]
