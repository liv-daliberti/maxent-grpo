"""Request/retry helpers separated from vLLM weight sync and scatter logic."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from maxent_grpo.patches.vllm import VLLMLogprobResult, safe_generate
from maxent_grpo.training.runtime.prompts import _truncate_prompt

from .vllm_state import _VLLMGenerationState

_DEFAULT_PROMPT_CHAR_LIMIT = 2048

LOG = logging.getLogger(__name__)


def _resolve_default_limit() -> int:
    """Return the current default prompt character cap from the environment."""
    if _DEFAULT_PROMPT_CHAR_LIMIT <= 0:
        try:
            env_val = os.environ.get("MAX_PROMPT_CHARS")
            if env_val is not None:
                return int(env_val)
        except (TypeError, ValueError):
            pass
        return _DEFAULT_PROMPT_CHAR_LIMIT
    try:
        env_val = os.environ.get("MAX_PROMPT_CHARS")
        if env_val is not None:
            return int(env_val)
    except (TypeError, ValueError):
        pass
    try:
        from maxent_grpo.training.runtime import prompts as prompts_mod

        baseline = getattr(prompts_mod, "PROMPT_CHAR_LIMIT", _DEFAULT_PROMPT_CHAR_LIMIT)
    except (ImportError, AttributeError):
        baseline = _DEFAULT_PROMPT_CHAR_LIMIT
    try:
        return max(int(baseline), int(_DEFAULT_PROMPT_CHAR_LIMIT))
    except (TypeError, ValueError):
        return _DEFAULT_PROMPT_CHAR_LIMIT


class VLLMRequestMixin:
    """Mix-in that isolates request building, retries, and aggregation."""

    ctx: Any
    _safe_generate: Any
    _time: Any
    _fallback_generate: Any

    def set_safe_generate(self, safe_fn: Callable[..., Any]) -> None:
        """Allow callers to override the vLLM ``safe_generate`` hook.

        :param safe_fn: Callable matching the ``safe_generate`` signature.
        :type safe_fn: Callable[..., Any]
        """
        self._safe_generate = safe_fn

    def set_time_provider(self, time_mod: Any) -> None:
        """Allow callers to override the time module for sleep/now calls.

        :param time_mod: Replacement module or object exposing ``sleep`` and
            ``time`` as needed.
        :type time_mod: Any
        """
        self._time = time_mod

    def set_fallback_generate(self, fallback_fn: Callable[..., Any]) -> None:
        """Allow callers to override the local fallback generation hook.

        :param fallback_fn: Callable invoked when vLLM cannot provide outputs.
        :type fallback_fn: Callable[..., Any]
        """
        self._fallback_generate = fallback_fn

    def set_request_executor(
        self, executor_fn: Callable[["_VLLMGenerationState, list[int]"], bool]
    ) -> None:
        """Allow callers to override the vLLM request executor.

        :param executor_fn: Function that performs one vLLM request round for
            pending indices and returns ``True`` on success.
        :type executor_fn: Callable[[maxent_grpo.generation.vllm_state._VLLMGenerationState, list[int]], bool]
        """
        setattr(self, "_execute_vllm_request", executor_fn)

    def set_request_batcher(
        self,
        batcher_fn: Callable[
            [list[str], int],
            Tuple[
                Optional[List[List[str]]],
                Optional[List[List[Optional[VLLMLogprobResult]]]],
            ],
        ],
    ) -> None:
        """Allow callers to override the vLLM batch request helper.

        :param batcher_fn: Callable used to build and dispatch a single vLLM
            request for a list of prompts and a target count.
        :type batcher_fn: Callable[[list[str], int], tuple[list[list[str]] | None, list[list[VLLMLogprobResult | None]] | None]]
        """
        setattr(self, "_request_vllm_batch", batcher_fn)

    def run_vllm_rounds(self, state: _VLLMGenerationState) -> None:
        """Public entry point for executing vLLM retry rounds.

        :param state: Mutable vLLM generation state tracked across retries.
        :type state: _VLLMGenerationState
        """
        self._run_vllm_rounds(state)

    @staticmethod
    def expand_dedup_results(
        grouped: List[List[str]],
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        mapping: Optional[List[int]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Public wrapper for expanding de-duplicated results.

        :param grouped: Grouped completions for unique prompts.
        :type grouped: list[list[str]]
        :param meta: Optional grouped metadata for unique prompts.
        :type meta: list[list[VLLMLogprobResult | None]] | None
        :param mapping: Mapping from original prompt indices to unique indices.
        :type mapping: list[int] | None
        :returns: Grouped completions and metadata expanded to the original
            prompt ordering.
        :rtype: tuple[list[list[str]], list[list[VLLMLogprobResult | None]] | None]
        """
        return VLLMRequestMixin._expand_dedup_results(grouped, meta, mapping)

    def _resolve_vllm_round_limit(self, requested_n: int) -> int:
        ctx = self.ctx
        rounds_cfg = getattr(ctx, "vllm_rounds_cfg", 0) or 0
        if rounds_cfg > 0:
            return max(1, rounds_cfg)
        return max(1, requested_n)

    def _prepare_vllm_targets(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]],
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        """Resolve per-prompt targets and deduplication mapping.

        :param prompts: Original prompt list.
        :type prompts: list[str]
        :param num_samples: Global completion target per prompt.
        :type num_samples: int
        :param per_prompt_counts: Optional per-prompt completion overrides.
        :type per_prompt_counts: list[int] | None
        :returns: Tuple containing prompts to request, target counts per prompt,
            and an optional mapping back to the original indices.
        :rtype: tuple[list[str], list[int], list[int] | None]
        :raises ValueError: If ``per_prompt_counts`` length does not match
            ``prompts`` length.
        """
        if per_prompt_counts is not None and len(per_prompt_counts) != len(prompts):
            raise ValueError("per_prompt_counts length must match prompts length")
        target_counts = (
            [max(0, int(count)) for count in per_prompt_counts]
            if per_prompt_counts is not None
            else [max(0, int(num_samples))] * len(prompts)
        )
        dedupe_enabled = os.environ.get("MAXENT_VLLM_DEDUP", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if not dedupe_enabled:
            return list(prompts), target_counts, None
        seen: Dict[str, int] = {}
        unique_prompts: List[str] = []
        unique_counts: List[int] = []
        mapping: List[int] = []
        for prompt, count in zip(prompts, target_counts):
            if prompt in seen:
                mapping.append(seen[prompt])
                continue
            seen[prompt] = len(unique_prompts)
            mapping.append(seen[prompt])
            unique_prompts.append(prompt)
            unique_counts.append(count)
        return unique_prompts, unique_counts, mapping

    def prepare_vllm_targets(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]],
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        """Public wrapper for resolving vLLM targets/dedup mapping.

        :param prompts: Original prompt list.
        :type prompts: list[str]
        :param num_samples: Global completion target per prompt.
        :type num_samples: int
        :param per_prompt_counts: Optional per-prompt completion overrides.
        :type per_prompt_counts: list[int] | None
        :returns: Tuple of deduplicated prompts, target counts, and mapping back
            to the original order when deduplication occurs.
        :rtype: tuple[list[str], list[int], list[int] | None]
        """
        return self._prepare_vllm_targets(prompts, num_samples, per_prompt_counts)

    @staticmethod
    def _expand_dedup_results(
        grouped: List[List[str]],
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        mapping: Optional[List[int]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Expand grouped completions back to match the original prompt ordering.

        :param grouped: Grouped completions for unique prompts.
        :type grouped: list[list[str]]
        :param meta: Optional grouped metadata for unique prompts.
        :type meta: list[list[VLLMLogprobResult | None]] | None
        :param mapping: Mapping from original prompt indices to unique indices.
        :type mapping: list[int] | None
        :returns: Grouped completions and metadata aligned to the original
            prompt list.
        :rtype: tuple[list[list[str]], list[list[VLLMLogprobResult | None]] | None]
        """
        if mapping is None:
            return grouped, meta
        expanded: List[List[str]] = []
        expanded_meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = (
            [] if meta is not None else None
        )
        for idx in mapping:
            expanded.append(list(grouped[idx]) if idx < len(grouped) else [])
            if expanded_meta is None:
                continue
            expanded_meta.append(
                list(meta[idx]) if meta is not None and idx < len(meta) else []
            )
        return expanded, expanded_meta

    def _run_vllm_rounds(self, state: _VLLMGenerationState) -> None:
        """Execute vLLM request rounds until targets are met or retries are exhausted.

        :param state: Mutable generation state containing prompts, targets, and
            aggregation buffers.
        :type state: _VLLMGenerationState
        """
        ctx = self.ctx
        attempt = 0
        while attempt < state.round_limit:
            pending_indices = state.pending_indices()
            if not pending_indices:
                break
            attempt += 1
            if attempt > 1:
                ctx.generation_stats["vllm_retry_rounds"] += 1
            try:
                success = self._execute_vllm_request(state, pending_indices)
            except RuntimeError as err:
                pending_count = len(pending_indices)
                LOG.warning(
                    "vLLM attempt %d/%d for %d prompts failed: %s",
                    attempt,
                    state.round_limit,
                    pending_count,
                    err,
                )
                if attempt >= state.round_limit:
                    break
                sleep_mod = getattr(self, "_time", time)
                if ctx.vllm_retry_sleep > 0 and hasattr(sleep_mod, "sleep"):
                    sleep_mod.sleep(ctx.vllm_retry_sleep)
                continue
            if not success:
                continue
        missing_indices = state.pending_indices()
        if missing_indices:
            self._backfill_missing(state, missing_indices)
            remaining = state.pending_indices()
            if remaining:
                self._record_vllm_failure(state, remaining)

    def _execute_vllm_request(
        self,
        state: _VLLMGenerationState,
        pending_indices: List[int],
    ) -> bool:
        """Issue batched vLLM requests for the pending prompts.

        :param state: Mutable generation state to update with results.
        :type state: _VLLMGenerationState
        :param pending_indices: Prompt indices that still need completions.
        :type pending_indices: list[int]
        :returns: ``True`` if all batches were accepted by vLLM, ``False`` on a
            recoverable failure.
        :rtype: bool
        """
        remaining_counts = state.remaining_counts(pending_indices)
        grouped_indices: Dict[int, List[int]] = {}
        for prompt_idx, need in zip(pending_indices, remaining_counts):
            if need <= 0:
                continue
            grouped_indices.setdefault(need, []).append(prompt_idx)
        for need, indices in grouped_indices.items():
            pending_prompts = [state.prompts[idx] for idx in indices]
            grouped, grouped_meta = self._request_vllm_batch(
                pending_prompts,
                need,
            )
            if grouped is None:
                return False
            self._merge_vllm_results(
                state,
                grouped,
                grouped_meta,
                indices,
            )
        return True

    def _prompt_char_limit(self) -> int:
        """Return the maximum prompt length enforced before calling vLLM.

        The limit prefers ``ctx.prompt_char_limit`` when set, otherwise derives
        from ``max_prompt_len`` or the static default constant.

        :returns: Maximum number of characters to send per prompt.
        :rtype: int
        """
        base_limit = _resolve_default_limit()
        limit_override = getattr(self.ctx, "prompt_char_limit", None)
        if isinstance(limit_override, int) and limit_override > 0:
            return limit_override
        approx_chars = 0
        max_len = getattr(self.ctx, "max_prompt_len", None)
        if isinstance(max_len, int) and max_len > 0:
            approx_chars = int(max_len * 4)
        try:
            from maxent_grpo.generation import vllm as _vllm_mod

            limit_const = getattr(_vllm_mod, "PROMPT_CHAR_LIMIT", base_limit)
        except (ImportError, AttributeError, RuntimeError):
            limit_const = base_limit
        if limit_const <= 0:
            return approx_chars
        if approx_chars <= 0:
            return limit_const
        return max(limit_const, approx_chars)

    def _request_vllm_batch(
        self,
        pending_prompts: List[str],
        request_count: int,
        invoke_fn: Optional[
            Callable[
                [List[str], int],
                Optional[
                    Tuple[
                        List[List[str]],
                        Optional[List[List[Optional[VLLMLogprobResult]]]],
                        float,
                    ]
                ],
            ]
        ] = None,
    ) -> Tuple[
        Optional[List[List[str]]],
        Optional[List[List[Optional[VLLMLogprobResult]]]],
    ]:
        """Build and dispatch a vLLM batch request for the given prompts.

        :param pending_prompts: Prompts to send to vLLM.
        :type pending_prompts: list[str]
        :param request_count: Number of completions to request per prompt.
        :type request_count: int
        :param invoke_fn: Optional callable to override the actual request
            invocation (useful for testing).
        :type invoke_fn: Callable[[list[str], int], tuple[list[list[str]], list[list[VLLMLogprobResult | None]], float] | None] | None
        :returns: Tuple of grouped completions and optional metadata; ``None``
            values indicate a hard failure that should trigger retries.
        :rtype: tuple[list[list[str]] | None, list[list[VLLMLogprobResult | None]] | None]
        """
        char_limit = self._prompt_char_limit()
        truncated = [_truncate_prompt(prompt, char_limit) for prompt in pending_prompts]
        request_impl = invoke_fn or self._invoke_vllm_requests
        response = request_impl(truncated, request_count)
        if response is None:
            return None, None
        grouped, grouped_meta, latency_ms = response
        self._record_vllm_latency(latency_ms)
        pending_count = len(pending_prompts)
        raw_group_count = len(grouped)
        if raw_group_count != pending_count:
            LOG.warning(
                "vLLM raw groups=%d for %d prompts (req_n=%d) | per-group preview: %s",
                raw_group_count,
                pending_count,
                request_count,
                self._summarize_grouped(grouped),
            )
        grouped, grouped_meta = self._coalesce_grouped_outputs(
            grouped,
            pending_count,
            request_count,
            grouped_meta,
        )
        if len(grouped) == pending_count:
            LOG.warning(
                (
                    "vLLM grouped outputs normalized to %d prompts "
                    "(req_n=%d) | per-prompt lengths=%s"
                ),
                len(grouped),
                request_count,
                [len(entry) for entry in grouped],
            )
            return grouped, grouped_meta
        LOG.warning(
            "vLLM grouped outputs len=%d vs pending=%d | per-prompt lengths=%s",
            len(grouped),
            pending_count,
            [len(entry) for entry in grouped],
        )
        return None, None

    def _record_vllm_latency(self, latency_ms: float) -> None:
        """Record latency stats for a vLLM request round.

        :param latency_ms: Observed latency in milliseconds.
        :type latency_ms: float
        """
        stats = self.ctx.generation_stats
        stats["vllm_last_latency_ms"] = float(latency_ms)
        stats["vllm_latency_total_ms"] = float(
            stats.get("vllm_latency_total_ms", 0.0)
        ) + float(latency_ms)
        stats["vllm_latency_calls"] = int(stats.get("vllm_latency_calls", 0)) + 1

    def _build_vllm_request_kwargs(
        self,
        prompts: List[str],
        request_count: int,
    ) -> Dict[str, Any]:
        """Assemble keyword arguments passed to ``safe_generate``.

        :param prompts: Prompt texts to send.
        :type prompts: list[str]
        :param request_count: Number of completions requested per prompt.
        :type request_count: int
        :returns: Keyword arguments for the vLLM request.
        :rtype: dict[str, Any]
        """
        ctx = self.ctx
        stop_sequences = (
            ctx.gen_stop_sequences
            if ctx.gen_stop_sequences is not None
            else ctx.vllm_stop_sequences
        )
        top_k = ctx.gen_top_k if ctx.gen_top_k is not None else ctx.vllm_top_k
        best_of = ctx.gen_best_of if ctx.gen_best_of is not None else ctx.vllm_best_of
        return {
            "prompts": prompts,
            "url": ctx.vllm_url,
            "max_tokens": ctx.max_completion_len,
            "temperature": ctx.gen_temperature,
            "top_p": ctx.gen_top_p,
            "top_k": top_k,
            "n": request_count,
            "best_of": best_of,
            "frequency_penalty": ctx.gen_frequency_penalty,
            "presence_penalty": ctx.gen_presence_penalty,
            "stop": stop_sequences,
            "logit_bias": ctx.vllm_logit_bias,
            "guided_json": ctx.vllm_guided_json,
            "guided_regex": ctx.vllm_guided_regex,
            "request_id_prefix": ctx.vllm_request_id_prefix,
            "stream": False,
            "tokenizer": ctx.tokenizer,
            "timeout": ctx.vllm_timeout,
            "max_retries": ctx.vllm_max_retries,
            "backoff": ctx.vllm_backoff,
            "return_logprobs": ctx.vllm_request_logprobs,
        }

    def _invoke_vllm_requests(
        self,
        prompts: List[str],
        request_count: int,
    ) -> Optional[
        Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]], float]
    ]:
        """Invoke vLLM requests with recursive fallback on failures.

        :param prompts: Prompt texts to send.
        :type prompts: list[str]
        :param request_count: Number of completions requested per prompt.
        :type request_count: int
        :returns: Tuple of grouped completions, grouped metadata when enabled,
            and latency in milliseconds, or ``None`` if the request fails.
        :rtype: tuple[list[list[str]], list[list[VLLMLogprobResult | None]] | None, float] | None
        """
        try:
            request_kwargs = self._build_vllm_request_kwargs(prompts, request_count)
            safe_gen = getattr(self, "_safe_generate", safe_generate)
            grouped, grouped_meta, latency_ms = safe_gen(**request_kwargs)
            return grouped, grouped_meta, latency_ms
        except RuntimeError as err:
            if len(prompts) <= 1:
                LOG.warning("vLLM request failed for single prompt: %s", err)
                return None
            mid = max(1, len(prompts) // 2)
            left_result = self._invoke_vllm_requests(prompts[:mid], request_count)
            right_result = self._invoke_vllm_requests(prompts[mid:], request_count)
            if left_result is None or right_result is None:
                return None
            combined_meta = None
            if left_result[1] is not None and right_result[1] is not None:
                combined_meta = left_result[1] + right_result[1]
            return (
                left_result[0] + right_result[0],
                combined_meta,
                left_result[2] + right_result[2],
            )

    @staticmethod
    def _summarize_grouped(groups: List[List[str]], limit: int = 8) -> str:
        """Return a concise string summary of grouped completions.

        :param groups: Grouped completions to summarize.
        :type groups: list[list[str]]
        :param limit: Maximum number of groups to include in the summary.
        :type limit: int
        :returns: Human-readable summary suitable for logging.
        :rtype: str
        """
        summary_parts: List[str] = []
        for idx, entry in enumerate(groups[:limit]):
            if isinstance(entry, list):
                preview = entry[0][:32] if entry else ""
                summary_parts.append(f"{idx}:len={len(entry)} sample={preview!r}")
            else:
                summary_parts.append(f"{idx}:type={type(entry).__name__}")
        if len(groups) > limit:
            summary_parts.append(f"...(+{len(groups) - limit})")
        return "; ".join(summary_parts)

    def _merge_vllm_results(
        self,
        state: _VLLMGenerationState,
        grouped: List[List[str]],
        grouped_meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        pending_indices: List[int],
    ) -> None:
        """Merge vLLM responses into the aggregated state with overflow trimming.

        :param state: Mutable generation state to update.
        :type state: _VLLMGenerationState
        :param grouped: Generated completions aligned to ``pending_indices``.
        :type grouped: list[list[str]]
        :param grouped_meta: Optional metadata aligned to ``pending_indices``.
        :type grouped_meta: list[list[VLLMLogprobResult | None]] | None
        :param pending_indices: Prompt indices associated with ``grouped``.
        :type pending_indices: list[int]
        """
        aggregated = state.aggregated
        aggregated_meta = state.aggregated_meta
        stats = self.ctx.generation_stats
        for idx, prompt_idx in enumerate(pending_indices):
            aggregated[prompt_idx].extend(grouped[idx])
            if aggregated_meta is not None and grouped_meta is not None:
                aggregated_meta[prompt_idx].extend(grouped_meta[idx])
            target = state.target_counts[prompt_idx]
            overflow = 0
            if 0 < target < len(aggregated[prompt_idx]):
                overflow = len(aggregated[prompt_idx]) - target
                aggregated[prompt_idx] = aggregated[prompt_idx][:target]
                if aggregated_meta is not None:
                    aggregated_meta[prompt_idx] = aggregated_meta[prompt_idx][:target]
            if overflow > 0:
                stats["vllm_excess_prompts"] = stats.get("vllm_excess_prompts", 0) + 1
                stats["vllm_excess_completions"] = (
                    stats.get(
                        "vllm_excess_completions",
                        0,
                    )
                    + overflow
                )

    def merge_vllm_results(
        self,
        state: _VLLMGenerationState,
        grouped: List[List[str]],
        grouped_meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        pending_indices: List[int],
    ) -> None:
        """Public wrapper for merging generated outputs.

        :param state: Generation state to update.
        :type state: _VLLMGenerationState
        :param grouped: Generated completions aligned to ``pending_indices``.
        :type grouped: list[list[str]]
        :param grouped_meta: Optional metadata aligned to ``pending_indices``.
        :type grouped_meta: list[list[VLLMLogprobResult | None]] | None
        :param pending_indices: Prompt indices associated with the provided
            completions.
        :type pending_indices: list[int]
        """
        self._merge_vllm_results(state, grouped, grouped_meta, pending_indices)

    def _backfill_missing(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        """Generate missing completions locally when vLLM fails.

        :param state: Mutable generation state.
        :type state: _VLLMGenerationState
        :param missing_indices: Prompt indices still missing completions.
        :type missing_indices: list[int]
        """
        ctx = self.ctx
        backfill_enabled = bool(getattr(ctx, "vllm_backfill_local", False))
        if not backfill_enabled:
            return
        missing_prompts = [state.prompts[idx] for idx in missing_indices]
        needed_per_prompt = state.remaining_counts(missing_indices)
        ctx.generation_stats["vllm_backfilled_prompts"] += len(missing_indices)
        max_need = max(needed_per_prompt) if needed_per_prompt else 0
        LOG.warning(
            (
                "Backfilling %d/%d prompts locally because vLLM failed to "
                "return the remaining completions (max_need=%d) after %d attempts."
            ),
            len(missing_indices),
            len(state.prompts),
            max_need,
            state.round_limit,
        )
        local_groups, _ = self._fallback_generate(
            missing_prompts,
            state.requested_n,
            needed_per_prompt,
        )
        aggregated = state.aggregated
        for local_idx, prompt_idx in enumerate(missing_indices):
            target = state.target_counts[prompt_idx]
            needed = max(0, target - len(aggregated[prompt_idx]))
            if needed <= 0:
                continue
            aggregated[prompt_idx].extend(local_groups[local_idx][:needed])
        state.drop_meta()

    def backfill_missing(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        """Public wrapper for local fallback generation.

        :param state: Generation state to update.
        :type state: _VLLMGenerationState
        :param missing_indices: Prompt indices still missing completions.
        :type missing_indices: list[int]
        """
        self._backfill_missing(state, missing_indices)

    def _record_vllm_failure(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        """Record metrics and warnings when vLLM could not satisfy requests.

        :param state: Generation state containing prompt counts.
        :type state: _VLLMGenerationState
        :param missing_indices: Indices that remain incomplete.
        :type missing_indices: list[int]
        """
        ctx = self.ctx
        missing_count = len(missing_indices)
        ctx.generation_stats["vllm_failed_prompts"] += missing_count
        backfill_enabled = bool(getattr(ctx, "vllm_backfill_local", False))
        suffix = " + local fallback" if backfill_enabled else ""
        remaining = state.remaining_counts(missing_indices)
        max_need = max(remaining) if remaining else 0
        LOG.warning(
            (
                "Unable to obtain the remaining completions (max_need=%d) for %d/%d prompts even "
                "after %d attempts%s."
            ),
            max_need,
            missing_count,
            len(state.prompts),
            state.round_limit,
            suffix,
        )

    def record_vllm_failure(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        """Public wrapper for reporting vLLM failures.

        :param state: Generation state containing prompt counts.
        :type state: _VLLMGenerationState
        :param missing_indices: Indices that remain incomplete.
        :type missing_indices: list[int]
        """
        self._record_vllm_failure(state, missing_indices)

    @staticmethod
    def _coalesce_grouped_outputs(
        groups: List[List[str]],
        prompt_count: int,
        requested_n: int,
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Normalize grouped outputs when vLLM returns multiple slices per prompt.

        :param groups: Raw grouped completions returned by vLLM.
        :type groups: list[list[str]]
        :param prompt_count: Number of prompts originally requested.
        :type prompt_count: int
        :param requested_n: Target completions per prompt.
        :type requested_n: int
        :param meta: Optional grouped metadata aligned with ``groups``.
        :type meta: list[list[VLLMLogprobResult | None]] | None
        :returns: Regrouped completions and metadata aligned to prompts. If
            regrouping is not possible, metadata may be dropped.
        :rtype: tuple[list[list[str]], list[list[VLLMLogprobResult | None]] | None]
        """
        if not groups or prompt_count <= 0:
            return groups, meta
        total_groups = len(groups)
        if total_groups == prompt_count:
            return groups, meta
        if total_groups % prompt_count != 0:
            return groups, None
        per_prompt = total_groups // prompt_count
        if (
            per_prompt <= 1
            or (requested_n > 0 and per_prompt != requested_n)
            or not all(len(entry) <= 1 for entry in groups)
        ):
            return groups, meta
        regrouped: List[List[str]] = []
        regrouped_meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = (
            [] if meta is not None else None
        )
        for chunk_start in range(0, total_groups, per_prompt):
            chunk = groups[chunk_start : chunk_start + per_prompt]
            meta_slice = (
                meta[chunk_start : chunk_start + per_prompt]
                if meta is not None
                else None
            )
            merged, merged_meta = VLLMRequestMixin._merge_group_chunk(
                chunk,
                meta_slice,
                requested_n,
            )
            regrouped.append(merged)
            if regrouped_meta is None:
                continue
            regrouped_meta.append(merged_meta if merged_meta is not None else [])
        return regrouped, regrouped_meta

    @staticmethod
    def coalesce_grouped_outputs(
        groups: List[List[str]],
        prompt_count: int,
        requested_n: int,
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Public wrapper for regrouping vLLM outputs.

        :param groups: Raw grouped completions returned by vLLM.
        :type groups: list[list[str]]
        :param prompt_count: Number of prompts originally requested.
        :type prompt_count: int
        :param requested_n: Target completions per prompt.
        :type requested_n: int
        :param meta: Optional grouped metadata aligned with ``groups``.
        :type meta: list[list[VLLMLogprobResult | None]] | None
        :returns: Regrouped completions and metadata aligned to prompts. If
            regrouping is not possible, metadata may be dropped.
        :rtype: tuple[list[list[str]], list[list[VLLMLogprobResult | None]] | None]
        """
        return VLLMRequestMixin._coalesce_grouped_outputs(
            groups, prompt_count, requested_n, meta
        )

    @staticmethod
    def _merge_group_chunk(
        chunk: List[List[str]],
        meta_chunk: Optional[List[List[Optional[VLLMLogprobResult]]]],
        requested_n: int,
    ) -> Tuple[List[str], Optional[List[Optional[VLLMLogprobResult]]]]:
        """Merge a contiguous chunk of grouped outputs for a single prompt.

        :param chunk: Subset of grouped outputs belonging to one prompt.
        :type chunk: list[list[str]]
        :param meta_chunk: Optional metadata aligned to ``chunk``.
        :type meta_chunk: list[list[VLLMLogprobResult | None]] | None
        :param requested_n: Target number of completions for the prompt.
        :type requested_n: int
        :returns: Flattened completions and optional flattened metadata trimmed
            to ``requested_n``.
        :rtype: tuple[list[str], list[VLLMLogprobResult | None] | None]
        """
        merged: List[str] = []
        merged_meta: Optional[List[Optional[VLLMLogprobResult]]] = (
            [] if meta_chunk is not None else None
        )
        for idx, entry in enumerate(chunk):
            merged.extend(entry)
            if merged_meta is None:
                continue
            if meta_chunk is None or idx >= len(meta_chunk):
                merged_meta = None
                continue
            merged_meta.extend(meta_chunk[idx])
        if requested_n > 0:
            merged = merged[:requested_n]
            if merged_meta is not None:
                merged_meta = merged_meta[:requested_n]
        return merged, merged_meta

    @staticmethod
    def merge_group_chunk(
        chunk: List[List[str]],
        meta_chunk: Optional[List[List[Optional[VLLMLogprobResult]]]],
        requested_n: int,
    ) -> Tuple[List[str], Optional[List[Optional[VLLMLogprobResult]]]]:
        """Public wrapper for merging grouped chunks.

        :param chunk: Subset of grouped outputs belonging to one prompt.
        :type chunk: list[list[str]]
        :param meta_chunk: Optional metadata aligned to ``chunk``.
        :type meta_chunk: list[list[VLLMLogprobResult | None]] | None
        :param requested_n: Target number of completions for the prompt.
        :type requested_n: int
        :returns: Flattened completions and optional flattened metadata trimmed
            to ``requested_n``.
        :rtype: tuple[list[str], list[VLLMLogprobResult | None] | None]
        """
        return VLLMRequestMixin._merge_group_chunk(chunk, meta_chunk, requested_n)


__all__ = ["VLLMRequestMixin"]
