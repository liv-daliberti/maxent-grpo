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

"""vLLM-specific helpers extracted from the MaxEnt-GRPO generation module."""

from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.vllm_patch import VLLMLogprobResult, safe_generate

from .run_helpers import (
    PROMPT_CHAR_LIMIT,
    _truncate_prompt,
    require_accelerator,
    require_torch,
)

torch = require_torch("generation_vllm")
Accelerator = require_accelerator("generation_vllm")
LOG = logging.getLogger(__name__)


def _optional_import(module_name: str) -> Any:
    """Import a module if available without raising errors downstream."""
    try:
        return __import__(module_name, fromlist=["dummy"])
    except ImportError:
        return None


def _zero3_gather_factory(
    accelerator: Accelerator,
) -> Callable[[Sequence[Any]], Any]:
    """Return a callable that gathers parameters when ZeRO-3 is active."""
    ds_plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    zero_stage = getattr(ds_plugin, "zero_stage", 0) or 0
    gather_cls = None
    if zero_stage == 3:
        deepspeed_mod = _optional_import("deepspeed")
        zero_mod = getattr(deepspeed_mod, "zero", None) if deepspeed_mod else None
        gather_cls = getattr(zero_mod, "GatheredParameters", None)

    if gather_cls is None:
        return lambda _params: nullcontext()

    def _factory(params: Sequence[Any]) -> Any:
        return gather_cls(params)

    return _factory


def _is_peft_model_safe(target: Any) -> bool:
    """Return True if accelerate.utils reports that the model uses PEFT adapters."""
    accelerate_utils = _optional_import("accelerate.utils")
    if accelerate_utils is None:
        return False
    is_peft_model = getattr(accelerate_utils, "is_peft_model", None)
    if not callable(is_peft_model):
        return False
    try:
        return bool(is_peft_model(target))
    except (TypeError, AttributeError, ValueError):
        return False


def _import_vllm_client_cls() -> Optional[type]:
    """Return TRL's VLLMClient class if available."""
    vllm_module = _optional_import("trl.extras.vllm_client")
    if vllm_module is None:
        return None
    return getattr(vllm_module, "VLLMClient", None)


@dataclass
class _VLLMGenerationState:
    """Track state shared across multiple vLLM retries."""

    prompts: List[str]
    target_counts: List[int]
    requested_n: int
    round_limit: int
    track_logprobs: bool
    aggregated: List[List[str]] = None  # type: ignore[assignment]
    aggregated_meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = None

    def __post_init__(self) -> None:
        if len(self.target_counts) != len(self.prompts):
            raise ValueError("target_counts must align with prompts for vLLM state")
        self.aggregated = [[] for _ in self.prompts]
        if self.track_logprobs:
            self.aggregated_meta = [[] for _ in self.prompts]

    def pending_indices(self) -> List[int]:
        pending: List[int] = []
        for idx, (completions, target) in enumerate(
            zip(self.aggregated, self.target_counts)
        ):
            if target > 0 and len(completions) < target:
                pending.append(idx)
        return pending

    def remaining_counts(self, indices: List[int]) -> List[int]:
        counts: List[int] = []
        for idx in indices:
            target = self.target_counts[idx]
            counts.append(max(0, target - len(self.aggregated[idx])))
        return counts

    def trim(
        self,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        trimmed = [
            group[: target]
            for group, target in zip(self.aggregated, self.target_counts)
        ]
        if self.aggregated_meta is None:
            return trimmed, None
        trimmed_meta = [
            meta_group[: target]
            for meta_group, target in zip(self.aggregated_meta, self.target_counts)
        ]
        return trimmed, trimmed_meta

    def drop_meta(self) -> None:
        self.aggregated_meta = None


class _ClientCallable:
    """Lightweight callable wrapper to keep static analyzers satisfied."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)


class VLLMGenerationHelper:
    """Encapsulate vLLM-specific logic so CompletionGenerator stays lean."""

    def __init__(
        self,
        ctx: Any,
        fallback_generate: Callable[[List[str], int, Optional[List[int]]], Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]],
    ) -> None:
        self.ctx = ctx
        self._fallback_generate = fallback_generate
        self._vllm_client: Any = None
        self._vllm_sync_ready = False
        self._last_vllm_synced_step: Optional[int] = None
        self._gather_factory = _zero3_gather_factory(ctx.accelerator)

    # ------------------------------------------------------------------
    # Weight sync helpers
    # ------------------------------------------------------------------

    def _vllm_base_url(self, url: str) -> str:
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.scheme and parsed.netloc:
            base = f"{parsed.scheme}://{parsed.netloc}"
            return base.rstrip("/")
        if "/generate" in url:
            return url.split("/generate", 1)[0].rstrip("/")
        return url.rstrip("/")

    def _ensure_vllm_client(self) -> bool:
        ctx = self.ctx
        if not ctx.vllm_sync_weights or not ctx.accelerator.is_main_process:
            return False
        if self._vllm_client is not None and self._vllm_sync_ready:
            return True
        client_cls = _import_vllm_client_cls()
        if client_cls is None:
            LOG.warning("vLLM weight sync requested but TRL VLLMClient is unavailable; skipping.")
            self._vllm_sync_ready = False
            return False
        try:
            base_url = self._vllm_base_url(ctx.vllm_url)
            self._vllm_client = client_cls(base_url=base_url)
            self._vllm_client.init_communicator()
            self._vllm_sync_ready = True
            return True
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - network dependent
            LOG.warning("Failed to initialize vLLMClient for weight sync: %s", exc)
            self._vllm_client = None
            self._vllm_sync_ready = False
            return False

    def maybe_sync_weights(self) -> None:
        """Synchronize weights to the vLLM server if configured."""
        if not self._ensure_vllm_client():
            return
        current_step = self.ctx.generation_stats.get("current_step")
        if current_step is not None and self._last_vllm_synced_step == int(current_step):
            return
        accelerator = self.ctx.accelerator
        try:
            model = accelerator.unwrap_model(self.ctx.model)
        except (AttributeError, TypeError):
            model = self.ctx.model
        try:
            self._sync_model_params_to_vllm(model)
            stats = self.ctx.generation_stats
            stats["vllm_weight_syncs"] = int(stats.get("vllm_weight_syncs", 0)) + 1
            if current_step is not None:
                self._last_vllm_synced_step = int(current_step)
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - runtime dependent
            LOG.warning("Skipping vLLM weight sync due to error: %s", exc)
        wait_for_all = getattr(accelerator, "wait_for_everyone", None)
        if callable(wait_for_all):
            wait_for_all()

    def _sync_model_params_to_vllm(self, model: Any) -> None:
        """Push model parameters to the vLLM side, handling FSDP/PEFT cases."""
        fsdp_cls = getattr(getattr(torch, "distributed", None), "fsdp", None)
        fsdp_cls = getattr(fsdp_cls, "FullyShardedDataParallel", None) if fsdp_cls else None
        if fsdp_cls is not None and isinstance(model, fsdp_cls):
            self._sync_fsdp_params(model, fsdp_cls)
            self._reset_vllm_cache()
            return
        if _is_peft_model_safe(model):
            self._sync_peft_params(model)
            self._reset_vllm_cache()
            return
        self._sync_standard_params(model)
        self._reset_vllm_cache()

    def _push_param_to_vllm(self, name: str, param: Any) -> None:
        if param is None:
            return
        update_fn = self._client_callable("update_named_param")
        if update_fn is None:
            return
        try:
            update_fn(name, param.data)  # type: ignore[union-attr]
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - network dependent
            LOG.warning("Failed to push param %s to vLLM: %s", name, exc)

    def _reset_vllm_cache(self) -> None:
        reset_fn = self._client_callable("reset_prefix_cache")
        if reset_fn is None:
            return
        try:
            reset_fn()
        except (RuntimeError, ValueError):
            return

    def _sync_standard_params(self, model: Any) -> None:
        params = list(model.parameters())
        with self._gather_factory(params):
            for name, param in model.named_parameters():
                self._push_param_to_vllm(name, param)

    def _sync_fsdp_params(self, module: Any, fsdp_cls: type) -> None:
        visited: set = set()
        for child_name, child in module.named_children():
            child_prefix = child_name
            self._sync_fsdp_params(child, fsdp_cls)
            if isinstance(child, fsdp_cls):
                with fsdp_cls.summon_full_params(child, recurse=False, writeback=False):  # type: ignore[attr-defined]
                    for pname, param in child.named_parameters():
                        full_name = f"{child_prefix}.{pname}"
                        for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                            full_name = full_name.replace(extra, "")
                        if full_name in visited:
                            continue
                        visited.add(full_name)
                        self._push_param_to_vllm(full_name, param)

    def _sync_peft_params(self, model: Any) -> None:
        merge_fn = getattr(model, "merge_adapter", None)
        unmerge_fn = getattr(model, "unmerge_adapter", None)
        params = list(model.parameters())
        with self._gather_factory(params):
            if callable(merge_fn):
                merge_fn()
            for name, param in model.named_parameters():
                clean = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if getattr(model, "prefix", None) and str(model.prefix) in clean:
                    continue
                if "original_module" in clean:
                    continue
                clean = clean.replace("modules_to_save.default.", "")
                self._push_param_to_vllm(clean, param)
            if callable(unmerge_fn):
                unmerge_fn()

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _client_callable(self, attr_name: str) -> Optional[_ClientCallable]:
        """Return a callable attribute from the vLLM client if available."""
        client = self._vllm_client
        if client is None:
            return None
        candidate = getattr(client, attr_name, None)
        if not callable(candidate):
            return None
        return _ClientCallable(candidate)

    def generate(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Generate completions for prompts via vLLM, optionally deduplicating."""
        self.maybe_sync_weights()
        prompts_local, target_counts, mapping = self._prepare_vllm_targets(
            prompts,
            num_samples,
            per_prompt_counts,
        )
        effective_target = max(target_counts) if target_counts else int(num_samples)
        round_limit = self._resolve_vllm_round_limit(max(effective_target, 1))
        state = _VLLMGenerationState(
            prompts=prompts_local,
            target_counts=target_counts,
            requested_n=num_samples,
            round_limit=round_limit,
            track_logprobs=self.ctx.vllm_request_logprobs,
        )
        self._run_vllm_rounds(state)
        grouped, meta = state.trim()
        return self._expand_dedup_results(grouped, meta, mapping)

    def generate_collective(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Broadcast prompts across ranks and gather vLLM generations collectively."""
        flat_prompts, offsets, flat_counts = self._flatten_prompts_for_broadcast(
            prompts,
            per_prompt_counts,
        )
        accelerator = self.ctx.accelerator
        if accelerator.is_main_process:
            grouped_all, meta_all = self.generate(
                flat_prompts,
                num_samples,
                flat_counts,
            )
        else:
            grouped_all = None
            meta_all = None
        return self._scatter_vllm_payload(flat_prompts, offsets, grouped_all, meta_all)

    def _resolve_vllm_round_limit(self, requested_n: int) -> int:
        ctx = self.ctx
        if ctx.vllm_rounds_cfg > 0:
            return max(1, ctx.vllm_rounds_cfg)
        return max(1, requested_n)

    def _prepare_vllm_targets(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]],
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        if per_prompt_counts is not None and len(per_prompt_counts) != len(prompts):
            raise ValueError("per_prompt_counts length must match prompts length")
        target_counts = (
            [max(0, int(count)) for count in per_prompt_counts]
            if per_prompt_counts is not None
            else [max(0, int(num_samples))] * len(prompts)
        )
        dedupe_enabled = os.environ.get("MAXENT_VLLM_DEDUP", "0").lower() in {"1", "true", "yes"}
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

    @staticmethod
    def _expand_dedup_results(
        grouped: List[List[str]],
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        mapping: Optional[List[int]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Expand grouped completions back to match the original prompt ordering."""
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
                if ctx.vllm_retry_sleep > 0:
                    time.sleep(ctx.vllm_retry_sleep)
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
        """Return the maximum prompt length enforced before calling vLLM."""
        limit_override = getattr(self.ctx, "prompt_char_limit", None)
        if isinstance(limit_override, int) and limit_override > 0:
            return limit_override
        return PROMPT_CHAR_LIMIT

    def _request_vllm_batch(
        self,
        pending_prompts: List[str],
        request_count: int,
    ) -> Tuple[
        Optional[List[List[str]]],
        Optional[List[List[Optional[VLLMLogprobResult]]]],
    ]:
        char_limit = self._prompt_char_limit()
        truncated = [_truncate_prompt(prompt, char_limit) for prompt in pending_prompts]
        response = self._invoke_vllm_requests(truncated, request_count)
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
        stats = self.ctx.generation_stats
        stats["vllm_last_latency_ms"] = float(latency_ms)
        stats["vllm_latency_total_ms"] = float(stats.get("vllm_latency_total_ms", 0.0)) + float(latency_ms)
        stats["vllm_latency_calls"] = int(stats.get("vllm_latency_calls", 0)) + 1

    def _build_vllm_request_kwargs(
        self,
        prompts: List[str],
        request_count: int,
    ) -> Dict[str, Any]:
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
        try:
            request_kwargs = self._build_vllm_request_kwargs(prompts, request_count)
            grouped, grouped_meta, latency_ms = safe_generate(**request_kwargs)
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

    # ------------------------------------------------------------------
    # Result post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_grouped(groups: List[List[str]], limit: int = 8) -> str:
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
                stats["vllm_excess_completions"] = stats.get(
                    "vllm_excess_completions",
                    0,
                ) + overflow

    def _backfill_missing(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        ctx = self.ctx
        if not ctx.vllm_backfill_local:
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

    def _record_vllm_failure(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        ctx = self.ctx
        missing_count = len(missing_indices)
        ctx.generation_stats["vllm_failed_prompts"] += missing_count
        suffix = " + local fallback" if ctx.vllm_backfill_local else ""
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

    def _coalesce_grouped_outputs(
        self,
        groups: List[List[str]],
        prompt_count: int,
        requested_n: int,
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
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
                meta[chunk_start : chunk_start + per_prompt] if meta is not None else None
            )
            merged, merged_meta = self._merge_group_chunk(
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
    def _merge_group_chunk(
        chunk: List[List[str]],
        meta_chunk: Optional[List[List[Optional[VLLMLogprobResult]]]],
        requested_n: int,
    ) -> Tuple[List[str], Optional[List[Optional[VLLMLogprobResult]]]]:
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

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------

    def _flatten_prompts_for_broadcast(
        self,
        prompts: List[str],
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        accelerator = self.ctx.accelerator
        gathered = _gather_object_list(accelerator, prompts)
        flat_prompts: List[str] = []
        offsets: List[int] = []
        running = 0
        for group in gathered:
            offsets.append(running)
            running += len(group)
            flat_prompts.extend(group)
        flat_counts: Optional[List[int]] = None
        if per_prompt_counts is not None:
            gathered_counts = _gather_object_list(accelerator, per_prompt_counts)
            flat_counts = []
            for group in gathered_counts:
                flat_counts.extend(int(val) for val in group)
        return flat_prompts, offsets, flat_counts

    def _build_scatter_payload(
        self,
        offsets: List[int],
        world_size: int,
        flat_prompts: List[str],
        grouped_all: Optional[List[List[str]]],
        meta_all: Optional[List[List[Optional[VLLMLogprobResult]]]],
    ) -> List[Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]]:
        """Build payload slices for scatter, trimming to each rank's prompt slice."""
        payload: List[Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]] = []
        total = len(flat_prompts)
        for rank in range(world_size):
            start = offsets[rank]
            end = offsets[rank + 1] if rank + 1 < len(offsets) else total
            slice_grouped = [] if grouped_all is None else grouped_all[start:end]
            slice_meta = None if meta_all is None else meta_all[start:end]
            payload.append((slice_grouped, slice_meta))
        return payload

    def _scatter_vllm_payload(
        self,
        flat_prompts: List[str],
        offsets: List[int],
        grouped_all: Optional[List[List[str]]],
        meta_all: Optional[List[List[Optional[VLLMLogprobResult]]]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        accelerator = self.ctx.accelerator
        world_size = accelerator.num_processes
        if world_size <= 1:
            return self._pluck_rank_outputs(
                grouped_all or [],
                meta_all,
                offsets,
                flat_prompts,
            )

        scatter_payload: Optional[List[Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]]] = None
        if accelerator.is_main_process:
            scatter_payload = self._build_scatter_payload(
                offsets,
                world_size,
                flat_prompts,
                grouped_all,
                meta_all,
            )
        grouped_local, meta_local = _scatter_object(accelerator, scatter_payload, src=0)
        filled_local: List[List[str]] = []
        for group in grouped_local or []:
            filled_local.append(group if group is not None else [])
        return filled_local, meta_local

    def _pluck_rank_outputs(
        self,
        grouped_all: List[List[str]],
        meta_all: Optional[List[List[Optional[VLLMLogprobResult]]]],
        offsets: List[int],
        prompts: List[str],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        accelerator = self.ctx.accelerator
        rank = accelerator.process_index
        start = offsets[rank]
        end = start + len(prompts)
        grouped_local = grouped_all[start:end]
        meta_local = None if meta_all is None else meta_all[start:end]
        filled_local: List[List[str]] = []
        for group in grouped_local:
            filled_local.append(group if group is not None else [])
        return filled_local, meta_local


def _gather_object_list(accelerator: Accelerator, value: List[Any]) -> List[List[Any]]:
    gather_fn = getattr(accelerator, "gather_object", None)
    if callable(gather_fn):
        return gather_fn(value)
    dist = getattr(torch, "distributed", None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered: List[List[str]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, value)
        return gathered
    return [value]


def _scatter_object(
    accelerator: Accelerator,
    input_list: Optional[List[Any]],
    *,
    src: int = 0,
) -> Any:
    if accelerator.num_processes <= 1:
        if input_list is None:
            return None
        return input_list[0]
    scatter_fn = getattr(accelerator, "scatter_object", None)
    if callable(scatter_fn):
        return scatter_fn(
            input_list if accelerator.process_index == src else None,
            src=src,
        )
    dist = getattr(torch, "distributed", None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        output: List[Any] = [None]
        dist.scatter_object_list(
            output,
            input_list if accelerator.process_index == src else None,
            src=src,
        )
        return output[0]
    if input_list is None:
        return None
    return input_list[accelerator.process_index]


__all__ = ["VLLMGenerationHelper"]
