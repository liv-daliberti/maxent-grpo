"""vLLM-focused helpers split away from the local generation path."""

from __future__ import annotations

import importlib
import sys
import logging
import time
from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from maxent_grpo.generation.common import (
    AggregatedGenerationState as _AggregatedGenerationState,
    retry_incomplete_prompts as _retry_incomplete_prompts_impl,
    seed_generation_groups as _seed_generation_groups_impl,
)
from maxent_grpo.generation.vllm import (
    VLLMGenerationHelper,
    _VLLMGenerationState as _BaseVLLMGenerationState,
)
from maxent_grpo.generation.vllm_utils import (
    import_vllm_client_cls as _shared_import_vllm_client_cls,
    zero3_gather_factory as _shared_zero3_gather_factory,
)
from maxent_grpo.patches.vllm import VLLMLogprobResult, safe_generate
from maxent_grpo.training.runtime import require_accelerator, require_torch
from maxent_grpo.training.runtime.prompts import _truncate_prompt
from maxent_grpo.utils.fallbacks import dist_with_fallback

from .context import GenerationContext

torch = require_torch("generation")
Accelerator = require_accelerator("generation")
dist = dist_with_fallback(getattr(torch, "distributed", None))
LOG = logging.getLogger(__name__)


def _optional_import(module_name: str) -> Any:
    """Import a module if available without triggering import errors."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _zero3_gather_factory(
    accelerator: Accelerator,
) -> Callable[[Sequence[Any]], AbstractContextManager[Any]]:
    return _shared_zero3_gather_factory(accelerator, import_fn=_optional_import)


def _import_vllm_client_cls(
    import_fn: Optional[Callable[[str], Any]] = None,
) -> Optional[type]:
    """Return TRL's VLLMClient using the provided import fn (defaults to optional_import)."""

    return _shared_import_vllm_client_cls(import_fn or _optional_import)


def _is_peft_model_safe(target: Any) -> bool:
    """Return True if accelerate.utils reports that the model uses PEFT adapters."""
    accel_utils = _optional_import("accelerate.utils")
    if accel_utils is None:
        return False
    is_peft_model = getattr(accel_utils, "is_peft_model", None)
    if not callable(is_peft_model):
        return False
    try:
        return bool(is_peft_model(target))
    except (TypeError, AttributeError, ValueError):
        return False


_VLLMGenerationState = _BaseVLLMGenerationState


class VLLMGenerationMixin:
    """All vLLM-specific plumbing extracted from the main generator."""

    # Access to helper internals is intentional for tests/patching.
    # pylint: disable=protected-access

    ctx: GenerationContext

    def __init__(self, ctx: GenerationContext) -> None:
        self.ctx = ctx
        self._vllm_helper = VLLMGenerationHelper(ctx, self._generate_local)
        # Surface patchable hooks for tests so monkeypatched helpers.* propagate.
        if hasattr(self._vllm_helper, "set_safe_generate"):
            self._vllm_helper.set_safe_generate(safe_generate)
        else:
            self._vllm_helper._safe_generate = (
                safe_generate  # pragma: no cover - legacy stubs
            )
        self._vllm_helper._scatter_object = _scatter_object
        if hasattr(self._vllm_helper, "set_time_provider"):
            self._vllm_helper.set_time_provider(time)
        else:
            self._vllm_helper._time = time  # pragma: no cover - legacy stubs
        self._vllm_helper._is_peft_model_safe = _is_peft_model_safe
        if hasattr(self._vllm_helper, "set_fallback_generate"):
            self._vllm_helper.set_fallback_generate(self._generate_local)
        else:
            self._vllm_helper._fallback_generate = (
                self._generate_local
            )  # pragma: no cover - legacy stubs

    @property
    def _vllm_client(self):
        return self._vllm_helper._vllm_client

    @_vllm_client.setter
    def _vllm_client(self, value) -> None:
        self._vllm_helper._vllm_client = value

    @property
    def _vllm_sync_ready(self) -> bool:
        return self._vllm_helper._vllm_sync_ready

    @_vllm_sync_ready.setter
    def _vllm_sync_ready(self, value: bool) -> None:
        self._vllm_helper._vllm_sync_ready = value

    @property
    def _last_vllm_synced_step(self) -> Optional[int]:
        return self._vllm_helper._last_vllm_synced_step

    @_last_vllm_synced_step.setter
    def _last_vllm_synced_step(self, value: Optional[int]) -> None:
        self._vllm_helper._last_vllm_synced_step = value

    @property
    def _fsdp_cls(self):
        return self._vllm_helper._fsdp_cls

    @_fsdp_cls.setter
    def _fsdp_cls(self, value) -> None:
        self._vllm_helper._fsdp_cls = value

    def _vllm_base_url(self, url: str) -> str:
        """Delegate to the shared vLLM helper to normalize the base URL."""
        return self._vllm_helper._vllm_base_url(url)

    def _ensure_vllm_client(self) -> bool:
        """Instantiate the TRL VLLMClient when weight sync is enabled."""
        try:
            helpers_mod = sys.modules.get(
                type(self).__module__
            ) or importlib.import_module("maxent_grpo.training.generation.helpers")
        except ImportError:
            helpers_mod = None
        import_fn = getattr(
            self,
            "_import_vllm_client_cls",
            getattr(helpers_mod, "_import_vllm_client_cls", _import_vllm_client_cls),
        )
        ctx = self.ctx
        if not getattr(ctx, "vllm_sync_weights", False) or not getattr(
            ctx.accelerator, "is_main_process", False
        ):
            return False
        if (
            self._vllm_helper._vllm_client is not None
            and self._vllm_helper._vllm_sync_ready
        ):
            return True
        client_cls = import_fn()
        if client_cls is None or not callable(client_cls):
            self._vllm_helper._vllm_sync_ready = False
            return False
        try:
            base_url = self._vllm_helper._vllm_base_url(ctx.vllm_url)
            try:
                client = client_cls(base_url=base_url)
            except TypeError:
                client = client_cls()
            init = getattr(client, "init_communicator", None)
            if callable(init):
                init()
            self._vllm_helper._vllm_client = client
            self._vllm_helper._vllm_sync_ready = True
            return True
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ):  # pragma: no cover - defensive
            self._vllm_helper._vllm_client = client_cls  # best-effort marker
            self._vllm_helper._vllm_sync_ready = True
            return True

    def _maybe_sync_vllm_weights(self) -> None:
        """Push current model weights to the vLLM server."""
        accelerator = self.ctx.accelerator
        try:
            self._vllm_helper.maybe_sync_weights(
                ensure_client=self._ensure_vllm_client,
                sync_model=lambda model: self._sync_model_params_to_vllm(
                    model, accelerator
                ),
            )
        except TypeError:
            # Allow lightweight stubs without keyword support.
            self._vllm_helper.maybe_sync_weights()

    def _sync_model_params_to_vllm(
        self,
        model: Any,
        accelerator: Accelerator,
    ) -> None:
        """Best-effort parameter broadcast mirroring HF GRPO's vLLM path."""
        del accelerator  # handled internally by the shared helper
        self._vllm_helper._sync_model_params_to_vllm(model)

    def _push_param_to_vllm(self, name: str, param: Any) -> None:
        """Send a single parameter tensor to the vLLM client."""
        self._vllm_helper._push_param_to_vllm(name, param)

    def _reset_vllm_cache(self) -> None:
        """Reset prefix caches when the vLLM client exposes the helper."""
        self._vllm_helper._reset_vllm_cache()

    def _sync_fsdp_params(self, model: Any) -> None:
        """Iterate FSDP shards and push full parameters to vLLM."""
        self._vllm_helper._sync_fsdp_params(model)

    def _sync_peft_params(
        self,
        model: Any,
        gather_factory: Callable[[Sequence[Any]], AbstractContextManager[Any]],
    ) -> None:
        """Push merged PEFT adapter weights to vLLM."""
        self._vllm_helper._sync_peft_params(model, gather_factory)

    def _sync_standard_params(
        self,
        model: Any,
        gather_factory: Callable[[Sequence[Any]], AbstractContextManager[Any]],
    ) -> None:
        """Push standard (non-PEFT/FSDP) parameters to vLLM."""
        self._vllm_helper._sync_standard_params(model, gather_factory)

    def _resolve_vllm_round_limit(self, requested_n: int) -> int:
        """Decide how many vLLM rounds to run for the current request."""
        return self._vllm_helper._resolve_vllm_round_limit(requested_n)

    @staticmethod
    def _seed_generation_groups(
        prompt_count: int,
        grouped_comps: Optional[List[List[str]]],
        grouped_meta: Optional[List[List[Optional[Any]]]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
        """Compatibility wrapper for older tests expecting this helper."""
        return _seed_generation_groups_impl(prompt_count, grouped_comps, grouped_meta)

    @staticmethod
    def _retry_incomplete_prompts(
        helper: "VLLMGenerationMixin",
        prompts: List[str],
        generator: Callable[
            [List[str], int, Optional[List[int]]],
            Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]],
        ],
        expected_generations: int,
        aggregated_comps: List[List[str]],
        aggregated_meta: Optional[List[List[Optional[Any]]]],
        max_retry_rounds: Optional[int],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
        """Retry helpers retained for backwards compatibility with older tests."""
        del helper  # helper is unused but kept for signature compatibility.
        state = _AggregatedGenerationState(aggregated_comps, aggregated_meta)
        updated = _retry_incomplete_prompts_impl(
            prompts,
            generator,
            expected_generations,
            state,
            max_retry_rounds,
        )
        return updated.completions, updated.metadata

    @staticmethod
    def _summarize_grouped(groups: List[List[str]], limit: int = 8) -> str:
        """Return a compact preview of grouped completions."""
        return VLLMGenerationHelper._summarize_grouped(groups, limit)

    def _request_vllm_batch(
        self,
        pending_prompts: List[str],
        request_count: int,
    ) -> Tuple[
        Optional[List[List[str]]],
        Optional[List[List[Optional[VLLMLogprobResult]]]],
    ]:
        """Request completions from vLLM for a subset of prompts."""
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
        coalesce_fn = getattr(
            self._vllm_helper,
            "_coalesce_grouped_outputs",
            self._coalesce_grouped_outputs,
        )
        grouped, grouped_meta = coalesce_fn(
            grouped,
            pending_count,
            request_count,
            meta=grouped_meta,
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
        """Track latency metrics for successful vLLM invocations."""
        self._vllm_helper._record_vllm_latency(latency_ms)

    def _build_vllm_request_kwargs(
        self,
        prompts: List[str],
        request_count: int,
    ) -> Dict[str, Any]:
        """Assemble keyword arguments for ``safe_generate`` requests."""
        return self._vllm_helper._build_vllm_request_kwargs(prompts, request_count)

    def _invoke_vllm_requests(
        self,
        prompts: List[str],
        request_count: int,
    ) -> Optional[
        Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]], float]
    ]:
        """Call vLLM with retries by splitting large prompt batches."""
        try:
            helpers_mod = sys.modules.get(
                type(self).__module__
            ) or importlib.import_module("maxent_grpo.training.generation.helpers")
        except ImportError:
            helpers_mod = None
        safe_gen = getattr(helpers_mod, "safe_generate", safe_generate)
        set_safe = getattr(self._vllm_helper, "set_safe_generate", None)
        if callable(set_safe):
            set_safe(safe_gen)
        else:
            self._vllm_helper._safe_generate = (
                safe_gen  # pragma: no cover - legacy stubs
            )
        set_time = getattr(self._vllm_helper, "set_time_provider", None)
        if callable(set_time):
            set_time(getattr(helpers_mod, "time", time))
        else:
            self._vllm_helper._time = getattr(
                helpers_mod, "time", time
            )  # pragma: no cover - legacy stubs
        return self._vllm_helper._invoke_vllm_requests(prompts, request_count)

    def _merge_vllm_results(
        self,
        state: _VLLMGenerationState,
        grouped: List[List[str]],
        grouped_meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        pending_indices: List[int],
    ) -> None:
        """Append vLLM outputs into the shared state aggregates."""
        self._vllm_helper.merge_vllm_results(
            state,
            grouped,
            grouped_meta,
            pending_indices,
        )

    def _backfill_missing(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        """Generate missing completions locally when vLLM under-delivers."""
        self._vllm_helper.set_fallback_generate(self._generate_local)
        self._vllm_helper.backfill_missing(state, missing_indices)

    def _record_vllm_failure(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        """Log a warning when vLLM fails to deliver even after retries/backfill."""
        self._vllm_helper.record_vllm_failure(state, missing_indices)

    # pylint: enable=protected-access

    @staticmethod
    def _coalesce_grouped_outputs(
        groups: List[List[str]],
        prompt_count: int,
        requested_n: int,
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Normalize grouped outputs when vLLM returns per-sample lists."""
        return VLLMGenerationHelper.coalesce_grouped_outputs(
            groups, prompt_count, requested_n, meta
        )

    @staticmethod
    def _merge_group_chunk(
        chunk: List[List[str]],
        meta_chunk: Optional[List[List[Optional[VLLMLogprobResult]]]],
        requested_n: int,
    ) -> Tuple[List[str], Optional[List[Optional[VLLMLogprobResult]]]]:
        """Merge consecutive micro-groups back into per-prompt lists."""
        return VLLMGenerationHelper.merge_group_chunk(chunk, meta_chunk, requested_n)

    def _prepare_vllm_targets(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]],
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        """Resolve target counts and optional dedup mapping for vLLM."""
        return self._vllm_helper.prepare_vllm_targets(
            prompts, num_samples, per_prompt_counts
        )

    def _run_vllm_rounds(self, state: _VLLMGenerationState) -> None:
        """Iteratively request completions until targets are satisfied."""
        # pylint: disable=protected-access
        try:
            helpers_mod = sys.modules.get(type(self).__module__)
            if helpers_mod is None or not hasattr(helpers_mod, "time"):
                helpers_mod = importlib.import_module(
                    "maxent_grpo.training.generation.helpers"
                )
        except ImportError:
            helpers_mod = helpers_mod if "helpers_mod" in locals() else None
        set_time = getattr(self._vllm_helper, "set_time_provider", None)
        if callable(set_time):
            set_time(getattr(helpers_mod, "time", time))
        else:
            self._vllm_helper._time = getattr(
                helpers_mod, "time", time
            )  # pragma: no cover - legacy stubs
        # Allow monkeypatched generator hooks to propagate into the helper.
        helper_exec = getattr(self._vllm_helper, "_execute_vllm_request", None)
        if (
            getattr(helper_exec, "__func__", helper_exec)
            is VLLMGenerationHelper._execute_vllm_request
        ):
            set_exec = getattr(self._vllm_helper, "set_request_executor", None)
            if callable(set_exec):
                set_exec(self._execute_vllm_request)
            else:
                self._vllm_helper._execute_vllm_request = (
                    self._execute_vllm_request
                )  # pragma: no cover - legacy stubs
        helper_batch = getattr(self._vllm_helper, "_request_vllm_batch", None)
        if (
            getattr(helper_batch, "__func__", helper_batch)
            is VLLMGenerationHelper._request_vllm_batch
        ):
            set_batcher = getattr(self._vllm_helper, "set_request_batcher", None)
            if callable(set_batcher):
                set_batcher(self._request_vllm_batch)
            else:
                self._vllm_helper._request_vllm_batch = (
                    self._request_vllm_batch
                )  # pragma: no cover - legacy stubs
        set_fallback = getattr(self._vllm_helper, "set_fallback_generate", None)
        if callable(set_fallback):
            set_fallback(self._generate_local)
        else:
            self._vllm_helper._fallback_generate = self._generate_local  # pragma: no cover - legacy stubs
        run_rounds = getattr(self._vllm_helper, "run_vllm_rounds", None)
        if callable(run_rounds):
            run_rounds(state)
        else:
            self._vllm_helper._run_vllm_rounds(state)  # pragma: no cover - legacy stubs
        # pylint: enable=protected-access

    @staticmethod
    def _expand_dedup_results(
        grouped: List[List[str]],
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        mapping: Optional[List[int]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Expand de-duplicated prompts back to the original ordering."""
        return VLLMGenerationHelper.expand_dedup_results(grouped, meta, mapping)

    def _generate_with_vllm(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Generate completions via vLLM, with dedupe/backoff handling."""
        if not prompts:
            return [], None
        # Keep prompt truncation aligned with the legacy helper implementation.
        self.ctx.prompt_char_limit = self._prompt_char_limit()
        accelerator = self.ctx.accelerator
        set_fallback = getattr(self._vllm_helper, "set_fallback_generate", None)
        if callable(set_fallback):
            set_fallback(self._generate_local)
        else:
            setattr(self._vllm_helper, "_fallback_generate", self._generate_local)
        generate_fn = getattr(self._vllm_helper, "generate", None)
        if not callable(generate_fn):
            return [], None
        return generate_fn(
            prompts,
            num_samples,
            per_prompt_counts,
            ensure_client=self._ensure_vllm_client,
            sync_model=lambda model: self._sync_model_params_to_vllm(
                model, accelerator
            ),
        )

    def _execute_vllm_request(
        self,
        state: _VLLMGenerationState,
        pending_indices: List[int],
    ) -> bool:
        """Request completions for specific prompts, grouped by need bucket."""
        return getattr(self._vllm_helper, "_execute_vllm_request")(state, pending_indices)

    def _flatten_prompts_for_broadcast(
        self,
        prompts: List[str],
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        return getattr(self._vllm_helper, "_flatten_prompts_for_broadcast")(
            prompts, per_prompt_counts
        )

    def _broadcast_vllm_payload(
        self,
        flat_prompts: List[str],
        payload: List[Any],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        _broadcast_object_list(self.ctx.accelerator, payload, src=0)
        grouped_all, meta_all = payload
        if grouped_all is None:
            grouped_all = [[] for _ in flat_prompts]
        return grouped_all, meta_all

    def _scatter_vllm_payload(
        self,
        flat_prompts: List[str],
        offsets: List[int],
        grouped_all: Optional[List[List[str]]],
        meta_all: Optional[List[List[Optional[VLLMLogprobResult]]]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Scatter per-rank slices instead of broadcasting full completions."""
        return getattr(self._vllm_helper, "_scatter_vllm_payload")(
            flat_prompts,
            offsets,
            grouped_all,
            meta_all,
        )

    def _pluck_rank_outputs(
        self,
        grouped_all: List[List[str]],
        meta_all: Optional[List[List[Optional[VLLMLogprobResult]]]],
        offsets: List[int],
        prompts: List[str],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        return getattr(self._vllm_helper, "_pluck_rank_outputs")(
            grouped_all,
            meta_all,
            offsets,
            prompts,
        )

    def _generate_vllm_collective(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Run vLLM once on rank 0 and scatter results back to all ranks."""
        self.ctx.prompt_char_limit = self._prompt_char_limit()
        accelerator = self.ctx.accelerator
        if getattr(accelerator, "num_processes", 1) <= 1:
            return self._generate_with_vllm(prompts, num_samples, per_prompt_counts)
        flat_prompts, offsets, flat_counts = self._flatten_prompts_for_broadcast(
            prompts,
            per_prompt_counts,
        )
        if accelerator.is_main_process:
            grouped_all, meta_all = self._generate_with_vllm(
                flat_prompts,
                num_samples,
                flat_counts,
            )
        else:
            grouped_all = None
            meta_all = None
        scatter_result = self._scatter_vllm_payload(
            flat_prompts, offsets, grouped_all, meta_all
        )
        if isinstance(scatter_result, tuple):
            if len(scatter_result) != 2:
                grouped_res, meta_res = [], None
            else:
                grouped_res = scatter_result[0]
                meta_res = scatter_result[1]
        else:
            grouped_res, meta_res = scatter_result, None
        if grouped_res is None:
            grouped_res = [[] for _ in prompts]
        return grouped_res, meta_res

    def generate(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Produce completions, preferring vLLM when configured."""
        if not prompts:
            return [], None
        if per_prompt_counts is not None and len(per_prompt_counts) != len(prompts):
            raise ValueError(
                "per_prompt_counts length must match prompts length in generate()"
            )
        if self.ctx.use_vllm:
            return self._generate_vllm_collective(
                prompts, num_samples, per_prompt_counts
            )
        return self._generate_local(prompts, num_samples, per_prompt_counts)


def _gather_object_list(accelerator: Accelerator, value: List[Any]) -> List[List[Any]]:
    """Gather Python lists across ranks with graceful Accelerate fallbacks."""
    gather_fn = getattr(accelerator, "gather_object", None)
    if callable(gather_fn):
        return gather_fn(value)
    if dist is not None and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered: List[List[str]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, value)
        return gathered
    # Single-process fallback
    return [value]


def _broadcast_object_list(
    accelerator: Accelerator, payload: List[Any], *, src: int = 0
) -> None:
    """Broadcast python objects even when Accelerate lacks the helper."""
    broadcast_fn = getattr(accelerator, "broadcast_object_list", None)
    if callable(broadcast_fn):
        broadcast_fn(payload, src=src)
        return
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=src)


def _scatter_object(
    accelerator: Accelerator,
    input_list: Optional[List[Any]],
    *,
    src: int = 0,
) -> Any:
    """Scatter python objects from src to all ranks."""
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
    if dist is not None and dist.is_available() and dist.is_initialized():
        output: List[Any] = [None]
        dist.scatter_object_list(
            output,
            input_list if accelerator.process_index == src else None,
            src=src,
        )
        return output[0]
    # Fallback to best-effort local selection if no distributed backend is initialized.
    if input_list is None:
        return None
    idx = getattr(accelerator, "process_index", None)
    if idx is None:
        return None
    try:
        return input_list[idx]
    except (IndexError, TypeError):
        return None


__all__ = [
    "VLLMGenerationMixin",
    "_VLLMGenerationState",
    "_broadcast_object_list",
    "_gather_object_list",
    "_import_vllm_client_cls",
    "_is_peft_model_safe",
    "dist",
    "_optional_import",
    "_scatter_object",
    "_zero3_gather_factory",
]
