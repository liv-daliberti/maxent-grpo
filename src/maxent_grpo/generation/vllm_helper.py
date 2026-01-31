"""Assemble the vLLMGenerationHelper from dedicated mixins."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from maxent_grpo.patches.vllm import VLLMLogprobResult, safe_generate
from maxent_grpo.training.runtime import require_accelerator, require_torch

from .vllm_distributed import VLLMDistributedMixin, _scatter_object
from .vllm_requests import VLLMRequestMixin, _resolve_served_model_id
from .vllm_state import _VLLMGenerationState
from .vllm_weight_sync import (
    VLLMWeightSyncMixin,
    _import_vllm_client_cls,
    _is_peft_model_safe,
    _optional_import,
    _zero3_gather_factory,
)

torch = require_torch("generation_vllm")
Accelerator = require_accelerator("generation_vllm")
LOG = logging.getLogger(__name__)


def _seed_stats_metadata(stats: Dict[str, Any], ctx: Any) -> None:
    """Ensure dataset/model identifiers are stored on generation stats."""

    if not stats.get("dataset_name"):
        label = getattr(ctx, "dataset_name", None)
        if not label:
            training_args = getattr(ctx, "training_args", None)
            label = getattr(training_args, "dataset_name", None)
            if not label:
                mixture = getattr(training_args, "dataset_mixture", None)
                if mixture:
                    label = str(mixture)
        if label:
            stats["dataset_name"] = label
    if not stats.get("model_id"):
        model_label = _resolve_served_model_id(ctx)
        if model_label:
            stats["model_id"] = model_label


class VLLMGenerationHelper(
    VLLMWeightSyncMixin,
    VLLMRequestMixin,
    VLLMDistributedMixin,
):
    """Encapsulate vLLM-specific logic so CompletionGenerator stays lean."""

    def __init__(
        self,
        ctx: Any,
        fallback_generate: Callable[
            [List[str], int, Optional[List[int]]],
            Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]],
        ],
    ) -> None:
        """Initialize the helper with TRL/Accelerate context and fallback flow.

        :param ctx: Training or generation context exposing accelerator,
            tokenizer, and configuration attributes consumed by the helper.
        :type ctx: Any
        :param fallback_generate: Callable used to produce completions locally
            when vLLM requests fail or when backfilling missing outputs.
        :type fallback_generate: Callable[[list[str], int, list[int] | None], tuple[list[list[str]], list[list[VLLMLogprobResult | None]] | None]]
        """
        self.ctx = ctx
        self._fallback_generate = fallback_generate
        self._vllm_client: Any = None
        self._vllm_sync_ready = False
        self._last_vllm_synced_step: Optional[int] = None
        self._fsdp_cls = getattr(getattr(torch, "distributed", None), "fsdp", None)
        if self._fsdp_cls is not None:
            self._fsdp_cls = getattr(self._fsdp_cls, "FullyShardedDataParallel", None)
        self._gather_factory = _zero3_gather_factory(ctx.accelerator)
        try:
            import maxent_grpo.generation.vllm as _vmod
        except ImportError:
            _vmod = None
        # Patchable hooks used in tests.
        self._safe_generate = (
            getattr(_vmod, "safe_generate", None) if _vmod is not None else None
        ) or safe_generate
        self._scatter_object = (
            getattr(_vmod, "_scatter_object", None) if _vmod is not None else None
        ) or _scatter_object
        self._time = getattr(_vmod, "time", time) if _vmod is not None else time
        self._is_peft_model_safe = (
            getattr(_vmod, "_is_peft_model_safe", None) if _vmod is not None else None
        ) or _is_peft_model_safe
        self._import_vllm_client_cls = (
            getattr(_vmod, "_import_vllm_client_cls", None)
            if _vmod is not None
            else None
        ) or _import_vllm_client_cls
        stats = getattr(ctx, "generation_stats", {})
        stats.setdefault("vllm_backfilled_prompts", 0)
        stats.setdefault("vllm_failed_prompts", 0)
        stats.setdefault("vllm_retry_rounds", 0)
        stats.setdefault("vllm_retry_failures", 0)
        stats.setdefault("vllm_last_error", None)
        _seed_stats_metadata(stats, ctx)
        ctx.generation_stats = stats
        if not getattr(ctx, "_maxent_vllm_helper_logged", False):
            sync_weights = bool(getattr(ctx, "vllm_sync_weights", False))
            backend_note = (
                "frozen server (no weight sync)" if not sync_weights else "live weight sync"
            )
            LOG.info(
                "vLLM helper configured | use_vllm=%s | endpoint=%s | request_logprobs=%s | sync_weights=%s (%s)",
                bool(getattr(ctx, "use_vllm", False)),
                getattr(ctx, "vllm_url", None),
                bool(getattr(ctx, "vllm_request_logprobs", False)),
                sync_weights,
                backend_note,
            )
            setattr(ctx, "_maxent_vllm_helper_logged", True)

    # Expose patchable state via public accessors for callers/tests.
    @property
    def vllm_client(self) -> Any:
        return self._vllm_client

    @vllm_client.setter
    def vllm_client(self, value: Any) -> None:
        self._vllm_client = value

    @property
    def vllm_sync_ready(self) -> bool:
        return self._vllm_sync_ready

    @vllm_sync_ready.setter
    def vllm_sync_ready(self, value: bool) -> None:
        self._vllm_sync_ready = bool(value)

    @property
    def last_vllm_synced_step(self) -> Optional[int]:
        return self._last_vllm_synced_step

    @last_vllm_synced_step.setter
    def last_vllm_synced_step(self, value: Optional[int]) -> None:
        self._last_vllm_synced_step = value

    @property
    def fsdp_cls(self) -> Any:
        return self._fsdp_cls

    @fsdp_cls.setter
    def fsdp_cls(self, value: Any) -> None:
        self._fsdp_cls = value

    def generate(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]],
        ensure_client: Optional[Callable[[], bool]] = None,
        sync_model: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Generate completions for prompts via vLLM, optionally deduplicating.

        The helper handles optional weight synchronization, deduplicates
        repeated prompts when enabled, and retries requests up to the configured
        round limit. Results are expanded back to the original prompt ordering.

        :param prompts: Prompts to generate completions for.
        :type prompts: list[str]
        :param num_samples: Requested completions per prompt.
        :type num_samples: int
        :param per_prompt_counts: Optional per-prompt completion counts; when
            provided overrides ``num_samples`` on a per-prompt basis.
        :type per_prompt_counts: list[int] | None
        :param ensure_client: Optional callable to guarantee the vLLM client is
            ready before issuing requests.
        :type ensure_client: Callable[[], bool] | None
        :param sync_model: Optional callable to push model weights before
            generation.
        :type sync_model: Callable[[Any], None] | None
        :returns: Grouped completions per prompt and optional grouped logprob
            metadata when enabled.
        :rtype: tuple[list[list[str]], list[list[VLLMLogprobResult | None]] | None]
        """
        sync_fn = self.maybe_sync_weights
        using_default_sync = getattr(sync_fn, "__func__", None) is VLLMGenerationHelper.maybe_sync_weights
        try:
            if ensure_client is None and sync_model is None:
                sync_fn()
            else:
                sync_fn(ensure_client, sync_model)
        except TypeError:
            sync_fn()
        stats = self.ctx.generation_stats
        if getattr(self.ctx, "vllm_sync_weights", False):
            if "vllm_weight_syncs" not in stats:
                stats["vllm_weight_syncs"] = 1
            if not using_default_sync and (
                ensure_client is not None or sync_model is not None
            ):
                if ensure_client is not None:
                    ensure_client()
                if sync_model is not None:
                    sync_model(getattr(self.ctx, "model", None))
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
        ensure_client: Optional[Callable[[], bool]] = None,
        sync_model: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Broadcast prompts across ranks and gather vLLM generations collectively.

        Prompts from every rank are gathered on the main process, generated in a
        single vLLM call, and the outputs are scattered back to each rank with
        metadata preserved when available.

        :param prompts: Local prompts for the current rank.
        :type prompts: list[str]
        :param num_samples: Requested completions per prompt.
        :type num_samples: int
        :param per_prompt_counts: Optional per-prompt completion counts.
        :type per_prompt_counts: list[int] | None
        :param ensure_client: Optional callable ensuring the vLLM client is
            ready on the main process.
        :type ensure_client: Callable[[], bool] | None
        :param sync_model: Optional callable to push model weights before
            generation.
        :type sync_model: Callable[[Any], None] | None
        :returns: Grouped completions and optional metadata corresponding to the
            current rank's prompts.
        :rtype: tuple[list[list[str]], list[list[VLLMLogprobResult | None]] | None]
        """
        flat_prompts, offsets, flat_counts = self._flatten_prompts_for_broadcast(
            prompts,
            per_prompt_counts,
        )
        accelerator = self.ctx.accelerator
        # When ZeRO-3 is active, weight synchronization gathers parameters
        # collectively even though only rank 0 issues the HTTP requests. Run the
        # sync hook on every rank before non-main processes block waiting for
        # the scatter payload.
        if getattr(self.ctx, "vllm_sync_weights", False):
            try:
                self.maybe_sync_weights(
                    ensure_client=ensure_client,
                    sync_model=sync_model or self._sync_model_params_to_vllm,
                )
            except TypeError:
                # Backwards-compatible fallback for lightweight stubs.
                self.maybe_sync_weights()
        if accelerator.is_main_process:
            if ensure_client is None and sync_model is None:
                grouped_all, meta_all = self.generate(
                    flat_prompts,
                    num_samples,
                    flat_counts,
                )
            else:
                grouped_all, meta_all = self.generate(
                    flat_prompts,
                    num_samples,
                    flat_counts,
                    ensure_client,
                    sync_model,
                )
        else:
            grouped_all = None
            meta_all = None
        return self._scatter_vllm_payload(flat_prompts, offsets, grouped_all, meta_all)


__all__ = [
    "Accelerator",
    "VLLMGenerationHelper",
    "_VLLMGenerationState",
    "_optional_import",
]
