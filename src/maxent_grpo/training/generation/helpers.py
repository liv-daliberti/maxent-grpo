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

"""Completion generation helpers for the MaxEnt-GRPO runner.

The training stack supports two generation backends:

``CompletionGenerator``
    Orchestrates prompt batching, tokenizer truncation, and sampling for both
    local Hugging Face models and remote vLLM servers.  It exposes a single
    ``generate`` method that returns grouped completions plus optional logprob
    metadata, abstracting away the backend-specific plumbing.
``GenerationContext``
    Dataclass that captures all runtime knobs (accelerator handles, tokenizer,
    device, sampling penalties, vLLM configuration).  The context is passed
    verbatim between the setup routines and generator so the training loop can
    modify generation parameters without touching the generator internals.

This module contains the local HF sampling path along with vLLM convenience
wrappers that delegate to :mod:`maxent_grpo.training.generation.vllm` for the
multi-round retry/aggregation logic.  The docstrings aim to make these pieces
discoverable in the Sphinx API docs, describing how prompts flow from the
training dataloader to grouped completions consumed by the reward pipeline.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from contextlib import AbstractContextManager, nullcontext
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, cast
from urllib.parse import urlparse

from maxent_grpo.patches.vllm import VLLMLogprobResult, safe_generate
from ..types import (
    Accelerator as TypesAccelerator,
    PreTrainedModel as TypesPreTrainedModel,
    PreTrainedTokenizer as TypesPreTrainedTokenizer,
)
from ..run_helpers import (
    GenerationPenaltyConfig,
    GenerationSamplingConfig,
    PROMPT_CHAR_LIMIT,
    _truncate_prompt,
    require_accelerator,
    require_torch,
    require_transformer_base_classes,
)

torch = require_torch("generation")
Accelerator = require_accelerator("generation")
PreTrainedModel, PreTrainedTokenizer = require_transformer_base_classes("generation")
dist = getattr(torch, "distributed", None)
if dist is None:

    class _DistFallback:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_initialized() -> bool:
            return False

        @staticmethod
        def get_world_size() -> int:
            return 1

        @staticmethod
        def all_gather_object(output_list, input_obj):
            if output_list:
                output_list[0] = input_obj

        @staticmethod
        def broadcast_object_list(_payload, _src=0):
            return None

    dist = _DistFallback()
else:
    if not hasattr(dist, "is_available"):
        dist.is_available = lambda: False
    if not hasattr(dist, "is_initialized"):
        dist.is_initialized = lambda: False
    if not hasattr(dist, "get_world_size"):
        dist.get_world_size = lambda: 1

LOG = logging.getLogger(__name__)
_DEFAULT_RETRY_LIMIT = 3


@dataclass
class _AggregatedGenerationState:
    """Internal container tracking completions and metadata."""

    completions: List[List[str]]
    metadata: Optional[List[List[Optional[Any]]]] = None


def _append_completion_group(
    grouped_comps: List[List[str]],
    grouped_meta: Optional[List[List[Optional[Any]]]],
    prompt_idx: int,
    completions: Optional[List[str]],
    meta_group: Optional[List[Optional[Any]]],
) -> Optional[List[List[Optional[Any]]]]:
    """Append completions and metadata for ``prompt_idx``."""
    if not completions:
        return grouped_meta
    entries = list(completions)
    start = len(grouped_comps[prompt_idx])
    grouped_comps[prompt_idx].extend(entries)
    if meta_group is None:
        if grouped_meta is not None:
            grouped_meta[prompt_idx].extend([None] * len(entries))
        return grouped_meta
    if grouped_meta is None:
        grouped_meta = [[None] * len(group) for group in grouped_comps]
    meta_entries = list(meta_group)
    if len(meta_entries) < len(entries):
        meta_entries.extend([None] * (len(entries) - len(meta_entries)))
    else:
        meta_entries = meta_entries[: len(entries)]
    end = start + len(entries)
    current_meta = grouped_meta[prompt_idx]
    if len(current_meta) < end:
        current_meta.extend([None] * (end - len(current_meta)))
    current_meta[start:end] = meta_entries
    return grouped_meta


def _seed_generation_groups_impl(
    prompt_count: int,
    grouped_comps: Optional[List[List[str]]],
    grouped_meta: Optional[List[List[Optional[Any]]]],
) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
    """Initialize completion/meta buffers aligned with prompts."""
    aggregated_comps: List[List[str]] = [[] for _ in range(prompt_count)]
    aggregated_meta: Optional[List[List[Optional[Any]]]] = None
    base_groups = grouped_comps or []
    for idx in range(prompt_count):
        comp_group: List[str] = []
        if idx < len(base_groups) and base_groups[idx]:
            comp_group = list(base_groups[idx])
        meta_group = None
        if grouped_meta is not None and idx < len(grouped_meta):
            meta_group = grouped_meta[idx]
        aggregated_meta = _append_completion_group(
            aggregated_comps,
            aggregated_meta,
            idx,
            comp_group,
            meta_group,
        )
    return aggregated_comps, aggregated_meta


def _pending_generation_indices(
    aggregated_comps: List[List[str]],
    expected_generations: int,
) -> List[int]:
    """Return prompt indices still missing completions."""
    if expected_generations <= 0:
        return []
    return [
        idx
        for idx, comps in enumerate(aggregated_comps)
        if len(comps) < expected_generations
    ]


def _determine_retry_limit(
    expected_generations: int,
    max_retry_rounds: Optional[int],
) -> int:
    """Return the retry limit for incomplete prompts."""
    if max_retry_rounds and max_retry_rounds > 0:
        return max_retry_rounds
    if expected_generations > 0:
        return expected_generations
    return _DEFAULT_RETRY_LIMIT


def _retry_incomplete_prompts_impl(
    prompts: List[str],
    generator: Callable[
        [List[str], int, Optional[List[int]]],
        Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]],
    ],
    expected_generations: int,
    aggregated: _AggregatedGenerationState,
    max_retry_rounds: Optional[int],
) -> _AggregatedGenerationState:
    """Retry prompts missing completions until the retry limit is reached."""
    incomplete_indices = _pending_generation_indices(
        aggregated.completions,
        expected_generations,
    )
    retry_limit = _determine_retry_limit(expected_generations, max_retry_rounds)
    retry_round = 0
    while incomplete_indices and retry_round < retry_limit:
        retry_round += 1
        retry_groups, retry_meta = generator(
            [prompts[idx] for idx in incomplete_indices],
            expected_generations,
            [
                max(expected_generations - len(aggregated.completions[idx]), 0)
                for idx in incomplete_indices
            ],
        )
        retry_groups = retry_groups or [[] for _ in incomplete_indices]
        meta_payload: Optional[List[List[Optional[Any]]]] = None
        if isinstance(retry_meta, list):
            meta_payload = retry_meta
        for local_idx, prompt_idx in enumerate(incomplete_indices):
            meta_group = None
            if meta_payload is not None and local_idx < len(meta_payload):
                meta_group = meta_payload[local_idx]
            group = retry_groups[local_idx] if local_idx < len(retry_groups) else []
            aggregated.metadata = _append_completion_group(
                aggregated.completions,
                aggregated.metadata,
                prompt_idx,
                group,
                meta_group,
            )
        incomplete_indices = _pending_generation_indices(
            aggregated.completions,
            expected_generations,
        )
    return aggregated


# Keep the aggregation helpers in sync with the shared generation module used by
# the distilabel pipeline to avoid divergence between codepaths.
from maxent_grpo.generation import helpers as gen_helpers

_AggregatedGenerationState = gen_helpers.AggregatedGenerationState
_append_completion_group = gen_helpers.append_completion_group
_seed_generation_groups_impl = gen_helpers.seed_generation_groups
_pending_generation_indices = gen_helpers.pending_generation_indices
_determine_retry_limit = gen_helpers.determine_retry_limit
_retry_incomplete_prompts_impl = gen_helpers.retry_incomplete_prompts
drop_empty_prompt_groups = gen_helpers.drop_empty_prompt_groups
_truncate_to_expected_counts = gen_helpers.truncate_to_expected_counts
flatten_ref_metadata = gen_helpers.flatten_ref_metadata


def _optional_import(module_name: str) -> Any:
    """Import a module if available without triggering pylint import errors.

    :param module_name: Fully-qualified module path to import.
    :type module_name: str
    :returns: Imported module or ``None`` when unavailable.
    :rtype: Any | None
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _zero3_gather_factory(
    accelerator: Accelerator,
) -> Callable[[Sequence[Any]], AbstractContextManager[Any]]:
    """Return a callable that gathers parameters when ZeRO-3 is active.

    :param accelerator: Active ``Accelerator`` instance.
    :type accelerator: accelerate.Accelerator
    :returns: Context manager factory that gathers parameters if ZeRO-3 is enabled.
    :rtype: Callable[[Sequence[Any]], contextlib.AbstractContextManager[Any]]
    """
    ds_plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    zero_stage = getattr(ds_plugin, "zero_stage", 0) or 0
    gather_cls = None
    if zero_stage == 3:
        deepspeed_mod = _optional_import("deepspeed")
        zero_mod = getattr(deepspeed_mod, "zero", None) if deepspeed_mod else None
        gather_cls = getattr(zero_mod, "GatheredParameters", None)

    if gather_cls is None:
        return lambda _params: nullcontext()

    def _factory(params: Sequence[Any]) -> AbstractContextManager[Any]:
        """Context manager that temporarily gathers ZeRO-3 partitions.

        :param params: Iterable of parameters to gather.
        :type params: Sequence[Any]
        :returns: Context manager handling the gather logic.
        :rtype: contextlib.AbstractContextManager[Any]
        """
        return gather_cls(params)

    return _factory


def _is_peft_model_safe(target: Any) -> bool:
    """Return True if accelerate.utils reports that the model uses PEFT adapters.

    :param target: Model or module potentially using PEFT adapters.
    :type target: Any
    :returns: ``True`` when Accelerate identifies the target as PEFT-enabled.
    :rtype: bool
    """
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


def _import_vllm_client_cls() -> Optional[Type[Any]]:
    """Return TRL's VLLMClient class if available.

    :returns: ``trl.extras.vllm_client.VLLMClient`` or ``None``.
    :rtype: type | None
    """
    vllm_module = _optional_import("trl.extras.vllm_client")
    if vllm_module is None:
        return None
    return getattr(vllm_module, "VLLMClient", None)


@dataclass
class GenerationContext(GenerationSamplingConfig):
    """Configuration required to produce completions for each training batch."""

    accelerator: TypesAccelerator
    model: TypesPreTrainedModel
    tokenizer: TypesPreTrainedTokenizer
    generation_stats: Dict[str, int]
    device: torch.device
    penalty: GenerationPenaltyConfig = field(default_factory=GenerationPenaltyConfig)

    @property
    def gen_top_k(self) -> Optional[int]:
        """Return the top-k sampling override.

        :returns: Value to override ``top_k`` or ``None`` to use defaults.
        :rtype: int | None
        """
        return self.penalty.gen_top_k

    @gen_top_k.setter
    def gen_top_k(self, value: Optional[int]) -> None:
        self.penalty.gen_top_k = value

    @property
    def gen_best_of(self) -> Optional[int]:
        """Return the best-of sampling override, if provided.

        :returns: ``best_of`` override or ``None`` when unset.
        :rtype: int | None
        """
        return self.penalty.gen_best_of

    @gen_best_of.setter
    def gen_best_of(self, value: Optional[int]) -> None:
        self.penalty.gen_best_of = value

    @property
    def gen_frequency_penalty(self) -> float:
        """Return the custom frequency penalty.

        :returns: Frequency penalty applied during generation.
        :rtype: float
        """
        return self.penalty.gen_frequency_penalty

    @gen_frequency_penalty.setter
    def gen_frequency_penalty(self, value: float) -> None:
        self.penalty.gen_frequency_penalty = value

    @property
    def gen_presence_penalty(self) -> float:
        """Return the custom presence penalty.

        :returns: Presence penalty applied to generation.
        :rtype: float
        """
        return self.penalty.gen_presence_penalty

    @gen_presence_penalty.setter
    def gen_presence_penalty(self, value: float) -> None:
        self.penalty.gen_presence_penalty = value

    @property
    def gen_stop_sequences(self) -> Optional[List[str]]:
        """Return stop sequences that should override the config defaults.

        :returns: Custom stop sequences or ``None`` when not overridden.
        :rtype: list[str] | None
        """
        return self.penalty.gen_stop_sequences

    @gen_stop_sequences.setter
    def gen_stop_sequences(self, value: Optional[List[str]]) -> None:
        self.penalty.gen_stop_sequences = value

    def as_dict(self) -> Dict[str, Any]:
        """Return a lightweight representation useful for logging/debugging.

        :returns: Dictionary with core generation context attributes.
        :rtype: dict[str, Any]
        """
        return {
            "device": str(self.device),
            "max_prompt_len": self.max_prompt_len,
            "max_completion_len": self.max_completion_len,
            "top_k": self.gen_top_k,
            "best_of": self.gen_best_of,
            "use_vllm": self.use_vllm,
            "vllm_url": self.vllm_url,
        }


@dataclass
class _VLLMGenerationState:
    """Track state shared across multiple vLLM retries."""

    prompts: List[str]
    target_counts: List[int]
    requested_n: int
    round_limit: int
    track_logprobs: bool
    aggregated: List[List[str]] = field(init=False)
    aggregated_meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = field(
        init=False
    )

    def __post_init__(self) -> None:
        """Validate counts and initialize aggregate storage."""
        if len(self.target_counts) != len(self.prompts):
            raise ValueError("target_counts must align with prompts for vLLM state")
        self.aggregated = [[] for _ in self.prompts]
        self.aggregated_meta = (
            [[] for _ in self.prompts] if self.track_logprobs else None
        )

    def pending_indices(self) -> List[int]:
        """Return prompt indices that still require completions."""
        pending: List[int] = []
        for idx, (completions, target) in enumerate(
            zip(self.aggregated, self.target_counts)
        ):
            if target > 0 and len(completions) < target:
                pending.append(idx)
        return pending

    def remaining_counts(self, indices: List[int]) -> List[int]:
        """Compute outstanding completions for the provided indices."""
        counts: List[int] = []
        for idx in indices:
            target = self.target_counts[idx]
            counts.append(max(0, target - len(self.aggregated[idx])))
        return counts

    def trim(
        self,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Trim aggregated completions/meta to the requested counts."""
        trimmed = [
            group[:target] for group, target in zip(self.aggregated, self.target_counts)
        ]
        if self.aggregated_meta is None:
            return trimmed, None
        trimmed_meta = [
            meta_group[:target]
            for meta_group, target in zip(self.aggregated_meta, self.target_counts)
        ]
        return trimmed, trimmed_meta

    def drop_meta(self) -> None:
        """Discard stored log-prob metadata."""
        self.aggregated_meta = None


class CompletionGenerator:
    """Stateful helper that handles both local HF and vLLM completions."""

    def __init__(self, ctx: GenerationContext) -> None:
        """Create a generator with handles to local/vLLM backends.

        :param ctx: Generation settings shared with the training loop.
        :type ctx: GenerationContext
        """
        self.ctx = ctx
        self._vllm_client = None
        self._vllm_sync_ready = False
        self._last_vllm_synced_step: Optional[int] = None
        self._fsdp_cls = getattr(getattr(torch, "distributed", None), "fsdp", None)
        if self._fsdp_cls is not None:
            self._fsdp_cls = getattr(self._fsdp_cls, "FullyShardedDataParallel", None)

    def _vllm_base_url(self, url: str) -> str:
        """Strip common /generate suffixes to yield the server base URL.

        :param url: Raw vLLM endpoint URL (may include ``/generate``).
        :type url: str
        :returns: Base URL pointing at the root vLLM server.
        :rtype: str
        """
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
        """Instantiate the TRL VLLMClient when weight sync is enabled.

        :returns: ``True`` if the client is ready for synchronization.
        :rtype: bool
        """
        ctx = self.ctx
        if not ctx.vllm_sync_weights or not ctx.accelerator.is_main_process:
            return False
        if self._vllm_client is not None and self._vllm_sync_ready:
            return True
        client_cls = _import_vllm_client_cls()
        if client_cls is None:
            LOG.warning(
                "vLLM weight sync requested but TRL VLLMClient is unavailable; skipping."
            )
            self._vllm_sync_ready = False
            return False
        try:
            base_url = self._vllm_base_url(ctx.vllm_url)
            self._vllm_client = client_cls(base_url=base_url)
            self._vllm_client.init_communicator()
            self._vllm_sync_ready = True
            return True
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            LOG.warning("Failed to initialize vLLMClient for weight sync: %s", exc)
            self._vllm_client = None
            self._vllm_sync_ready = False
            return False

    def _maybe_sync_vllm_weights(self) -> None:
        """Push current model weights to the vLLM server."""
        if not self._ensure_vllm_client():
            return
        current_step = self.ctx.generation_stats.get("current_step")
        if current_step is not None and self._last_vllm_synced_step == int(
            current_step
        ):
            return
        accelerator = self.ctx.accelerator
        try:
            model = accelerator.unwrap_model(self.ctx.model)
        except (AttributeError, TypeError):
            model = self.ctx.model
        try:
            self._sync_model_params_to_vllm(model, accelerator)
            stats = self.ctx.generation_stats
            stats["vllm_weight_syncs"] = int(stats.get("vllm_weight_syncs", 0)) + 1
            if current_step is not None:
                self._last_vllm_synced_step = int(current_step)
        except (RuntimeError, ValueError) as exc:  # pragma: no cover
            LOG.warning("Skipping vLLM weight sync due to error: %s", exc)
        wait_for_all = getattr(accelerator, "wait_for_everyone", None)
        if callable(wait_for_all):
            wait_for_all()

    def _sync_model_params_to_vllm(
        self,
        model: PreTrainedModel,
        accelerator: Accelerator,
    ) -> None:
        """Best-effort parameter broadcast mirroring HF GRPO's vLLM path.

        :param model: Model whose parameters are being synchronized.
        :type model: transformers.PreTrainedModel
        :param accelerator: Accelerator managing distributed context.
        :type accelerator: accelerate.Accelerator
        """
        gather_factory = _zero3_gather_factory(accelerator)
        if self._fsdp_cls is not None and isinstance(model, self._fsdp_cls):
            self._sync_fsdp_params(model)
            self._reset_vllm_cache()
            return
        if _is_peft_model_safe(model):
            self._sync_peft_params(model, gather_factory)
            self._reset_vllm_cache()
            return
        self._sync_standard_params(model, gather_factory)
        self._reset_vllm_cache()

    def _push_param_to_vllm(self, name: str, param: Any) -> None:
        """Send a single parameter tensor to the vLLM client.

        :param name: Fully-qualified parameter name.
        :type name: str
        :param param: Parameter tensor or proxy to upload.
        :type param: Any
        """
        if param is None or self._vllm_client is None:
            return
        update_raw = getattr(self._vllm_client, "update_named_param", None)
        if not callable(update_raw):
            return
        update_fn = cast(Callable[[str, Any], None], update_raw)
        try:
            update_fn(name, param.data)
        except (
            RuntimeError,
            ValueError,
        ) as exc:  # pragma: no cover - network/dependency dependent
            LOG.warning("Failed to push param %s to vLLM: %s", name, exc)

    def _reset_vllm_cache(self) -> None:
        """Reset prefix caches when the vLLM client exposes the helper."""
        client = self._vllm_client
        reset_raw = getattr(client, "reset_prefix_cache", None)
        if not callable(reset_raw):
            return
        reset_fn = cast(Callable[[], Any], reset_raw)
        try:
            reset_fn()
        except (AttributeError, RuntimeError):
            pass

    def _sync_fsdp_params(self, model: Any) -> None:
        """Iterate FSDP shards and push full parameters to vLLM."""
        fsdp_cls = self._fsdp_cls
        if fsdp_cls is None or not isinstance(model, fsdp_cls):
            return
        visited: Set[str] = set()

        def _sync_tree(module: Any, prefix: str = "") -> None:
            """Traverse child modules pushing gathered parameters to vLLM."""
            for child_name, child in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                _sync_tree(child, child_prefix)
            if isinstance(module, fsdp_cls):
                with fsdp_cls.summon_full_params(module, recurse=False, writeback=False):
                    for pname, param in module.named_parameters():
                        full_name = f"{prefix}.{pname}" if prefix else pname
                        for extra in (
                            "_fsdp_wrapped_module.",
                            "_checkpoint_wrapped_module.",
                        ):
                            full_name = full_name.replace(extra, "")
                        if full_name in visited:
                            continue
                        visited.add(full_name)
                        self._push_param_to_vllm(full_name, param)

        _sync_tree(model)

    def _sync_peft_params(
        self,
        model: Any,
        gather_factory: Callable[[Sequence[Any]], AbstractContextManager[Any]],
    ) -> None:
        """Push merged PEFT adapter weights to vLLM."""
        params = list(model.parameters())
        merge_fn = getattr(model, "merge_adapter", None)
        unmerge_fn = getattr(model, "unmerge_adapter", None)
        with gather_factory(params):
            if callable(merge_fn):
                merge_fn()
            for name, param in model.named_parameters():
                clean = name.removeprefix("base_model.model.").replace(
                    ".base_layer", ""
                )
                if getattr(model, "prefix", None) and str(model.prefix) in clean:
                    continue
                if "original_module" in clean:
                    continue
                clean = clean.replace("modules_to_save.default.", "")
                self._push_param_to_vllm(clean, param)
            if callable(unmerge_fn):
                unmerge_fn()

    def _sync_standard_params(
        self,
        model: Any,
        gather_factory: Callable[[Sequence[Any]], AbstractContextManager[Any]],
    ) -> None:
        """Push standard (non-PEFT/FSDP) parameters to vLLM."""
        params = list(model.parameters())
        with gather_factory(params):
            for name, param in model.named_parameters():
                self._push_param_to_vllm(name, param)

    def describe(self) -> Dict[str, Any]:
        """Expose the underlying generation configuration for logging.

        :returns: Dictionary describing the generation context.
        :rtype: dict[str, Any]
        """
        return self.ctx.as_dict()

    def _resolve_vllm_round_limit(self, requested_n: int) -> int:
        """Decide how many vLLM rounds to run for the current request.

        :param requested_n: Number of completions requested per prompt.
        :type requested_n: int
        :returns: Maximum rounds allowed for each batch.
        :rtype: int
        """
        ctx = self.ctx
        if ctx.vllm_rounds_cfg > 0:
            return max(1, ctx.vllm_rounds_cfg)
        return max(1, requested_n)

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
        helper: "CompletionGenerator",
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

    def _build_local_prompt_requests(
        self,
        prompts: List[str],
        target_counts: List[int],
    ) -> Tuple[List[str], List[int]]:
        """Expand prompts by their requested counts for local sampling.

        :param prompts: Prompt strings to replicate.
        :type prompts: list[str]
        :param target_counts: Desired completions per prompt.
        :type target_counts: list[int]
        :returns: Tuple of expanded prompts and original prompt indices.
        :rtype: tuple[list[str], list[int]]
        """
        expanded_prompts: List[str] = []
        prompt_indices: List[int] = []
        for idx, (prompt, target_count) in enumerate(zip(prompts, target_counts)):
            adjusted_target = max(0, int(target_count))
            if adjusted_target <= 0:
                continue
            expanded_prompts.extend([prompt] * adjusted_target)
            prompt_indices.extend([idx] * adjusted_target)
        return expanded_prompts, prompt_indices

    def _prompt_char_limit(self) -> int:
        """Return the character limit applied to prompts for vLLM/local calls.

        :returns: Maximum characters allowed per prompt.
        :rtype: int
        """
        approx_chars = 0
        if self.ctx.max_prompt_len and self.ctx.max_prompt_len > 0:
            approx_chars = int(self.ctx.max_prompt_len * 4)
        if PROMPT_CHAR_LIMIT <= 0:
            return approx_chars
        if approx_chars <= 0:
            return PROMPT_CHAR_LIMIT
        return max(PROMPT_CHAR_LIMIT, approx_chars)

    def _tokenize_expanded_prompts(
        self,
        expanded_prompts: List[str],
    ) -> Tuple[Any, List[int]]:
        """Tokenize prompts for local generation and track prompt lengths.

        :param expanded_prompts: Prompt strings after applying counts.
        :type expanded_prompts: list[str]
        :returns: Tuple of tokenizer inputs and per-prompt lengths.
        :rtype: tuple[Any, list[int]]
        """
        ctx = self.ctx
        encoder_inputs = self.ctx.tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ctx.max_prompt_len,
        ).to(ctx.device)
        prompt_lengths = (
            encoder_inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()
        )
        return encoder_inputs, prompt_lengths

    def _run_local_model(
        self,
        encoder_inputs: Any,
        prompt_lengths: List[int],
    ) -> List[str]:
        """Run the HF model locally and decode completions.

        :param encoder_inputs: Tokenized prompt batch.
        :type encoder_inputs: Any
        :param prompt_lengths: Prompt token lengths for slicing completions.
        :type prompt_lengths: list[int]
        :returns: Decoded completion strings per expanded prompt.
        :rtype: list[str]
        """
        ctx = self.ctx
        gen_model = ctx.accelerator.unwrap_model(ctx.model)
        with torch.no_grad():
            gen_out = gen_model.generate(
                **encoder_inputs,
                do_sample=True,
                temperature=ctx.gen_temperature,
                top_p=ctx.gen_top_p,
                top_k=ctx.gen_top_k if ctx.gen_top_k is not None else None,
                max_new_tokens=ctx.max_completion_len,
                num_return_sequences=1,
            )
        return self._decode_sequences(gen_out, prompt_lengths, ctx.tokenizer)

    def _generate_local(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Generate completions using the local HF model.

        :param prompts: Prompt strings collected from the batch.
        :type prompts: list[str]
        :param num_samples: Default number of completions per prompt.
        :type num_samples: int
        :param per_prompt_counts: Optional per-prompt overrides.
        :type per_prompt_counts: list[int] | None
        :returns: Tuple of grouped completions and optional metadata (``None`` for local generation).
        :rtype: tuple[list[list[str]], list[list[Optional[VLLMLogprobResult]]] | None
        """
        grouped: List[List[str]] = [[] for _ in prompts]
        if not prompts:
            return grouped, None
        char_limit = self._prompt_char_limit()
        prompts = [_truncate_prompt(prompt, char_limit) for prompt in prompts]
        target_counts = self._resolve_local_counts(
            prompts, num_samples, per_prompt_counts
        )
        expanded_prompts, prompt_indices = self._build_local_prompt_requests(
            prompts,
            target_counts,
        )
        if not expanded_prompts:
            return grouped, None
        enc_inputs, prompt_lengths = self._tokenize_expanded_prompts(expanded_prompts)
        decoded = self._run_local_model(enc_inputs, prompt_lengths)
        for text, prompt_idx in zip(decoded, prompt_indices):
            grouped[prompt_idx].append(text)
        return grouped, None

    @staticmethod
    def _summarize_grouped(groups: List[List[str]], limit: int = 8) -> str:
        """Return a compact preview of grouped completions.

        :param groups: Nested list of completions per prompt.
        :type groups: list[list[str]]
        :param limit: Maximum groups to display in the preview.
        :type limit: int
        :returns: Summary string with lengths/sample text.
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

    @staticmethod
    def _resolve_local_counts(
        prompts: List[str],
        default_count: int,
        overrides: Optional[List[int]],
    ) -> List[int]:
        """Resolve per-prompt generation counts for local sampling.

        :param prompts: Prompt strings that require completions.
        :type prompts: list[str]
        :param default_count: Baseline completions per prompt.
        :type default_count: int
        :param overrides: Optional per-prompt overrides.
        :type overrides: list[int] | None
        :returns: Resolved completion counts aligned with ``prompts``.
        :rtype: list[int]
        """
        if overrides is None:
            return [default_count] * len(prompts)
        if len(overrides) != len(prompts):
            raise ValueError("per_prompt_counts length must match prompts length")
        return overrides

    @staticmethod
    def _decode_sequences(
        sequences: torch.Tensor,
        prompt_lengths: List[int],
        tokenizer: PreTrainedTokenizer,
    ) -> List[str]:
        """Decode model outputs into completion strings.

        :param sequences: Full token sequences returned by ``generate``.
        :type sequences: torch.Tensor
        :param prompt_lengths: Token counts for each prompt in ``sequences``.
        :type prompt_lengths: list[int]
        :param tokenizer: Tokenizer used to decode completions.
        :type tokenizer: transformers.PreTrainedTokenizer
        :returns: List of decoded completions aligned with ``prompt_lengths``.
        :rtype: list[str]
        """
        outputs: List[str] = []
        for row, prompt_len in zip(sequences, prompt_lengths):
            completion_ids = row[int(prompt_len) :]
            outputs.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
        return outputs

    def _request_vllm_batch(
        self,
        pending_prompts: List[str],
        request_count: int,
    ) -> Tuple[
        Optional[List[List[str]]],
        Optional[List[List[Optional[VLLMLogprobResult]]]],
    ]:
        """Request completions from vLLM for a subset of prompts.

        :param pending_prompts: Prompts that still require completions.
        :type pending_prompts: list[str]
        :param request_count: Number of completions to request per prompt.
        :type request_count: int
        :returns: Tuple of grouped completions/logprob metadata or ``None`` on failure.
        :rtype: tuple[list[list[str]] | None, list[list[Optional[VLLMLogprobResult]]] | None]
        """
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
            ("vLLM grouped outputs len=%d vs pending=%d | per-prompt lengths=%s"),
            len(grouped),
            pending_count,
            [len(entry) for entry in grouped],
        )
        return None, None

    def _record_vllm_latency(self, latency_ms: float) -> None:
        """Track latency metrics for successful vLLM invocations.

        :param latency_ms: Measured latency (milliseconds).
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
        """Assemble keyword arguments for ``safe_generate`` requests.

        :param prompts: Prompt strings for the batch.
        :type prompts: list[str]
        :param request_count: Completions requested per prompt.
        :type request_count: int
        :returns: Dictionary passed to ``safe_generate``.
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
        """Call vLLM with retries by splitting large prompt batches.

        :param prompts: Prompts to send in a single request (may recurse).
        :type prompts: list[str]
        :param request_count: Completions requested per prompt.
        :type request_count: int
        :returns: Grouped completions, optional logprob metadata, and latency.
        :rtype: tuple[list[list[str]], list[list[Optional[VLLMLogprobResult]]] | None, float] | None
        """
        try:
            request_kwargs = self._build_vllm_request_kwargs(prompts, request_count)
            grouped, grouped_meta, latency_ms = safe_generate(**request_kwargs)
            return grouped, grouped_meta, latency_ms
        except RuntimeError as err:
            if len(prompts) <= 1:
                LOG.warning("vLLM request failed for single prompt: %s", err)
                return None
            mid = max(1, len(prompts) // 2)
            left = self._invoke_vllm_requests(prompts[:mid], request_count)
            right = self._invoke_vllm_requests(prompts[mid:], request_count)
            if left is None or right is None:
                return None
            left_groups, left_meta, left_latency = left
            right_groups, right_meta, right_latency = right
            combined_groups = left_groups + right_groups
            if left_meta is None or right_meta is None:
                combined_meta = None
            else:
                combined_meta = left_meta + right_meta
            return combined_groups, combined_meta, left_latency + right_latency

    def _merge_vllm_results(
        self,
        state: _VLLMGenerationState,
        grouped: List[List[str]],
        grouped_meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        pending_indices: List[int],
    ) -> None:
        """Append vLLM outputs into the shared state aggregates.

        :param state: Active vLLM generation state.
        :type state: _VLLMGenerationState
        :param grouped: Completions returned per pending prompt.
        :type grouped: list[list[str]]
        :param grouped_meta: Optional logprob metadata aligned with ``grouped``.
        :type grouped_meta: list[list[Optional[VLLMLogprobResult]]] | None
        :param pending_indices: Prompt indices corresponding to ``grouped`` entries.
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
            if target > 0 and len(aggregated[prompt_idx]) > target:
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

    def _backfill_missing(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        """Generate missing completions locally when vLLM under-delivers.

        :param state: Active vLLM generation state.
        :type state: _VLLMGenerationState
        :param missing_indices: Prompt indices lacking enough completions.
        :type missing_indices: list[int]
        """
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
        local_groups, _ = self._generate_local(
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
        """Log a warning when vLLM fails to deliver even after retries/backfill.

        :param state: Active vLLM generation state.
        :type state: _VLLMGenerationState
        :param missing_indices: Prompts still missing completions.
        :type missing_indices: list[int]
        """
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

    @staticmethod
    def _coalesce_grouped_outputs(
        groups: List[List[str]],
        prompt_count: int,
        requested_n: int,
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Normalize grouped outputs when vLLM returns per-sample lists.

        :param groups: Grouped completions for each prompt/sample.
        :type groups: list[list[str]]
        :param prompt_count: Number of prompts tracked in the request.
        :type prompt_count: int
        :param requested_n: Requested completions per prompt.
        :type requested_n: int
        :param meta: Optional grouped logprob metadata.
        :type meta: list[list[Optional[VLLMLogprobResult]]] | None
        :returns: Regrouped completions and metadata (or original input on mismatch).
        :rtype: tuple[list[list[str]], list[list[Optional[VLLMLogprobResult]]] | None]
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
            merged, merged_meta = CompletionGenerator._merge_group_chunk(
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
        """Merge consecutive micro-groups back into per-prompt lists.

        :param chunk: Consecutive grouped completions for one prompt.
        :type chunk: list[list[str]]
        :param meta_chunk: Optional logprob metadata aligned with ``chunk``.
        :type meta_chunk: list[list[Optional[VLLMLogprobResult]]] | None
        :param requested_n: Requested completions per prompt (trims overflow).
        :type requested_n: int
        :returns: Flattened completions and optional metadata slice.
        :rtype: tuple[list[str], list[Optional[VLLMLogprobResult]] | None]
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

    def _prepare_vllm_targets(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]],
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        """Resolve target counts and optional dedup mapping for vLLM.

        :param prompts: Original prompt list (possibly with duplicates).
        :type prompts: list[str]
        :param num_samples: Default completions per prompt.
        :type num_samples: int
        :param per_prompt_counts: Optional overrides per prompt.
        :type per_prompt_counts: list[int] | None
        :returns: Tuple of prompts, target counts, and optional dedupe mapping.
        :rtype: tuple[list[str], list[int], list[int] | None]
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

    def _run_vllm_rounds(self, state: _VLLMGenerationState) -> None:
        """Iteratively request completions until targets are satisfied.

        :param state: vLLM generation state tracking pending targets.
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

    @staticmethod
    def _expand_dedup_results(
        grouped: List[List[str]],
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        mapping: Optional[List[int]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Expand de-duplicated prompts back to the original ordering.

        :param grouped: Grouped completions for unique prompts.
        :type grouped: list[list[str]]
        :param meta: Optional metadata for each grouped prompt.
        :type meta: list[list[Optional[VLLMLogprobResult]]] | None
        :param mapping: Mapping from original prompt index to deduped index.
        :type mapping: list[int] | None
        :returns: Completions/meta aligned with the original prompt order.
        :rtype: tuple[list[list[str]], list[list[Optional[VLLMLogprobResult]]] | None]
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

    def _generate_with_vllm(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Generate completions via vLLM, with dedupe/backoff handling.

        :param prompts: Prompt strings to send to vLLM.
        :type prompts: list[str]
        :param num_samples: Default completions per prompt.
        :type num_samples: int
        :param per_prompt_counts: Optional overrides per prompt.
        :type per_prompt_counts: list[int] | None
        :returns: Grouped completions and optional logprob metadata.
        :rtype: tuple[list[list[str]], list[list[Optional[VLLMLogprobResult]]] | None]
        """
        if not prompts:
            return [], None
        self._maybe_sync_vllm_weights()
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

    def _execute_vllm_request(
        self,
        state: _VLLMGenerationState,
        pending_indices: List[int],
    ) -> bool:
        """Request completions for specific prompts, grouped by need bucket.

        :param state: vLLM generation state containing prompts/meta.
        :type state: _VLLMGenerationState
        :param pending_indices: Prompt indices that still need completions.
        :type pending_indices: list[int]
        :returns: ``True`` if every bucket succeeded, ``False`` otherwise.
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
            grouped, grouped_meta = self._request_vllm_batch(pending_prompts, need)
            if grouped is None:
                return False
            self._merge_vllm_results(state, grouped, grouped_meta, indices)
        return True

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
        accelerator = self.ctx.accelerator
        world_size = accelerator.num_processes
        if world_size <= 1:
            grouped_local, meta_local = self._pluck_rank_outputs(
                grouped_all or [],
                meta_all,
                offsets,
                flat_prompts,
            )
            return grouped_local, meta_local

        scatter_payload: Optional[
            List[
                Tuple[
                    List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]
                ]
            ]
        ] = None
        if accelerator.is_main_process:
            scatter_payload = []
            total = len(flat_prompts)
            for rank in range(world_size):
                start = offsets[rank]
                end = offsets[rank + 1] if rank + 1 < len(offsets) else total
                slice_grouped = [] if grouped_all is None else grouped_all[start:end]
                slice_meta = None if meta_all is None else meta_all[start:end]
                scatter_payload.append((slice_grouped, slice_meta))
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

    def _generate_vllm_collective(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Run vLLM once on rank 0 and scatter results back to all ranks.

        :param prompts: Local rank's prompts (gathered across ranks internally).
        :type prompts: list[str]
        :param num_samples: Default completions to request per prompt.
        :type num_samples: int
        :param per_prompt_counts: Optional per-prompt overrides.
        :type per_prompt_counts: list[int] | None
        :returns: Grouped completions and optional logprob metadata for this rank.
        :rtype: tuple[list[list[str]], list[list[Optional[VLLMLogprobResult]]] | None]
        """
        flat_prompts, offsets, flat_counts = self._flatten_prompts_for_broadcast(
            prompts,
            per_prompt_counts,
        )
        accelerator = self.ctx.accelerator
        if accelerator.is_main_process:
            grouped_all, meta_all = self._generate_with_vllm(
                flat_prompts,
                num_samples,
                flat_counts,
            )
        else:
            grouped_all = None
            meta_all = None
        return self._scatter_vllm_payload(flat_prompts, offsets, grouped_all, meta_all)

    def generate(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Produce completions, preferring vLLM when configured.

        :param prompts: Prompt strings grouped per batch entry.
        :type prompts: list[str]
        :param num_samples: Completions requested per prompt when ``per_prompt_counts`` is ``None``.
        :type num_samples: int
        :param per_prompt_counts: Optional override counts (must align with ``prompts``).
        :type per_prompt_counts: list[int] | None
        :returns: Grouped completions and optional log-prob metadata per prompt.
        :rtype: tuple[list[list[str]], list[list[Optional[VLLMLogprobResult]]] | None]
        :raises ValueError: If ``per_prompt_counts`` length mismatches ``prompts``.
        """
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


__all__ = ["CompletionGenerator", "GenerationContext"]


def _gather_object_list(accelerator: Accelerator, value: List[Any]) -> List[List[Any]]:
    """Gather Python lists across ranks with graceful Accelerate fallbacks.

    :param accelerator: Accelerator providing collective utilities.
    :type accelerator: accelerate.Accelerator
    :param value: Local list payload to gather across ranks.
    :type value: list[Any]
    :returns: List of gathered payloads ordered by rank.
    :rtype: list[list[Any]]
    """
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
    """Scatter python objects from src to all ranks.

    :param accelerator: Accelerator providing distributed process metadata.
    :type accelerator: accelerate.Accelerator
    :param input_list: Objects available on ``src`` used for scattering.
    :type input_list: list[Any] | None
    :param src: Rank to scatter from.
    :type src: int
    :returns: Object assigned to the current rank (or ``None`` if unavailable).
    :rtype: Any
    """
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
    return input_list[accelerator.process_index]
