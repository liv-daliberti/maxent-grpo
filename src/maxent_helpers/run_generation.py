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

"""Completion generation helpers for the MaxEnt-GRPO runner."""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from contextlib import AbstractContextManager, nullcontext
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, cast
from urllib.parse import urlparse

from utils.vllm_patch import VLLMLogprobResult, safe_generate
from .run_helpers import (
    PROMPT_CHAR_LIMIT,
    _truncate_prompt,
    GenerationPenaltyConfig,
    GenerationSamplingConfig,
    require_accelerator,
    require_torch,
    require_transformer_base_classes,
)

torch = require_torch("generation")
Accelerator = require_accelerator("generation")
PreTrainedModel, PreTrainedTokenizer = require_transformer_base_classes("generation")
dist = getattr(torch, "distributed", None)

LOG = logging.getLogger(__name__)


def _optional_import(module_name: str) -> Any:
    """Import a module if available without triggering pylint import errors."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _zero3_gather_factory(
    accelerator: Accelerator,
) -> Callable[[Sequence[Any]], AbstractContextManager[Any]]:
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

    def _factory(params: Sequence[Any]) -> AbstractContextManager[Any]:
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


def _import_vllm_client_cls() -> Optional[Type[Any]]:
    """Return TRL's VLLMClient class if available."""
    vllm_module = _optional_import("trl.extras.vllm_client")
    if vllm_module is None:
        return None
    return getattr(vllm_module, "VLLMClient", None)


@dataclass
class GenerationContext(GenerationSamplingConfig):
    """Configuration required to produce completions for each training batch."""

    accelerator: Accelerator
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    generation_stats: Dict[str, int]
    device: torch.device
    penalty: GenerationPenaltyConfig = field(default_factory=GenerationPenaltyConfig)

    @property
    def gen_top_k(self) -> Optional[int]:
        """Return the top-k sampling override."""
        return self.penalty.gen_top_k

    @gen_top_k.setter
    def gen_top_k(self, value: Optional[int]) -> None:
        self.penalty.gen_top_k = value

    @property
    def gen_best_of(self) -> Optional[int]:
        """Return the best-of sampling override, if provided."""
        return self.penalty.gen_best_of

    @gen_best_of.setter
    def gen_best_of(self, value: Optional[int]) -> None:
        self.penalty.gen_best_of = value

    @property
    def gen_frequency_penalty(self) -> float:
        """Return the custom frequency penalty."""
        return self.penalty.gen_frequency_penalty

    @gen_frequency_penalty.setter
    def gen_frequency_penalty(self, value: float) -> None:
        self.penalty.gen_frequency_penalty = value

    @property
    def gen_presence_penalty(self) -> float:
        """Return the custom presence penalty."""
        return self.penalty.gen_presence_penalty

    @gen_presence_penalty.setter
    def gen_presence_penalty(self, value: float) -> None:
        self.penalty.gen_presence_penalty = value

    @property
    def gen_stop_sequences(self) -> Optional[List[str]]:
        """Return stop sequences that should override the config defaults."""
        return self.penalty.gen_stop_sequences

    @gen_stop_sequences.setter
    def gen_stop_sequences(self, value: Optional[List[str]]) -> None:
        self.penalty.gen_stop_sequences = value

    def as_dict(self) -> Dict[str, Any]:
        """Return a lightweight representation useful for logging/debugging."""
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
        if len(self.target_counts) != len(self.prompts):
            raise ValueError("target_counts must align with prompts for vLLM state")
        self.aggregated = [[] for _ in self.prompts]
        self.aggregated_meta = (
            [[] for _ in self.prompts] if self.track_logprobs else None
        )

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


class CompletionGenerator:
    """Stateful helper that handles both local HF and vLLM completions."""

    def __init__(self, ctx: GenerationContext) -> None:
        self.ctx = ctx
        self._vllm_client = None
        self._vllm_sync_ready = False
        self._last_vllm_synced_step: Optional[int] = None
        self._fsdp_cls = getattr(getattr(torch, "distributed", None), "fsdp", None)
        if self._fsdp_cls is not None:
            self._fsdp_cls = getattr(self._fsdp_cls, "FullyShardedDataParallel", None)

    def _vllm_base_url(self, url: str) -> str:
        """Strip common /generate suffixes to yield the server base URL."""
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
        """Instantiate the TRL VLLMClient when weight sync is enabled."""
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
        if current_step is not None and self._last_vllm_synced_step == int(current_step):
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
        """Best-effort parameter broadcast mirroring HF GRPO's vLLM path."""
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
        if param is None or self._vllm_client is None:
            return
        update_raw = getattr(self._vllm_client, "update_named_param", None)
        if not callable(update_raw):
            return
        update_fn = cast(Callable[[str, Any], None], update_raw)
        try:
            update_fn(name, param.data)  # type: ignore[union-attr]  # pylint: disable=not-callable
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - network/dependency dependent
            LOG.warning("Failed to push param %s to vLLM: %s", name, exc)

    def _reset_vllm_cache(self) -> None:
        """Reset prefix caches when the vLLM client exposes the helper."""
        client = self._vllm_client
        reset_raw = getattr(client, "reset_prefix_cache", None)
        if not callable(reset_raw):
            return
        reset_fn = cast(Callable[[], Any], reset_raw)
        try:
            reset_fn()  # pylint: disable=not-callable
        except (AttributeError, RuntimeError):
            pass

    def _sync_fsdp_params(self, model: Any) -> None:
        fsdp_cls = self._fsdp_cls
        if fsdp_cls is None or not isinstance(model, fsdp_cls):
            return
        visited: Set[str] = set()

        def _sync_tree(module: Any, prefix: str = "") -> None:
            for child_name, child in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                _sync_tree(child, child_prefix)
            if isinstance(module, fsdp_cls):
                with fsdp_cls.summon_full_params(module, recurse=False, writeback=False):  # type: ignore[attr-defined]
                    for pname, param in module.named_parameters():
                        full_name = f"{prefix}.{pname}" if prefix else pname
                        for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
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
        params = list(model.parameters())
        merge_fn = getattr(model, "merge_adapter", None)
        unmerge_fn = getattr(model, "unmerge_adapter", None)
        with gather_factory(params):
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

    def _sync_standard_params(
        self,
        model: Any,
        gather_factory: Callable[[Sequence[Any]], AbstractContextManager[Any]],
    ) -> None:
        params = list(model.parameters())
        with gather_factory(params):
            for name, param in model.named_parameters():
                self._push_param_to_vllm(name, param)

    def describe(self) -> Dict[str, Any]:
        """Expose the underlying generation configuration for logging."""
        return self.ctx.as_dict()

    def _resolve_vllm_round_limit(self, requested_n: int) -> int:
        ctx = self.ctx
        if ctx.vllm_rounds_cfg > 0:
            return max(1, ctx.vllm_rounds_cfg)
        return max(1, requested_n)

    def _build_local_prompt_requests(
        self,
        prompts: List[str],
        target_counts: List[int],
    ) -> Tuple[List[str], List[int]]:
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
        grouped: List[List[str]] = [[] for _ in prompts]
        if not prompts:
            return grouped, None
        char_limit = self._prompt_char_limit()
        prompts = [_truncate_prompt(prompt, char_limit) for prompt in prompts]
        target_counts = self._resolve_local_counts(prompts, num_samples, per_prompt_counts)
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
        """Return a compact preview of grouped completions."""
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
            (
                "vLLM grouped outputs len=%d vs pending=%d | per-prompt lengths=%s"
            ),
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

    @staticmethod
    def _expand_dedup_results(
        grouped: List[List[str]],
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        mapping: Optional[List[int]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
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

        scatter_payload: Optional[List[Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]]] = None
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
        """Produce completions, preferring vLLM when configured."""
        if not prompts:
            return [], None
        if per_prompt_counts is not None and len(per_prompt_counts) != len(prompts):
            raise ValueError(
                "per_prompt_counts length must match prompts length in generate()"
            )
        if self.ctx.use_vllm:
            return self._generate_vllm_collective(prompts, num_samples, per_prompt_counts)
        return self._generate_local(prompts, num_samples, per_prompt_counts)


__all__ = ["CompletionGenerator", "GenerationContext"]
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
    accelerator: Accelerator,
    payload: List[Any],
    *,
    src: int = 0,
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
    return input_list[accelerator.process_index]
