"""Local HF generation helpers split from the vLLM adapter."""

from __future__ import annotations

from contextlib import nullcontext
import logging
import os
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, cast

from maxent_grpo.training.runtime import require_torch, require_transformer_base_classes
from maxent_grpo.training.runtime.prompts import PROMPT_CHAR_LIMIT, _truncate_prompt

from .context import GenerationContext

LOG = logging.getLogger(__name__)

torch = require_torch("generation")
try:
    PreTrainedModel, PreTrainedTokenizer = require_transformer_base_classes("generation")
except (ImportError, RuntimeError, ModuleNotFoundError):  # pragma: no cover - stub fallback
    PreTrainedModel = Any
    PreTrainedTokenizer = Any

if TYPE_CHECKING:
    import torch as torch_types
    from transformers.tokenization_utils import (
        PreTrainedTokenizer as PreTrainedTokenizerType,
    )

    Tensor = torch_types.Tensor
else:
    PreTrainedTokenizerType = Any
    Tensor = Any


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class LocalGenerationMixin:
    """Handle prompt expansion, tokenization, and local HF sampling."""

    ctx: GenerationContext

    def __init__(self, ctx: GenerationContext) -> None:
        self.ctx = ctx

    def describe(self) -> dict[str, Any]:
        """Expose the underlying generation configuration for logging."""
        return self.ctx.as_dict()

    def _prompt_char_limit(self) -> int:
        """Return the token limit applied to prompts for vLLM/local calls."""
        try:
            helpers_mod = __import__(
                "maxent_grpo.training.rollout.helpers",
                fromlist=["PROMPT_CHAR_LIMIT"],
            )
            limit_base = getattr(helpers_mod, "PROMPT_CHAR_LIMIT", PROMPT_CHAR_LIMIT)
        except ImportError:
            limit_base = PROMPT_CHAR_LIMIT
        if self.ctx.max_prompt_len and self.ctx.max_prompt_len > 0:
            return int(self.ctx.max_prompt_len)
        return int(limit_base) if limit_base is not None else 0

    def _build_local_prompt_requests(
        self,
        prompts: List[str],
        target_counts: List[int],
    ) -> Tuple[List[str], List[int]]:
        """Expand prompts by their requested counts for local sampling."""
        expanded_prompts: List[str] = []
        prompt_indices: List[int] = []
        for idx, (prompt, target_count) in enumerate(zip(prompts, target_counts)):
            adjusted_target = max(0, int(target_count))
            if adjusted_target <= 0:
                continue
            expanded_prompts.extend([prompt] * adjusted_target)
            prompt_indices.extend([idx] * adjusted_target)
        return expanded_prompts, prompt_indices

    def _tokenize_expanded_prompts(
        self,
        expanded_prompts: List[str],
    ) -> Tuple[Any, List[int]]:
        """Tokenize prompts for local generation and track prompt lengths."""
        tokenizer = self.ctx.tokenizer
        if callable(tokenizer):
            try:
                encoder_inputs = tokenizer(
                    expanded_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.ctx.max_prompt_len,
                )
            except TypeError:
                encoder_inputs = tokenizer(expanded_prompts)
            if hasattr(encoder_inputs, "to"):
                encoder_inputs = encoder_inputs.to(self.ctx.device)
            mask = cast(Any, encoder_inputs["attention_mask"])
            prompt_lengths = mask.sum(dim=1).detach().cpu().tolist()
            return encoder_inputs, prompt_lengths

        # Fallback for lightweight stubs that only provide ``decode``.
        lengths = [len(p) for p in expanded_prompts]

        class _Mask:
            def __init__(self, vals: List[int]) -> None:
                self._vals = vals

            def sum(self, _dim: int = 1) -> "_Mask":
                return self

            def detach(self) -> "_Mask":
                return self

            def cpu(self) -> "_Mask":
                return self

            def tolist(self) -> List[int]:
                return list(self._vals)

        class _Inputs(dict):
            def __init__(self, lens: List[int]) -> None:
                super().__init__(attention_mask=_Mask(lens))

            def to(self, _device: Any) -> "_Inputs":
                return self

        return _Inputs(lengths), lengths

    def _run_local_model(
        self,
        encoder_inputs: Any,
        prompt_lengths: List[int],
    ) -> List[str]:
        """Run the HF model locally and decode completions."""
        unwrap = getattr(self.ctx.accelerator, "unwrap_model", None)
        gen_model = unwrap(self.ctx.model) if callable(unwrap) else self.ctx.model
        no_grad = getattr(torch, "no_grad", None) or nullcontext
        dist = getattr(torch, "distributed", None)
        dist_initialized = bool(
            dist
            and hasattr(dist, "is_available")
            and hasattr(dist, "is_initialized")
            and dist.is_available()
            and dist.is_initialized()
        )
        synced_gpus = _env_flag("MAXENT_LOCAL_SYNCED_GPUS", False)
        disable_dynamo = _env_flag("MAXENT_LOCAL_DISABLE_DYNAMO", dist_initialized)
        max_new_tokens = self.ctx.max_completion_len
        max_time: Optional[float] = None
        env_max_new = os.getenv("MAXENT_LOCAL_MAX_NEW_TOKENS")
        if env_max_new is not None:
            try:
                cap_val = int(env_max_new)
                if cap_val > 0:
                    max_new_tokens = min(int(max_new_tokens), cap_val)
            except (TypeError, ValueError):
                LOG.warning(
                    "Invalid MAXENT_LOCAL_MAX_NEW_TOKENS=%r; using max_new_tokens=%s",
                    env_max_new,
                    max_new_tokens,
                )
        env_max_time = os.getenv("MAXENT_LOCAL_MAX_TIME_S")
        if env_max_time is not None:
            try:
                max_time_val = float(env_max_time)
                if max_time_val > 0:
                    max_time = max_time_val
            except (TypeError, ValueError):
                LOG.warning(
                    "Invalid MAXENT_LOCAL_MAX_TIME_S=%r; ignoring max_time override.",
                    env_max_time,
                )
        empty_cache = _env_flag("MAXENT_LOCAL_EMPTY_CACHE", False)
        if empty_cache and hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        dynamo_ctx = nullcontext()
        if disable_dynamo:
            dynamo = getattr(torch, "_dynamo", None)
            disable_fn = getattr(dynamo, "disable", None) if dynamo is not None else None
            if callable(disable_fn):
                dynamo_ctx = disable_fn()
        LOG.debug(
            "HF generate start | model=%s | max_new_tokens=%s | max_time=%s | temp=%.3f | top_p=%.3f | top_k=%s | synced_gpus=%s | disable_dynamo=%s | empty_cache=%s",
            gen_model.__class__.__name__ if gen_model is not None else "None",
            max_new_tokens,
            max_time,
            self.ctx.gen_temperature,
            self.ctx.gen_top_p,
            self.ctx.gen_top_k,
            synced_gpus,
            disable_dynamo,
            empty_cache,
        )
        with no_grad(), dynamo_ctx:
            generate_fn = getattr(gen_model, "generate", None)
            if callable(generate_fn):
                gen_cfg = getattr(gen_model, "generation_config", None)
                if gen_cfg is not None and hasattr(gen_cfg, "synced_gpus"):
                    try:
                        setattr(gen_cfg, "synced_gpus", bool(synced_gpus))
                    except Exception:
                        pass
                generate_kwargs = dict(
                    do_sample=True,
                    temperature=self.ctx.gen_temperature,
                    top_p=self.ctx.gen_top_p,
                    top_k=(
                        self.ctx.gen_top_k if self.ctx.gen_top_k is not None else None
                    ),
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    synced_gpus=synced_gpus,
                )
                if max_time is not None:
                    generate_kwargs["max_time"] = max_time
                try:
                    gen_out = generate_fn(**encoder_inputs, **generate_kwargs)
                except TypeError as exc:
                    msg = str(exc)
                    retry = False
                    if "synced_gpus" in msg:
                        generate_kwargs.pop("synced_gpus", None)
                        retry = True
                    if "max_time" in msg:
                        generate_kwargs.pop("max_time", None)
                        retry = True
                    if retry:
                        gen_out = generate_fn(**encoder_inputs, **generate_kwargs)
                    else:
                        raise
            else:
                # Fallback for lightweight stubs without generation support.
                gen_out = encoder_inputs
        gen_out_any = cast(Any, gen_out)
        return self._decode_sequences(gen_out_any, prompt_lengths, self.ctx.tokenizer)

    def _generate_local(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
        """Generate completions using the local HF model."""
        try:
            helpers_mod = __import__(
                "maxent_grpo.training.rollout.helpers", fromlist=["_truncate_prompt"]
            )
            trunc_fn = getattr(helpers_mod, "_truncate_prompt", _truncate_prompt)
        except ImportError:
            trunc_fn = _truncate_prompt
        grouped: List[List[str]] = [[] for _ in prompts]
        if not prompts:
            return grouped, None
        char_limit = self._prompt_char_limit()
        prompts = [trunc_fn(prompt, char_limit) for prompt in prompts]
        target_counts = self._resolve_local_counts(
            prompts, num_samples, per_prompt_counts
        )
        LOG.debug(
            "Local generation | prompts=%d | num_samples=%d | char_limit=%d | per_prompt_counts=%s",
            len(prompts),
            num_samples,
            char_limit,
            f"len={len(target_counts)}" if target_counts is not None else "none",
        )
        expanded_prompts, prompt_indices = self._build_local_prompt_requests(
            prompts,
            target_counts,
        )
        if not expanded_prompts:
            return grouped, None
        enc_inputs, prompt_lengths = self._tokenize_expanded_prompts(expanded_prompts)
        LOG.debug(
            "Local generation tokenize | expanded_prompts=%d | prompt_indices=%d | prompt_lengths_sample=%s",
            len(expanded_prompts),
            len(prompt_indices),
            prompt_lengths[: min(3, len(prompt_lengths))],
        )
        decoded = self._run_local_model(enc_inputs, prompt_lengths)
        LOG.debug(
            "Local generation decode done | decoded=%d | first_prompt_count=%d",
            len(decoded),
            len(grouped[0]) if grouped else 0,
        )
        for text, prompt_idx in zip(decoded, prompt_indices):
            grouped[prompt_idx].append(text)
        return grouped, None

    @staticmethod
    def _resolve_local_counts(
        prompts: List[str],
        default_count: int,
        overrides: Optional[List[int]],
    ) -> List[int]:
        """Resolve per-prompt generation counts for local sampling."""
        if overrides is None:
            return [default_count] * len(prompts)
        if len(overrides) != len(prompts):
            raise ValueError("per_prompt_counts length must match prompts length")
        return overrides

    @staticmethod
    def _decode_sequences(
        sequences: Any,
        prompt_lengths: List[int],
        tokenizer: PreTrainedTokenizerType,
    ) -> List[str]:
        """Decode model outputs into completion strings."""
        outputs: List[str] = []
        for row, prompt_len in zip(sequences, prompt_lengths):
            completion_ids = row[int(prompt_len) :]
            try:
                outputs.append(
                    tokenizer.decode(completion_ids, skip_special_tokens=True)
                )
            except AttributeError:
                # Minimal tokenizer fallback: stringify the ids.
                try:
                    outputs.append(" ".join(str(int(tok)) for tok in completion_ids))
                except (TypeError, ValueError):
                    outputs.append(str(completion_ids))
        return outputs


__all__ = ["LocalGenerationMixin"]
