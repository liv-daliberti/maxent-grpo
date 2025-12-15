"""Local HF generation helpers split from the vLLM adapter."""

from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Any, List, Optional, Tuple

from maxent_grpo.training.runtime import require_torch, require_transformer_base_classes
from maxent_grpo.training.runtime.prompts import PROMPT_CHAR_LIMIT, _truncate_prompt

from .context import GenerationContext

LOG = logging.getLogger(__name__)

torch = require_torch("generation")
PreTrainedModel, PreTrainedTokenizer = require_transformer_base_classes("generation")


class LocalGenerationMixin:
    """Handle prompt expansion, tokenization, and local HF sampling."""

    ctx: GenerationContext

    def __init__(self, ctx: GenerationContext) -> None:
        self.ctx = ctx

    def describe(self) -> dict[str, Any]:
        """Expose the underlying generation configuration for logging."""
        return self.ctx.as_dict()

    def _prompt_char_limit(self) -> int:
        """Return the character limit applied to prompts for vLLM/local calls."""
        try:
            helpers_mod = __import__(
                "maxent_grpo.training.generation.helpers",
                fromlist=["PROMPT_CHAR_LIMIT"],
            )
            limit_base = getattr(helpers_mod, "PROMPT_CHAR_LIMIT", PROMPT_CHAR_LIMIT)
        except ImportError:
            limit_base = PROMPT_CHAR_LIMIT
        approx_chars = 0
        if self.ctx.max_prompt_len and self.ctx.max_prompt_len > 0:
            approx_chars = int(self.ctx.max_prompt_len * 4)
        if limit_base <= 0:
            return approx_chars
        if approx_chars <= 0:
            return limit_base
        return max(limit_base, approx_chars)

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
            prompt_lengths = (
                encoder_inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()
            )
            return encoder_inputs, prompt_lengths

        # Fallback for lightweight stubs that only provide ``decode``.
        lengths = [len(p) for p in expanded_prompts]

        class _Mask:
            def __init__(self, vals):
                self._vals = vals

            def sum(self, _dim=1):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self._vals)

        class _Inputs(dict):
            def __init__(self, lens):
                super().__init__(attention_mask=_Mask(lens))

            def to(self, _device):
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
        LOG.debug(
            "HF generate start | model=%s | max_new_tokens=%s | temp=%.3f | top_p=%.3f | top_k=%s",
            gen_model.__class__.__name__ if gen_model is not None else "None",
            self.ctx.max_completion_len,
            self.ctx.gen_temperature,
            self.ctx.gen_top_p,
            self.ctx.gen_top_k,
        )
        with no_grad():
            generate_fn = getattr(gen_model, "generate", None)
            if callable(generate_fn):
                gen_out = generate_fn(
                    **encoder_inputs,
                    do_sample=True,
                    temperature=self.ctx.gen_temperature,
                    top_p=self.ctx.gen_top_p,
                    top_k=(
                        self.ctx.gen_top_k if self.ctx.gen_top_k is not None else None
                    ),
                    max_new_tokens=self.ctx.max_completion_len,
                    num_return_sequences=1,
                )
            else:
                # Fallback for lightweight stubs without generation support.
                gen_out = encoder_inputs
        return self._decode_sequences(gen_out, prompt_lengths, self.ctx.tokenizer)

    def _generate_local(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
        """Generate completions using the local HF model."""
        try:
            helpers_mod = __import__(
                "maxent_grpo.training.generation.helpers", fromlist=["_truncate_prompt"]
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
        sequences: torch.Tensor,
        prompt_lengths: List[int],
        tokenizer: PreTrainedTokenizer,
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
