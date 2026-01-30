"""Public CompletionGenerator that wires local and vLLM helpers together."""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from maxent_grpo.patches.vllm import VLLMLogprobResult, safe_generate

from .context import GenerationContext
from .distributed import _scatter_object
from .local import LocalGenerationMixin
from .vllm_adapter import (
    VLLMGenerationHelper,
    VLLMGenerationMixin,
    _is_peft_model_safe,
)

LOG = logging.getLogger(__name__)


class CompletionGenerator(LocalGenerationMixin, VLLMGenerationMixin):
    """Stateful helper that handles both local HF and vLLM completions."""

    def __init__(self, ctx: GenerationContext) -> None:
        LocalGenerationMixin.__init__(self, ctx)
        if hasattr(ctx, "accelerator"):
            try:
                VLLMGenerationMixin.__init__(self, ctx)
            except (ImportError, RuntimeError, AttributeError, ValueError):
                self._vllm_helper = None
        else:
            self._vllm_helper = None
        helper_cls = globals().get("VLLMGenerationHelper", VLLMGenerationHelper)
        self._vllm_helper = helper_cls(ctx, self._generate_local)
        # Surface patchable hooks for tests so monkeypatched helpers.* propagate.
        self._vllm_helper._safe_generate = safe_generate
        self._vllm_helper._scatter_object = _scatter_object
        self._vllm_helper._time = time
        self._vllm_helper._is_peft_model_safe = _is_peft_model_safe
        self._vllm_helper._fallback_generate = self._generate_local

    def describe(self) -> Dict[str, Any]:
        """Expose the underlying generation configuration for logging."""
        return self.ctx.as_dict()

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
        LOG.debug(
            "CompletionGenerator.generate | prompts=%d | num_samples=%d | use_vllm=%s | per_prompt_counts=%s",
            len(prompts),
            num_samples,
            getattr(self.ctx, "use_vllm", False),
            f"len={len(per_prompt_counts)}" if per_prompt_counts is not None else "none",
        )
        if self.ctx.use_vllm:
            return self._generate_vllm_collective(
                prompts, num_samples, per_prompt_counts
            )
        LOG.debug("CompletionGenerator.generate using local HF path")
        return self._generate_local(prompts, num_samples, per_prompt_counts)


__all__ = [
    "CompletionGenerator",
    "GenerationContext",
    "safe_generate",
    "_scatter_object",
    "_is_peft_model_safe",
    "time",
]
