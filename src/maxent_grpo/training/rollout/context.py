"""Shared generation context dataclass used by local and vLLM paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from maxent_grpo.training.runtime import GenerationSamplingConfig
from maxent_grpo.training.runtime.prompts import (
    GenerationPenaltyConfig,
    GenerationPenaltyPassthroughMixin,
)
from ..types import (
    Accelerator as TypesAccelerator,
    PreTrainedModel as TypesPreTrainedModel,
    PreTrainedTokenizer as TypesPreTrainedTokenizer,
)


@dataclass
class GenerationContext(GenerationPenaltyPassthroughMixin, GenerationSamplingConfig):
    """Configuration required to produce completions for each training batch."""

    accelerator: TypesAccelerator
    model: TypesPreTrainedModel
    tokenizer: TypesPreTrainedTokenizer
    generation_stats: Dict[str, int]
    device: Any
    penalty: GenerationPenaltyConfig = field(default_factory=GenerationPenaltyConfig)
    prompt_char_limit: int | None = None

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


__all__ = ["GenerationContext"]
