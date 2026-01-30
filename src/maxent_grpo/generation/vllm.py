"""vLLM-specific helpers extracted from the MaxEnt-GRPO generation module."""

from __future__ import annotations

from maxent_grpo.training.runtime import require_torch
from .errors import GenerationServiceError as _GenerationServiceError
from .vllm_helper import (
    VLLMGenerationHelper,
    _VLLMGenerationState,
)

PROMPT_CHAR_LIMIT = 2048

torch = require_torch("generation_vllm")

__all__ = [
    "VLLMGenerationHelper",
    "_VLLMGenerationState",
    "GenerationServiceError",
]
GenerationServiceError = _GenerationServiceError
