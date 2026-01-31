"""vLLM-specific helpers extracted from the MaxEnt-GRPO generation module."""

from __future__ import annotations

from contextlib import nullcontext as _stdlib_nullcontext

from maxent_grpo.training.runtime import require_torch
from .errors import GenerationServiceError as _GenerationServiceError
from .vllm_helper import (
    VLLMGenerationHelper,
    _VLLMGenerationState,
)

PROMPT_CHAR_LIMIT = 2048

torch = require_torch("generation_vllm")


def nullcontext(enter_result=None):
    """Return a no-op context manager.

    This wrapper avoids Sphinx parsing issues with the stdlib docstring.
    """

    return _stdlib_nullcontext(enter_result)


__all__ = [
    "VLLMGenerationHelper",
    "_VLLMGenerationState",
    "GenerationServiceError",
    "nullcontext",
]
GenerationServiceError = _GenerationServiceError
