"""vLLM-specific helpers extracted from the MaxEnt-GRPO generation module."""

from __future__ import annotations

from contextlib import nullcontext as _nullcontext
import time as _time

from maxent_grpo.patches.vllm import safe_generate as _safe_generate
from maxent_grpo.training.runtime import require_torch
from .vllm_distributed import (
    VLLMDistributedMixin as _VLLMDistributedMixin,
    _gather_object_list as _imported_gather_object_list,
    _scatter_object as _imported_scatter_object,
)
from .errors import GenerationServiceError as _GenerationServiceError
from .vllm_helper import (
    VLLMGenerationHelper,
    _VLLMGenerationState,
    _optional_import as _imported_optional_import,
)
from .vllm_requests import VLLMRequestMixin as _VLLMRequestMixin
from .vllm_weight_sync import (
    VLLMWeightSyncMixin as _VLLMWeightSyncMixin,
    _ClientCallable as _ImportedClientCallable,
    _import_vllm_client_cls as _imported_vllm_client_cls,
    _is_peft_model_safe as _imported_is_peft_model_safe,
    _zero3_gather_factory as _imported_zero3_gather_factory,
)

PROMPT_CHAR_LIMIT = 2048

torch = require_torch("generation_vllm")

__all__ = [
    "VLLMGenerationHelper",
    "_VLLMGenerationState",
    "GenerationServiceError",
    "VLLMServiceError",
]

# Backwards compatibility for callers/tests importing private helpers directly.
_gather_object_list = _imported_gather_object_list
_scatter_object = _imported_scatter_object
_ClientCallable = _ImportedClientCallable
_import_vllm_client_cls = _imported_vllm_client_cls
_is_peft_model_safe = _imported_is_peft_model_safe
_optional_import = _imported_optional_import
_zero3_gather_factory = _imported_zero3_gather_factory
VLLMRequestMixin = _VLLMRequestMixin
VLLMWeightSyncMixin = _VLLMWeightSyncMixin
VLLMDistributedMixin = _VLLMDistributedMixin
nullcontext = _nullcontext
safe_generate = _safe_generate
time = _time
GenerationServiceError = _GenerationServiceError
VLLMServiceError = _GenerationServiceError
