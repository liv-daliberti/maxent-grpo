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
import time

import maxent_grpo.training.generation.vllm_adapter as _vllm_adapter
from maxent_grpo.patches.vllm import safe_generate
from maxent_grpo.generation.vllm import VLLMGenerationHelper
from maxent_grpo.training.runtime import require_torch
from maxent_grpo.training.runtime.prompts import PROMPT_CHAR_LIMIT, _truncate_prompt
from maxent_grpo.generation.common import (
    AggregatedGenerationState as _AggregatedGenerationState,
    append_completion_group as _append_completion_group,
    determine_retry_limit as _determine_retry_limit,
    pending_generation_indices as _pending_generation_indices,
    retry_incomplete_prompts as _retry_incomplete_prompts,
    seed_generation_groups as _seed_generation_groups_impl,
)
from maxent_grpo.utils.fallbacks import dist_with_fallback

from .context import GenerationContext
from .local import LocalGenerationMixin
from .vllm_adapter import (
    VLLMGenerationMixin,
    _VLLMGenerationState,
    _is_peft_model_safe,
    _optional_import,
    _zero3_gather_factory,
    _import_vllm_client_cls as _adapter_import_vllm_client_cls,
)

torch = require_torch("generation")
_retry_incomplete_prompts_impl = _retry_incomplete_prompts

# Recreate the dist fallback on reload to pick up the current torch stub.
dist = dist_with_fallback(getattr(torch, "distributed", None))


def _refresh_vllm_globals() -> None:
    """Keep vLLM adapter globals in sync with test monkeypatches."""

    _vllm_adapter.dist = dist
    _vllm_adapter.safe_generate = safe_generate
    _vllm_adapter.time = importlib.import_module("time")
    globals()["_retry_incomplete_prompts_impl"] = _retry_incomplete_prompts


def _gather_object_list_wrapper(accelerator, value):
    # pylint: disable=protected-access
    _refresh_vllm_globals()
    return _vllm_adapter._gather_object_list(accelerator, value)


def _broadcast_object_list_wrapper(accelerator, payload, *, src=0):
    # pylint: disable=protected-access
    _refresh_vllm_globals()
    return _vllm_adapter._broadcast_object_list(accelerator, payload, src=src)


def _scatter_object_wrapper(accelerator, input_list, *, src=0):
    # pylint: disable=protected-access
    _refresh_vllm_globals()
    return _vllm_adapter._scatter_object(accelerator, input_list, src=src)


# Expose wrapper functions that honor patched globals.
_broadcast_object_list = _broadcast_object_list_wrapper
_gather_object_list = _gather_object_list_wrapper
_scatter_object = _scatter_object_wrapper


def _import_vllm_client_cls(import_fn=None):
    """Import the TRL VLLMClient using the caller-provided optional import hook."""

    resolved_import = import_fn or globals().get("_optional_import") or _optional_import
    return _adapter_import_vllm_client_cls(resolved_import)


class CompletionGenerator(LocalGenerationMixin, VLLMGenerationMixin):
    """Stateful helper that handles both local HF and vLLM completions."""

    def __init__(self, ctx: GenerationContext) -> None:
        LocalGenerationMixin.__init__(self, ctx)
        VLLMGenerationMixin.__init__(self, ctx)


__all__ = [
    "VLLMGenerationHelper",
    "CompletionGenerator",
    "GenerationContext",
    "PROMPT_CHAR_LIMIT",
    "_truncate_prompt",
    "require_torch",
    "safe_generate",
    "torch",
    "time",
    "dist",
    "_AggregatedGenerationState",
    "_append_completion_group",
    "_determine_retry_limit",
    "_pending_generation_indices",
    "_retry_incomplete_prompts",
    "_retry_incomplete_prompts_impl",
    "_seed_generation_groups_impl",
    "_VLLMGenerationState",
    "_broadcast_object_list",
    "_gather_object_list",
    "_import_vllm_client_cls",
    "_is_peft_model_safe",
    "_optional_import",
    "_scatter_object",
    "_zero3_gather_factory",
]
