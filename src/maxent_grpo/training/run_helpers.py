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

"""Shared helper utilities for the MaxEnt-GRPO training pipeline.

This module re-exports common runtime dependency/prompt helpers while keeping
lightweight tensor utilities used in scoring and loss computation. Logging
helpers now live in :mod:`maxent_grpo.training.runtime.logging`.
"""

from __future__ import annotations

from typing import Any, List, Tuple, TYPE_CHECKING

from maxent_grpo.training.runtime import (
    PROMPT_CHAR_LIMIT,
    _TRUNC_STATE,
    _build_torch_stub,
    _import_module,
    _maybe_create_deepspeed_plugin,
    _optional_dependency,
    _prompt_char_limit_from_tokens,
    _report_to_contains,
    _require_dependency,
    _to_prompt,
    _wandb_error_types,
    truncate_prompt,
    ChatTokenizer,
    GenerationPenaltyConfig,
    GenerationPenaltyPassthroughMixin,
    GenerationSamplingConfig,
    MaxEntOptions,
    VLLMClientConfig,
    get_trl_prepare_deepspeed,
    require_accelerator,
    require_dataloader,
    require_deepspeed,
    require_torch,
    require_transformer_base_classes,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from torch import Tensor
else:  # pragma: no cover - runtime fallback
    Tensor = Any

# Backwards compatibility for legacy imports that referenced the private alias.
_truncate_prompt = truncate_prompt


def _group_softmax(
    values: List[float],
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> List[float]:
    """Numerically stable softmax with optional temperature and epsilon floor."""

    if len(values) == 0:
        return []
    torch_module = _require_dependency(
        "torch",
        (
            "MaxEnt softmax weighting requires PyTorch. "
            "Install it via `pip install torch`."
        ),
    )
    value_tensor = torch_module.tensor(values, dtype=torch_module.float32)
    value_tensor = value_tensor / max(temperature, 1e-8)
    value_tensor = value_tensor - value_tensor.max()
    probs = torch_module.softmax(value_tensor, dim=0)
    probs = probs * (1.0 - eps * len(values)) + eps
    probs = probs / probs.sum()
    return probs.tolist()


def _prepare_labels_for_ce(
    input_ids: "Tensor",
    prompt_lengths: List[int],
) -> "Tensor":
    """Create labels tensor with prompt tokens masked as -100 for CE."""

    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lengths):
        labels[i, :plen] = -100
    return labels


def _batch_tokenize_pairs(
    tokenizer: Any,
    prompts: List[str],
    completions: List[str],
) -> Tuple["Tensor", "Tensor", List[int]]:
    """Tokenize prompt+completion pairs and return tensors + prompt lengths."""

    pairs = [p + c for p, c in zip(prompts, completions)]
    enc_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    prompt_lengths = (
        enc_prompts["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1).tolist()
    )
    enc = tokenizer(
        pairs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    return input_ids, attn, prompt_lengths


__all__ = [
    "ChatTokenizer",
    "GenerationSamplingConfig",
    "GenerationPenaltyConfig",
    "GenerationPenaltyPassthroughMixin",
    "MaxEntOptions",
    "VLLMClientConfig",
    "PROMPT_CHAR_LIMIT",
    "_TRUNC_STATE",
    "truncate_prompt",
    "_truncate_prompt",
    "require_accelerator",
    "require_dataloader",
    "require_torch",
    "require_transformer_base_classes",
    "require_deepspeed",
    "get_trl_prepare_deepspeed",
    "_batch_tokenize_pairs",
    "_group_softmax",
    "_maybe_create_deepspeed_plugin",
    "_prepare_labels_for_ce",
    "_to_prompt",
    "_import_module",
    "_require_dependency",
    "_optional_dependency",
    "_wandb_error_types",
    "_report_to_contains",
    "_prompt_char_limit_from_tokens",
    "_build_torch_stub",
    "_truncate_prompt",
]
