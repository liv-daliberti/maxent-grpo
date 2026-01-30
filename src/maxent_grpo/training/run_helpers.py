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

import logging
from typing import Any, List, Tuple, TYPE_CHECKING, cast

from maxent_grpo.training.runtime import (
    PROMPT_CHAR_LIMIT,
    _TRUNC_STATE,
    _maybe_create_deepspeed_plugin,
    _prompt_char_limit_from_tokens,
    _report_to_contains,
    _to_prompt,
    _wandb_error_types,
    truncate_prompt,
    ChatTokenizer,
    GenerationPenaltyConfig,
    GenerationPenaltyPassthroughMixin,
    GenerationSamplingConfig,
    MaxEntOptions,
    SeedAugmentationConfig,
    VLLMClientConfig,
    get_trl_prepare_deepspeed,
    require_accelerator,
    require_dataloader,
    require_deepspeed,
    require_torch,
    require_transformer_base_classes,
)

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from torch import Tensor
else:  # pragma: no cover - runtime fallback
    Tensor = Any


def _group_softmax(
    values: List[float],
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> List[float]:
    """Numerically stable softmax with optional temperature and epsilon floor."""

    if len(values) == 0:
        return []
    torch_module = require_torch("softmax weighting")
    try:
        value_tensor = torch_module.tensor(
            values, dtype=getattr(torch_module, "float32", None)
        )
        value_tensor = value_tensor / max(temperature, 1e-8)
        value_tensor = value_tensor - value_tensor.max()
    except (TypeError, ValueError, RuntimeError):
        value_tensor = None
    softmax_fn = getattr(torch_module, "softmax", None)
    if value_tensor is None or not callable(softmax_fn):
        try:
            import numpy as _np

            arr = _np.array(values, dtype=float)
            arr = arr / max(temperature, 1e-8)
            arr = arr - arr.max()
            exps = _np.exp(arr)
            denom = exps.sum() if exps.size else 1.0
            probs_arr = exps / denom
            probs = torch_module.tensor(
                probs_arr.tolist(),
                dtype=getattr(torch_module, "float32", None),
            )
        except (ImportError, TypeError, ValueError, OverflowError):
            probs = torch_module.full((len(values),), 1.0 / max(len(values), 1))
    else:
        probs = softmax_fn(value_tensor, dim=0)
    if isinstance(probs, list):
        try:
            import numpy as _np

            probs = _np.array(probs, dtype=float)
        except (ImportError, TypeError, ValueError):
            LOG.debug("Failed to coerce softmax probs to numpy array.")
    try:
        probs_any = cast(Any, probs)
        probs_any = probs_any * (1.0 - eps * len(values)) + eps
        probs_any = probs_any / probs_any.sum()
        return probs_any.tolist()
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        try:
            # Fallback when probs is still a sequence, e.g., torch stub list.
            import numpy as _np

            arr = _np.array(probs, dtype=float)
            arr = arr * (1.0 - eps * len(values)) + eps
            arr = arr / arr.sum()
            return arr.tolist()
        except (ImportError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            return [1.0 / max(len(values), 1)] * len(values)


def _prepare_labels_for_ce(
    input_ids: "Tensor",
    prompt_lengths: List[int],
) -> "Tensor":
    """Create labels tensor with prompt tokens masked as -100 for CE."""

    labels = input_ids.clone()
    ndim = getattr(labels, "ndim", None)
    if ndim is not None and ndim < 2:
        # Lift 1D inputs to at least 2D so masking logic does not throw.
        try:
            labels = labels.view(max(len(prompt_lengths), 1), -1)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            try:
                labels = labels.unsqueeze(0)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return labels
    for i, plen in enumerate(prompt_lengths):
        try:
            labels[i, :plen] = -100
        except (IndexError, TypeError, ValueError):
            try:
                labels[i][:plen] = -100
            except (AttributeError, IndexError, TypeError, ValueError):
                # Best-effort masking for stub tensors that lack full indexing support.
                LOG.debug("Failed to mask labels for prompt index %s", i)
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
    "SeedAugmentationConfig",
    "VLLMClientConfig",
    "PROMPT_CHAR_LIMIT",
    "_TRUNC_STATE",
    "truncate_prompt",
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
    "_wandb_error_types",
    "_report_to_contains",
    "_prompt_char_limit_from_tokens",
]
