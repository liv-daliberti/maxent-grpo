"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tokenizer/model loading helpers for training scripts.

This module exposes two utilities:

- ``get_tokenizer``: Load a tokenizer with optional chat template override. A
  minimal fallback tokenizer is provided for offline/CI environments.
- ``get_model``: Load an ``AutoModelForCausalLM`` with optional quantization
  and device map resolution via TRL helpers, respecting attention impl/dtype
  choices and gradient checkpointing compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict, TYPE_CHECKING, Union

import torch

from maxent_grpo.utils.stubs import (
    AutoModelForCausalLMStub,
    AutoTokenizerStub,
    PreTrainedTokenizerStub,
)

try:  # pragma: no cover - optional dependency (offline/CI fallback)
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
except (
    ImportError,
    RuntimeError,
    AttributeError,
):  # degrade gracefully when transformers partially missing
    AutoTokenizer = AutoTokenizerStub
    PreTrainedTokenizer = PreTrainedTokenizerStub
    AutoModelForCausalLM = AutoModelForCausalLMStub

try:
    from trl import ModelConfig, get_kbit_device_map, get_quantization_config
except (
    ImportError,
    RuntimeError,
    AttributeError,
):  # fallback for partially installed TRL/httpx

    class ModelConfig:
        """Fallback stub for ``trl.ModelConfig`` when TRL is unavailable."""

        model_name_or_path: str = ""
        model_revision: Optional[str] = None
        trust_remote_code: bool = False
        attn_implementation: Optional[str] = None
        torch_dtype: Optional[str] = None

    def get_kbit_device_map(*_args, **_kwargs):
        """Stub used when TRL's ``get_kbit_device_map`` is unavailable."""
        return None

    def get_quantization_config(*_args, **_kwargs):
        """Stub used when TRL's ``get_quantization_config`` is unavailable."""
        return None


from maxent_grpo.config import GRPOConfig

if TYPE_CHECKING:
    from torch import dtype as TorchDType  # pragma: no cover
else:  # pragma: no cover - runtime fallback when torch.dtype is missing
    TorchDType = getattr(torch, "dtype", Any)


class ChatMessage(TypedDict):
    """Type definition for chat message format."""

    role: str
    content: str


def get_tokenizer(
    model_args: ModelConfig, training_args: GRPOConfig
) -> PreTrainedTokenizer | Any:
    """Load and optionally customize the tokenizer.

    The function first attempts to download a tokenizer from the Hub using the
    provided model identifiers. If the environment lacks the ``transformers``
    dependency or network access, it falls back to a lightweight stub that
    preserves the API surface required by downstream code. When a
    ``chat_template`` override is configured it is injected into the tokenizer
    before returning.

    :param model_args: Model configuration (name, revision, trust flags) used to
        locate the tokenizer on the Hub.
    :type model_args: ``trl.ModelConfig``
    :param training_args: Training configuration, specifically the optional
        ``chat_template`` used to override the tokenizer template.
    :type training_args: GRPOConfig
    :returns: A pre-trained tokenizer instance or a stub with matching
        interface for offline/CI environments.
    :rtype: ``transformers.PreTrainedTokenizer``
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
    ):  # pragma: no cover - offline/CI fallback
        # Always fall back to the lightweight stub to avoid network access.
        tokenizer = PreTrainedTokenizerStub()

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(
    model_args: ModelConfig, training_args: GRPOConfig
) -> AutoModelForCausalLM:
    """Construct the causal LM with optional quantization and device map.

    :param model_args: Model configuration (quantization, dtype, attention
        implementation, revision, trust settings) forwarded to
        ``from_pretrained``.
    :type model_args: ``trl.ModelConfig``
    :param training_args: Training configuration (used for ``use_cache`` and
        gradient checkpointing compatibility).
    :type training_args: GRPOConfig
    :returns: A loaded ``AutoModelForCausalLM`` instance, configured with a
        device map and quantization settings when available.
    :rtype: ``transformers.AutoModelForCausalLM``
    :raises ValueError: Propagated from underlying model loading if identifiers
        or revisions are invalid.
    """
    # Accept strings ("float16"), special values ("auto"/None), or actual torch.dtype
    torch_dtype: Union[str, TorchDType, None] = getattr(model_args, "torch_dtype", None)
    if torch_dtype in ["auto", None]:
        torch_dtype = model_args.torch_dtype
    elif isinstance(model_args.torch_dtype, str):
        torch_dtype = getattr(torch, model_args.torch_dtype, model_args.torch_dtype)
    else:
        torch_dtype = model_args.torch_dtype
    quantization_config: Optional[Any] = get_quantization_config(model_args)
    device_map: Optional[Dict[str, Any]] = (
        get_kbit_device_map() if quantization_config is not None else None
    )

    model_kwargs: Dict[str, Any] = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
        "device_map": device_map,
        "quantization_config": quantization_config,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    if getattr(training_args, "gradient_checkpointing", False):
        enable_fn = getattr(model, "gradient_checkpointing_enable", None)
        if callable(enable_fn):
            gc_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None)
            try:
                if isinstance(gc_kwargs, dict):
                    enable_fn(**gc_kwargs)
                else:
                    enable_fn()
            except TypeError:
                enable_fn()
    return model
