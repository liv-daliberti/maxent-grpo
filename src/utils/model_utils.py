"""
Tokenizer/model loading helpers for training scripts.

This module exposes two utilities:
- ``get_tokenizer``: Load a tokenizer with optional chat template override.
  A minimal fallback tokenizer is provided for offline/CI environments.
- ``get_model``: Load an ``AutoModelForCausalLM`` with optional quantization
  and device map resolution via TRL helpers, respecting attention impl/dtype
  choices and gradient checkpointing compatibility.

License
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import torch
from typing import Any, Dict, List, Optional, TypedDict, TYPE_CHECKING, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from configs import GRPOConfig

if TYPE_CHECKING:
    from torch import dtype as TorchDType  # pragma: no cover
else:  # pragma: no cover - runtime fallback when torch.dtype is missing
    TorchDType = getattr(torch, "dtype", Any)  # type: ignore[assignment]


class ChatMessage(TypedDict):
    """Type definition for chat message format."""
    role: str
    content: str


def get_tokenizer(
    model_args: ModelConfig, training_args: GRPOConfig
) -> PreTrainedTokenizer | Any:
    """Load and optionally customize the tokenizer.

    :param model_args: Model configuration (name, revision, trust flags).
    :type model_args: trl.ModelConfig
    :param training_args: Training configuration (used for ``chat_template``).
    :type training_args: GRPOConfig
    :returns: A pre‑trained tokenizer instance.
    :rtype: transformers.PreTrainedTokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    except (OSError, ValueError, RuntimeError):  # pragma: no cover - offline/CI fallback
        class _FallbackTok:  # minimal surface for tests/docs
            chat_template: Optional[str] = None
            eos_token_id: Optional[int] = None
            pad_token_id: Optional[int] = None

            def apply_chat_template(
                self,
                messages: List[ChatMessage],
                tokenize: bool = False,
                add_generation_prompt: bool = True
            ) -> Union[str, List[int]]:
                text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                )
                if add_generation_prompt:
                    text += "\nASSISTANT:"
                # Provide a deterministic, minimal behavior when tokenize=True
                # to mirror HF's API surface without external dependencies.
                if tokenize:
                    # Naive byte-level tokenization for CI/docs environments.
                    return list(text.encode("utf-8"))
                return text

        tokenizer = _FallbackTok()

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(
    model_args: ModelConfig, training_args: GRPOConfig
) -> AutoModelForCausalLM:
    """Construct the causal LM with optional quantization and device map.

    :param model_args: Model configuration (quantization, dtype, attn impl,
        revision, trust settings).
    :type model_args: trl.ModelConfig
    :param training_args: Training configuration (used for ``use_cache`` and
        gradient checkpointing compatibility).
    :type training_args: GRPOConfig
    :returns: A loaded ``AutoModelForCausalLM`` instance.
    :rtype: transformers.AutoModelForCausalLM
    """
    # Accept strings ("float16"), special values ("auto"/None), or actual torch.dtype
    torch_dtype: Union[str, TorchDType, None] = getattr(model_args, "torch_dtype", None)
    if torch_dtype in ["auto", None]:
        torch_dtype = model_args.torch_dtype
    elif isinstance(model_args.torch_dtype, str):
        torch_dtype = getattr(torch, model_args.torch_dtype)
    else:
        torch_dtype = model_args.torch_dtype
    quantization_config: Optional[Any] = get_quantization_config(model_args)
    device_map: Optional[Dict[str, Any]] = get_kbit_device_map() if quantization_config is not None else None

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
