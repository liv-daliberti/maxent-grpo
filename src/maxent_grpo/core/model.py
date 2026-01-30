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

- ``get_tokenizer``: Load a tokenizer with optional chat template override.
- ``get_model``: Load an ``AutoModelForCausalLM`` with optional quantization
  and device map resolution via TRL helpers, respecting attention impl/dtype
  choices and gradient checkpointing compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict, TYPE_CHECKING, Union, cast
import logging

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer as _PreTrainedTokenizer

PreTrainedTokenizer = _PreTrainedTokenizer
if TYPE_CHECKING:
    AutoModelForCausalLMType = AutoModelForCausalLM
    PreTrainedTokenizerType = PreTrainedTokenizer
else:
    AutoModelForCausalLMType = Any
    PreTrainedTokenizerType = Any

from trl import (  # type: ignore[reportMissingTypeStubs]
    ModelConfig,
    get_kbit_device_map,
    get_quantization_config,
)

if TYPE_CHECKING:
    ModelConfigType = ModelConfig
else:
    ModelConfigType = Any


from maxent_grpo.config import GRPOConfig

if TYPE_CHECKING:
    from torch import dtype as TorchDType  # pragma: no cover
else:  # pragma: no cover - runtime fallback when torch.dtype is missing
    TorchDType = getattr(torch, "dtype", Any)

LOG = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    """Type definition for chat message format."""

    role: str
    content: str


def get_tokenizer(
    model_args: ModelConfigType, training_args: GRPOConfig
) -> PreTrainedTokenizerType | Any:
    """Load and optionally customize the tokenizer.

    The function downloads a tokenizer from the Hub using the provided model
    identifiers. When a ``chat_template`` override is configured it is injected
    into the tokenizer before returning.

    :param model_args: Model configuration (name, revision, trust flags) used to
        locate the tokenizer on the Hub.
    :type model_args: ``trl.ModelConfig``
    :param training_args: Training configuration, specifically the optional
        ``chat_template`` used to override the tokenizer template.
    :type training_args: GRPOConfig
    :returns: A pre-trained tokenizer instance.
    :rtype: ``transformers.PreTrainedTokenizer``
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    pad_token = getattr(tokenizer, "pad_token", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if pad_token is None and eos_token is not None:
        try:
            setattr(tokenizer, "pad_token", eos_token)
        except (AttributeError, TypeError, ValueError):
            LOG.debug("Failed to set tokenizer.pad_token from eos_token.")

    return tokenizer


def get_model(
    model_args: ModelConfigType, training_args: GRPOConfig
) -> AutoModelForCausalLMType:
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
    quantization_config: Optional[Any] = get_quantization_config(
        cast(ModelConfig, model_args)
    )
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

    model_name_or_path = getattr(model_args, "model_name_or_path", None)
    if not model_name_or_path:
        raise ValueError("model_name_or_path must be set in model_args")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None and getattr(cfg, "pad_token_id", None) is None:
            eos_token_id = getattr(cfg, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                cfg.pad_token_id = eos_token_id
            elif isinstance(eos_token_id, (list, tuple)) and eos_token_id:
                first = eos_token_id[0]
                if isinstance(first, int):
                    cfg.pad_token_id = first
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
            cfg = getattr(model, "config", None)
            pad_token_id = (
                getattr(cfg, "pad_token_id", None) if cfg is not None else None
            )
            if isinstance(pad_token_id, int):
                gen_cfg.pad_token_id = pad_token_id
    except (AttributeError, TypeError, ValueError):
        LOG.debug("Failed to align model pad_token_id settings.")
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
    # Optional seed classification head for InfoSeed auxiliary objectives.
    if getattr(training_args, "info_seed_enabled", False):
        num_seeds = max(int(getattr(training_args, "info_seed_num_seeds", 0)), 0)
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if num_seeds > 0 and hidden_size:
            if not hasattr(model, "seed_head"):
                setattr(model, "seed_head", nn.Linear(hidden_size, num_seeds))
    if getattr(training_args, "torch_compile", False):
        # torch.compile is fragile with DeepSpeed/ZeRO wrapping; skip when deepspeed config is present.
        prev_suppress = None
        dynamo_mod = None
        dynamo_config = None
        if getattr(training_args, "deepspeed", None):
            LOG.warning(
                "Skipping torch.compile because deepspeed is enabled; set torch_compile=false to silence."
            )
        else:
            try:
                import torch._dynamo as dynamo_mod  # type: ignore[attr-defined]

                dynamo_config = getattr(dynamo_mod, "config", None)
                prev_suppress = (
                    getattr(dynamo_config, "suppress_errors", None)
                    if dynamo_config is not None
                    else None
                )
                try:
                    if dynamo_config is not None:
                        dynamo_config.suppress_errors = True  # fall back to eager on compile errors
                except (AttributeError, TypeError):
                    LOG.debug("Failed to set torch._dynamo suppress_errors flag.")
            except (ImportError, AttributeError, RuntimeError):
                dynamo_mod = None
        compile_fn = getattr(torch, "compile", None)
        if callable(compile_fn):
            try:
                model = compile_fn(model, mode="max-autotune")
            except TypeError:
                try:
                    model = compile_fn(model)
                except (RuntimeError, TypeError, ValueError):
                    LOG.warning("torch.compile failed; falling back to eager mode.")
            except (RuntimeError, ValueError):
                # Best-effort: ignore compilation failures and keep the original model.
                LOG.warning("torch.compile failed; falling back to eager mode.")
            finally:
                if dynamo_config is not None and prev_suppress is not None:
                    try:
                        dynamo_config.suppress_errors = prev_suppress
                    except (AttributeError, TypeError):
                        LOG.debug("Failed to restore torch._dynamo suppress_errors flag.")
    return cast(AutoModelForCausalLMType, model)
