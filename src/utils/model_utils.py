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
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from configs import GRPOConfig, SFTConfig


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig
) -> PreTrainedTokenizer:
    """Load and optionally customize the tokenizer.

    :param model_args: Model configuration (name, revision, trust flags).
    :type model_args: trl.ModelConfig
    :param training_args: Training configuration (used for ``chat_template``).
    :type training_args: SFTConfig | GRPOConfig
    :returns: A pre‑trained tokenizer instance.
    :rtype: transformers.PreTrainedTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig
) -> AutoModelForCausalLM:
    """Construct the causal LM with optional quantization and device map.

    :param model_args: Model configuration (quantization, dtype, attn impl,
        revision, trust settings).
    :type model_args: trl.ModelConfig
    :param training_args: Training configuration (used for ``use_cache`` and
        gradient checkpointing compatibility).
    :type training_args: SFTConfig | GRPOConfig
    :returns: A loaded ``AutoModelForCausalLM`` instance.
    :rtype: transformers.AutoModelForCausalLM
    """
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    device_map = get_kbit_device_map() if quantization_config is not None else None

    model_kwargs = {
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
    return model
