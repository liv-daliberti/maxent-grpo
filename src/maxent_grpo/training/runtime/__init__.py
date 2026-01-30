"""
Runtime utilities split by concern for the MaxEnt-GRPO training stack.

This package separates setup/dependency loading, logging, and prompt handling
so callers can import only what they need without pulling the full helper
module.
"""

from __future__ import annotations

from .setup import (
    Accelerator,
    DeepSpeedPlugin,
    GenerationSamplingConfig,
    MaxEntOptions,
    SeedAugmentationConfig,
    VLLMClientConfig,
    _maybe_create_deepspeed_plugin,
    get_trl_prepare_deepspeed,
    require_accelerator,
    require_dataloader,
    require_deepspeed,
    require_torch,
    require_transformer_base_classes,
)
from .logging import (
    _FIRST_WANDB_LOGGED_RUNS,
    _log_wandb,
    _maybe_init_wandb_run,
    _report_to_contains,
    _wandb_error_types,
    log_run_header,
    resolve_run_metadata,
)
from .prompts import (
    PROMPT_CHAR_LIMIT,
    _TRUNC_STATE,
    ChatTokenizer,
    GenerationPenaltyConfig,
    GenerationPenaltyPassthroughMixin,
    _prompt_char_limit_from_tokens,
    _to_prompt,
    _truncate_prompt,
    truncate_prompt,
)

__all__ = [
    "PROMPT_CHAR_LIMIT",
    "_FIRST_WANDB_LOGGED_RUNS",
    "_TRUNC_STATE",
    "Accelerator",
    "DeepSpeedPlugin",
    "_log_wandb",
    "_maybe_create_deepspeed_plugin",
    "_maybe_init_wandb_run",
    "_prompt_char_limit_from_tokens",
    "_report_to_contains",
    "_to_prompt",
    "_truncate_prompt",
    "_wandb_error_types",
    "log_run_header",
    "resolve_run_metadata",
    "ChatTokenizer",
    "GenerationPenaltyConfig",
    "GenerationPenaltyPassthroughMixin",
    "GenerationSamplingConfig",
    "MaxEntOptions",
    "SeedAugmentationConfig",
    "VLLMClientConfig",
    "get_trl_prepare_deepspeed",
    "require_accelerator",
    "require_dataloader",
    "require_deepspeed",
    "require_torch",
    "require_transformer_base_classes",
    "truncate_prompt",
]
