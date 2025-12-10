"""Shared helpers for constructing vLLM generation contexts in tests."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict

__all__ = ["make_vllm_context"]

_DEFAULT_GENERATION_STATS: Dict[str, Any] = {
    "vllm_retry_rounds": 0,
    "vllm_retry_failures": 0,
    "vllm_backfilled_prompts": 0,
    "vllm_failed_prompts": 0,
    "vllm_excess_prompts": 0,
    "vllm_excess_completions": 0,
    "vllm_last_error": None,
}

_DEFAULT_CTX: Dict[str, Any] = {
    "accelerator": SimpleNamespace(
        is_main_process=True,
        process_index=0,
        num_processes=1,
        state=None,
    ),
    "vllm_sync_weights": False,
    "vllm_url": "http://localhost:8000/generate",
    "vllm_rounds_cfg": 0,
    "vllm_request_logprobs": False,
    "vllm_retry_sleep": 0.0,
    "vllm_backfill_local": False,
    "vllm_stop_sequences": None,
    "vllm_top_k": None,
    "vllm_best_of": None,
    "vllm_logit_bias": None,
    "vllm_guided_json": None,
    "vllm_guided_regex": None,
    "vllm_request_id_prefix": None,
    "vllm_timeout": 10,
    "vllm_max_retries": 0,
    "vllm_backoff": None,
    "vllm_backoff_multiplier": 1.0,
    "gen_stop_sequences": None,
    "gen_top_k": None,
    "gen_best_of": None,
    "gen_temperature": 0.1,
    "gen_top_p": 0.9,
    "gen_frequency_penalty": 0.0,
    "gen_presence_penalty": 0.0,
    "max_completion_len": 8,
    "tokenizer": None,
    "prompt_char_limit": None,
    "generation_stats": _DEFAULT_GENERATION_STATS,
}


def make_vllm_context(**overrides: Any) -> SimpleNamespace:
    """Return a ``SimpleNamespace`` mimicking the real generation context.

    :param overrides: Attribute overrides applied on top of the defaults.
    :returns: SimpleNamespace populated with commonly accessed vLLM fields.
    """

    ctx_payload: Dict[str, Any] = {
        key: deepcopy(value) for key, value in _DEFAULT_CTX.items()
    }
    stats_override = overrides.pop("generation_stats", None)
    if stats_override is not None:
        ctx_payload["generation_stats"].update(stats_override)
    ctx_payload.update(overrides)
    return SimpleNamespace(**ctx_payload)
