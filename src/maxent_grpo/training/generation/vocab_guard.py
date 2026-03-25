"""Shared helpers for masking model-only token IDs during generation."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, cast

from maxent_grpo.training.scoring_common import (
    _coerce_optional_int,
    _get_config_value,
    _get_embedding_vocab_size,
)

LOG = logging.getLogger(__name__)
_INVALID_TOKEN_BLOCK_BIAS = -1.0e9


def _resolve_served_model_id(ctx: Any) -> Optional[str]:
    """Best-effort resolution of the external vLLM-served model identifier."""

    env_model = os.getenv("MAXENT_VLLM_SERVER_MODEL_NAME")
    if isinstance(env_model, str) and env_model.strip():
        return env_model.strip()

    for key in (
        "vllm_model_id",
        "served_model_id",
        "model_name",
        "model_id",
        "hub_model_id",
        "model_name_or_path",
    ):
        value = getattr(ctx, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    training_args = getattr(ctx, "training_args", None)
    if training_args is not None:
        for key in ("model_name_or_path", "hub_model_id", "model_id"):
            value = getattr(training_args, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _resolve_served_model_vocab_limit(ctx: Any) -> Optional[int]:
    """Return the output-vocab width exposed by the external vLLM model."""

    cached = getattr(ctx, "_served_model_vocab_limit", None)
    if isinstance(cached, int) and cached > 0:
        return int(cached)

    env_limit = _coerce_optional_int(os.getenv("MAXENT_VLLM_SERVER_MODEL_VOCAB_LIMIT"))
    if isinstance(env_limit, int) and env_limit > 0:
        setattr(ctx, "_served_model_vocab_limit", int(env_limit))
        return int(env_limit)

    model_id = _resolve_served_model_id(ctx)
    if not model_id:
        return None

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug("Unable to resolve served model vocab limit for %s: %s", model_id, exc)
        return None

    vocab_limit = _coerce_optional_int(_get_config_value(config, "vocab_size", None))
    if not isinstance(vocab_limit, int) or vocab_limit <= 0:
        return None

    setattr(ctx, "_served_model_vocab_limit", int(vocab_limit))
    if not bool(getattr(ctx, "_served_model_vocab_limit_logged", False)):
        LOG.warning(
            "Resolved served-model vocab limit=%d for server-mode vLLM generation (model=%s).",
            int(vocab_limit),
            model_id,
        )
        setattr(ctx, "_served_model_vocab_limit_logged", True)
    return int(vocab_limit)


def resolve_tokenizer_vocab_limit(tokenizer: Any) -> Optional[int]:
    """Return the maximum token id addressable by the tokenizer plus one."""

    if tokenizer is None:
        return None
    candidates: List[int] = []
    for attr in ("vocab_size",):
        value = _coerce_optional_int(getattr(tokenizer, attr, None))
        if value is not None and value > 0:
            candidates.append(int(value))
    try:
        tokenizer_len = len(tokenizer)
    except Exception:
        tokenizer_len = None
    if isinstance(tokenizer_len, int) and tokenizer_len > 0:
        candidates.append(int(tokenizer_len))
    if not candidates:
        return None
    return max(candidates)


def resolve_model_vocab_limit(ctx: Any) -> Optional[int]:
    """Return the model output-vocab width exposed to generation."""

    model = getattr(ctx, "model", None)
    accelerator = getattr(ctx, "accelerator", None)
    unwrap_fn = getattr(accelerator, "unwrap_model", None)
    base_model = model
    if callable(unwrap_fn):
        try:
            base_model = unwrap_fn(model)
        except Exception:
            base_model = model
    if base_model is None:
        return None
    config = getattr(base_model, "config", None)
    embedding_vocab = _get_embedding_vocab_size(base_model, config)
    config_vocab = _coerce_optional_int(_get_config_value(config, "vocab_size", None))
    candidates = [
        int(value)
        for value in (embedding_vocab, config_vocab)
        if isinstance(value, int) and int(value) > 0
    ]
    use_vllm = bool(getattr(ctx, "use_vllm", False))
    vllm_mode = str(getattr(ctx, "vllm_mode", "server") or "server").strip().lower()
    if use_vllm and vllm_mode == "server":
        served_model_vocab = _resolve_served_model_vocab_limit(ctx)
        if isinstance(served_model_vocab, int) and served_model_vocab > 0:
            candidates.append(int(served_model_vocab))
    if not candidates:
        return None
    return max(candidates)


def merge_invalid_token_block_logit_bias(
    ctx: Any,
    existing_bias: Any,
) -> Optional[Dict[str, float]]:
    """Block model-only token IDs that the tokenizer cannot represent."""

    tokenizer = getattr(ctx, "tokenizer", None)
    tokenizer_limit = resolve_tokenizer_vocab_limit(tokenizer)
    model_limit = resolve_model_vocab_limit(ctx)
    if (
        not isinstance(tokenizer_limit, int)
        or tokenizer_limit <= 0
        or not isinstance(model_limit, int)
        or model_limit <= tokenizer_limit
    ):
        return cast(Optional[Dict[str, float]], existing_bias)

    merged: Dict[str, float] = {}
    if isinstance(existing_bias, dict):
        for key, value in existing_bias.items():
            try:
                merged[str(int(key))] = float(value)
            except (TypeError, ValueError):
                continue

    blocked = 0
    for token_id in range(int(tokenizer_limit), int(model_limit)):
        key = str(int(token_id))
        prev = merged.get(key)
        if prev is None or prev > _INVALID_TOKEN_BLOCK_BIAS:
            merged[key] = _INVALID_TOKEN_BLOCK_BIAS
        blocked += 1

    if not bool(getattr(ctx, "_vllm_invalid_token_block_logged", False)):
        LOG.warning(
            "Blocking %d tokenizer-inaccessible token IDs for vLLM generation (tokenizer_limit=%d, model_limit=%d).",
            blocked,
            tokenizer_limit,
            model_limit,
        )
        setattr(ctx, "_vllm_invalid_token_block_logged", True)
    stats = getattr(ctx, "generation_stats", None)
    if isinstance(stats, dict):
        stats["vllm_invalid_token_block_count"] = blocked
        stats["vllm_invalid_token_block_min_id"] = int(tokenizer_limit)
        stats["vllm_invalid_token_block_max_id"] = int(model_limit - 1)
    return merged


def resolve_allowed_token_ids(ctx: Any) -> Optional[List[int]]:
    """Return a cached hard allowlist for tokenizer-addressable token IDs."""

    tokenizer = getattr(ctx, "tokenizer", None)
    tokenizer_limit = resolve_tokenizer_vocab_limit(tokenizer)
    model_limit = resolve_model_vocab_limit(ctx)
    if (
        not isinstance(tokenizer_limit, int)
        or tokenizer_limit <= 0
        or not isinstance(model_limit, int)
        or model_limit <= tokenizer_limit
    ):
        return None

    cached = getattr(ctx, "_vllm_allowed_token_ids", None)
    cached_limit = getattr(ctx, "_vllm_allowed_token_ids_limit", None)
    if isinstance(cached, list) and cached_limit == int(tokenizer_limit):
        return cached

    allowed = list(range(int(tokenizer_limit)))
    setattr(ctx, "_vllm_allowed_token_ids", allowed)
    setattr(ctx, "_vllm_allowed_token_ids_limit", int(tokenizer_limit))
    if not bool(getattr(ctx, "_vllm_allowed_token_ids_logged", False)):
        LOG.warning(
            "Allowing only %d tokenizer-addressable token IDs for vLLM generation (tokenizer_limit=%d, model_limit=%d).",
            tokenizer_limit,
            tokenizer_limit,
            model_limit,
        )
        setattr(ctx, "_vllm_allowed_token_ids_logged", True)
    stats = getattr(ctx, "generation_stats", None)
    if isinstance(stats, dict):
        stats["vllm_allowed_token_ids_count"] = int(tokenizer_limit)
    return allowed


def resolve_blocked_token_ids(ctx: Any) -> List[int]:
    """Return tokenizer-inaccessible model token IDs for local generation guards."""

    tokenizer = getattr(ctx, "tokenizer", None)
    tokenizer_limit = resolve_tokenizer_vocab_limit(tokenizer)
    model_limit = resolve_model_vocab_limit(ctx)
    if (
        not isinstance(tokenizer_limit, int)
        or tokenizer_limit <= 0
        or not isinstance(model_limit, int)
        or model_limit <= tokenizer_limit
    ):
        return []

    cached = getattr(ctx, "_local_blocked_token_ids", None)
    cached_limits = getattr(ctx, "_local_blocked_token_ids_limits", None)
    if (
        isinstance(cached, list)
        and isinstance(cached_limits, tuple)
        and cached_limits == (int(tokenizer_limit), int(model_limit))
    ):
        return cached

    blocked = list(range(int(tokenizer_limit), int(model_limit)))
    setattr(ctx, "_local_blocked_token_ids", blocked)
    setattr(
        ctx,
        "_local_blocked_token_ids_limits",
        (int(tokenizer_limit), int(model_limit)),
    )
    if not bool(getattr(ctx, "_local_invalid_token_block_logged", False)):
        LOG.warning(
            "Blocking %d tokenizer-inaccessible token IDs for local generation (tokenizer_limit=%d, model_limit=%d).",
            len(blocked),
            tokenizer_limit,
            model_limit,
        )
        setattr(ctx, "_local_invalid_token_block_logged", True)
    stats = getattr(ctx, "generation_stats", None)
    if isinstance(stats, dict):
        stats["local_invalid_token_block_count"] = len(blocked)
    return blocked
