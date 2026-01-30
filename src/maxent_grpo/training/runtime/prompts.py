"""Prompt-related helpers and sampling penalties."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING, Union

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from transformers import PreTrainedTokenizer

LOG = logging.getLogger(__name__)
PROMPT_CHAR_LIMIT = int(os.environ.get("MAX_PROMPT_CHARS", "2048"))
_TRUNC_STATE = {"warned": False}


class ChatTokenizer(Protocol):
    """Protocol for tokenizers with chat template capabilities."""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> Union[str, List[int]]:
        """Render a conversation into a model-ready prompt."""
        raise NotImplementedError

    @property
    def eos_token_id(self) -> Optional[int]:
        """Expose the EOS token id used by the tokenizer."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow chat tokenizers to be invoked like standard HF tokenizers."""
        raise NotImplementedError


@dataclass
class GenerationPenaltyConfig:
    """Shared penalty/stop sequence overrides for completion sampling."""

    gen_top_k: Optional[int] = None
    gen_best_of: Optional[int] = None
    gen_frequency_penalty: float = 0.0
    gen_presence_penalty: float = 0.0
    gen_stop_sequences: Optional[List[str]] = None


class GenerationPenaltyPassthroughMixin:
    """Expose penalty overrides via legacy ``gen_*`` accessors."""

    penalty: GenerationPenaltyConfig

    @property
    def gen_top_k(self) -> Optional[int]:
        """Backward-compatible alias for the top-k sampling limit."""
        return self.penalty.gen_top_k

    @gen_top_k.setter
    def gen_top_k(self, value: Optional[int]) -> None:
        self.penalty.gen_top_k = value

    @property
    def gen_best_of(self) -> Optional[int]:
        """Backward-compatible alias for the best-of sampling count."""
        return self.penalty.gen_best_of

    @gen_best_of.setter
    def gen_best_of(self, value: Optional[int]) -> None:
        self.penalty.gen_best_of = value

    @property
    def gen_frequency_penalty(self) -> float:
        """Backward-compatible alias for the frequency penalty strength."""
        return self.penalty.gen_frequency_penalty

    @gen_frequency_penalty.setter
    def gen_frequency_penalty(self, value: float) -> None:
        self.penalty.gen_frequency_penalty = value

    @property
    def gen_presence_penalty(self) -> float:
        """Backward-compatible alias for the presence penalty strength."""
        return self.penalty.gen_presence_penalty

    @gen_presence_penalty.setter
    def gen_presence_penalty(self, value: float) -> None:
        self.penalty.gen_presence_penalty = value

    @property
    def gen_stop_sequences(self) -> Optional[List[str]]:
        """Backward-compatible alias for stop sequences."""
        return self.penalty.gen_stop_sequences

    @gen_stop_sequences.setter
    def gen_stop_sequences(self, value: Optional[List[str]]) -> None:
        self.penalty.gen_stop_sequences = value


def truncate_prompt(prompt: str, char_limit: Optional[int] = None) -> str:
    """Clamp prompt strings to a safe length for vLLM/http payloads.

    :param prompt: Prompt string to clamp.
    :param char_limit: Optional character limit override. When ``None`` the
        module-level ``PROMPT_CHAR_LIMIT`` is used.
    :returns: The original prompt when under the limit, otherwise a truncated
        prefix.
    :rtype: str
    """

    limit = char_limit if char_limit is not None else PROMPT_CHAR_LIMIT
    if limit <= 0 or len(prompt) <= limit:
        return prompt
    if not _TRUNC_STATE["warned"]:
        LOG.warning(
            "Prompt length exceeded %d characters; truncating. "
            "Override via MAX_PROMPT_CHARS if needed.",
            limit,
        )
        _TRUNC_STATE["warned"] = True
    return prompt[:limit]


def sync_trunc_state(state: Dict[str, Any]) -> None:
    """Merge external truncation state into the shared warning cache.

    :param state: Dictionary of state keys to merge (e.g., ``{"warned": True}``).
    :returns: ``None``.
    """

    if isinstance(state, dict):
        _TRUNC_STATE.update(state)


# Backwards compatibility for existing imports.
_truncate_prompt = truncate_prompt


def _prompt_char_limit_from_tokens(max_prompt_len: int) -> int:
    """Derive a character cap from the token cap (approx. 4 chars/token).

    :param max_prompt_len: Maximum number of tokens allowed for prompts.
    :returns: Character limit used by ``truncate_prompt``.
    :rtype: int
    """

    approx_char_limit = (
        int(max_prompt_len * 4) if max_prompt_len and max_prompt_len > 0 else 0
    )
    if approx_char_limit <= 0:
        return PROMPT_CHAR_LIMIT
    return max(PROMPT_CHAR_LIMIT, approx_char_limit)


def _require_prompt_column(example: Dict[str, Any], prompt_column: str) -> None:
    """Raise if the configured prompt column is missing from a dataset row."""

    if prompt_column in example:
        return
    try:
        available = ", ".join(sorted(str(key) for key in example.keys()))
    except (AttributeError, TypeError):
        available = "<unknown>"
    raise KeyError(
        f"Missing prompt column '{prompt_column}' in dataset row. "
        f"Available columns: {available}"
    )


def _to_prompt(
    example: Dict[str, Any],
    tokenizer: Union["PreTrainedTokenizer", ChatTokenizer],
    prompt_column: str,
    system_prompt: Optional[str],
    char_limit: Optional[int] = None,
) -> Dict[str, str]:
    """Shared prompt/answer builder used across training pipelines.

    :param example: Dataset row containing a prompt and optional answer fields.
    :param tokenizer: Tokenizer or chat template adapter used to render prompts.
    :param prompt_column: Column name to read the user prompt from.
    :param system_prompt: Optional system prompt prepended to the conversation.
    :param char_limit: Optional character cap applied after formatting.
    :returns: Mapping with ``prompt`` and ``answer`` string fields.
    :rtype: dict[str, str]
    :raises KeyError: If the prompt column is missing from the example.
    """

    resolved_column = prompt_column
    if prompt_column not in example and prompt_column == "problem":
        for candidate in ("prompt", "question"):
            if candidate in example:
                LOG.info(
                    "Prompt column '%s' missing; falling back to '%s'.",
                    prompt_column,
                    candidate,
                )
                resolved_column = candidate
                break
    _require_prompt_column(example, resolved_column)
    user_val = example.get(resolved_column)
    user = "" if user_val is None else str(user_val)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})

    try:
        apply_fn = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_fn):
            raise AttributeError("chat template missing or not callable")
        prompt = apply_fn(messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(prompt, str):
            raise TypeError("chat template did not return a string prompt")
    except (AttributeError, TypeError, ValueError, RuntimeError):
        prompt = (
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            + "\nASSISTANT:"
        )
    effective_limit = char_limit if char_limit is not None else PROMPT_CHAR_LIMIT
    min_required = len("USER: ") + len(user) + len("\nASSISTANT:")
    if effective_limit and effective_limit > 0 and effective_limit < min_required:
        effective_limit = min_required
    prompt = truncate_prompt(prompt, effective_limit)
    # Defensive: ensure the user message survives even if truncation or a template
    # removed it entirely.
    if user and user not in prompt:
        prompt = f"{prompt}\n{user}"
    return {
        "prompt": prompt,
        "answer": str(example.get("answer", example.get("solution", ""))),
    }


__all__ = [
    "ChatTokenizer",
    "GenerationPenaltyConfig",
    "GenerationPenaltyPassthroughMixin",
    "PROMPT_CHAR_LIMIT",
    "_TRUNC_STATE",
    "_prompt_char_limit_from_tokens",
    "_to_prompt",
    "_truncate_prompt",
    "sync_trunc_state",
    "truncate_prompt",
]
