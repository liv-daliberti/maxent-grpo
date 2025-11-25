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
    """Clamp prompt strings to a safe length for vLLM/http payloads (shared warning state)."""

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


# Backwards compatibility for existing imports.
_truncate_prompt = truncate_prompt


def _prompt_char_limit_from_tokens(max_prompt_len: int) -> int:
    """Derive a character cap from the token cap (≈4 chars/token) with env floor."""

    approx_char_limit = (
        int(max_prompt_len * 4) if max_prompt_len and max_prompt_len > 0 else 0
    )
    if approx_char_limit <= 0:
        return PROMPT_CHAR_LIMIT
    return max(PROMPT_CHAR_LIMIT, approx_char_limit)


def _to_prompt(
    example: Dict[str, Any],
    tokenizer: Union["PreTrainedTokenizer", ChatTokenizer],
    prompt_column: str,
    system_prompt: Optional[str],
    char_limit: Optional[int] = None,
) -> Dict[str, str]:
    """Shared prompt/answer builder used across training pipelines."""

    user = str(example.get(prompt_column, example.get("prompt", "")))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        prompt = (
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            + "\nASSISTANT:"
        )
    prompt = truncate_prompt(prompt, char_limit)
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
    "truncate_prompt",
]
