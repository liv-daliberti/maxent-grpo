"""Lightweight fallbacks for optional transformer/TRL dependencies."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional, Union


class PreTrainedTokenizerStub:
    """Minimal tokenizer surface used when transformers is unavailable.

    The stub preserves only the methods invoked by the training and evaluation
    code paths so that imports succeed in offline or lightweight test
    environments.
    """

    chat_template: Optional[str] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "PreTrainedTokenizerStub":
        """Return a stub tokenizer regardless of inputs.

        :param _args: Positional arguments mirroring the transformers API (ignored).
        :type _args: tuple
        :param _kwargs: Keyword arguments mirroring the transformers API (ignored).
        :type _kwargs: dict
        :returns: Instance of ``PreTrainedTokenizerStub``.
        :rtype: PreTrainedTokenizerStub
        """
        return cls()

    def apply_chat_template(
        self,
        messages: List[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> Union[str, List[int]]:
        """Render messages into deterministic text or naive byte tokens.

        :param messages: Sequence of ``{\"role\": str, \"content\": str}`` chat
            turns to render.
        :type messages: list[dict]
        :param tokenize: If ``True``, return a list of UTF-8 bytes rather than
            the concatenated string.
        :type tokenize: bool
        :param add_generation_prompt: Whether to append a trailing assistant
            prompt marker.
        :type add_generation_prompt: bool
        :returns: Combined chat string or byte tokens emulating tokenizer output.
        :rtype: str | list[int]
        """
        text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nASSISTANT:"
        if tokenize:
            return list(text.encode("utf-8"))
        return text

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Provide a simple byte-level decode for stubbed tokenizers.

        :param token_ids: Token identifiers to decode. For the stub these are
            treated as raw bytes.
        :type token_ids: list[int]
        :param skip_special_tokens: Present for API compatibility; ignored in
            the stub implementation.
        :type skip_special_tokens: bool
        :returns: Decoded UTF-8 string or a naive concatenation of token IDs on
            decode failure.
        :rtype: str
        """
        if not skip_special_tokens:
            # Stub keeps signature parity; no special tokens to strip.
            pass
        try:
            return bytes(token_ids).decode("utf-8")
        except (UnicodeDecodeError, ValueError, TypeError):
            return "".join(str(t) for t in token_ids)


class AutoTokenizerStub(PreTrainedTokenizerStub):
    """Alias matching transformers.AutoTokenizer."""


class AutoModelForCausalLMStub:
    """Minimal causal LM stub mirroring transformers' loader API."""

    def __init__(self) -> None:
        self.config = SimpleNamespace()

    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "AutoModelForCausalLMStub":
        """Return a stub causal LM regardless of inputs.

        :param _args: Positional arguments accepted for API parity (ignored).
        :type _args: tuple
        :param _kwargs: Keyword arguments accepted for API parity (ignored).
        :type _kwargs: dict
        :returns: Lightweight model stub exposing ``config`` and
            ``gradient_checkpointing_enable``.
        :rtype: AutoModelForCausalLMStub
        """
        return cls()

    def gradient_checkpointing_enable(self, *_args: Any, **_kwargs: Any) -> None:
        """No-op placeholder to mirror the transformers API.

        :param _args: Positional arguments accepted for compatibility.
        :type _args: tuple
        :param _kwargs: Keyword arguments accepted for compatibility.
        :type _kwargs: dict
        :returns: ``None``.
        :rtype: None
        """
        return None


class AutoConfigStub:
    """Lightweight AutoConfig stand-in used when transformers is missing."""

    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> Any:
        """Return a stub ``AutoConfig`` object with minimal fields.

        :param _args: Positional arguments matching the transformers signature
            (ignored).
        :type _args: tuple
        :param _kwargs: Keyword arguments matching the transformers signature
            (ignored).
        :type _kwargs: dict
        :returns: Simple namespace containing ``num_attention_heads``,
            ``hidden_size`` and ``model_type`` defaults.
        :rtype: types.SimpleNamespace
        """
        return SimpleNamespace(num_attention_heads=0, hidden_size=0, model_type="stub")
