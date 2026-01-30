# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared lightweight stubs used when optional dependencies are missing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional, Union


def _assign_module(obj: Any, module_name: str) -> None:
    """Best-effort helper to set ``__module__`` when possible.

    :param obj: Object whose ``__module__`` attribute will be updated.
    :param module_name: Module path to assign for display and Sphinx rendering.
    :returns: ``None``. Fails silently when the attribute cannot be set.
    """
    try:
        obj.__module__ = module_name
    except (AttributeError, TypeError):
        return


class AutoConfigStub:
    """Minimal AutoConfig stub when transformers is unavailable.

    :param _kwargs: Ignored keyword arguments mirroring ``AutoConfig`` signature.
    """

    __maxent_stub__ = True

    def __init__(self, **_kwargs: Any) -> None:
        self.model_type = None

    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "AutoConfigStub":
        """Return a stub config object.

        :param _args: Ignored positional args.
        :param _kwargs: Ignored keyword args.
        :returns: Instance of :class:`AutoConfigStub`.
        """
        return cls()


class FallbackTokenizer:
    """Minimal tokenizer stub used when ``transformers`` is missing."""

    __maxent_stub__ = True

    chat_template: Optional[str] = None
    eos_token: Optional[str] = None
    eos_token_id: Optional[int] = None
    pad_token: Optional[str] = None
    pad_token_id: Optional[int] = None
    padding_side: str = "right"

    def __init__(self) -> None:
        self._vocab_size = 0

    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "FallbackTokenizer":
        """Return a basic tokenizer placeholder.

        :param _args: Ignored positional args matching transformers API.
        :param _kwargs: Ignored keyword args matching transformers API.
        :returns: Instance of :class:`FallbackTokenizer`.
        """
        return cls()

    def apply_chat_template(
        self,
        messages: List[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> Union[str, List[int]]:
        """Render chat messages when the real tokenizer implementation is absent.

        :param messages: Sequence of chat message dicts with ``role`` and ``content`` keys.
        :param tokenize: If ``True``, return a naive byte list instead of a string.
        :param add_generation_prompt: Whether to append a final assistant prompt line.
        :returns: Rendered chat transcript or byte token list when ``tokenize`` is set.
        """
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        if tokenize:
            return list(text.encode("utf-8"))
        return text

    def add_special_tokens(
        self, special_tokens_dict: dict[str, str], *_args: Any, **_kwargs: Any
    ) -> int:
        """Register special tokens on the stub tokenizer.

        :param special_tokens_dict: Mapping of token names to token strings.
        :returns: Number of tokens added (best-effort stubbed value).
        """
        added = 0
        pad_token = special_tokens_dict.get("pad_token")
        if pad_token is not None:
            self.pad_token = pad_token
            if self.pad_token_id is None:
                self.pad_token_id = 0
            added += 1
        eos_token = special_tokens_dict.get("eos_token")
        if eos_token is not None:
            self.eos_token = eos_token
            if self.eos_token_id is None:
                self.eos_token_id = 0
            added += 1
        if added:
            self._vocab_size = max(self._vocab_size, added)
        return added

    def __len__(self) -> int:
        return self._vocab_size

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Minimal callable interface mirroring HF tokenizers."""
        _ = kwargs
        if not args:
            return {"input_ids": [], "attention_mask": []}
        text = args[0]
        if isinstance(text, list):
            ids = [list(str(item).encode("utf-8")) for item in text]
            return {"input_ids": ids, "attention_mask": [[1] * len(seq) for seq in ids]}
        token_ids = list(str(text).encode("utf-8"))
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Best-effort decode for compatibility with HF tokenizer API."""
        _ = skip_special_tokens
        try:
            return bytes(token_ids).decode("utf-8")
        except (UnicodeDecodeError, ValueError, TypeError):
            return "".join(str(t) for t in token_ids)


class AutoModelForCausalLMStub:
    """Tiny model stub for environments without transformers."""

    __maxent_stub__ = True

    def __init__(self) -> None:
        self.config = SimpleNamespace()

    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "AutoModelForCausalLMStub":
        """Return a stub model instance.

        :param _args: Ignored positional args matching transformers API.
        :param _kwargs: Ignored keyword args matching transformers API.
        :returns: Instance of :class:`AutoModelForCausalLMStub`.
        """
        return cls()

    def gradient_checkpointing_enable(self, *_args: Any, **_kwargs: Any) -> None:
        """No-op placeholder mirroring transformers API.

        :param _args: Ignored positional args.
        :param _kwargs: Ignored keyword args.
        :returns: ``None``; provided for API compatibility.
        """
        return None


# Align stubs with expected module names for Sphinx/tests when possible.
for _cls in (AutoConfigStub, FallbackTokenizer, AutoModelForCausalLMStub):
    _assign_module(_cls, "transformers")

# Aliases matching upstream class names for convenience.
AutoTokenizerStub = FallbackTokenizer
PreTrainedTokenizerStub = FallbackTokenizer
