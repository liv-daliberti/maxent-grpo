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

    def __init__(self, **_kwargs):
        self.model_type = None

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        """Return a stub config object.

        :param _args: Ignored positional args.
        :param _kwargs: Ignored keyword args.
        :returns: Instance of :class:`AutoConfigStub`.
        """
        return cls()


class FallbackTokenizer:
    """Minimal tokenizer stub used when ``transformers`` is missing."""

    chat_template: Optional[str] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        """Return a basic tokenizer placeholder.

        :param _args: Ignored positional args matching transformers API.
        :param _kwargs: Ignored keyword args matching transformers API.
        :returns: Instance of :class:`FallbackTokenizer`.
        """
        return cls()

    def apply_chat_template(
        self,
        messages: List[Any],
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


class AutoModelForCausalLMStub:
    """Tiny model stub for environments without transformers."""

    def __init__(self):
        self.config = SimpleNamespace()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        """Return a stub model instance.

        :param _args: Ignored positional args matching transformers API.
        :param _kwargs: Ignored keyword args matching transformers API.
        :returns: Instance of :class:`AutoModelForCausalLMStub`.
        """
        return cls()

    def gradient_checkpointing_enable(self, *_args, **_kwargs):
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
