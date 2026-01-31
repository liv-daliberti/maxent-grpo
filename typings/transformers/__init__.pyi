from __future__ import annotations

from typing import Any

from .modeling_utils import PreTrainedModel as PreTrainedModel
from .tokenization_utils import PreTrainedTokenizer as PreTrainedTokenizer
from .tokenization_utils_base import (
    PreTrainedTokenizerBase as PreTrainedTokenizerBase,
)

class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any: ...

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any: ...

class AutoConfig:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any: ...


def set_seed(seed: int) -> None: ...

def __getattr__(name: str) -> Any: ...
