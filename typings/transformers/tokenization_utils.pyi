from __future__ import annotations

from typing import Any

class PreTrainedTokenizer:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
