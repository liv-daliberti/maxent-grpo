from __future__ import annotations

from typing import Any

class TrainerCallback:
    def __getattr__(self, name: str) -> Any: ...
