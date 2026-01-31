from __future__ import annotations

from typing import Any

class _Config:
    suppress_errors: bool

config: _Config

def __getattr__(name: str) -> Any: ...
