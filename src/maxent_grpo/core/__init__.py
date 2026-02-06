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

"""Core domain helpers (data, evaluation, hub, model access)."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, TYPE_CHECKING

_SUBMODULES = ("data", "evaluation", "hub", "model")

__all__ = list(_SUBMODULES)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from . import data as data
    from . import evaluation as evaluation
    from . import hub as hub
    from . import model as model


def __getattr__(name: str) -> Any:
    """Lazily import core submodules on first access."""

    if name in _SUBMODULES:
        module: ModuleType = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(__all__)
