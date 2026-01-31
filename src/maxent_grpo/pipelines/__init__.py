"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Application-layer orchestration helpers for the MaxEnt-GRPO toolkit.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, TYPE_CHECKING

from .base import PipelineResult, log_pipeline_banner

_LAZY_SUBMODULES = {
    "generation": "maxent_grpo.pipelines.generation",
    "math_inference": "maxent_grpo.pipelines.math_inference",
    "training": "maxent_grpo.pipelines.training",
}

class _LazyModuleProxy:
    """Proxy that lazily imports a module on first attribute access."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            self._module = import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        module = self._load()
        value = getattr(module, name)
        setattr(self, name, value)
        return value

    def __dir__(self) -> list[str]:  # pragma: no cover - trivial
        if self._module is None:
            return sorted(self.__dict__.keys())
        return sorted(dir(self._module))

if TYPE_CHECKING:  # pragma: no cover - typing-only imports for __all__
    from . import generation as generation
    from . import math_inference as math_inference
    from . import training as training
else:
    generation = _LazyModuleProxy(_LAZY_SUBMODULES["generation"])
    math_inference = _LazyModuleProxy(_LAZY_SUBMODULES["math_inference"])
    training = _LazyModuleProxy(_LAZY_SUBMODULES["training"])

__all__ = [
    "generation",
    "math_inference",
    "training",
    "PipelineResult",
    "log_pipeline_banner",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(__all__)
