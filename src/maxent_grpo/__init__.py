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

MaxEnt-GRPO Python package namespace.

All public modules live under this package (e.g., ``maxent_grpo.training``,
``maxent_grpo.pipelines``, ``maxent_grpo.cli``).  Importing :mod:`maxent_grpo`
exposes those submodules through a light lazy-loader so code can use
``from maxent_grpo import training`` without pulling heavy dependencies until
they are actually accessed.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Dict

_LAZY_MODULES: Dict[str, str] = {
    "cli": "maxent_grpo.cli",
    "config": "maxent_grpo.config",
    "core": "maxent_grpo.core",
    "generation": "maxent_grpo.generation",
    "inference": "maxent_grpo.inference",
    "patches": "maxent_grpo.patches",
    "pipelines": "maxent_grpo.pipelines",
    "rewards": "maxent_grpo.rewards",
    "telemetry": "maxent_grpo.telemetry",
    "training": "maxent_grpo.training",
    "generate": "maxent_grpo.generate",
}

__all__ = sorted(_LAZY_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily import submodules on first access."""

    if name not in _LAZY_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module: ModuleType = import_module(_LAZY_MODULES[name])
    globals()[name] = module
    return module


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(__all__)
