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
from typing import Any

from .base import PipelineResult, log_pipeline_banner

_LAZY_SUBMODULES = {
    "generation": "maxent_grpo.pipelines.generation",
    "math_inference": "maxent_grpo.pipelines.math_inference",
    "training": "maxent_grpo.pipelines.training",
}

__all__ = [
    *list(_LAZY_SUBMODULES.keys()),
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
