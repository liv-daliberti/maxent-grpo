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
from typing import Any, Dict, TYPE_CHECKING, Tuple

_PUBLIC_SUBMODULES = [
    "cli",
    "config",
    "core",
    "generation",
    "patches",
    "pipelines",
    "rewards",
    "telemetry",
    "training",
]
_PUBLIC_ATTRS = {
    "hydra_cli": ("maxent_grpo.cli.hydra_cli", None),
}

__all__ = [
    "cli",
    "config",
    "core",
    "generation",
    "patches",
    "pipelines",
    "rewards",
    "telemetry",
    "training",
    "main",
    "parse_grpo_args",
    "hydra_cli",
]

_LAZY_MODULES: Dict[str, str] = {
    name: f"maxent_grpo.{name}" for name in _PUBLIC_SUBMODULES
}
_LAZY_ATTRS: Dict[str, Tuple[str, str | None]] = dict(_PUBLIC_ATTRS)


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


# Provide a lightweight handle for tests/consumers to monkeypatch without
# importing the full hydra CLI stack.
hydra_cli = _LazyModuleProxy("maxent_grpo.cli.hydra_cli")

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from . import cli as cli
    from . import config as config
    from . import core as core
    from . import generation as generation
    from . import patches as patches
    from . import pipelines as pipelines
    from . import rewards as rewards
    from . import telemetry as telemetry
    from . import training as training
    from .cli import hydra_cli as hydra_cli


def parse_grpo_args():
    """Parse GRPO CLI args via the training CLI parser."""

    from maxent_grpo.cli import parse_grpo_args as _parse_grpo_args

    return _parse_grpo_args()


def main(
    script_args: Any = None,
    training_args: Any = None,
    model_args: Any = None,
) -> Any:
    """Run the MaxEnt trainer when configs are provided, else delegate to Hydra."""

    if script_args is None or training_args is None or model_args is None:
        try:
            script_args, training_args, model_args = parse_grpo_args()
        except (ImportError, RuntimeError, SystemExit, ValueError):
            hydra_mod = globals().get("hydra_cli")
            if hydra_mod is None:
                from maxent_grpo.cli import hydra_cli as hydra_mod

                globals()["hydra_cli"] = hydra_mod
            return hydra_mod.maxent_entry()
    from maxent_grpo.pipelines.training.maxent import run_maxent_training

    return run_maxent_training(script_args, training_args, model_args)


def __getattr__(name: str) -> Any:
    """Lazily import submodules on first access.

    :param name: Attribute name corresponding to a lazy module entry.
    :returns: Imported module instance.
    :raises AttributeError: If ``name`` does not map to a known submodule.
    """

    if name in _LAZY_MODULES:
        module: ModuleType = import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module_name, attr = _LAZY_ATTRS[name]
        module = import_module(module_name)
        value = module if attr is None else getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(__all__)
