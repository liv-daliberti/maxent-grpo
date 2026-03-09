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

Hydra-powered multi-command CLI for MaxEnt-GRPO workflows.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from contextlib import nullcontext
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, cast
from collections.abc import Mapping

from maxent_grpo.cli._test_hooks import ensure_usercustomize_loaded

from maxent_grpo.cli.config_validation import validate_training_config
from maxent_grpo.config import (
    GRPOConfig,
    GRPOScriptArguments,
    load_grpo_recipe,
)


class _HydraStub:
    """Minimal Hydra-like stub used when hydra is absent."""

    def main(
        self, *_args: Any, **_kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that forwards directly to the wrapped function.

        :param _args: Positional arguments ignored by the stub.
        :param _kwargs: Keyword arguments ignored by the stub.
        :returns: Decorator mimicking :func:`hydra.main`.
        """

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def _wrapped(*_a: Any, **_k: Any) -> Any:
                return fn(*_a, **_k)

            return _wrapped

        return _decorator


if TYPE_CHECKING:  # pragma: no cover - type hints only
    import hydra
    from omegaconf import DictConfig, OmegaConf, open_dict
else:
    DictConfig: type[Any]
    OmegaConf: type[Any]
    open_dict: Any
    try:  # Optional dependency; provide stubs so linting/tests can import.
        import hydra
        from omegaconf import DictConfig as _DictConfig
        from omegaconf import OmegaConf as _OmegaConf
        from omegaconf import open_dict as _open_dict
    except ImportError:  # pragma: no cover - hydra not installed in minimal envs
        hydra = _HydraStub()

        class _DictConfigStub(dict):
            """Minimal stub so type hints resolve without hydra installed."""

        class _OmegaConfStub:
            @staticmethod
            def to_object(cfg: Any) -> Any:
                return cfg

            @staticmethod
            def to_yaml(cfg: Any) -> str:
                return str(cfg)

            @staticmethod
            def create(payload: Any) -> Any:
                return payload

            @staticmethod
            def structured(obj: Any) -> Any:
                return obj

        _DictConfig = _DictConfigStub
        _OmegaConf = _OmegaConfStub
        _open_dict = nullcontext
    DictConfig = _DictConfig
    OmegaConf = _OmegaConf
    open_dict = _open_dict


class _FallbackModelConfig:
    """Trivial stand-in for :class:`trl.ModelConfig` when TRL is absent."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _resolve_model_config_cls() -> type[Any]:
    """Return the TRL ``ModelConfig`` type or a stub when TRL is unavailable."""

    try:
        from trl import ModelConfig as _model_config_cls
    except (ImportError, ModuleNotFoundError, AttributeError):
        return _FallbackModelConfig
    return _model_config_cls


@dataclass
class BaselineCommand:
    """GRPO training command options for the baseline recipe.

    :param recipe: Optional recipe file path to load default configs from.
    :param script: Script-level overrides passed to GRPO script arguments.
    :param training: Training argument overrides passed to GRPO config.
    :param model: Model argument overrides passed to TRL model config.
    """

    recipe: Optional[str] = None
    script: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaxentCommand:
    """GRPO training command options for the MaxEnt recipe.

    :param recipe: Optional recipe file path to load default configs from.
    :param script: Script-level overrides passed to GRPO script arguments.
    :param training: Training argument overrides passed to GRPO config.
    :param model: Model argument overrides passed to TRL model config.
    """

    recipe: Optional[str] = None
    script: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HydraRootConfig:
    """Hydra root configuration covering all supported CLI commands.

    :param command: Name of the subcommand to run.
    :param baseline: Baseline training command configuration.
    :param maxent: MaxEnt training command configuration.
    """

    command: str = "train-baseline"
    baseline: BaselineCommand = field(default_factory=BaselineCommand)
    maxent: MaxentCommand = field(default_factory=MaxentCommand)


_HYDRA_CONFIG_NAME = "maxent_grpo_cli"
_HYDRA_CONFIG_STATE = {"registered": False}


def _load_config_store() -> Optional[type[Any]]:
    """Return Hydra's ConfigStore class if available."""

    try:
        module = importlib.import_module("hydra.core.config_store")
    except ImportError:
        return None
    return getattr(module, "ConfigStore", None)


def _register_hydra_config() -> Optional[str]:
    """
    Register :class:`HydraRootConfig` with Hydra's config store.

    Hydra validates CLI overrides (``command=``) against the registered
    config schema.  When Hydra is present and the config has not been
    registered, overrides like ``command=train-maxent`` raise
    ``Could not override 'command'`` before :func:`hydra_main` runs.
    """

    if not isinstance(hydra, types.ModuleType):
        return None
    if _HYDRA_CONFIG_STATE["registered"]:
        return _HYDRA_CONFIG_NAME
    config_store_cls = _load_config_store()
    if config_store_cls is None:
        return None
    cs = config_store_cls.instance()
    try:
        cs.store(name=_HYDRA_CONFIG_NAME, node=HydraRootConfig)
    except (KeyError, RuntimeError, TypeError, ValueError, AssertionError):
        # Defensive: ignore double registration or mismatched config errors.
        _HYDRA_CONFIG_STATE["registered"] = True
        return _HYDRA_CONFIG_NAME
    _HYDRA_CONFIG_STATE["registered"] = True
    return _HYDRA_CONFIG_NAME


def _maybe_insert_command(default_command: str) -> None:
    """Ensure hydra sees a command override for convenience entrypoints.

    :param default_command: Command name inserted when no explicit ``command=`` is present.
    :returns: ``None``; updates ``sys.argv`` in-place when needed.
    """

    if not any(
        arg.startswith("command=") or arg.startswith("+command=")
        for arg in sys.argv[1:]
    ):
        sys.argv.insert(1, f"command={default_command}")


def _resolve_recipe_path(cmd: BaselineCommand | MaxentCommand) -> Optional[str]:
    """Return the explicit recipe path or fall back to ``$GRPO_RECIPE``."""

    return getattr(cmd, "recipe", None) or os.environ.get("GRPO_RECIPE")


def _build_grpo_configs(
    cmd: BaselineCommand | MaxentCommand,
) -> tuple[GRPOScriptArguments, GRPOConfig, Any]:
    """Construct GRPO config objects from a command block.

    :param cmd: Command payload defining script, training, and model sections.
    :returns: Tuple of ``(script_args, training_args, model_config)`` ready to pass to training pipelines.
    """

    recipe_path = _resolve_recipe_path(cmd)
    model_config_cls = _resolve_model_config_cls()
    if recipe_path:
        script_args, training_args, model_args = load_grpo_recipe(
            recipe_path, model_config_cls=model_config_cls
        )
        _apply_overrides(script_args, getattr(cmd, "script", None))
        _apply_overrides(training_args, getattr(cmd, "training", None))
        _apply_overrides(model_args, getattr(cmd, "model", None))
        return script_args, training_args, model_args

    return (
        GRPOScriptArguments(**cmd.script),
        GRPOConfig(**cmd.training),
        model_config_cls(**cmd.model),
    )


def _merge_mapping(
    base: Mapping[str, Any], updates: Mapping[str, Any]
) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if (
            isinstance(value, Mapping)
            and not isinstance(value, str)
            and isinstance(merged.get(key), Mapping)
        ):
            merged[key] = _merge_mapping(
                merged[key], value
            )
        else:
            merged[key] = value
    return merged


def _apply_overrides(target: Any, overrides: Optional[Mapping[str, Any]]) -> Any:
    if target is None or not overrides:
        return target
    for key, value in overrides.items():
        if value is None:
            continue
        if (
            isinstance(value, Mapping)
            and not isinstance(value, str)
        ):
            current = getattr(target, key, None)
            if isinstance(current, Mapping):
                setattr(target, key, _merge_mapping(current, value))
            elif current is not None and (
                is_dataclass(current) or hasattr(current, "__dict__")
            ):
                _apply_overrides(current, value)
            else:
                setattr(target, key, dict(value))
        else:
            setattr(target, key, value)
    return target


def hydra_main(cfg: Optional[DictConfig] = None) -> Any:
    """Dispatch hydra-configured subcommands (direct-call friendly).

    :param cfg: Optional Hydra configuration object or plain dict derived from CLI files.
    :returns: Result of the executed command, or ``None`` for commands that only have side effects.
    :raises ValueError: If an unsupported command name is supplied.
    """

    # When hydra is monkeypatched to a stub (tests), delegate to it directly.
    if not isinstance(hydra, types.ModuleType):
        return hydra.main()(lambda *_a, **_k: None)(cfg)

    if isinstance(cfg, HydraRootConfig):
        root = cfg
    elif hasattr(OmegaConf, "structured"):
        structured_root = OmegaConf.structured(HydraRootConfig())
        if cfg is not None:
            with open_dict(structured_root):
                structured_root.merge_with(cfg)
        conf = OmegaConf.to_object(structured_root)
        if isinstance(conf, HydraRootConfig):
            root = conf
        elif isinstance(conf, Mapping):
            conf_map = cast(Mapping[str, Any], conf)
            root = HydraRootConfig(**dict(conf_map))
        else:
            root = HydraRootConfig()
    else:
        payload = cfg or {}
        if isinstance(payload, HydraRootConfig):
            root = payload
        elif isinstance(payload, dict):
            root = HydraRootConfig(**payload)
        else:
            root = HydraRootConfig()
    # Allow CLI-style `command=` overrides from sys.argv even when cfg is absent.
    cmd = root.command
    for arg in sys.argv[1:]:
        if arg.startswith("command=") or arg.startswith("+command="):
            cmd = arg.split("=", 1)[1]
            root.command = cmd
            break
    if cmd == "train-baseline":
        from maxent_grpo.training.baseline import run_baseline_training

        script_args, training_args, model_args = _build_grpo_configs(root.baseline)
        baseline_recipe = _resolve_recipe_path(root.baseline)
        validate_training_config(
            training_args,
            command="train-baseline",
            source=baseline_recipe,
        )
        run_baseline_training(script_args, training_args, model_args)
    elif cmd == "train-maxent":
        from maxent_grpo.training.baseline import run_baseline_training

        script_args, training_args, model_args = _build_grpo_configs(root.maxent)
        maxent_recipe = _resolve_recipe_path(root.maxent)
        validate_training_config(
            training_args,
            command="train-maxent",
            source=maxent_recipe,
        )
        run_baseline_training(script_args, training_args, model_args)
    else:
        raise ValueError(f"Unsupported command: {cmd}")


def hydra_entry() -> None:
    """Entry point for the top-level Hydra CLI.

    :returns: ``None`` after invoking the configured command.
    """
    _invoke_hydra_cli()


def baseline_entry() -> None:
    """Console script wrapper for baseline training.

    :returns: ``None`` after dispatching to Hydra.
    """
    _maybe_insert_command("train-baseline")
    _invoke_hydra_cli()


def _invoke_hydra_cli() -> Any:
    """Invoke hydra_main through Hydra's decorator wrapper for CLI use.

    :returns: Result of :func:`hydra_main`, forwarded directly.
    """
    ensure_usercustomize_loaded()
    if not isinstance(hydra, types.ModuleType):
        return hydra_main()
    config_name = _register_hydra_config()
    decorated = hydra.main(
        version_base=None,
        config_name=config_name,
    )(hydra_main)
    return decorated()
