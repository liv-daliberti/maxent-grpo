"""Dependency loading utilities used by the training runtime."""

from __future__ import annotations

import os
import sys
from types import ModuleType
import importlib
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from maxent_grpo.utils.imports import (
    cached_import as _import_module,
    optional_import as _optional_dependency,
    require_dependency as _require_dependency,
)

from .torch_stub import _build_torch_stub

if TYPE_CHECKING:
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin
else:  # pragma: no cover - typing fallbacks
    Accelerator = Any
    DeepSpeedPlugin = Any

try:  # Optional dependency for reading accelerate config files
    import yaml
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional
    yaml = None


def require_torch(_context: str) -> Any:
    """Return the torch module or a stub for test environments."""

    existing = sys.modules.get("torch")
    if existing is not None:
        return existing
    try:
        torch_mod = _import_module("torch")
    except (ModuleNotFoundError, RuntimeError):  # pragma: no cover - import guard
        torch_mod = None
        try:
            _bootstrap = importlib.import_module("ops.sitecustomize")
            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                _import_module.cache_clear()
                torch_mod = _import_module("torch")
        except (ImportError, AttributeError, RuntimeError):
            torch_mod = None
        if torch_mod is None:
            torch_mod = _build_torch_stub()
    required_attrs = ("tensor", "full", "ones_like", "zeros")

    def _missing_required(mod: Any) -> bool:
        return mod is None or any(not hasattr(mod, attr) for attr in required_attrs)

    if _missing_required(torch_mod):
        try:
            _bootstrap = importlib.import_module("ops.sitecustomize")
            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                _import_module.cache_clear()
                torch_mod = _import_module("torch")
        except (ImportError, AttributeError, RuntimeError):
            torch_mod = None

    if _missing_required(torch_mod):
        torch_mod = _build_torch_stub()
    if "torch" not in sys.modules:
        sys.modules["torch"] = torch_mod
    return torch_mod


def require_dataloader(context: str) -> Any:
    """Return torch.utils.data.DataLoader with a descriptive error on failure."""

    hint = f"Torch's DataLoader is required for MaxEnt-GRPO {context}. Install torch first."
    try:
        torch_data = _import_module("torch.utils.data")
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
    ):  # pragma: no cover - import guard
        torch_data = None
        try:
            _bootstrap = importlib.import_module("ops.sitecustomize")
            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                _import_module.cache_clear()
                torch_data = _import_module("torch.utils.data")
        except (ImportError, AttributeError, RuntimeError):
            torch_data = None

        if torch_data is None:
            # Minimal stub for environments running without torch (e.g., perf scripts).
            torch_mod = sys.modules.get("torch")
            if torch_mod is None:
                torch_mod = _build_torch_stub()
                sys.modules["torch"] = torch_mod
            utils_mod = getattr(torch_mod, "utils", None)
            if utils_mod is None:
                utils_mod = ModuleType("torch.utils")
                sys.modules["torch.utils"] = utils_mod
                torch_mod.utils = utils_mod
            data_mod = ModuleType("torch.utils.data")
            data_mod.DataLoader = type("DataLoader", (), {})
            data_mod.Sampler = type("Sampler", (), {})
            sys.modules["torch.utils.data"] = data_mod
            utils_mod.data = data_mod
            torch_data = data_mod
    if torch_data is None:
        raise RuntimeError(hint)
    try:
        return getattr(torch_data, "DataLoader")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "torch.utils.data.DataLoader is missing; update your torch installation."
        ) from exc


def require_accelerator(context: str) -> Any:
    """Return accelerate.Accelerator or raise a helpful RuntimeError."""

    hint = (
        f"Accelerate is required for MaxEnt-GRPO {context}. "
        "Install it with `pip install accelerate`."
    )
    try:
        accelerate_mod = _import_module("accelerate")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(hint) from exc
    try:
        return getattr(accelerate_mod, "Accelerator")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "accelerate.Accelerator is unavailable; upgrade the accelerate package."
        ) from exc


def require_transformer_base_classes(context: str) -> Tuple[Any, Any]:
    """Return (PreTrainedModel, PreTrainedTokenizer) with clear failure messages."""

    hint = (
        f"Transformers is required for MaxEnt-GRPO {context}. "
        "Install it with `pip install transformers`."
    )
    try:
        transformers_mod = _import_module("transformers")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(hint) from exc
    try:
        model_cls = getattr(transformers_mod, "PreTrainedModel")
        tokenizer_cls = getattr(transformers_mod, "PreTrainedTokenizer")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Transformers does not expose PreTrainedModel/Tokenizer; upgrade transformers."
        ) from exc
    return model_cls, tokenizer_cls


def require_deepspeed(context: str, module: str = "deepspeed") -> ModuleType:
    """Return a DeepSpeed module import or raise a contextual RuntimeError."""

    hint = (
        f"DeepSpeed is required for MaxEnt-GRPO {context}. "
        "Install it with `pip install deepspeed`."
    )
    try:
        return _require_dependency(module, hint)
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(hint) from exc


def get_trl_prepare_deepspeed() -> Optional[Any]:
    """Return TRL's prepare_deepspeed helper when available."""

    utils_module = _optional_dependency("trl.trainer.utils")
    if utils_module is None:
        return None
    prepare = getattr(utils_module, "prepare_deepspeed", None)
    if not callable(prepare):
        return None
    return prepare


def _maybe_create_deepspeed_plugin() -> Optional[DeepSpeedPlugin]:
    """Construct a DeepSpeedPlugin from Accelerate env/config when available."""

    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() != "true":
        return None
    ds_module = _require_dependency(
        "accelerate.utils",
        (
            "DeepSpeed integration requires the Accelerate package. "
            "Install it via `pip install accelerate[deepspeed]`."
        ),
    )
    try:
        ds_class = getattr(ds_module, "DeepSpeedPlugin")
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "accelerate.utils does not expose DeepSpeedPlugin; update Accelerate."
        ) from exc

    ds_cfg: Dict[str, Any] = {}
    cfg_path = os.environ.get("ACCELERATE_CONFIG_FILE")
    if cfg_path and yaml is not None and os.path.isfile(cfg_path):
        handled_exceptions: Tuple[type, ...] = (OSError, ValueError)
        yaml_error = getattr(yaml, "YAMLError", None)
        if isinstance(yaml_error, type):
            handled_exceptions = handled_exceptions + (yaml_error,)
        try:
            with open(cfg_path, "r", encoding="utf-8") as cfg_file:
                raw = yaml.safe_load(cfg_file) or {}
            ds_cfg = raw.get("deepspeed_config") or {}
        except handled_exceptions:
            ds_cfg = {}
    zero_stage_raw = ds_cfg.get("zero_stage", 3)
    zero_stage = int(zero_stage_raw) if zero_stage_raw is not None else None
    offload_param = ds_cfg.get("offload_param_device")
    offload_optim = ds_cfg.get("offload_optimizer_device")
    zero3_init_flag = ds_cfg.get("zero3_init_flag")
    zero3_save = ds_cfg.get("zero3_save_16bit_model")

    kwargs = {
        "zero_stage": zero_stage,
        "offload_param_device": offload_param,
        "offload_optimizer_device": offload_optim,
        "zero3_init_flag": zero3_init_flag,
        "zero3_save_16bit_model": zero3_save,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if not kwargs:
        return None
    return ds_class(**kwargs)


def sync_import_helpers(import_fn: Any, optional_fn: Any, require_fn: Any) -> None:
    """Synchronize import helper references for callers that monkeypatch them."""
    globals()["_import_module"] = import_fn
    globals()["_optional_dependency"] = optional_fn
    globals()["_require_dependency"] = require_fn


def maybe_create_deepspeed_plugin() -> Optional[DeepSpeedPlugin]:
    """Public wrapper for DeepSpeed plugin creation."""
    return _maybe_create_deepspeed_plugin()


__all__ = [
    "Accelerator",
    "DeepSpeedPlugin",
    "_import_module",
    "_maybe_create_deepspeed_plugin",
    "_optional_dependency",
    "_require_dependency",
    "maybe_create_deepspeed_plugin",
    "get_trl_prepare_deepspeed",
    "require_accelerator",
    "require_dataloader",
    "require_deepspeed",
    "require_torch",
    "require_transformer_base_classes",
    "sync_import_helpers",
]
