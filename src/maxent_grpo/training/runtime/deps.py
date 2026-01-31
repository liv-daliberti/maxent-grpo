"""Dependency loading utilities used by the training runtime."""

from __future__ import annotations

import logging
import os
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from maxent_grpo.utils.imports import (
    cached_import as _import_module,
    optional_import as _optional_dependency,
    require_dependency as _require_dependency,
)

LOG = logging.getLogger(__name__)

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


def require_torch(context: str) -> ModuleType:
    """Return the torch module or raise a helpful RuntimeError."""

    hint = (
        f"PyTorch is required for MaxEnt-GRPO {context}. "
        "Install it via `pip install torch`."
    )
    try:
        return _require_dependency("torch", hint)
    except ImportError as exc:
        raise RuntimeError(hint) from exc


def require_dataloader(context: str) -> Any:
    """Return ``torch.utils.data.DataLoader`` with a descriptive error on failure."""

    hint = (
        f"Torch's DataLoader is required for MaxEnt-GRPO {context}. "
        "Install torch first."
    )
    try:
        torch_data = _require_dependency("torch.utils.data", hint)
    except ImportError as exc:
        raise RuntimeError(hint) from exc
    dataloader_cls = getattr(torch_data, "DataLoader", None)
    if dataloader_cls is None:
        raise RuntimeError(hint)
    return dataloader_cls


def require_accelerator(context: str) -> Any:
    """Return accelerate.Accelerator or raise a helpful RuntimeError."""

    hint = (
        f"Accelerate is required for MaxEnt-GRPO {context}. "
        "Install it via `pip install accelerate`."
    )
    try:
        accelerate_mod = _require_dependency("accelerate", hint)
    except ImportError as exc:
        raise RuntimeError(hint) from exc
    accelerator_cls = getattr(accelerate_mod, "Accelerator", None)
    if accelerator_cls is None:
        raise RuntimeError(hint)
    return accelerator_cls


def require_transformer_base_classes(context: str) -> Tuple[Any, Any]:
    """Return (PreTrainedModel, PreTrainedTokenizer) with clear failure messages."""

    hint = (
        f"Transformers is required for MaxEnt-GRPO {context}. "
        "Install it via `pip install transformers`."
    )
    try:
        transformers_mod = _require_dependency("transformers", hint)
    except ImportError as exc:
        raise RuntimeError(hint) from exc
    model_cls = getattr(transformers_mod, "PreTrainedModel", None)
    tokenizer_cls = getattr(transformers_mod, "PreTrainedTokenizer", None)
    if model_cls is None or tokenizer_cls is None:
        try:
            from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel
            from transformers.tokenization_utils import (
                PreTrainedTokenizer as _PreTrainedTokenizer,
            )
        except Exception as exc:
            raise RuntimeError(hint) from exc
        model_cls = model_cls or _PreTrainedModel
        tokenizer_cls = tokenizer_cls or _PreTrainedTokenizer
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
    ds_class = getattr(ds_module, "DeepSpeedPlugin", None)
    if ds_class is None:
        raise ImportError(
            "accelerate.utils does not expose DeepSpeedPlugin; update Accelerate."
        )

    ds_cfg: Dict[str, Any] = {}
    cfg_path = os.environ.get("ACCELERATE_CONFIG_FILE")
    if cfg_path and yaml is not None and os.path.isfile(cfg_path):
        handled_exceptions: Tuple[type[BaseException], ...] = (OSError, ValueError)
        yaml_error = getattr(yaml, "YAMLError", None)
        if isinstance(yaml_error, type) and issubclass(yaml_error, BaseException):
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


def maybe_create_deepspeed_plugin() -> Optional[DeepSpeedPlugin]:
    """Public wrapper for DeepSpeed plugin creation."""

    return _maybe_create_deepspeed_plugin()


__all__ = [
    "Accelerator",
    "DeepSpeedPlugin",
    "_import_module",
    "_optional_dependency",
    "_require_dependency",
    "_maybe_create_deepspeed_plugin",
    "maybe_create_deepspeed_plugin",
    "get_trl_prepare_deepspeed",
    "require_accelerator",
    "require_dataloader",
    "require_deepspeed",
    "require_torch",
    "require_transformer_base_classes",
]
