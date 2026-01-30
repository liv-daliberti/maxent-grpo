"""DeepSpeed and Accelerate integration helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from maxent_grpo.utils.imports import optional_import as _optional_dependency
from maxent_grpo.utils.imports import require_dependency as _require_dependency

try:  # Optional dependency for reading accelerate config files
    import yaml
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional
    yaml = None

LOG = logging.getLogger(__name__)


def require_deepspeed(context: str, module: str = "deepspeed") -> Any:
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


def _maybe_create_deepspeed_plugin() -> Optional[Any]:
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


__all__ = [
    "_maybe_create_deepspeed_plugin",
    "get_trl_prepare_deepspeed",
    "require_deepspeed",
]
