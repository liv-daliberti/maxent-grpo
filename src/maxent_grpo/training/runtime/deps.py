"""Dependency loading utilities used by the training runtime."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from maxent_grpo.utils.imports import (
    cached_import as _import_module,
    optional_import as _optional_dependency,
    require_dependency as _require_dependency,
)
from .torch_stub import _build_torch_stub

LOG = logging.getLogger(__name__)

# Ensure cached importer exposes a no-op cache_clear even when replaced in tests.
def __setattr__(name: str, value: Any) -> None:  # type: ignore[override]
    if name == "_import_module" and not hasattr(value, "cache_clear"):
        try:
            value.cache_clear = lambda: None  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            LOG.debug("Unable to attach cache_clear to custom importer.")
    globals()[name] = value


# Ensure cached importer exposes a no-op cache_clear when the underlying
# implementation does not provide one (defensive for test stubs).
if not hasattr(_import_module, "cache_clear"):
    _import_module.cache_clear = lambda: None  # type: ignore[attr-defined]
else:
    # Some cache wrappers remove cache_clear at runtime; keep a defensive alias.
    try:
        _ = _import_module.cache_clear
    except AttributeError:  # pragma: no cover - defensive
        _import_module.cache_clear = lambda: None  # type: ignore[attr-defined]

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
        # If tests or external code inserted a lightweight object into
        # ``sys.modules['torch']`` that is not a proper module, coerce it
        # into a real ModuleType so Python can import submodules
        # (e.g. ``torch._dynamo``) without raising "'torch' is not a package".
        try:
            if not isinstance(existing, ModuleType):
                # Wrap attributes on a fresh ModuleType to make it package-like.
                mod = ModuleType("torch")
                for k, v in getattr(existing, "__dict__", {}).items():
                    try:
                        setattr(mod, k, v)
                    except (TypeError, ValueError, AttributeError):
                        # Ignore attributes that can't be set.
                        LOG.debug("Skipping torch stub attribute %s during module wrap.", k)
                # Mark as package to allow submodule imports.
                mod.__spec__ = importlib.machinery.ModuleSpec(
                    "torch", loader=None, is_package=True
                )
                mod.__path__ = []
                sys.modules["torch"] = mod
                existing = mod
            # Ensure minimal helpers exist on the module so callers can use them.
            if not hasattr(existing, "SymBool"):
                sym_cls = getattr(_build_torch_stub(), "SymBool", None)
                if sym_cls is not None:
                    existing.SymBool = sym_cls
            if not hasattr(existing, "_dynamo"):
                setattr(
                    existing,
                    "_dynamo",
                    SimpleNamespace(disable=lambda fn=None, recursive=False: fn),
                )
            else:
                dyn = getattr(existing, "_dynamo", None)
                if dyn is None or not hasattr(dyn, "disable"):
                    setattr(
                        existing,
                        "_dynamo",
                        SimpleNamespace(disable=lambda fn=None, recursive=False: fn),
                    )
        except (TypeError, ValueError, AttributeError):
            # Conservative fallback: leave existing as-is.
            LOG.debug("Failed to normalize existing torch module; leaving as-is.")
        # Ensure torch._dynamo resolves to the real subpackage when available.
        if "torch._dynamo" not in sys.modules:
            try:
                sys.modules["torch._dynamo"] = importlib.import_module("torch._dynamo")
            except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError):
                # Fallback stub marked as a package so nested imports don't fail
                # with "'torch._dynamo' is not a package".
                dynamo_stub = ModuleType("torch._dynamo")
                dynamo_stub.__spec__ = importlib.machinery.ModuleSpec(
                    "torch._dynamo", loader=None, is_package=True
                )
                dynamo_stub.__path__ = []
                sys.modules["torch._dynamo"] = dynamo_stub
        # Always expose a _dynamo attribute on the module to satisfy downstream imports.
        if not hasattr(existing, "_dynamo"):
            setattr(existing, "_dynamo", sys.modules.get("torch._dynamo"))
        try:
            if not hasattr(existing, "version") or existing.version is None:
                existing.version = SimpleNamespace()
            if isinstance(existing.version, SimpleNamespace):
                if not hasattr(existing.version, "version"):
                    existing.version.version = "0.0.0"
                if not hasattr(existing.version, "__version__"):
                    existing.version.__version__ = "0.0.0"
                if not hasattr(existing.version, "cuda"):
                    existing.version.cuda = None
            if not hasattr(existing, "__version__"):
                existing.__version__ = "0.0.0"
        except (AttributeError, TypeError, ValueError) as exc:
            LOG.debug("Failed to normalize torch version metadata: %s", exc)
        try:
            if not hasattr(existing, "nested"):
                nested_mod = ModuleType("torch.nested")
                nested_mod.__spec__ = importlib.machinery.ModuleSpec(
                    "torch.nested", loader=None, is_package=True
                )
                sys.modules.setdefault("torch.nested", nested_mod)
                setattr(existing, "nested", nested_mod)
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            optim_mod = sys.modules.get("torch.optim")
            if not isinstance(optim_mod, ModuleType):
                optim_mod = ModuleType("torch.optim")
                optim_mod.__spec__ = importlib.machinery.ModuleSpec(
                    "torch.optim", loader=None, is_package=True
                )
                optim_mod.__path__ = []
                optim_mod.Optimizer = type("Optimizer", (), {})
                sys.modules["torch.optim"] = optim_mod
            if getattr(optim_mod, "__path__", None) is None:
                optim_mod.__path__ = []
            lr_sched_mod = sys.modules.get("torch.optim.lr_scheduler")
            if not isinstance(lr_sched_mod, ModuleType):
                lr_sched_mod = ModuleType("torch.optim.lr_scheduler")
                lr_sched_mod.__spec__ = importlib.machinery.ModuleSpec(
                    "torch.optim.lr_scheduler", loader=None, is_package=True
                )
                lr_sched_mod.__path__ = []
                lr_sched_mod.LRScheduler = type("LRScheduler", (), {})
                protected_name = "_" + "LRScheduler"
                setattr(lr_sched_mod, protected_name, lr_sched_mod.LRScheduler)
                sys.modules["torch.optim.lr_scheduler"] = lr_sched_mod
            optim_mod.lr_scheduler = lr_sched_mod
            if not hasattr(existing, "optim"):
                setattr(existing, "optim", optim_mod)
        except (AttributeError, TypeError, ValueError) as exc:
            LOG.debug("Failed to ensure torch.optim stubs: %s", exc)
        # Patch common helpers on existing stubs so downstream code can rely on them.
        try:
            stub = _build_torch_stub()
            if not hasattr(existing, "nn"):
                existing.nn = getattr(stub, "nn", None)
            if not hasattr(existing, "stack"):
                existing.stack = getattr(stub, "stack", None)
            if not hasattr(existing, "unique"):
                existing.unique = getattr(stub, "unique", None)
            if not hasattr(existing, "arange"):
                existing.arange = getattr(stub, "arange", None)
            if not hasattr(existing, "cuda"):
                existing.cuda = SimpleNamespace(is_available=lambda: False)
            # Ensure CUDA memory helpers exist on lightweight stubs.
            cuda_mod = getattr(existing, "cuda", None)
            if cuda_mod is not None:
                if not hasattr(cuda_mod, "memory_stats"):
                    try:
                        cuda_mod.memory_stats = lambda *_a, **_k: {}
                    except (AttributeError, TypeError):
                        pass
                for name in (
                    "current_allocated_memory",
                    "current_reserved_memory",
                    "memory_allocated",
                    "memory_reserved",
                    "max_memory_allocated",
                    "max_memory_reserved",
                ):
                    if not hasattr(cuda_mod, name):
                        try:
                            setattr(cuda_mod, name, lambda *_a, **_k: 0)
                        except (AttributeError, TypeError):
                            pass
            if not hasattr(existing, "xpu"):
                existing.xpu = getattr(
                    stub, "xpu", SimpleNamespace(is_available=lambda: False)
                )
            if not hasattr(existing, "mps"):
                existing.mps = getattr(
                    stub, "mps", SimpleNamespace(is_available=lambda: False)
                )
            if not hasattr(existing, "no_grad"):
                existing.no_grad = getattr(stub, "no_grad", None)
            if not hasattr(existing, "manual_seed"):
                existing.manual_seed = lambda *_a, **_k: None
            if existing.nn is not None:
                sys.modules.setdefault("torch.nn", existing.nn)
                func_mod = getattr(existing.nn, "functional", None)
                if func_mod is not None:
                    sys.modules.setdefault("torch.nn.functional", func_mod)
        except (AttributeError, TypeError, ValueError) as exc:
            LOG.debug("Failed to finalize existing torch stub: %s", exc)
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
    # Tests may reuse lightweight stubs; patch missing helpers when present.
    try:
        if not hasattr(torch_mod, "SymBool"):
            sym_cls = getattr(_build_torch_stub(), "SymBool", None)
            if sym_cls is not None:
                torch_mod.SymBool = sym_cls
        stub = _build_torch_stub()
        if not hasattr(torch_mod, "SymBool"):
            sym_cls = getattr(stub, "SymBool", None)
            if sym_cls is not None:
                torch_mod.SymBool = sym_cls
        if not hasattr(torch_mod, "manual_seed"):
            torch_mod.manual_seed = lambda *_a, **_k: None
        if not hasattr(torch_mod, "stack"):
            torch_mod.stack = getattr(stub, "stack", None)
        if not hasattr(torch_mod, "unique"):
            torch_mod.unique = getattr(stub, "unique", None)
        if not hasattr(torch_mod, "nn"):
            torch_mod.nn = getattr(stub, "nn", None)
        if not hasattr(torch_mod, "optim"):
            torch_mod.optim = getattr(stub, "optim", None)
        if not hasattr(torch_mod, "softmax"):
            torch_mod.softmax = getattr(stub, "softmax", None)
        if not hasattr(torch_mod, "log"):
            torch_mod.log = getattr(stub, "log", None)
        if not hasattr(torch_mod, "stack"):
            torch_mod.stack = getattr(stub, "stack", None)
        if not hasattr(torch_mod, "unique"):
            torch_mod.unique = getattr(stub, "unique", None)
        if not hasattr(torch_mod, "cuda"):
            torch_mod.cuda = SimpleNamespace(is_available=lambda: False)
        if not hasattr(torch_mod, "multiprocessing"):
            torch_mod.multiprocessing = SimpleNamespace(
                _is_in_bad_fork=False,
                current_process=lambda: SimpleNamespace(
                    _config=SimpleNamespace(_parallel=False)
                ),
            )
        else:
            mp = getattr(torch_mod, "multiprocessing", None)
            if mp is not None:
                if not hasattr(mp, "_is_in_bad_fork"):
                    setattr(mp, "_is_in_bad_fork", False)
                if not hasattr(mp, "current_process"):
                    mp.current_process = lambda: SimpleNamespace(
                        _config=SimpleNamespace(_parallel=False)
                    )
    except (AttributeError, TypeError):
        LOG.debug("Failed to finalize torch stub attributes.")
    return torch_mod


def require_dataloader(context: str) -> Any:
    """Return torch.utils.data.DataLoader with a descriptive error on failure."""

    hint = f"Torch's DataLoader is required for MaxEnt-GRPO {context}. Install torch first."
    try:
        torch_data = _import_module("torch.utils.data")
        if torch_data is None:
            # When the importer returns ``None`` explicitly, treat it as a hard
            # failure rather than attempting stub installation.
            raise RuntimeError(hint)
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - import guard
        torch_data = None
        try:
            _bootstrap = importlib.import_module("ops.sitecustomize")
            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                cache_clear = getattr(_import_module, "cache_clear", None)
                if callable(cache_clear):
                    cache_clear()
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
            data_mod.DataLoader = type(
                "DataLoader", (), {"__init__": lambda self, *a, **k: None}
            )
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

    if not hasattr(_import_module, "cache_clear"):
        _import_module.cache_clear = lambda: None  # type: ignore[attr-defined]
    hint = (
        f"Transformers is required for MaxEnt-GRPO {context}. "
        "Install it with `pip install transformers`."
    )
    try:
        transformers_mod = _import_module("transformers")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(hint) from exc

    # Accept SimpleNamespace/test stubs.
    if not isinstance(transformers_mod, ModuleType):
        if hasattr(transformers_mod, "PreTrainedModel") and hasattr(
            transformers_mod, "PreTrainedTokenizer"
        ):
            return (
                getattr(transformers_mod, "PreTrainedModel"),
                getattr(transformers_mod, "PreTrainedTokenizer"),
            )
        raise RuntimeError(hint)

    # Ensure nested package for transformers.models exists for downstream imports,
    # but avoid clobbering the real package when transformers is installed.
    if "transformers.models" not in sys.modules:
        try:
            importlib.import_module("transformers.models")
        except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError):
            models_mod = ModuleType("transformers.models")
            models_mod.__spec__ = importlib.machinery.ModuleSpec(
                "transformers.models", loader=None, is_package=True
            )
            models_mod.__path__ = []
            sys.modules["transformers.models"] = models_mod

    try:
        model_cls = getattr(transformers_mod, "PreTrainedModel", None)
        tokenizer_cls = getattr(transformers_mod, "PreTrainedTokenizer", None)
    except (ImportError, ModuleNotFoundError, RuntimeError):
        # When optional torch/transformers pieces are missing (e.g., flex attention
        # imports), fall back to lightweight stubs so test environments can still
        # import the training package.
        model_cls = None
        tokenizer_cls = None
    if model_cls is None or tokenizer_cls is None:
        # Create minimal stub base classes to satisfy type checks in tests.
        model_cls = type("PreTrainedModel", (), {})
        tokenizer_cls = type("PreTrainedTokenizer", (), {})
        transformers_mod.PreTrainedModel = model_cls  # type: ignore[attr-defined]
        transformers_mod.PreTrainedTokenizer = tokenizer_cls  # type: ignore[attr-defined]
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
