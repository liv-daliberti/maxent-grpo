"""Dependency loading utilities used by the training runtime."""

from __future__ import annotations

import logging
import os
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
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


def _allow_stubbed_deps() -> bool:
    """Return True when lightweight dependency stubs are allowed."""

    return "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def _install_torch_stub(hint: str) -> ModuleType:
    """Install a lightweight torch stub so imports can succeed in tests."""

    existing = sys.modules.get("torch")
    if isinstance(existing, ModuleType) and getattr(existing, "__MAXENT_STUB__", False):
        return existing

    try:
        import numpy as _np
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover - be defensive
        _np = None

    class _StubDType:
        def __init__(self, name: str, np_dtype: Any = None) -> None:
            self.name = f"torch.{name}"
            self.np_dtype = np_dtype

        def __repr__(self) -> str:  # pragma: no cover - representational only
            return self.name

    class _StubTensor:
        def __init__(self, data: Any, dtype: Any = None) -> None:
            self.arr = _np.array(data) if _np is not None else data
            self.dtype = dtype

        def detach(self) -> "_StubTensor":
            return self

        def float(self) -> "_StubTensor":
            return self

        def cpu(self) -> "_StubTensor":
            return self

        def to(self, *_args: Any, **_kwargs: Any) -> "_StubTensor":
            return self

        def view(self, *shape: int) -> "_StubTensor":
            if _np is None:
                return self
            return _StubTensor(self.arr.reshape(*shape), dtype=self.dtype)

        def reshape(self, *shape: int) -> "_StubTensor":
            return self.view(*shape)

        def clamp(
            self,
            min_value: Optional[float] = None,
            max_value: Optional[float] = None,
            **kwargs: Any,
        ) -> "_StubTensor":
            if "min" in kwargs and min_value is None:
                min_value = kwargs["min"]
            if "max" in kwargs and max_value is None:
                max_value = kwargs["max"]
            if _np is None:
                return self
            data = self.arr
            if min_value is not None:
                data = _np.maximum(data, min_value)
            if max_value is not None:
                data = _np.minimum(data, max_value)
            return _StubTensor(data, dtype=self.dtype)

        def numel(self) -> int:
            if _np is None:
                try:
                    return len(self.arr)
                except TypeError:
                    return 1
            return int(self.arr.size)

        def size(self, dim: Optional[int] = None) -> Any:
            if _np is None:
                return len(self.arr) if dim is not None else ()
            if dim is None:
                return self.arr.shape
            return self.arr.shape[dim]

        def tolist(self) -> Any:
            if _np is None:
                return list(self.arr) if isinstance(self.arr, list) else self.arr
            return self.arr.tolist()

        def item(self) -> Any:
            if _np is None:
                return self.arr
            return self.arr.item()

        def sum(self, dim: Optional[int] = None) -> "_StubTensor":
            if _np is None:
                return self
            return _StubTensor(self.arr.sum(axis=dim), dtype=self.dtype)

        def max(self) -> Any:
            if _np is None:
                return self.arr
            return self.arr.max()

        def __len__(self) -> int:
            try:
                return len(self.arr)
            except TypeError:
                return 1

        def __iter__(self):
            return iter(self.arr)

        def __getitem__(self, idx: Any) -> Any:
            if _np is None:
                return self.arr[idx]
            value = self.arr[idx]
            if isinstance(value, (_np.ndarray, _np.generic)):
                return _StubTensor(value, dtype=self.dtype)
            return value

        def new_full(self, shape: Any, fill_value: float) -> "_StubTensor":
            if _np is None:
                return _StubTensor([fill_value], dtype=self.dtype)
            return _StubTensor(_np.full(shape, fill_value), dtype=self.dtype)

        def __float__(self) -> float:
            if _np is None:
                try:
                    return float(self.arr)
                except Exception:
                    return 0.0
            return float(self.arr.item())

        def _binary_op(self, other: Any, op) -> "_StubTensor":
            if isinstance(other, _StubTensor):
                other = other.arr
            if _np is None:
                try:
                    return _StubTensor(op(self.arr, other), dtype=self.dtype)
                except (TypeError, ValueError, OverflowError):
                    return self
            return _StubTensor(op(self.arr, other), dtype=self.dtype)

        def __add__(self, other: Any) -> "_StubTensor":
            return self._binary_op(other, lambda a, b: a + b)

        def __sub__(self, other: Any) -> "_StubTensor":
            return self._binary_op(other, lambda a, b: a - b)

        def __mul__(self, other: Any) -> "_StubTensor":
            return self._binary_op(other, lambda a, b: a * b)

        def __truediv__(self, other: Any) -> "_StubTensor":
            return self._binary_op(other, lambda a, b: a / b)

        def __lt__(self, other: Any) -> "_StubTensor":
            return self._binary_op(other, lambda a, b: a < b)

        def __le__(self, other: Any) -> "_StubTensor":
            return self._binary_op(other, lambda a, b: a <= b)

        def __gt__(self, other: Any) -> "_StubTensor":
            return self._binary_op(other, lambda a, b: a > b)

        def __ge__(self, other: Any) -> "_StubTensor":
            return self._binary_op(other, lambda a, b: a >= b)

        def lt(self, other: Any) -> "_StubTensor":
            return self.__lt__(other)

        def le(self, other: Any) -> "_StubTensor":
            return self.__le__(other)

        def gt(self, other: Any) -> "_StubTensor":
            return self.__gt__(other)

        def ge(self, other: Any) -> "_StubTensor":
            return self.__ge__(other)

        def __array__(self, dtype: Any = None) -> Any:  # pragma: no cover - numpy interop
            if _np is None:
                return self.arr
            return self.arr.astype(dtype) if dtype is not None else self.arr

    def _wrap(data: Any, dtype: Any = None) -> _StubTensor:
        return _StubTensor(data, dtype=dtype)

    def _zeros(shape: Any, dtype: Any = None) -> _StubTensor:
        if _np is None:
            return _wrap([0 for _ in range(int(shape[0]) if isinstance(shape, (list, tuple)) else 1)], dtype)
        return _wrap(_np.zeros(shape), dtype=dtype)

    def _ones(shape: Any, dtype: Any = None) -> _StubTensor:
        if _np is None:
            return _wrap([1 for _ in range(int(shape[0]) if isinstance(shape, (list, tuple)) else 1)], dtype)
        return _wrap(_np.ones(shape), dtype=dtype)

    def _full(shape: Any, fill_value: float, dtype: Any = None) -> _StubTensor:
        if _np is None:
            return _wrap([fill_value for _ in range(int(shape[0]) if isinstance(shape, (list, tuple)) else 1)], dtype)
        return _wrap(_np.full(shape, fill_value), dtype=dtype)

    def _arange(*args: Any, **_kwargs: Any) -> _StubTensor:
        if _np is None:
            return _wrap(list(range(*args)))
        return _wrap(_np.arange(*args))

    def _cat(tensors: Any, dim: int = 0) -> _StubTensor:
        if _np is None:
            data = []
            for t in tensors:
                data.extend(getattr(t, "arr", t))
            return _wrap(data)
        arrs = [_np.array(getattr(t, "arr", t)) for t in tensors]
        return _wrap(_np.concatenate(arrs, axis=dim))

    def _stack(tensors: Any, dim: int = 0) -> _StubTensor:
        if _np is None:
            return _wrap([getattr(t, "arr", t) for t in tensors])
        arrs = [_np.array(getattr(t, "arr", t)) for t in tensors]
        return _wrap(_np.stack(arrs, axis=dim))

    stub = ModuleType("torch")
    stub.__MAXENT_STUB__ = True
    stub.__spec__ = getattr(stub, "__spec__", None) or ModuleSpec("torch", loader=None)
    stub.__path__ = getattr(stub, "__path__", [])
    stub.Tensor = _StubTensor
    stub.tensor = _wrap
    stub.as_tensor = _wrap
    stub.zeros = _zeros
    stub.ones = _ones
    def _shape_from(value: Any) -> Any:
        shape = getattr(value, "shape", None)
        if shape is not None:
            return shape
        try:
            return (len(value),)
        except TypeError:
            return (1,)

    stub.ones_like = lambda tensor, *_a, **_k: _ones(
        _shape_from(getattr(tensor, "arr", tensor))
    )
    stub.zeros_like = lambda tensor, *_a, **_k: _zeros(
        _shape_from(getattr(tensor, "arr", tensor))
    )
    stub.full = _full
    stub.arange = _arange
    stub.cat = _cat
    stub.stack = _stack
    stub.float32 = _StubDType("float32", _np.float32 if _np is not None else None)
    stub.float16 = _StubDType("float16", _np.float16 if _np is not None else None)
    stub.bfloat16 = _StubDType("bfloat16")
    long_dtype = _StubDType("int64", _np.int64 if _np is not None else None)
    stub.long = long_dtype
    stub.int64 = long_dtype
    stub.bool = _StubDType("bool", _np.bool_ if _np is not None else None)
    try:
        from contextlib import nullcontext as _nullcontext
    except ImportError:  # pragma: no cover - stdlib fallback
        _nullcontext = None

    def _no_grad() -> Any:
        if _nullcontext is None:
            return SimpleNamespace(
                __enter__=lambda *_a, **_k: None,
                __exit__=lambda *_a, **_k: False,
            )
        return _nullcontext()

    stub.no_grad = _no_grad
    stub.device = lambda *_a, **_k: "cpu"
    stub.cuda = SimpleNamespace(is_available=lambda: False)
    stub.distributed = None
    stub.__version__ = "0.0.0-stub"
    stub.nn = SimpleNamespace(
        Module=object,
        Linear=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError(hint)),
        utils=SimpleNamespace(clip_grad_norm_=lambda *_a, **_k: 0.0),
        functional=SimpleNamespace(),
    )
    stub.optim = SimpleNamespace(Optimizer=object, AdamW=None)

    sys.modules["torch"] = stub
    nn_mod = sys.modules.setdefault("torch.nn", ModuleType("torch.nn"))
    if not hasattr(nn_mod, "Module"):
        nn_mod.Module = object
    if not hasattr(nn_mod, "Linear"):
        nn_mod.Linear = stub.nn.Linear
    nn_fn_mod = sys.modules.setdefault("torch.nn.functional", ModuleType("torch.nn.functional"))
    setattr(nn_mod, "functional", nn_fn_mod)
    optim_mod = sys.modules.setdefault("torch.optim", ModuleType("torch.optim"))
    if not hasattr(optim_mod, "Optimizer"):
        optim_mod.Optimizer = object
    if not hasattr(optim_mod, "AdamW"):
        optim_mod.AdamW = None
    utils_mod = sys.modules.setdefault("torch.utils", ModuleType("torch.utils"))
    data_mod = sys.modules.setdefault("torch.utils.data", ModuleType("torch.utils.data"))
    if not hasattr(data_mod, "DataLoader"):
        class _DataLoader:  # pragma: no cover - stubbed fallback
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError(hint)

        data_mod.DataLoader = _DataLoader
    if not hasattr(data_mod, "Sampler"):
        class _Sampler:  # pragma: no cover - stubbed fallback
            pass

        data_mod.Sampler = _Sampler
    setattr(utils_mod, "data", data_mod)
    return stub


def _install_accelerate_stub(_hint: str) -> ModuleType:
    """Install a lightweight accelerate stub for tests."""

    existing = sys.modules.get("accelerate")
    if isinstance(existing, ModuleType) and getattr(existing, "__MAXENT_STUB__", False):
        return existing

    class _Accelerator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)
            self.is_main_process = True
            self.num_processes = 1
            self.process_index = 0
            self.device = "cpu"

        def wait_for_everyone(self) -> None:
            return None

    accel_mod = ModuleType("accelerate")
    accel_mod.__MAXENT_STUB__ = True
    accel_mod.__spec__ = getattr(accel_mod, "__spec__", None) or ModuleSpec(
        "accelerate", loader=None
    )
    accel_mod.__path__ = getattr(accel_mod, "__path__", [])
    setattr(accel_mod, "Accelerator", _Accelerator)

    state_mod = ModuleType("accelerate.state")
    state_mod.__spec__ = getattr(state_mod, "__spec__", None) or ModuleSpec(
        "accelerate.state", loader=None
    )

    class _DistributedType:
        DEEPSPEED = "deepspeed"

    setattr(state_mod, "DistributedType", _DistributedType)

    sys.modules["accelerate"] = accel_mod
    sys.modules["accelerate.state"] = state_mod
    return accel_mod


def require_torch(context: str) -> ModuleType:
    """Return the torch module or raise a helpful RuntimeError."""

    hint = (
        f"PyTorch is required for MaxEnt-GRPO {context}. "
        "Install it via `pip install torch`."
    )
    try:
        return _require_dependency("torch", hint)
    except ImportError as exc:
        if _allow_stubbed_deps():
            return _install_torch_stub(hint)
        raise RuntimeError(hint) from exc


def require_dataloader(context: str) -> Any:
    """Return ``torch.utils.data.DataLoader`` with a descriptive error on failure."""

    hint = (
        f"Torch's DataLoader is required for MaxEnt-GRPO {context}. "
        "Install torch first."
    )

    class DataLoader:  # pragma: no cover - import-time fallback
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(hint)

    try:
        torch_data = _require_dependency("torch.utils.data", hint)
    except ImportError as exc:
        torch_mod = _optional_dependency("torch")
        if torch_mod is None and _allow_stubbed_deps():
            torch_mod = _install_torch_stub(hint)
        if torch_mod is None:
            raise RuntimeError(hint) from exc
        torch_data = ModuleType("torch.utils.data")
        setattr(torch_data, "DataLoader", DataLoader)
        sys.modules["torch.utils.data"] = torch_data
        torch_utils = getattr(torch_mod, "utils", None)
        if torch_utils is None:
            torch_utils = ModuleType("torch.utils")
            sys.modules["torch.utils"] = torch_utils
            setattr(torch_mod, "utils", torch_utils)
        setattr(torch_utils, "data", torch_data)
    dataloader_cls = getattr(torch_data, "DataLoader", None)
    if dataloader_cls is None:
        setattr(torch_data, "DataLoader", DataLoader)
        dataloader_cls = DataLoader
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
        if _allow_stubbed_deps():
            accelerate_mod = _install_accelerate_stub(hint)
        else:
            raise RuntimeError(hint) from exc
    accelerator_cls = getattr(accelerate_mod, "Accelerator", None)
    if accelerator_cls is None:
        if _allow_stubbed_deps():
            accelerate_mod = _install_accelerate_stub(hint)
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

    utils_module = _optional_dependency("trl.models.utils")
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
