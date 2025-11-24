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
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import os
import numpy as np
from types import ModuleType, SimpleNamespace

_ROOT_DIR = Path(__file__).resolve().parent.parent
_VAR_ROOT = _ROOT_DIR / "var"
_VAR_ROOT.mkdir(parents=True, exist_ok=True)
_PYCACHE_DIR = _VAR_ROOT / "pycache"
_PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(_PYCACHE_DIR))

_SRC_ROOT = _ROOT_DIR / "src"
if _SRC_ROOT.exists():
    src_str = str(_SRC_ROOT)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
# Allow legacy top-level imports (e.g., ``training``) to resolve to the
# current ``maxent_grpo`` package layout without requiring callers to update
# their PYTHONPATH.
_PKG_ROOT = _SRC_ROOT / "maxent_grpo"
if _PKG_ROOT.exists():
    pkg_str = str(_PKG_ROOT)
    if pkg_str not in sys.path:
        sys.path.append(pkg_str)


_ORIG_FIND_SPEC = importlib.util.find_spec


def _maxent_find_spec(name: str, package: str | None = None):
    """Treat lightweight stubs (missing files) as absent for optional deps."""
    if name == "torch":
        module = sys.modules.get("torch")
        if isinstance(module, ModuleType) and getattr(module, "__file__", None) is None:
            return None
        if isinstance(module, ModuleType) and getattr(module, "__spec__", None) is None:
            return None
    try:
        return _ORIG_FIND_SPEC(name, package)
    except ValueError:
        if name == "torch":
            return None
        raise


importlib.util.find_spec = _maxent_find_spec


def _install_transformers_stub() -> None:
    """Register a lightweight transformers stub for tests when missing."""
    if "transformers" in sys.modules:
        return
    if importlib.util.find_spec("transformers") is not None:
        return
    tf_stub = ModuleType("transformers")
    tf_stub.__spec__ = None
    tf_stub.__path__ = []
    tf_stub.set_seed = lambda *_args, **_kwargs: None
    tf_stub.PreTrainedModel = type("PreTrainedModel", (), {})
    tf_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    trainer_utils = ModuleType("transformers.trainer_utils")
    trainer_utils.get_last_checkpoint = lambda *_args, **_kwargs: None
    utils_mod = ModuleType("transformers.utils")
    utils_mod.logging = SimpleNamespace(
        set_verbosity=lambda *args, **kwargs: None,
        enable_default_handler=lambda *args, **kwargs: None,
        enable_explicit_format=lambda *args, **kwargs: None,
    )

    class _AutoConfig:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return SimpleNamespace(num_attention_heads=8)

    tf_stub.AutoConfig = _AutoConfig
    sys.modules["transformers"] = tf_stub
    sys.modules["transformers.trainer_utils"] = trainer_utils
    sys.modules["transformers.utils"] = utils_mod
    tf_stub.trainer_utils = trainer_utils
    tf_stub.utils = utils_mod


_install_transformers_stub()


def _install_accelerate_stub() -> None:
    """Provide a tiny accelerate stub for environments without the package."""
    if "accelerate" in sys.modules:
        return
    spec = None
    try:
        spec = importlib.util.find_spec("accelerate")
    except ValueError:
        spec = None
    if spec is not None:
        return
    accel_mod = ModuleType("accelerate")

    class _Accelerator:
        def __init__(self, **_kwargs):
            self.is_main_process = True
            self.process_index = 0
            self.num_processes = 1
            self.device = "cpu"
            self.gradient_accumulation_steps = 1
            self.sync_gradients = True
            self.gradient_state = SimpleNamespace(
                set_gradient_accumulation_steps=lambda _steps: None
            )

        def clip_grad_norm_(self, *_args, **_kwargs):
            return 0.0

        def gather(self, value):
            return value

        def gather_object(self, value):
            return [value]

        def broadcast_object_list(self, *_args, **_kwargs):
            return None

        def wait_for_everyone(self):
            return None

        def accumulate(self, _model):
            @contextmanager
            def _ctx():
                yield

            return _ctx()

        def backward(self, _loss):
            return None

        def load_state(self, _path):
            return None

        def save_state(self, _path):
            return None

        def unwrap_model(self, model):
            return model

        def prepare(self, *objects):
            if len(objects) == 1:
                return objects[0]
            return objects

        def set_gradient_accumulation_steps(self, steps):
            self.gradient_accumulation_steps = steps
            setter = getattr(
                self.gradient_state, "set_gradient_accumulation_steps", None
            )
            if callable(setter):
                setter(steps)

    accel_mod.Accelerator = _Accelerator
    accel_state = ModuleType("accelerate.state")
    accel_state.DistributedType = SimpleNamespace(DEEPSPEED="deepspeed")
    accel_mod.state = accel_state
    sys.modules.setdefault("accelerate", accel_mod)
    sys.modules.setdefault("accelerate.state", accel_state)


def _install_torch_stub() -> None:
    """Register a lightweight torch stub when the real package is unavailable."""
    try:
        importlib.util.find_spec("torch")
    except ValueError:
        return
    try:  # pragma: no cover - prefer the real library when present
        import torch as _real_torch  # type: ignore

        if _real_torch is not None:
            return
    except ModuleNotFoundError:
        pass
    existing = sys.modules.get("torch")
    if existing is not None and getattr(existing, "tensor", None):
        return

    def _resolve_dtype(dtype: object | None):
        if dtype is None:
            return None
        np_dtype = getattr(dtype, "np_dtype", None)
        if np_dtype is not None:
            return np_dtype
        try:
            return np.dtype(dtype)
        except Exception:
            return None

    class _TorchDType:
        def __init__(self, name: str, np_dtype: np.dtype):
            self.name = name
            self.np_dtype = np_dtype

        def __repr__(self):
            return f"torch.{self.name}"

    torch_float32 = _TorchDType("float32", np.float32)
    torch_int64 = _TorchDType("int64", np.int64)
    torch_bool = _TorchDType("bool", np.bool_)

    def _as_array(value):
        if isinstance(value, TorchTensor):
            return value._arr
        return np.asarray(value)

    class TorchTensor:
        __array_priority__ = 100

        def __init__(self, data, dtype=None):
            np_dtype = _resolve_dtype(dtype)
            self._arr = np.array(data, dtype=np_dtype)
            self.dtype = dtype or self._arr.dtype
            self.device = "cpu"

        def __array__(self, dtype=None):
            if dtype is None:
                return self._arr
            return np.asarray(self._arr, dtype=dtype)

        def __iter__(self):
            return iter(self._arr)

        def __len__(self):
            return len(self._arr)

        @property
        def shape(self):
            return self._arr.shape

        @property
        def ndim(self):
            return self._arr.ndim

        def numel(self):
            return self._arr.size

        def item(self):
            return self._arr.item()

        def clone(self):
            return TorchTensor(self._arr.copy(), dtype=self.dtype)

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, *args, **kwargs):
            dtype = kwargs.get("dtype", None)
            if len(args) >= 2 and dtype is None:
                dtype = args[1]
            if dtype is not None:
                return TorchTensor(self._arr, dtype=dtype)
            return self

        def long(self):
            return TorchTensor(self._arr.astype(np.int64), dtype=torch_int64)

        def float(self):
            return TorchTensor(self._arr.astype(np.float32), dtype=torch_float32)

        def reshape(self, *shape):
            return TorchTensor(self._arr.reshape(*shape), dtype=self.dtype)

        def unsqueeze(self, dim: int):
            return TorchTensor(np.expand_dims(self._arr, axis=dim), dtype=self.dtype)

        def squeeze(self, dim: int | None = None):
            if dim is None:
                return TorchTensor(np.squeeze(self._arr), dtype=self.dtype)
            return TorchTensor(np.squeeze(self._arr, axis=dim), dtype=self.dtype)

        def sum(self, dim: int | None = None):
            return TorchTensor(self._arr.sum(axis=dim))

        def clamp(self, min=None, max=None):
            return TorchTensor(
                np.clip(self._arr, a_min=min, a_max=max), dtype=self.dtype
            )

        def mean(self, dim: int | None = None):
            return TorchTensor(self._arr.mean(axis=dim))

        def min(self, dim: int | None = None):
            if dim is None:
                return TorchTensor(self._arr.min())
            return TorchTensor(self._arr.min(axis=dim))

        def max(self, dim: int | None = None):
            if dim is None:
                return TorchTensor(self._arr.max())
            return TorchTensor(self._arr.max(axis=dim))

        def gather(self, dim: int, index):
            gathered = np.take_along_axis(self._arr, _as_array(index), axis=dim)
            return TorchTensor(gathered, dtype=self.dtype)

        def __getitem__(self, idx):
            return TorchTensor(self._arr[idx], dtype=self.dtype)

        def __setitem__(self, idx, value):
            self._arr[idx] = _as_array(value)

        def __mul__(self, other):
            return TorchTensor(self._arr * _as_array(other), dtype=self.dtype)

        def __add__(self, other):
            return TorchTensor(self._arr + _as_array(other), dtype=self.dtype)

        def __sub__(self, other):
            return TorchTensor(self._arr - _as_array(other), dtype=self.dtype)

        def __eq__(self, other):
            return TorchTensor(self._arr == _as_array(other), dtype=torch_bool)

        def __ne__(self, other):
            return TorchTensor(self._arr != _as_array(other), dtype=torch_bool)

        def __truediv__(self, other):
            return TorchTensor(self._arr / _as_array(other), dtype=self.dtype)

        def __rtruediv__(self, other):
            return TorchTensor(_as_array(other) / self._arr, dtype=self.dtype)

        def __ge__(self, other):
            return TorchTensor(self._arr >= _as_array(other), dtype=torch_bool)

        def __le__(self, other):
            return TorchTensor(self._arr <= _as_array(other), dtype=torch_bool)

        def __invert__(self):
            return TorchTensor(np.logical_not(self._arr), dtype=self.dtype)

        def __repr__(self):
            return f"TorchTensor(shape={self._arr.shape}, dtype={self.dtype})"

        def tolist(self):
            return self._arr.tolist()

        def size(self, dim: int | None = None):
            if dim is None:
                return self._arr.shape
            return self._arr.shape[dim]

        def __float__(self):
            return float(self._arr)

    class TorchDevice:
        def __init__(self, device: str = "cpu"):
            self.type = str(device).split(":")[0]
            self.index = 0

        def __repr__(self):
            return f"torch.device('{self.type}')"

    def tensor(data, dtype=None, device=None):
        return TorchTensor(data, dtype=dtype)

    def arange(end, dtype=None):
        return TorchTensor(np.arange(end, dtype=_resolve_dtype(dtype)), dtype=dtype)

    def ones_like(x, **kwargs):
        dtype = kwargs.get("dtype", getattr(x, "dtype", None))
        return TorchTensor(
            np.ones_like(_as_array(x), dtype=_resolve_dtype(dtype)), dtype=dtype
        )

    def zeros(shape, dtype=None):
        return TorchTensor(np.zeros(shape, dtype=_resolve_dtype(dtype)), dtype=dtype)

    def zeros_like(x, **kwargs):
        dtype = kwargs.get("dtype", getattr(x, "dtype", None))
        return TorchTensor(
            np.zeros_like(_as_array(x), dtype=_resolve_dtype(dtype)), dtype=dtype
        )

    def full(shape, fill_value, dtype=None):
        return TorchTensor(
            np.full(shape, fill_value, dtype=_resolve_dtype(dtype)), dtype=dtype
        )

    def empty(shape, dtype=None):
        return TorchTensor(np.empty(shape, dtype=_resolve_dtype(dtype)), dtype=dtype)

    def cat(tensors, dim=0):
        arrays = [_as_array(t) for t in tensors if t is not None]
        if not arrays:
            return TorchTensor([], dtype=None)
        return TorchTensor(
            np.concatenate(arrays, axis=dim),
            dtype=getattr(tensors[0], "dtype", None),
        )

    def all_fn(x):
        return bool(np.all(_as_array(x)))

    def log_softmax(logits, dim=-1):
        arr = _as_array(logits)
        shifted = arr - arr.max(axis=dim, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.maximum(exps.sum(axis=dim, keepdims=True), 1e-12)
        return TorchTensor(np.log(probs + 1e-12), dtype=getattr(logits, "dtype", None))

    @contextmanager
    def no_grad():
        yield

    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        torch_mod = ModuleType("torch")
    if not getattr(torch_mod, "__spec__", None):
        torch_mod.__spec__ = SimpleNamespace(name="torch", origin="<maxent-torch-stub>")
    torch_mod.__file__ = getattr(torch_mod, "__file__", "<maxent-torch-stub>")
    torch_mod.Tensor = TorchTensor
    torch_mod.float32 = torch_float32
    torch_mod.int64 = torch_int64
    torch_mod.long = torch_int64
    torch_mod.bool = torch_bool
    torch_mod.tensor = tensor
    torch_mod.arange = arange
    torch_mod.ones_like = ones_like
    torch_mod.zeros = zeros
    torch_mod.zeros_like = zeros_like
    torch_mod.full = full
    torch_mod.empty = empty
    torch_mod.cat = cat
    torch_mod.all = all_fn
    torch_mod.device = TorchDevice
    if not hasattr(torch_mod, "autocast"):
        torch_mod.autocast = lambda **_kwargs: nullcontext()
    torch_mod.no_grad = no_grad
    torch_mod.cuda = SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None
    )
    torch_mod.distributed = SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False
    )

    nn_functional = SimpleNamespace(log_softmax=log_softmax)
    nn_functional.__spec__ = SimpleNamespace(
        name="torch.nn.functional", origin="<maxent-torch-stub>"
    )
    nn_mod = SimpleNamespace(
        Module=type("Module", (), {}),
        Parameter=type("Parameter", (), {}),
        functional=nn_functional,
    )
    nn_mod.__spec__ = SimpleNamespace(name="torch.nn", origin="<maxent-torch-stub>")
    torch_mod.nn = nn_mod
    utils_mod = ModuleType("torch.utils")
    utils_mod.__spec__ = SimpleNamespace(
        name="torch.utils", origin="<maxent-torch-stub>"
    )
    data_mod = ModuleType("torch.utils.data")
    data_mod.__spec__ = SimpleNamespace(
        name="torch.utils.data", origin="<maxent-torch-stub>"
    )
    data_mod.DataLoader = type("DataLoader", (), {})
    data_mod.Sampler = type("Sampler", (), {})
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod
    optim_mod = ModuleType("torch.optim")
    optim_mod.__spec__ = SimpleNamespace(
        name="torch.optim", origin="<maxent-torch-stub>"
    )
    optim_mod.Optimizer = type("Optimizer", (), {})
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = nn_functional
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod
    sys.modules["torch.optim"] = optim_mod


_install_accelerate_stub()
_install_torch_stub()
