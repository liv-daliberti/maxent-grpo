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

import sys
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace


class FakeTensor:
    def __init__(self, data, requires_grad: bool = True):
        if isinstance(data, FakeTensor):
            data = data.as_lists()
        self._data = data
        self.requires_grad = requires_grad
        self.device = "cpu"
        self.dtype = "fp32"

    def detach(self):
        return self

    def clone(self):
        copied = (
            [row[:] for row in self._data]
            if self._data and isinstance(self._data[0], list)
            else list(self._data)
        )
        return self.__class__(copied, requires_grad=self.requires_grad)

    def to(self, **_kwargs):
        return self

    def numel(self) -> int:
        if not self._data:
            return 0
        if isinstance(self._data[0], list):
            return sum(len(row) for row in self._data)
        return len(self._data)

    @property
    def ndim(self) -> int:
        if not self._data:
            return 1
        return 2 if isinstance(self._data[0], list) else 1

    @property
    def shape(self):
        if self.ndim == 2:
            return (len(self._data), len(self._data[0]) if self._data else 0)
        return (len(self._data),)

    def as_lists(self):
        return [row[:] for row in self._data] if self.ndim == 2 else list(self._data)


class FakeParameter(FakeTensor):
    pass


class FakeEmbedding:
    def __init__(self, matrix):
        self.weight = FakeParameter(matrix)


class FakeLM:
    def __init__(self, matrix):
        self.embed_tokens = FakeEmbedding(matrix)
        self.lm_head = SimpleNamespace(weight=self.embed_tokens.weight)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _torch_stub():
    class _Device:
        def __init__(self, device="cpu"):
            self.type = str(device)

        def __repr__(self):
            return f"device('{self.type}')"

    class _Tensor:
        __array_priority__ = 100.0

        def __init__(self, data, dtype=None):
            self.arr = np.array(data, dtype=dtype)

        # Basic container/utility helpers
        @property
        def shape(self):
            return self.arr.shape

        @property
        def dtype(self):
            return self.arr.dtype

        @property
        def device(self):
            return _Device("cpu")

        def size(self, dim=None):
            return self.arr.size if dim is None else self.arr.shape[dim]

        def numel(self):
            return int(self.arr.size)

        def item(self):
            return self.arr.item()

        def to(self, device=None, dtype=None, non_blocking=False):
            return _Tensor(self.arr.astype(dtype) if dtype is not None else self.arr)

        def cpu(self):
            return self

        def detach(self):
            return self

        def any(self):
            return bool(np.any(self.arr))

        def clone(self):
            return _Tensor(self.arr.copy())

        def float(self):
            return _Tensor(self.arr.astype(np.float32))

        def long(self):
            return _Tensor(self.arr.astype(np.int64))

        # Tensor math
        def sum(self, dim=None):
            return _Tensor(np.sum(self.arr, axis=dim))

        def mean(self, dim=None):
            return _Tensor(np.mean(self.arr, axis=dim))

        def min(self):
            return _Tensor(np.min(self.arr))

        def max(self):
            return _Tensor(np.max(self.arr))

        def clamp(self, min=None, max=None):
            lo = min if min is not None else None
            hi = max if max is not None else None
            arr = self.arr
            if lo is None and hi is None:
                return _Tensor(arr)
            return _Tensor(
                np.clip(
                    arr,
                    lo if lo is not None else arr.min(),
                    hi if hi is not None else arr.max(),
                )
            )

        def unsqueeze(self, dim):
            return _Tensor(np.expand_dims(self.arr, axis=dim))

        def squeeze(self, dim=None):
            return _Tensor(
                np.squeeze(self.arr, axis=dim)
                if dim is not None
                else np.squeeze(self.arr)
            )

        def reshape(self, *shape):
            return _Tensor(self.arr.reshape(*shape))

        def gather(self, dim, index):
            idx = index.arr if isinstance(index, _Tensor) else index
            result = np.take_along_axis(self.arr, idx, axis=dim)
            return _Tensor(result)

        # Comparisons/logic
        def ne(self, other):
            return _Tensor(self.arr != other)

        def eq(self, other):
            return _Tensor(self.arr == other)

        def ge(self, other):
            return _Tensor(self.arr >= other)

        def __eq__(self, other):
            return self.eq(other)

        def __ne__(self, other):
            return self.ne(other)

        def __ge__(self, other):
            return self.ge(other)

        # Reductions returning Python
        def __iter__(self):
            for item in self.arr:
                yield item

        def __array__(self, dtype=None):
            return np.array(self.arr, dtype=dtype)

        def tolist(self):
            return self.arr.tolist()

        def __len__(self):
            return len(self.arr)

        def __getitem__(self, key):
            return _Tensor(self.arr[key])

        def __setitem__(self, key, value):
            self.arr[key] = value.arr if isinstance(value, _Tensor) else value

        # Binary ops
        def __add__(self, other):
            return _Tensor(
                self.arr + (other.arr if isinstance(other, _Tensor) else other)
            )

        def __sub__(self, other):
            return _Tensor(
                self.arr - (other.arr if isinstance(other, _Tensor) else other)
            )

        def __mul__(self, other):
            return _Tensor(
                self.arr * (other.arr if isinstance(other, _Tensor) else other)
            )

        def __truediv__(self, other):
            return _Tensor(
                self.arr / (other.arr if isinstance(other, _Tensor) else other)
            )

        def __gt__(self, other):
            return _Tensor(
                self.arr > (other.arr if isinstance(other, _Tensor) else other)
            )

        def __lt__(self, other):
            return _Tensor(
                self.arr < (other.arr if isinstance(other, _Tensor) else other)
            )

        def __neg__(self):
            return _Tensor(-self.arr)

    def _tensor(data, dtype=None, device=None):
        return _Tensor(data, dtype=dtype)

    def _ones_like(t, dtype=None):
        return _Tensor(
            np.ones_like(
                t.arr if hasattr(t, "arr") else t,
                dtype=dtype if dtype is not None else None,
            )
        )

    def _zeros_like(t, dtype=None):
        return _Tensor(
            np.zeros_like(
                t.arr if hasattr(t, "arr") else t,
                dtype=dtype if dtype is not None else None,
            )
        )

    def _zeros(shape, dtype=None):
        return _Tensor(np.zeros(shape, dtype=dtype))

    def _ones(shape, dtype=None):
        return _Tensor(np.ones(shape, dtype=dtype))

    def _full(shape, fill_value, dtype=None):
        return _Tensor(np.full(shape, fill_value, dtype=dtype))

    def _empty(shape, dtype=None):
        return _Tensor(np.empty(shape, dtype=dtype))

    def _arange(end, dtype=None):
        return _Tensor(np.arange(end, dtype=dtype))

    def _cat(tensors, dim=0):
        arrays = [t.arr if isinstance(t, _Tensor) else np.array(t) for t in tensors]
        return _Tensor(np.concatenate(arrays, axis=dim))

    def _all(tensor):
        arr = tensor.arr if isinstance(tensor, _Tensor) else np.array(tensor)
        return bool(np.all(arr))

    def _no_grad():
        return contextmanager(lambda: (yield))()

    def _log_softmax(logits, dim=-1):
        arr = logits.arr if isinstance(logits, _Tensor) else np.array(logits)
        max_val = np.max(arr, axis=dim, keepdims=True)
        exps = np.exp(arr - max_val)
        logsum = np.log(np.sum(exps, axis=dim, keepdims=True))
        return _Tensor(arr - max_val - logsum)

    def _unique(x):
        arr = x.arr if isinstance(x, _Tensor) else np.array(x)
        return _Tensor(np.unique(arr))

    nn_functional = SimpleNamespace(log_softmax=_log_softmax)
    nn_mod = SimpleNamespace(Parameter=FakeParameter, functional=nn_functional)
    data_loader_cls = type("DataLoader", (), {})
    sampler_cls = type("Sampler", (), {})
    data_mod = SimpleNamespace(DataLoader=data_loader_cls, Sampler=sampler_cls)
    utils_mod = SimpleNamespace(data=data_mod)
    autograd_mod = SimpleNamespace(no_grad=lambda: contextmanager(lambda: (yield))())

    return SimpleNamespace(
        nn=nn_mod,
        Tensor=_Tensor,
        tensor=_tensor,
        ones_like=_ones_like,
        zeros_like=_zeros_like,
        zeros=_zeros,
        ones=_ones,
        full=_full,
        empty=_empty,
        arange=_arange,
        cat=_cat,
        all=_all,
        unique=_unique,
        float32=np.float32,
        float64=np.float64,
        long=np.int64,
        int64=np.int64,
        dtype=np.dtype,
        device=lambda x="cpu": _Device(x),
        autograd=autograd_mod,
        autocast=lambda *a, **k: contextmanager(lambda: (yield))(),
        utils=utils_mod,
    )


class FakeAccelerator:
    def __init__(self):
        self.device = "cpu"
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0
        self.gradient_accumulation_steps = 1
        self.sync_gradients = True

    def gather(self, obj):
        return obj

    def gather_object(self, obj):
        return [obj]

    def log(self, metrics, step=None):
        return None

    def wait_for_everyone(self):
        return None

    @contextmanager
    def accumulate(self, _model):
        yield

    def backward(self, _loss):
        return None

    def clip_grad_norm_(self, *_args, **_kwargs):
        return 0.0

    def unwrap_model(self, model):
        return model

    def save_state(self, _path):
        return None

    def load_state(self, _path):
        return None


TORCH_STUB = _torch_stub()
TORCH_STUB.__spec__ = SimpleNamespace()
TORCH_STUB.utils.__spec__ = SimpleNamespace()
TORCH_STUB.utils.data.__spec__ = SimpleNamespace()
TORCH_STUB.nn.__spec__ = SimpleNamespace()
TORCH_STUB.nn.functional.__spec__ = SimpleNamespace()


def _accelerate_stub():
    accel_module = ModuleType("accelerate")
    accel_state = ModuleType("accelerate.state")
    accel_state.DistributedType = SimpleNamespace(DEEPSPEED="deepspeed")
    accel_module.state = accel_state
    accel_module.Accelerator = lambda **_kwargs: FakeAccelerator()
    return accel_module, accel_state


ACCELERATE_MODULE, ACCELERATE_STATE = _accelerate_stub()


class _AutoModel:
    def __init__(self):
        self.config = SimpleNamespace()

    @classmethod
    def from_pretrained(cls, *_a, **_k):
        return cls()


class _AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *_a, **_k):
        obj = SimpleNamespace(chat_template=None, eos_token_id=None, pad_token_id=None)
        obj.add_special_tokens = lambda *_args, **_kwargs: None
        obj.resize_token_embeddings = lambda *_args, **_kwargs: None
        return obj


tf_logging = SimpleNamespace(
    set_verbosity=lambda *_a, **_k: None,
    enable_default_handler=lambda: None,
    enable_explicit_format=lambda: None,
)
tf_utils = SimpleNamespace(logging=tf_logging)
TRANSFORMERS_STUB = SimpleNamespace(
    PreTrainedModel=type("PreTrainedModel", (), {}),
    PreTrainedTokenizer=type("PreTrainedTokenizer", (), {}),
    utils=tf_utils,
    AutoModelForCausalLM=_AutoModel,
    AutoTokenizer=_AutoTokenizer,
)


class _TrlScriptArguments:
    def __init__(self, **_kwargs):
        return


class _TrlGRPOConfig:
    def __init__(self, **_kwargs):
        return


class _TrlModelConfig:
    def __init__(self, **_kwargs):
        return


def _trl_kbit_device_map(*_args, **_kwargs):
    return {}


def _trl_quant_config(*_args, **_kwargs):
    return None


TRL_STUB = ModuleType("trl")
TRL_STUB.ScriptArguments = _TrlScriptArguments
TRL_STUB.GRPOConfig = _TrlGRPOConfig
TRL_STUB.ModelConfig = _TrlModelConfig
TRL_STUB.get_kbit_device_map = _trl_kbit_device_map
TRL_STUB.get_quantization_config = _trl_quant_config


def load_run_setup(monkeypatch):
    """Legacy alias kept for tests that previously imported run_setup."""
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    _install_training_stubs(monkeypatch)
    return None


def _install_training_stubs(monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    monkeypatch.setitem(sys.modules, "torch", TORCH_STUB)
    monkeypatch.setitem(sys.modules, "torch.utils", TORCH_STUB.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", TORCH_STUB.utils.data)
    monkeypatch.setitem(sys.modules, "torch.nn", TORCH_STUB.nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", TORCH_STUB.nn.functional)
    optim_mod = ModuleType("torch.optim")
    optim_mod.Optimizer = type("Optimizer", (), {})
    monkeypatch.setitem(sys.modules, "torch.optim", optim_mod)
    monkeypatch.setitem(sys.modules, "accelerate", ACCELERATE_MODULE)
    monkeypatch.setitem(sys.modules, "accelerate.state", ACCELERATE_STATE)
    monkeypatch.setitem(sys.modules, "transformers", TRANSFORMERS_STUB)
    monkeypatch.setitem(sys.modules, "trl", TRL_STUB)
    return SimpleNamespace(
        torch=TORCH_STUB,
        accelerate=ACCELERATE_MODULE,
        transformers=TRANSFORMERS_STUB,
        trl=TRL_STUB,
    )


def install_training_stubs(monkeypatch):
    """Public shim for installing lightweight training dependency stubs."""
    return _install_training_stubs(monkeypatch)


__all__ = [
    "FakeTensor",
    "FakeParameter",
    "FakeEmbedding",
    "FakeLM",
    "FakeAccelerator",
    "load_run_setup",
    "install_training_stubs",
]
