"""Reusable stubs/fixtures for training.run_setup tests."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from importlib import import_module, reload
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
    nn_functional = SimpleNamespace(log_softmax=lambda *_a, **_k: None)
    nn_mod = SimpleNamespace(Parameter=FakeParameter, functional=nn_functional)
    tensor_cls = type("Tensor", (), {})
    data_loader_cls = type("DataLoader", (), {})
    sampler_cls = type("Sampler", (), {})
    data_mod = SimpleNamespace(DataLoader=data_loader_cls, Sampler=sampler_cls)
    utils_mod = SimpleNamespace(data=data_mod)
    return SimpleNamespace(nn=nn_mod, Tensor=tensor_cls, utils=utils_mod)


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
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    import importlib.util

    package_root = src_path / "training"
    package_init = package_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "training",
        package_init,
        submodule_search_locations=[str(package_root)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "training", module)
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
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return reload(import_module("training.run_setup"))


def build_framework_handles(run_setup_module):
    return run_setup_module.FrameworkHandles(
        torch=TORCH_STUB,
        data_loader_cls=TORCH_STUB.utils.data.DataLoader,
        transformers=TRANSFORMERS_STUB,
        accelerator_cls=lambda **_k: FakeAccelerator(),
    )


__all__ = [
    "FakeTensor",
    "FakeParameter",
    "FakeEmbedding",
    "FakeLM",
    "FakeAccelerator",
    "load_run_setup",
    "build_framework_handles",
]
