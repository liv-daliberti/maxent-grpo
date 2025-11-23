"""Tests covering reference embedding hydration for the training setup."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from importlib import import_module, reload
from types import ModuleType, SimpleNamespace


class _FakeTensor:
    """Minimal tensor-like object supporting the methods used in hydration."""

    def __init__(self, data, requires_grad: bool = True):
        if isinstance(data, _FakeTensor):
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
        return (
            [row[:] for row in self._data]
            if self.ndim == 2
            else list(self._data)
        )


class _FakeParameter(_FakeTensor):
    pass


class _FakeEmbedding:
    def __init__(self, matrix):
        self.weight = _FakeParameter(matrix)


class _FakeLM:
    """Minimal causal LM skeleton exposing embedding accessors."""

    def __init__(self, matrix):
        self.embed_tokens = _FakeEmbedding(matrix)
        self.lm_head = SimpleNamespace(weight=self.embed_tokens.weight)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _torch_stub():
    """Return a torch-like stub exposing the nn.Parameter entrypoint."""

    nn_functional = SimpleNamespace(log_softmax=lambda *_a, **_k: None)
    nn_mod = SimpleNamespace(Parameter=_FakeParameter, functional=nn_functional)
    tensor_cls = type("Tensor", (), {})
    data_loader_cls = type("DataLoader", (), {})
    data_mod = SimpleNamespace(DataLoader=data_loader_cls)
    utils_mod = SimpleNamespace(data=data_mod)
    return SimpleNamespace(nn=nn_mod, Tensor=tensor_cls, utils=utils_mod)


def _load_run_setup(monkeypatch):
    torch_stub = _torch_stub()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_stub.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_stub.utils.data)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_stub.nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", torch_stub.nn.functional)
    class _FakeAccelerator:
        def __init__(self):
            self.device = "cpu"
            self.is_main_process = True
            self.num_processes = 1
            self.process_index = 0
            self.logged = []
            self.gradient_accumulation_steps = 1
            self.sync_gradients = True

        def gather(self, obj):
            return obj

        def gather_object(self, obj):
            return [obj]

        def log(self, metrics, step=None):
            self.logged.append((metrics, step))

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

    accel_module = ModuleType("accelerate")
    accel_module.Accelerator = lambda **_kwargs: _FakeAccelerator()
    accel_state = ModuleType("accelerate.state")
    accel_state.DistributedType = SimpleNamespace(DEEPSPEED="deepspeed")
    accel_module.state = accel_state
    monkeypatch.setitem(sys.modules, "accelerate", accel_module)
    monkeypatch.setitem(sys.modules, "accelerate.state", accel_state)
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
    tf_stub = SimpleNamespace(
        PreTrainedModel=type("PreTrainedModel", (), {}),
        PreTrainedTokenizer=type("PreTrainedTokenizer", (), {}),
        utils=tf_utils,
        AutoModelForCausalLM=_AutoModel,
        AutoTokenizer=_AutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", tf_stub)
    return reload(import_module("maxent_helpers.run_setup"))


def test_reference_embedding_hydration_rebuilds_flat_weights(monkeypatch):
    """Hydration copies trainable embeddings and re-ties the LM head."""

    run_setup = _load_run_setup(monkeypatch)

    vocab = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    train_model = _FakeLM(vocab)
    ref_model = _FakeLM([[0.0]])  # overwritten immediately
    ref_model.embed_tokens.weight = _FakeParameter([], requires_grad=False)
    ref_model.lm_head.weight = ref_model.embed_tokens.weight

    updated = run_setup._hydrate_reference_embeddings(train_model, ref_model)
    assert updated is True

    ref_weight = ref_model.get_input_embeddings().weight
    train_weight = train_model.get_input_embeddings().weight
    assert ref_weight.shape == train_weight.shape
    assert ref_weight.requires_grad is False
    assert ref_model.get_output_embeddings().weight is ref_weight
    assert ref_weight.as_lists() == train_weight.as_lists()


def test_get_state_dict_fallback(monkeypatch):
    """_get_state_dict_for_reference falls back to the unwrapped model."""

    run_setup = _load_run_setup(monkeypatch)

    class _Model:
        def __init__(self):
            self.module = self

        def state_dict(self):
            return {"foo": 1}

    class _Accel:
        def __init__(self):
            self.device = "cpu"

        def get_state_dict(self, _model):
            return None

        def unwrap_model(self, model):
            return model

    state_dict = run_setup._get_state_dict_for_reference(_Accel(), _Model())
    assert state_dict == {"foo": 1}


def test_get_state_dict_without_accelerate_helper(monkeypatch):
    """When get_state_dict is missing entirely we still gather via state_dict()."""

    run_setup = _load_run_setup(monkeypatch)

    class _Model:
        def __init__(self):
            self.module = self

        def state_dict(self):
            return {"bar": 2}

    class _Accel:
        def __init__(self):
            self.device = "cpu"
            self.is_main_process = True

        def unwrap_model(self, model):
            return model

    state_dict = run_setup._get_state_dict_for_reference(_Accel(), _Model())
    assert state_dict == {"bar": 2}


def test_state_dict_broadcast_for_non_main_rank(monkeypatch):
    """Non-main ranks receive the broadcasted state dict."""

    run_setup = _load_run_setup(monkeypatch)

    class _Model:
        def __init__(self):
            self.module = self

    class _Accel:
        def __init__(self):
            self.device = "cpu"
            self.is_main_process = False

        def broadcast_object_list(self, payload, src=0):
            payload[0] = {"from_main": True}

    state_dict = run_setup._get_state_dict_for_reference(_Accel(), _Model())
    assert state_dict == {"from_main": True}


def test_sync_reference_model_invokes_wait(monkeypatch):
    """_sync_reference_model triggers accelerator.wait_for_everyone when available."""

    run_setup = _load_run_setup(monkeypatch)

    class _Accel:
        def __init__(self):
            self.called = False

        def wait_for_everyone(self):
            self.called = True

    accel = _Accel()
    run_setup._sync_reference_model(accel)
    assert accel.called is True
