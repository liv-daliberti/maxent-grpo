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
import types
from importlib import import_module, reload
from types import SimpleNamespace

import pytest

import maxent_grpo.training.runtime.torch_utils as torch_utils


@pytest.fixture
def run_helpers(monkeypatch, training_stubs):
    """Reload training.run_helpers with lightweight dependency stubs installed."""
    # training_stubs fixture installs torch/accelerate/transformers shims
    import maxent_grpo.training  # noqa: F401

    module = import_module("maxent_grpo.training.run_helpers")
    return reload(module)


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"{messages[0]['content']}|{messages[1]['content']}"


def test_to_prompt_prefers_chat_template(run_helpers):
    example = {"instruction": "Hello", "answer": "42"}
    out = run_helpers._to_prompt(
        example, _Tokenizer(), "instruction", "SYS", char_limit=100
    )
    assert "SYS" in out["prompt"]
    assert out["answer"] == "42"


def test_to_prompt_fallback_when_chat_template_missing(run_helpers):
    class _BrokenTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise AttributeError

    example = {"instruction": "Hello"}
    out = run_helpers._to_prompt(
        example, _BrokenTokenizer(), "instruction", None, char_limit=64
    )
    assert out["prompt"].startswith("USER:")
    assert out["prompt"].endswith("ASSISTANT:")


def test_group_softmax_normalizes_values(monkeypatch, run_helpers):
    values = [0.0, 1.0, 2.0]

    class _Tensor(list):
        def __init__(self, data):
            super().__init__(float(v) for v in data)

        def __truediv__(self, other):
            return _Tensor([a / other for a in self])

        def __sub__(self, other):
            return _Tensor([a - other for a in self])

        def max(self):
            return max(self)

        def __mul__(self, scalar):
            return _Tensor([a * scalar for a in self])

        __rmul__ = __mul__

        def __add__(self, scalar):
            return _Tensor([a + scalar for a in self])

        def sum(self):
            return sum(self)

        def tolist(self):
            return list(self)

    torch_like = SimpleNamespace(
        float32=float,
        tensor=lambda data, dtype=None: _Tensor(data),
        softmax=lambda tensor, dim=0: _Tensor([val / sum(tensor) for val in tensor]),
    )
    original_dep = run_helpers._require_dependency
    monkeypatch.setattr(
        run_helpers,
        "_require_dependency",
        lambda name, hint: torch_like if name == "torch" else original_dep(name, hint),
    )
    probs = run_helpers._group_softmax(values, temperature=1.0, eps=1e-6)
    assert pytest.approx(sum(probs), rel=1e-6) == 1.0
    assert all(p >= 0 for p in probs)


def test_group_softmax_empty_returns_empty(run_helpers):
    assert run_helpers._group_softmax([]) == []


def test_require_torch_builds_stub_when_missing(monkeypatch, run_helpers):
    torch_utils._import_module.cache_clear()
    real_import = torch_utils._import_module

    def _missing(name: str):
        if name == "torch":
            raise ModuleNotFoundError("no torch")
        return real_import(name)

    monkeypatch.setattr(torch_utils, "_import_module", _missing)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    torch_mod = run_helpers.require_torch("ctx")
    assert hasattr(torch_mod, "tensor")
    assert hasattr(torch_mod, "Tensor")


def test_require_torch_patches_missing_attrs(monkeypatch, run_helpers):
    stub_missing = SimpleNamespace(tensor=lambda *_a, **_k: "t")
    stub_patched = SimpleNamespace(
        tensor=lambda *_a, **_k: "t",
        zeros=lambda *_a, **_k: "z",
        full=lambda *_a, **_k: None,
        ones_like=lambda *_a, **_k: None,
    )
    torch_utils._import_module.cache_clear()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    stubs = [stub_missing, stub_patched]

    def _fake_import(name):
        if name == "torch":
            return stubs.pop(0) if stubs else stub_patched
        return import_module(name)

    _fake_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(torch_utils, "_import_module", _fake_import)
    torch_mod = run_helpers.require_torch("ctx2")
    assert hasattr(torch_mod, "zeros")
    zeros_val = torch_mod.zeros((1,))
    assert zeros_val is not None


def test_require_torch_installs_via_bootstrap(monkeypatch, run_helpers):
    torch_utils._import_module.cache_clear()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    installed = SimpleNamespace(
        tensor=lambda *_a, **_k: "t",
        zeros=lambda *_a, **_k: "z",
        full=lambda *_a, **_k: "f",
        ones_like=lambda *_a, **_k: "o",
    )

    def _install_stub():
        sys.modules["torch"] = installed

    bootstrap = SimpleNamespace(_install_torch_stub=_install_stub)
    monkeypatch.setitem(sys.modules, "ops.sitecustomize", bootstrap)

    def _fake_import(name):
        if name in sys.modules:
            return sys.modules[name]
        if name == "torch":
            return SimpleNamespace(
                tensor=lambda *_a, **_k: "missing"
            )  # missing zeros attr
        if name == "ops.sitecustomize":
            return bootstrap
        return import_module(name)

    _fake_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(torch_utils, "_import_module", _fake_import)
    torch_mod = run_helpers.require_torch("bootstrap")
    assert callable(torch_mod.zeros)
    assert torch_mod.zeros((1,)) == "z"


def test_require_torch_reinstalls_when_attrs_missing(monkeypatch, run_helpers):
    torch_utils._import_module.cache_clear()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    installed = SimpleNamespace(
        tensor=lambda *_a, **_k: "t",
        zeros=lambda *_a, **_k: "z",
        full=lambda *_a, **_k: "f",
        ones_like=lambda *_a, **_k: "o",
    )

    def _install_stub():
        sys.modules["torch"] = installed

    bootstrap = SimpleNamespace(_install_torch_stub=_install_stub)
    monkeypatch.setitem(sys.modules, "ops.sitecustomize", bootstrap)

    def _fake_import(name):
        if name == "torch":
            return SimpleNamespace(tensor=lambda *_a, **_k: None)  # missing attrs
        if name == "ops.sitecustomize":
            return bootstrap
        return import_module(name)

    _fake_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(torch_utils, "_import_module", _fake_import)
    torch_mod = run_helpers.require_torch("bootstrap_missing")
    assert torch_mod.full((1,), 1) == "f"


def test_require_torch_builds_stub_when_all_imports_fail(monkeypatch, run_helpers):
    torch_utils._import_module.cache_clear()

    def _missing(name):
        raise ModuleNotFoundError("no torch")

    _missing.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(torch_utils, "_import_module", _missing)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    torch_mod = run_helpers.require_torch("fallback")
    assert hasattr(torch_mod, "Tensor")


def test_build_torch_stub_tensor_ops(run_helpers):
    stub = run_helpers._build_torch_stub()
    tensor = stub.tensor([[0, 0], [0, 0]])
    tensor[(0, slice(0, 2))] = [1, 2]
    assert tensor.tolist()[0] == [1, 2]
    tensor[1] = [3, 4]
    assert tensor[1, 1].tolist() == [4]
    tensor1d = stub.tensor([0, 3])
    clamped = tensor1d.clamp(min=2)
    assert clamped.tolist()[0] == 2
    assert stub.cat([stub.tensor([1]), [2, 3]]).tolist() == [1, 2, 3]
    assert stub.all([[True, True]])


def test_torch_stub_indexing_and_math_edges(run_helpers):
    stub = run_helpers._build_torch_stub()
    t = stub.tensor([])
    assert len(t) == 0
    t[(0, slice(0, 1))] = [5]  # extend row and assign list
    assert t.tolist() == [[5]]
    t[(0, slice(0, 1))] = 9  # scalar slice assignment
    assert t.tolist()[0] == [9]
    with pytest.raises(TypeError):
        t[(slice(0, 1), slice(0, 1))] = [0]  # non-int row key

    t2 = stub.tensor([[1, 2], [3, 4]])
    selected = t2[(slice(0, 2), 1)]
    assert selected.tolist() == [2, 4]
    t1d = stub.tensor([1, 2])
    summed = t2.sum(dim=1)
    assert summed.tolist() == [3, 7]
    assert t1d.sum().tolist() == [3]
    assert stub.tensor([]).item() == []
    nested_added = t2._binary([[10, 20], [30, 40]], lambda a, b: a + b)
    assert nested_added.tolist()[0] == [11, 22]
    assert (t1d + [1, 1]).tolist() == [2, 3]
    assert (t1d - [1, 1]).tolist() == [0, 1]
    assert t1d.__rsub__(5).tolist()[0] == 4
    assert (t1d / 2).tolist() == [0.5, 1.0]
    assert (t1d == [1, 2]).tolist() == [True, True]
    assert (t1d >= [0, 3]).tolist() == [True, False]


def test_torch_stub_setitem_tuple_row_non_int_raises(run_helpers):
    stub = run_helpers._build_torch_stub()
    tensor = stub.tensor([[1, 2]])
    with pytest.raises(TypeError):
        tensor[(object(), slice(0, 1))] = 3


def test_torch_stub_long_and_sum_non_nested_and_full_1d(run_helpers):
    stub = run_helpers._build_torch_stub()
    t = stub.tensor([1, 2])
    assert t.long() is t
    summed = t.sum(dim=0)
    assert summed.tolist() == [3]
    filled = stub.full((3,), 7)
    assert filled.tolist() == [7, 7, 7]


def test_torch_stub_utilities_and_clamp(run_helpers):
    stub = run_helpers._build_torch_stub()
    zeros_2d = stub.zeros((2, 3))
    assert zeros_2d.shape == (2, 3)
    ones_empty = stub.ones_like(None)
    assert ones_empty.tolist() == []
    full_2d = stub.full((2, 2), 7)
    assert full_2d.tolist() == [[7, 7], [7, 7]]
    assert stub.size(None) == 0
    with stub.autocast():
        pass
    clamped_max = stub.tensor([1, 5]).clamp(max=2)
    assert clamped_max.tolist() == [1, 2]


def test_maxent_options_env_overrides(monkeypatch, run_helpers):
    monkeypatch.setenv("MAXENT_TAU", "0.5")
    monkeypatch.setenv("MAXENT_Q_TEMPERATURE", "0.7")
    monkeypatch.setenv("MAXENT_Q_EPS", "0.123")
    monkeypatch.setenv("MAXENT_LENGTH_NORM_REF", "0")
    opts = run_helpers.MaxEntOptions()
    assert opts.tau == pytest.approx(0.5)
    assert opts.q_temperature == pytest.approx(0.7)
    assert opts.q_epsilon == pytest.approx(0.123)
    assert opts.length_normalize_ref is False


def test_truncate_prompt_warns_once(monkeypatch, caplog, run_helpers):
    caplog.set_level("WARNING")
    run_helpers._TRUNC_STATE["warned"] = False
    truncated = run_helpers._truncate_prompt("x" * 10, char_limit=5)
    assert truncated == "xxxxx"
    assert "Prompt length exceeded" in caplog.text

    caplog.clear()
    truncated = run_helpers._truncate_prompt("y" * 10, char_limit=5)
    assert truncated == "yyyyy"
    assert "Prompt length exceeded" not in caplog.text


def test_generation_sampling_config_proxies_vllm_fields(run_helpers):
    vllm_cfg = run_helpers.VLLMClientConfig(
        url="http://vllm",
        rounds_cfg=3,
        retry_sleep=0.5,
        backfill_local=True,
        request_logprobs=True,
        best_of=4,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        top_k=5,
        stop_sequences=["</s>"],
        timeout=30.0,
        max_retries=2,
        backoff=1.5,
        guided_json=None,
        guided_regex=None,
        logit_bias={"0": -2.0},
        request_id_prefix="eval",
        sync_weights=True,
    )
    cfg = run_helpers.GenerationSamplingConfig(
        max_prompt_len=128,
        max_completion_len=64,
        gen_temperature=0.7,
        gen_top_p=0.95,
        use_vllm=True,
        vllm=vllm_cfg,
    )
    assert cfg.vllm_url == "http://vllm"
    assert cfg.vllm_rounds_cfg == 3
    assert cfg.vllm_retry_sleep == 0.5
    assert cfg.vllm_backfill_local is True
    assert cfg.vllm_request_logprobs is True
    assert cfg.vllm_best_of == 4
    assert cfg.vllm_frequency_penalty == pytest.approx(0.2)
    assert cfg.vllm_presence_penalty == pytest.approx(0.1)
    assert cfg.vllm_top_k == 5
    assert cfg.vllm_stop_sequences == ["</s>"]
    assert cfg.vllm_timeout == 30.0
    assert cfg.vllm_max_retries == 2
    assert cfg.vllm_backoff == pytest.approx(1.5)
    assert cfg.vllm_guided_json is None
    assert cfg.vllm_guided_regex is None
    assert cfg.vllm_logit_bias == {"0": -2.0}
    assert cfg.vllm_request_id_prefix == "eval"
    assert cfg.vllm_sync_weights is True


def test_require_dataloader_installs_stub(monkeypatch):
    import maxent_grpo.training.run_helpers as run_helpers

    torch_utils._import_module.cache_clear()
    for mod in ("torch.utils.data", "torch.utils", "torch"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    attempts = {"count": 0}
    real_import = import_module

    def _fake_import(name, *args, **kwargs):
        if name == "torch.utils.data" and attempts["count"] == 0:
            attempts["count"] += 1
            raise ModuleNotFoundError("missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(torch_utils, "_import_module", _fake_import)
    loader_cls = run_helpers.require_dataloader("test_stub")

    assert loader_cls.__name__ == "DataLoader"
    assert "torch.utils.data" in sys.modules


def test_require_dependency_and_optional_dependency(monkeypatch, run_helpers):
    run_helpers._import_module.cache_clear()
    demo_mod = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "demo_mod", demo_mod)
    assert run_helpers._require_dependency("demo_mod", "hint") is demo_mod
    run_helpers._import_module.cache_clear()
    monkeypatch.delitem(sys.modules, "missing_mod_demo", raising=False)
    assert run_helpers._optional_dependency("missing_mod_demo") is None


def test_wandb_error_types_defaults_and_custom_error(monkeypatch, run_helpers):
    run_helpers._wandb_error_types.cache_clear()
    monkeypatch.delitem(sys.modules, "wandb.errors", raising=False)
    assert run_helpers._wandb_error_types() == (RuntimeError, ValueError)

    run_helpers._wandb_error_types.cache_clear()
    errors_mod = types.ModuleType("wandb.errors")

    class CustomError(RuntimeError):
        pass

    errors_mod.Error = CustomError
    monkeypatch.setitem(sys.modules, "wandb.errors", errors_mod)
    err_types = run_helpers._wandb_error_types()
    assert CustomError in err_types
    assert RuntimeError in err_types and ValueError in err_types


def test_wandb_error_types_ignores_non_exception_error(monkeypatch, run_helpers):
    run_helpers._wandb_error_types.cache_clear()
    errors_mod = types.ModuleType("wandb.errors")
    errors_mod.Error = "not-an-exception"
    monkeypatch.setitem(sys.modules, "wandb.errors", errors_mod)
    err_types = run_helpers._wandb_error_types()
    assert RuntimeError in err_types and ValueError in err_types


def test_report_to_contains_handles_variants(run_helpers):
    assert run_helpers._report_to_contains(None, "wandb") is False
    assert run_helpers._report_to_contains("WandB", "wandb") is True
    assert run_helpers._report_to_contains(["mlflow", "wandb"], "WANDB") is True


def test_maybe_create_deepspeed_plugin_returns_none_when_disabled(
    run_helpers, monkeypatch
):
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "false")
    assert run_helpers._maybe_create_deepspeed_plugin() is None


def test_chat_tokenizer_protocol_methods_raise(run_helpers):
    class _Dummy(run_helpers.ChatTokenizer):
        pass

    dummy = _Dummy()
    with pytest.raises(NotImplementedError):
        dummy.apply_chat_template([])
    with pytest.raises(NotImplementedError):
        _ = dummy.eos_token_id
    with pytest.raises(NotImplementedError):
        dummy("x")


def test_prepare_labels_for_ce_masks_prompt_tokens(run_helpers):
    class _Tensor2D:
        def __init__(self, data):
            self.data = [list(row) for row in data]

        def clone(self):
            return _Tensor2D(self.data)

        def __getitem__(self, key):
            if isinstance(key, tuple):
                row_idx, slc = key
                return self.data[row_idx][slc]
            return self.data[key]

        def __setitem__(self, key, value):
            if isinstance(key, tuple):
                row_idx, slc = key
                row = self.data[row_idx]
                if isinstance(slc, slice):
                    start = slc.start or 0
                    stop = slc.stop if slc.stop is not None else len(row)
                    row[start:stop] = [value] * (stop - start)
                    self.data[row_idx] = row
            else:
                self.data[key] = value

    tensor = _Tensor2D([[1, 2, 3], [4, 5]])
    labels = run_helpers._prepare_labels_for_ce(tensor, [2, 1])
    assert labels.data[0][:2] == [-100, -100]
    assert labels.data[1][0] == -100


def test_batch_tokenize_pairs_returns_lengths(run_helpers):
    class _FakeTensor:
        def __init__(self, data):
            self.data = data

        def ne(self, value):
            masked = [[1 if v != value else 0 for v in row] for row in self.data]
            return _FakeTensor(masked)

        def sum(self, dim):
            assert dim == 1
            return _FakeTensor1D([sum(row) for row in self.data])

        def tolist(self):
            return self.data

    class _FakeTensor1D(_FakeTensor):
        def sum(self, dim=None):
            return sum(self.data)

    class _TokenizerStub:
        pad_token_id = 0

        def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
            encoded = [[1] * len(txt) for txt in texts]
            tensor = _FakeTensor(encoded)
            return {"input_ids": tensor, "attention_mask": tensor}

    prompts = ["hi", "ok"]
    completions = [" there", "!"]
    input_ids, attn, prompt_lengths = run_helpers._batch_tokenize_pairs(
        _TokenizerStub(), prompts, completions
    )
    assert isinstance(input_ids, _FakeTensor)
    assert isinstance(attn, _FakeTensor)
    assert prompt_lengths == [len(p) for p in prompts]
