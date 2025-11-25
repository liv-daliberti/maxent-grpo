"""Additional unit tests for maxent_grpo.training.scoring edge cases."""

from __future__ import annotations

from types import SimpleNamespace
import types
import sys
import numpy as np

import maxent_grpo.training.scoring as scoring


def test_refresh_torch_import_error_falls_back(monkeypatch):
    """Ensure _refresh_torch uses require_torch when ops.sitecustomize import fails."""

    class _Stub:
        def tensor(self, data, dtype=None):
            return ("tensor", data)

    # Force import failure path
    monkeypatch.setattr(
        "importlib.import_module",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError()),
    )
    monkeypatch.setattr(scoring, "require_torch", lambda _ctx: _Stub())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    torch_mod = scoring._refresh_torch()
    assert hasattr(torch_mod, "tensor")
    assert torch_mod.tensor([1])[0] == "tensor"


def test_size_hint_handles_missing_len_and_shape():
    class _Obj:
        def size(self, *args, **kwargs):
            raise TypeError("bad size")

        @property
        def shape(self):
            return None

        def __len__(self):
            raise TypeError("no len")

    assert scoring._size_hint(_Obj(), dim=0) == 0


def test_to_numpy_array_prefers_data_when_arr_fails(monkeypatch):
    class _Arr:
        def __init__(self):
            self.arr = _BadArr()
            self.data = [9, 8, 7]

    class _BadArr:
        def __array__(self, *_a, **_k):
            raise ValueError("fail")

    out = scoring._to_numpy_array(_Arr())
    assert np.array_equal(out, np.asarray([9, 8, 7]))


def test_build_score_batch_uses_callable_cache(monkeypatch):
    reward_comp = SimpleNamespace(
        pairs=SimpleNamespace(prompts=["p"], completions=["c"])
    )

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, texts, **_kwargs):
            return {
                "input_ids": np.array([[1]]),
                "attention_mask": np.array([[1]]),
            }

    class _BatchCfg(SimpleNamespace):
        score_slice = 0

        def __call__(self, prompt):
            return SimpleNamespace(input_ids=[1], attention_mask=[1])

    batching_cfg = _BatchCfg()
    gen_cfg = SimpleNamespace(max_prompt_len=1, max_completion_len=1)
    sb = scoring.build_score_batch(reward_comp, _Tok(), gen_cfg, batching_cfg)
    assert sb is not None
    assert sb.slice_size == 1


def test_autocast_context_prefers_accelerator():
    class _Ctx:
        pass

    class _Accel:
        def autocast(self):
            return _Ctx()

    ctx = scoring._autocast_context(_Accel(), SimpleNamespace(type="cuda"))
    assert isinstance(ctx, _Ctx)


def test_gather_reference_logprobs_none_passthrough(monkeypatch):
    sb = SimpleNamespace()
    runtime = SimpleNamespace()
    batching = SimpleNamespace()
    monkeypatch.setattr(scoring, "reference_from_model", lambda *_a, **_k: None)
    assert scoring.gather_reference_logprobs(sb, runtime, batching) is None

    fake_tensors = (SimpleNamespace(), SimpleNamespace())
    called = {}
    monkeypatch.setattr(scoring, "reference_from_model", lambda *_a, **_k: fake_tensors)
    monkeypatch.setattr(
        scoring,
        "finalize_reference_stats",
        lambda *args: called.setdefault("args", args),
    )
    scoring.gather_reference_logprobs(sb, runtime, batching)
    assert called["args"] == fake_tensors
