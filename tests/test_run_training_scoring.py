"""Tests for scoring helpers."""

from importlib import import_module, reload
from types import SimpleNamespace
import sys

import pytest

from test_run_setup_reference import _load_run_setup


@pytest.fixture
def scoring_mod(monkeypatch):
    _load_run_setup(monkeypatch)
    torch_stub = sys.modules["torch"]
    torch_stub.long = object()
    class _FakeTensor:
        def __init__(self, dtype):
            self.dtype = dtype

        def long(self):
            self.dtype = torch_stub.long
            return self

    class _Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": _FakeTensor(dtype="float"),
                "attention_mask": _FakeTensor(dtype="float"),
            }

    module = reload(import_module("maxent_helpers.run_training_scoring"))
    module._TEST_FAKE_TOKENIZER = _Tokenizer
    module._TEST_FAKE_TENSOR_DTYPE = torch_stub.long
    return module


def test_tokenize_completions_forces_long_dtype(scoring_mod):
    tensors = scoring_mod._tokenize_completions(
        ["foo"],
        scoring_mod._TEST_FAKE_TOKENIZER(),
        SimpleNamespace(max_completion_len=8),
    )
    assert tensors.ids.dtype == scoring_mod._TEST_FAKE_TENSOR_DTYPE
    assert tensors.mask.dtype == scoring_mod._TEST_FAKE_TENSOR_DTYPE
