"""Unit tests for inference package re-exports."""

from __future__ import annotations

import importlib

import inference
import pipelines.inference.math500 as math500


def test_inference_reexports_math500_symbols():
    assert inference.InferenceModelSpec is math500.InferenceModelSpec
    assert inference.Math500EvalConfig is math500.Math500EvalConfig
    assert inference.Math500InferenceResult is math500.Math500InferenceResult
    assert inference.run_math500_inference is math500.run_math500_inference
    assert inference.load_math500_dataset is math500.load_math500_dataset
    for name in (
        "InferenceModelSpec",
        "Math500EvalConfig",
        "Math500InferenceResult",
        "run_math500_inference",
        "load_math500_dataset",
    ):
        assert name in inference.__all__


def test_load_math500_dataset_delegates(monkeypatch):
    cfg = math500.Math500EvalConfig(
        dataset_name="demo", dataset_config=None, split="test"
    )
    sentinel = object()

    def _fake_load_dataset(name, config=None, split=None):
        assert name == "demo"
        assert config is None
        assert split == "test"
        return sentinel

    monkeypatch.setattr(math500, "load_dataset", _fake_load_dataset)
    # reload inference to ensure it still points to math500.load_math500_dataset
    importlib.reload(inference)
    assert inference.load_math500_dataset(cfg) is sentinel
