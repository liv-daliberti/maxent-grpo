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

Unit tests for inference package re-exports.
"""

from __future__ import annotations

import importlib

import maxent_grpo.inference as inference
import maxent_grpo.pipelines.inference.inference as math_inference


def test_inference_reexports_math_inference_symbols():
    assert inference.InferenceModelSpec is math_inference.InferenceModelSpec
    assert inference.MathEvalConfig is math_inference.MathEvalConfig
    assert inference.MathInferenceResult is math_inference.MathInferenceResult
    assert inference.run_math_inference is math_inference.run_math_inference
    assert inference.run_math_eval_inference is math_inference.run_math_eval_inference
    assert inference.load_math_dataset is math_inference.load_math_dataset
    assert inference.INFERENCE_DATASETS is math_inference.INFERENCE_DATASETS
    assert inference.list_inference_datasets is math_inference.list_inference_datasets
    assert (
        inference.resolve_inference_dataset is math_inference.resolve_inference_dataset
    )
    for name in (
        "INFERENCE_DATASETS",
        "InferenceModelSpec",
        "MathEvalConfig",
        "MathInferenceResult",
        "list_inference_datasets",
        "run_math_eval_inference",
        "resolve_inference_dataset",
        "run_math_inference",
        "load_math_dataset",
    ):
        assert name in inference.__all__


def test_load_math_dataset_delegates(monkeypatch):
    cfg = math_inference.MathEvalConfig(
        dataset_name="demo", dataset_config=None, split="test"
    )
    sentinel = object()

    def _fake_load_dataset(name, config=None, split=None):
        assert name == "demo"
        assert config is None
        assert split == "test"
        return sentinel

    monkeypatch.setattr(math_inference, "load_dataset", _fake_load_dataset)
    # reload inference to ensure it still points to math_inference.load_math_dataset
    importlib.reload(inference)
    assert inference.load_math_dataset(cfg) is sentinel
