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

Unit tests for distilabel generation pipeline helpers.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
import sys

import pytest

from maxent_grpo.generation.errors import GenerationServiceError, ServiceErrorPayload
from maxent_grpo.pipelines.generation import distilabel as distilabel_mod


def test_generation_config_roundtrip_namespace():
    args = Namespace(hf_dataset="org/repo", model="demo-model")
    cfg = distilabel_mod.DistilabelGenerationConfig.from_namespace(args)
    ns = cfg.to_namespace()
    assert ns.hf_dataset == "org/repo"
    assert ns.model == "demo-model"


def test_run_generation_job_uses_custom_builder(monkeypatch):
    captured = {}

    datasets_mod = SimpleNamespace(load_dataset=lambda *a, **k: {"rows": ["ok"]})
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    def fake_builder(cfg):
        captured["cfg"] = cfg
        return SimpleNamespace(
            run=lambda **kwargs: captured.setdefault("run_kwargs", kwargs)
        )

    cfg = distilabel_mod.DistilabelGenerationConfig(hf_dataset="x", model="y")
    distilabel_mod.run_generation_job(cfg, builder=fake_builder)

    assert isinstance(captured["cfg"], distilabel_mod.DistilabelPipelineConfig)
    assert captured["run_kwargs"]["dataset"] == {"rows": ["ok"]}
    assert captured["run_kwargs"]["dataset_batch_size"] == cfg.input_batch_size * 1000


def test_run_generation_job_accepts_namespace(monkeypatch):
    datasets_mod = SimpleNamespace(load_dataset=lambda *a, **k: {"rows": ["ns"]})
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    captured = {}

    def _builder(cfg):
        captured["cfg"] = cfg
        return SimpleNamespace(run=lambda **kwargs: kwargs)

    args = Namespace(
        hf_dataset="ds",
        hf_dataset_config=None,
        hf_dataset_split="train",
        model="m",
        vllm_server_url="http://v",
        prompt_template=None,
        prompt_column="col",
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=8,
        num_generations=1,
        input_batch_size=2,
        client_replicas=1,
        timeout=1,
        retries=0,
        hf_output_dataset=None,
        private=False,
    )
    distilabel_mod.run_generation_job(args, builder=_builder)
    assert isinstance(captured["cfg"], distilabel_mod.DistilabelPipelineConfig)
    assert captured["cfg"].model == "m"


def test_run_generation_job_plain_object(monkeypatch):
    datasets_mod = SimpleNamespace(load_dataset=lambda *a, **k: {"rows": ["obj"]})
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    captured = {}

    class _Cfg:
        def __init__(self):
            self.hf_dataset = "ds"
            self.hf_dataset_config = None
            self.hf_dataset_split = "test"
            self.model = "m2"
            self.vllm_server_url = "http://v2"
            self.prompt_template = None
            self.prompt_column = "c2"
            self.temperature = 0.2
            self.top_p = 0.8
            self.max_new_tokens = 16
            self.num_generations = 3
            self.input_batch_size = 1
            self.client_replicas = 1
            self.timeout = 5
            self.retries = 0
            self.hf_output_dataset = None
            self.private = False

    def _builder(cfg):
        captured["cfg"] = cfg
        return SimpleNamespace(run=lambda **kwargs: kwargs)

    distilabel_mod.run_generation_job(_Cfg(), builder=_builder)
    assert captured["cfg"].base_url == "http://v2"
    assert captured["cfg"].prompt_column == "c2"


def test_run_generation_job_augments_error_payload(monkeypatch):
    datasets_mod = SimpleNamespace(load_dataset=lambda *a, **k: {"rows": ["ok"]})
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    payload = ServiceErrorPayload(
        service="vllm",
        endpoint="http://host",
        model="m",
        prompt_count=1,
        payload_chars=10,
        payload_size_bytes=20,
        status_code=500,
        attempt=1,
        max_attempts=1,
        exception_type="RuntimeError",
        exception_message="boom",
    )
    error = GenerationServiceError("boom", payload)

    class _Pipeline:
        def run(self, **_kwargs):
            raise error

    def _builder(_cfg):
        return _Pipeline()

    logged = {}

    def _log(_logger, _stage, exc):
        logged["payload"] = exc.payload

    monkeypatch.setattr(
        distilabel_mod, "log_generation_service_error", _log, raising=False
    )
    cfg = distilabel_mod.DistilabelGenerationConfig(
        hf_dataset="hf/train",
        hf_dataset_config="alpha",
        hf_dataset_split="eval",
        model="org/model",
    )
    with pytest.raises(RuntimeError):
        distilabel_mod.run_generation_job(cfg, builder=_builder)
    extra = logged["payload"].extra
    assert extra["dataset"] == "hf/train"
    assert extra["dataset_config"] == "alpha"
    assert extra["dataset_split"] == "eval"
    assert extra["model_id"] == "org/model"
