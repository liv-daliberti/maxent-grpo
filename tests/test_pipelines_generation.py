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


from maxent_grpo.pipelines.generation import distilabel as distilabel_mod


def test_generation_config_roundtrip_namespace():
    args = Namespace(hf_dataset="org/repo", model="demo-model")
    cfg = distilabel_mod.DistilabelGenerationConfig.from_namespace(args)
    ns = cfg.to_namespace()
    assert ns.hf_dataset == "org/repo"
    assert ns.model == "demo-model"


def test_run_generation_job_uses_custom_builder(monkeypatch):
    captured = {}

    def fake_builder(cfg):
        captured["cfg"] = cfg
        return SimpleNamespace(
            run=lambda **kwargs: captured.setdefault("run_kwargs", kwargs)
        )

    def fake_cli(args, builder):
        captured["args"] = args
        pipeline = builder(args)
        pipeline.run(dataset="ds", dataset_batch_size=1, use_cache=False)

    monkeypatch.setattr(distilabel_mod, "run_distilabel_cli", fake_cli)

    cfg = distilabel_mod.DistilabelGenerationConfig(hf_dataset="x", model="y")
    distilabel_mod.run_generation_job(cfg, builder=fake_builder)

    assert isinstance(captured["args"], Namespace)
    assert captured["cfg"] == captured["args"]
    assert captured["run_kwargs"]["dataset_batch_size"] == 1