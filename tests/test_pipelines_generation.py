"""Unit tests for distilabel generation pipeline helpers."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace


from pipelines.generation import distilabel as distilabel_mod


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
