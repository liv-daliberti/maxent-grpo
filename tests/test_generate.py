"""Unit tests for the thin ``src/generate`` CLI wrapper."""

from __future__ import annotations

from argparse import Namespace
from types import ModuleType
import runpy
import sys

import generate


def test_run_cli_builds_config_and_executes_job(monkeypatch):
    captured = {}

    class _Cfg:
        def __init__(self, ns):
            self.namespace = ns

        @classmethod
        def from_namespace(cls, ns):
            captured["ns"] = ns
            return cls(ns)

    monkeypatch.setattr(generate, "DistilabelGenerationConfig", _Cfg)
    monkeypatch.setattr(
        generate,
        "run_generation_job",
        lambda cfg, **kwargs: captured.setdefault("cfg", cfg),
    )

    args = Namespace(model="demo")
    generate.run_cli(args)

    assert captured["ns"] is args
    assert isinstance(captured["cfg"], _Cfg)
    assert captured["cfg"].namespace is args


def test_main_invokes_typer_app(monkeypatch):
    called = {}
    monkeypatch.setattr(
        generate, "generate_cli_app", lambda: called.setdefault("invoked", True)
    )
    generate.main()
    assert called["invoked"] is True


def test_generate_module_executes_main_guard(monkeypatch):
    called = {}
    cli_generate = ModuleType("cli.generate")
    cli_generate.app = lambda: called.setdefault("invoked", True)
    cli_generate.build_generate_parser = lambda: None
    monkeypatch.setitem(sys.modules, "cli.generate", cli_generate)

    distilabel_mod = ModuleType("pipelines.generation.distilabel")

    class _Cfg:
        @classmethod
        def from_namespace(cls, _ns):
            return cls()

    distilabel_mod.DistilabelGenerationConfig = _Cfg
    distilabel_mod.DistilabelPipelineConfig = object
    distilabel_mod.build_distilabel_pipeline = lambda *_args, **_kwargs: None
    distilabel_mod.run_generation_job = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "pipelines.generation.distilabel", distilabel_mod)

    runpy.run_module("generate", run_name="__main__")

    assert called["invoked"] is True
