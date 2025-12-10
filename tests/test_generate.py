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

Unit tests for the thin ``maxent_grpo.generate`` CLI wrapper.
"""

from __future__ import annotations

import importlib
from argparse import Namespace
from types import ModuleType
import runpy
import sys


def test_run_cli_proxies_cli_generate(monkeypatch):
    called = {}
    cli_generate = importlib.import_module("maxent_grpo.cli.generate")
    monkeypatch.setattr(cli_generate, "run_cli", lambda ns: called.setdefault("ns", ns))
    generate = importlib.reload(importlib.import_module("maxent_grpo.generate"))

    args = Namespace(model="demo")
    generate.run_cli(args)

    assert called["ns"] is args


def test_main_invokes_typer_app(monkeypatch):
    called = {}
    generate = importlib.reload(importlib.import_module("maxent_grpo.generate"))
    monkeypatch.setattr(
        generate, "generate_cli_app", lambda: called.setdefault("invoked", True)
    )
    generate.main()
    assert called["invoked"] is True


def test_generate_module_executes_main_guard(monkeypatch):
    called = {}
    cli_generate = ModuleType("maxent_grpo.cli.generate")
    cli_generate.app = lambda: called.setdefault("invoked", True)
    cli_generate.build_generate_parser = lambda: None
    cli_generate.run_cli = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "maxent_grpo.cli.generate", cli_generate)

    distilabel_mod = ModuleType("maxent_grpo.pipelines.generation.distilabel")

    class _Cfg:
        @classmethod
        def from_namespace(cls, _ns):
            return cls()

    distilabel_mod.DistilabelGenerationConfig = _Cfg
    distilabel_mod.DistilabelPipelineConfig = object
    distilabel_mod.build_distilabel_pipeline = lambda *_args, **_kwargs: None
    distilabel_mod.run_generation_job = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.pipelines.generation.distilabel", distilabel_mod
    )

    # Ensure the module reloads under __main__ without runpy warnings.
    sys.modules.pop("maxent_grpo.generate", None)

    runpy.run_module("maxent_grpo.generate", run_name="__main__")

    assert called["invoked"] is True


def test_run_generation_job_forwards_args(monkeypatch):
    generate = importlib.reload(importlib.import_module("maxent_grpo.generate"))
    sentinel = object()
    seen = {}

    def _fake_run(cfg, builder):
        seen["cfg"] = cfg
        seen["builder"] = builder
        return sentinel

    monkeypatch.setattr(generate, "_run_generation_job", _fake_run)

    cfg = Namespace(task="demo")
    builder = object()

    result = generate.run_generation_job(cfg, builder)

    assert result is sentinel
    assert seen["cfg"] is cfg
    assert seen["builder"] is builder


def test_run_generation_job_defaults_builder(monkeypatch):
    generate = importlib.reload(importlib.import_module("maxent_grpo.generate"))
    called = {}

    def _fake_run(cfg, builder):
        called["cfg"] = cfg
        called["builder"] = builder
        return "ok"

    monkeypatch.setattr(generate, "_run_generation_job", _fake_run)
    cfg = Namespace(task="demo")

    result = generate.run_generation_job(cfg)

    assert result == "ok"
    assert called["cfg"] is cfg
    assert called["builder"] is None
