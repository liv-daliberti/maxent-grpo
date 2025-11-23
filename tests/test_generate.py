"""Unit tests for the thin ``src/generate`` CLI wrapper."""

from __future__ import annotations

from argparse import Namespace

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
