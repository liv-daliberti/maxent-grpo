"""Tests for the hydra CLI shim with minimal stubs."""

from __future__ import annotations

import sys


import cli.hydra_cli as hydra_cli


def test_hydra_entrypoints_route_to_hydra(monkeypatch):
    calls = {}

    class _HydraStub:
        def main(self, *args, **kwargs):
            calls["decorator_args"] = (args, kwargs)

            def _decorator(fn):
                calls["wrapped"] = fn

                def _wrapped(cfg=None):
                    calls["cfg"] = cfg
                    return "ok"

                return _wrapped

            return _decorator

    monkeypatch.setattr(hydra_cli, "hydra", _HydraStub())
    cfg_obj = hydra_cli.HydraRootConfig(
        command="generate",
        generate=hydra_cli.GenerateCommand(
            args={"hf_dataset": "org/repo", "model": "m"}
        ),
    )
    result = hydra_cli.hydra_main(cfg=cfg_obj)
    assert result == "ok"

    # Entry wrappers should insert the command argument
    sys.argv = ["prog"]
    marker = {}
    monkeypatch.setattr(
        hydra_cli, "hydra_main", lambda cfg=None: marker.setdefault("called", cfg)
    )
    hydra_cli.generate_entry()
    assert "called" in marker
