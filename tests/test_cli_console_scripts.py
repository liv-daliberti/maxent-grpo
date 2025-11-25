"""
Smoke tests for console script entrypoints (baseline and maxent).
"""

from __future__ import annotations

import sys


def _call_entry(entry_fn, expected_command, monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = hydra_cli._HydraStub()
    calls = {}
    monkeypatch.setattr(
        hydra_cli,
        "hydra_main",
        lambda cfg=None: calls.setdefault("argv", list(sys.argv)) or "ok",
    )
    monkeypatch.setattr(sys, "argv", ["prog"])
    entry_fn()
    argv = calls.get("argv", [])
    assert any(arg.startswith(f"command={expected_command}") for arg in argv[1:])


def test_console_script_baseline(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    _call_entry(hydra_cli.baseline_entry, "train-baseline", monkeypatch)


def test_console_script_maxent(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    _call_entry(hydra_cli.maxent_entry, "train-maxent", monkeypatch)
