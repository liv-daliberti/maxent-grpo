"""Legacy stub installer for training module tests."""

from __future__ import annotations

from tests.helpers.run_setup_stubs import install_training_stubs


def _load_run_setup(monkeypatch):
    """Install lightweight training dependency stubs for downstream imports."""
    install_training_stubs(monkeypatch)
    return None


__all__ = ["_load_run_setup"]
