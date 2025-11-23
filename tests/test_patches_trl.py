"""Unit tests for TRL patch helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import patches.trl as p


def test_resolve_port_from_env_prefers_primary(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    monkeypatch.setenv("VLLM_GROUP_PORT", "12345")
    monkeypatch.setenv("MASTER_PORT", "9999")
    port = p._resolve_port_from_env("VLLM_GROUP_PORT", ("MASTER_PORT",))
    assert port == 12345

    # Non-integer should be ignored with a warning
    monkeypatch.setenv("VLLM_GROUP_PORT", "not-int")
    port = p._resolve_port_from_env("VLLM_GROUP_PORT", ("MASTER_PORT",))
    assert port == 9999
    assert any("not a valid integer" in rec.message for rec in caplog.records)


def test_ensure_vllm_group_port_patches_once(monkeypatch):
    # Stub VLLMClient so we can observe kwargs
    captured = {}

    class _Client:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setitem(p.__dict__, "_PATCH_STATE", {"vllm_group_port": False})
    monkeypatch.setitem(os.environ, "VLLM_GROUP_PORT", "1234")
    monkeypatch.setitem(os.environ, "MASTER_PORT", "2222")
    # Inject stub module into sys.modules path used by import
    import sys

    sys.modules["trl"] = SimpleNamespace()
    sys.modules["trl.extras"] = SimpleNamespace()
    sys.modules["trl.extras.vllm_client"] = SimpleNamespace(VLLMClient=_Client)

    p.ensure_vllm_group_port()
    # Instantiate to exercise patched __init__
    _Client()
    # Second call should no-op
    p.ensure_vllm_group_port()

    assert captured.get("kwargs", {}).get("group_port") == 1234
    assert p._PATCH_STATE["vllm_group_port"] is True
