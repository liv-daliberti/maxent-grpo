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

Tests for TRL patch helpers in patches.trl.
"""

from __future__ import annotations

import types
import sys

import pytest

from maxent_grpo.patches.trl import _PATCH_STATE, _resolve_port_from_env, ensure_vllm_group_port


def test_resolve_port_from_env_prefers_primary(monkeypatch):
    monkeypatch.setenv("PRIMARY_PORT", "1234")
    monkeypatch.setenv("FALLBACK", "9999")
    assert _resolve_port_from_env("PRIMARY_PORT", ("FALLBACK",)) == 1234


def test_resolve_port_from_env_warns_on_invalid(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setenv("PRIMARY_PORT", "not-an-int")
    assert _resolve_port_from_env("PRIMARY_PORT", ("FALLBACK",)) is None
    assert "not a valid integer" in caplog.text


def test_resolve_port_from_env_skips_empty(monkeypatch):
    monkeypatch.delenv("PRIMARY_PORT", raising=False)
    monkeypatch.setenv("FALLBACK", "")
    assert _resolve_port_from_env("PRIMARY_PORT", ("FALLBACK",)) is None


def test_ensure_vllm_group_port_injects_env(monkeypatch):
    _PATCH_STATE["vllm_group_port"] = False

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    trl_stub = types.SimpleNamespace(extras=types.SimpleNamespace(vllm_client=types.SimpleNamespace(VLLMClient=_Client)))
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "trl.extras", trl_stub.extras)
    monkeypatch.setitem(sys.modules, "trl.extras.vllm_client", trl_stub.extras.vllm_client)

    monkeypatch.setenv("VLLM_GROUP_PORT", "4242")
    ensure_vllm_group_port()
    client = trl_stub.extras.vllm_client.VLLMClient()
    assert client.kwargs["group_port"] == 4242


def test_ensure_vllm_group_port_is_idempotent(monkeypatch):
    _PATCH_STATE["vllm_group_port"] = False
    calls = {}

    class _Client:
        def __init__(self, **kwargs):
            calls.setdefault("count", 0)
            calls["count"] += 1

    trl_stub = types.SimpleNamespace(extras=types.SimpleNamespace(vllm_client=types.SimpleNamespace(VLLMClient=_Client)))
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "trl.extras", trl_stub.extras)
    monkeypatch.setitem(sys.modules, "trl.extras.vllm_client", trl_stub.extras.vllm_client)

    ensure_vllm_group_port()
    ensure_vllm_group_port()
    _ = trl_stub.extras.vllm_client.VLLMClient()
    assert calls["count"] == 1


def test_ensure_vllm_group_port_no_override_logs_debug(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    _PATCH_STATE["vllm_group_port"] = False

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    trl_stub = types.SimpleNamespace(extras=types.SimpleNamespace(vllm_client=types.SimpleNamespace(VLLMClient=_Client)))
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "trl.extras", trl_stub.extras)
    monkeypatch.setitem(sys.modules, "trl.extras.vllm_client", trl_stub.extras.vllm_client)
    monkeypatch.delenv("VLLM_GROUP_PORT", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    monkeypatch.delenv("MAIN_PROCESS_PORT", raising=False)

    ensure_vllm_group_port()
    client = trl_stub.extras.vllm_client.VLLMClient()
    assert "VLLMClient will use default" in caplog.text
    assert "group_port" not in client.kwargs


def test_ensure_vllm_group_port_import_error_logs(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    _PATCH_STATE["vllm_group_port"] = False
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "trl.extras.vllm_client":
            raise ImportError("missing trl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    ensure_vllm_group_port()
    assert "TRL is unavailable; skipping" in caplog.text


def test_ensure_vllm_group_port_no_trl_logs_debug(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    _PATCH_STATE["vllm_group_port"] = False
    monkeypatch.setitem(sys.modules, "trl", None)
    pytest.importorskip("patches.trl")
    ensure_vllm_group_port()
    assert "TRL is unavailable; skipping" in caplog.text