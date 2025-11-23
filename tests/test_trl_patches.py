"""
Tests for patches.trl.ensure_vllm_group_port.
"""

from __future__ import annotations

import sys
import types

import patches.trl as patches


def test_ensure_vllm_group_port_uses_env(monkeypatch):
    patched = {}

    class _Client:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def __init_subclass__(cls):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def init_communicator(self):
            return None

    trl_stub = types.ModuleType("trl")
    extras = types.ModuleType("trl.extras")
    vllm_client_mod = types.ModuleType("trl.extras.vllm_client")

    class _VClient:
        def __init__(self, **kwargs):
            patched["kwargs"] = kwargs

        def init_communicator(self):
            return None

    vllm_client_mod.VLLMClient = _VClient
    sys.modules["trl"] = trl_stub
    sys.modules["trl.extras"] = extras
    sys.modules["trl.extras.vllm_client"] = vllm_client_mod
    monkeypatch.setenv("VLLM_GROUP_PORT", "12345")
    patches._PATCH_STATE["vllm_group_port"] = False
    patches.ensure_vllm_group_port()
    client = vllm_client_mod.VLLMClient(base_url="http://localhost")
    assert client is not None
    assert patched["kwargs"]["group_port"] == 12345
