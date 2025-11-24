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

Unit tests for core.hub and core.model helpers.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace
import types

import pytest


def test_push_to_hub_revision_handles_missing_commits(monkeypatch):
    """list_repo_commits failure should fall back to default branch and honor ignore list."""
    calls = {}
    future = object()

    class _Args(SimpleNamespace):
        hub_model_id: str = "org/repo"
        hub_model_revision: str = "branch"
        output_dir: str = "/tmp/out"

    def _list_commits(repo_id):
        assert repo_id == "org/repo"
        raise importlib.import_module("core.hub").HfHubHTTPError("oops")  # type: ignore[arg-type]

    monkeypatch.setattr("core.hub.create_repo", lambda **_: "https://hub/repo")
    monkeypatch.setattr("core.hub.list_repo_commits", _list_commits)
    monkeypatch.setattr(
        "core.hub.create_branch",
        lambda **kwargs: calls.setdefault("create_branch", kwargs.copy()),
    )

    def _upload_folder(**kwargs):
        calls["upload_folder"] = kwargs.copy()
        return future

    monkeypatch.setattr("core.hub.upload_folder", _upload_folder)

    result = importlib.import_module("core.hub").push_to_hub_revision(_Args())
    assert result is future
    assert calls["create_branch"]["revision"] is None
    assert "checkpoint-*" in calls["upload_folder"]["ignore_patterns"]


def test_check_hub_revision_exists_raises_when_readme_present(monkeypatch):
    args = SimpleNamespace(
        hub_model_id="org/repo",
        hub_model_revision="branch",
        push_to_hub_revision=True,
        overwrite_hub_revision=False,
    )
    monkeypatch.setattr("core.hub.repo_exists", lambda *_: True)
    monkeypatch.setattr(
        "core.hub.list_repo_refs",
        lambda *_: SimpleNamespace(branches=[SimpleNamespace(name="branch")]),
    )
    monkeypatch.setattr("core.hub.list_repo_files", lambda **_: ["README.md"])

    with pytest.raises(ValueError):
        importlib.import_module("core.hub").check_hub_revision_exists(args)


def test_get_param_count_from_repo_id_handles_patterns_and_fallback(monkeypatch):
    hub = importlib.import_module("core.hub")
    # Pattern parsing should pick the largest derived count
    assert hub.get_param_count_from_repo_id("org/8x7b-and-42m") == 56_000_000_000

    # When no pattern matches and metadata lookup fails, return -1
    monkeypatch.setattr(
        "core.hub.get_safetensors_metadata",
        lambda *_: (_ for _ in ()).throw(hub.HfHubHTTPError("404")),
    )
    assert hub.get_param_count_from_repo_id("org/unknown-model") == -1


def test_get_gpu_count_for_vllm_decrements_until_divisible(monkeypatch):
    class _Cfg:
        num_attention_heads = 30

    monkeypatch.setattr(
        "core.hub.AutoConfig",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _Cfg()),
    )

    result = importlib.import_module("core.hub").get_gpu_count_for_vllm(
        model_name="org/repo",
        revision="main",
        num_gpus=8,
    )
    # 30 % 8 != 0 and 64 % 8 == 0, so it should decrement down to 2
    assert result == 2


def test_get_model_resolves_dtype_and_enables_gradient_checkpointing(monkeypatch):
    import maxent_grpo.core.model as model

    captured = {}

    class _DummyModel:
        def __init__(self):
            self.config = SimpleNamespace()
            self.gc_calls = []

        def gradient_checkpointing_enable(self, **kwargs):
            self.gc_calls.append(kwargs)

    def _from_pretrained(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return _DummyModel()

    monkeypatch.setattr(model, "get_quantization_config", lambda *_: {"q": 1})
    monkeypatch.setattr(model, "get_kbit_device_map", lambda *_: {"layer": 0})
    monkeypatch.setattr(
        model, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=_from_pretrained)
    )

    args = SimpleNamespace(
        model_name_or_path="demo/model",
        model_revision="rev",
        trust_remote_code=False,
        attn_implementation="sdpa",
        torch_dtype="float16",
    )
    train_args = SimpleNamespace(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    mdl = model.get_model(args, train_args)
    assert captured["name"] == "demo/model"
    assert captured["kwargs"]["device_map"] == {"layer": 0}
    assert captured["kwargs"]["quantization_config"] == {"q": 1}
    assert captured["kwargs"]["torch_dtype"] == getattr(
        model.torch, "float16", "float16"
    )
    assert mdl.gc_calls == [train_args.gradient_checkpointing_kwargs]


def test_get_tokenizer_fallback_sets_chat_template(monkeypatch):
    import maxent_grpo.core.model as model

    class _Err:
        @staticmethod
        def from_pretrained(*_a, **_k):
            raise OSError("offline")

    monkeypatch.setattr(model, "AutoTokenizer", _Err)
    args = SimpleNamespace(
        model_name_or_path="demo", model_revision=None, trust_remote_code=False
    )
    train_args = SimpleNamespace(chat_template="You are helpful.")

    tokenizer = model.get_tokenizer(args, train_args)
    assert tokenizer.chat_template == "You are helpful."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=True, add_generation_prompt=True
    )
    assert isinstance(rendered, list)
    assert rendered  # non-empty bytes


def test_model_import_fallback_without_transformers(monkeypatch):
    """Reload core.model with transformers/trl missing to exercise fallback classes."""
    import maxent_grpo.core.model as model

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("transformers") or name.startswith("trl"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    fallback_mod = importlib.reload(model)

    tok = fallback_mod.PreTrainedTokenizer()
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert "user" in rendered
    assert isinstance(fallback_mod.get_quantization_config(None), type(None))

    # Restore normal module state for subsequent tests
    monkeypatch.setattr(builtins, "__import__", real_import, raising=False)
    importlib.reload(model)


def test_hub_commitinfo_fallback(monkeypatch):
    """Reload core.hub with a stubbed huggingface_hub lacking CommitInfo."""
    import maxent_grpo.core.hub as hub

    hf_stub = types.ModuleType("huggingface_hub")
    hf_stub.create_branch = lambda **_kwargs: None
    hf_stub.create_repo = lambda **_kwargs: None
    hf_stub.get_safetensors_metadata = lambda *_args, **_kwargs: {}
    hf_stub.list_repo_commits = lambda *_args, **_kwargs: []
    hf_stub.list_repo_files = lambda **_kwargs: []
    hf_stub.list_repo_refs = lambda *_args, **_kwargs: SimpleNamespace(branches=[])
    hf_stub.repo_exists = lambda *_args, **_kwargs: False
    hf_stub.upload_folder = lambda **_kwargs: None

    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_stub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", None)
    reloaded = importlib.reload(hub)

    from typing import Any

    assert reloaded.CommitInfo is Any
    assert issubclass(reloaded.HfHubHTTPError, Exception)