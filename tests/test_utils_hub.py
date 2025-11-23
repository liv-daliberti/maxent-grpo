"""
Unit tests for Hugging Face Hub helpers in core.hub.
"""

from __future__ import annotations

from concurrent.futures import Future
import sys
import types
from types import SimpleNamespace

import pytest

transformers_stub = types.ModuleType("transformers")


class _AutoConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return types.SimpleNamespace(num_attention_heads=8)


transformers_stub.AutoConfig = _AutoConfig
sys.modules.setdefault("transformers", transformers_stub)

import core.hub as hub  # noqa: E402


def test_push_to_hub_revision_calls_hub_apis(monkeypatch):
    calls = {}

    def record(name):
        def _rec(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            if name == "list_repo_commits":
                return [SimpleNamespace(commit_id="abc123")]
            if name == "upload_folder":
                fut: Future[str] = Future()
                fut.set_result("ok")
                return fut
            return "repo-url"

        return _rec

    monkeypatch.setattr(hub, "create_repo", record("create_repo"))
    monkeypatch.setattr(hub, "list_repo_commits", record("list_repo_commits"))
    monkeypatch.setattr(hub, "create_branch", record("create_branch"))
    monkeypatch.setattr(hub, "upload_folder", record("upload_folder"))

    training_args = SimpleNamespace(
        hub_model_id="org/model",
        hub_model_revision="dev",
        output_dir="/tmp/out",
    )
    fut = hub.push_to_hub_revision(training_args, extra_ignore_patterns=["*.bin"])
    assert fut.result() == "ok"
    assert calls["create_repo"]
    assert calls["create_branch"][0][0] == ()
    _, kwargs = calls["upload_folder"][0]
    assert "*.bin" in kwargs["ignore_patterns"]


def test_push_to_hub_revision_requires_model_id():
    with pytest.raises(ValueError):
        hub.push_to_hub_revision(SimpleNamespace(hub_model_id=None))


def test_check_hub_revision_blocks_without_overwrite(monkeypatch):
    monkeypatch.setattr(hub, "repo_exists", lambda *_: True)
    monkeypatch.setattr(
        hub,
        "list_repo_refs",
        lambda *_: SimpleNamespace(branches=[SimpleNamespace(name="dev")]),
    )
    monkeypatch.setattr(
        hub,
        "list_repo_files",
        lambda **_kwargs: ["README.md"],
    )
    training_args = SimpleNamespace(
        hub_model_id="org/model",
        hub_model_revision="dev",
        push_to_hub_revision=True,
        overwrite_hub_revision=False,
    )
    with pytest.raises(ValueError):
        hub.check_hub_revision_exists(training_args)


def test_check_hub_revision_allows_when_overwrite(monkeypatch):
    monkeypatch.setattr(hub, "repo_exists", lambda *_: True)
    monkeypatch.setattr(
        hub,
        "list_repo_refs",
        lambda *_: SimpleNamespace(branches=[SimpleNamespace(name="dev")]),
    )
    monkeypatch.setattr(
        hub,
        "list_repo_files",
        lambda **_kwargs: ["README.md"],
    )
    training_args = SimpleNamespace(
        hub_model_id="org/model",
        hub_model_revision="dev",
        push_to_hub_revision=True,
        overwrite_hub_revision=True,
    )
    hub.check_hub_revision_exists(training_args)
