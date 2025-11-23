"""Smoke tests for the TRL vllm_serve patcher in setup.py."""

from __future__ import annotations

import importlib
import importlib.util
import importlib.metadata as importlib_metadata
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_setup_module(monkeypatch, module_name: str = "setup_under_test"):
    """Import setup.py with setuptools.setup stubbed out."""

    monkeypatch.setattr("setuptools.setup", lambda *args, **kwargs: None)
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    spec = importlib.util.spec_from_file_location(module_name, setup_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _install_fake_trl(monkeypatch, target_path: Path) -> None:
    """Stub importlib lookups to point at a fake trl.scripts.vllm_serve."""

    fake_mod = ModuleType("trl.scripts.vllm_serve")
    fake_mod.__file__ = str(target_path)
    real_import = importlib.import_module
    real_version = importlib_metadata.version

    def _fake_import(name: str, *args, **kwargs):
        if name == "trl.scripts.vllm_serve":
            return fake_mod
        return real_import(name, *args, **kwargs)

    def _fake_version(pkg: str):
        if pkg == "trl":
            return "0.18.0"
        return real_version(pkg)

    monkeypatch.setattr("importlib.import_module", _fake_import)
    monkeypatch.setattr("importlib.metadata.version", _fake_version)


def test_trl_patch_idempotent(tmp_path, monkeypatch):
    """Applying the patch twice should only insert the hook once."""

    target = tmp_path / "vllm_serve.py"
    target.write_text(
        "llm = LLM(\n        enforce_eager=script_args.enforce_eager,\n"
        "        dtype=script_args.dtype,\n)\n",
        encoding="utf-8",
    )
    _install_fake_trl(monkeypatch, target)
    setup_mod = _load_setup_module(monkeypatch, module_name="setup_patch_idempotent")

    setup_mod._patch_trl_vllm_serve()
    setup_mod._patch_trl_vllm_serve()

    patched = target.read_text(encoding="utf-8")
    assert patched.count("use_tqdm_on_load=True") == 1


def test_trl_patch_raises_when_anchor_missing(tmp_path, monkeypatch):
    """Fail loudly when expected anchors are absent."""

    target = tmp_path / "vllm_serve.py"
    target.write_text("def noop():\n    return None\n", encoding="utf-8")
    _install_fake_trl(monkeypatch, target)
    setup_mod = _load_setup_module(monkeypatch, module_name="setup_patch_missing_anchor")

    with pytest.raises(RuntimeError) as excinfo:
        setup_mod._patch_trl_vllm_serve()
    assert "Could not find insertion point" in str(excinfo.value)
    assert "trl==" in str(excinfo.value)
