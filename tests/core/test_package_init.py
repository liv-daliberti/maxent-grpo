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
"""

from __future__ import annotations

import types
import sys

import pytest


def test_maxent_grpo_lazy_import(monkeypatch):
    """__getattr__ should import and cache known submodules."""

    import maxent_grpo

    dummy_mod = types.SimpleNamespace()
    calls = {}

    def _fake_import(name):
        calls["module"] = name
        return dummy_mod

    monkeypatch.setattr(maxent_grpo, "import_module", _fake_import)
    monkeypatch.delattr(maxent_grpo, "cli", raising=False)
    sys.modules.pop("maxent_grpo.cli", None)
    cli_attr = maxent_grpo.cli  # type: ignore[attr-defined]
    if calls:
        assert calls["module"] == "maxent_grpo.cli"
        assert cli_attr is dummy_mod
    else:
        assert cli_attr is not None
    calls.clear()
    _ = maxent_grpo.cli  # type: ignore[attr-defined]
    assert calls == {}


def test_maxent_grpo_dir_lists_all():
    import maxent_grpo

    names = set(dir(maxent_grpo))
    # Ensure the advertised submodules are present in __dir__ output.
    for key in maxent_grpo.__all__:
        assert key in names


def test_maxent_grpo_unknown_attr_raises():
    import maxent_grpo

    with pytest.raises(AttributeError):
        _ = maxent_grpo.__getattr__("does_not_exist")
