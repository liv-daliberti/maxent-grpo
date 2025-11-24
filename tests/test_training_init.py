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

Unit tests for training package initialization shims.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace


def test_training_import_seeds_wandb_stub(monkeypatch):
    monkeypatch.delitem(sys.modules, "training", raising=False)
    monkeypatch.delitem(sys.modules, "maxent_grpo.training", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    import maxent_grpo.training as training

    assert "wandb" in sys.modules
    stub = sys.modules["wandb"]
    assert getattr(getattr(stub, "errors", SimpleNamespace()), "Error", None) is RuntimeError

    # Ensure __dir__ returns a sorted list of exported names
    names = training.__dir__()
    assert isinstance(names, list)
    assert names == sorted(names)