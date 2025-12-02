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

import sys
from types import SimpleNamespace


def test_parse_grpo_args_uses_conflict_resolve(monkeypatch):
    """Ensure duplicate CLI flags are resolved instead of erroring."""

    from maxent_grpo.training.cli import trl as cli_trl

    calls = {}

    class DummyParser:
        def __init__(self, dataclass_types, conflict_handler=None):
            calls["dataclass_types"] = tuple(dataclass_types)
            calls["conflict_handler"] = conflict_handler

        def parse_args_and_config(self):
            calls["parsed"] = True
            return ("script", "train", "model")

    dummy_trl = SimpleNamespace(ModelConfig=object, TrlParser=DummyParser)
    monkeypatch.setitem(sys.modules, "trl", dummy_trl)

    result = cli_trl.parse_grpo_args()

    assert calls["conflict_handler"] == "resolve"
    assert calls["dataclass_types"][0].__name__ == "GRPOScriptArguments"
    assert calls["dataclass_types"][1].__name__ == "GRPOConfig"
    assert result == ("script", "train", "model")
