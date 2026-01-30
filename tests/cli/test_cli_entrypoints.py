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

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import maxent_grpo.cli as cli


def _reload_cli_trl() -> ModuleType:
    """Load training.cli.trl via its source path to avoid heavy package imports."""
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "maxent_grpo"
        / "training"
        / "cli"
        / "trl.py"
    )
    spec = importlib.util.spec_from_file_location("test_maxent_training_cli_trl", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_maxent_entry_module() -> ModuleType:
    """Load the MaxEnt GRPO entry module from its source path for testing."""
    root = Path(__file__).resolve().parents[2]
    path = root / "src" / "maxent_grpo" / "maxent_grpo.py"
    spec = importlib.util.spec_from_file_location("maxent_grpo_entry_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_parse_grpo_args_uses_trl_parser(monkeypatch):
    calls = {}

    class _DummyParser:
        def __init__(self, types_tuple):
            calls["types"] = types_tuple

        def parse_args_and_config(self):
            return ("script", "training", "model")

    trl_stub = ModuleType("trl")
    trl_stub.ModelConfig = type("ModelConfig", (), {})
    trl_stub.TrlParser = lambda classes: _DummyParser(classes)
    monkeypatch.setitem(sys.modules, "trl", trl_stub)

    cli_trl = _reload_cli_trl()
    assert cli_trl.parse_grpo_args() == ("script", "training", "model")
    assert calls["types"][0].__name__ == "GRPOScriptArguments"
    assert calls["types"][1].__name__ == "GRPOConfig"


def test_cli_parse_grpo_args_wrapper(monkeypatch):
    training_cli = ModuleType("maxent_grpo.training.cli")
    training_cli.parse_grpo_args = lambda: ("stub_script", "stub_train", "stub_model")
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.cli", training_cli)
    importlib.reload(cli)
    assert cli.parse_grpo_args() == ("stub_script", "stub_train", "stub_model")


def test_grpo_cli_invokes_main(monkeypatch):
    module = importlib.reload(importlib.import_module("maxent_grpo.grpo"))
    called = {}
    baseline_mod = ModuleType("maxent_grpo.pipelines.training.baseline")
    baseline_mod.run_baseline_training = lambda *args: called.setdefault("args", args)
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.pipelines.training.baseline", baseline_mod
    )
    module.main("s", "t", "m")
    assert called.get("args") == ("s", "t", "m")


def test_maxent_entrypoint_calls_training_runner(monkeypatch):
    module = _load_maxent_entry_module()
    called = {}
    maxent_stub = ModuleType("maxent_grpo.pipelines.training.maxent")
    maxent_stub.run_maxent_training = lambda *args: called.setdefault("args", args)
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.pipelines.training.maxent", maxent_stub
    )
    module.main("s_args", "t_args", "m_args")
    assert called.get("args") == ("s_args", "t_args", "m_args")


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
