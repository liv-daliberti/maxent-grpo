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

import cli


def _reload_cli_trl() -> ModuleType:
    """Load training.cli.trl via its source path to avoid heavy package imports."""
    path = Path(__file__).resolve().parents[1] / "src" / "training" / "cli" / "trl.py"
    spec = importlib.util.spec_from_file_location("test_training_cli_trl", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_maxent_entry_module() -> ModuleType:
    """Load src/maxent-grpo.py as an importable module for testing."""
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "maxent-grpo.py"
    spec = importlib.util.spec_from_file_location("maxent_grpo_entry", path)
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
    training_cli = ModuleType("training.cli")
    training_cli.parse_grpo_args = lambda: ("stub_script", "stub_train", "stub_model")
    monkeypatch.setitem(sys.modules, "training.cli", training_cli)
    importlib.reload(cli)
    assert cli.parse_grpo_args() == ("stub_script", "stub_train", "stub_model")


def test_maxent_entrypoint_calls_training_runner(monkeypatch):
    training_pkg = ModuleType("training")
    training_pkg.run_maxent_grpo = lambda *_args: None
    training_cli_mod = ModuleType("training.cli")
    training_cli_mod.parse_grpo_args = lambda: ("s_args", "t_args", "m_args")
    setattr(training_pkg, "cli", training_cli_mod)
    monkeypatch.setitem(sys.modules, "training", training_pkg)
    monkeypatch.setitem(sys.modules, "training.cli", training_cli_mod)
    module = _load_maxent_entry_module()
    captured = {}
    monkeypatch.setattr(
        module, "parse_grpo_args", lambda: ("s_args", "t_args", "m_args")
    )
    monkeypatch.setattr(
        module, "run_maxent_grpo", lambda *args: captured.setdefault("args", args)
    )
    module.main()
    assert captured["args"] == ("s_args", "t_args", "m_args")


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
