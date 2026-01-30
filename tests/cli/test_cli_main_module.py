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

Tests for the module entrypoint under maxent_grpo.cli.__main__.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import textwrap

import pytest


def test_cli_main_invokes_hydra_entry(monkeypatch):
    called = {}

    def _hydra_entry():
        called["ran"] = True

    import maxent_grpo.cli.__main__ as cli_main

    monkeypatch.setattr(cli_main, "hydra_entry", _hydra_entry)
    cli_main.main()
    assert called.get("ran") is True


def test_maxent_cli_round_trip(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    var_dir = tmp_path / "var"
    recipe_path = project_root / "tests/fixtures/recipes/maxent_smoke.yaml"
    usercustomize_path = shim_dir / "usercustomize.py"
    usercustomize_path.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys
            from pathlib import Path
            from types import ModuleType, SimpleNamespace
            import importlib

            ROOT = Path({root!r})
            SRC = ROOT / "src"
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            importlib.import_module("ops.sitecustomize")
            VAR_DIR = Path(os.environ.get("VAR_DIR", ROOT / "var"))
            LOG_DIR = VAR_DIR / "artifacts" / "logs"
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            import maxent_grpo.pipelines.training.maxent as _maxent_mod

            def _fake_run_maxent_training(_script_args, _training_args, _model_args):
                payload = {{"train/kl": 0.05, "train/beta": 0.04, "train/tau": 0.3}}
                (LOG_DIR / "roundtrip.json").write_text(json.dumps(payload))
                return "ok"

            _maxent_mod.run_maxent_training = _fake_run_maxent_training

            stub_cli = ModuleType("maxent_grpo.training.cli")
            def _parse(*_args, **_kwargs):
                return SimpleNamespace(), SimpleNamespace(output_dir=str(VAR_DIR)), SimpleNamespace()

            stub_cli.parse_grpo_args = _parse
            sys.modules["maxent_grpo.training.cli"] = stub_cli
            """
        ).format(root=str(project_root))
    )
    env = os.environ.copy()
    env["VAR_DIR"] = str(var_dir)
    env["GRPO_RECIPE"] = str(recipe_path)
    env.pop("PYTEST_CURRENT_TEST", None)
    existing = env.get("PYTHONPATH", "")
    src_path = project_root / "src"
    env["PYTHONPATH"] = ":".join(
        [str(shim_dir), str(src_path), existing] if existing else [str(shim_dir), str(src_path)]
    )
    cmd = [sys.executable, "-m", "maxent_grpo.maxent_grpo"]
    result = subprocess.run(cmd, cwd=str(project_root), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    log_path = var_dir / "artifacts" / "logs" / "roundtrip.json"
    assert log_path.exists(), f"stdout={result.stdout} stderr={result.stderr}"
    payload = json.loads(log_path.read_text())
    assert payload["train/kl"] == pytest.approx(0.05)
    assert payload["train/beta"] == pytest.approx(0.04)
    assert payload["train/tau"] == pytest.approx(0.3)
