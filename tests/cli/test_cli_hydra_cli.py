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
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments


def _stub_hydra_module():
    """Return a minimal Hydra stub that behaves like a ModuleType."""
    hydra_mod = types.ModuleType("hydra")

    def main(version_base=None, config_name=None):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    hydra_mod.main = main
    return hydra_mod


def _stub_omegaconf():
    class _OmegaConf:
        @staticmethod
        def to_object(cfg):
            return cfg

        @staticmethod
        def to_yaml(cfg):
            return str(cfg)

        @staticmethod
        def create(payload):
            return payload

    return _OmegaConf


def _stub_trl(monkeypatch):
    trl_mod = types.ModuleType("trl")

    class _ModelConfig:
        def __init__(self, **kwargs):
            self.payload = kwargs

    trl_mod.ModelConfig = _ModelConfig
    monkeypatch.setitem(sys.modules, "trl", trl_mod)
    return _ModelConfig


@pytest.fixture(autouse=True)
def _reset_sysargv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])


@pytest.fixture(autouse=True)
def _disable_validation(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    monkeypatch.setattr(hydra_cli, "validate_training_config", lambda *_, **__: None)


def test_build_grpo_configs_recipe(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    _stub_trl(monkeypatch)
    stub_result = ("script", "train", "model")
    monkeypatch.setattr(
        hydra_cli, "load_grpo_recipe", lambda recipe, model_config_cls: stub_result
    )
    cmd = hydra_cli.BaselineCommand(recipe="demo")
    assert hydra_cli._build_grpo_configs(cmd) == stub_result


def test_build_grpo_configs_without_recipe(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    class _ModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules, "trl", types.SimpleNamespace(ModelConfig=_ModelConfig)
    )
    cmd = hydra_cli.BaselineCommand(
        script={"dataset_name": "demo"},
        training={"reward_funcs": ["x"]},
        model={"a": 1},
    )
    script_args, train_args, model_cfg = hydra_cli._build_grpo_configs(cmd)
    assert train_args.reward_funcs == ["x"]
    assert hasattr(train_args, "benchmarks")
    assert model_cfg.kwargs == {"a": 1}


def test_hydra_main_runs_baseline(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    _stub_trl(monkeypatch)
    calls = {}
    baseline_mod = SimpleNamespace(
        run_baseline_training=lambda *args: calls.setdefault("baseline", args)
    )
    monkeypatch.setitem(
        sys.modules, "maxent_grpo.training.baseline", baseline_mod
    )
    monkeypatch.setattr(hydra_cli, "_build_grpo_configs", lambda _cmd: ("s", "t", "m"))

    cfg = hydra_cli.HydraRootConfig(command="train-baseline")
    hydra_cli.hydra_main(cfg)
    assert calls["baseline"] == ("s", "t", "m")


def test_hydra_main_runs_maxent(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    _stub_trl(monkeypatch)
    calls = {}
    monkeypatch.setattr(
        "maxent_grpo.training.baseline.run_baseline_training",
        lambda *args: calls.setdefault("maxent", args),
    )
    monkeypatch.setattr(hydra_cli, "_build_grpo_configs", lambda _cmd: ("s", "t", "m"))

    cfg = hydra_cli.HydraRootConfig(command="train-maxent")
    hydra_cli.hydra_main(cfg)
    assert calls["maxent"] == ("s", "t", "m")


@pytest.mark.parametrize(
    "command,recipe_key",
    [
        ("train-baseline", "baseline"),
        ("train-maxent", "maxent"),
    ],
)
def test_hydra_recipes_route_to_expected_pipeline(
    monkeypatch, command, recipe_key
):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    _stub_trl(monkeypatch)
    recipe_calls: list[str] = []

    def _fake_loader(recipe, model_config_cls):
        recipe_calls.append(recipe)
        cfg = GRPOConfig()
        if recipe == "maxent":
            cfg.train_grpo_objective = False
        else:
            cfg.train_grpo_objective = True
        return (
            GRPOScriptArguments(dataset_name=recipe),
            cfg,
            model_config_cls(),
        )

    monkeypatch.setattr(hydra_cli, "load_grpo_recipe", _fake_loader)

    baseline_calls = {}
    monkeypatch.setattr(
        "maxent_grpo.training.baseline.run_baseline_training",
        lambda *args: baseline_calls.setdefault("args", args),
    )
    cfg = hydra_cli.HydraRootConfig(command=command)
    cfg.baseline.recipe = "baseline"
    cfg.maxent.recipe = "maxent"
    hydra_cli.hydra_main(cfg)
    assert recipe_calls == [recipe_key]
    assert "args" in baseline_calls
    if command == "train-maxent":
        assert baseline_calls["args"][1].train_grpo_objective is False


def test_hydra_main_wraps_not_implemented(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    _stub_trl(monkeypatch)
    monkeypatch.setattr(hydra_cli, "_build_grpo_configs", lambda _cmd: ("s", "t", "m"))
    monkeypatch.setattr(
        "maxent_grpo.training.baseline.run_baseline_training",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    cfg = hydra_cli.HydraRootConfig(command="train-maxent")
    with pytest.raises(RuntimeError, match="boom"):
        hydra_cli.hydra_main(cfg)


def test_hydra_main_command_override_from_argv(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    _stub_trl(monkeypatch)
    calls = {}
    monkeypatch.setattr(
        "maxent_grpo.training.baseline.run_baseline_training",
        lambda *args: calls.setdefault("maxent", args),
    )
    monkeypatch.setattr(hydra_cli, "_build_grpo_configs", lambda _cmd: ("s", "t", "m"))
    sys.argv = ["prog", "command=train-maxent"]
    cfg = {"baseline": {"recipe": "ignored"}}
    hydra_cli.hydra_main(cfg)
    assert calls["maxent"] == ("s", "t", "m")


def test_cli_smoke_runs_recipe_and_logs(monkeypatch, tmp_path):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    _stub_trl(monkeypatch)

    recipe_path = Path("tests/fixtures/recipes/maxent_smoke.yaml")
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("GRPO_RECIPE", str(recipe_path))
    monkeypatch.setenv("VAR_DIR", str(tmp_path))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("MAXENT_ALLOW_HYDRA_EXEC", "1")

    def _fake_run_baseline(script_args, training_args, model_args):
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "cli-smoke.log"
        payload = {
            "train/kl": 0.05,
            "train/beta": 0.04,
            "train/tau": 0.3,
            "train/maxent_objective": 1.0,
        }
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return "ok"

    monkeypatch.setattr(
        "maxent_grpo.training.baseline.run_baseline_training", _fake_run_baseline
    )

    monkeypatch.setattr(sys, "argv", ["prog", "command=train-maxent"])
    hydra_cli.hydra_entry()
    log_path = log_dir / "cli-smoke.log"
    assert log_path.exists()
    data = json.loads(log_path.read_text("utf-8"))
    assert data["train/kl"] == pytest.approx(0.05)
    assert data["train/beta"] == pytest.approx(0.04)
    assert data["train/tau"] == pytest.approx(0.3)
    assert data["train/maxent_objective"] == 1.0


def test_entrypoints_insert_command(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    calls = []
    monkeypatch.setattr(
        hydra_cli, "_invoke_hydra_cli", lambda: calls.append(list(sys.argv))
    )

    sys.argv = ["prog"]
    hydra_cli.baseline_entry()
    assert any(arg.startswith("command=train-baseline") for arg in calls[-1])

    sys.argv = ["prog", "command=train-maxent"]
    hydra_cli.hydra_entry()
    assert any(arg.startswith("command=train-maxent") for arg in calls[-1])

def test_maybe_insert_command_skips_when_present(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    sys.argv = ["prog", "command=train-maxent"]
    hydra_cli._maybe_insert_command("ignored")
    assert sys.argv[1] == "command=train-maxent"


def test_hydra_main_delegates_when_hydra_stub(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = hydra_cli._HydraStub()
    assert hydra_cli.hydra_main(hydra_cli.HydraRootConfig()) is None


def test_hydra_entry_invokes_cli(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    called = {}
    monkeypatch.setattr(
        hydra_cli, "_invoke_hydra_cli", lambda: called.setdefault("invoked", True)
    )
    hydra_cli.hydra_entry()
    assert called["invoked"] is True


def test_invoke_hydra_cli_decorates_when_module(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    sentinel = object()
    monkeypatch.setattr(hydra_cli, "hydra_main", lambda: sentinel)
    result = hydra_cli._invoke_hydra_cli()
    assert result is sentinel


def test_invoke_hydra_cli_calls_hydra_main_when_stub(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = hydra_cli._HydraStub()
    monkeypatch.setattr(hydra_cli, "hydra_main", lambda: "ok")
    assert hydra_cli._invoke_hydra_cli() == "ok"


def test_hydra_main_unsupported_command(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    with pytest.raises(ValueError):
        hydra_cli.hydra_main(hydra_cli.HydraRootConfig(command="unknown"))
