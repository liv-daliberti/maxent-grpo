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

import pytest


def _stub_hydra_module():
    """Return a minimal Hydra stub that behaves like a ModuleType."""
    hydra_mod = types.ModuleType("hydra")

    def main(version_base=None, config_name=None):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    hydra_mod.main = main  # type: ignore[attr-defined]
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

    trl_mod.ModelConfig = _ModelConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trl", trl_mod)
    return _ModelConfig


@pytest.fixture(autouse=True)
def _reset_sysargv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])


def test_build_grpo_configs_recipe(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    _stub_trl(monkeypatch)
    stub_result = ("script", "train", "model")
    monkeypatch.setattr(
        hydra_cli, "load_grpo_recipe", lambda recipe, model_config_cls: stub_result
    )
    cmd = hydra_cli.BaselineCommand(recipe="demo")
    assert hydra_cli._build_grpo_configs(cmd) is stub_result


def test_build_grpo_configs_without_recipe(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    class _ModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules, "trl", types.SimpleNamespace(ModelConfig=_ModelConfig)
    )
    cmd = hydra_cli.BaselineCommand(
        script={"reward_funcs": ["x"], "dataset_name": "demo"},
        training={},
        model={"a": 1},
    )
    script_args, train_args, model_cfg = hydra_cli._build_grpo_configs(cmd)
    assert script_args.reward_funcs  # defaults from GRPOScriptArguments applied
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
        sys.modules, "maxent_grpo.pipelines.training.baseline", baseline_mod
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
        "maxent_grpo.pipelines.training.maxent.run_maxent_training",
        lambda *args: calls.setdefault("maxent", args),
    )
    monkeypatch.setattr(hydra_cli, "_build_grpo_configs", lambda _cmd: ("s", "t", "m"))

    cfg = hydra_cli.HydraRootConfig(command="train-maxent")
    hydra_cli.hydra_main(cfg)
    assert calls["maxent"] == ("s", "t", "m")


def test_hydra_main_wraps_not_implemented(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    _stub_trl(monkeypatch)
    monkeypatch.setattr(hydra_cli, "_build_grpo_configs", lambda _cmd: ("s", "t", "m"))
    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.maxent.run_maxent_training",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    cfg = hydra_cli.HydraRootConfig(command="train-maxent")
    with pytest.raises(RuntimeError, match="boom"):
        hydra_cli.hydra_main(cfg)


def test_hydra_main_generate_test_mode(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "hydra-main-generate")
    monkeypatch.delenv("MAXENT_ALLOW_HYDRA_EXEC", raising=False)
    called = {}
    monkeypatch.setattr(
        hydra_cli,
        "DistilabelGenerationConfig",
        lambda **kwargs: called.setdefault("cfg", kwargs),
    )
    monkeypatch.setattr(
        hydra_cli,
        "run_generation_job",
        lambda *_args, **_kwargs: called.setdefault("ran", True),
    )
    cfg = hydra_cli.HydraRootConfig(
        command="generate", generate=hydra_cli.GenerateCommand(args={"foo": "bar"})
    )
    result = hydra_cli.hydra_main(cfg)
    assert result == "ok"
    assert called["cfg"] == {"foo": "bar"}
    assert "ran" not in called  # short-circuited in test mode


def test_hydra_main_command_override_from_argv(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "override")
    monkeypatch.delenv("MAXENT_ALLOW_HYDRA_EXEC", raising=False)
    called = {}
    monkeypatch.setattr(
        hydra_cli,
        "DistilabelGenerationConfig",
        lambda **kwargs: called.setdefault("cfg", kwargs),
    )
    monkeypatch.setattr(
        hydra_cli,
        "run_generation_job",
        lambda *_args, **_kwargs: called.setdefault("ran", True),
    )
    sys.argv = ["prog", "command=generate"]
    cfg = {"generate": {"args": {"foo": "bar"}}}
    result = hydra_cli.hydra_main(cfg)
    assert result == "ok"
    assert called["cfg"] == {"args": {"foo": "bar"}}
    assert "ran" not in called


def test_hydra_main_inference_validates(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "hydra-main-inference")
    monkeypatch.delenv("MAXENT_ALLOW_HYDRA_EXEC", raising=False)
    monkeypatch.setattr(
        hydra_cli, "InferenceModelSpec", lambda **kwargs: ("spec", kwargs)
    )
    monkeypatch.setattr(
        hydra_cli, "Math500EvalConfig", lambda **kwargs: ("eval", kwargs)
    )
    called = {}
    monkeypatch.setattr(
        hydra_cli,
        "run_math500_inference",
        lambda *args, **kwargs: called.setdefault("ran", (args, kwargs)),
    )
    cfg = hydra_cli.HydraRootConfig(
        command="inference",
        inference=hydra_cli.InferenceCommand(
            models=[{"name": "model-a"}],
            eval={"shots": 1},
            limit=5,
            collect_generations=True,
        ),
    )
    result = hydra_cli.hydra_main(cfg)
    assert result == "ok"
    assert "ran" not in called  # not executed in test mode


def test_hydra_main_inference_requires_models(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    cfg = hydra_cli.HydraRootConfig(command="inference")
    with pytest.raises(ValueError):
        hydra_cli.hydra_main(cfg)


def test_hydra_main_generate_runs_when_allowed(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    monkeypatch.setenv("MAXENT_ALLOW_HYDRA_EXEC", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "generate")
    called = {}

    class _Cfg:
        pass

    monkeypatch.setattr(
        hydra_cli,
        "DistilabelGenerationConfig",
        lambda **kwargs: called.setdefault("cfg", _Cfg()),
    )
    monkeypatch.setattr(
        hydra_cli, "run_generation_job", lambda cfg: called.setdefault("run", cfg)
    )
    cfg = hydra_cli.HydraRootConfig(
        command="generate", generate=hydra_cli.GenerateCommand(args={"foo": "bar"})
    )
    hydra_cli.hydra_main(cfg)
    assert isinstance(called["run"], _Cfg)


def test_hydra_main_inference_executes_when_allowed(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    monkeypatch.setenv("MAXENT_ALLOW_HYDRA_EXEC", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "inference")
    monkeypatch.setattr(
        hydra_cli,
        "InferenceModelSpec",
        lambda **kwargs: SimpleNamespace(payload=kwargs),
    )
    monkeypatch.setattr(
        hydra_cli,
        "Math500EvalConfig",
        lambda **kwargs: SimpleNamespace(payload=kwargs),
    )
    called = {}

    class _Result(SimpleNamespace):
        pass

    def _run_inference(*args, **kwargs):
        called["run"] = (args, kwargs)
        return [_Result(data="ok")]

    monkeypatch.setattr(hydra_cli, "run_math500_inference", _run_inference)
    prints = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: prints.append(args))
    cfg = hydra_cli.HydraRootConfig(
        command="inference",
        inference=hydra_cli.InferenceCommand(
            models=[{"name": "m1"}],
            eval={"shots": 1},
            limit=2,
            collect_generations=False,
        ),
    )
    hydra_cli.hydra_main(cfg)
    assert called["run"][0][0][0].payload["name"] == "m1"
    assert called["run"][1]["eval_cfg"].payload["shots"] == 1
    assert prints, "expected results to be printed"


def test_entrypoints_insert_command(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    calls = []
    monkeypatch.setattr(
        hydra_cli, "_invoke_hydra_cli", lambda: calls.append(list(sys.argv))
    )

    sys.argv = ["prog"]
    hydra_cli.generate_entry()
    assert any(arg.startswith("command=generate") for arg in calls[-1])

    sys.argv = ["prog"]
    hydra_cli.baseline_entry()
    assert any(arg.startswith("command=train-baseline") for arg in calls[-1])

    sys.argv = ["prog"]
    hydra_cli.maxent_entry()
    assert any(arg.startswith("command=train-maxent") for arg in calls[-1])

    sys.argv = ["prog"]
    hydra_cli.inference_entry()
    assert any(arg.startswith("command=inference") for arg in calls[-1])


def test_maybe_insert_command_skips_when_present(monkeypatch):
    from maxent_grpo.cli import hydra_cli

    sys.argv = ["prog", "command=generate"]
    hydra_cli._maybe_insert_command("ignored")
    assert sys.argv[1] == "command=generate"


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
