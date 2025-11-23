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
    from src.cli import hydra_cli

    _stub_trl(monkeypatch)
    stub_result = ("script", "train", "model")
    monkeypatch.setattr(
        hydra_cli, "load_grpo_recipe", lambda recipe, model_config_cls: stub_result
    )
    cmd = hydra_cli.BaselineCommand(recipe="demo")
    assert hydra_cli._build_grpo_configs(cmd) is stub_result


def test_hydra_main_runs_baseline(monkeypatch):
    from src.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    _stub_trl(monkeypatch)
    calls = {}
    baseline_mod = SimpleNamespace(
        run_baseline_training=lambda *args: calls.setdefault("baseline", args)
    )
    monkeypatch.setitem(sys.modules, "pipelines.training.baseline", baseline_mod)
    monkeypatch.setattr(hydra_cli, "_build_grpo_configs", lambda _cmd: ("s", "t", "m"))

    cfg = hydra_cli.HydraRootConfig(command="train-baseline")
    hydra_cli.hydra_main(cfg)
    assert calls["baseline"] == ("s", "t", "m")


def test_hydra_main_runs_maxent(monkeypatch):
    from src.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    _stub_trl(monkeypatch)
    calls = {}
    training_mod = SimpleNamespace(
        run_maxent_grpo=lambda *args: calls.setdefault("maxent", args)
    )
    monkeypatch.setitem(sys.modules, "training", training_mod)
    monkeypatch.setattr(hydra_cli, "_build_grpo_configs", lambda _cmd: ("s", "t", "m"))

    cfg = hydra_cli.HydraRootConfig(command="train-maxent")
    hydra_cli.hydra_main(cfg)
    assert calls["maxent"] == ("s", "t", "m")


def test_hydra_main_generate_test_mode(monkeypatch):
    from src.cli import hydra_cli

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


def test_hydra_main_inference_validates(monkeypatch):
    from src.cli import hydra_cli

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
    from src.cli import hydra_cli

    hydra_cli.hydra = _stub_hydra_module()
    hydra_cli.OmegaConf = _stub_omegaconf()
    cfg = hydra_cli.HydraRootConfig(command="inference")
    with pytest.raises(ValueError):
        hydra_cli.hydra_main(cfg)


def test_entrypoints_insert_command(monkeypatch):
    from src.cli import hydra_cli

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
