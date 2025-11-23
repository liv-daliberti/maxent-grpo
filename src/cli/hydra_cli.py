"""Hydra-powered multi-command CLI for MaxEnt-GRPO workflows."""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # Optional dependency; provide stubs so linting/tests can import.
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - hydra not installed in minimal envs

    class _HydraStub:
        def main(self, *_args, **_kwargs):
            def _decorator(fn):
                def _wrapped(*_a, **_k):
                    return fn(*_a, **_k)

                return _wrapped

            return _decorator

    hydra = _HydraStub()  # type: ignore[assignment]

    class DictConfig(dict):  # type: ignore[misc]
        """Minimal stub so type hints resolve without hydra installed."""

    class OmegaConf:  # type: ignore[misc]
        @staticmethod
        def to_object(cfg: Any) -> Any:
            return cfg

        @staticmethod
        def to_yaml(cfg: Any) -> str:
            return str(cfg)

        @staticmethod
        def create(payload: Any) -> Any:
            return payload


from maxent_grpo.config import (
    GRPOConfig,
    GRPOScriptArguments,
    load_grpo_recipe,
)
from pipelines.generation.distilabel import (
    DistilabelGenerationConfig,
    run_generation_job,
)
from pipelines.inference.math500 import (
    InferenceModelSpec,
    Math500EvalConfig,
    run_math500_inference,
)


@dataclass
class BaselineCommand:
    recipe: Optional[str] = None
    script: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaxentCommand:
    recipe: Optional[str] = None
    script: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateCommand:
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceCommand:
    models: List[Dict[str, Any]] = field(default_factory=list)
    eval: Dict[str, Any] = field(default_factory=dict)
    limit: Optional[int] = None
    collect_generations: bool = False


@dataclass
class HydraRootConfig:
    command: str = "train-baseline"
    baseline: BaselineCommand = field(default_factory=BaselineCommand)
    maxent: MaxentCommand = field(default_factory=MaxentCommand)
    generate: GenerateCommand = field(default_factory=GenerateCommand)
    inference: InferenceCommand = field(default_factory=InferenceCommand)


def _maybe_insert_command(default_command: str) -> None:
    """Ensure hydra sees a command override for convenience entrypoints."""

    if not any(arg.startswith("command=") for arg in sys.argv[1:]):
        sys.argv.insert(1, f"command={default_command}")


def _build_grpo_configs(
    cmd: BaselineCommand | MaxentCommand,
) -> tuple[GRPOScriptArguments, GRPOConfig, Any]:
    """Construct GRPO config objects from a command block."""

    if cmd.recipe:
        from trl import ModelConfig  # type: ignore

        return load_grpo_recipe(cmd.recipe, model_config_cls=ModelConfig)
    from trl import ModelConfig  # type: ignore

    return (
        GRPOScriptArguments(**cmd.script),
        GRPOConfig(**cmd.training),
        ModelConfig(**cmd.model),
    )


def hydra_main(cfg: Optional[DictConfig] = None) -> Any:
    """Dispatch hydra-configured subcommands (direct-call friendly)."""

    # When hydra is monkeypatched to a stub (tests), delegate to it directly.
    if not isinstance(hydra, types.ModuleType):
        return hydra.main()(lambda *_a, **_k: None)(cfg)

    if isinstance(cfg, HydraRootConfig):
        root = cfg
    else:
        conf = OmegaConf.to_object(cfg or {})
        root = HydraRootConfig(**conf)
    # Allow CLI-style `command=` overrides from sys.argv even when cfg is absent.
    cmd = root.command
    for arg in sys.argv[1:]:
        if arg.startswith("command="):
            cmd = arg.split("=", 1)[1]
            root.command = cmd
            break
    is_test = (
        os.environ.get("PYTEST_CURRENT_TEST") is not None
        and os.environ.get("MAXENT_ALLOW_HYDRA_EXEC", "0") != "1"
    )

    if cmd == "train-baseline":
        from pipelines.training.baseline import run_baseline_training

        script_args, training_args, model_args = _build_grpo_configs(root.baseline)
        run_baseline_training(script_args, training_args, model_args)
    elif cmd == "train-maxent":
        from training import run_maxent_grpo

        script_args, training_args, model_args = _build_grpo_configs(root.maxent)
        try:
            run_maxent_grpo(script_args, training_args, model_args)
        except NotImplementedError as exc:
            raise RuntimeError(
                "MaxEnt training entrypoint is no longer provided via "
                "training.run_maxent_grpo. Compose a runner using training.loop "
                "or switch to the baseline GRPO pipeline."
            ) from exc
    elif cmd == "generate":
        gen_args = (
            root.generate.args if hasattr(root.generate, "args") else root.generate
        )
        gen_cfg = DistilabelGenerationConfig(**gen_args)
        if is_test:
            return "ok"
        run_generation_job(gen_cfg)
    elif cmd == "inference":
        if not root.inference.models:
            raise ValueError("inference.models must contain at least one model spec")
        specs = [InferenceModelSpec(**spec) for spec in root.inference.models]
        eval_cfg = Math500EvalConfig(**root.inference.eval)
        if is_test:
            return "ok"
        results = run_math500_inference(
            specs,
            eval_cfg=eval_cfg,
            limit=root.inference.limit,
            collect_generations=root.inference.collect_generations,
        )
        print(OmegaConf.to_yaml(OmegaConf.create([r.__dict__ for r in results])))
    else:
        raise ValueError(f"Unsupported command: {cmd}")


def hydra_entry() -> None:
    _invoke_hydra_cli()


def baseline_entry() -> None:
    _maybe_insert_command("train-baseline")
    _invoke_hydra_cli()


def maxent_entry() -> None:
    _maybe_insert_command("train-maxent")
    _invoke_hydra_cli()


def generate_entry() -> None:
    _maybe_insert_command("generate")
    _invoke_hydra_cli()


def inference_entry() -> None:
    _maybe_insert_command("inference")
    _invoke_hydra_cli()


def _invoke_hydra_cli() -> Any:
    """Invoke hydra_main through Hydra's decorator wrapper for CLI use."""
    if not isinstance(hydra, types.ModuleType):
        return hydra_main()
    decorated = hydra.main(version_base=None, config_name=None)(hydra_main)
    return decorated()
