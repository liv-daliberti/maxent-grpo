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

Hydra-powered multi-command CLI for MaxEnt-GRPO workflows.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from maxent_grpo.config import (
    GRPOConfig,
    GRPOScriptArguments,
    load_grpo_recipe,
)
from maxent_grpo.pipelines.generation.distilabel import (
    DistilabelGenerationConfig,
    run_generation_job,
)
from maxent_grpo.pipelines.inference.inference import (
    InferenceArtifactConfig,
    InferenceModelSpec,
    resolve_inference_dataset,
    run_math_inference,
)


class _HydraStub:
    """Minimal Hydra-like stub used when hydra is absent."""

    def main(self, *_args, **_kwargs):
        """Return a decorator that forwards directly to the wrapped function.

        :param _args: Positional arguments ignored by the stub.
        :param _kwargs: Keyword arguments ignored by the stub.
        :returns: Decorator mimicking :func:`hydra.main`.
        """

        def _decorator(fn):
            def _wrapped(*_a, **_k):
                return fn(*_a, **_k)

            return _wrapped

        return _decorator


try:  # Optional dependency; provide stubs so linting/tests can import.
    import hydra
    from omegaconf import DictConfig, OmegaConf, open_dict
except ImportError:  # pragma: no cover - hydra not installed in minimal envs

    hydra = _HydraStub()

    class DictConfig(dict):
        """Minimal stub so type hints resolve without hydra installed."""

    class OmegaConf:
        @staticmethod
        def to_object(cfg: Any) -> Any:
            return cfg

        @staticmethod
        def to_yaml(cfg: Any) -> str:
            return str(cfg)

        @staticmethod
        def create(payload: Any) -> Any:
            return payload

        @staticmethod
        def structured(obj: Any) -> Any:
            return obj


@dataclass
class BaselineCommand:
    """GRPO training command options for the baseline recipe.

    :param recipe: Optional recipe file path to load default configs from.
    :param script: Script-level overrides passed to GRPO script arguments.
    :param training: Training argument overrides passed to GRPO config.
    :param model: Model argument overrides passed to TRL model config.
    """

    recipe: Optional[str] = None
    script: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaxentCommand:
    """GRPO training command options for the MaxEnt recipe.

    :param recipe: Optional recipe file path to load default configs from.
    :param script: Script-level overrides passed to GRPO script arguments.
    :param training: Training argument overrides passed to GRPO config.
    :param model: Model argument overrides passed to TRL model config.
    """

    recipe: Optional[str] = None
    script: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfoSeedCommand:
    """GRPO training command options for the InfoSeed recipe."""

    recipe: Optional[str] = None
    script: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateCommand:
    """Generation job configuration block.

    :param args: Keyword arguments forwarded to :class:`DistilabelGenerationConfig`.
    """

    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceCommand:
    """Inference job configuration block.

    :param models: Sequence of model specs to evaluate.
    :param dataset: Named inference preset to evaluate.
    :param eval: Evaluation configuration overrides for math benchmarks.
    :param seeds: RNG seeds to average metrics over.
    :param num_generations: Number of completions per prompt (Pass@k with k=num_generations).
    :param temperature: Temperature override for rollout sampling.
    :param limit: Optional cap on number of items to evaluate.
    :param collect_generations: Whether to return collected generations.
    :param artifacts: Artifact persistence/resume configuration.
    """

    models: List[Dict[str, Any]] = field(default_factory=list)
    dataset: str = "math_500"
    eval: Dict[str, Any] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_generations: int = 8
    temperature: float = 0.6
    limit: Optional[int] = None
    collect_generations: bool = False
    artifacts: InferenceArtifactConfig = field(default_factory=InferenceArtifactConfig)


@dataclass
class HydraRootConfig:
    """Hydra root configuration covering all supported CLI commands.

    :param command: Name of the subcommand to run.
    :param baseline: Baseline training command configuration.
    :param maxent: MaxEnt training command configuration.
    :param generate: Generation job configuration.
    :param inference: Inference job configuration.
    """

    command: str = "train-baseline"
    baseline: BaselineCommand = field(default_factory=BaselineCommand)
    maxent: MaxentCommand = field(default_factory=MaxentCommand)
    infoseed: InfoSeedCommand = field(default_factory=InfoSeedCommand)
    generate: GenerateCommand = field(default_factory=GenerateCommand)
    inference: InferenceCommand = field(default_factory=InferenceCommand)


def _maybe_insert_command(default_command: str) -> None:
    """Ensure hydra sees a command override for convenience entrypoints.

    :param default_command: Command name inserted when no explicit ``command=`` is present.
    :returns: ``None``; updates ``sys.argv`` in-place when needed.
    """

    if not any(
        arg.startswith("command=") or arg.startswith("+command=")
        for arg in sys.argv[1:]
    ):
        sys.argv.insert(1, f"command={default_command}")


def _build_grpo_configs(
    cmd: BaselineCommand | MaxentCommand | InfoSeedCommand,
) -> tuple[GRPOScriptArguments, GRPOConfig, Any]:
    """Construct GRPO config objects from a command block.

    :param cmd: Command payload defining script, training, and model sections.
    :returns: Tuple of ``(script_args, training_args, model_config)`` ready to pass to training pipelines.
    """

    if getattr(cmd, "recipe", None):
        try:
            from trl import ModelConfig  # type: ignore
        except Exception:
            class ModelConfig:  # type: ignore[override]
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

        return load_grpo_recipe(cmd.recipe, model_config_cls=ModelConfig)
    try:
        from trl import ModelConfig  # type: ignore
    except Exception:
        class ModelConfig:  # type: ignore[override]
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

    # Avoid parser conflicts by keeping reward-related flags on the training config.
    training_payload = dict(cmd.training)
    script_payload = dict(cmd.script)
    if "reward_funcs" in script_payload and "reward_funcs" not in training_payload:
        training_payload["reward_funcs"] = script_payload.pop("reward_funcs")
    if "reward_weights" in script_payload and "reward_weights" not in training_payload:
        training_payload["reward_weights"] = script_payload.pop("reward_weights")

    return (
        GRPOScriptArguments(**script_payload),
        GRPOConfig(**training_payload),
        ModelConfig(**cmd.model),
    )


def hydra_main(cfg: Optional[DictConfig] = None) -> Any:
    """Dispatch hydra-configured subcommands (direct-call friendly).

    :param cfg: Optional Hydra configuration object or plain dict derived from CLI files.
    :returns: Result of the executed command, or ``None`` for commands that only have side effects.
    :raises ValueError: If an unsupported command name is supplied.
    """

    # When hydra is monkeypatched to a stub (tests), delegate to it directly.
    if not isinstance(hydra, types.ModuleType):
        return hydra.main()(lambda *_a, **_k: None)(cfg)

    if isinstance(cfg, HydraRootConfig):
        root = cfg
    elif hasattr(OmegaConf, "structured"):
        structured_root = OmegaConf.structured(HydraRootConfig())
        if cfg is not None:
            with open_dict(structured_root):
                structured_root.merge_with(cfg)
        conf = OmegaConf.to_object(structured_root)
        if isinstance(conf, HydraRootConfig):
            root = conf
        else:
            root = HydraRootConfig(**conf)
    else:
        payload = cfg or {}
        if isinstance(payload, HydraRootConfig):
            root = payload
        elif isinstance(payload, dict):
            root = HydraRootConfig(**payload)
        else:
            root = HydraRootConfig()
    # Allow CLI-style `command=` overrides from sys.argv even when cfg is absent.
    cmd = root.command
    for arg in sys.argv[1:]:
        if arg.startswith("command=") or arg.startswith("+command="):
            cmd = arg.split("=", 1)[1]
            root.command = cmd
            break
    is_test = (
        os.environ.get("PYTEST_CURRENT_TEST") is not None
        and os.environ.get("MAXENT_ALLOW_HYDRA_EXEC", "0") != "1"
    )

    if cmd == "train-baseline":
        from maxent_grpo.pipelines.training.baseline import run_baseline_training

        script_args, training_args, model_args = _build_grpo_configs(root.baseline)
        run_baseline_training(script_args, training_args, model_args)
    elif cmd == "train-maxent":
        from maxent_grpo.pipelines.training.maxent import run_maxent_training

        script_args, training_args, model_args = _build_grpo_configs(root.maxent)
        run_maxent_training(script_args, training_args, model_args)
    elif cmd == "train-infoseed":
        from maxent_grpo.pipelines.training.infoseed import run_infoseed_training

        script_args, training_args, model_args = _build_grpo_configs(root.infoseed)
        run_infoseed_training(script_args, training_args, model_args)
    elif cmd == "generate":
        gen_args = (
            root.generate.args if hasattr(root.generate, "args") else root.generate
        )
        gen_cfg = DistilabelGenerationConfig(**gen_args)
        if is_test:
            return "ok"
        run_generation_job(gen_cfg)
    elif cmd in ("inference", "math-eval", "math_eval"):
        if not root.inference.models:
            raise ValueError("inference.models must contain at least one model spec")
        specs = [InferenceModelSpec(**spec) for spec in root.inference.models]
        dataset_name = getattr(root.inference, "dataset", None) or "math_500"
        eval_cfg = resolve_inference_dataset(
            dataset_name,
            root.inference.eval,
        )
        if is_test:
            return "ok"
        results = run_math_inference(
            specs,
            eval_cfg=eval_cfg,
            limit=root.inference.limit,
            collect_generations=root.inference.collect_generations,
            num_generations=root.inference.num_generations,
            seeds=root.inference.seeds,
            temperature=root.inference.temperature,
            dataset_id=dataset_name,
            artifact_config=root.inference.artifacts,
        )
        print(OmegaConf.to_yaml(OmegaConf.create([r.__dict__ for r in results])))
    else:
        raise ValueError(f"Unsupported command: {cmd}")


def hydra_entry() -> None:
    """Entry point for the top-level Hydra CLI.

    :returns: ``None`` after invoking the configured command.
    """
    _invoke_hydra_cli()


def baseline_entry() -> None:
    """Console script wrapper for baseline training.

    :returns: ``None`` after dispatching to Hydra.
    """
    _maybe_insert_command("train-baseline")
    _invoke_hydra_cli()


def maxent_entry() -> None:
    """Console script wrapper for MaxEnt training.

    :returns: ``None`` after dispatching to Hydra.
    """
    _maybe_insert_command("train-maxent")
    _invoke_hydra_cli()


def generate_entry() -> None:
    """Console script wrapper for dataset generation.

    :returns: ``None`` after dispatching to Hydra.
    """
    _maybe_insert_command("generate")
    _invoke_hydra_cli()


def inference_entry() -> None:
    """Console script wrapper for math inference evaluation.

    :returns: ``None`` after dispatching to Hydra.
    """
    _maybe_insert_command("inference")
    _invoke_hydra_cli()


def math_eval_entry() -> None:
    """Console script wrapper for multi-benchmark math inference."""

    _maybe_insert_command("math-eval")
    _invoke_hydra_cli()


def infoseed_entry() -> None:
    """Console script wrapper for InfoSeed training."""

    _maybe_insert_command("train-infoseed")
    _invoke_hydra_cli()


def _invoke_hydra_cli() -> Any:
    """Invoke hydra_main through Hydra's decorator wrapper for CLI use.

    :returns: Result of :func:`hydra_main`, forwarded directly.
    """
    if not isinstance(hydra, types.ModuleType):
        return hydra_main()
    decorated = hydra.main(version_base=None, config_name=None)(hydra_main)
    return decorated()
