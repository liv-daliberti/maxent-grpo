"""Legacy training setup helpers preserved for test compatibility."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional

from core.data import get_dataset
from training.types import PromptCacheEntry
from training.weighting import (
    QDistributionSettings,
    WeightNormalizationSettings,
    WeightingSettings,
    TauSchedule,
    KlControllerSettings,
)
from .run_types import CheckpointConfig, HubPushConfig, PromptIOConfig, TrainDataBundle

# Ensure wandb import is harmless in test environments.
if "wandb" not in sys.modules:
    sys.modules["wandb"] = SimpleNamespace(errors=SimpleNamespace(Error=RuntimeError))


@dataclass
class EnvironmentConfig:
    """Captured environment variables (lightweight shim)."""

    values: dict[str, str]

    @classmethod
    def capture(cls) -> "EnvironmentConfig":
        return cls(values=dict(os.environ))


@dataclass
class DatasetContext:
    """Inputs required to map datasets to prompts."""

    script_args: Any
    training_args: Any
    tokenizer: Any


@dataclass
class DataLoaderRuntimeOptions:
    """Runtime knobs forwarded to the dataloader constructor."""

    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False


@dataclass
class DataPrepConfig:
    """Aggregate configuration for preparing the training dataloader."""

    context: DatasetContext
    batch_size: int
    max_prompt_len: int
    data_loader_cls: Any
    runtime: DataLoaderRuntimeOptions


@dataclass
class TrainingHyperparams:
    batch_size: int
    num_epochs: int
    grad_accum_steps: int
    learning_rate: float
    max_grad_norm: float


def _hydrate_reference_embeddings(train_model: Any, ref_model: Any) -> bool:
    """Copy train embeddings into the reference model and retie lm_head."""
    try:
        train_embed = train_model.get_input_embeddings()
        ref_embed = ref_model.get_input_embeddings()
        ref_head = ref_model.get_output_embeddings()
    except AttributeError:
        return False
    if train_embed is None or ref_embed is None or ref_head is None:
        return False
    weight = getattr(train_embed, "weight", None)
    if weight is None or getattr(weight, "numel", lambda: 0)() == 0:
        return False
    cloned = weight.detach().clone()
    setattr(cloned, "requires_grad", False)
    ref_embed.weight = cloned
    ref_head.weight = cloned
    return True


def prepare_with_accelerator(
    accelerator: Any, bundle: Any, optimizer: Any, train_data: TrainDataBundle
):
    """Wrap model/optimizer/dataloader with accelerator.prepare and update steps."""
    model, opt, loader = accelerator.prepare(
        bundle.model, optimizer, train_data.train_loader
    )
    bundle.model = model
    train_data.train_loader = loader
    try:
        train_data.steps_per_epoch = len(loader)
    except (TypeError, AttributeError):
        pass
    return bundle, opt, train_data


def resolve_training_hyperparams(args: Any) -> TrainingHyperparams:
    """Validate core hyperparameters and return a structured bundle."""
    if args.per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be > 0")
    if args.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if args.max_grad_norm is not None and args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be >= 0")
    return TrainingHyperparams(
        batch_size=args.per_device_train_batch_size,
        num_epochs=args.num_train_epochs,
        grad_accum_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
    )


def _resolve_q_distribution(args: Any) -> QDistributionSettings:
    """Build q-distribution settings, neutralising when training GRPO objective."""
    if getattr(args, "train_grpo_objective", False):
        return QDistributionSettings(temperature=1.0, epsilon=1e-8)
    return QDistributionSettings(
        temperature=getattr(args, "maxent_q_temperature", 1.0),
        epsilon=getattr(args, "maxent_q_epsilon", 0.0),
    )


def _resolve_weighting_settings(
    args: Any, env_config: Optional[EnvironmentConfig] = None
) -> WeightingSettings:
    """Create weighting settings with beta/tau bookkeeping."""
    values = env_config.values if env_config else {}
    beta = (
        getattr(args, "init_kl_coeff", None)
        if getattr(args, "init_kl_coeff", None) is not None
        else getattr(args, "init_kl_coef", None)
    )
    if beta is None:
        beta = getattr(args, "beta", 0.0)
    tau = getattr(args, "maxent_tau", None)
    if tau is None:
        tau = float(values.get("MAXENT_TAU", 0.0))
    train_grpo = bool(getattr(args, "train_grpo_objective", False))
    denom = beta if train_grpo else beta + float(tau or 0.0)
    q_temp = getattr(args, "maxent_q_temperature", None)
    q_eps = getattr(args, "maxent_q_epsilon", None)
    len_norm = getattr(args, "maxent_length_normalize_ref", None)
    if q_temp is None:
        q_temp = float(values.get("MAXENT_Q_TEMPERATURE", 1.0))
    if q_eps is None:
        q_eps = float(values.get("MAXENT_Q_EPS", 0.0))
    if len_norm is None:
        len_norm = values.get("MAXENT_LENGTH_NORM_REF", "1").strip() not in {
            "0",
            "false",
            "False",
        }
    if getattr(args, "train_grpo_objective", False):
        q_temp, q_eps = 1.0, 1e-8
    q_dist = QDistributionSettings(temperature=q_temp, epsilon=q_eps)
    weight_norm = WeightNormalizationSettings(denom=denom, len_norm_ref=bool(len_norm))
    tau_schedule = TauSchedule(
        target_entropy=getattr(args, "maxent_target_weight_entropy", None),
        learning_rate=getattr(args, "maxent_tau_lr", 0.0),
        minimum_value=getattr(args, "maxent_tau_min", 0.0),
        maximum_value=getattr(args, "maxent_tau_max", 0.0),
        warmup_steps=getattr(
            args, "maxent_tau_warmup_steps", getattr(args, "warmup_steps", -1)
        ),
    )
    kl_ctrl = KlControllerSettings(
        target=getattr(args, "kl_target", 0.0),
        horizon=getattr(args, "kl_horizon", 0),
        step_size=getattr(args, "kl_ctl_step_size", 0.0),
    )
    return WeightingSettings(
        tau=tau,
        beta=beta,
        normalization=weight_norm,
        q_distribution=q_dist,
        tau_schedule=tau_schedule,
        kl_controller=kl_ctrl,
        train_grpo_objective=train_grpo,
    )


def build_checkpoint_config(
    args: Any,
    ensure_output_dir: bool = True,
    makedirs: Callable[..., None] = os.makedirs,
) -> CheckpointConfig:
    """Normalize checkpoint arguments into a CheckpointConfig."""
    strategy = args.save_strategy
    if hasattr(strategy, "value"):
        strategy = strategy.value
    strategy = str(strategy)
    if ensure_output_dir:
        makedirs(args.output_dir, exist_ok=True)
    hub_cfg = HubPushConfig(
        enabled=bool(getattr(args, "push_to_hub", False)),
        model_id=args.hub_model_id,
        token=args.hub_token,
    )
    return CheckpointConfig(
        output_dir=args.output_dir,
        save_strategy=strategy,
        save_steps=int(getattr(args, "save_steps", 0)),
        save_total_limit=int(getattr(args, "save_total_limit", 0)),
        hub=hub_cfg,
    )


def _maybe_prepare_with_trl(accelerator: Any, model: Any) -> Any:
    """Delegate to TRL's prepare_deepspeed when available."""
    plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    try:
        import trl.trainer.utils as trl_utils  # type: ignore
    except ImportError:
        return model
    prepare = getattr(trl_utils, "prepare_deepspeed", None)
    if callable(prepare):
        prepare(model, accelerator, plugin)
    return model


def _build_prompt_length_cache(
    tokenizer: Any, max_prompt_len: int
) -> Callable[[str], PromptCacheEntry]:
    """Return a callable that caches prompt tokenization lengths."""

    def _cache(prompt: str) -> PromptCacheEntry:
        encoded = tokenizer(prompt)
        input_ids = list(encoded.get("input_ids", []))
        attn = list(encoded.get("attention_mask", []))
        return PromptCacheEntry(
            input_ids=input_ids[:max_prompt_len], attention_mask=attn[:max_prompt_len]
        )

    return _cache


def _map_training_dataset(config: DatasetContext, _columns: Optional[list[str]] = None):
    """Load the training split from the configured dataset."""
    dataset = get_dataset(config.script_args)
    split = getattr(config.script_args, "dataset_train_split", "train")
    return dataset[split]


def prepare_training_data(config: DataPrepConfig) -> TrainDataBundle:
    """Prepare dataset/loader artifacts using the provided config."""
    dataset = _map_training_dataset(config.context, None)
    prompt_io = PromptIOConfig(
        prompt_column=getattr(
            config.context.script_args, "dataset_prompt_column", "prompt"
        ),
        solution_column=getattr(
            config.context.script_args, "dataset_solution_column", "answer"
        ),
        prompt_length_cache_get=_build_prompt_length_cache(
            config.context.tokenizer, config.max_prompt_len
        ),
    )
    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
        drop_last=config.runtime.drop_last,
        persistent_workers=config.runtime.persistent_workers,
    )
    loader = config.data_loader_cls(dataset, **loader_kwargs)
    steps = len(loader) if hasattr(loader, "__len__") else 0
    return TrainDataBundle(
        train_dataset=dataset,
        train_loader=loader,
        train_sampler=None,
        prompt_io=prompt_io,
        steps_per_epoch=steps,
        batch_size=config.batch_size,
    )


def bootstrap_for_tests(frameworks: Any, _env_config: EnvironmentConfig):
    """Return accelerator and helper namespace for test shims."""
    accelerator = frameworks.accelerator_cls()
    helpers = SimpleNamespace()
    return accelerator, helpers


def EnvironmentConfig_capture() -> EnvironmentConfig:  # pragma: no cover - legacy alias
    return EnvironmentConfig.capture()


__all__ = [
    "_hydrate_reference_embeddings",
    "prepare_with_accelerator",
    "resolve_training_hyperparams",
    "_resolve_weighting_settings",
    "_resolve_q_distribution",
    "build_checkpoint_config",
    "_maybe_prepare_with_trl",
    "_build_prompt_length_cache",
    "prepare_training_data",
    "EnvironmentConfig",
    "DatasetContext",
    "DataLoaderRuntimeOptions",
    "DataPrepConfig",
    "TrainingHyperparams",
    "bootstrap_for_tests",
]
