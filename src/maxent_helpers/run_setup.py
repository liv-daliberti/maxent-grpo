"""Helper functions for preparing the MaxEnt-GRPO runner."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import sys
import threading
import traceback
import faulthandler
from contextlib import contextmanager, nullcontext
import inspect
from dataclasses import dataclass, replace
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from configs import GRPOConfig, GRPOScriptArguments
from rewards import get_reward_funcs
from utils.data import get_dataset, load_dataset_split
from utils.model_utils import get_model, get_tokenizer

from .run_helpers import (
    GenerationPenaltyConfig,
    VLLMClientConfig,
    _maybe_create_deepspeed_plugin,
    _prompt_char_limit_from_tokens,
    _to_prompt,
    _truncate_prompt,
)
from .zero_utils import _maybe_zero_gather_embedding, _maybe_zero_gather_params
from .run_training_types import (
    BatchingSettings,
    ClipSettings,
    EvaluationSettings,
    GenerationSettings,
    KlControllerSettings,
    OptimizationSchedule,
    OptimizationSettings,
    OptimizerHandles,
    PromptCacheEntry,
    QDistributionSettings,
    RewardSpec,
    ScoringSettings,
    TauSchedule,
    WeightingSettings,
    WeightNormalizationSettings,
)
from .run_types import (
    CheckpointConfig,
    DatasetColumns,
    DatasetContext,
    DataPrepConfig,
    DataLoaderRuntimeOptions,
    EvaluationConfig,
    FrameworkHandles,
    GenerationConfig,
    HubPushConfig,
    LearningConfig,
    ModelBundle,
    OptimizerContext,
    PromptIOConfig,
    RunnerSetup,
    RuntimeArtifacts,
    LengthSettings,
    SamplingParams,
    SamplingPenalties,
    SamplingStopConfig,
    TrainingComponents,
    TrainingHyperParams,
    TrainingScheduleConfig,
    TrainDataBundle,
)

LOG = logging.getLogger(__name__)


@dataclass
class StageLogger:
    """Wrapper around logging.Logger to simplify testing."""

    logger: logging.Logger

    def log(self, message: str, accelerator: Optional[Any] = None) -> None:
        if accelerator is None or getattr(accelerator, "is_main_process", True):
            self.logger.info(message)


class WatchdogFactory:
    """Factory producing watchdog contexts for long-running operations."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def __call__(
        self,
        label: str,
        timeout: float = 120.0,
        heartbeat: float = 30.0,
    ):
        @contextmanager
        def _context():
            cancelled = threading.Event()
            timers: List[threading.Timer] = []

            def _dump_stack() -> None:
                self._logger.warning(
                    "Reference model gather appears stuck (%s). Stack:\n%s",
                    label,
                    "".join(traceback.format_stack()),
                )
                try:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                except (OSError, RuntimeError):  # pragma: no cover
                    pass

            def _heartbeat() -> None:
                if cancelled.is_set():
                    return
                self._logger.info("Reference model gather still in progress (%s)...", label)
                timer = threading.Timer(heartbeat, _heartbeat)
                timer.daemon = True
                timer.start()
                timers.append(timer)

            timeout_timer = threading.Timer(timeout, _dump_stack)
            timeout_timer.daemon = True
            timeout_timer.start()
            timers.append(timeout_timer)
            if heartbeat > 0:
                hb_timer = threading.Timer(heartbeat, _heartbeat)
                hb_timer.daemon = True
                hb_timer.start()
                timers.append(hb_timer)
            try:
                yield
            finally:
                cancelled.set()
                for timer in timers:
                    timer.cancel()

        return _context()


_STAGE_LOGGER = StageLogger(LOG)
_WATCHDOG_FACTORY = WatchdogFactory(LOG)


@dataclass(frozen=True)
class EnvironmentConfig:
    """Thin wrapper around environment variables for easier injection/testing."""

    values: Dict[str, str]

    @classmethod
    def capture(cls) -> "EnvironmentConfig":
        """Snapshot os.environ into a test-friendly config."""
        return cls(dict(os.environ))

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.values.get(key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.values.get(key)
        if val is None:
            return default
        return str(val).lower() in {"1", "true", "yes"}


def _log_stage(
    message: str,
    accelerator: Optional[Any] = None,
    stage_logger: Optional[StageLogger] = None,
) -> None:
    """Emit setup progress logs on the main process.

    Logs every rank when the accelerator handle is absent.
    """
    logger = stage_logger or _STAGE_LOGGER
    logger.log(message, accelerator)


def _import_dependency(module_name: str, hint: str) -> Any:
    """Import a dependency or raise a descriptive error."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ImportError(hint) from exc


def resolve_frameworks() -> FrameworkHandles:
    """Resolve torch/transformers/accelerate lazily for pylint-friendly imports."""
    torch_module = _import_dependency(
        "torch",
        "PyTorch is required for MaxEnt-GRPO. Install it via `pip install torch`.",
    )
    torch_data = _import_dependency(
        "torch.utils.data",
        "torch.utils.data is unavailable. Ensure PyTorch is installed correctly.",
    )
    transformers_module = _import_dependency(
        "transformers",
        (
            "Transformers is required for MaxEnt-GRPO. "
            "Install it via `pip install transformers`."
        ),
    )
    accelerate_module = _import_dependency(
        "accelerate",
        (
            "Accelerate is required for this training entrypoint. "
            "Install it via `pip install accelerate`."
        ),
    )
    accelerator_cls = getattr(accelerate_module, "Accelerator")
    data_loader_cls = getattr(torch_data, "DataLoader")
    return FrameworkHandles(
        torch=torch_module,
        data_loader_cls=data_loader_cls,
        transformers=transformers_module,
        accelerator_cls=accelerator_cls,
    )


def create_accelerator(frameworks: FrameworkHandles, env_config: EnvironmentConfig) -> Any:
    """Create an Accelerator instance (optionally with DeepSpeed)."""
    ds_plugin = _maybe_create_deepspeed_plugin()
    # We already construct a DistributedSampler; avoid splitting batches a second
    # time inside Accelerate, which would shrink steps_per_epoch by world size.
    accelerator = frameworks.accelerator_cls(
        deepspeed_plugin=ds_plugin,
        split_batches=False,
    )
    os.makedirs(env_config.get("LOG_DIR", "logs") or "logs", exist_ok=True)
    return accelerator


def configure_logging(training_args: GRPOConfig, transformers_module: Any) -> None:
    """Apply standard logging configuration + HF verbosity."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = (
        training_args.get_process_log_level()
        if hasattr(training_args, "get_process_log_level")
        else logging.INFO
    )
    logging.getLogger(__name__).setLevel(log_level)
    transformers_module.utils.logging.set_verbosity(log_level)
    transformers_module.utils.logging.enable_default_handler()
    transformers_module.utils.logging.enable_explicit_format()


def seed_everything(training_args: GRPOConfig, transformers_module: Any) -> None:
    """Seed Python/torch/random through transformers.set_seed when requested."""
    seed = getattr(training_args, "seed", None)
    if seed is None:
        return
    set_seed = getattr(transformers_module, "set_seed", None)
    if callable(set_seed):
        set_seed(seed)


def _ensure_tokenizer_padding(tokenizer: Any, model: Any) -> None:
    """Guarantee PAD tokens exist for both tokenizer + model configs."""
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    try:
        tokenizer.padding_side = "left"
    except AttributeError:
        pass


def _assign_reference_model(bundle: ModelBundle, ref_model: Any) -> None:
    """Store the constructed reference model inside a bundle-specific holder."""
    holder = getattr(bundle, "_ref_holder", None)
    if holder is None:
        holder = {}
        setattr(bundle, "_ref_holder", holder)
    holder["model"] = ref_model


def _require_ref_model(bundle: ModelBundle) -> Any:
    holder = getattr(bundle, "_ref_holder", None)
    if not holder or holder.get("model") is None:
        raise RuntimeError(
            "Reference model has not been initialized; "
            "call _initialize_reference_model after accelerator.prepare."
        )
    return holder["model"]


def _log_deepspeed_plugin(accelerator: Any) -> None:
    """Log basic DeepSpeed plugin settings to aid debugging."""
    state = getattr(accelerator, "state", None)
    plugin = getattr(state, "deepspeed_plugin", None)
    if plugin is None:
        return
    LOG.info(
        "DeepSpeed plugin | stage=%s | offload_param=%s | offload_optimizer=%s | "
        "reduce_bucket_size=%s | allgather_bucket_size=%s",
        getattr(plugin, "zero_stage", None),
        getattr(plugin, "offload_param", None),
        getattr(plugin, "offload_optimizer", None),
        getattr(plugin, "zero_reduce_bucket_size", None),
        getattr(plugin, "zero_allgather_bucket_size", None),
    )


def _gather_watchdog(
    label: str,
    timeout: float = 120.0,
    heartbeat: float = 30.0,
    watchdog_factory: Optional[WatchdogFactory] = None,
):
    """Log a warning if a long-running gather appears stuck, with heartbeats."""
    factory = watchdog_factory or _WATCHDOG_FACTORY
    return factory(label, timeout=timeout, heartbeat=heartbeat)


def _maybe_prepare_with_trl(accelerator: Any, model: Any) -> None:
    """Call TRL's prepare_deepspeed helper when available."""
    try:
        from trl.trainer.utils import prepare_deepspeed as trl_prepare_deepspeed  # type: ignore
    except ImportError:
        LOG.debug("TRL not installed; skipping prepare_deepspeed hook.")
        return
    plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    if plugin is None:
        return
    try:
        sig = inspect.signature(trl_prepare_deepspeed)
    except (TypeError, ValueError):
        sig = None
    try:
        if sig is not None:
            param_count = len(sig.parameters)
        else:
            param_count = 0
        if param_count <= 1:
            trl_prepare_deepspeed(model)  # type: ignore[call-arg]
        elif param_count == 2:
            trl_prepare_deepspeed(model, accelerator)  # type: ignore[call-arg]
        else:
            trl_prepare_deepspeed(model, accelerator, plugin)  # type: ignore[call-arg]
        LOG.info("Applied TRL prepare_deepspeed hook (params=%s).", param_count or "unknown")
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:  # pragma: no cover - defensive logging
        LOG.warning("Failed to run TRL prepare_deepspeed: %s", exc)


def _maybe_compile_model(
    frameworks: FrameworkHandles,
    accelerator: Any,
    model: Any,
    training_args: GRPOConfig,
    env_config: EnvironmentConfig,
) -> Any:
    """Optionally compile the trainable model when ZeRO-3 is not active."""
    torch_module = frameworks.torch
    compile_fn = getattr(torch_module, "compile", None)
    if not callable(compile_fn):
        return False
    ds_plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    try:
        zero_stage = int(getattr(ds_plugin, "zero_stage", 0) or 0)
    except (TypeError, ValueError):
        zero_stage = 0
    if zero_stage >= 3:
        LOG.info("Skipping torch.compile because DeepSpeed ZeRO-%s is active.", zero_stage)
        return False
    flag = env_config.get("MAXENT_COMPILE", None)
    compile_enabled = (
        flag.lower() in {"1", "true", "yes"}
        if flag is not None
        else bool(getattr(training_args, "maxent_compile", False))
    )
    if not compile_enabled:
        return False
    try:
        LOG.info("Compiling trainable model with torch.compile before accelerator.prepare.")
        compiled_model = compile_fn(model)
        if compiled_model is None:
            return False
        return compiled_model
    except (RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover - torch/runtime dependent
        LOG.warning("torch.compile failed; continuing without compilation: %s", exc)
        return False


def _broadcast_object(
    accelerator: Any,
    payload: List[Any],
    *,
    src: int = 0,
) -> None:
    """Broadcast python objects even when Accelerate lacks the helper."""
    broadcast_fn = getattr(accelerator, "broadcast_object_list", None)
    if callable(broadcast_fn):
        broadcast_fn(payload, src=src)
        return
    try:
        torch_module = import_module("torch")
        dist_module = getattr(torch_module, "distributed", None)
    except ModuleNotFoundError:
        dist_module = None
    if (
        dist_module is not None
        and dist_module.is_available()
        and dist_module.is_initialized()
    ):
        dist_module.broadcast_object_list(payload, src=src)


def _fetch_reference_state_dict(
    accelerator: Any,
    model: Any,
) -> Dict[str, Any]:
    """Gather a full-precision state dict on the main process and broadcast it."""
    _log_deepspeed_plugin(accelerator)
    wait_for_everyone = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait_for_everyone):
        LOG.info("Reference model: waiting for all ranks before gathering state dict...")
        wait_for_everyone()
    state_dict: Optional[Dict[str, Any]] = None
    is_main = bool(getattr(accelerator, "is_main_process", True))
    if is_main:
        unwrap_model = getattr(accelerator, "unwrap_model", None)
        base_model = unwrap_model(model) if callable(unwrap_model) else model
        base_model = getattr(base_model, "module", base_model)
        state_dict_fn = getattr(base_model, "state_dict", None)
        if callable(state_dict_fn):
            LOG.info("Reference model: entering ZeRO gather context (state_dict()).")
            with _gather_watchdog("state_dict()", timeout=300.0):
                with _maybe_zero_gather_params(base_model, True):
                    state_dict = state_dict_fn()
            LOG.info("Reference model: exited ZeRO gather context (state_dict()).")
        else:
            LOG.warning(
                "Reference model: base model lacks state_dict(); attempting "
                "accelerator.get_state_dict()."
            )
            get_sd = getattr(accelerator, "get_state_dict", None)
            if callable(get_sd):
                LOG.info("Reference model: entering ZeRO gather context (get_state_dict)...")
                with _gather_watchdog("get_state_dict()", timeout=300.0):
                    with _maybe_zero_gather_params(model, True):
                        state_dict = get_sd(model)
                LOG.info("Reference model: exited ZeRO gather context (get_state_dict).")
        if state_dict is None:
            raise RuntimeError(
                "Unable to gather reference state dict from the main process."
            )
        LOG.info(
            "Reference model: state_dict collected (keys=%d); broadcasting to ranks...",
            len(state_dict),
        )
    payload = [state_dict]
    _broadcast_object(accelerator, payload, src=0)
    result = payload[0]
    if result is None:
        raise RuntimeError(
            "Broadcasted reference state dict is None; verify distributed init."
        )
    LOG.info("Reference model: broadcast complete.")
    return result


def _get_state_dict_for_reference(accelerator: Any, model: Any) -> Dict[str, Any]:
    """Backward-compatible alias retained for unit tests."""
    return _fetch_reference_state_dict(accelerator, model)


def _materialize_reference_embeddings(train_model: Any, ref_model: Any) -> bool:
    """Ensure the frozen reference embedding matrix is fully materialized."""
    if train_model is None or ref_model is None:
        return False
    try:
        torch_module = import_module("torch")
    except ModuleNotFoundError:
        LOG.warning("Torch is unavailable; cannot hydrate reference embeddings.")
        return False
    nn_module = getattr(torch_module, "nn", None)
    parameter_cls = getattr(nn_module, "Parameter", None) if nn_module else None
    if parameter_cls is None:
        return False
    base_train = getattr(train_model, "module", train_model)
    base_ref = getattr(ref_model, "module", ref_model)
    train_embed = getattr(base_train, "get_input_embeddings", lambda: None)()
    ref_embed = getattr(base_ref, "get_input_embeddings", lambda: None)()
    if train_embed is None or ref_embed is None:
        return False
    ref_weight = getattr(ref_embed, "weight", None)
    if ref_weight is None:
        return False
    ref_device = getattr(ref_weight, "device", None)
    ref_dtype = getattr(ref_weight, "dtype", None)
    with _maybe_zero_gather_embedding(train_model):
        src_weight = getattr(train_embed, "weight", None)
        if src_weight is None:
            return False
        gathered = src_weight.detach().clone()
    if getattr(gathered, "ndim", 0) != 2 or gathered.numel() == 0:
        LOG.warning(
            "Gathered reference embedding has invalid shape: %s",
            getattr(gathered, "shape", None),
        )
        return False
    if hasattr(gathered, "to"):
        kwargs: Dict[str, Any] = {}
        if ref_device is not None:
            kwargs["device"] = ref_device
        if ref_dtype is not None:
            kwargs["dtype"] = ref_dtype
        if kwargs:
            gathered = gathered.to(**kwargs)
    new_weight = parameter_cls(gathered, requires_grad=False)  # type: ignore[arg-type]
    ref_embed.weight = new_weight
    ref_output = getattr(base_ref, "get_output_embeddings", lambda: None)()
    if ref_output is not None and hasattr(ref_output, "weight"):
        ref_output.weight = new_weight
    LOG.info(
        "Hydrated reference embeddings from trainable model (shape=%s)",
        tuple(getattr(new_weight, "shape", [])),
    )
    return True


def _hydrate_reference_embeddings(train_model: Any, ref_model: Any) -> bool:
    """Backward-compatible alias retained for unit tests."""
    return _materialize_reference_embeddings(train_model, ref_model)


def prepare_model_bundle(
    model_args: Any,
    training_args: GRPOConfig,
    accelerator: Any,
) -> ModelBundle:
    """Instantiate the trainable model/tokenizer pair."""
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)
    model.train()
    _ensure_tokenizer_padding(tokenizer, model)
    _log_stage("Trainable model and tokenizer instantiated.", accelerator)

    ref_holder: Dict[str, Any] = {"model": None}

    def _get_ref_model() -> Any:
        """Return the prepared frozen reference model."""
        return _require_ref_model(bundle)

    bundle = ModelBundle(model=model, tokenizer=tokenizer, get_ref_model=_get_ref_model)
    setattr(bundle, "_ref_holder", ref_holder)
    return bundle


def _initialize_reference_model(
    accelerator: Any,
    model_bundle: ModelBundle,
    model_args: Any,
    training_args: GRPOConfig,
    env_config: EnvironmentConfig,
) -> None:
    """Load the frozen reference model after ZeRO materializes weights."""
    _log_stage("Reference model: preparing base configuration...", accelerator)
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    base_train = unwrap_model(model_bundle.model) if callable(unwrap_model) else model_bundle.model
    base_train = getattr(base_train, "module", base_train)
    reuse_flag, reuse_reason = _should_reuse_train_reference(env_config, accelerator)
    if reuse_flag:
        _reuse_trainable_model_as_reference(base_train, model_bundle, accelerator, reuse_reason)
        return
    ref_model = _load_reference_from_checkpoint(model_args, training_args, accelerator)
    if ref_model is None:
        ref_model = _build_reference_from_train_state(
            accelerator,
            model_bundle.model,
            base_train,
        )
    ref_model.to(accelerator.device)
    _assign_reference_model(model_bundle, ref_model)


def _should_reuse_train_reference(
    env_config: EnvironmentConfig,
    accelerator: Any,
) -> Tuple[bool, str]:
    reuse_train = env_config.get_bool("MAXENT_USE_TRAIN_AS_REFERENCE")
    if reuse_train:
        return True, "env override"
    main_only = env_config.get_bool("MAXENT_REFERENCE_MAIN_ONLY")
    if main_only and not accelerator.is_main_process:
        return True, "non-main rank shortcut"
    return False, ""


def _reuse_trainable_model_as_reference(
    base_train: Any,
    model_bundle: ModelBundle,
    accelerator: Any,
    reason: str,
) -> None:
    _log_stage(
        f"Reference model: reusing trainable model as reference ({reason}).",
        accelerator,
    )
    _freeze_model_parameters(base_train)
    _assign_reference_model(model_bundle, base_train)


def _load_reference_from_checkpoint(
    model_args: Any,
    training_args: GRPOConfig,
    accelerator: Any,
) -> Optional[Any]:
    try:
        _log_stage(
            "Reference model: loading from checkpoint per rank (no ZeRO gather).",
            accelerator,
        )
        ref_model = get_model(model_args, training_args)
        _freeze_model_parameters(ref_model)
        return ref_model
    except (OSError, RuntimeError, ValueError, TypeError):
        LOG.warning(
            "Reference model: failed to load from checkpoint; falling back to train-state gather.",
            exc_info=True,
        )
        return None


def _build_reference_from_train_state(
    accelerator: Any,
    train_model: Any,
    base_train: Any,
) -> Any:
    ref_model = _instantiate_reference_clone(base_train, accelerator)
    loaded = _load_reference_weights_from_accelerator(accelerator, train_model, ref_model)
    if not loaded:
        _log_stage(
            "Reference model: skipping full state_dict load; hydrating embeddings only.",
            accelerator,
        )
    if not _materialize_reference_embeddings(base_train, ref_model):
        LOG.warning(
            "Failed to hydrate reference embeddings; reference scoring may remain sharded."
        )
    else:
        _log_stage("Reference model: embeddings materialized.", accelerator)
    return ref_model


def _instantiate_reference_clone(base_train: Any, accelerator: Any) -> Any:
    model_cls = base_train.__class__
    config = copy.deepcopy(getattr(base_train, "config", None))
    if config is None:
        raise RuntimeError("Cannot clone reference model without a base configuration.")
    ds_plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    zero3_ctx = getattr(ds_plugin, "zero3_init_context_manager", None)
    context = zero3_ctx(enable=False) if callable(zero3_ctx) else nullcontext()
    with context:
        _log_stage("Reference model: instantiating new copy...", accelerator)
        return model_cls(config)


def _load_reference_weights_from_accelerator(
    accelerator: Any,
    train_model: Any,
    ref_model: Any,
) -> bool:
    try:
        _log_stage("Reference model: fetching full state_dict from accelerator...", accelerator)
        state_dict = _fetch_reference_state_dict(accelerator, train_model)
        _log_stage("Reference model: state_dict fetched; loading weights...", accelerator)
        missing_keys, unexpected_keys = ref_model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            LOG.warning(
                "Reference model load_state_dict mismatches (missing=%d unexpected=%d)",
                len(missing_keys),
                len(unexpected_keys),
            )
        del state_dict
        _freeze_model_parameters(ref_model)
        return True
    except (RuntimeError, ValueError, OSError):
        LOG.warning(
            "Reference model: failed to load full state_dict; falling back to "
            "embedding hydration only.",
            exc_info=True,
        )
        _freeze_model_parameters(ref_model)
        return False


def _freeze_model_parameters(model: Any) -> None:
    if hasattr(model, "eval"):
        model.eval()
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for param in parameters():
            if hasattr(param, "requires_grad_"):
                param.requires_grad_(False)


def _sync_reference_model(accelerator: Any) -> None:
    """Barrier to ensure reference initialization finishes on all ranks."""
    wait_for_everyone = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait_for_everyone):
        wait_for_everyone()


def prepare_training_data(
    config: DataPrepConfig,
    env_config: Optional[EnvironmentConfig] = None,
) -> TrainDataBundle:
    """Load datasets, map prompts, and construct the dataloader."""
    columns = _resolve_dataset_columns(config.context.script_args)
    _log_stage(
        (
            "Loading training dataset %s (split=%s, workers=%d)"
            % (
                getattr(config.context.script_args, "dataset_name", "unknown"),
                getattr(config.context.script_args, "dataset_split", "train"),
                int(config.num_workers),
            )
        ),
        accelerator=config.accelerator,
    )
    train_dataset = _map_training_dataset(config, columns)
    prompt_length_cache_get = _build_prompt_length_cache(
        config.context.tokenizer, config.max_prompt_len
    )
    env = env_config or EnvironmentConfig.capture()
    sampler = _build_distributed_sampler(train_dataset, config, env)
    persistent_workers = bool(config.persistent_workers and config.num_workers > 0)
    train_loader = config.data_loader_cls(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        collate_fn=_collate_rows,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        drop_last=bool(config.drop_last),
        persistent_workers=persistent_workers,
        sampler=sampler,
    )
    steps_per_epoch = _infer_steps_per_epoch(train_loader)
    try:
        raw_len = len(train_loader)
    except (TypeError, AttributeError):
        raw_len = "unknown"
    _log_stage(
        f"Dataloader length (pre-accelerate)={raw_len}",
        accelerator=config.accelerator,
    )
    dataset_size = getattr(train_dataset, "num_rows", None)
    if dataset_size is None:
        try:
            dataset_size = len(train_dataset)  # type: ignore[arg-type]
        except (TypeError, AttributeError):  # pragma: no cover - defensive
            dataset_size = None
    _log_stage(
        (
            "Training dataloader ready | rows=%s | batch_size=%d | steps_per_epoch=%s"
            % (
                str(dataset_size) if dataset_size is not None else "unknown",
                config.batch_size,
                str(steps_per_epoch) if steps_per_epoch is not None else "unknown",
            )
        ),
        accelerator=config.accelerator,
    )
    prompt_io = PromptIOConfig(
        prompt_column=columns.prompt,
        solution_column=columns.solution,
        prompt_length_cache_get=prompt_length_cache_get,
    )
    return TrainDataBundle(
        train_dataset=train_dataset,
        train_loader=train_loader,
        train_sampler=sampler,
        prompt_io=prompt_io,
        steps_per_epoch=steps_per_epoch,
        batch_size=config.batch_size,
    )


def prepare_with_accelerator(
    accelerator: Any,
    model_bundle: ModelBundle,
    optimizer: Any,
    train_data: TrainDataBundle,
) -> Tuple[ModelBundle, Any, TrainDataBundle]:
    """Move the model/optimizer/dataloader onto the accelerator stack."""
    pre_steps = _infer_steps_per_epoch(train_data.train_loader)
    _log_stage(
        (
            "Calling accelerator.prepare | pre_len=%s | split_batches_attr=%s | "
            "state.split_batches=%s"
            % (
                str(pre_steps) if pre_steps is not None else "unknown",
                getattr(accelerator, "split_batches", None),
                getattr(getattr(accelerator, "state", None), "split_batches", None),
            )
        ),
        accelerator,
    )
    model, optimizer, prepared_loader = accelerator.prepare(
        model_bundle.model,
        optimizer,
        train_data.train_loader,
    )
    model_bundle.model = model
    prepared_steps = _infer_steps_per_epoch(prepared_loader)
    # Guard against double-sharding: keep the pre-accelerate estimate when
    # Accelerate reports a smaller length (e.g., when split_batches=False and
    # the loader already uses a DistributedSampler).
    if (
        train_data.steps_per_epoch is not None
        and prepared_steps is not None
        and prepared_steps < train_data.steps_per_epoch
        and not getattr(accelerator, "split_batches", True)
    ):
        prepared_steps = train_data.steps_per_epoch
    if prepared_steps is None:
        prepared_steps = train_data.steps_per_epoch
    _log_stage(
        (
            "accelerator.prepare complete | pre_len=%s | post_len=%s | "
            "split_batches_attr=%s | world_size=%s"
            % (
                str(pre_steps) if pre_steps is not None else "unknown",
                str(prepared_steps) if prepared_steps is not None else "unknown",
                getattr(accelerator, "split_batches", None),
                getattr(getattr(accelerator, "state", None), "num_processes", None),
            )
        ),
        accelerator,
    )
    updated_loader = replace(
        train_data,
        train_loader=prepared_loader,
        steps_per_epoch=prepared_steps,
    )
    return model_bundle, optimizer, updated_loader


def _resolve_dataset_columns(script_args: GRPOScriptArguments) -> DatasetColumns:
    prompt = getattr(script_args, "dataset_prompt_column", "problem")
    solution = getattr(script_args, "dataset_solution_column", "answer")
    return DatasetColumns(prompt=prompt, solution=solution)


def _map_training_dataset(
    config: DataPrepConfig,
    columns: DatasetColumns,
) -> Any:
    raw_ds = get_dataset(config.context.script_args)
    char_limit = _prompt_char_limit_from_tokens(config.max_prompt_len)

    def _map_fn(example: Dict[str, Any]) -> Dict[str, str]:
        mapped = _to_prompt(
            example,
            config.context.tokenizer,
            columns.prompt,
            config.context.training_args.system_prompt,
            char_limit=char_limit,
        )
        mapped["answer"] = str(example.get(columns.solution, mapped.get("answer", "")))
        return mapped

    dataset = raw_ds.map(_map_fn)
    for split_name in list(dataset.keys()):
        if "messages" in dataset[split_name].column_names:
            dataset[split_name] = dataset[split_name].remove_columns("messages")
    train_split = getattr(config.context.script_args, "dataset_train_split", "train")
    return dataset[train_split]


def _build_prompt_length_cache(
    tokenizer: Any,
    max_prompt_len: int,
) -> Callable[[str], PromptCacheEntry]:
    prompt_cache: Dict[str, PromptCacheEntry] = {}
    char_limit = _prompt_char_limit_from_tokens(max_prompt_len)

    def _cache_get(prompt: str) -> PromptCacheEntry:
        normalized = _truncate_prompt(prompt, char_limit)
        cached = prompt_cache.get(normalized)
        if cached is not None:
            return cached
        encoded = tokenizer(
            normalized,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_len,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        entry = PromptCacheEntry(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        prompt_cache[normalized] = entry
        return entry

    return _cache_get


def _collate_rows(batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Convert dataset rows into the prompt/answer dict expected by DataLoader."""
    return {
        "prompt": [row["prompt"] for row in batch],
        "answer": [row.get("answer", "") for row in batch],
    }


def _build_distributed_sampler(
    dataset: Any,
    config: DataPrepConfig,
    env_config: EnvironmentConfig,
) -> Optional[Any]:
    """Create a DistributedSampler mirroring TRL's Trainer behavior."""
    # Allow opting out when you want each rank to see the full dataset.
    training_args = getattr(config.context, "training_args", None)
    if env_config.get_bool("MAXENT_DISABLE_DISTRIBUTED_SAMPLER") or bool(
        getattr(training_args, "disable_distributed_sampler", False)
    ):
        _log_stage(
            "Distributed sampler disabled; each rank will see the full dataset.",
            config.accelerator,
        )
        return None
    torch_module = getattr(config, "torch_module", None)
    accelerator = getattr(config, "accelerator", None)
    if torch_module is None or accelerator is None:
        return None
    utils_mod = getattr(torch_module, "utils", None)
    data_mod = getattr(utils_mod, "data", None) if utils_mod is not None else None
    dist_mod = getattr(data_mod, "distributed", None) if data_mod is not None else None
    sampler_cls = getattr(dist_mod, "DistributedSampler", None)
    if sampler_cls is None:
        return None
    world_size = getattr(accelerator, "num_processes", None)
    if world_size is None:
        world_size = getattr(getattr(accelerator, "state", None), "num_processes", None)
    if not world_size or world_size <= 1:
        return None
    rank = getattr(accelerator, "process_index", None)
    if rank is None:
        rank = getattr(getattr(accelerator, "state", None), "process_index", 0)
    seed = getattr(config.context.training_args, "seed", None)
    try:
        seed_val = int(seed)
    except (TypeError, ValueError):
        seed_val = 0
    sampler = sampler_cls(
        dataset,
        num_replicas=int(world_size),
        rank=int(rank or 0),
        shuffle=True,
        seed=seed_val,
        drop_last=bool(config.drop_last),
    )
    _log_stage(
        (
            "Distributed sampler enabled | world_size=%s | rank=%s | drop_last=%s"
            % (world_size, rank, bool(config.drop_last))
        ),
        config.accelerator,
    )
    return sampler


def _infer_steps_per_epoch(loader: Any) -> Optional[int]:
    """Best-effort estimation of how many steps an epoch contains."""
    try:
        return int(len(loader))
    except TypeError:
        return None


def build_reward_spec(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    tokenizer: Any,
) -> RewardSpec:
    """Resolve reward functions and default weights from the CLI arguments."""
    reward_funcs = get_reward_funcs(script_args, None, tokenizer)
    reward_weights = getattr(training_args, "reward_weights", None)
    if reward_weights is None or len(reward_weights) != len(reward_funcs):
        reward_weights = [1.0 for _ in range(len(reward_funcs))]
    if not reward_funcs:
        raise RuntimeError("No reward functions resolved for MaxEnt-GRPO training.")
    return RewardSpec(reward_funcs=reward_funcs, reward_weights=reward_weights)


def resolve_generation_config(training_args: GRPOConfig) -> GenerationConfig:
    """Read inference-related knobs from the training arguments."""

    def _coalesce_setting(attr_name: str, env_name: str) -> Optional[str]:
        attr_val = getattr(training_args, attr_name, None)
        if attr_val is None and attr_name == "vllm_timeout":
            # Backward-compatible alias used in some configs.
            attr_val = getattr(training_args, "vllm_request_timeout", None)
        if attr_val is not None and attr_val != "":
            return attr_val
        env_val = os.environ.get(env_name)
        return env_val if env_val not in {None, ""} else None

    def _maybe_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _maybe_bool(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _maybe_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_stop_sequences(raw_value: Optional[str]) -> Optional[List[str]]:
        if raw_value is None:
            return None
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [str(entry) for entry in parsed if entry is not None]
        except (json.JSONDecodeError, TypeError):
            pass
        return [segment.strip() for segment in str(raw_value).split("||") if segment.strip()]

    def _parse_logit_bias(raw_value: Optional[str]) -> Optional[Dict[str, float]]:
        if raw_value is None:
            return None
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        bias: Dict[str, float] = {}
        for key, val in parsed.items():
            try:
                bias[str(key)] = float(val)
            except (TypeError, ValueError):
                continue
        return bias or None

    max_prompt_len = int(getattr(training_args, "max_prompt_length", 512))
    max_completion_len = int(getattr(training_args, "max_completion_length", 256))
    num_generations = int(getattr(training_args, "num_generations", 4))
    gen_temperature = float(getattr(training_args, "gen_temperature", 0.8))
    gen_top_p = float(getattr(training_args, "gen_top_p", 0.9))
    use_vllm = bool(getattr(training_args, "use_vllm", False))

    vllm_url = str(
        getattr(
            training_args,
            "vllm_url",
            os.environ.get("VLLM_URL", "http://localhost:8000/generate"),
        )
    )
    vllm_rounds_cfg = int(
        getattr(training_args, "vllm_max_completion_rounds", 0) or 0
    )
    vllm_retry_sleep = max(
        0.0,
        float(getattr(training_args, "vllm_retry_sleep", 0.5)),
    )
    vllm_backfill_local = bool(
        getattr(training_args, "vllm_backfill_with_model", True)
    )
    vllm_request_logprobs = bool(
        getattr(training_args, "vllm_return_logprobs", True)
    )
    vllm_best_of = _maybe_int(_coalesce_setting("vllm_best_of", "VLLM_BEST_OF"))
    vllm_frequency_penalty = float(
        _maybe_float(
            _coalesce_setting("vllm_frequency_penalty", "VLLM_FREQUENCY_PENALTY"),
            0.0,
        )
    )
    vllm_presence_penalty = float(
        _maybe_float(
            _coalesce_setting("vllm_presence_penalty", "VLLM_PRESENCE_PENALTY"),
            0.0,
        )
    )
    vllm_top_k = _maybe_int(_coalesce_setting("vllm_top_k", "VLLM_TOP_K"))
    vllm_stop_sequences = _parse_stop_sequences(
        _coalesce_setting("vllm_stop_sequences", "VLLM_STOP")
    )
    vllm_timeout = float(
        _maybe_float(_coalesce_setting("vllm_timeout", "VLLM_TIMEOUT"), 120.0)
    )
    vllm_max_retries = int(
        _maybe_int(_coalesce_setting("vllm_max_retries", "VLLM_MAX_RETRIES")) or 3
    )
    vllm_backoff = float(
        _maybe_float(_coalesce_setting("vllm_backoff", "VLLM_BACKOFF"), 1.0)
    )
    vllm_guided_json = _coalesce_setting("vllm_guided_json", "VLLM_GUIDED_JSON")
    vllm_guided_regex = _coalesce_setting("vllm_guided_regex", "VLLM_GUIDED_REGEX")
    vllm_logit_bias = _parse_logit_bias(
        _coalesce_setting("vllm_logit_bias", "VLLM_LOGIT_BIAS")
    )
    vllm_request_id_prefix = _coalesce_setting(
        "vllm_request_id_prefix",
        "VLLM_REQUEST_ID_PREFIX",
    )
    vllm_sync_weights = _maybe_bool(
        _coalesce_setting("vllm_sync_weights", "VLLM_SYNC_WEIGHTS"),
        default=False,
    )
    client_cfg = VLLMClientConfig(
        url=vllm_url,
        rounds_cfg=vllm_rounds_cfg,
        retry_sleep=vllm_retry_sleep,
        backfill_local=vllm_backfill_local,
        request_logprobs=vllm_request_logprobs,
        best_of=vllm_best_of,
        frequency_penalty=vllm_frequency_penalty,
        presence_penalty=vllm_presence_penalty,
        top_k=vllm_top_k,
        stop_sequences=vllm_stop_sequences,
        timeout=vllm_timeout,
        max_retries=vllm_max_retries,
        backoff=vllm_backoff,
        guided_json=vllm_guided_json,
        guided_regex=vllm_guided_regex,
        logit_bias=vllm_logit_bias,
        request_id_prefix=vllm_request_id_prefix,
        sync_weights=vllm_sync_weights,
    )
    return GenerationConfig(
        lengths=LengthSettings(prompt=max_prompt_len, completion=max_completion_len),
        sampling=SamplingParams(
            num_generations=num_generations,
            temperature=gen_temperature,
            top_p=gen_top_p,
            top_k=vllm_top_k,
            penalties=SamplingPenalties(
                frequency=vllm_frequency_penalty,
                presence=vllm_presence_penalty,
            ),
            stop_config=SamplingStopConfig(
                stop_sequences=vllm_stop_sequences,
                best_of=vllm_best_of,
            ),
        ),
        use_vllm=use_vllm,
        vllm=client_cfg,
    )


def resolve_training_hyperparams(training_args: GRPOConfig) -> TrainingHyperParams:
    """Extract core hyperparameters (batch size, epochs, etc.) from the args."""
    batch_size = int(training_args.per_device_train_batch_size)
    if batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be a positive integer.")
    num_epochs = int(training_args.num_train_epochs)
    if num_epochs <= 0:
        raise ValueError("num_train_epochs must be a positive integer.")
    grad_accum_steps = int(training_args.gradient_accumulation_steps)
    if grad_accum_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer.")
    learning_rate = float(training_args.learning_rate)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    max_grad_norm = float(training_args.max_grad_norm)
    if max_grad_norm < 0.0:
        raise ValueError("max_grad_norm must be non-negative.")
    return TrainingHyperParams(
        batch_size=batch_size,
        num_epochs=num_epochs,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
    )


def build_generation_settings(cfg: GenerationConfig) -> GenerationSettings:
    """Convert a resolved ``GenerationConfig`` into trainer settings."""
    generation_stats: Dict[str, int] = {
        "vllm_retry_rounds": 0,
        "vllm_backfilled_prompts": 0,
        "vllm_failed_prompts": 0,
        "dropped_prompts": 0,
        "partial_prompts": 0,
        "vllm_excess_prompts": 0,
        "vllm_excess_completions": 0,
        "vllm_latency_calls": 0,
        "vllm_latency_total_ms": 0.0,
        "vllm_last_latency_ms": 0.0,
        "vllm_weight_syncs": 0,
    }
    penalty_cfg = GenerationPenaltyConfig(
        gen_top_k=cfg.sampling.top_k,
        gen_best_of=cfg.sampling.best_of,
        gen_frequency_penalty=cfg.sampling.frequency_penalty,
        gen_presence_penalty=cfg.sampling.presence_penalty,
        gen_stop_sequences=cfg.sampling.stop_sequences,
    )
    return GenerationSettings(
        generation_stats=generation_stats,
        penalty=penalty_cfg,
        max_prompt_len=cfg.lengths.prompt,
        max_completion_len=cfg.lengths.completion,
        gen_temperature=cfg.sampling.temperature,
        gen_top_p=cfg.sampling.top_p,
        use_vllm=cfg.use_vllm,
        vllm=cfg.vllm,
    )


def build_evaluation_settings(config: EvaluationConfig) -> EvaluationSettings:
    """Construct EvaluationSettings by optionally processing a dataset."""
    script_args = config.context.script_args
    training_args = config.context.training_args
    eval_dataset_name = getattr(script_args, "eval_dataset_name", None)
    eval_prompt_col = getattr(
        script_args, "eval_dataset_prompt_column", None
    ) or config.columns.prompt
    eval_solution_col = getattr(
        script_args, "eval_dataset_solution_column", None
    ) or config.columns.solution
    eval_split = getattr(script_args, "eval_dataset_split", "validation")
    eval_enabled = bool(getattr(training_args, "do_eval", False))
    eval_every = int(getattr(training_args, "eval_steps", 0))
    eval_every = eval_every if eval_every > 0 else None
    eval_bsz = int(
        getattr(
            training_args,
            "per_device_eval_batch_size",
            config.default_batch_size,
        )
    )
    eval_rows: List[Dict[str, str]] = []

    if eval_enabled and eval_dataset_name:
        eval_rows = _load_eval_rows(
            config,
            eval_dataset_name,
            eval_prompt_col,
            eval_solution_col,
            eval_split,
        )
        if not eval_rows:
            eval_enabled = False
        else:
            # Downsample to a consistent 10% subset using the training seed for repeatability.
            seed = getattr(training_args, "seed", 0) or 0
            rng = random.Random(seed)
            rng.shuffle(eval_rows)
            original_len = len(eval_rows)
            subset_size = max(1, int(math.ceil(original_len * 0.1)))
            eval_rows = eval_rows[:subset_size]
            LOG.info(
                "Eval rows downsampled to %d/%d using seed=%d",
                len(eval_rows),
                original_len,
                seed,
            )
    else:
        eval_enabled = False

    return EvaluationSettings(
        enabled=eval_enabled,
        rows=eval_rows,
        batch_size=eval_bsz,
        every_n_steps=eval_every,
    )


def _load_eval_rows(
    config: EvaluationConfig,
    dataset_name: str,
    prompt_column: str,
    solution_column: str,
    split: str,
) -> List[Dict[str, str]]:
    """Load and normalize the evaluation dataset into prompt/answer rows."""
    eval_raw = load_dataset_split(
        dataset_name,
        getattr(config.context.script_args, "eval_dataset_config", None),
        split,
    )
    char_limit = _prompt_char_limit_from_tokens(
        int(getattr(config.context.training_args, "max_prompt_length", 0))
    )

    def _map_eval_fn(example: Dict[str, Any]) -> Dict[str, str]:
        mapped = _to_prompt(
            example,
            config.context.tokenizer,
            prompt_column,
            config.context.training_args.system_prompt,
            char_limit=char_limit,
        )
        mapped["answer"] = str(
            example.get(solution_column, mapped.get("answer", ""))
        )
        return mapped

    eval_processed = eval_raw.map(_map_eval_fn)
    if "messages" in eval_processed.column_names:
        eval_processed = eval_processed.remove_columns("messages")
    return eval_processed.to_list()


def _read_bool_env(name: str, default: str = "1") -> bool:
    """Interpret environment flags that may use string-based booleans."""
    value = os.environ.get(name, default)
    return value not in {"0", "false", "False"}


def _resolve_beta_and_tau(training_args: GRPOConfig) -> Tuple[float, float]:
    """Extract initial beta/tau coefficients."""

    def _first_configured_beta() -> Optional[float]:
        """Return the first non-None beta-style attribute that was set."""

        for attr in ("init_kl_coeff", "init_kl_coef", "beta"):
            value = getattr(training_args, attr, None)
            if value is not None:
                return float(value)
        return None

    beta_attr = _first_configured_beta()
    beta = float(beta_attr) if beta_attr is not None else 0.0
    tau = float(getattr(training_args, "maxent_tau", os.environ.get("MAXENT_TAU", 0.0)))
    return beta, tau


def _resolve_q_distribution(training_args: GRPOConfig) -> QDistributionSettings:
    """Return temperature and epsilon for the MaxEnt softmax."""
    q_temp = float(
        getattr(
            training_args,
            "maxent_q_temperature",
            os.environ.get("MAXENT_Q_TEMPERATURE", 1.0),
        )
    )
    q_eps = float(
        getattr(
            training_args,
            "maxent_q_epsilon",
            os.environ.get("MAXENT_Q_EPS", 1e-6),
        )
    )
    if bool(getattr(training_args, "train_grpo_objective", False)):
        # When training the GRPO objective, ignore MaxEnt-specific q shaping.
        q_temp = 1.0
        q_eps = 1e-8
    return QDistributionSettings(temperature=q_temp, epsilon=q_eps)


def _resolve_len_norm_flag(training_args: GRPOConfig) -> bool:
    """Determine whether to length-normalize reference log-probs."""
    len_norm_ref_attr = getattr(training_args, "maxent_length_normalize_ref", None)
    if len_norm_ref_attr is None:
        return _read_bool_env("MAXENT_LENGTH_NORM_REF")
    return bool(len_norm_ref_attr)


def _resolve_kl_controller(training_args: GRPOConfig) -> KlControllerSettings:
    """Parse KL-controller settings from the training args."""
    kl_target = float(getattr(training_args, "kl_target", 0.0) or 0.0)
    kl_horizon = int(getattr(training_args, "kl_horizon", 0) or 0)
    kl_ctl_step_size = float(getattr(training_args, "kl_ctl_step_size", 0.0) or 0.0)
    return KlControllerSettings(
        target=kl_target,
        horizon=kl_horizon,
        step_size=kl_ctl_step_size,
    )


def _resolve_tau_schedule(training_args: GRPOConfig) -> TauSchedule:
    """Convert tau adaptation hyperparameters into a structured config."""
    tau_target_entropy_attr = getattr(training_args, "maxent_target_weight_entropy", None)
    tau_target_entropy = (
        float(tau_target_entropy_attr) if tau_target_entropy_attr is not None else None
    )
    tau_lr = float(getattr(training_args, "maxent_tau_lr", 0.0))
    tau_min = float(getattr(training_args, "maxent_tau_min", 0.0))
    tau_max = float(getattr(training_args, "maxent_tau_max", 0.0))
    tau_warmup_cfg = int(getattr(training_args, "maxent_tau_warmup_steps", -1))
    if tau_warmup_cfg < 0:
        tau_warmup_cfg = int(getattr(training_args, "warmup_steps", 0) or 0)
    return TauSchedule(
        target_entropy=tau_target_entropy,
        learning_rate=tau_lr,
        minimum_value=tau_min,
        maximum_value=tau_max,
        warmup_steps=max(0, tau_warmup_cfg),
    )


def _resolve_weighting_settings(training_args: GRPOConfig) -> WeightingSettings:
    """Return weighting hyperparameters derived from training args/environment."""
    beta, tau = _resolve_beta_and_tau(training_args)
    q_dist = _resolve_q_distribution(training_args)
    len_norm_ref = _resolve_len_norm_flag(training_args)
    kl_controller = _resolve_kl_controller(training_args)
    tau_schedule = _resolve_tau_schedule(training_args)
    train_grpo_objective = bool(getattr(training_args, "train_grpo_objective", False))
    denom = (
        beta if (train_grpo_objective and beta > 0.0) else (tau + beta if (tau + beta) > 0 else 1.0)
    )
    normalization = WeightNormalizationSettings(denom=denom, len_norm_ref=len_norm_ref)
    return WeightingSettings(
        tau=tau,
        beta=beta,
        normalization=normalization,
        q_distribution=q_dist,
        tau_schedule=tau_schedule,
        kl_controller=kl_controller,
        train_grpo_objective=train_grpo_objective,
    )


def _resolve_clip_settings(training_args: GRPOConfig) -> ClipSettings:
    """Build PPO-style clipping settings based on training args/environment."""
    clip_range_attr = getattr(training_args, "ppo_clip_range", None)
    if clip_range_attr is None:
        clip_range_attr = getattr(training_args, "clip_range", None)
    if clip_range_attr is None:
        clip_range_attr = getattr(training_args, "maxent_clip_range", None)
    clip_range = abs(float(clip_range_attr if clip_range_attr is not None else 0.2))
    use_clip_objective = bool(
        getattr(
            training_args,
            "maxent_use_clip_objective",
            getattr(training_args, "use_clip_objective", False),
        )
    )
    clip_objective_coef = float(
        getattr(training_args, "maxent_clip_objective_coef", 1.0)
    )
    clip_adv_baseline = getattr(training_args, "maxent_clip_adv_baseline", None)
    clip_adv_baseline = (
        float(clip_adv_baseline) if clip_adv_baseline is not None else None
    )
    return ClipSettings(
        clip_range=clip_range,
        use_clip_objective=use_clip_objective,
        clip_objective_coef=clip_objective_coef,
        clip_adv_baseline=clip_adv_baseline,
    )


def _resolve_batching_settings(
    training_args: GRPOConfig,
    prompt_length_cache_get: Callable[[str], PromptCacheEntry],
) -> BatchingSettings:
    """Return batching-related scoring settings."""
    logprob_chunk_size = int(
        getattr(training_args, "maxent_logprob_chunk_size", 0)
        or getattr(training_args, "logprob_chunk_size", 0)
        or int(os.environ.get("MAXENT_LOGPROB_CHUNK_SIZE", "0"))
    )
    score_slice_cfg = getattr(training_args, "maxent_score_slice", None)
    if score_slice_cfg is None:
        score_slice_cfg = os.environ.get("MAXENT_SCORE_SLICE", None)
    score_slice = int(score_slice_cfg) if score_slice_cfg not in (None, "") else 0
    score_slice = max(score_slice, 0)
    return BatchingSettings(
        logprob_chunk_size=logprob_chunk_size,
        score_slice=score_slice,
        prompt_length_cache_get=prompt_length_cache_get,
    )


def build_scoring_settings(
    training_args: GRPOConfig,
    prompt_length_cache_get: Callable[[str], PromptCacheEntry],
) -> ScoringSettings:
    """Generate weighting, clipping, and batching settings for scoring."""
    weighting = _resolve_weighting_settings(training_args)
    clipping = _resolve_clip_settings(training_args)
    batching = _resolve_batching_settings(training_args, prompt_length_cache_get)
    return ScoringSettings(weighting=weighting, clipping=clipping, batching=batching)


def _normalize_save_strategy(strategy: Any, default: str = "no") -> str:
    """Return a lowercase save strategy string from enums/strings."""
    if strategy is None:
        return default
    value = getattr(strategy, "value", strategy)
    text = str(value).strip().lower()
    # Transformers enums stringify to e.g. "savestrategy.steps" or "intervalstrategy.steps".
    for prefix in ("savestrategy.", "intervalstrategy."):
        if text.startswith(prefix):
            return text.split(".", 1)[1]
    return text or default


def build_checkpoint_config(training_args: GRPOConfig) -> CheckpointConfig:
    """Generate checkpoint preferences (frequency, pruning, and Hub pushes)."""
    output_dir = getattr(training_args, "output_dir", "./maxent-grpo-out")
    os.makedirs(output_dir, exist_ok=True)
    save_strategy = _normalize_save_strategy(getattr(training_args, "save_strategy", "no"))
    save_steps = int(getattr(training_args, "save_steps", 0) or 0)
    save_total_limit = int(getattr(training_args, "save_total_limit", 0) or 0)
    push_to_hub = bool(getattr(training_args, "push_to_hub", False))
    hub_model_id = getattr(training_args, "hub_model_id", None)
    hub_token = getattr(training_args, "hub_token", None)
    return CheckpointConfig(
        output_dir=output_dir,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        hub=HubPushConfig(
            enabled=push_to_hub,
            model_id=hub_model_id,
            token=hub_token,
        ),
    )


def _estimate_total_steps(schedule_cfg: TrainingScheduleConfig) -> Optional[int]:
    """Estimate the total optimizer steps implied by the schedule."""
    if schedule_cfg.steps_per_epoch is None:
        return None
    estimated_micro_steps = schedule_cfg.steps_per_epoch * schedule_cfg.num_epochs
    return math.ceil(estimated_micro_steps / schedule_cfg.grad_accum_steps)


def _resolve_warmup_steps(
    total_steps: int, configured_steps: int, warmup_ratio: float
) -> int:
    """Determine how many warmup steps to use given the schedule constraints."""
    if configured_steps > 0 or total_steps <= 0:
        return min(configured_steps, total_steps) if total_steps > 0 else configured_steps
    if warmup_ratio <= 0.0 or total_steps <= 0:
        return 0
    computed = max(1, int(total_steps * warmup_ratio))
    return min(computed, total_steps)


def _resolve_total_training_steps(
    training_args: GRPOConfig, schedule_cfg: TrainingScheduleConfig
) -> int:
    """Compute the total number of optimizer steps for the run."""
    max_steps_cfg = int(getattr(training_args, "max_steps", 0) or 0)
    estimated_total_steps = _estimate_total_steps(schedule_cfg)
    return max_steps_cfg if max_steps_cfg > 0 else (estimated_total_steps or 0)


def _maybe_build_scheduler(
    optimizer_ctx: OptimizerContext,
    base_optimizer: Any,
    lr_warmup_steps: int,
    total_training_steps: int,
    learning_rate: float,
) -> Optional[Any]:
    """Construct a learning-rate scheduler when total steps are known."""
    if total_training_steps <= 0:
        return None
    training_args = optimizer_ctx.training_args
    transformers_module = optimizer_ctx.transformers_module
    scheduler_name = str(getattr(training_args, "lr_scheduler_type", "cosine")).lower()
    try:
        scheduler = transformers_module.get_scheduler(
            name=scheduler_name,
            optimizer=base_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=total_training_steps,
        )
    except (RuntimeError, ValueError):
        return None
    if lr_warmup_steps > 0:
        for group in base_optimizer.param_groups:  # type: ignore[attr-defined]
            group.setdefault("initial_lr", learning_rate)
            group["lr"] = 0.0
    return scheduler


def build_optimization_settings(
    optimizer_ctx: OptimizerContext,
    schedule_cfg: TrainingScheduleConfig,
    learning_cfg: LearningConfig,
) -> tuple[OptimizationSettings, int]:
    """Create the OptimizationSettings dataclass and compute LR warmup steps."""
    training_args = optimizer_ctx.training_args
    total_training_steps = _resolve_total_training_steps(training_args, schedule_cfg)
    configured_warmup = int(getattr(training_args, "warmup_steps", 0) or 0)
    warmup_ratio = float(getattr(training_args, "warmup_ratio", 0.0) or 0.0)
    lr_warmup_steps = _resolve_warmup_steps(
        total_training_steps, configured_warmup, warmup_ratio
    )

    base_optimizer = getattr(optimizer_ctx.optimizer, "optimizer", optimizer_ctx.optimizer)
    lr_scheduler = _maybe_build_scheduler(
        optimizer_ctx,
        base_optimizer,
        lr_warmup_steps,
        total_training_steps,
        learning_cfg.learning_rate,
    )

    schedule = OptimizationSchedule(
        num_epochs=schedule_cfg.num_epochs,
        num_generations=schedule_cfg.num_generations,
        grad_accum_steps=schedule_cfg.grad_accum_steps,
        max_grad_norm=learning_cfg.max_grad_norm,
        steps_per_epoch=schedule_cfg.steps_per_epoch,
        total_training_steps=total_training_steps,
        warmup_steps=lr_warmup_steps,
    )
    handles = OptimizerHandles(
        optimizer=optimizer_ctx.optimizer,
        lr_scheduler=lr_scheduler,
        base_optimizer=base_optimizer,
        learning_rate=learning_cfg.learning_rate,
    )
    return OptimizationSettings(schedule=schedule, handles=handles), lr_warmup_steps


def build_wandb_config(
    *,
    model_args: Any,
    script_args: GRPOScriptArguments,
    hyperparams: TrainingHyperParams,
    generation_cfg: GenerationConfig,
    scoring_settings: ScoringSettings,
    optimization_settings: OptimizationSettings,
    lr_warmup_steps: int,
    training_args: GRPOConfig,
) -> Dict[str, Any]:
    """Compose the base Weights & Biases configuration for logging."""
    return {
        "script": "maxent-grpo",
        "model_name_or_path": getattr(model_args, "model_name_or_path", None),
        "dataset_name": getattr(script_args, "dataset_name", None),
        "per_device_train_batch_size": hyperparams.batch_size,
        "learning_rate": hyperparams.learning_rate,
        "num_generations": generation_cfg.sampling.num_generations,
        "max_prompt_length": generation_cfg.lengths.prompt,
        "max_completion_length": generation_cfg.lengths.completion,
        "gradient_accumulation_steps": hyperparams.grad_accum_steps,
        "tau": scoring_settings.weighting.tau,
        "beta": scoring_settings.weighting.beta,
        "q_temperature": scoring_settings.weighting.q_temperature,
        "q_epsilon": scoring_settings.weighting.q_epsilon,
        "length_norm_ref": scoring_settings.weighting.len_norm_ref,
        "kl_target": scoring_settings.weighting.kl_target,
        "kl_horizon": scoring_settings.weighting.kl_horizon,
        "kl_ctl_step_size": scoring_settings.weighting.kl_ctl_step_size,
        "use_vllm": generation_cfg.use_vllm,
        "seed": getattr(training_args, "seed", None),
        "lr_warmup_steps": lr_warmup_steps,
        "max_steps": optimization_settings.schedule.total_training_steps,
    }


def bootstrap_runner(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: Any,
) -> RunnerSetup:
    """Prepare frameworks, accelerator, model bundle, and training data."""
    frameworks = resolve_frameworks()
    env_config = EnvironmentConfig.capture()
    accelerator = create_accelerator(frameworks, env_config)
    _log_stage(
        f"Accelerator initialized (device={getattr(accelerator, 'device', 'unknown')})",
        accelerator,
    )
    seed_everything(training_args, frameworks.transformers)
    configure_logging(training_args, frameworks.transformers)

    hyperparams = resolve_training_hyperparams(training_args)
    generation_cfg = resolve_generation_config(training_args)
    checkpoint_cfg = build_checkpoint_config(training_args)

    _log_stage("Loading model and tokenizer...", accelerator)
    model_bundle = prepare_model_bundle(model_args, training_args, accelerator)
    _log_stage("Model and tokenizer ready.", accelerator)
    compiled_model = _maybe_compile_model(
        frameworks,
        accelerator,
        model_bundle.model,
        training_args,
        env_config,
    )
    if compiled_model:
        model_bundle.model = compiled_model
        _log_stage("torch.compile applied to trainable model.", accelerator)
    loader_runtime = DataLoaderRuntimeOptions(
        torch_module=frameworks.torch,
        accelerator=accelerator,
        num_workers=int(getattr(training_args, "dataloader_num_workers", 0) or 0),
        pin_memory=bool(getattr(training_args, "dataloader_pin_memory", True)),
        drop_last=bool(getattr(training_args, "dataloader_drop_last", False)),
        persistent_workers=bool(
            getattr(training_args, "dataloader_persistent_workers", False)
        ),
    )
    data_config = DataPrepConfig(
        context=DatasetContext(script_args, training_args, model_bundle.tokenizer),
        batch_size=hyperparams.batch_size,
        max_prompt_len=generation_cfg.lengths.prompt,
        data_loader_cls=frameworks.data_loader_cls,
        runtime=loader_runtime,
    )
    train_data = prepare_training_data(data_config, env_config)

    optimizer = frameworks.torch.optim.AdamW(
        model_bundle.model.parameters(),
        lr=hyperparams.learning_rate,
        betas=(0.9, 0.99),
        eps=1e-6,
    )
    _log_stage("Wrapping model/optimizer/dataloader with accelerator...", accelerator)
    model_bundle, optimizer, train_data = prepare_with_accelerator(
        accelerator,
        model_bundle,
        optimizer,
        train_data,
    )
    _maybe_prepare_with_trl(accelerator, model_bundle.model)
    _log_stage("Accelerator prepare() complete.", accelerator)
    _log_stage("Instantiating frozen reference model...", accelerator)
    _initialize_reference_model(
        accelerator,
        model_bundle,
        model_args,
        training_args,
        env_config,
    )
    _sync_reference_model(accelerator)
    _log_stage("Reference model ready.", accelerator)

    runtime = RuntimeArtifacts(frameworks=frameworks, accelerator=accelerator)

    return RunnerSetup(
        runtime=runtime,
        hyperparams=hyperparams,
        generation=generation_cfg,
        checkpoint=checkpoint_cfg,
        model_bundle=model_bundle,
        train_data=train_data,
        optimizer=optimizer,
    )


def bootstrap_for_tests(
    frameworks: FrameworkHandles,
    env_config: Optional[EnvironmentConfig] = None,
) -> Tuple[Any, SimpleNamespace]:
    """Utility factory for unit tests to avoid sys.modules monkeypatching.

    Returns a tuple of (accelerator, helper namespace) where the helper exposes
    wrappers that already inject the provided frameworks/env config.
    """
    env = env_config or EnvironmentConfig.capture()
    accelerator = create_accelerator(frameworks, env)

    def _prepare_training_data(
        config: DataPrepConfig,
    ) -> TrainDataBundle:
        return prepare_training_data(config, env)

    def _init_reference(
        model_bundle: ModelBundle,
        model_args: Any,
        training_args: GRPOConfig,
    ) -> None:
        _initialize_reference_model(accelerator, model_bundle, model_args, training_args, env)

    helpers = SimpleNamespace(
        environment=env,
        accelerator=accelerator,
        prepare_training_data=_prepare_training_data,
        initialize_reference_model=_init_reference,
        prepare_model_bundle=prepare_model_bundle,
        create_accelerator=lambda: accelerator,
    )
    return accelerator, helpers


def build_training_components(
    setup: RunnerSetup,
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: Any,
) -> TrainingComponents:
    """Construct generation/scoring/eval/optimization components for training."""
    scoring_settings = build_scoring_settings(
        training_args, setup.train_data.prompt_length_cache_get
    )
    evaluation_settings = build_evaluation_settings(
        EvaluationConfig(
            context=DatasetContext(
                script_args, training_args, setup.model_bundle.tokenizer
            ),
            columns=DatasetColumns(
                prompt=setup.train_data.prompt_column,
                solution=setup.train_data.solution_column,
            ),
            default_batch_size=setup.train_data.batch_size,
        )
    )
    optimization_settings, lr_warmup_steps = build_optimization_settings(
        OptimizerContext(
            training_args=training_args,
            optimizer=setup.optimizer,
            transformers_module=setup.frameworks.transformers,
        ),
        TrainingScheduleConfig(
            num_epochs=setup.hyperparams.num_epochs,
            num_generations=setup.generation.sampling.num_generations,
            grad_accum_steps=setup.hyperparams.grad_accum_steps,
            steps_per_epoch=setup.train_data.steps_per_epoch,
        ),
        LearningConfig(
            learning_rate=setup.hyperparams.learning_rate,
            max_grad_norm=setup.hyperparams.max_grad_norm,
        ),
    )
    wandb_config = build_wandb_config(
        model_args=model_args,
        script_args=script_args,
        hyperparams=setup.hyperparams,
        generation_cfg=setup.generation,
        scoring_settings=scoring_settings,
        optimization_settings=optimization_settings,
        lr_warmup_steps=lr_warmup_steps,
        training_args=training_args,
    )
    reward_spec = build_reward_spec(
        script_args,
        training_args,
        setup.model_bundle.tokenizer,
    )
    generation_settings = build_generation_settings(setup.generation)

    return TrainingComponents(
        scoring=scoring_settings,
        evaluation=evaluation_settings,
        optimization=optimization_settings,
        generation=generation_settings,
        reward=reward_spec,
        wandb_config=wandb_config,
        lr_warmup_steps=lr_warmup_steps,
    )


__all__ = [
    "EnvironmentConfig",
    "bootstrap_runner",
    "build_checkpoint_config",
    "build_evaluation_settings",
    "build_generation_settings",
    "build_reward_spec",
    "build_scoring_settings",
    "build_training_components",
    "build_optimization_settings",
    "build_wandb_config",
    "configure_logging",
    "create_accelerator",
    "prepare_model_bundle",
    "prepare_training_data",
    "prepare_with_accelerator",
    "resolve_frameworks",
    "bootstrap_for_tests",
    "resolve_generation_config",
    "resolve_training_hyperparams",
    "seed_everything",
]
