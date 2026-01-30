"""
MaxEnt-GRPO training entrypoint that mirrors the baseline GRPO pipeline.

This keeps data loading, prompt construction, and trainer wiring aligned with
``pipelines.training.baseline`` while letting callers flip between vanilla GRPO
and entropy-weighted MaxEnt behavior via config toggles
(``training_args.train_grpo_objective`` and related ``maxent_*`` fields).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from contextlib import contextmanager
from typing import Type

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.core.hub import ensure_hf_repo_ready
from maxent_grpo.pipelines.training.baseline import (
    GRPOTrainerOverride,
    get_peft_config_override,
    run_baseline_training as _run_baseline_training,
    _to_prompt,
    ChatTemplate,
)
from maxent_grpo.telemetry.trl_logging import ensure_weighting_logging
from maxent_grpo.training import run_training_loop
from maxent_grpo.training.runtime.logging import log_run_header
from maxent_grpo.utils.deps_guard import ensure_real_dependencies
from .loop_common import build_training_loop_context

LOG = logging.getLogger(__name__)

__all__ = [
    "run_maxent_training",
    "GRPOTrainerOverride",
    "get_peft_config_override",
    "_to_prompt",
    "ChatTemplate",
]


def _configure_custom_loop_logging(training_args: GRPOConfig) -> None:
    """Set up Python/logging backends before launching the custom loop."""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    LOG.setLevel(log_level)
    log_run_header(training_args)
    try:
        import datasets as _hf_datasets

        _hf_datasets.utils.logging.set_verbosity(log_level)
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        LOG.debug("Skipping datasets logging setup: %s", exc)
    try:
        import transformers as _transformers
    except (ImportError, ModuleNotFoundError):
        _transformers = None
    tf_logging_module = getattr(
        getattr(_transformers, "utils", None), "logging", None
    )
    if tf_logging_module is not None:
        set_verbosity = getattr(tf_logging_module, "set_verbosity", None)
        if callable(set_verbosity):
            set_verbosity(log_level)
        enable_default_handler = getattr(
            tf_logging_module, "enable_default_handler", None
        )
        if callable(enable_default_handler):
            enable_default_handler()
        enable_explicit_format = getattr(
            tf_logging_module, "enable_explicit_format", None
        )
        if callable(enable_explicit_format):
            enable_explicit_format()


def _build_maxent_trainer(parent_cls: Type) -> Type:
    """Return a lightweight MaxEnt-aware GRPOTrainer subclass.

    :param parent_cls: Base GRPO trainer class to extend.
    :type parent_cls: type
    :returns: Concrete subclass that tags losses/metrics with MaxEnt metadata.
    :rtype: type
    """

    class MaxEntGRPOTrainer(parent_cls):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            # Force vLLM to honor fp16/bf16 flags even when the base model's
            # config defaults to bfloat16 (e.g., Qwen2.5). TRL's GRPOTrainer
            # does not forward a dtype to LLM(...), so colocated vLLM falls
            # back to the model config and can crash on GPUs without bf16.
            patch_llm = None
            orig_llm = None
            try:
                import trl.trainer.grpo_trainer as grpo_mod
                from vllm import LLM as _LLM

                # Args are always passed by name in our pipeline.
                t_args = kwargs.get("args")
                dtype_override = None
                if getattr(t_args, "fp16", False):
                    dtype_override = "float16"
                elif getattr(t_args, "bf16", False):
                    dtype_override = "bfloat16"
                if dtype_override and getattr(t_args, "use_vllm", False):
                    orig_llm = getattr(grpo_mod, "LLM", None)

                    def _patched_llm(*llm_args, **llm_kwargs):
                        llm_kwargs.setdefault("dtype", dtype_override)
                        return _LLM(*llm_args, **llm_kwargs)

                    patch_llm = _patched_llm
                    if orig_llm is not None:
                        grpo_mod.LLM = patch_llm
            except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
                patch_llm = None
                orig_llm = None
                LOG.debug("Unable to patch TRL LLM constructor: %s", exc)

            try:
                super().__init__(*args, **kwargs)
            finally:
                # Restore the original LLM constructor to avoid side effects.
                try:
                    if patch_llm is not None and orig_llm is not None:
                        grpo_mod.LLM = orig_llm
                except (AttributeError, RuntimeError, TypeError) as exc:
                    LOG.debug("Failed to restore TRL LLM constructor: %s", exc)
            # Mark the args so downstream hooks/metrics can key off it.
            if hasattr(self, "args"):
                setattr(self.args, "train_grpo_objective", False)
            self.maxent_enabled = True

        def compute_loss(self, *args, **kwargs):  # pragma: no cover - thin wrapper
            """Call parent compute_loss but tag outputs for MaxEnt bookkeeping."""
            loss = super().compute_loss(*args, **kwargs)
            if isinstance(loss, tuple):
                # (loss, metrics) tuple in some TRL versions
                _loss, metrics = loss
                if isinstance(metrics, dict):
                    metrics.setdefault("maxent_enabled", True)
                return (_loss, metrics)
            if hasattr(loss, "item"):
                setattr(loss, "maxent_enabled", True)
            return loss

    MaxEntGRPOTrainer = ensure_weighting_logging(MaxEntGRPOTrainer)
    MaxEntGRPOTrainer.__name__ = "MaxEntGRPOTrainer"
    return MaxEntGRPOTrainer


@contextmanager
def _maybe_patch_trainer(training_args: GRPOConfig):
    """Temporarily install a MaxEnt-aware trainer override when requested.

    :param training_args: Trainer configuration controlling GRPO vs MaxEnt mode.
    :type training_args: GRPOConfig
    :yields: Context where the baseline module exposes a MaxEnt-aware trainer
        unless ``train_grpo_objective`` is truthy.
    :rtype: collections.abc.Generator[None, None, None]
    """

    from maxent_grpo.pipelines.training import baseline as baseline_mod

    prev_override = baseline_mod.GRPOTrainerOverride
    if getattr(training_args, "train_grpo_objective", True):
        yield
        return
    try:
        from trl import GRPOTrainer as _TRLGRPOTrainer
    except (
        ImportError,
        RuntimeError,
        AttributeError,
    ):  # pragma: no cover - optional dependency
        LOG.warning(
            "MaxEnt trainer override requested but TRL not available; "
            "falling back to baseline GRPOTrainer."
        )
        yield
        return
    baseline_mod.GRPOTrainerOverride = _build_maxent_trainer(_TRLGRPOTrainer)
    try:
        yield
    finally:
        baseline_mod.GRPOTrainerOverride = prev_override


def run_maxent_training(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: Any,
) -> None:
    """Run MaxEnt/GRPO training using the shared baseline pipeline.

    The same prompt construction and dataset loader are reused from
    ``pipelines.training.baseline``. The ``train_grpo_objective`` flag controls
    whether the run behaves like vanilla GRPO (``True``) or entropy-weighted
    MaxEnt (``False``), keeping the codepath unified and testable.

    :param script_args: Data/reward configuration mirroring the baseline pipeline.
    :type script_args: GRPOScriptArguments
    :param training_args: GRPO trainer options controlling MaxEnt toggles.
    :type training_args: GRPOConfig
    :param model_args: Model configuration consumed by TRL/transformers.
    :type model_args: Any
    :returns: ``None`` after delegating to the baseline run helper.
    :rtype: None
    """

    ensure_real_dependencies(context="MaxEnt-GRPO training")
    ensure_hf_repo_ready(training_args)

    train_grpo_flag = bool(getattr(training_args, "train_grpo_objective", False))
    meta_enabled = bool(getattr(training_args, "controller_meta_enabled", False))
    force_custom_loop = bool(getattr(training_args, "force_custom_loop", False))
    use_custom_loop = force_custom_loop or (not train_grpo_flag) or meta_enabled
    if use_custom_loop:
        _configure_custom_loop_logging(training_args)
        LOG.info(
            "Launching custom training loop | objective=%s | controller_meta_enabled=%s | force_custom_loop=%s",
            "GRPO" if train_grpo_flag else "MaxEnt",
            meta_enabled,
            force_custom_loop,
        )
        # Accelerate 1.4.0+ guards against calling ``AcceleratorState`` before the
        # accelerator is initialized. Pre-warm the shared state so downstream
        # ``Accelerator()`` construction does not raise when used outside the CLI.
        ctx = build_training_loop_context(
            script_args,
            training_args,
            model_args,
            deps_namespace="maxent",
            apply_info_seed_cfg=True,
            force_grpo_objective=True if train_grpo_flag else None,
        )
        return run_training_loop(ctx)
    LOG.info(
        "Launching vanilla GRPO pipeline via TRL | controller_meta_enabled=%s",
        meta_enabled,
    )
    with _maybe_patch_trainer(training_args):
        return _run_baseline_training(script_args, training_args, model_args)
