"""
MaxEnt-GRPO training entrypoint that mirrors the baseline GRPO pipeline.

This keeps data loading, prompt construction, and trainer wiring aligned with
``pipelines.training.baseline`` while letting callers flip between vanilla GRPO
and entropy-weighted MaxEnt behavior via config toggles
(``training_args.train_grpo_objective`` and related ``maxent_*`` fields).
"""

from __future__ import annotations

import logging
from typing import Any

from contextlib import contextmanager
from typing import Type

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.pipelines.training.baseline import (
    GRPOTrainerOverride,
    get_peft_config_override,
    run_baseline_training as _run_baseline_training,
    _to_prompt,
    ChatTemplate,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "run_maxent_training",
    "GRPOTrainerOverride",
    "get_peft_config_override",
    "_to_prompt",
    "ChatTemplate",
]


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
            except Exception:
                patch_llm = None
                orig_llm = None

            try:
                super().__init__(*args, **kwargs)
            finally:
                # Restore the original LLM constructor to avoid side effects.
                try:
                    if patch_llm is not None and orig_llm is not None:
                        grpo_mod.LLM = orig_llm
                except Exception:
                    pass
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

    use_vanilla = bool(getattr(training_args, "train_grpo_objective", False))
    LOG.info(
        "Launching %s pipeline | train_grpo_objective=%s",
        "vanilla GRPO" if use_vanilla else "MaxEnt (entropy-weighted)",
        use_vanilla,
    )
    with _maybe_patch_trainer(training_args):
        return _run_baseline_training(script_args, training_args, model_args)
