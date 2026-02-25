"""
Training entrypoint that mirrors the baseline GRPO pipeline.

This keeps data loading, prompt construction, and trainer wiring aligned with
``pipelines.training.baseline`` while letting callers flip between:

* vanilla GRPO (``train_grpo_objective=true``),
* GRPO + policy-entropy reward bonus (``policy_entropy_bonus_coef>0``), and
* entropy-weighted MaxEnt (``train_grpo_objective=false``).

As of the Trainer refactor, this entrypoint always runs through the TRL/HF
Trainer loop (no custom loop). MaxEnt behavior is handled inside
``CustomGRPOTrainer``.
"""

from __future__ import annotations

import logging
from typing import Any

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.core.hub import ensure_hf_repo_ready
from maxent_grpo.pipelines.training.baseline import (
    GRPOTrainerOverride,
    get_peft_config_override,
    run_baseline_training as _run_baseline_training,
    _to_prompt,
    ChatTemplate,
)
from maxent_grpo.utils.deps_guard import ensure_real_dependencies

LOG = logging.getLogger(__name__)

__all__ = [
    "run_maxent_training",
    "GRPOTrainerOverride",
    "get_peft_config_override",
    "_to_prompt",
    "ChatTemplate",
]


def run_maxent_training(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: Any,
) -> None:
    """Run GRPO (vanilla or entropy-bonus) or MaxEnt training via the shared pipeline.

    The same prompt construction and dataset loader are reused from
    ``pipelines.training.baseline``. The ``train_grpo_objective`` flag controls
    whether the run behaves like GRPO (``True``) or entropy-weighted MaxEnt
    (``False``); when ``policy_entropy_bonus_coef>0`` and ``train_grpo_objective``
    is ``True`` the run becomes GRPO + entropy bonus.

    :param script_args: Data/reward configuration mirroring the baseline pipeline.
    :type script_args: GRPOScriptArguments
    :param training_args: GRPO trainer options controlling MaxEnt toggles.
    :type training_args: GRPOConfig
    :param model_args: Model configuration consumed by TRL/transformers.
    :type model_args: Any
    :returns: ``None`` after delegating to the baseline run helper.
    :rtype: None
    """

    ensure_real_dependencies(context="GRPO/MaxEnt training")
    ensure_hf_repo_ready(training_args)

    if getattr(training_args, "controller_meta_enabled", False):
        LOG.info(
            "controller_meta_enabled is set; CustomGRPOTrainer will handle controller/meta updates."
        )
    return _run_baseline_training(script_args, training_args, model_args)
