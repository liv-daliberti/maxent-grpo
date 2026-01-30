# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop utilities for the MaxEnt-GRPO runner.

The functions in this module orchestrate the end-to-end training flow:

``run_training_loop``
    Entry point that constructs reusable helpers (generation, validation, step
    resources) and drives epoch iteration.
``_run_epoch``
    Consumes the PyTorch DataLoader to process each batch and stops early when
    the optimizer signals convergence/termination.
``_train_step``
    Encapsulates the per-batch workflow of generation, reward computation,
    sequence scoring, loss evaluation, gradient accumulation, controller
    updates, and logging hooks.

The helpers favor explicit parameters over implicit globals so that unit tests
can stub individual pieces (generation, checkpointing, metrics) without running
the full distributed stack.  All docstrings are Sphinx-friendly to power the
reference documentation for the training CLI.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from collections.abc import MutableMapping
import math
from contextlib import nullcontext

from dataclasses import replace
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast

from .rollout import CompletionGenerator, GenerationContext
from maxent_grpo.training.runtime import require_accelerator, require_torch
from maxent_grpo.core.hub import push_to_hub_revision
from .eval import run_validation_step
from .weighting.loss import LossInputConfig, build_loss_inputs, evaluate_losses
from . import pipeline as pipeline_mod
from .metrics import (
    log_local_step,
    log_training_step,
    summarize_reward_stats,
    summarize_weight_stats,
)
from .optim import (
    configure_accumulation_steps,
    detect_deepspeed_state,
    epoch_progress,
    optimizer_step,
    require_accumulation_context,
    scheduled_learning_rate,
    sync_gradients_enabled,
)
from .state import (
    check_stop_condition,
    load_controller_state_chain,
    maybe_checkpoint,
    maybe_load_accelerator_state,
)
from .controller_objective import ControllerMetaContext
from .weighting.logic import (
    apply_meta_controller_update,
    broadcast_controller_state,
    maybe_update_beta,
    maybe_update_tau,
    save_controller_state,
    _sync_controller_state,
)
from .scoring import vllm_meta_has_logprobs
from .types import (
    EvaluationSettings,
    LogStepArtifacts,
    StepBatchInfo,
    StepResources,
    TrainingLoopContext,
    TrainingLoopState,
    ValidationContext,
)
from .zero_utils import _maybe_patch_zero_no_sync

torch = require_torch("training")
Tensor = torch.Tensor

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
else:  # pragma: no cover - runtime uses optional torch stub
    TorchTensor = Any
Accelerator = require_accelerator("training")
prepare_training_batch = pipeline_mod.prepare_training_batch

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from maxent_grpo.config import GRPOConfig as GRPOConfigType
else:
    GRPOConfigType = Any

LOG = logging.getLogger(__name__)
_PROMPT_OBJECTIVE_ENV_VAR = "MAXENT_LOG_PROMPT_OBJECTIVE"
_PROMPT_OBJECTIVE_PREVIEW_LEN = 160
_VLLM_LOGPROB_FAIL_AFTER_ENV = "MAXENT_VLLM_LOGPROB_FAIL_AFTER"
_VLLM_LOGPROB_FALLBACK_ENV = "MAXENT_VLLM_LOGPROB_FALLBACK"
_scheduled_learning_rate = scheduled_learning_rate
_epoch_progress = epoch_progress
_optimizer_step = optimizer_step


def _coerce_bool(value: Any) -> Optional[bool]:
    """Return a parsed boolean when possible, else None."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _maybe_save_seed_heatmap(seed_heatmap: Optional[dict], step: int) -> None:
    """Persist a correlation heatmap when enabled via env flag.

    Controlled by ``INFOSEED_SAVE_HEATMAP=1``; writes a JSON file under
    ``INFOSEED_HEATMAP_DIR`` (default: ``var/artifacts/logs/seed_heatmaps``).
    """

    if not seed_heatmap:
        return
    import json

    if os.environ.get("INFOSEED_SAVE_HEATMAP", "0") not in {"1", "true", "True"}:
        return
    out_dir = os.environ.get("INFOSEED_HEATMAP_DIR", "var/artifacts/logs/seed_heatmaps")
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"seed_heatmap_step{int(step)}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(seed_heatmap, f)
    except (OSError, TypeError, ValueError):  # pragma: no cover - best-effort
        LOG.warning("Failed to save seed heatmap at step %d", step)


def _prompt_objective_logging_enabled(ctx: TrainingLoopContext) -> bool:
    """Return True when per-prompt objective logging is requested."""

    training_args = getattr(ctx, "training_args", None)
    args_enabled = bool(getattr(training_args, "log_prompt_objective", False))
    env_val = os.environ.get(_PROMPT_OBJECTIVE_ENV_VAR, "")
    if isinstance(env_val, str):
        env_enabled = env_val.strip().lower() in {"1", "true", "yes", "on"}
    else:
        env_enabled = False
    return args_enabled or env_enabled


def _to_cpu_tensor(value: Any) -> TorchTensor:
    """Best-effort conversion of tensors/arrays to 1D CPU float tensors."""

    if value is None:
        return torch.tensor([], dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        try:
            return value.detach().float().cpu().view(-1)
        except (RuntimeError, TypeError, ValueError):  # pragma: no cover - defensive conversion
            return torch.tensor([], dtype=torch.float32)
    arr = getattr(value, "arr", value)
    try:
        tensor = torch.tensor(arr, dtype=torch.float32)
    except (TypeError, ValueError):
        return torch.tensor([], dtype=torch.float32)
    return tensor.view(-1)


def _entropy_from_probs(probs: Optional[List[float]]) -> float:
    """Return natural-log entropy for a probability vector."""

    if not probs:
        return 0.0
    filtered = [max(float(p), 1e-12) for p in probs if isinstance(p, (int, float))]
    if not filtered:
        return 0.0
    total = sum(filtered)
    if total <= 0.0:
        return 0.0
    normalized = [val / total for val in filtered]
    return float(-sum(val * math.log(val) for val in normalized))


def _per_sequence_kl_values(
    scores: Any,
    ref_stats: Any,
    weighting_cfg: Any,
) -> List[float]:
    """Return per-sequence KL estimates used for prompt-level logging."""

    if scores is None or ref_stats is None or weighting_cfg is None:
        return []
    cur_logp = _to_cpu_tensor(getattr(scores, "cur_logp_sum", None))
    denom = _to_cpu_tensor(getattr(scores, "denom_tok_tensor", None)).clamp(min=1.0)
    count = min(cur_logp.numel(), denom.numel())
    if count <= 0:
        return []
    cur_logp = cur_logp[:count]
    denom = denom[:count]
    cur_per_tok = cur_logp / denom
    ref_source = (
        getattr(ref_stats, "ref_logp_sum", None)
        if getattr(weighting_cfg, "len_norm_ref", False)
        else getattr(ref_stats, "ref_logp_sum_raw", None)
    )
    if ref_source is None:
        # Fallback to whichever tensor is available so logging still works.
        ref_source = getattr(ref_stats, "ref_logp_sum", None)
    ref_tensor = _to_cpu_tensor(ref_source)
    if ref_tensor.numel() == 0:
        return []
    if ref_tensor.numel() < count:
        pad_val = ref_tensor[-1]
        pad = pad_val.new_full((count - ref_tensor.numel(),), float(pad_val))
        ref_tensor = torch.cat([ref_tensor, pad], dim=0)
    ref_tensor = ref_tensor[:count]
    if getattr(weighting_cfg, "len_norm_ref", False):
        ref_per_tok = ref_tensor
    else:
        ref_per_tok = ref_tensor / denom
    delta = (ref_per_tok - cur_per_tok).clamp(min=-60.0, max=60.0)
    per_seq = delta.exp() - delta - 1.0
    return per_seq.detach().cpu().tolist()


def _prompt_preview(text: str) -> str:
    """Return a compact preview of the prompt for logging."""

    if not text:
        return ""
    compact = " ".join(str(text).strip().split())
    if len(compact) <= _PROMPT_OBJECTIVE_PREVIEW_LEN:
        return compact
    return compact[: _PROMPT_OBJECTIVE_PREVIEW_LEN - 1] + "…"


def _resolve_vllm_logprob_fail_after(ctx: TrainingLoopContext) -> int:
    """Return the consecutive-step threshold for missing vLLM logprobs."""

    training_args = getattr(ctx, "training_args", None)
    cfg_val = getattr(training_args, "vllm_logprob_fail_after", None)
    if cfg_val is not None:
        try:
            return max(0, int(cfg_val))
        except (TypeError, ValueError):
            LOG.warning(
                "Invalid vllm_logprob_fail_after=%s; falling back to env/default.",
                cfg_val,
            )
    raw = os.environ.get(_VLLM_LOGPROB_FAIL_AFTER_ENV)
    if raw is None:
        return 3
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 3


def _vllm_logprob_fallback_enabled(ctx: TrainingLoopContext) -> bool:
    training_args = getattr(ctx, "training_args", None)
    cfg_val = getattr(training_args, "vllm_logprob_fallback", None)
    parsed = _coerce_bool(cfg_val)
    if parsed is not None:
        return parsed
    raw = os.environ.get(_VLLM_LOGPROB_FALLBACK_ENV, "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _maybe_guard_vllm_logprobs(
    ctx: TrainingLoopContext, prepared: Any, global_step: int
) -> None:
    """Fail fast or force fallback when vLLM logprobs are repeatedly missing."""

    gen_cfg = getattr(ctx, "generation", None)
    if gen_cfg is None or not getattr(gen_cfg, "use_vllm", False):
        return
    if not getattr(gen_cfg, "vllm_request_logprobs", False):
        return
    scoring_cfg = getattr(ctx, "scoring", None)
    ref_source = str(
        getattr(scoring_cfg, "reference_logprobs_source", "auto") or "auto"
    ).lower()
    if ref_source in {"model", "reference", "reference_model", "ref_model"}:
        return
    reward_comp = getattr(prepared, "reward_comp", None)
    flat_meta = getattr(reward_comp, "ref_logprob_meta", None)
    total_sequences = None
    batch_stats = getattr(prepared, "batch_stats", None)
    score_batch = getattr(batch_stats, "score_batch", None) if batch_stats else None
    if score_batch is not None:
        total_sequences = getattr(score_batch, "total_sequences", None)
    has_meta = vllm_meta_has_logprobs(flat_meta, total_sequences)
    counter = int(getattr(ctx.runtime, "_vllm_logprob_miss_steps", 0) or 0)
    if has_meta:
        if counter:
            setattr(ctx.runtime, "_vllm_logprob_miss_steps", 0)
        return
    counter += 1
    setattr(ctx.runtime, "_vllm_logprob_miss_steps", counter)
    limit = _resolve_vllm_logprob_fail_after(ctx)
    if limit <= 0:
        return
    if counter < limit:
        LOG.warning(
            "vLLM logprob metadata missing | step=%d | consecutive_missing=%d/%d",
            global_step,
            counter,
            limit,
        )
        return
    if _vllm_logprob_fallback_enabled(ctx):
        LOG.error(
            (
                "vLLM logprobs missing for %d consecutive steps; forcing "
                "reference_logprobs_source=model."
            ),
            counter,
        )
        try:
            ctx.scoring.reference_logprobs_source = "model"
        except (AttributeError, TypeError, ValueError):
            LOG.warning("Failed to set scoring.reference_logprobs_source to model.")
        try:
            ctx.settings.scoring.reference_logprobs_source = "model"
        except (AttributeError, TypeError, ValueError):
            LOG.warning("Failed to set settings.scoring.reference_logprobs_source to model.")
        setattr(ctx.runtime, "_vllm_logprob_miss_steps", 0)
        return
    raise RuntimeError(
        (
            "vLLM logprob metadata missing for "
            f"{counter} consecutive steps (>= {limit}). "
            "Check vLLM client_tag echo and return_logprobs settings, or set "
            "vllm_logprob_fallback=true "
            f"(or {_VLLM_LOGPROB_FALLBACK_ENV}=1) to force reference-model fallback."
        )
    )


def _build_prompt_objective_entries(
    prepared: Any,
    weighting_cfg: Any,
) -> List[Dict[str, Any]]:
    """Return per-prompt summaries of reward, KL, and entropy."""

    if prepared is None or weighting_cfg is None:
        return []
    grouped = getattr(prepared, "grouped_completions", None) or []
    if not grouped:
        return []
    reward_comp = getattr(prepared, "reward_comp", None)
    weight_stats = getattr(prepared, "weight_stats", None)
    if reward_comp is None:
        return []
    rewards_flat = list(getattr(reward_comp, "total_utils", []) or [])
    q_grouped = getattr(reward_comp, "q_grouped", None)
    if q_grouped is None:
        q_dist = getattr(reward_comp, "q_distribution", None)
        q_grouped = getattr(q_dist, "grouped", None)
    if q_grouped is None:
        q_grouped = []
    prompt_pairs = getattr(reward_comp, "pairs", None)
    prompt_texts = list(getattr(prompt_pairs, "prompts", []) or [])
    weight_groups = getattr(weight_stats, "weights_grouped", None) or []
    use_weight_entropy = not (
        weighting_cfg is not None
        and getattr(weighting_cfg, "train_grpo_objective", False)
    )
    kl_values = _per_sequence_kl_values(
        getattr(prepared, "scores", None),
        getattr(prepared, "ref_stats", None),
        weighting_cfg,
    )
    entries: List[Dict[str, Any]] = []
    offset = 0
    for idx, comp_group in enumerate(grouped):
        size = len(comp_group)
        if size <= 0:
            continue
        reward_slice = rewards_flat[offset : offset + size]
        kl_slice = kl_values[offset : offset + size] if kl_values else []
        reward_mean = (
            float(sum(reward_slice) / len(reward_slice)) if reward_slice else 0.0
        )
        kl_mean = float(sum(kl_slice) / len(kl_slice)) if kl_slice else 0.0
        q_entropy = _entropy_from_probs(q_grouped[idx] if idx < len(q_grouped) else [])
        weight_entropy = (
            _entropy_from_probs(weight_groups[idx] if idx < len(weight_groups) else [])
            if use_weight_entropy
            else 0.0
        )
        prompt_text = prompt_texts[offset] if offset < len(prompt_texts) else ""
        entries.append(
            {
                "index": idx,
                "prompt": prompt_text,
                "reward": reward_mean,
                "reward_values": reward_slice,
                "kl": kl_mean,
                "kl_values": kl_slice,
                "q_entropy": q_entropy,
                "weight_entropy": weight_entropy,
                "objective": reward_mean + kl_mean + q_entropy,
                "group_size": size,
            }
        )
        offset += size
    return entries


def _log_prompt_objective(
    ctx: TrainingLoopContext,
    prepared: Any,
    step: int,
) -> None:
    """Emit per-prompt objective breakdown when explicitly requested."""

    if not _prompt_objective_logging_enabled(ctx):
        return
    weighting_cfg = getattr(getattr(ctx, "scoring", None), "weighting", None)
    entries = _build_prompt_objective_entries(prepared, weighting_cfg)
    if not entries:
        return
    for entry in entries:
        LOG.info(
            (
                "Prompt objective | step=%d | idx=%d | comps=%d | reward=%.4f | "
                "kl=%.4f | entropy=%.4f | weight_entropy=%.4f | objective=%.4f | prompt=%s"
            ),
            step,
            entry["index"],
            entry["group_size"],
            entry["reward"],
            entry["kl"],
            entry["q_entropy"],
            entry["weight_entropy"],
            entry["objective"],
            _prompt_preview(entry.get("prompt", "")),
        )


def _maybe_validate(
    evaluation_cfg: EvaluationSettings,
    validation_ctx: ValidationContext,
    global_step: int,
) -> None:
    """Optionally trigger the evaluation loop.

    :param evaluation_cfg: Evaluation schedule/settings.
    :type evaluation_cfg: EvaluationSettings
    :param validation_ctx: Handles required to run validation.
    :type validation_ctx: ValidationContext
    :param global_step: Current optimizer step.
    :type global_step: int
    """
    if (
        evaluation_cfg.enabled
        and evaluation_cfg.every_n_steps
        and (global_step % evaluation_cfg.every_n_steps == 0)
    ):
        run_validation_step(global_step, validation_ctx)


def _maybe_overwrite_controller_state_from_config(
    ctx: TrainingLoopContext, controller_resumed: bool = False
) -> None:
    """Optionally force controller scalars to match the active recipe."""

    training_args = getattr(ctx, "training_args", None)
    if training_args is None:
        return
    if not getattr(training_args, "controller_overwrite_from_config", False):
        return
    if controller_resumed:
        LOG.info(
            "Controller state resumed from checkpoint; skipping config overwrite."
        )
        return
    weighting = getattr(getattr(ctx, "scoring", None), "weighting", None)
    if weighting is None:
        return

    def _coerce_scalar(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    tau_override = _coerce_scalar(getattr(training_args, "maxent_tau", None))
    beta_override = _coerce_scalar(getattr(training_args, "beta", None))
    updated = False
    prev_tau = getattr(weighting, "tau", None)
    prev_beta = getattr(weighting, "beta", None)
    if tau_override is not None:
        weighting.tau = tau_override
        updated = True
    if beta_override is not None:
        weighting.beta = beta_override
        updated = True
    if not updated:
        return
    if getattr(weighting, "train_grpo_objective", False):
        weighting.denom = 1.0
    else:
        denom_sum = float(weighting.tau) + float(weighting.beta)
        weighting.denom = denom_sum if denom_sum > 0 else 1.0
    try:
        setattr(weighting, "_tau_entropy_ema", float(weighting.tau))
        setattr(weighting, "_tau_log", math.log(max(float(weighting.tau), 1e-8)))
    except (TypeError, ValueError):
        LOG.debug("Failed to refresh weighting tau tracking fields.")
    _sync_controller_state(weighting)
    accelerator = getattr(getattr(ctx, "runtime", None), "accelerator", None)
    if accelerator is not None:
        broadcast_controller_state(accelerator, weighting)
    prev_tau_float = _coerce_scalar(prev_tau)
    prev_beta_float = _coerce_scalar(prev_beta)
    LOG.info(
        "Overwrote controller state from config | tau=%.4f (prev=%s) | beta=%.4f (prev=%s)",
        float(weighting.tau),
        "nan" if prev_tau_float is None else f"{prev_tau_float:.4f}",
        float(weighting.beta),
        "nan" if prev_beta_float is None else f"{prev_beta_float:.4f}",
    )


def _apply_weighting_overrides_from_config(ctx: TrainingLoopContext) -> None:
    """Apply non-controller weighting toggles from the active training config."""

    training_args = getattr(ctx, "training_args", None)
    if training_args is None:
        return
    scoring_cfg = getattr(ctx, "scoring", None)
    weighting = getattr(scoring_cfg, "weighting", None) if scoring_cfg else None
    if weighting is None:
        return
    fallback_flag = getattr(training_args, "maxent_allow_empty_weight_fallback", None)
    if fallback_flag is not None:
        weighting.allow_empty_weight_fallback = bool(fallback_flag)


def _train_step(
    ctx: TrainingLoopContext,
    state: TrainingLoopState,
    step_info: StepBatchInfo,
    resources: StepResources,
) -> bool:
    """Process a single batch and update training state.

    :param ctx: Training loop context containing configs and runtime handles.
    :type ctx: :class:`~training.types.TrainingLoopContext`
    :param state: Mutable counters shared across batches and epochs.
    :type state: :class:`~training.types.TrainingLoopState`
    :param step_info: Metadata describing the current batch (epoch index,
        micro-step, and raw data).
    :type step_info: :class:`~training.types.StepBatchInfo`
    :param resources: Reusable handles for generation and validation.
    :type resources: :class:`~training.types.StepResources`
    :returns: ``True`` when training should stop (schedule exhausted or
        controller requested halt), otherwise ``False``.
    :rtype: bool
    """
    schedule = ctx.optimization.schedule
    accelerator = ctx.runtime.accelerator
    ds_state = detect_deepspeed_state(accelerator)
    LOG.debug(
        "Train step begin | epoch=%d | step_in_epoch=%d | global_step=%d",
        step_info.epoch,
        step_info.step_in_epoch,
        state.global_step,
    )
    gen_stats = getattr(ctx.generation, "generation_stats", None)
    if isinstance(gen_stats, MutableMapping):
        gen_stats["current_step"] = int(state.global_step)
    prepared = prepare_training_batch(ctx, resources.generator, step_info.batch)
    if prepared is None:
        skip_stage = getattr(ctx.runtime, "_last_skip_stage", "unknown")
        LOG.warning(
            "Skipping training batch | epoch=%d | step_in_epoch=%d | global_step=%d | stage=%s",
            step_info.epoch,
            step_info.step_in_epoch,
            state.global_step,
            skip_stage,
        )
        try:
            delattr(ctx.runtime, "_last_skip_stage")
        except (AttributeError, TypeError):
            LOG.debug("Failed to clear runtime skip stage marker.")
        return False
    _maybe_guard_vllm_logprobs(ctx, prepared, state.global_step)
    state.num_input_tokens_seen += float(prepared.total_input_tokens)
    loss_outputs, diagnostics = evaluate_losses(
        *build_loss_inputs(
            prepared.grouped_completions,
            prepared.weight_stats,
            prepared.scores,
            LossInputConfig(
                clip_cfg=ctx.scoring.clipping,
                weighting_cfg=ctx.scoring.weighting,
                ref_stats=prepared.ref_stats,
            ),
        ),
        seed_inputs=(
            prepared.scores.seed_aux if hasattr(prepared.scores, "seed_aux") else None
        ),
        info_seed_lambda=getattr(ctx.scoring, "info_seed_lambda", 0.0),
        info_seed_temperature=getattr(ctx.scoring, "info_seed_temperature", 0.1),
        info_seed_loss_type=getattr(ctx.scoring, "info_seed_loss_type", "infonce"),
        info_seed_alpha_entropy=getattr(ctx.scoring, "info_seed_alpha_entropy", 0.0),
    )
    _maybe_save_seed_heatmap(
        getattr(prepared, "seed_heatmap", None),
        state.global_step,
    )
    _log_prompt_objective(ctx, prepared, state.global_step)
    current_lr = _scheduled_learning_rate(
        schedule, ctx.optimization.handles, state.global_step
    )
    log_artifacts = LogStepArtifacts(
        loss_outputs=loss_outputs,
        diagnostics=diagnostics,
        grad_norm_scalar=None,
        epoch_progress=_epoch_progress(
            schedule, step_info.epoch, step_info.step_in_epoch
        ),
    )
    reward_stats_global = summarize_reward_stats(
        accelerator, getattr(prepared, "reward_comp", None)
    )
    weight_stats_global = summarize_weight_stats(
        accelerator, prepared.weight_stats
    )
    scalars = getattr(loss_outputs, "scalars", None)
    if scalars is not None:
        LOG.debug(
            "Loss scalars | total=%.4f | policy=%.4f | kl=%.4f",
            getattr(scalars, "total_loss", 0.0),
            getattr(scalars, "policy_loss", 0.0),
            getattr(scalars, "kl_loss", 0.0),
        )
    if accelerator.__class__.__name__ == "SimpleNamespace":
        accumulation_ctx = nullcontext()
    else:
        accumulation_ctx = require_accumulation_context(
            accelerator,
            ctx.runtime.model,
        )
        if not hasattr(accumulation_ctx, "__enter__"):
            accumulation_ctx = nullcontext()
    grad_accum_total = int(getattr(schedule, "grad_accum_steps", 1) or 1)
    accum_position = (step_info.step_in_epoch % grad_accum_total) + 1
    grad_norm_scalar: Optional[float] = None
    LOG.debug(
        "Entering accumulate context | grad_accum_steps=%d | accum_progress=%d/%d | step_in_epoch=%d",
        schedule.grad_accum_steps,
        accum_position,
        grad_accum_total,
        step_info.step_in_epoch,
    )
    with accumulation_ctx:
        # Some test stubs may produce a loss tensor that does not require
        # gradients (e.g., detached or aggregated into a float). Guard the
        # backward call to avoid RuntimeError when autograd is not active.
        loss_val = loss_outputs.loss
        if getattr(loss_val, "requires_grad", False):
            accelerator.backward(loss_val)
        else:
            LOG.debug("Skipping backward: loss does not require grad")
        if ds_state.use_deepspeed and ds_state.zero_stage >= 2:
            if (step_info.step_in_epoch + 1) % grad_accum_total != 0:
                LOG.debug(
                    "DeepSpeed accumulate | deferring optimizer step | epoch=%d | step_in_epoch=%d | accum_progress=%d/%d",
                    step_info.epoch,
                    step_info.step_in_epoch,
                    accum_position,
                    grad_accum_total,
                )
                log_local_step(
                    ctx,
                    state,
                    prepared,
                    log_artifacts,
                    current_lr,
                    reward_view=reward_stats_global,
                    weight_view=weight_stats_global,
                    emit=False,
                )
                return False
            grad_norm_scalar = _optimizer_step(ctx, state, current_lr)
            LOG.debug(
                "Optimizer step executed | global_step=%d | grad_norm=%s",
                state.global_step,
                grad_norm_scalar,
            )
        else:
            if not sync_gradients_enabled(accelerator, state.global_step):
                LOG.debug(
                    "Deferring optimizer step until sync | epoch=%d | step_in_epoch=%d | accum_progress=%d/%d",
                    step_info.epoch,
                    step_info.step_in_epoch,
                    accum_position,
                    grad_accum_total,
                )
                log_local_step(
                    ctx,
                    state,
                    prepared,
                    log_artifacts,
                    current_lr,
                    reward_view=reward_stats_global,
                    weight_view=weight_stats_global,
                    emit=False,
                )
                return False
            grad_norm_scalar = _optimizer_step(ctx, state, current_lr)
            LOG.debug(
                "Optimizer step executed | global_step=%d | grad_norm=%s",
                state.global_step,
                grad_norm_scalar,
            )
    LOG.debug("Exiting accumulate context | synced_step=%d", state.global_step)
    log_artifacts.grad_norm_scalar = grad_norm_scalar
    log_local_step(
        ctx,
        state,
        prepared,
        log_artifacts,
        current_lr,
        reward_view=reward_stats_global,
        weight_view=weight_stats_global,
    )
    log_training_step(
        ctx,
        state,
        prepared,
        log_artifacts,
        current_lr,
        reward_view=reward_stats_global,
        weight_view=weight_stats_global,
    )
    meta_cfg = getattr(getattr(ctx.scoring, "weighting", None), "controller_meta", None)
    meta_enabled = bool(getattr(meta_cfg, "enabled", False))
    if not meta_enabled:
        maybe_update_beta(ctx.scoring.weighting, loss_outputs.kl_loss_scalar)
    base_lr = max(float(getattr(ctx.optimization.handles, "learning_rate", 0.0)), 1e-12)
    lr_scale = float(current_lr) / base_lr if base_lr > 0 else 1.0
    weight_stats_for_tau = (
        weight_stats_global
        if weight_stats_global is not None
        else getattr(prepared, "weight_stats", None)
    )
    if not meta_enabled:
        try:
            maybe_update_tau(
                ctx.scoring.weighting,
                weight_stats_for_tau,
                state.global_step,
                lr_scale=lr_scale,
            )
        except TypeError:
            maybe_update_tau(
                ctx.scoring.weighting, weight_stats_for_tau, state.global_step
            )
    settings_obj = getattr(ctx, "settings", None)
    controller_objective = getattr(settings_obj, "controller_objective", None)
    controller_manager = getattr(settings_obj, "controller_meta_manager", None)
    if controller_manager is None:
        controller_manager = getattr(ctx, "controller_meta_manager", None)
    if controller_objective is None:
        controller_objective = getattr(ctx, "controller_objective", None)
    if controller_objective is not None:
        should_run_meta = (
            controller_manager.should_run(state.global_step)
            if controller_manager
            else True
        )
    else:
        should_run_meta = False
    if controller_objective is not None and should_run_meta:
        _cache_meta_stats(ctx.scoring.weighting, weight_stats_global, loss_outputs)
        meta_ctx = ControllerMetaContext(
            weighting=ctx.scoring.weighting,
            weight_stats=weight_stats_global,
            loss_outputs=loss_outputs,
            prepared_batch=prepared,
            global_step=state.global_step,
            lr_scale=lr_scale,
            kl_value=loss_outputs.kl_loss_scalar,
            backprop_fn=controller_manager.make_backprop_fn()
            if controller_manager
            else None,
        )
        try:
            gradients = controller_objective.compute(meta_ctx)
        except (RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover - defensive logging
            gradients = None
            LOG.warning("Controller objective failed: %s", exc)
        if controller_manager:
            controller_manager.apply_gradients(gradients, lr_scale=lr_scale)
        elif gradients and gradients.has_updates():
            apply_meta_controller_update(
                ctx.scoring.weighting,
                tau_grad=gradients.tau_grad,
                beta_grad=gradients.beta_grad,
                lr_scale=lr_scale,
            )
    broadcast_controller_state(accelerator, ctx.scoring.weighting)
    if ctx.runtime.accelerator.is_main_process:
        save_controller_state(ctx.controller.state_path, ctx.scoring.weighting)
    _maybe_validate(ctx.evaluation, resources.validation_ctx, state.global_step)
    maybe_checkpoint(ctx.logging, ctx.runtime.accelerator, state.global_step)
    check_stop_condition(ctx.optimization.schedule, state)
    return state.stop_training


def _run_epoch(
    ctx: TrainingLoopContext,
    state: TrainingLoopState,
    epoch: int,
    resources: StepResources,
) -> bool:
    """Run a full epoch and return ``True`` when training should stop.

    :param ctx: Training loop context with loaders, configs, and handles.
    :type ctx: TrainingLoopContext
    :param state: Mutable loop counters shared across epochs.
    :type state: TrainingLoopState
    :param epoch: Epoch index used for schedulers/samplers.
    :type epoch: int
    :param resources: Reusable generation/validation handles.
    :type resources: StepResources
    :returns: ``True`` if the loop signaled an early stop.
    :rtype: bool
    """
    sampler = getattr(ctx.runtime, "train_sampler", None)
    if sampler is not None:
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            try:
                set_epoch(epoch)
            except TypeError:
                set_epoch(int(epoch))
    for step_in_epoch, batch in enumerate(ctx.runtime.train_loader):
        step_info = StepBatchInfo(epoch=epoch, step_in_epoch=step_in_epoch, batch=batch)
        if _train_step(ctx, state, step_info, resources):
            return True
    return state.stop_training


def run_training_loop(ctx: TrainingLoopContext) -> None:
    """Execute the training loop using the supplied context.

    :param ctx: Fully-populated :class:`~training.types.TrainingLoopContext`
        describing runtime handles, configurations, logging hooks, and controller paths.
    :type ctx: :class:`~training.types.TrainingLoopContext`
    :returns: ``None``. Side effects are driven through the provided handles.
    :rtype: None
    """

    def _maybe_load_optimizer_state(path: Optional[str]) -> None:
        opt = getattr(ctx.optimization.handles, "optimizer", None)
        if not path or opt is None:
            return
        opt_path = os.path.join(path, "optimizer.pt")
        if not os.path.isfile(opt_path):
            return
        try:
            state_dict = torch.load(opt_path, map_location=runtime.device)
            opt.load_state_dict(state_dict)
            LOG.info("Loaded optimizer state from %s", opt_path)
        except (OSError, RuntimeError, ValueError) as exc:
            LOG.warning("Failed to load optimizer state from %s: %s", opt_path, exc)

    def _maybe_push_final() -> None:
        training_args = getattr(ctx, "training_args", None)
        if training_args is None:
            return
        push_enabled = bool(
            getattr(training_args, "push_to_hub", False)
            or getattr(training_args, "push_to_hub_revision", False)
        )
        if not push_enabled:
            return
        hub_strategy = str(getattr(training_args, "hub_strategy", "end") or "end").lower()
        if hub_strategy not in {"end", "last", "final"}:
            return
        if not runtime.accelerator.is_main_process:
            return
        try:
            push_args = SimpleNamespace(**getattr(training_args, "__dict__", {}))
            push_args.push_to_hub_revision = True
            push_to_hub_revision(
                cast(GRPOConfigType, push_args),
                extra_ignore_patterns=[],
                include_checkpoints=True,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - optional hub deps
            LOG.warning("Failed to push final output_dir to Hub: %s", exc)

    runtime = ctx.runtime
    generation_cfg = ctx.generation
    generation_ctx = GenerationContext(
        accelerator=runtime.accelerator,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        generation_stats=generation_cfg.generation_stats,
        device=runtime.device,
        max_prompt_len=generation_cfg.max_prompt_len,
        max_completion_len=generation_cfg.max_completion_len,
        gen_temperature=generation_cfg.gen_temperature,
        gen_top_p=generation_cfg.gen_top_p,
        use_vllm=generation_cfg.use_vllm,
        vllm=generation_cfg.vllm,
        penalty=replace(generation_cfg.penalty),
    )
    training_args = getattr(ctx, "training_args", None)
    if training_args is not None:
        setattr(generation_ctx, "training_args", training_args)
        cfg_val = getattr(training_args, "vllm_client_tag_fail_fast", None)
        if cfg_val is not None:
            setattr(generation_ctx, "vllm_client_tag_fail_fast", cfg_val)
        sync_interval = getattr(training_args, "vllm_sync_interval_steps", None)
        if sync_interval is not None:
            setattr(generation_ctx, "vllm_sync_interval_steps", sync_interval)
    completion_generator = CompletionGenerator(generation_ctx)
    validation_ctx = ValidationContext(
        evaluation=ctx.evaluation,
        accelerator=runtime.accelerator,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        reward=ctx.reward,
        eval_reward=getattr(ctx, "eval_reward", None),
        generator=completion_generator.generate,
        logging=ctx.logging,
    )
    configure_accumulation_steps(
        runtime.accelerator, ctx.optimization.schedule.grad_accum_steps
    )
    _maybe_patch_zero_no_sync(runtime.model)
    resume_state = getattr(ctx, "resume_state", {}) or {}
    starting_step = int(resume_state.get("global_step", 0) or 0)
    starting_tokens = float(resume_state.get("num_input_tokens_seen", 0.0) or 0.0)
    state = TrainingLoopState(
        global_step=starting_step,
        num_input_tokens_seen=starting_tokens,
    )
    if runtime.accelerator.is_main_process:
        LOG.info(
            "Training schedule | steps_per_epoch=%s | total_training_steps=%s | grad_accum_steps=%s",
            ctx.optimization.schedule.steps_per_epoch,
            ctx.optimization.schedule.total_training_steps,
            ctx.optimization.schedule.grad_accum_steps,
        )
        if starting_step > 0:
            LOG.info(
                "Resumed training state | checkpoint=%s | global_step=%d | num_input_tokens_seen=%.0f",
                getattr(ctx, "resume_checkpoint", None),
                starting_step,
                starting_tokens,
            )
    resources = StepResources(
        generator=completion_generator.generate,
        validation_ctx=validation_ctx,
    )
    state_ref = getattr(ctx, "checkpoint_state_ref", None)
    if isinstance(state_ref, dict):
        state_ref["state"] = state
    accel_state_path = getattr(ctx, "resume_checkpoint", None) or getattr(
        ctx.controller, "resume_from", None
    )
    controller_loaded = load_controller_state_chain(
        ctx.controller,
        runtime.accelerator,
        ctx.scoring.weighting,
    )
    _maybe_overwrite_controller_state_from_config(
        ctx, controller_resumed=bool(controller_loaded)
    )
    _apply_weighting_overrides_from_config(ctx)
    maybe_load_accelerator_state(accel_state_path, runtime.accelerator)
    _maybe_load_optimizer_state(accel_state_path)
    try:
        ctx.optimization.handles.optimizer.zero_grad(set_to_none=True)
        for epoch in range(ctx.optimization.schedule.num_epochs):
            if _run_epoch(ctx, state, epoch, resources):
                break
    finally:
        if runtime.accelerator.is_main_process:
            LOG.info(
                (
                    "Generation stats | retries=%d | backfilled_prompts=%d | "
                    "vllm_failures=%d | dropped_prompts=%d | partial_prompts=%d | "
                    "excess_prompts=%d | excess_completions=%d"
                ),
                generation_cfg.generation_stats["vllm_retry_rounds"],
                generation_cfg.generation_stats["vllm_backfilled_prompts"],
                generation_cfg.generation_stats["vllm_failed_prompts"],
                generation_cfg.generation_stats["dropped_prompts"],
                generation_cfg.generation_stats.get("partial_prompts", 0),
                generation_cfg.generation_stats.get("vllm_excess_prompts", 0),
                generation_cfg.generation_stats.get("vllm_excess_completions", 0),
            )
            _maybe_push_final()
        if ctx.logging.wandb_run is not None and runtime.accelerator.is_main_process:
            try:
                ctx.logging.wandb_run.finish()
            except (
                RuntimeError,
                ValueError,
            ) as exc:  # pragma: no cover - defensive logging
                LOG.warning("Failed to close W&B run cleanly: %s", exc)


def _cache_meta_stats(weighting_cfg: Any, weight_view: Any, loss_outputs: Any) -> None:
    entropy_val = getattr(weight_view, "weight_entropy", None)
    if entropy_val is None:
        entropy_val = getattr(weight_view, "entropy", None)
    if isinstance(entropy_val, (int, float)):
        setattr(weighting_cfg, "_meta_entropy_value", float(entropy_val))
    kl_val = getattr(loss_outputs, "kl_loss_scalar", None)
    if isinstance(kl_val, (int, float)):
        setattr(weighting_cfg, "_meta_kl_value", float(kl_val))
    meta_loss = 0.0
    target_entropy = getattr(weighting_cfg, "tau_target_entropy", None)
    if target_entropy is not None and isinstance(entropy_val, (int, float)):
        entropy_error = float(entropy_val) - float(target_entropy)
        meta_loss += 0.5 * entropy_error * entropy_error
    kl_target = getattr(weighting_cfg, "kl_target", None)
    if kl_target and isinstance(kl_val, (int, float)):
        kl_error = float(kl_val) - float(kl_target)
        meta_loss += 0.5 * kl_error * kl_error
    setattr(weighting_cfg, "_meta_last_loss", float(meta_loss))
