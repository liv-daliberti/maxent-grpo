"""Custom TRL GRPOTrainer wrapper used by the MaxEnt-GRPO pipelines.

This module provides a light subclass hook so we can gradually add MaxEnt/InfoSeed
behavior while still relying on TRL's Trainer loop and logging cadence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from functools import partial
import inspect
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import torch
from accelerate.utils import gather, gather_object
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template

from maxent_grpo.pipelines.training.loop_common import (
    _controller_paths,
    _build_prompt_cache_fn,
    _resolve_prompt_cache_size,
    build_evaluation_settings,
    build_generation_settings,
    build_scoring_settings,
    build_weighting_settings,
)
from maxent_grpo.telemetry.trl_logging import ensure_weighting_logging
from maxent_grpo.training.controller_objective import (
    ControllerMetaContext,
    build_controller_objective,
)
from maxent_grpo.training.controller_optimizer import ControllerMetaManager
from maxent_grpo.training.context_builder import apply_info_seed
from maxent_grpo.training.trainer_hooks import (
    _apply_weighting_overrides_from_config,
    _cache_meta_stats,
    _log_prompt_objective,
    _maybe_overwrite_controller_state_from_config,
    _maybe_save_seed_heatmap,
)
from maxent_grpo.training.metrics import (
    _build_metrics_payload,
    build_training_metrics_dict,
    summarize_reward_stats,
    summarize_weight_stats,
)
from maxent_grpo.training.optim import scheduled_learning_rate
from maxent_grpo.training.pipeline import (
    _completion_diversity_metrics,
    prepare_training_batch,
)
from maxent_grpo.training.rewards import prepare_generation_batch
from maxent_grpo.training.rollout import CompletionGenerator, GenerationContext
from maxent_grpo.training.scoring import (
    _apply_eos_completion_mask,
    _prepare_prompt_slice,
)
from maxent_grpo.training.state import load_controller_state_chain
from maxent_grpo.training.types import (
    ControllerPaths,
    LogStepArtifacts,
    LoggingHandles,
    LoopSettings,
    OptimizationSchedule,
    OptimizationSettings,
    OptimizerHandles,
    RewardSpec,
    RuntimeHandles,
    TrainingLoopContext,
)
from maxent_grpo.training.weighting.logic import (
    apply_meta_controller_update,
    broadcast_controller_state,
    maybe_update_beta,
    maybe_update_tau,
    save_controller_state,
)
from maxent_grpo.training.weighting.loss import (
    LossInputConfig,
    build_loss_inputs,
    evaluate_losses,
)

LOG = logging.getLogger(__name__)


def _build_seed_worker(num_workers: int, rank: int):
    """Return a worker_init_fn compatible with the active transformers seed_worker signature."""
    try:
        from transformers.trainer_utils import seed_worker as hf_seed_worker
    except Exception:  # pragma: no cover - transformers is required for training
        return None
    try:
        params = list(inspect.signature(hf_seed_worker).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return hf_seed_worker
    if len(params) <= 1:
        return hf_seed_worker
    return partial(hf_seed_worker, num_workers=num_workers, rank=rank)

try:  # Optional dependency for callback-based controller updates
    from transformers.trainer_callback import TrainerCallback
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    TrainerCallback = None


def _nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """Match TRL's nanstd helper for reward logging."""
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    variance = variance * (count / (count - 1))
    return torch.sqrt(variance)


def _numeric_or_none(value: Any) -> Optional[float]:
    """Best-effort numeric conversion used for logging filters."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        item_fn = getattr(value, "item", None)
        if callable(item_fn):
            try:
                return float(item_fn())
            except (TypeError, ValueError):
                return None
    return None


_BOOL_TRUE = {"1", "true", "t", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "f", "no", "n", "off", ""}


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Convert flexible config values to bool without surprising string truthiness."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _coerce_non_negative_float(value: Any, *, default: float = 0.0) -> float:
    """Convert config values to a finite non-negative float."""
    numeric = _numeric_or_none(value)
    if numeric is None or not math.isfinite(numeric):
        return default
    return max(float(numeric), 0.0)


def _coerce_examples(inputs: Any) -> List[Dict[str, Any]]:
    """Normalize trainer inputs into a list of example dicts."""
    if isinstance(inputs, list):
        return [cast(Dict[str, Any], ex) for ex in inputs]
    if isinstance(inputs, dict):
        values = list(inputs.values())
        if not values:
            return [cast(Dict[str, Any], inputs)]
        batch_len = None
        for val in values:
            if isinstance(val, (list, tuple)):
                batch_len = len(val)
                break
            if torch.is_tensor(val):
                batch_len = int(val.size(0)) if val.dim() > 0 else 1
                break
        if not batch_len:
            return [cast(Dict[str, Any], inputs)]
        batch: List[Dict[str, Any]] = []
        for idx in range(batch_len):
            row: Dict[str, Any] = {}
            for key, val in inputs.items():
                if isinstance(val, (list, tuple)):
                    row[key] = val[idx]
                elif torch.is_tensor(val):
                    row[key] = val[idx] if val.dim() > 0 else val
                else:
                    row[key] = val
            batch.append(row)
        return batch
    return [cast(Dict[str, Any], inputs)]


def _strip_mode_prefix(key: str, mode: str) -> str:
    """Remove a train/eval prefix from metric keys when applicable."""
    if mode == "train" and key.startswith("train/"):
        return key[len("train/") :]
    if mode == "eval" and key.startswith("eval/"):
        return key[len("eval/") :]
    return key


_CANONICAL_METRIC_KEYS: Dict[str, str] = {
    "completions/mean_length": "completions/mean_length_sampled",
    "completions/min_length": "completions/min_length_sampled",
    "completions/max_length": "completions/max_length_sampled",
    "completions/clipped_ratio": "completions/clipped_frac",
    "completions/mean_terminated_length": "completions/mean_length_terminated",
    "completions/min_terminated_length": "completions/min_length_terminated",
    "completions/max_terminated_length": "completions/max_length_terminated",
}
_LEGACY_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "completions/mean_length_sampled": ("completions/mean_length",),
    "completions/min_length_sampled": ("completions/min_length",),
    "completions/max_length_sampled": ("completions/max_length",),
    "completions/clipped_frac": ("completions/clipped_ratio",),
    "completions/mean_length_terminated": ("completions/mean_terminated_length",),
    "completions/min_length_terminated": ("completions/min_terminated_length",),
    "completions/max_length_terminated": ("completions/max_terminated_length",),
}


def _canonical_metric_key(key: str) -> str:
    """Normalize metric aliases to one canonical key namespace."""
    if key.startswith("diversity/"):
        return f"completions/{key}"
    return _CANONICAL_METRIC_KEYS.get(key, key)


def _legacy_metric_aliases(key: str) -> Tuple[str, ...]:
    """Return compatibility aliases for a canonical metric key."""
    aliases: List[str] = list(_LEGACY_METRIC_ALIASES.get(key, ()))
    if key.startswith("completions/diversity/"):
        aliases.append(key[len("completions/") :])
    if not aliases:
        return ()
    return tuple(dict.fromkeys(aliases))


class _NoopMetricWriter:
    def log(self, _metrics: Dict[str, Any], _step: int) -> None:
        return

    def flush(self) -> None:
        return


if TrainerCallback is not None:  # pragma: no cover - thin wrapper

    class _ControllerUpdateCallback(TrainerCallback):
        """Trigger controller updates after optimizer steps."""

        def __init__(self) -> None:
            self._last_step = -1

        def on_step_end(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            trainer = kwargs.get("trainer")
            if trainer is None:
                return control
            if not getattr(trainer, "maxent_enabled", False):
                return control
            step = getattr(state, "global_step", None)
            if step is None or step == self._last_step:
                return control
            self._last_step = int(step)
            try:
                trainer._apply_controller_updates()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOG.warning("Controller update failed: %s", exc)
            return control

else:  # pragma: no cover - optional dependency

    _ControllerUpdateCallback = None


def build_custom_grpo_trainer(parent_cls: Type[Any]) -> Type[Any]:
    """Return a GRPOTrainer subclass with MaxEnt hooks enabled.

    :param parent_cls: Base TRL GRPOTrainer class.
    :returns: Wrapped GRPOTrainer subclass.
    """

    if getattr(parent_cls, "_MAXENT_CUSTOM_TRAINER", False):
        return parent_cls

    class CustomGRPOTrainer(parent_cls):
        """Thin GRPOTrainer subclass used as a future extension point."""

        _MAXENT_CUSTOM_TRAINER = True

        @staticmethod
        def _resolve_parent_training_args(
            init_args: Tuple[Any, ...],
            init_kwargs: Dict[str, Any],
        ) -> Any:
            """Best-effort retrieval of TRL trainer args from constructor inputs."""
            if "args" in init_kwargs:
                return init_kwargs.get("args")
            if len(init_args) >= 3:
                return init_args[2]
            return None

        @staticmethod
        def _is_maxent_requested(training_args: Any) -> bool:
            """Return True when requested objective is MaxEnt (not vanilla GRPO)."""
            return not _coerce_bool(
                getattr(training_args, "train_grpo_objective", True),
                default=True,
            )

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            parent_args = self._resolve_parent_training_args(args, kwargs)
            maxent_requested = self._is_maxent_requested(parent_args)
            parent_alpha_default = 1.0 if maxent_requested else 0.0
            parent_maxent_alpha = _coerce_non_negative_float(
                getattr(parent_args, "maxent_alpha", parent_alpha_default)
                if parent_args is not None
                else parent_alpha_default,
                default=parent_alpha_default,
            )
            # Native TRL pathway policy: always keep parent rollout/compute enabled
            # and apply MaxEnt only as a small objective adjustment.
            shared_rollout_requested = False
            parent_use_vllm_original: Optional[bool] = None
            disabled_parent_vllm = False
            parent_vllm_mode = "server"

            # Keep GRPO on the native TRL compute path. Only MaxEnt uses the shared
            # rollout/loss path and needs parent vLLM init suppressed.
            if parent_args is not None:
                try:
                    parent_use_vllm_original = bool(
                        getattr(parent_args, "use_vllm", False)
                    )
                except (AttributeError, TypeError, ValueError):
                    parent_use_vllm_original = None
                try:
                    parent_vllm_mode = str(
                        getattr(parent_args, "vllm_mode", "server") or "server"
                    ).strip().lower()
                except Exception:
                    parent_vllm_mode = "server"
                if parent_use_vllm_original and shared_rollout_requested:
                    try:
                        setattr(parent_args, "use_vllm", False)
                        disabled_parent_vllm = True
                        LOG.info(
                            "Shared rollout path: disabling parent TRL vLLM initialization during trainer construction "
                            "(objective=maxent, vllm_mode=%s).",
                            parent_vllm_mode,
                        )
                    except Exception as exc:
                        LOG.warning(
                            "Failed to disable parent TRL vLLM initialization before construction: %s",
                            exc,
                        )

            try:
                super().__init__(*args, **kwargs)
            finally:
                if (
                    disabled_parent_vllm
                    and parent_args is not None
                    and parent_use_vllm_original is not None
                ):
                    try:
                        setattr(parent_args, "use_vllm", parent_use_vllm_original)
                    except Exception as exc:
                        LOG.warning(
                            "Failed to restore args.use_vllm after parent construction: %s",
                            exc,
                        )

            self.maxent_enabled = not _coerce_bool(
                getattr(getattr(self, "args", None), "train_grpo_objective", True),
                default=True,
            )
            self.maxent_alpha = _coerce_non_negative_float(
                getattr(getattr(self, "args", None), "maxent_alpha", parent_maxent_alpha),
                default=parent_maxent_alpha,
            )
            # Keep compute path identical to TRL for both GRPO and MaxEnt.
            self._maxent_custom_path = False
            self._shared_rollout_vllm = bool(
                parent_use_vllm_original and shared_rollout_requested
            )
            self._shared_rollout_vllm_mode = parent_vllm_mode
            if self._shared_rollout_vllm:
                if getattr(self, "vllm_client", None) is not None:
                    self._release_parent_vllm_client()
                elif disabled_parent_vllm:
                    LOG.info(
                        "Shared rollout path: parent TRL vLLM client initialization skipped; custom rollout client will handle vLLM sync "
                        "(objective=maxent, vllm_mode=%s).",
                        parent_vllm_mode,
                    )
            if not self.maxent_enabled:
                route_mode = "grpo_native"
            elif self.maxent_alpha > 0.0:
                route_mode = "maxent_native_plus_alpha"
            else:
                route_mode = "maxent_native_alpha0"
            LOG.info(
                "Objective routing selected | mode=%s | train_grpo_objective=%s | maxent_alpha=%s | shared_rollout_vllm=%s | maxent_custom_path=%s",
                route_mode,
                getattr(getattr(self, "args", None), "train_grpo_objective", None),
                self.maxent_alpha,
                self._shared_rollout_vllm,
                self._maxent_custom_path,
            )
            self._maxent_ctx: Optional[TrainingLoopContext] = None
            self._maxent_generator: Optional[CompletionGenerator] = None
            self._last_prepared_batch: Any = None
            self._last_loss_outputs: Any = None
            self._last_batch_diagnostics: Any = None
            self._last_reward_view: Any = None
            self._last_weight_view: Any = None
            self._last_metrics_mode: Optional[str] = None
            self._last_lr: float = 0.0
            self._controller_update_step: int = -1
            self._last_grpo_debug_step: Optional[int] = None
            if self._maxent_custom_path and _ControllerUpdateCallback is not None:
                try:
                    self.add_callback(_ControllerUpdateCallback())
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOG.debug("Failed to attach controller update callback: %s", exc)

        def _release_parent_vllm_client(self) -> None:
            """Fallback cleanup when parent TRL initialized a vLLM client.

            TRL initializes `self.vllm_client` during parent construction when
            `use_vllm=true`. MaxEnt uses a separate shared rollout helper that
            owns vLLM lifecycle; keeping both clients active can cause duplicate
            `init_communicator` calls.
            """

            args = getattr(self, "args", None)
            if args is None or not bool(getattr(args, "use_vllm", False)):
                return
            mode = str(
                getattr(self, "vllm_mode", getattr(args, "vllm_mode", "")) or ""
            ).strip().lower()
            if mode != "server":
                return
            client = getattr(self, "vllm_client", None)
            if client is None:
                return

            if bool(getattr(self.accelerator, "is_main_process", False)):
                close_fn = getattr(client, "close_communicator", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception as exc:
                        LOG.warning(
                            "Failed to close parent TRL vLLM communicator in shared rollout mode: %s",
                            exc,
                        )
            try:
                self.accelerator.wait_for_everyone()
            except Exception:
                pass
            self.vllm_client = None
            LOG.info(
                "Shared rollout path: released parent TRL vLLM client; custom rollout client will handle vLLM sync."
            )

        def get_train_dataloader(self):  # type: ignore[override]
            # Preserve native TRL batching/sampling behavior for GRPO/MaxEnt while
            # adapting worker_init_fn to the active transformers seed_worker signature.
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator

            try:
                from transformers.utils import is_datasets_available
            except Exception:  # pragma: no cover - transformers is required for training
                is_datasets_available = lambda: False  # type: ignore

            if is_datasets_available():
                try:
                    import datasets
                except Exception:
                    datasets = None  # type: ignore
                if datasets is not None and isinstance(train_dataset, datasets.Dataset):
                    train_dataset = self._remove_unused_columns(train_dataset, description="training")
                else:
                    data_collator = self._get_collator_with_removed_columns(
                        data_collator, description="training"
                    )
            else:
                data_collator = self._get_collator_with_removed_columns(
                    data_collator, description="training"
                )

            dataloader_params = {
                "batch_size": self._train_batch_size * self.args.steps_per_generation,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_train_sampler()
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
                worker_init_fn = _build_seed_worker(
                    self.args.dataloader_num_workers, self.args.process_index
                )
                if worker_init_fn is not None:
                    dataloader_params["worker_init_fn"] = worker_init_fn

            return self.accelerator.prepare(
                torch.utils.data.DataLoader(train_dataset, **dataloader_params)
            )

        def _ensure_maxent_context(
            self,
        ) -> Tuple[TrainingLoopContext, CompletionGenerator]:
            ctx = self._maxent_ctx
            generator = self._maxent_generator
            if ctx is not None and generator is not None:
                return ctx, generator

            cfg = getattr(self, "args", None)
            if cfg is None:
                raise RuntimeError("CustomGRPOTrainer missing training args/config.")

            weighting = build_weighting_settings(cfg)
            if hasattr(self, "scale_rewards"):
                try:
                    weighting.scale_rewards = bool(self.scale_rewards)
                except (TypeError, ValueError):
                    pass
            scoring = build_scoring_settings(cfg, weighting)
            generation = build_generation_settings(cfg)
            evaluation = build_evaluation_settings(cfg)
            apply_info_seed(generation, scoring, evaluation, cfg)

            penalty = generation.penalty
            gen_top_k = getattr(cfg, "gen_top_k", None)
            if gen_top_k is None:
                gen_top_k = getattr(cfg, "vllm_top_k", None)
            penalty.gen_top_k = gen_top_k
            gen_best_of = getattr(cfg, "gen_best_of", None)
            if gen_best_of is None:
                gen_best_of = getattr(cfg, "vllm_best_of", None)
            penalty.gen_best_of = gen_best_of
            penalty.gen_frequency_penalty = float(
                getattr(cfg, "gen_frequency_penalty", None)
                or getattr(cfg, "vllm_frequency_penalty", 0.0)
                or 0.0
            )
            penalty.gen_presence_penalty = float(
                getattr(cfg, "gen_presence_penalty", None)
                or getattr(cfg, "vllm_presence_penalty", 0.0)
                or 0.0
            )
            stop_sequences = getattr(cfg, "gen_stop_sequences", None)
            if stop_sequences is None:
                stop_sequences = getattr(cfg, "vllm_stop_sequences", None)
            penalty.gen_stop_sequences = stop_sequences

            cache_size = _resolve_prompt_cache_size(cfg)
            prompt_cache_get = _build_prompt_cache_fn(
                self.processing_class,
                generation.max_prompt_len,
                cache_size,
            )
            scoring.batching.prompt_length_cache_get = prompt_cache_get
            scoring.batching.prompt_cache_size = cache_size

            train_loader = self.get_train_dataloader()
            train_sampler = getattr(train_loader, "sampler", None)

            def _get_ref_model() -> Any:
                ref_model = getattr(self, "ref_model", None)
                return ref_model if ref_model is not None else self.model

            runtime = RuntimeHandles(
                accelerator=self.accelerator,
                model=self.model,
                tokenizer=self.processing_class,
                train_loader=train_loader,
                train_sampler=train_sampler,
                device=self.accelerator.device,
                get_ref_model=_get_ref_model,
                reference_model=getattr(self, "ref_model", None),
                prompt_cache_get=prompt_cache_get,
            )

            reward_funcs = list(getattr(self, "reward_funcs", []) or [])
            reward_weights_raw = getattr(self, "reward_weights", None)
            if isinstance(reward_weights_raw, torch.Tensor):
                reward_weights = [
                    float(val)
                    for val in reward_weights_raw.detach().cpu().tolist()
                ]
            elif reward_weights_raw is None:
                reward_weights = [1.0 for _ in reward_funcs]
            else:
                reward_weights = [float(val) for val in list(reward_weights_raw)]
            reward = RewardSpec(
                reward_funcs=reward_funcs,
                reward_weights=reward_weights,
            )

            num_epochs = int(getattr(cfg, "num_train_epochs", 1) or 1)
            num_generations = int(
                getattr(self, "num_generations", None)
                or getattr(cfg, "num_generations", 1)
                or 1
            )
            grad_accum_steps = int(
                getattr(cfg, "gradient_accumulation_steps", 1) or 1
            )
            max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0) or 0.0)
            total_steps = int(
                getattr(self.state, "max_steps", 0)
                or getattr(cfg, "max_steps", 0)
                or 0
            )
            warmup_steps = int(getattr(cfg, "warmup_steps", 0) or 0)
            lr_scheduler_type = str(
                getattr(cfg, "lr_scheduler_type", "linear") or "linear"
            )
            schedule = OptimizationSchedule(
                num_epochs=num_epochs,
                num_generations=num_generations,
                grad_accum_steps=grad_accum_steps,
                max_grad_norm=max_grad_norm,
                steps_per_epoch=None,
                total_training_steps=total_steps,
                warmup_steps=warmup_steps,
                lr_scheduler_type=lr_scheduler_type,
            )
            handles = OptimizerHandles(
                optimizer=cast(Any, self.optimizer),
                lr_scheduler=self.lr_scheduler,
                base_optimizer=cast(Any, self.optimizer),
                learning_rate=float(getattr(cfg, "learning_rate", 0.0) or 0.0),
            )
            optimization = OptimizationSettings(schedule=schedule, handles=handles)

            controller_paths = _controller_paths(cfg)
            controller_objective = build_controller_objective(cfg, weighting)
            controller_meta_manager = ControllerMetaManager(cfg, weighting)
            loop_settings = LoopSettings(
                generation=generation,
                evaluation=evaluation,
                optimization=optimization,
                scoring=scoring,
                controller=controller_paths,
                controller_objective=controller_objective,
                controller_meta_manager=controller_meta_manager,
            )
            logging_handles = LoggingHandles(
                metric_writer=_NoopMetricWriter(),
                save_checkpoint=lambda _path: None,
                save_strategy="no",
                save_steps=0,
                wandb_run=None,
            )
            ctx = TrainingLoopContext(
                runtime=runtime,
                reward=reward,
                settings=loop_settings,
                logging=logging_handles,
                training_args=cfg,
            )
            controller_loaded = load_controller_state_chain(
                controller_paths, runtime.accelerator, weighting
            )
            _maybe_overwrite_controller_state_from_config(
                ctx, controller_resumed=controller_loaded
            )
            _apply_weighting_overrides_from_config(ctx)

            generation_ctx = GenerationContext(
                accelerator=runtime.accelerator,
                model=runtime.model,
                tokenizer=runtime.tokenizer,
                generation_stats=generation.generation_stats,
                device=runtime.device,
                max_prompt_len=generation.max_prompt_len,
                max_completion_len=generation.max_completion_len,
                gen_temperature=generation.gen_temperature,
                gen_top_p=generation.gen_top_p,
                use_vllm=generation.use_vllm,
                vllm_mode=getattr(generation, "vllm_mode", "server"),
                vllm=generation.vllm,
                penalty=replace(generation.penalty),
            )
            setattr(generation_ctx, "training_args", cfg)
            fail_fast = getattr(cfg, "vllm_client_tag_fail_fast", None)
            if fail_fast is not None:
                setattr(generation_ctx, "vllm_client_tag_fail_fast", fail_fast)
            sync_interval = getattr(cfg, "vllm_sync_interval_steps", None)
            if sync_interval is not None:
                setattr(generation_ctx, "vllm_sync_interval_steps", sync_interval)

            generator = CompletionGenerator(generation_ctx)
            self._maxent_ctx = ctx
            self._maxent_generator = generator
            return ctx, generator

        def _resolve_answer_column(self, mode: str) -> str:
            cfg = getattr(self, "args", None)
            if cfg is None:
                return "answer"
            if mode == "eval":
                eval_col = getattr(cfg, "eval_dataset_solution_column", None)
                if eval_col:
                    return str(eval_col)
            col = getattr(cfg, "dataset_solution_column", None)
            if col:
                return str(col)
            return "answer"

        def _append_metric_value(
            self,
            mode: str,
            key: str,
            value: Any,
            *,
            include_legacy_aliases: bool = True,
        ) -> None:
            numeric = _numeric_or_none(value)
            if numeric is None:
                return
            normalized = _strip_mode_prefix(str(key), mode)
            canonical = _canonical_metric_key(normalized)
            store = self._metrics[mode]
            if canonical == "num_tokens":
                store[canonical] = [numeric]
            else:
                store.setdefault(canonical, []).append(numeric)
            if not include_legacy_aliases:
                return
            for alias in _legacy_metric_aliases(canonical):
                if alias == canonical:
                    continue
                if canonical == "num_tokens":
                    store[alias] = [numeric]
                else:
                    store.setdefault(alias, []).append(numeric)

        def _append_metric_dict(
            self,
            mode: str,
            metrics: Dict[str, Any],
            *,
            include_legacy_aliases: bool = True,
        ) -> None:
            for key, value in metrics.items():
                self._append_metric_value(
                    mode,
                    str(key),
                    value,
                    include_legacy_aliases=include_legacy_aliases,
                )

        def _pack_core_rollout_metrics(
            self,
            *,
            num_tokens: Any,
            completion_mean_length_sampled: Any,
            completion_min_length_sampled: Any,
            completion_max_length_sampled: Any,
            completion_clipped_frac: Any,
            completion_mean_length_terminated: Any,
            completion_min_length_terminated: Any,
            completion_max_length_terminated: Any,
            reward_mean: Any,
            reward_std: Any,
            frac_reward_zero_std: Any,
            rewards_per_func: torch.Tensor,
            reward_func_names: List[str],
            seed_metrics: Optional[Dict[str, Any]],
            diversity_metrics: Optional[Dict[str, Any]],
        ) -> Dict[str, float]:
            def _coerce(value: Any, default: float = 0.0) -> float:
                numeric = _numeric_or_none(value)
                if numeric is None or not math.isfinite(numeric):
                    return float(default)
                return float(numeric)

            clipped_frac = _coerce(completion_clipped_frac)
            clipped_frac = max(0.0, min(1.0, clipped_frac))
            metrics: Dict[str, float] = {
                "num_tokens": _coerce(num_tokens),
                "completions/mean_length_sampled": _coerce(
                    completion_mean_length_sampled
                ),
                "completions/min_length_sampled": _coerce(
                    completion_min_length_sampled
                ),
                "completions/max_length_sampled": _coerce(
                    completion_max_length_sampled
                ),
                "completions/clipped_frac": clipped_frac,
                "completions/mean_length_terminated": _coerce(
                    completion_mean_length_terminated
                ),
                "completions/min_length_terminated": _coerce(
                    completion_min_length_terminated
                ),
                "completions/max_length_terminated": _coerce(
                    completion_max_length_terminated
                ),
                "reward": _coerce(reward_mean),
                "reward_std": _coerce(reward_std),
                "frac_reward_zero_std": _coerce(frac_reward_zero_std),
                "grpo_objective": 0.0 if self.maxent_enabled else 1.0,
                "maxent_objective": 1.0 if self.maxent_enabled else 0.0,
            }
            num_rewards = rewards_per_func.size(1) if rewards_per_func.ndim == 2 else 0
            for idx in range(num_rewards):
                name = (
                    reward_func_names[idx]
                    if idx < len(reward_func_names)
                    else f"reward_{idx}"
                )
                mean_rewards = torch.nanmean(rewards_per_func[:, idx]).item()
                metrics[f"rewards/{name}/mean"] = _coerce(mean_rewards)
                std_rewards = _nanstd(rewards_per_func[:, idx]).item()
                metrics[f"rewards/{name}/std"] = _coerce(std_rewards)
            if seed_metrics:
                for key, val in seed_metrics.items():
                    metrics[f"seed/{key}"] = _coerce(val)
            if diversity_metrics:
                for key, val in diversity_metrics.items():
                    metrics[f"completions/diversity/{key}"] = _coerce(val)
            return metrics

        def _gather_reward_matrix(
            self,
            prepared: Any,
        ) -> Tuple[torch.Tensor, List[str]]:
            num_rewards = len(getattr(self, "reward_funcs", []) or [])
            reward_func_names = list(getattr(self, "reward_func_names", []) or [])
            pairs = prepared.reward_comp.pairs
            local_count = len(getattr(pairs, "completions", []) or [])
            rewards_per_func = torch.full(
                (local_count, num_rewards),
                torch.nan,
                device=self.accelerator.device,
            )
            for idx in range(num_rewards):
                reward_key = f"reward_{idx}"
                reward_vals = prepared.reward_comp.per_reward_values.get(reward_key)
                if reward_vals is not None:
                    rewards_per_func[:, idx] = torch.tensor(
                        reward_vals,
                        dtype=torch.float32,
                        device=self.accelerator.device,
                    )
            return gather(rewards_per_func), reward_func_names

        def _append_textual_rollout_logs(
            self,
            prepared: Any,
            rewards_per_func: torch.Tensor,
            reward_func_names: List[str],
            advantages: Optional[torch.Tensor],
        ) -> None:
            pairs = prepared.reward_comp.pairs
            self._textual_logs["prompt"].extend(gather_object(pairs.prompts))
            self._textual_logs["completion"].extend(gather_object(pairs.completions))
            num_rewards = rewards_per_func.size(1) if rewards_per_func.ndim == 2 else 0
            for idx in range(num_rewards):
                name = (
                    reward_func_names[idx]
                    if idx < len(reward_func_names)
                    else f"reward_{idx}"
                )
                self._textual_logs["rewards"][name].extend(
                    rewards_per_func[:, idx].tolist()
                )
            if advantages is not None:
                self._textual_logs["advantages"].extend(advantages.tolist())

        def _reset_cached_generation_state(self) -> None:
            setattr(self, "_maxent_cached_generation_cycle", -1)
            setattr(self, "_maxent_cached_generation_step", -1)
            setattr(self, "_maxent_cached_generation_slices", [])
            setattr(self, "_maxent_cached_generation_cursor", 0)

        def _answer_from_example(self, mode: str, example: Dict[str, Any]) -> str:
            answer_column = self._resolve_answer_column(mode)
            fallback_keys = [
                answer_column,
                "answer",
                "solution",
                "final_answer",
                "target",
            ]
            for key in fallback_keys:
                if key in example and example[key] is not None:
                    return str(example[key])
            return ""

        def _build_prompt_answer_batch(
            self,
            mode: str,
            examples: List[Dict[str, Any]],
            *,
            dedupe_grouped_inputs: bool,
            log_prefix: str,
        ) -> Dict[str, List[str]]:
            prompts_all = [
                maybe_apply_chat_template(example, self.processing_class)["prompt"]
                for example in examples
            ]
            answers_all = [
                self._answer_from_example(mode, example) for example in examples
            ]
            prompts = prompts_all
            answers = answers_all
            if dedupe_grouped_inputs:
                group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
                if (
                    group_size > 1
                    and len(prompts_all) >= group_size
                    and len(prompts_all) % group_size == 0
                ):
                    dedup_ok = True
                    for start in range(0, len(prompts_all), group_size):
                        prompt_ref = prompts_all[start]
                        answer_ref = answers_all[start]
                        for offset in range(1, group_size):
                            idx = start + offset
                            if (
                                prompts_all[idx] != prompt_ref
                                or answers_all[idx] != answer_ref
                            ):
                                dedup_ok = False
                                break
                        if not dedup_ok:
                            break
                    if dedup_ok:
                        prompts = prompts_all[::group_size]
                        answers = answers_all[::group_size]
                        if self.accelerator.is_main_process:
                            LOG.info(
                                "%s input dedup applied | original=%d unique=%d num_generations=%d",
                                log_prefix,
                                len(prompts_all),
                                len(prompts),
                                group_size,
                            )
            return {"prompt": prompts, "answer": answers}

        def _prepare_training_batch_from_cache(
            self,
            mode: str,
            ctx: TrainingLoopContext,
            generator: CompletionGenerator,
            batch: Dict[str, List[str]],
        ) -> Any:
            if mode != "train":
                self._reset_cached_generation_state()
                return prepare_training_batch(ctx, generator.generate, batch)

            optimizer_step = int(getattr(self.state, "global_step", 0) or 0)
            num_iterations = max(int(getattr(self, "num_iterations", 1) or 1), 1)
            generation_cycle = optimizer_step // num_iterations
            cached_cycle_raw = getattr(self, "_maxent_cached_generation_cycle", -1)
            try:
                cached_cycle = int(cached_cycle_raw)
            except (TypeError, ValueError):
                cached_cycle = -1
            cached_slices = cast(
                List[Dict[str, Any]],
                getattr(self, "_maxent_cached_generation_slices", []),
            )
            if cached_cycle != generation_cycle or not cached_slices:
                retry_limit = int(getattr(ctx.generation, "vllm_rounds_cfg", 0) or 0)
                if retry_limit <= 0:
                    retry_limit = int(
                        getattr(ctx.optimization.schedule, "num_generations", 1) or 1
                    )
                full_gen_batch = prepare_generation_batch(
                    batch,
                    generator.generate,
                    ctx.generation.generation_stats,
                    int(getattr(ctx.optimization.schedule, "num_generations", 1) or 1),
                    max_retry_rounds=retry_limit,
                    seed_augmentation=getattr(ctx.generation, "seed_augmentation", None),
                )
                if full_gen_batch is None:
                    raise RuntimeError(
                        "prepare_generation_batch returned None while caching MaxEnt generations."
                    )
                total_prompts = len(getattr(full_gen_batch, "prompts", []) or [])
                if total_prompts <= 0:
                    raise RuntimeError(
                        "Cached MaxEnt generation batch contained no prompts."
                    )
                steps_cfg = int(getattr(self.args, "steps_per_generation", 0) or 0)
                grad_accum = int(
                    getattr(self.args, "gradient_accumulation_steps", 1) or 1
                )
                target_slices = steps_cfg if steps_cfg > 0 else grad_accum
                target_slices = max(1, target_slices)
                target_slices = min(target_slices, total_prompts)
                chunk_size = max(1, math.ceil(total_prompts / target_slices))
                grouped_completions = list(
                    getattr(full_gen_batch, "grouped_completions", []) or []
                )
                grouped_ref_meta = getattr(full_gen_batch, "grouped_ref_meta", None)
                prompts_full = list(getattr(full_gen_batch, "prompts", []) or [])
                answers_full = list(getattr(full_gen_batch, "answers", []) or [])
                rebuilt_slices: List[Dict[str, Any]] = []
                for start in range(0, total_prompts, chunk_size):
                    end = min(start + chunk_size, total_prompts)
                    comp_subset = [
                        list(group) for group in grouped_completions[start:end]
                    ]
                    if grouped_ref_meta is None:
                        meta_subset = None
                    else:
                        meta_subset = []
                        for group in grouped_ref_meta[start:end]:
                            if group is None:
                                meta_subset.append(None)
                            else:
                                meta_subset.append(list(group))
                    rebuilt_slices.append(
                        {
                            "batch": {
                                "prompt": prompts_full[start:end],
                                "answer": answers_full[start:end],
                            },
                            "grouped_completions": comp_subset,
                            "grouped_ref_meta": meta_subset,
                        }
                    )
                setattr(self, "_maxent_cached_generation_cycle", generation_cycle)
                setattr(self, "_maxent_cached_generation_step", optimizer_step)
                setattr(self, "_maxent_cached_generation_slices", rebuilt_slices)
                setattr(self, "_maxent_cached_generation_cursor", 0)
                cached_slices = rebuilt_slices
                if self.accelerator.is_main_process:
                    LOG.info(
                        "Generation cache refreshed | step=%d cycle=%d num_iterations=%d prompts=%d slices=%d grad_accum=%d steps_per_generation=%d",
                        optimizer_step,
                        generation_cycle,
                        num_iterations,
                        total_prompts,
                        len(rebuilt_slices),
                        grad_accum,
                        steps_cfg,
                    )

            cursor = int(getattr(self, "_maxent_cached_generation_cursor", 0) or 0)
            if not cached_slices:
                raise RuntimeError("Generation cache is unexpectedly empty.")
            slice_idx = cursor % len(cached_slices)
            setattr(self, "_maxent_cached_generation_cursor", cursor + 1)
            slice_payload = cached_slices[slice_idx]
            cached_batch = cast(Dict[str, List[str]], slice_payload["batch"])
            cached_grouped = cast(
                List[List[str]], slice_payload["grouped_completions"]
            )
            cached_meta = cast(
                Optional[List[Optional[List[Any]]]],
                slice_payload["grouped_ref_meta"],
            )

            def _cached_generator(
                _prompts: List[str],
                _num_samples: int,
                per_prompt_counts: Optional[List[int]] = None,
            ) -> Tuple[
                List[List[str]],
                Optional[List[Optional[List[Any]]]],
            ]:
                del _prompts, _num_samples, per_prompt_counts
                grouped_copy = [list(group) for group in cached_grouped]
                if cached_meta is None:
                    meta_copy = None
                else:
                    meta_copy = []
                    for group in cached_meta:
                        if group is None:
                            meta_copy.append(None)
                        else:
                            meta_copy.append(list(group))
                return grouped_copy, meta_copy

            return prepare_training_batch(ctx, _cached_generator, cached_batch)

        def _prepare_rollout_batch(
            self,
            mode: str,
            ctx: TrainingLoopContext,
            generator: CompletionGenerator,
            batch: Dict[str, List[str]],
            *,
            use_generation_cache: bool,
        ) -> Any:
            if use_generation_cache:
                prepared = self._prepare_training_batch_from_cache(
                    mode, ctx, generator, batch
                )
            else:
                if mode != "train":
                    self._reset_cached_generation_state()
                prepared = prepare_training_batch(ctx, generator.generate, batch)
            if prepared is None:
                raise RuntimeError(
                    "prepare_training_batch returned None while preparing rollout batch."
                )
            return prepared

        def _log_grpo_debug(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            if mode != "train":
                return
            step = int(getattr(self.state, "global_step", 0))
            if self._last_grpo_debug_step == step:
                return
            self._last_grpo_debug_step = step

            completion_mask = outputs.get("completion_mask")
            advantages = outputs.get("advantages")
            completion_ids = outputs.get("completion_ids")

            token_mask_sum = None
            completion_length_mean = None
            if isinstance(completion_mask, torch.Tensor):
                try:
                    completion_lengths = completion_mask.sum(1)
                    agg_lengths = self.accelerator.gather(completion_lengths)
                    completion_length_mean = float(agg_lengths.float().mean().item())
                    token_mask_sum = float(agg_lengths.sum().item())
                except Exception:
                    completion_length_mean = None
                    token_mask_sum = None

            advantages_std = None
            if isinstance(advantages, torch.Tensor):
                try:
                    agg_adv = self.accelerator.gather(advantages)
                    advantages_std = float(agg_adv.float().std().item())
                except Exception:
                    advantages_std = None

            reward_std = None
            try:
                reward_history = self._metrics.get(mode, {}).get("reward_std")
                if reward_history:
                    reward_std = float(reward_history[-1])
            except Exception:
                reward_std = None

            local_expected = len(inputs)
            local_actual = (
                int(completion_ids.shape[0])
                if isinstance(completion_ids, torch.Tensor)
                else local_expected
            )
            local_dropped = max(local_expected - local_actual, 0)
            try:
                counts = torch.tensor(
                    [local_expected, local_actual],
                    device=self.accelerator.device,
                    dtype=torch.long,
                )
                agg_counts = self.accelerator.gather(counts)
                expected_total = int(agg_counts[0::2].sum().item())
                actual_total = int(agg_counts[1::2].sum().item())
            except Exception:
                expected_total = local_expected
                actual_total = local_actual
            dropped_total = max(expected_total - actual_total, 0)

            if self.accelerator.is_main_process:
                LOG.info(
                    "GRPO debug | step=%d | token_mask_sum=%s | completion_length_mean=%s | "
                    "advantages_std=%s | reward_std=%s | num_sequences=%d | dropped_groups=%d",
                    step,
                    token_mask_sum,
                    completion_length_mean,
                    advantages_std,
                    reward_std,
                    expected_total,
                    dropped_total,
                )
            self._maybe_update_grpo_beta(mode)

        def _maybe_update_grpo_beta(self, mode: str) -> None:
            if self.maxent_enabled:
                return
            args = getattr(self, "args", None)
            if args is None:
                return
            kl_target = float(getattr(args, "kl_target", 0.0) or 0.0)
            kl_horizon = int(getattr(args, "kl_horizon", 0) or 0)
            kl_ctl_step_size = float(getattr(args, "kl_ctl_step_size", 0.0) or 0.0)
            if kl_target <= 0.0 or kl_horizon <= 0 or kl_ctl_step_size <= 0.0:
                return
            kl_history = self._metrics.get(mode, {}).get("kl")
            if not kl_history:
                return
            try:
                measured_kl = float(kl_history[-1])
            except (TypeError, ValueError):
                return
            if not math.isfinite(measured_kl):
                return
            current_beta = float(getattr(self, "beta", 0.0) or 0.0)
            if current_beta <= 0.0:
                return
            ratio = measured_kl / max(kl_target, 1e-8)
            error = ratio - 1.0
            if abs(error) < 1e-8:
                return
            limit = kl_ctl_step_size
            clipped_error = max(min(error, limit), -limit)
            horizon = max(1, kl_horizon)
            scale = 1.0 + clipped_error / float(horizon)
            if scale <= 0.0:
                scale = 1e-6
            new_beta = max(0.0, current_beta * scale)
            self.beta = new_beta

        def _log_grpo_diversity(
            self,
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            completion_ids = outputs.get("completion_ids")
            if not isinstance(completion_ids, torch.Tensor):
                return
            tokenizer = getattr(self, "processing_class", None)
            decode = getattr(tokenizer, "batch_decode", None)
            if not callable(decode):
                return
            try:
                completions_text = decode(
                    completion_ids, skip_special_tokens=True
                )
            except Exception:
                return
            group_size = max(
                int(getattr(self, "num_generations", 1) or 1), 1
            )
            usable = len(completions_text) - (len(completions_text) % group_size)
            if usable <= 0:
                return
            if usable != len(completions_text):
                completions_text = completions_text[:usable]
            grouped = [
                completions_text[i : i + group_size]
                for i in range(0, usable, group_size)
            ]
            use_tokenizer = (
                tokenizer
                if callable(getattr(tokenizer, "encode", None)) or callable(tokenizer)
                else None
            )
            metrics = _completion_diversity_metrics(
                grouped,
                tokenizer=use_tokenizer,
                accelerator=self.accelerator,
            )
            if metrics:
                for key, val in metrics.items():
                    self._append_metric_value(
                        mode,
                        f"completions/diversity/{key}",
                        float(val),
                    )

        def _compute_policy_maxent_term(
            self,
            outputs: Dict[str, Any],
        ) -> Optional[torch.Tensor]:
            completion_ids = outputs.get("completion_ids")
            if not isinstance(completion_ids, torch.Tensor):
                return None
            completion_mask = outputs.get("completion_mask")
            if not isinstance(completion_mask, torch.Tensor):
                completion_mask = _apply_eos_completion_mask(
                    completion_ids,
                    getattr(self.processing_class, "eos_token_id", None),
                )
            if not isinstance(completion_mask, torch.Tensor):
                return None
            completion_mask = completion_mask.to(
                device=completion_ids.device, dtype=torch.float32
            )
            per_token_logps = outputs.get("old_per_token_logps")
            if not isinstance(per_token_logps, torch.Tensor):
                prompt_ids = outputs.get("prompt_ids")
                prompt_mask = outputs.get("prompt_mask")
                if not isinstance(prompt_ids, torch.Tensor) or not isinstance(
                    prompt_mask, torch.Tensor
                ):
                    return None
                input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                attention_mask = torch.cat([prompt_mask, completion_mask.long()], dim=1)
                logits_to_keep = completion_ids.size(1)
                batch_size = (
                    int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
                    if self.model.training
                    else int(getattr(self.args, "per_device_eval_batch_size", 1) or 1)
                )
                with torch.no_grad():
                    per_token_logps = self._get_per_token_logps(
                        self.model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                    )
                outputs["old_per_token_logps"] = per_token_logps.detach()
            if not isinstance(per_token_logps, torch.Tensor):
                return None
            token_counts = completion_mask.sum(dim=1).clamp(min=1.0)
            sequence_scores_local = (
                per_token_logps.to(torch.float32) * completion_mask
            ).sum(dim=1) / token_counts
            sequence_scores = gather(sequence_scores_local)
            group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
            if sequence_scores.numel() == 0:
                return None
            if sequence_scores.numel() % group_size != 0:
                return None
            grouped_scores = sequence_scores.view(-1, group_size)
            grouped_log_probs = grouped_scores - torch.logsumexp(
                grouped_scores, dim=1, keepdim=True
            )
            surprisal = (-grouped_log_probs).reshape(-1)
            return torch.nan_to_num(surprisal, nan=0.0, posinf=0.0, neginf=0.0)

        def _recompute_global_rewards_for_outputs(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
        ) -> Optional[torch.Tensor]:
            if not inputs:
                return None
            completion_ids = outputs.get("completion_ids")
            if not isinstance(completion_ids, torch.Tensor):
                return None
            completion_mask = outputs.get("completion_mask")
            if not isinstance(completion_mask, torch.Tensor):
                completion_mask = _apply_eos_completion_mask(
                    completion_ids,
                    getattr(self.processing_class, "eos_token_id", None),
                )
            if not isinstance(completion_mask, torch.Tensor):
                return None
            completion_mask = completion_mask.to(
                device=completion_ids.device, dtype=torch.long
            )

            prompts = [example["prompt"] for example in inputs]
            completions_text = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )
            if is_conversational(inputs[0]):
                completions: List[Any] = []
                for prompt, completion in zip(prompts, completions_text):
                    bootstrap = ""
                    if (
                        isinstance(prompt, list)
                        and prompt
                        and isinstance(prompt[-1], dict)
                        and prompt[-1].get("role") == "assistant"
                    ):
                        bootstrap = str(prompt[-1].get("content", ""))
                    completions.append(
                        [{"role": "assistant", "content": f"{bootstrap}{completion}"}]
                    )
            else:
                completions = completions_text

            completion_ids_list = [
                [int(tok.item()) for tok, keep in zip(row, mask_row) if int(keep.item()) != 0]
                for row, mask_row in zip(completion_ids, completion_mask)
            ]
            rewards_per_func_local = torch.zeros(
                (len(prompts), len(self.reward_funcs)),
                device=completion_ids.device,
                dtype=torch.float32,
            )

            keys = [
                key
                for key in inputs[0]
                if key not in {"prompt", "completion", "completion_ids"}
            ]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            reward_processing_classes = list(
                getattr(self, "reward_processing_classes", [None] * len(self.reward_funcs))
            )

            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, torch.nn.Module):
                    reward_processing_class = (
                        reward_processing_classes[i]
                        if i < len(reward_processing_classes)
                        else None
                    )
                    if reward_processing_class is None:
                        return None
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func_local[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    if not callable(reward_func):
                        return None
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        completion_ids=completion_ids_list,
                        **reward_kwargs,
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    rewards_per_func_local[:, i] = torch.tensor(
                        output_reward_func,
                        dtype=torch.float32,
                        device=completion_ids.device,
                    )

            rewards_per_func = gather(rewards_per_func_local)
            reward_weights = getattr(self, "reward_weights", None)
            if isinstance(reward_weights, torch.Tensor):
                weights = reward_weights.to(device=rewards_per_func.device, dtype=torch.float32)
            elif isinstance(reward_weights, (list, tuple)):
                weights = torch.tensor(
                    list(reward_weights),
                    dtype=torch.float32,
                    device=rewards_per_func.device,
                )
            else:
                weights = torch.ones(
                    (len(self.reward_funcs),),
                    dtype=torch.float32,
                    device=rewards_per_func.device,
                )
            if weights.numel() != rewards_per_func.size(1):
                weights = torch.ones(
                    (rewards_per_func.size(1),),
                    dtype=torch.float32,
                    device=rewards_per_func.device,
                )
            rewards = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1)
            return torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        def _apply_maxent_reward_shaping(
            self,
            inputs: List[Dict[str, Any]],
            outputs: Dict[str, Any],
            *,
            mode: str,
        ) -> None:
            if not self.maxent_enabled or self.maxent_alpha <= 0.0:
                return
            advantages = outputs.get("advantages")
            if not isinstance(advantages, torch.Tensor):
                return
            try:
                rewards = self._recompute_global_rewards_for_outputs(inputs, outputs)
                maxent_term = self._compute_policy_maxent_term(outputs)
            except Exception as exc:
                LOG.warning("Skipping MaxEnt reward shaping due to error: %s", exc)
                return
            if rewards is None or maxent_term is None:
                return
            if rewards.numel() != maxent_term.numel():
                return

            alpha = float(self.maxent_alpha)
            shaped_rewards = rewards + alpha * maxent_term.to(rewards.device)
            group_size = max(int(getattr(self, "num_generations", 1) or 1), 1)
            if shaped_rewards.numel() % group_size != 0:
                return
            grouped_rewards = shaped_rewards.view(-1, group_size)
            mean_grouped_rewards = grouped_rewards.mean(dim=1)
            std_grouped_rewards = grouped_rewards.std(dim=1)
            is_std_zero = torch.isclose(
                std_grouped_rewards,
                torch.zeros_like(std_grouped_rewards),
            )
            mean_rep = mean_grouped_rewards.repeat_interleave(group_size, dim=0)
            std_rep = std_grouped_rewards.repeat_interleave(group_size, dim=0)
            all_advantages = shaped_rewards - mean_rep
            if bool(getattr(self, "scale_rewards", False)):
                all_advantages = all_advantages / (std_rep + 1e-4)

            local_batch_size = int(advantages.size(0))
            start = int(getattr(self.accelerator, "process_index", 0) or 0) * local_batch_size
            end = start + local_batch_size
            local_advantages = all_advantages[start:end].to(
                device=advantages.device,
                dtype=advantages.dtype,
            )
            if local_advantages.numel() != advantages.numel():
                return
            outputs["advantages"] = local_advantages

            metric_store = self._metrics[mode]
            reward_val = float(mean_grouped_rewards.mean().item())
            reward_std_val = float(std_grouped_rewards.mean().item())
            reward_zero_std = float(is_std_zero.float().mean().item())
            for key, value in (
                ("reward", reward_val),
                ("reward_std", reward_std_val),
                ("frac_reward_zero_std", reward_zero_std),
            ):
                bucket = metric_store.get(key)
                if isinstance(bucket, list) and bucket:
                    bucket[-1] = value
                else:
                    metric_store.setdefault(key, []).append(value)

            reward_bonus = alpha * maxent_term
            self._append_metric_value(mode, "maxent/alpha", alpha)
            self._append_metric_value(
                mode,
                "maxent/policy_surprisal_mean",
                float(maxent_term.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/policy_surprisal_std",
                float(maxent_term.std(unbiased=False).item()),
            )
            self._append_metric_value(
                mode,
                "maxent/reward_bonus_mean",
                float(reward_bonus.mean().item()),
            )
            self._append_metric_value(
                mode,
                "maxent/reward_bonus_abs_mean",
                float(reward_bonus.abs().mean().item()),
            )

            advantage_log = self._textual_logs.get("advantages")
            if isinstance(advantage_log, list):
                updated = all_advantages.detach().cpu().tolist()
                if len(updated) <= len(advantage_log):
                    advantage_log[-len(updated) :] = updated

        def _get_per_token_logps(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
            logps = super()._get_per_token_logps(*args, **kwargs)
            if self.maxent_enabled:
                return logps
            if self.accelerator.is_main_process:
                step = int(getattr(self.state, "global_step", 0))
                try:
                    requires_grad = bool(getattr(logps, "requires_grad", False))
                except Exception:
                    requires_grad = False
                LOG.info(
                    "GRPO debug | step=%d | token_logp_requires_grad=%s | grad_enabled=%s | logps_shape=%s",
                    step,
                    requires_grad,
                    torch.is_grad_enabled(),
                    getattr(logps, "shape", None),
                )
            return logps

        def _compute_shared_objective_loss(
            self,
            *,
            model: Any,
            inputs: Any,
            return_outputs: bool,
            mode: str,
            ctx: Optional[TrainingLoopContext],
            prepared: Any,
        ) -> Any:
            del model, inputs
            if not self.maxent_enabled:
                raise RuntimeError(
                    "Shared objective loss is MaxEnt-only; GRPO must use TRL native compute_loss."
                )
            if ctx is None or prepared is None:
                raise RuntimeError(
                    "Shared objective loss requires prepared rollout artifacts."
                )

            if mode == "train":
                try:
                    self.state.num_input_tokens_seen += float(prepared.total_input_tokens)
                except (AttributeError, TypeError, ValueError):
                    pass

            loss_outputs, diagnostics = evaluate_losses(
                *build_loss_inputs(
                    prepared.grouped_completions,
                    prepared.weight_stats,
                    prepared.scores,
                    LossInputConfig(
                        clip_cfg=ctx.scoring.clipping,
                        weighting_cfg=ctx.scoring.weighting,
                        ref_stats=prepared.ref_stats,
                        grpo_loss_type=str(
                            (
                                getattr(getattr(ctx, "training_args", None), "loss_type", None)
                                or getattr(
                                    getattr(ctx, "training_args", None),
                                    "grpo_loss_type",
                                    None,
                                )
                                or "bnpo"
                            )
                        ),
                        max_completion_length=int(
                            getattr(ctx.generation, "max_completion_len", 0) or 0
                        ),
                    ),
                ),
                seed_inputs=(
                    prepared.scores.seed_aux
                    if hasattr(prepared.scores, "seed_aux")
                    else None
                ),
                info_seed_lambda=getattr(ctx.scoring, "info_seed_lambda", 0.0),
                info_seed_temperature=getattr(
                    ctx.scoring, "info_seed_temperature", 0.1
                ),
                info_seed_loss_type=getattr(
                    ctx.scoring, "info_seed_loss_type", "infonce"
                ),
                info_seed_alpha_entropy=getattr(
                    ctx.scoring, "info_seed_alpha_entropy", 0.0
                ),
            )
            _maybe_save_seed_heatmap(
                getattr(prepared, "seed_heatmap", None),
                self.state.global_step,
            )
            _log_prompt_objective(ctx, prepared, self.state.global_step)

            current_lr = None
            scheduler = getattr(self, "lr_scheduler", None)
            if scheduler is not None and hasattr(scheduler, "get_last_lr"):
                try:
                    last_lrs = scheduler.get_last_lr()
                    if last_lrs:
                        current_lr = float(last_lrs[0])
                except (TypeError, ValueError, IndexError):
                    current_lr = None
            if current_lr is None:
                optimizer = getattr(self, "optimizer", None)
                param_groups = (
                    getattr(optimizer, "param_groups", None) if optimizer else None
                )
                if param_groups:
                    try:
                        current_lr = float(param_groups[0].get("lr", 0.0))
                    except (TypeError, ValueError):
                        current_lr = None
            if current_lr is None:
                current_lr = scheduled_learning_rate(
                    ctx.optimization.schedule,
                    ctx.optimization.handles,
                    int(self.state.global_step),
                )
            self._last_lr = float(current_lr)

            log_artifacts = LogStepArtifacts(
                loss_outputs=loss_outputs,
                diagnostics=diagnostics,
                grad_norm_scalar=None,
                epoch_progress=float(getattr(self.state, "epoch", 0.0) or 0.0),
            )
            reward_view = summarize_reward_stats(
                self.accelerator, getattr(prepared, "reward_comp", None)
            )
            weight_view = summarize_weight_stats(
                self.accelerator, prepared.weight_stats
            )
            metric_state = SimpleNamespace(
                global_step=int(self.state.global_step),
                num_input_tokens_seen=float(
                    getattr(self.state, "num_input_tokens_seen", 0.0) or 0.0
                ),
                metric_sums={},
                metric_counts={},
            )
            payload = _build_metrics_payload(
                ctx,
                metric_state,
                prepared,
                log_artifacts,
                float(current_lr),
                reward_view=reward_view,
                weight_view=weight_view,
            )
            metrics = build_training_metrics_dict(payload, int(self.state.global_step))
            rewards_per_func, reward_func_names = self._gather_reward_matrix(prepared)

            def _metric_scalar(key: str, default: float = 0.0) -> float:
                value = metrics.get(key, default)
                numeric = _numeric_or_none(value)
                if numeric is None or not math.isfinite(numeric):
                    return float(default)
                return float(numeric)

            core_rollout_metrics = self._pack_core_rollout_metrics(
                num_tokens=float(getattr(self.state, "num_input_tokens_seen", 0.0) or 0.0),
                completion_mean_length_sampled=_metric_scalar(
                    "train/completions/mean_length_sampled"
                ),
                completion_min_length_sampled=_metric_scalar(
                    "train/completions/min_length_sampled"
                ),
                completion_max_length_sampled=_metric_scalar(
                    "train/completions/max_length_sampled"
                ),
                completion_clipped_frac=_metric_scalar("train/completions/clipped_frac"),
                completion_mean_length_terminated=_metric_scalar(
                    "train/completions/mean_length_terminated"
                ),
                completion_min_length_terminated=_metric_scalar(
                    "train/completions/min_length_terminated"
                ),
                completion_max_length_terminated=_metric_scalar(
                    "train/completions/max_length_terminated"
                ),
                reward_mean=_metric_scalar("train/reward"),
                reward_std=_metric_scalar("train/reward_std"),
                frac_reward_zero_std=_metric_scalar("train/frac_reward_zero_std"),
                rewards_per_func=rewards_per_func,
                reward_func_names=reward_func_names,
                seed_metrics=prepared.seed_metrics,
                diversity_metrics=prepared.diversity_metrics,
            )
            metrics.update({f"train/{key}": val for key, val in core_rollout_metrics.items()})
            self._append_metric_dict(mode, metrics)

            adv_samples = getattr(prepared.reward_comp.advantage, "samples", None)
            adv_global = None
            if adv_samples is not None:
                adv_tensor = torch.tensor(
                    adv_samples, dtype=torch.float32, device=self.accelerator.device
                )
                adv_global = gather(adv_tensor)
            self._append_textual_rollout_logs(
                prepared,
                rewards_per_func,
                reward_func_names,
                adv_global,
            )

            self._last_prepared_batch = prepared
            self._last_loss_outputs = loss_outputs
            self._last_batch_diagnostics = diagnostics
            self._last_reward_view = reward_view
            self._last_weight_view = weight_view
            self._last_metrics_mode = mode

            if return_outputs:
                return loss_outputs.loss, {
                    "loss_outputs": loss_outputs,
                    "diagnostics": diagnostics,
                }
            return loss_outputs.loss

        def _compute_grpo_native_loss(
            self,
            *,
            model: Any,
            inputs: Any,
            return_outputs: bool,
            num_items_in_batch: Any = None,
        ) -> Any:
            """Run GRPO through the parent TRL loss implementation only."""
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError as exc:
                # Older TRL signatures may not accept num_items_in_batch.
                if "num_items_in_batch" not in str(exc):
                    raise
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                )

        def _compute_grpo_objective_loss(
            self,
            *,
            model: Any,
            inputs: Any,
            return_outputs: bool,
            mode: str,
            ctx: Optional[TrainingLoopContext],
            prepared: Any,
        ) -> Any:
            del mode, ctx, prepared
            return self._compute_grpo_native_loss(
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
            )

        def _compute_maxent_objective_loss(
            self,
            *,
            model: Any,
            inputs: Any,
            return_outputs: bool,
            mode: str,
            ctx: Optional[TrainingLoopContext],
            prepared: Any,
        ) -> Any:
            return self._compute_shared_objective_loss(
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
                mode=mode,
                ctx=ctx,
                prepared=prepared,
            )

        def compute_loss(  # type: ignore[override]
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            return self._compute_grpo_native_loss(
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        def _prepare_inputs(self, inputs: Any) -> Any:  # type: ignore[override]
            if not self._maxent_custom_path:
                return super()._prepare_inputs(inputs)
            return inputs

        def _apply_controller_updates(self) -> None:
            if not self._maxent_custom_path or not self.model.training:
                return
            if self._last_metrics_mode != "train":
                return
            if self._last_prepared_batch is None or self._last_loss_outputs is None:
                return
            step = int(getattr(self.state, "global_step", 0))
            if step == self._controller_update_step:
                return
            self._controller_update_step = step

            ctx, _ = self._ensure_maxent_context()
            loss_outputs = self._last_loss_outputs
            weight_view = self._last_weight_view
            meta_cfg = getattr(getattr(ctx.scoring, "weighting", None), "controller_meta", None)
            meta_enabled = bool(getattr(meta_cfg, "enabled", False))
            if not meta_enabled:
                maybe_update_beta(ctx.scoring.weighting, loss_outputs.kl_loss_scalar)
            base_lr = max(float(getattr(ctx.optimization.handles, "learning_rate", 0.0)), 1e-12)
            lr_scale = float(self._last_lr) / base_lr if base_lr > 0 else 1.0
            if not meta_enabled:
                try:
                    maybe_update_tau(
                        ctx.scoring.weighting,
                        weight_view,
                        step,
                        lr_scale=lr_scale,
                    )
                except TypeError:
                    maybe_update_tau(ctx.scoring.weighting, weight_view, step)

            controller_manager = getattr(ctx, "controller_meta_manager", None)
            controller_objective = getattr(ctx, "controller_objective", None)
            if controller_objective is not None:
                should_run_meta = (
                    controller_manager.should_run(step)
                    if controller_manager
                    else True
                )
            else:
                should_run_meta = False
            if controller_objective is not None and should_run_meta:
                _cache_meta_stats(ctx.scoring.weighting, weight_view, loss_outputs)
                meta_ctx = ControllerMetaContext(
                    weighting=ctx.scoring.weighting,
                    weight_stats=weight_view,
                    loss_outputs=loss_outputs,
                    prepared_batch=self._last_prepared_batch,
                    global_step=step,
                    lr_scale=lr_scale,
                    kl_value=loss_outputs.kl_loss_scalar,
                    backprop_fn=controller_manager.make_backprop_fn()
                    if controller_manager
                    else None,
                )
                try:
                    gradients = controller_objective.compute(meta_ctx)
                except (RuntimeError, ValueError, TypeError) as exc:
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

            broadcast_controller_state(self.accelerator, ctx.scoring.weighting)
            if self.accelerator.is_main_process:
                save_controller_state(ctx.controller.state_path, ctx.scoring.weighting)

        def _generate_and_score_completions(  # type: ignore[override]
            self, inputs: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            if not self._maxent_custom_path:
                outputs = super()._generate_and_score_completions(inputs)
                mode = "train" if self.model.training else "eval"
                self._log_grpo_diversity(outputs, mode=mode)
                self._apply_maxent_reward_shaping(inputs, outputs, mode=mode)
                if not self.maxent_enabled:
                    self._log_grpo_debug(inputs, outputs, mode=mode)
                return outputs

            device = self.accelerator.device
            mode = "train" if self.model.training else "eval"
            ctx, generator = self._ensure_maxent_context()

            examples = _coerce_examples(inputs)
            if not examples:
                raise RuntimeError("Empty inputs provided to rollout generation.")
            batch = self._build_prompt_answer_batch(
                mode,
                examples,
                dedupe_grouped_inputs=True,
                log_prefix="Rollout",
            )
            prepared = self._prepare_rollout_batch(
                mode,
                ctx,
                generator,
                batch,
                use_generation_cache=False,
            )

            score_batch = prepared.batch_stats.score_batch
            if score_batch is None:
                raise RuntimeError("prepare_training_batch did not build a ScoreBatch.")

            prompt_ids, prompt_mask, _ = _prepare_prompt_slice(
                score_batch.prompt_entries,
                score_batch.max_prompt_len,
                score_batch.pad_token_id,
                score_batch.completion_ids.dtype,
                torch.long,
            )
            prompt_ids = prompt_ids.to(device)
            prompt_mask = prompt_mask.to(device)
            completion_ids = score_batch.completion_ids.to(device)

            eos_token_id = getattr(self.processing_class, "eos_token_id", None)
            completion_mask = _apply_eos_completion_mask(completion_ids, eos_token_id)
            is_eos = (
                completion_ids == eos_token_id
                if eos_token_id is not None
                else torch.zeros_like(completion_ids, dtype=torch.bool)
            )
            completion_lengths = completion_mask.sum(1)

            if self.mask_truncated_completions:
                truncated_completions = ~is_eos.any(dim=1)
                completion_mask = completion_mask * (
                    ~truncated_completions
                ).unsqueeze(1).int()

            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            logits_to_keep = completion_ids.size(1)
            batch_size = (
                self.args.per_device_train_batch_size
                if mode == "train"
                else self.args.per_device_eval_batch_size
            )

            with torch.no_grad():
                if (
                    self.num_iterations > 1
                    or self.args.steps_per_generation
                    > self.args.gradient_accumulation_steps
                ):
                    old_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )
                else:
                    old_per_token_logps = None

            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

            local_count = completion_ids.size(0)
            rewards_per_func, reward_func_names = self._gather_reward_matrix(prepared)
            num_rewards = rewards_per_func.size(1) if rewards_per_func.ndim == 2 else 0

            reward_weights_raw = getattr(self, "reward_weights", None)
            if isinstance(reward_weights_raw, torch.Tensor):
                reward_weights = reward_weights_raw.to(device).unsqueeze(0)
            elif reward_weights_raw is None:
                reward_weights = torch.ones(
                    (1, num_rewards), dtype=torch.float32, device=device
                )
            else:
                reward_weights = torch.tensor(
                    list(reward_weights_raw), dtype=torch.float32, device=device
                ).unsqueeze(0)

            rewards = (rewards_per_func * reward_weights).nansum(dim=1)

            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            is_std_zero = torch.isclose(
                std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
            )
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)

            process_slice = slice(
                self.accelerator.process_index * local_count,
                (self.accelerator.process_index + 1) * local_count,
            )
            all_process_advantages = advantages.clone()
            advantages = advantages[process_slice]

            if mode == "train":
                self.state.num_input_tokens_seen += (
                    self.accelerator.gather(attention_mask.sum()).sum().item()
                )

            agg_completion_lengths = self.accelerator.gather(completion_lengths)
            agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
            term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
            clipped_completions_ratio = 1.0 - len(term_completion_lengths) / max(
                len(agg_completion_lengths), 1
            )
            if len(term_completion_lengths) == 0:
                term_completion_lengths = torch.zeros(1, device=device)
            rollout_metrics = self._pack_core_rollout_metrics(
                num_tokens=float(self.state.num_input_tokens_seen),
                completion_mean_length_sampled=agg_completion_lengths.float().mean().item(),
                completion_min_length_sampled=agg_completion_lengths.float().min().item(),
                completion_max_length_sampled=agg_completion_lengths.float().max().item(),
                completion_clipped_frac=float(clipped_completions_ratio),
                completion_mean_length_terminated=term_completion_lengths.float()
                .mean()
                .item(),
                completion_min_length_terminated=term_completion_lengths.float()
                .min()
                .item(),
                completion_max_length_terminated=term_completion_lengths.float()
                .max()
                .item(),
                reward_mean=mean_grouped_rewards.mean().item(),
                reward_std=std_grouped_rewards.mean().item(),
                frac_reward_zero_std=is_std_zero.float().mean().item(),
                rewards_per_func=rewards_per_func,
                reward_func_names=reward_func_names,
                seed_metrics=prepared.seed_metrics,
                diversity_metrics=prepared.diversity_metrics,
            )
            self._append_metric_dict(mode, rollout_metrics)

            self._append_textual_rollout_logs(
                prepared,
                rewards_per_func,
                reward_func_names,
                all_process_advantages,
            )

            outputs = {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "advantages": advantages,
                "old_per_token_logps": old_per_token_logps,
            }
            return outputs

    CustomGRPOTrainer.__name__ = "CustomGRPOTrainer"
    return ensure_weighting_logging(CustomGRPOTrainer)


def wrap_trl_trainer(trainer_cls: Type[Any]) -> Type[Any]:
    """Ensure a trainer class emits TRL-style logs and metrics."""

    return ensure_weighting_logging(trainer_cls)


__all__ = ["build_custom_grpo_trainer", "wrap_trl_trainer"]
