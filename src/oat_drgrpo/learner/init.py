"""Learner initialization and fixed-run configuration helpers."""

from __future__ import annotations

import functools
import logging
import math
import os
from typing import List

import torch
from datasets import load_from_disk
from oat.actors.base import ActorBase
from oat.utils.ops import masked_mean, masked_sum

from ..args import ZeroMathArgs, build_fixed_listwise_config
from ..controllers import ListwiseControllerState
from ..listwise import normalize_oat_objective
from ..runtime import patch_oat_learner_datetime, resolve_fixed_oat_exp_suffix
from ..trajectory_dataset import ZeroMathTrajectoryDataset


class ZeroMathInitMixin:
    """Initialize OAT learner state and Dr.X/Dr.GRPO runtime configuration."""

    def _init(self, args: ZeroMathArgs, actors: List[ActorBase]) -> None:
        requested_use_wb = args.use_wb
        args.use_wb = False
        fixed_exp_suffix = resolve_fixed_oat_exp_suffix()
        if fixed_exp_suffix:
            logging.info(
                "Using fixed OAT experiment suffix %s for save_path coordination",
                fixed_exp_suffix,
            )
        with patch_oat_learner_datetime(fixed_exp_suffix):
            super()._init(args, actors)
        self.dataset_builder = ZeroMathTrajectoryDataset
        args.use_wb = requested_use_wb
        if hasattr(self, "strategy") and hasattr(self.strategy, "args"):
            self.strategy.args.use_wb = requested_use_wb
        self.eval_dataset_dict = load_from_disk(args.eval_data)
        if args.test_split != "all":
            self.eval_dataset_dict = {
                k: v for k, v in self.eval_dataset_dict.items() if k in args.test_split
            }
        self.args = args
        self._requested_use_wb = requested_use_wb
        self._wandb = None
        self._wandb_run_id: str | None = None
        self._wandb_run_name = os.path.basename(self.save_path)
        # Dr. GRPO Modification 1: Remove length bias with a constant normalizer.
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )
        self.objective = normalize_oat_objective(args.objective)
        self._listwise_grad_norm_logging_disabled_warned = False
        self._listwise_zero_signal_skip_warned = False
        self._listwise_backward_token_budget_safety_warned = False
        self._listwise_branch_grad_probe_warned = False
        self._listwise_branch_grad_probe_runtime_disabled = False
        self._baseline_grad_norm_logging_disabled_warned = False
        self._invalid_scoring_token_ids_warned_contexts = set()
        self._invalid_logit_columns_warned_contexts = set()
        self._fixed_listwise_tau: float | None = None
        self._fixed_listwise_beta: float | None = None
        self._fixed_listwise_config: dict[str, object] = {}
        self._maxent_correctness_schedule_any_correct_ema: float | None = None
        self._policy_grad_probe_params: tuple[torch.nn.Parameter, ...] | None = None
        if self.objective == "maxent_listwise":
            self._fixed_listwise_config = build_fixed_listwise_config(args)
            if not bool(args.maxent_tau_learnable) and not bool(
                args.maxent_tau_controller_enabled
            ):
                self._fixed_listwise_tau = float(args.maxent_tau)
            if not bool(args.maxent_beta_controller_enabled):
                self._fixed_listwise_beta = float(args.beta)
        self._maxent_controller_state = ListwiseControllerState(
            tau_log=math.log(max(float(args.maxent_tau), 1e-8))
        )
        self._maxent_tau_log: torch.nn.Parameter | None = None
        self._maxent_tau_optimizer: torch.optim.Optimizer | None = None
        if bool(args.maxent_tau_learnable):
            self._maxent_tau_log = torch.nn.Parameter(
                torch.tensor(
                    float(self._maxent_controller_state.tau_log),
                    dtype=torch.float32,
                )
            )
            self._maxent_tau_optimizer = torch.optim.Adam(
                [self._maxent_tau_log],
                lr=float(args.maxent_tau_lr),
            )
            self._sync_maxent_tau_from_state()
        if bool(args.maxent_beta_controller_enabled) and float(args.beta) <= 0.0:
            logging.warning(
                "Listwise beta controller is enabled with initial beta=%s; the "
                "multiplicative KL controller will remain at zero until beta is "
                "initialized above zero.",
                args.beta,
            )
        if (
            self._fixed_listwise_tau is not None
            or self._fixed_listwise_beta is not None
        ):
            logging.info(
                "Locking listwise hyperparameters: tau=%s beta=%s "
                "(tau_learnable=%s tau_controller=%s beta_controller=%s)",
                (
                    self._fixed_listwise_tau
                    if self._fixed_listwise_tau is not None
                    else float(args.maxent_tau)
                ),
                (
                    self._fixed_listwise_beta
                    if self._fixed_listwise_beta is not None
                    else float(args.beta)
                ),
                bool(args.maxent_tau_learnable),
                bool(args.maxent_tau_controller_enabled),
                bool(args.maxent_beta_controller_enabled),
            )
            self._enforce_fixed_listwise_hparams()
        self._prompt_batches_consumed_total = 0
