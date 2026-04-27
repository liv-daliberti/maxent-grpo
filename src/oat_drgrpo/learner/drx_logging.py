"""Controller and runtime logging helpers for Dr.X/listwise learning."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from ..controllers import (
    clamp_listwise_tau,
    compute_learnable_tau_loss,
    maybe_update_listwise_beta,
    maybe_update_listwise_tau,
    resolve_listwise_target_entropy,
    update_listwise_tau_metric_ema,
)


class ZeroMathDrxLoggingMixin:
    """Adaptive controller and scalar logging helpers for Dr.X/listwise paths."""

    def _enforce_fixed_listwise_hparams(self) -> None:
        if self.objective != "maxent_listwise":
            return
        for name, value in self._fixed_listwise_config.items():
            setattr(self.args, name, value)
        if self._fixed_listwise_tau is not None:
            fixed_tau = clamp_listwise_tau(
                float(self._fixed_listwise_tau),
                tau_min=self.args.maxent_tau_min,
                tau_max=self.args.maxent_tau_max,
            )
            self.args.maxent_tau = float(fixed_tau)
            self._maxent_controller_state.tau_log = math.log(max(fixed_tau, 1e-8))
            if self._maxent_tau_log is not None:
                with torch.no_grad():
                    self._maxent_tau_log.fill_(math.log(max(fixed_tau, 1e-8)))
        if self._fixed_listwise_beta is not None:
            self.args.beta = float(self._fixed_listwise_beta)

    def _sync_maxent_tau_from_state(self) -> float:
        self._enforce_fixed_listwise_hparams()
        if self._fixed_listwise_tau is not None:
            current_tau = clamp_listwise_tau(
                float(self._fixed_listwise_tau),
                tau_min=self.args.maxent_tau_min,
                tau_max=self.args.maxent_tau_max,
            )
            self.args.maxent_tau = float(current_tau)
            self._maxent_controller_state.tau_log = math.log(max(current_tau, 1e-8))
            if self._maxent_tau_log is not None:
                with torch.no_grad():
                    self._maxent_tau_log.fill_(math.log(max(current_tau, 1e-8)))
            return float(current_tau)
        current_tau = float(self.args.maxent_tau)
        if self._maxent_tau_log is not None:
            current_tau = math.exp(float(self._maxent_tau_log.detach().item()))
        current_tau = clamp_listwise_tau(
            current_tau,
            tau_min=self.args.maxent_tau_min,
            tau_max=self.args.maxent_tau_max,
        )
        if self._maxent_tau_log is not None:
            with torch.no_grad():
                self._maxent_tau_log.fill_(math.log(max(current_tau, 1e-8)))
            self._maxent_controller_state.tau_log = float(
                self._maxent_tau_log.detach().item()
            )
        else:
            self._maxent_controller_state.tau_log = math.log(max(current_tau, 1e-8))
        self.args.maxent_tau = float(current_tau)
        return float(current_tau)

    def _maybe_update_learnable_tau(
        self,
        *,
        measured_metric: float | None,
        target_metric: float | None,
        global_step: int,
    ) -> tuple[float, float | None]:
        if self._fixed_listwise_tau is not None:
            return self._sync_maxent_tau_from_state(), None
        current_tau = self._sync_maxent_tau_from_state()
        if self._maxent_tau_log is None or self._maxent_tau_optimizer is None:
            return current_tau, None
        if target_metric is None:
            return current_tau, None
        if global_step <= max(0, int(self.args.maxent_tau_warmup_steps)):
            return current_tau, None
        if not isinstance(measured_metric, (int, float)) or not math.isfinite(
            float(measured_metric)
        ):
            return current_tau, None

        tau_loss = compute_learnable_tau_loss(
            self._maxent_tau_log,
            measured_metric=float(measured_metric),
            target_metric=float(target_metric),
        )
        if tau_loss is None:
            return current_tau, None

        tau_loss_value = float(tau_loss.detach().cpu().item())
        self._maxent_tau_optimizer.zero_grad(set_to_none=True)
        tau_loss.backward()
        self._maxent_tau_optimizer.step()
        return self._sync_maxent_tau_from_state(), tau_loss_value

    def _resolve_tau_target_metric(self, *, global_step: int) -> float | None:
        return resolve_listwise_target_entropy(
            target_entropy=self.args.maxent_tau_target_metric,
            target_entropy_start=self.args.maxent_tau_target_metric_start,
            target_entropy_peak=self.args.maxent_tau_target_metric_peak,
            target_entropy_peak_step=self.args.maxent_tau_target_metric_peak_step,
            target_entropy_final=self.args.maxent_tau_target_metric_final,
            target_entropy_horizon=self.args.maxent_tau_target_metric_horizon,
            global_step=int(global_step),
        )

    def _select_tau_adaptation_metric(
        self,
        *,
        semantic_entropy_mu: float | None,
        exploration_gain_any_correct: float | None,
        exploration_gain_drgrpo: float | None,
    ) -> tuple[str, float | None]:
        metric_name = str(self.args.maxent_tau_adaptation_metric)
        metric_map = {
            "semantic_entropy_mu": semantic_entropy_mu,
            "exploration_gain_any_correct": exploration_gain_any_correct,
            "exploration_gain_drgrpo": exploration_gain_drgrpo,
        }
        return metric_name, metric_map.get(metric_name)

    def _record_listwise_controller_infos(
        self,
        infos: dict[str, torch.Tensor],
        *,
        device: torch.device,
        tau_loss_value: float | None = None,
        active_tau_target_metric: float | None = None,
    ) -> None:
        """Record current controller state into learner infos."""

        args = self.args
        infos["tau"] = torch.tensor(float(args.maxent_tau), device=device)
        infos["beta"] = torch.tensor(float(args.beta), device=device)
        infos["weight_norm_denom"] = torch.tensor(
            float(max(args.maxent_tau, 1e-8)),
            device=device,
        )
        infos["kl_controller_enabled"] = torch.tensor(
            1.0 if bool(args.maxent_beta_controller_enabled) else 0.0,
            device=device,
        )
        infos["tau_learnable_enabled"] = torch.tensor(
            1.0 if bool(args.maxent_tau_learnable) else 0.0,
            device=device,
        )
        infos["tau_controller_enabled"] = torch.tensor(
            1.0 if bool(args.maxent_tau_controller_enabled) else 0.0,
            device=device,
        )
        if tau_loss_value is not None:
            infos["tau_loss"] = torch.tensor(float(tau_loss_value), device=device)
        if active_tau_target_metric is not None:
            infos["listwise_tau_target_metric"] = torch.tensor(
                float(active_tau_target_metric),
                device=device,
            )
        tau_metric_ema = getattr(self._maxent_controller_state, "tau_metric_ema", None)
        if isinstance(tau_metric_ema, (int, float)) and math.isfinite(
            float(tau_metric_ema)
        ):
            infos["listwise_tau_metric_ema"] = torch.tensor(
                float(tau_metric_ema),
                device=device,
            )

    def _record_listwise_tau_adaptation_infos(
        self,
        infos: dict[str, torch.Tensor],
        *,
        device: torch.device,
        tau_metric_name: str,
        tau_metric_value: float | None,
        semantic_entropy_mu: float | None,
        exploration_gain_any_correct: float | None,
        exploration_gain_drgrpo: float | None,
        weight_entropy_controller: float | None = None,
    ) -> None:
        """Record the tau adaptation metric selection and available signals."""

        if weight_entropy_controller is not None:
            infos["listwise_weight_entropy_controller"] = torch.tensor(
                float(weight_entropy_controller),
                device=device,
            )
        if tau_metric_value is not None:
            infos["listwise_tau_adaptation_metric_value"] = torch.tensor(
                float(tau_metric_value),
                device=device,
            )
        infos["listwise_tau_adaptation_metric_is_semantic_entropy_mu"] = torch.tensor(
            1.0 if tau_metric_name == "semantic_entropy_mu" else 0.0,
            device=device,
        )
        infos["listwise_tau_adaptation_metric_is_exploration_gain_any_correct"] = (
            torch.tensor(
                1.0 if tau_metric_name == "exploration_gain_any_correct" else 0.0,
                device=device,
            )
        )
        infos["listwise_tau_adaptation_metric_is_exploration_gain_drgrpo"] = (
            torch.tensor(
                1.0 if tau_metric_name == "exploration_gain_drgrpo" else 0.0,
                device=device,
            )
        )
        if semantic_entropy_mu is not None:
            infos["listwise_tau_signal_semantic_entropy_mu"] = torch.tensor(
                float(semantic_entropy_mu),
                device=device,
            )
        if exploration_gain_any_correct is not None:
            infos["listwise_tau_signal_exploration_gain_any_correct"] = torch.tensor(
                float(exploration_gain_any_correct),
                device=device,
            )
        if exploration_gain_drgrpo is not None:
            infos["listwise_tau_signal_exploration_gain_drgrpo"] = torch.tensor(
                float(exploration_gain_drgrpo),
                device=device,
            )

    def _record_listwise_runtime_infos(
        self,
        infos: dict[str, torch.Tensor],
        *,
        device: torch.device,
        skip_zero_signal_update: bool,
        stats: dict[str, list[torch.Tensor]] | None = None,
    ) -> None:
        """Record static listwise runtime settings for this learner update."""

        args = self.args
        infos["listwise_logprob_chunk_size"] = torch.tensor(
            float(args.maxent_logprob_chunk_size),
            device=device,
        )
        infos["listwise_backward_chunk_size"] = torch.tensor(
            float(args.maxent_backward_chunk_size),
            device=device,
        )
        infos["listwise_backward_token_budget"] = torch.tensor(
            float(args.maxent_backward_token_budget),
            device=device,
        )
        infos["listwise_reference_logprobs_from_model"] = torch.tensor(
            1.0 if args.maxent_reference_logprobs_source == "model" else 0.0,
            device=device,
        )
        infos["listwise_reference_logprobs_from_behavior"] = torch.tensor(
            1.0 if args.maxent_reference_logprobs_source == "behavior" else 0.0,
            device=device,
        )
        infos["listwise_zero_signal_skip"] = torch.tensor(
            1.0 if skip_zero_signal_update else 0.0,
            device=device,
        )
        if stats is not None:
            stats["listwise_zero_signal_skip"].append(
                infos["listwise_zero_signal_skip"].detach()
            )

    def _apply_listwise_controller_updates(
        self,
        infos: dict[str, torch.Tensor],
        *,
        device: torch.device,
        skip_zero_signal_update: bool,
        weight_entropy_controller: float | None,
        semantic_entropy_mu: float | None,
        exploration_gain_any_correct: float | None,
        exploration_gain_drgrpo: float | None,
        measured_kl: float | None = None,
        active_tau_target_metric: float | None = None,
        update_beta_when_skipped: bool = True,
    ) -> None:
        """Apply tau/beta controller updates and record their public state."""

        args = self.args
        tau_loss_value = None
        if not skip_zero_signal_update:
            tau_metric_name, tau_metric_value = self._select_tau_adaptation_metric(
                semantic_entropy_mu=semantic_entropy_mu,
                exploration_gain_any_correct=exploration_gain_any_correct,
                exploration_gain_drgrpo=exploration_gain_drgrpo,
            )
            self._record_listwise_tau_adaptation_infos(
                infos,
                device=device,
                tau_metric_name=tau_metric_name,
                tau_metric_value=tau_metric_value,
                weight_entropy_controller=weight_entropy_controller,
                semantic_entropy_mu=semantic_entropy_mu,
                exploration_gain_any_correct=exploration_gain_any_correct,
                exploration_gain_drgrpo=exploration_gain_drgrpo,
            )
            if bool(args.maxent_tau_learnable):
                update_listwise_tau_metric_ema(
                    self._maxent_controller_state,
                    measured_metric=tau_metric_value,
                )
                target_metric = active_tau_target_metric
                if target_metric is None:
                    target_metric = self._resolve_tau_target_metric(
                        global_step=int(self.global_step),
                    )
                with torch.enable_grad():
                    args.maxent_tau, tau_loss_value = self._maybe_update_learnable_tau(
                        measured_metric=tau_metric_value,
                        target_metric=target_metric,
                        global_step=int(self.global_step),
                    )
            elif bool(args.maxent_tau_controller_enabled):
                args.maxent_tau = maybe_update_listwise_tau(
                    args.maxent_tau,
                    measured_metric=tau_metric_value,
                    global_step=int(self.global_step),
                    state=self._maxent_controller_state,
                    target_metric=args.maxent_tau_target_metric,
                    target_metric_start=args.maxent_tau_target_metric_start,
                    target_metric_peak=args.maxent_tau_target_metric_peak,
                    target_metric_peak_step=args.maxent_tau_target_metric_peak_step,
                    target_metric_final=args.maxent_tau_target_metric_final,
                    target_metric_horizon=args.maxent_tau_target_metric_horizon,
                    tau_lr=args.maxent_tau_lr,
                    tau_min=args.maxent_tau_min,
                    tau_max=args.maxent_tau_max,
                    tau_warmup_steps=args.maxent_tau_warmup_steps,
                )
        if bool(args.maxent_beta_controller_enabled) and (
            update_beta_when_skipped or not skip_zero_signal_update
        ):
            args.beta = maybe_update_listwise_beta(
                args.beta,
                measured_kl=measured_kl,
                kl_target=args.kl_target,
                kl_horizon=args.kl_horizon,
                kl_ctl_step_size=args.kl_ctl_step_size,
            )

        self._enforce_fixed_listwise_hparams()
        self._record_listwise_controller_infos(
            infos,
            device=device,
            tau_loss_value=tau_loss_value,
            active_tau_target_metric=active_tau_target_metric,
        )

    def _listwise_scalar(
        self,
        value: float,
        *,
        device: torch.device | int | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = torch.cuda.current_device()
        return torch.tensor(value, device=device, dtype=torch.float32)

    def _all_gather_prompt_values(
        self,
        value: torch.Tensor,
        *,
        world_size: int | None = None,
    ) -> torch.Tensor:
        if world_size is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = int(getattr(self.strategy, "world_size", 1))
        payload = value.reshape(-1).contiguous()
        gathered = self._all_gather_same_shape_tensor(payload)
        return gathered.reshape(int(world_size), -1)

    def _masked_prompt_mean(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.to(values.dtype)
        return (values * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    def _masked_prompt_corr(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if int(mask.to(torch.int64).sum().item()) < 2:
            return lhs.new_zeros(())
        lhs_sel = lhs[mask].to(torch.float32)
        rhs_sel = rhs[mask].to(torch.float32)
        lhs_centered = lhs_sel - lhs_sel.mean()
        rhs_centered = rhs_sel - rhs_sel.mean()
        denom = lhs_centered.norm() * rhs_centered.norm()
        if float(denom.item()) <= 1e-12:
            return lhs.new_zeros(())
        return ((lhs_centered * rhs_centered).sum() / denom).to(
            dtype=lhs.dtype,
            device=lhs.device,
        )
