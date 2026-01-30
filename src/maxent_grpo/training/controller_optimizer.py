"""Meta-optimizer orchestration for controller updates."""

from __future__ import annotations

import math
import logging
from contextlib import nullcontext
from typing import Optional

from maxent_grpo.config import GRPOConfig
from maxent_grpo.training.runtime import require_torch
from .weighting.logic import _ensure_tau_history
from .weighting.types import WeightingSettings
from .controller_objective import ControllerGradients

LOG = logging.getLogger(__name__)


def _no_grad_context(torch_mod):
    """Return a torch.no_grad context when available, else a nullcontext."""

    ctx = getattr(torch_mod, "no_grad", None)
    if callable(ctx):
        return ctx()
    if ctx is not None and hasattr(ctx, "__enter__"):
        return ctx
    return nullcontext()


class ControllerMetaManager:
    """Manage meta-controller optimizer state and cadence."""

    def __init__(self, cfg: GRPOConfig, weighting: WeightingSettings):
        meta_cfg = getattr(weighting, "controller_meta", None)
        disable_reason = None
        self.enabled = bool(meta_cfg and meta_cfg.enabled)
        if not self.enabled:
            disable_reason = "controller_meta_enabled flag is false"
        self.update_interval = max(
            1, int(getattr(meta_cfg, "update_interval", getattr(cfg, "controller_meta_update_interval", 1)))
        )
        legacy_lr = float(getattr(meta_cfg, "learning_rate", cfg.controller_meta_lr))
        tau_lr = float(getattr(meta_cfg, "tau_learning_rate", getattr(cfg, "controller_meta_tau_lr", 0.0)))
        beta_lr = float(getattr(meta_cfg, "beta_learning_rate", getattr(cfg, "controller_meta_beta_lr", 0.0)))
        self.tau_learning_rate = tau_lr if tau_lr > 0.0 else legacy_lr
        self.beta_learning_rate = beta_lr if beta_lr > 0.0 else legacy_lr
        self.learning_rate = max(self.tau_learning_rate, self.beta_learning_rate, 0.0)
        self.beta_grad_clip = float(getattr(meta_cfg, "beta_grad_clip", getattr(cfg, "controller_meta_beta_grad_clip", 0.0)))
        self.method = str(getattr(meta_cfg, "method", "analytic") or "analytic").lower()
        self.optimizer = str(getattr(meta_cfg, "optimizer", "sgd") or "sgd").lower()
        self.objective_name = str(getattr(meta_cfg, "objective", cfg.controller_meta_objective))
        self.analytic_steps = max(
            1, int(getattr(meta_cfg, "analytic_steps", cfg.controller_meta_analytic_steps or 1))
        )
        self.truncation_steps = max(
            1,
            int(
                getattr(
                    meta_cfg,
                    "truncation_steps",
                    getattr(cfg, "controller_meta_truncation_steps", cfg.controller_meta_analytic_steps or 1),
                )
            ),
        )
        self.use_hessian = bool(getattr(meta_cfg, "use_hessian", getattr(cfg, "controller_meta_use_hessian", False)))
        self._torch = None
        self._controller_state = getattr(weighting, "controller_state", None)
        self._meta_optimizer = None
        self._weighting = weighting
        # NOTE: These updates are not "true" meta-gradients by default. The
        # controller objective returns tau/beta update signals, and (when an
        # optimizer is enabled) we map those signals onto Parameter.grads so we
        # can use standard torch.optim state (e.g., Adam momentum).
        self._requires_optimizer = self.method in (
            "analytic",
            "analytic_grad",
            "potential",
            "first_order",
            "truncated",
            "truncated_backprop",
            "backprop",
        )
        if self.optimizer not in ("sgd", "adam", "adamw"):
            LOG.warning(
                "Unsupported controller_meta_optimizer=%s; falling back to analytic updates.",
                self.optimizer,
            )
            self._requires_optimizer = False
        if self._controller_state is not None:
            if self.enabled:
                self._controller_state.enable_grad()
            else:
                self._controller_state.disable_grad()
        if self.enabled and self._requires_optimizer:
            try:
                self._torch = require_torch("controller_meta")
                if self._controller_state is None:
                    raise RuntimeError("controller_state required for meta optimizer")
                params = self._controller_state.parameters()
                if not params:
                    raise RuntimeError("controller_state missing parameters")
                self._meta_optimizer = self._build_optimizer()
            except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as exc:  # pragma: no cover
                LOG.warning(
                    "Controller meta-optimizer falling back to analytic updates: %s",
                    exc,
                )
                self._requires_optimizer = False
                self._torch = None
                self._meta_optimizer = None
        if self.tau_learning_rate <= 0.0 and self.beta_learning_rate <= 0.0:
            self.enabled = False
            disable_reason = "controller_meta learning rates <= 0"
        proc_index = getattr(cfg, "process_index", getattr(cfg, "local_process_index", 0))
        if proc_index in (0, None):
            if self.enabled:
                update_mode = self.optimizer if self._requires_optimizer else "analytic"
                LOG.info(
                    "Controller meta enabled | method=%s | objective=%s | tau_lr=%.4g | beta_lr=%.4g | beta_grad_clip=%s | update_interval=%d | update_mode=%s",
                    self.method,
                    self.objective_name,
                    self.tau_learning_rate,
                    self.beta_learning_rate,
                    f"{self.beta_grad_clip:.4g}"
                    if self.beta_grad_clip and self.beta_grad_clip > 0
                    else "off",
                    self.update_interval,
                    update_mode,
                )
            else:
                LOG.info(
                    "Controller meta disabled | reason=%s",
                    disable_reason or "flag disabled",
                )

    def should_run(self, global_step: int) -> bool:
        if not self.enabled:
            return False
        return (global_step + 1) % self.update_interval == 0

    def make_backprop_fn(self):
        """Return a callback that computes gradients via autograd."""

        if not (
            self.enabled
            and self._requires_optimizer
            and self._controller_state is not None
            and self._torch is not None
        ):
            return None
        state = self._controller_state
        def _backprop_fn(_inner_steps: int) -> Optional[ControllerGradients]:
            tau_param = state.tau_param
            beta_param = state.beta_param
            tau_grad = tau_param.grad
            beta_grad = beta_param.grad
            if tau_grad is None and beta_grad is None:
                return None
            tau_grad_val = None
            beta_grad_val = None
            if tau_grad is not None:
                val = tau_grad.detach() if hasattr(tau_grad, "detach") else tau_grad
                try:
                    tau_grad_val = float(val.item())
                except (AttributeError, TypeError, ValueError):  # pragma: no cover - numeric fallback
                    tau_grad_val = float(val)
            if beta_grad is not None:
                val = beta_grad.detach() if hasattr(beta_grad, "detach") else beta_grad
                try:
                    beta_grad_val = float(val.item())
                except (AttributeError, TypeError, ValueError):  # pragma: no cover - numeric fallback
                    beta_grad_val = float(val)
            if tau_grad_val is None and beta_grad_val is None:
                return None
            return ControllerGradients(
                tau_grad=tau_grad_val,
                beta_grad=beta_grad_val,
            )

        return _backprop_fn

    def apply_gradients(
        self,
        gradients: Optional[ControllerGradients],
        *,
        lr_scale: float,
    ) -> None:
        """Apply controller updates based on the configured method."""

        if not gradients:
            return
        if self._requires_optimizer and self._meta_optimizer is not None:
            # Populate parameter grads from the controller objective signals when
            # running in analytic mode. Optimizer-based modes expect the grads
            # to already be set on the controller parameters.
            if self.method in ("analytic", "analytic_grad", "potential"):
                self._set_optimizer_grads(gradients)
            self._apply_optimizer_step(lr_scale)
            setattr(self._weighting, "_meta_last_tau_grad", float(gradients.tau_grad or 0.0))
            setattr(self._weighting, "_meta_last_beta_grad", float(gradients.beta_grad or 0.0))
            meta_cfg = getattr(self._weighting, "controller_meta", None)
            if meta_cfg:
                meta_cfg.last_tau_grad = float(gradients.tau_grad or 0.0)
                meta_cfg.last_beta_grad = float(gradients.beta_grad or 0.0)
            return
        self._manual_update(gradients, lr_scale=lr_scale)

    def _set_optimizer_grads(self, gradients: ControllerGradients) -> None:
        """Map controller update signals onto controller Parameter.grads."""

        if self._controller_state is None or self._torch is None:
            return

        state = self._controller_state
        torch_mod = self._torch
        try:
            state.zero_grad()
        except (AttributeError, RuntimeError, TypeError):
            pass

        def _as_grad_tensor(param, value: float):
            dtype = getattr(param, "dtype", None)
            device = getattr(param, "device", None)
            try:
                return torch_mod.tensor(float(value), dtype=dtype, device=device)
            except TypeError:
                return torch_mod.tensor(float(value), dtype=dtype)

        # tau: gradient-descent style update (tau -= lr * tau_grad)
        if isinstance(gradients.tau_grad, (int, float)) and math.isfinite(float(gradients.tau_grad)):
            state.tau_param.grad = _as_grad_tensor(state.tau_param, float(gradients.tau_grad))

        # beta: "tighten KL when kl > target" update (beta += lr * beta_grad),
        # so we flip sign to match optimizer descent convention.
        if isinstance(gradients.beta_grad, (int, float)) and math.isfinite(float(gradients.beta_grad)):
            grad_update_val = float(gradients.beta_grad)
            clip = float(self.beta_grad_clip or 0.0)
            if clip > 0.0 and math.isfinite(clip):
                grad_update_val = max(min(grad_update_val, clip), -clip)
            state.beta_param.grad = _as_grad_tensor(state.beta_param, -grad_update_val)

    def _manual_update(self, gradients: ControllerGradients, *, lr_scale: float) -> None:
        meta_cfg = getattr(self._weighting, "controller_meta", None)
        base_lr = 0.0
        if meta_cfg is not None:
            try:
                base_lr = float(getattr(meta_cfg, "learning_rate", 0.0) or 0.0)
            except (TypeError, ValueError):
                base_lr = 0.0

        lr_scale_val = float(lr_scale) if isinstance(lr_scale, (int, float)) else 1.0
        if not math.isfinite(lr_scale_val):
            lr_scale_val = 1.0
        lr_scale_val = max(lr_scale_val, 0.0)

        lr_tau = base_lr if base_lr > 0.0 else float(self.tau_learning_rate)
        lr_beta = base_lr if base_lr > 0.0 else float(self.beta_learning_rate)
        lr_tau *= lr_scale_val
        lr_beta *= lr_scale_val
        if lr_tau <= 0.0 and lr_beta <= 0.0:
            return

        updated = False
        tau_projected = False
        if isinstance(gradients.tau_grad, (int, float)) and math.isfinite(
            float(gradients.tau_grad)
        ):
            grad_val = float(gradients.tau_grad)
            raw_tau = self._weighting.tau - lr_tau * grad_val
            new_tau = max(self._weighting.tau_min, raw_tau)
            tau_max = float(self._weighting.tau_max)
            if tau_max > 0.0:
                clipped = min(new_tau, tau_max)
                tau_projected = tau_projected or clipped != new_tau
                new_tau = clipped
            if self._weighting.tau_min > 0.0 and new_tau <= self._weighting.tau_min:
                tau_projected = True
            self._weighting.tau = float(new_tau)
            try:
                setattr(self._weighting, "_tau_log", math.log(max(self._weighting.tau, 1e-8)))
            except (AttributeError, TypeError, ValueError):
                pass
            updated = True
            setattr(self._weighting, "_meta_last_tau_grad", float(grad_val))
            if meta_cfg is not None:
                try:
                    meta_cfg.last_tau_grad = float(grad_val)
                except (AttributeError, TypeError, ValueError):
                    pass

        beta_projected = False
        if isinstance(gradients.beta_grad, (int, float)) and math.isfinite(
            float(gradients.beta_grad)
        ):
            raw_grad_val = float(gradients.beta_grad)
            if lr_beta > 0.0:
                clip = float(self.beta_grad_clip or 0.0)
                grad_update_val = raw_grad_val
                if clip > 0.0 and math.isfinite(clip):
                    grad_update_val = max(min(grad_update_val, clip), -clip)
                # Beta tightens KL: when kl > target (grad > 0) beta must increase.
                raw_beta = self._weighting.beta + lr_beta * grad_update_val
                if raw_beta < 0.0:
                    beta_projected = True
                self._weighting.beta = max(float(raw_beta), 0.0)
                updated = True
                setattr(self._weighting, "_meta_last_beta_grad", float(raw_grad_val))
                if meta_cfg is not None:
                    try:
                        meta_cfg.last_beta_grad = float(raw_grad_val)
                    except (AttributeError, TypeError, ValueError):
                        pass

        if not updated:
            return

        _ensure_tau_history(self._weighting)
        if self._weighting.train_grpo_objective:
            self._weighting.denom = 1.0
        else:
            denom_sum = float(self._weighting.tau) + float(self._weighting.beta)
            self._weighting.denom = denom_sum if denom_sum > 0 else 1.0
        setattr(self._weighting, "_meta_tau_projected", bool(tau_projected))
        setattr(self._weighting, "_meta_beta_projected", bool(beta_projected))

        state = getattr(self._weighting, "controller_state", None)
        if state is not None:
            try:
                state.sync_from_scalars(self._weighting.tau, self._weighting.beta)
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                state.zero_grad()
            except (AttributeError, RuntimeError, TypeError):
                pass

    def _build_optimizer(self):
        torch_mod = self._torch
        if torch_mod is None:
            raise RuntimeError("torch is not available for meta optimizer")
        state = self._controller_state
        if state is None:
            raise RuntimeError("controller_state required for meta optimizer")
        # Use param groups so tau/beta can have independent learning rates and
        # optimizer state (e.g., Adam moments).
        param_groups = [
            {"params": [state.tau_param], "base_lr": float(self.tau_learning_rate)},
            {"params": [state.beta_param], "base_lr": float(self.beta_learning_rate)},
        ]
        if self.optimizer == "sgd":
            return torch_mod.optim.SGD(param_groups, lr=self.learning_rate)
        if self.optimizer == "adam":
            return torch_mod.optim.Adam(param_groups, lr=self.learning_rate)
        if self.optimizer == "adamw":
            return torch_mod.optim.AdamW(param_groups, lr=self.learning_rate)
        raise RuntimeError(f"Unsupported controller_meta_optimizer={self.optimizer}")

    def _apply_optimizer_step(self, lr_scale: float) -> None:
        if self._controller_state is None or self._meta_optimizer is None or self._torch is None:
            return
        for group in self._meta_optimizer.param_groups:
            base_lr = group.get("base_lr", self.learning_rate)
            try:
                base_lr_val = float(base_lr)
            except (TypeError, ValueError):
                base_lr_val = float(self.learning_rate)
            group["lr"] = float(base_lr_val * lr_scale)
        self._meta_optimizer.step()
        self._meta_optimizer.zero_grad(set_to_none=True)
        torch_mod = self._torch
        state = self._controller_state
        ctx = _no_grad_context(torch_mod)
        with ctx:
            tau_min = float(self._weighting.tau_min)
            tau_max = float(self._weighting.tau_max)
            if tau_min <= 0.0 and tau_max <= 0.0:
                pass
            else:
                min_val = tau_min if tau_min > 0.0 else None
                max_val = tau_max if tau_max > 0.0 else None
                clamp_fn = getattr(state.tau_param, "clamp_", None)
                if callable(clamp_fn):
                    if min_val is not None or max_val is not None:
                        clamp_fn(min=min_val, max=max_val)
                else:
                    clamp_res = state.tau_param.clamp(min=min_val, max=max_val)
                    state.tau_param.copy_(clamp_res)
            beta_clamp = getattr(state.beta_param, "clamp_", None)
            if callable(beta_clamp):
                beta_clamp(min=0.0)
            else:
                state.beta_param.copy_(state.beta_param.clamp(min=0.0))
        tau_val = float(state.tau_param.detach().item())
        beta_val = float(state.beta_param.detach().item())
        if tau_max > 0.0:
            tau_val = min(tau_val, tau_max)
        if tau_min > 0.0:
            tau_val = max(tau_val, tau_min)
        self._weighting.tau = tau_val
        self._weighting.beta = max(0.0, beta_val)
        try:
            setattr(self._weighting, "_tau_log", math.log(max(self._weighting.tau, 1e-8)))
        except (AttributeError, TypeError, ValueError):
            pass
        state.zero_grad()
        _ensure_tau_history(self._weighting)
        setattr(self._weighting, "_meta_tau_projected", False)
        setattr(self._weighting, "_meta_beta_projected", beta_val <= 0.0)
        if self._weighting.train_grpo_objective:
            self._weighting.denom = 1.0
        else:
            denom_sum = self._weighting.tau + self._weighting.beta
            self._weighting.denom = denom_sum if denom_sum > 0 else 1.0
        try:
            state.sync_from_scalars(self._weighting.tau, self._weighting.beta)
        except (AttributeError, TypeError, ValueError):
            pass
