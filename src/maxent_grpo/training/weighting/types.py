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

"""Weighting-related dataclasses shared across the MaxEnt training loop."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Optional

@dataclass
class ControllerMetaSettings:
    """Meta-controller knobs governing tau/beta adaptation."""

    enabled: bool = False
    method: str = "analytic"
    learning_rate: float = 0.0
    update_interval: int = 1
    objective: str = "potential"
    analytic_steps: int = 1
    optimizer: str = "sgd"
    truncation_steps: int = 1
    use_hessian: bool = False
    last_tau_grad: float = 0.0
    last_beta_grad: float = 0.0

    def to_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the meta-controller settings."""

        return {
            "enabled": bool(self.enabled),
            "method": str(self.method),
            "learning_rate": float(self.learning_rate),
            "update_interval": int(self.update_interval),
            "objective": str(self.objective),
            "analytic_steps": int(self.analytic_steps),
            "optimizer": str(self.optimizer),
            "truncation_steps": int(self.truncation_steps),
            "use_hessian": bool(self.use_hessian),
            "last_tau_grad": float(self.last_tau_grad),
            "last_beta_grad": float(self.last_beta_grad),
        }

    def apply_state(self, payload: Mapping[str, Any]) -> None:
        """Update the meta-controller settings from a serialized payload."""

        if "enabled" in payload:
            self.enabled = bool(payload["enabled"])
        if "method" in payload and payload["method"]:
            self.method = str(payload["method"])
        if "learning_rate" in payload:
            try:
                self.learning_rate = float(payload["learning_rate"])
            except (TypeError, ValueError):
                pass
        if "update_interval" in payload:
            try:
                self.update_interval = max(1, int(payload["update_interval"]))
            except (TypeError, ValueError):
                pass
        if "objective" in payload and payload["objective"]:
            self.objective = str(payload["objective"])
        if "analytic_steps" in payload:
            try:
                self.analytic_steps = max(1, int(payload["analytic_steps"]))
            except (TypeError, ValueError):
                pass
        if "optimizer" in payload and payload["optimizer"]:
            self.optimizer = str(payload["optimizer"])
        if "truncation_steps" in payload:
            try:
                self.truncation_steps = max(1, int(payload["truncation_steps"]))
            except (TypeError, ValueError):
                pass
        if "use_hessian" in payload:
            self.use_hessian = bool(payload["use_hessian"])
        if "last_tau_grad" in payload:
            try:
                self.last_tau_grad = float(payload["last_tau_grad"])
            except (TypeError, ValueError):
                pass
        if "last_beta_grad" in payload:
            try:
                self.last_beta_grad = float(payload["last_beta_grad"])
            except (TypeError, ValueError):
                pass


@dataclass
class QDistributionSettings:
    """Softmax temperature and smoothing for weighting."""

    temperature: float
    epsilon: float


@dataclass
class TauSchedule:
    """Hyperparameters controlling tau adaptation."""

    target_entropy: Optional[float]
    learning_rate: float
    minimum_value: float
    maximum_value: float
    warmup_steps: int


@dataclass
class KlControllerSettings:
    """Controller settings for KL regularization."""

    target: float
    horizon: int
    step_size: float


@dataclass
class WeightNormalizationSettings:
    """Length-normalization flag and denominator scaling."""

    denom: float
    len_norm_ref: bool


@dataclass
class WeightingSettings:
    """Sequence weighting hyperparameters with convenience accessors."""

    tau: float
    beta: float
    normalization: WeightNormalizationSettings
    q_distribution: QDistributionSettings
    tau_schedule: TauSchedule
    kl_controller: KlControllerSettings
    train_grpo_objective: bool
    controller_meta: ControllerMetaSettings = field(default_factory=ControllerMetaSettings)
    controller_state: Optional["TorchControllerState"] = None
    allow_empty_weight_fallback: bool = False

    @property
    def denom(self) -> float:
        """Return the denominator used for weight normalization.

        :returns: Normalization denominator applied to weights.
        :rtype: float
        """
        return self.normalization.denom

    @denom.setter
    def denom(self, value: float) -> None:
        """Update the denominator used for weight normalization.

        :param value: New denominator applied to the weights.
        :type value: float
        """
        self.normalization.denom = value

    @property
    def len_norm_ref(self) -> bool:
        """Return whether reference log-probs are length-normalized.

        :returns: ``True`` when reference stats are length-normalized.
        :rtype: bool
        """
        return self.normalization.len_norm_ref

    @len_norm_ref.setter
    def len_norm_ref(self, value: bool) -> None:
        """Update the reference length-normalization flag.

        :param value: Flag enabling/disabling reference length normalization.
        :type value: bool
        """
        self.normalization.len_norm_ref = value

    @property
    def q_temperature(self) -> float:
        """Return the q-distribution temperature.

        :returns: Temperature applied to the q-distribution softmax.
        :rtype: float
        """
        return self.q_distribution.temperature

    @q_temperature.setter
    def q_temperature(self, value: float) -> None:
        """Update the q-distribution temperature.

        :param value: New softmax temperature.
        :type value: float
        """
        self.q_distribution.temperature = value

    @property
    def q_epsilon(self) -> float:
        """Return the epsilon smoothing factor.

        :returns: Epsilon smoothing applied to the q-distribution.
        :rtype: float
        """
        return self.q_distribution.epsilon

    @q_epsilon.setter
    def q_epsilon(self, value: float) -> None:
        """Update the epsilon smoothing factor.

        :param value: New smoothing factor for the q-distribution.
        :type value: float
        """
        self.q_distribution.epsilon = value

    @property
    def tau_target_entropy(self) -> Optional[float]:
        """Return the target weight entropy.

        :returns: Desired entropy target (``None`` to disable adaptation).
        :rtype: float | None
        """
        return self.tau_schedule.target_entropy

    @tau_target_entropy.setter
    def tau_target_entropy(self, value: Optional[float]) -> None:
        """Update the target weight entropy.

        :param value: Desired entropy target (``None`` disables adaptation).
        :type value: float | None
        """
        self.tau_schedule.target_entropy = value

    @property
    def tau_lr(self) -> float:
        """Return the learning rate for tau adaptation.

        :returns: Scalar learning rate for tau updates.
        :rtype: float
        """
        return self.tau_schedule.learning_rate

    @tau_lr.setter
    def tau_lr(self, value: float) -> None:
        """Update the learning rate for tau adaptation.

        :param value: Learning-rate scalar for tau updates.
        :type value: float
        """
        self.tau_schedule.learning_rate = value

    @property
    def tau_min(self) -> float:
        """Return the minimum tau value.

        :returns: Lower bound applied to tau.
        :rtype: float
        """
        return self.tau_schedule.minimum_value

    @tau_min.setter
    def tau_min(self, value: float) -> None:
        """Update the minimum tau value.

        :param value: Lower bound enforced on tau.
        :type value: float
        """
        self.tau_schedule.minimum_value = value

    @property
    def tau_max(self) -> float:
        """Return the maximum tau value.

        :returns: Upper bound applied to tau.
        :rtype: float
        """
        return self.tau_schedule.maximum_value

    @tau_max.setter
    def tau_max(self, value: float) -> None:
        """Update the maximum tau value.

        :param value: Upper bound enforced on tau.
        :type value: float
        """
        self.tau_schedule.maximum_value = value

    @property
    def tau_warmup_steps(self) -> int:
        """Return the tau warmup horizon.

        :returns: Number of steps used to warm up tau updates.
        :rtype: int
        """
        return self.tau_schedule.warmup_steps

    @tau_warmup_steps.setter
    def tau_warmup_steps(self, value: int) -> None:
        """Update the tau warmup horizon.

        :param value: Number of steps used for tau warmup.
        :type value: int
        """
        self.tau_schedule.warmup_steps = value

    @property
    def kl_target(self) -> float:
        """Return the KL target.

        :returns: Desired KL divergence target.
        :rtype: float
        """
        return self.kl_controller.target

    @kl_target.setter
    def kl_target(self, value: float) -> None:
        """Update the KL target.

        :param value: Desired KL divergence target.
        :type value: float
        """
        self.kl_controller.target = value

    @property
    def kl_horizon(self) -> int:
        """Return the KL controller horizon.

        :returns: Number of steps used for the KL controller horizon.
        :rtype: int
        """
        return self.kl_controller.horizon

    @kl_horizon.setter
    def kl_horizon(self, value: int) -> None:
        """Update the KL controller horizon.

        :param value: Number of steps over which to integrate KL error.
        :type value: int
        """
        self.kl_controller.horizon = value

    @property
    def kl_ctl_step_size(self) -> float:
        """Return the KL controller step size.

        :returns: Step size multiplier used by the KL controller.
        :rtype: float
        """
        return self.kl_controller.step_size

    @kl_ctl_step_size.setter
    def kl_ctl_step_size(self, value: float) -> None:
        """Update the KL controller step size.

        :param value: Step-size multiplier for the KL controller.
        :type value: float
        """
        self.kl_controller.step_size = value


@dataclass
class ControllerStateSnapshot:
    """Serializable controller state describing tau/beta parameters."""

    beta: float
    tau: float
    tau_log: float
    tau_entropy_ema: float
    meta: Dict[str, Any] = field(default_factory=dict)

    STATE_VERSION: ClassVar[int] = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the snapshot to a JSON-friendly mapping."""

        payload: Dict[str, Any] = {
            "beta": float(self.beta),
            "tau": float(self.tau),
            "tau_log": float(self.tau_log),
            "tau_entropy_ema": float(self.tau_entropy_ema),
        }
        if self.meta:
            meta_payload = dict(self.meta)
            meta_payload.setdefault("version", self.STATE_VERSION)
            payload["meta"] = meta_payload
        return payload

    @classmethod
    def from_weighting(cls, weighting_cfg: "WeightingSettings") -> "ControllerStateSnapshot":
        """Build a controller snapshot from the active weighting settings."""

        tau_val = float(weighting_cfg.tau)
        tau_log = getattr(weighting_cfg, "_tau_log", math.log(max(tau_val, 1e-8)))
        tau_entropy_field = getattr(weighting_cfg, "_tau_entropy_ema", float("nan"))
        if not isinstance(tau_entropy_field, (int, float)) or not math.isfinite(
            float(tau_entropy_field)
        ):
            tau_entropy_ema = tau_val
        else:
            tau_entropy_ema = float(tau_entropy_field)
        meta_payload: Dict[str, Any] = {}
        meta_cfg = getattr(weighting_cfg, "controller_meta", None)
        if hasattr(meta_cfg, "to_state"):
            meta_payload = {
                "version": cls.STATE_VERSION,
                "controller": meta_cfg.to_state(),
            }
        return cls(
            beta=float(weighting_cfg.beta),
            tau=tau_val,
            tau_log=float(tau_log),
            tau_entropy_ema=tau_entropy_ema,
            meta=meta_payload,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControllerStateSnapshot":
        """Instantiate a snapshot from a serialized payload."""

        beta = payload.get("beta")
        tau = payload.get("tau")
        tau_log_val = payload.get("tau_log")
        tau_entropy_ema_val = payload.get("tau_entropy_ema")
        if not isinstance(beta, (int, float)) or not isinstance(tau, (int, float)):
            raise ValueError("controller state requires numeric beta/tau")
        tau = float(tau)
        if not isinstance(tau_log_val, (int, float)):
            tau_log = math.log(max(tau, 1e-8))
        else:
            tau_log = float(tau_log_val)
        if not isinstance(tau_entropy_ema_val, (int, float)):
            tau_entropy_ema = tau
        else:
            tau_entropy_ema = float(tau_entropy_ema_val)
        meta_payload = payload.get("meta")
        if not isinstance(meta_payload, dict):
            meta_payload = {}
        return cls(
            beta=float(beta),
            tau=tau,
            tau_log=float(tau_log),
            tau_entropy_ema=float(tau_entropy_ema),
            meta=meta_payload,
        )

    def apply_to_weighting(self, weighting_cfg: "WeightingSettings") -> None:
        """Apply the snapshot contents to a weighting configuration."""

        weighting_cfg.beta = float(self.beta)
        weighting_cfg.tau = float(self.tau)
        if weighting_cfg.train_grpo_objective:
            weighting_cfg.denom = 1.0
        else:
            denom_sum = weighting_cfg.tau + weighting_cfg.beta
            weighting_cfg.denom = denom_sum if denom_sum > 0 else 1.0
        tau_log_val = float(self.tau_log)
        if not math.isfinite(tau_log_val):
            tau_log_val = math.log(max(weighting_cfg.tau, 1e-8))
        setattr(weighting_cfg, "_tau_log", tau_log_val)
        tau_entropy_val = float(self.tau_entropy_ema)
        if not math.isfinite(tau_entropy_val):
            tau_entropy_val = weighting_cfg.tau
        setattr(weighting_cfg, "_tau_entropy_ema", tau_entropy_val)
        meta_payload = {}
        if isinstance(self.meta, dict):
            meta_payload = self.meta.get("controller", {})
        if isinstance(meta_payload, dict):
            meta_cfg = getattr(weighting_cfg, "controller_meta", None)
            if hasattr(meta_cfg, "apply_state"):
                meta_cfg.apply_state(meta_payload)
                setattr(weighting_cfg, "_meta_last_tau_grad", float(getattr(meta_cfg, "last_tau_grad", 0.0)))
                setattr(weighting_cfg, "_meta_last_beta_grad", float(getattr(meta_cfg, "last_beta_grad", 0.0)))
        state = getattr(weighting_cfg, "controller_state", None)
        if state is not None:
            try:
                state.sync_from_scalars(weighting_cfg.tau, weighting_cfg.beta)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass


@dataclass
class WeightStats:
    """Weights per completion and entropy diagnostics."""

    weights_grouped: List[List[float]]
    flat_weights: List[float]
    weight_entropy: float
    weight_entropy_min: float
    weight_entropy_max: float
    advantage_entropy: List[float]


@dataclass
class WeightLoggingView:
    """Aggregated entropy statistics for logging."""

    entropy: float
    entropy_min: float
    entropy_max: float
    advantage_entropy_mean: float
    advantage_entropy_std: float


class TorchControllerState:
    """Torch-backed parameters for tau/beta with sync helpers."""

    def __init__(
        self,
        torch_mod: Any,
        tau_init: float,
        beta_init: float,
        *,
        requires_grad: bool = False,
    ):
        self.torch = torch_mod
        param_cls = getattr(torch_mod.nn, "Parameter")
        tensor_cls = getattr(torch_mod, "tensor")
        dtype = getattr(torch_mod, "float32", None)
        tau_tensor = tensor_cls(float(tau_init), dtype=dtype)
        beta_tensor = tensor_cls(float(beta_init), dtype=dtype)
        self.tau_param = param_cls(tau_tensor, requires_grad=requires_grad)
        self.beta_param = param_cls(beta_tensor, requires_grad=requires_grad)
        self.last_weights = None

    def enable_grad(self) -> None:
        self.tau_param.requires_grad_(True)
        self.beta_param.requires_grad_(True)

    def disable_grad(self) -> None:
        self.tau_param.requires_grad_(False)
        self.beta_param.requires_grad_(False)

    def sync_from_scalars(self, tau: float, beta: float) -> None:
        no_grad = getattr(self.torch, "no_grad", None)
        ctx = no_grad() if callable(no_grad) else nullcontext()
        with ctx:
            self.tau_param.copy_(
                self.torch.tensor(float(tau), dtype=getattr(self.tau_param, "dtype", None))
            )
            self.beta_param.copy_(
                self.torch.tensor(float(beta), dtype=getattr(self.beta_param, "dtype", None))
            )

    def tau_tensor(self, detach: bool = False):
        tensor = self.tau_param.detach() if detach else self.tau_param
        return tensor

    def beta_tensor(self, detach: bool = False):
        tensor = self.beta_param.detach() if detach else self.beta_param
        return tensor

    def parameters(self) -> List[Any]:
        return [self.tau_param, self.beta_param]

    def zero_grad(self) -> None:
        for param in self.parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            zero = getattr(grad, "zero_", None)
            if callable(zero):
                zero()
            else:
                try:
                    param.grad = None  # type: ignore[attr-defined]
                except AttributeError:  # pragma: no cover - defensive cleanup
                    pass


__all__ = [
    "TorchControllerState",
    "ControllerMetaSettings",
    "ControllerStateSnapshot",
    "KlControllerSettings",
    "QDistributionSettings",
    "TauSchedule",
    "WeightLoggingView",
    "WeightNormalizationSettings",
    "WeightStats",
    "WeightingSettings",
]
