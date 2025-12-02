"""Lightweight logging helpers to mirror MaxEnt metrics inside TRL trainers.

These utilities attach a small mixin to the GRPOTrainer so per-step logs also
include the tau/beta and controller diagnostics used by the custom MaxEnt loop.
The helpers are dependency-light and tolerate missing transformer/TRL pieces so
unit tests can exercise them with SimpleNamespace stubs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _numeric_or_none(value: Any) -> Optional[float]:
    """Return a finite float or ``None`` when conversion fails."""

    if isinstance(value, bool):
        return None
    try:  # Handle plain numbers, numpy scalars, and torch scalars
        candidate = float(value)
    except (TypeError, ValueError):
        item_fn = getattr(value, "item", None)
        if callable(item_fn):
            try:
                candidate = float(item_fn())
            except (TypeError, ValueError):
                return None
        else:
            return None
    return candidate if math.isfinite(candidate) else None


def _with_prefix(prefix: str, key: str) -> str:
    """Helper to attach a prefix if not already present."""

    return key if key.startswith(prefix) else f"{prefix}{key}"


def _fix_clipped_ratio(metrics: Dict[str, Any], args: Any) -> None:
    """Clamp and normalize TRL's negative clipped_ratio counts into a [0, 1] ratio."""

    num_generations = _numeric_or_none(getattr(args, "num_generations", None)) or 0.0
    denom = max(1.0, num_generations)
    for key in list(metrics.keys()):
        if "completions/clipped_ratio" not in key:
            continue
        val = _numeric_or_none(metrics.pop(key))
        if val is None:
            continue
        val = float(val)
        if val < 0.0:
            val = -val / denom  # TRL currently emits a negative count
        metrics[key.replace("clipped_ratio", "clipped_frac")] = max(
            0.0, min(1.0, float(val))
        )


def _augment_loss_metrics(metrics: Dict[str, Any]) -> None:
    """Mirror base loss/KL logs under train-prefixed keys for consistency."""

    loss_val = _numeric_or_none(metrics.get("loss"))
    if loss_val is not None:
        metrics.setdefault("train/loss/total", float(loss_val))
    kl_val = _numeric_or_none(metrics.get("kl"))
    if kl_val is not None:
        metrics.setdefault("train/kl", float(kl_val))
        metrics.setdefault("train/loss/kl", float(kl_val))
    # Heuristically mirror policy/clip losses if TRL exposed them in metrics.
    key_aliases = {
        "train/loss/policy": ["policy_loss", "loss/policy", "train/policy_loss"],
        "train/loss/clip": ["clip_loss", "loss/clip", "train/clip_loss"],
        "train/loss/value": ["value_loss", "loss/value", "train/value_loss"],
    }
    for target, candidates in key_aliases.items():
        if target in metrics:
            continue
        for cand in candidates:
            cand_val = _numeric_or_none(metrics.get(cand))
            if cand_val is not None:
                metrics[target] = float(cand_val)
                break


def _merge_loss_components_from_trainer(metrics: Dict[str, Any], trainer: Any) -> None:
    """Inject loss sub-components captured from compute_loss into metrics."""

    comp = getattr(trainer, "_last_loss_components", None)
    if not isinstance(comp, dict):
        return
    for key, val in comp.items():
        val_num = _numeric_or_none(val)
        if val_num is None:
            continue
        metrics.setdefault(f"train/loss/{key}", float(val_num))


def _normalize_prefixes(
    metrics: Dict[str, Any], is_eval: bool = False
) -> Dict[str, Any]:
    """Return a copy of metrics with bare keys moved under train/ or eval/."""

    prefix = "eval/" if is_eval else "train/"
    out: Dict[str, Any] = {}
    for key, val in metrics.items():
        if key.startswith("eval/") or key.startswith("train/"):
            out[key] = val
            if key.endswith("weighting/tau"):
                out.setdefault("train/tau", val)
            if key.endswith("weighting/beta"):
                out.setdefault("train/beta", val)
            continue
        if key.startswith("train/weighting/"):
            if key.endswith("tau"):
                out.setdefault("train/tau", val)
            out[key] = val
            continue
        # Eval-prefixed keys from TRL (e.g., eval_loss, eval_reward)
        if key.startswith("eval_"):
            subkey = key[len("eval_") :]
            out[_with_prefix("eval/", subkey)] = val
            continue
        # Train-prefixed aliases (e.g., train_loss) — normalize to train/
        if key.startswith("train_"):
            subkey = key[len("train_") :]
            out[_with_prefix("train/", subkey)] = val
            continue
        if key == "loss":
            out[f"{prefix}loss/total"] = val
            continue
        if key == "kl":
            out[f"{prefix}kl"] = val
            out[f"{prefix}loss/kl"] = val
            continue
        if key in {"reward", "reward_std"}:
            out[f"{prefix}{key}"] = val
            continue
        if key in {"eval_reward", "eval_reward_std"}:
            out[f"eval/{key.replace('eval_', '')}"] = val
            continue
        if key.startswith("rewards/"):
            out[f"{prefix}{key}"] = val
            continue
        if key.startswith("eval_rewards/"):
            out[f"eval/{key[len('eval_'):]}"] = val
            continue
        if key.startswith("completions/"):
            out[f"{prefix}{key}"] = val
            continue
        if key.startswith("eval_completions/"):
            out[f"eval/{key[len('eval_'):]}"] = val
            continue
        if key == "frac_reward_zero_std":
            out[f"{prefix}reward/zero_fraction"] = val
            continue
        if key == "eval_frac_reward_zero_std":
            out["eval/reward/zero_fraction"] = val
            continue
        if key in {"beta", "kl_coeff", "kl_coef", "kl_coefficient"}:
            out[f"{prefix}weighting/beta"] = val
            continue
        if key == "tau":
            out[f"{prefix}weighting/tau"] = val
            continue
        if key.startswith("kl_controller_"):
            out[f"{prefix}kl_controller/{key[len('kl_controller_'):] }"] = val
            continue
        if key.startswith("train/weighting/"):
            if key.endswith("tau"):
                out.setdefault("train/tau", val)
            if key.endswith("beta"):
                out.setdefault("train/beta", val)
            out[key] = val
            continue
        out[_with_prefix(prefix, key)] = val
    return out


class _WeightingMetricHelper:
    """Helper that derives tau/beta metrics from a trainer + its args."""

    def __init__(self, args: Any):
        self._args = args
        self._prev_tau: Optional[float] = None
        self._prev_beta: Optional[float] = None

    def _current_tau(self, trainer: Any) -> float:
        for attr in ("tau", "maxent_tau"):
            tau_val = _numeric_or_none(getattr(trainer, attr, None))
            if tau_val is not None:
                return tau_val
        args = getattr(trainer, "args", self._args)
        tau_val = _numeric_or_none(getattr(args, "maxent_tau", None))
        return tau_val if tau_val is not None else 0.0

    def _current_beta(self, trainer: Any) -> float:
        kl_ctl = getattr(trainer, "kl_ctl", None)
        for candidate in (
            getattr(kl_ctl, "value", None),
            getattr(kl_ctl, "current_kl_coef", None),
            getattr(kl_ctl, "kl_coef", None),
            getattr(trainer, "kl_coef", None),
            getattr(trainer, "kl_coefficient", None),
            getattr(trainer, "beta", None),
        ):
            beta_val = _numeric_or_none(candidate)
            if beta_val is not None:
                return beta_val
        args = getattr(trainer, "args", self._args)
        for init_field in ("init_kl_coef", "init_kl_coeff"):
            init_val = _numeric_or_none(getattr(args, init_field, None))
            if init_val is not None:
                return init_val
        return 0.0

    def metrics_for_trainer(self, trainer: Any) -> Dict[str, float]:
        """Build the extra weighting metrics for the provided trainer."""

        args = getattr(trainer, "args", self._args)
        tau = float(self._current_tau(trainer))
        beta = float(self._current_beta(trainer))
        denom = max(tau + beta, 1.0)
        delta_tau = tau - self._prev_tau if self._prev_tau is not None else 0.0
        delta_beta = beta - self._prev_beta if self._prev_beta is not None else 0.0
        self._prev_tau = tau
        self._prev_beta = beta

        warmup_steps = getattr(args, "maxent_tau_warmup_steps", -1)
        warmup_steps = warmup_steps if isinstance(warmup_steps, int) else -1
        target_entropy = getattr(args, "maxent_target_weight_entropy", None)
        target_entropy_val = _numeric_or_none(target_entropy)
        state = getattr(trainer, "state", None)
        global_step = (
            getattr(state, "global_step", 0)
            if state is not None
            else getattr(args, "global_step", 0)
        )
        schedule_active = target_entropy_val is not None and global_step > max(
            0, warmup_steps
        )

        q_temperature = _numeric_or_none(getattr(args, "maxent_q_temperature", None))
        q_epsilon = _numeric_or_none(getattr(args, "maxent_q_epsilon", None))
        tau_lr = _numeric_or_none(getattr(args, "maxent_tau_lr", None))
        tau_min = _numeric_or_none(getattr(args, "maxent_tau_min", None))
        tau_max = _numeric_or_none(getattr(args, "maxent_tau_max", None))

        metrics: Dict[str, Optional[float]] = {
            "train/weighting/tau": tau,
            "train/weighting/beta": beta,
            "train/tau": tau,
            "train/beta": beta,
            "train/kl_coeff": beta,
            "train/weighting/weight_norm_denom": denom,
            "train/weight_norm_denom": denom,
            "train/weighting/tau_log": math.log(max(tau, 1e-8)),
            "train/weighting/q_temperature": q_temperature,
            "train/weighting/q_epsilon": q_epsilon,
            "train/weighting/tau_lr": tau_lr,
            "train/weighting/tau_min": tau_min,
            "train/weighting/tau_max": tau_max,
            "train/weighting/tau_warmup_steps": float(warmup_steps),
            "train/weighting/tau_target_entropy": (
                target_entropy_val if target_entropy_val is not None else None
            ),
            "train/weighting/tau_schedule_active": 1.0 if schedule_active else 0.0,
            "train/tau_target_enabled": 1.0 if target_entropy_val is not None else 0.0,
            "train/tau_schedule_active": 1.0 if schedule_active else 0.0,
            "train/weighting/delta_tau": delta_tau,
            "train/weighting/delta_tau_abs": abs(delta_tau),
            "train/weighting/delta_beta": delta_beta,
            "train/weighting/delta_beta_abs": abs(delta_beta),
            "train/delta_tau": delta_tau,
            "train/delta_beta": delta_beta,
            "train/kl_controller/target": _numeric_or_none(
                getattr(args, "kl_target", None)
            ),
            "train/kl_controller/horizon": _numeric_or_none(
                getattr(args, "kl_horizon", None)
            ),
            "train/kl_controller/step_size": _numeric_or_none(
                getattr(args, "kl_ctl_step_size", None)
            ),
            "train/grpo_objective": _numeric_or_none(
                getattr(args, "grpo_objective", 1.0)
            )
            or 1.0,
            "train/maxent_objective": _numeric_or_none(
                getattr(args, "maxent_objective", 0.0)
            )
            or 0.0,
            "train/kl_controller/enabled": (
                1.0
                if (
                    _numeric_or_none(getattr(args, "kl_target", None))
                    not in {None, 0.0}
                    and _numeric_or_none(getattr(args, "kl_horizon", None))
                    not in {
                        None,
                        0.0,
                    }
                    and _numeric_or_none(getattr(args, "kl_ctl_step_size", None))
                    not in {None, 0.0}
                )
                else 0.0
            ),
            "train/kl_controller_enabled": (
                1.0
                if (
                    _numeric_or_none(getattr(args, "kl_target", None))
                    not in {None, 0.0}
                    and _numeric_or_none(getattr(args, "kl_horizon", None))
                    not in {None, 0.0}
                    and _numeric_or_none(getattr(args, "kl_ctl_step_size", None))
                    not in {None, 0.0}
                )
                else 0.0
            ),
        }
        return {
            k: float(v) for k, v in metrics.items() if _numeric_or_none(v) is not None
        }


class _WeightingLoggingMixin:
    """Mixin that injects weighting metrics into Trainer.log."""

    def _init_weighting_logger(self) -> None:
        if getattr(self, "_weighting_metric_helper", None) is None:
            self._weighting_metric_helper = _WeightingMetricHelper(
                getattr(self, "args", None)
            )

    def log(self, logs: Dict[str, Any], *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._init_weighting_logger()
        helper = getattr(self, "_weighting_metric_helper", None)
        merged = dict(logs or {})
        if helper is not None:
            try:
                extra = helper.metrics_for_trainer(self)
                for key, value in extra.items():
                    merged.setdefault(key, value)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # Defensive: helper can rely on optional Trainer attributes
                # that may be missing in lightweight stubs.
                pass
        _merge_loss_components_from_trainer(merged, self)
        _augment_loss_metrics(merged)
        _fix_clipped_ratio(merged, getattr(self, "args", None))
        normalized = _normalize_prefixes(merged, is_eval=False)
        return super().log(normalized, *args, **kwargs)


def ensure_weighting_logging(trainer_cls: type) -> type:
    """Wrap a Trainer subclass to include weighting metric logging once."""

    if not isinstance(trainer_cls, type):
        # Allow callables (e.g., stubs returning SimpleNamespace) to be used like classes.
        callable_trainer = trainer_cls

        class _CallableTrainer(_WeightingLoggingMixin):  # type: ignore[misc]
            _MAXENT_WEIGHTING_LOGGING = True

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._inner = callable_trainer(*args, **kwargs)
                self.logged_kwargs = kwargs

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def log(self, logs: Dict[str, Any], *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
                merged = dict(logs or {})
                _merge_loss_components_from_trainer(merged, self)
                _augment_loss_metrics(merged)
                _fix_clipped_ratio(merged, getattr(self, "args", None))
                normalized = _normalize_prefixes(merged, is_eval=False)
                if hasattr(self._inner, "log"):
                    return self._inner.log(normalized, *args, **kwargs)
                return None

        _CallableTrainer.__name__ = (
            f"WeightingLogged{getattr(trainer_cls, '__name__', 'Callable')}"
        )
        return _CallableTrainer
    if getattr(trainer_cls, "_MAXENT_WEIGHTING_LOGGING", False):
        return trainer_cls

    class _LossCaptureMixin:
        """Capture loss component dicts returned by compute_loss for logging."""

        def compute_loss(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            loss = super().compute_loss(*args, **kwargs)  # type: ignore[misc]
            setattr(self, "_last_loss_components", None)
            if isinstance(loss, tuple) and len(loss) >= 2:
                maybe_components = loss[1]
                if isinstance(maybe_components, dict):
                    try:
                        self._last_loss_components = {
                            str(k): v for k, v in maybe_components.items()
                        }
                    except (
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):  # pragma: no cover - defensive
                        pass
            return loss

    class WeightingLoggedTrainer(_LossCaptureMixin, _WeightingLoggingMixin, trainer_cls):  # type: ignore[misc]
        _MAXENT_WEIGHTING_LOGGING = True

    WeightingLoggedTrainer.__name__ = f"WeightingLogged{trainer_cls.__name__}"
    return WeightingLoggedTrainer


__all__ = ["ensure_weighting_logging"]
