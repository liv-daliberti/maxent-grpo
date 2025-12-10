"""Lightweight logging helpers to mirror MaxEnt metrics inside TRL trainers.

These utilities attach a small mixin to the GRPOTrainer so per-step logs also
include the tau/beta and controller diagnostics used by the custom MaxEnt loop.
The helpers are dependency-light and tolerate missing transformer/TRL pieces so
unit tests can exercise them with SimpleNamespace stubs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

try:  # Optional dependency for callback-based logging patch
    from transformers import TrainerCallback
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    TrainerCallback = None

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

    # Best-effort world size for correcting cross-rank aggregation mistakes.
    world_size = _numeric_or_none(getattr(args, "world_size", None))
    if world_size in (None, 0.0):
        world_size = _numeric_or_none(getattr(args, "num_processes", None))
    if world_size in (None, 0.0):
        world_size = _numeric_or_none(getattr(args, "process_count", None))

    for key in list(metrics.keys()):
        if "completions/clipped_ratio" not in key:
            continue
        val = _numeric_or_none(metrics.pop(key))
        if val is None:
            continue
        val = float(val)
        if val < 0.0:
            if world_size and world_size > 1.0:
                # TRL gathers completions across ranks but divides by local batch size.
                val = 1.0 - ((1.0 - val) / world_size)
            else:
                # Fallback: negative count -> clamp to zero to avoid inflating ratios.
                val = 0.0
        metrics[key.replace("clipped_ratio", "clipped_frac")] = max(0.0, min(1.0, val))


class _WeightingLogCallback(TrainerCallback if TrainerCallback is not None else object):
    """Normalize/log metrics even if a trainer bypasses the log override."""

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs is None:
            return control
        # Inputs are unused but kept for TrainerCallback signature compatibility.
        _ = state
        _ = kwargs
        merged = dict(logs)
        _augment_loss_metrics(merged)
        _fix_clipped_ratio(merged, args)
        normalized = _normalize_prefixes(merged, is_eval=False)
        logs.clear()
        logs.update(normalized)
        return control


def _fix_clipped_ratio_metrics(trainer: Any) -> None:
    """Sanitize in-memory `_metrics` before GRPOTrainer aggregates them."""

    metrics_map = getattr(trainer, "_metrics", None)
    if not isinstance(metrics_map, dict):
        return
    args = getattr(trainer, "args", None)
    num_generations = _numeric_or_none(getattr(args, "num_generations", None)) or 1.0
    denom = max(1.0, num_generations)

    def _normalize(val: Any) -> Optional[float]:
        v = _numeric_or_none(val)
        if v is None:
            return None
        v = float(v)
        if v < 0.0:
            v = -v / denom  # TRL emits a negative count; convert to fraction.
        return max(0.0, min(1.0, v))

    for mode_metrics in metrics_map.values():
        if not isinstance(mode_metrics, dict):
            continue
        for key in list(mode_metrics.keys()):
            if "completions/clipped_ratio" not in key:
                continue
            values = mode_metrics.get(key, [])
            if not isinstance(values, (list, tuple)):
                continue
            normalized: list = []
            for val in values:
                fixed = _normalize(val)
                if fixed is not None:
                    normalized.append(fixed)
            # Keep the original list if nothing was normalized.
            if normalized:
                mode_metrics[key] = normalized
                # Also inject a normalized copy for downstream consumers to avoid
                # surfacing raw negative counts.
                mode_metrics.setdefault(
                    key.replace("clipped_ratio", "clipped_frac"), normalized
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


def patch_trl_grpo_clipped_ratio() -> bool:
    """Best-effort monkeypatch to sanitize GRPOTrainer logs even if unwrapped.

    When users invoke TRL's GRPOTrainer directly (bypassing ensure_weighting_logging),
    this patch adjusts the ``log`` method so negative ``completions/clipped_ratio``
    counts are clamped and renamed to ``clipped_frac`` before they are emitted.
    """

    try:
        import trl.trainer.grpo_trainer as grpo_mod  # type: ignore
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False
    trainer_cls = getattr(grpo_mod, "GRPOTrainer", None)
    if not isinstance(trainer_cls, type):
        return False
    if getattr(trainer_cls, "_MAXENT_CLIPPED_RATIO_PATCH", False):
        return True
    orig_log = getattr(trainer_cls, "log", None)
    if not callable(orig_log):
        return False

    def _patched_log(self, logs: Dict[str, Any], *args: Any, **kwargs: Any):
        merged = dict(logs or {})
        _fix_clipped_ratio(merged, getattr(self, "args", None))
        _fix_clipped_ratio_metrics(self)
        return orig_log(self, merged, *args, **kwargs)

    trainer_cls.log = _patched_log  # type: ignore[assignment]
    setattr(trainer_cls, "_MAXENT_CLIPPED_RATIO_PATCH", True)
    return True


# Attempt the global patch once on import; ignore failures so environments
# without TRL remain unaffected.
try:  # pragma: no cover - exercised in dedicated tests
    patch_trl_grpo_clipped_ratio()
except (ImportError, ModuleNotFoundError, AttributeError):
    pass


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

        def _bool_flag(val: Any) -> Optional[bool]:
            """Return a bool for truthy/falsey flags, preserving ``None``."""

            if isinstance(val, bool):
                return val
            if val is None:
                return None
            try:
                return bool(val)
            except (TypeError, ValueError):
                return None

        train_grpo_flag = _bool_flag(getattr(args, "train_grpo_objective", None))
        # Back-compat aliases used by some recipes/TRL surfaces.
        if train_grpo_flag is None:
            train_grpo_flag = _bool_flag(getattr(args, "grpo_objective", None))
        maxent_flag = _bool_flag(getattr(args, "maxent_objective", None))
        if train_grpo_flag is None and maxent_flag is not None:
            train_grpo_flag = not maxent_flag
        if train_grpo_flag is None:
            train_grpo_flag = True  # default to GRPO when unspecified
        if maxent_flag is None:
            maxent_flag = not train_grpo_flag

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
            "train/grpo_objective": 1.0 if train_grpo_flag else 0.0,
            "train/maxent_objective": 1.0 if maxent_flag else 0.0,
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
        meta_enabled = _bool_flag(getattr(args, "controller_meta_enabled", None))
        meta_lr = _numeric_or_none(getattr(args, "controller_meta_lr", None)) or 0.0
        meta_interval = _numeric_or_none(
            getattr(args, "controller_meta_update_interval", None)
        ) or 0.0
        meta_trunc = _numeric_or_none(
            getattr(args, "controller_meta_truncation_steps", None)
            or getattr(args, "controller_meta_analytic_steps", None)
        ) or 0.0
        meta_use_hessian = 1.0 if _bool_flag(getattr(args, "controller_meta_use_hessian", None)) else 0.0
        metrics.update(
            {
                "train/meta/enabled": 1.0 if meta_enabled else 0.0,
                "train/meta/lr": meta_lr if meta_enabled else 0.0,
                "train/meta/update_interval": meta_interval if meta_enabled else 0.0,
                "train/meta/truncation_steps": meta_trunc if meta_enabled else 0.0,
                "train/meta/use_hessian": meta_use_hessian if meta_enabled else 0.0,
                "train/meta/tau_grad": 0.0,
                "train/meta/beta_grad": 0.0,
                "train/meta/grad_norm": 0.0,
                "train/meta/loss": 0.0,
                "train/meta/tau_projected": 0.0,
                "train/meta/beta_projected": 0.0,
            }
        )
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
        # Prefer the precise loss captured during compute_loss to avoid the
        # 4-decimal rounding applied by the upstream Trainer logger.
        precise_loss = _numeric_or_none(getattr(self, "_last_loss_scalar", None))
        logged_loss = _numeric_or_none(merged.get("loss"))
        if precise_loss is not None:
            merged.setdefault("train/loss/total_raw", precise_loss)
            if logged_loss is None or logged_loss == 0.0:
                merged["loss"] = precise_loss
        _merge_loss_components_from_trainer(merged, self)
        _augment_loss_metrics(merged)
        _fix_clipped_ratio(merged, getattr(self, "args", None))
        # Sanitize TRL's internal metrics so the downstream GRPOTrainer log doesn't
        # reintroduce negative clipped_ratio values.
        _fix_clipped_ratio_metrics(self)
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
            setattr(self, "_last_loss_scalar", None)
            loss_value = loss
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
                loss_value = loss[0]
            try:
                # Capture a precise scalar before upstream rounding.
                if hasattr(loss_value, "mean"):
                    loss_value = loss_value.mean()
                self._last_loss_scalar = float(loss_value.item())  # type: ignore[arg-type]
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            return loss

    class WeightingLoggedTrainer(_LossCaptureMixin, _WeightingLoggingMixin, trainer_cls):  # type: ignore[misc]
        _MAXENT_WEIGHTING_LOGGING = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Belt-and-suspenders: attach callback-based normalization in case
            # parent classes bypass the log override.
            try:
                cb_handler = getattr(self, "callback_handler", None)
                callbacks = getattr(cb_handler, "callbacks", []) if cb_handler else []
                already_added = any(
                    isinstance(cb, _WeightingLogCallback) for cb in callbacks
                )
                if not already_added and hasattr(self, "add_callback"):
                    self.add_callback(_WeightingLogCallback())
            except (AttributeError, RuntimeError, TypeError):
                pass

    WeightingLoggedTrainer.__name__ = f"WeightingLogged{trainer_cls.__name__}"
    return WeightingLoggedTrainer


__all__ = ["ensure_weighting_logging", "patch_trl_grpo_clipped_ratio"]
