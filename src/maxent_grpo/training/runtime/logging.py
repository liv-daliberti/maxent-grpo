"""Logging utilities (primarily W&B) for the training stack."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
import importlib
import subprocess
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from maxent_grpo.config import GRPOConfig
from maxent_grpo.telemetry.wandb import init_wandb_training

from .setup import _optional_dependency
import sys

LOG = logging.getLogger(__name__)
_FIRST_WANDB_LOGGED_RUNS: set[Any] = set()
_RUN_META_CACHE: Dict[str, str] = {}


def _ensure_wandb_installed() -> Optional[Any]:
    """Best-effort runtime install of wandb when it's missing."""

    wandb_mod = _optional_dependency("wandb")
    if wandb_mod is not None:
        return wandb_mod
    if os.environ.get("MAXENT_WANDB_INSTALL_ATTEMPTED") == "1":
        return None
    os.environ["MAXENT_WANDB_INSTALL_ATTEMPTED"] = "1"
    pip_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "wandb",
    ]
    try:
        proc = subprocess.run(pip_cmd, check=True, capture_output=True, text=True)
        msg = proc.stdout.strip() or proc.stderr.strip()
        if msg:
            LOG.info("Runtime wandb install output: %s", msg)
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        LOG.warning("Failed to install wandb at runtime: %s", exc)
        return None
    return _optional_dependency("wandb")


def _git_sha() -> str:
    """Return the short git SHA for the current repo if available."""

    env_sha = os.environ.get("MAXENT_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or "unknown"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
        ValueError,
    ):  # pragma: no cover - best-effort metadata
        return "unknown"


def resolve_run_metadata(training_args: Any | None = None) -> Dict[str, str]:
    """Return run-level metadata (git SHA, recipe path) for logging consistency.

    :param training_args: Optional training config used to read ``recipe_path``.
    :returns: Mapping with ``run/git_sha`` and ``run/recipe_path`` keys.
    :rtype: dict[str, str]
    """

    if _RUN_META_CACHE:
        return _RUN_META_CACHE
    recipe_path = None
    if training_args is not None:
        recipe_path = getattr(training_args, "recipe_path", None)
    recipe_path = (
        recipe_path
        or os.environ.get("GRPO_RECIPE_USED")
        or os.environ.get("GRPO_RECIPE")
    )
    _RUN_META_CACHE.update(
        {
            "run/git_sha": _git_sha(),
            "run/recipe_path": recipe_path or "unknown",
        }
    )
    return _RUN_META_CACHE


@lru_cache(maxsize=None)
def _wandb_error_types() -> Tuple[type[BaseException], ...]:
    """Return exception types that should be suppressed during W&B logging.

    :returns: Tuple of exception classes treated as non-fatal during W&B calls.
    :rtype: tuple[type, ...]
    """

    base_exceptions: Tuple[type[BaseException], ...] = (RuntimeError, ValueError)
    error_list: list[type[BaseException]] = []

    modules = []
    if (os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules) and "wandb.errors" not in sys.modules:
        return base_exceptions
    for name in ("wandb.errors", "wandb.errors.errors"):
        mod = sys.modules.get(name)
        if mod is None and not (os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules):
            try:
                mod = importlib.import_module(name)
            except (ImportError, ModuleNotFoundError, ValueError):
                mod = None
        if mod is not None:
            modules.append(mod)

    for errors_module in modules:
        wandb_error = getattr(errors_module, "Error", None)
        comm_error = getattr(errors_module, "CommError", None)
        if isinstance(wandb_error, type) and issubclass(wandb_error, BaseException):
            if wandb_error not in error_list:
                error_list.append(wandb_error)
        if isinstance(comm_error, type) and issubclass(comm_error, BaseException):
            # Ensure specific communication failures (e.g., HTTP 4xx/5xx) never
            # crash training runs even when they do not inherit from Error.
            if comm_error not in error_list:
                error_list.append(comm_error)
    for base in base_exceptions:
        if base not in error_list:
            error_list.append(base)
    return tuple(error_list)


def _report_to_contains(
    report_to: Union[str, Sequence[str], None], target: str
) -> bool:
    """Case-insensitive membership check for ``TrainingArguments.report_to``.

    :param report_to: ``report_to`` value from training arguments (string or list).
    :param target: Target backend name (e.g., ``"wandb"``).
    :returns: ``True`` when ``target`` is present in ``report_to``.
    :rtype: bool
    """

    if report_to is None:
        return False
    if isinstance(report_to, str):
        entries = [report_to]
    else:
        entries = list(report_to)
    target = target.lower()
    return any(str(item).lower() == target for item in entries)


def _torch_rank_zero() -> Optional[bool]:
    """Return True when a torch distributed rank 0 process is running."""

    try:
        import torch

        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except (ImportError, AttributeError):
        return None
    return None


def _env_rank_zero() -> Optional[bool]:
    """Return True when common environment hints describe the primary node."""

    for key in (
        "RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PROCESS_RANK",
        "LOCAL_RANK",
    ):
        value = os.environ.get(key)
        if not value:
            continue
        try:
            return int(value) == 0
        except ValueError:
            continue
    return None


def _is_primary_wandb_process(accelerator: Any) -> bool:
    """Return True when the current process should initialize a W&B run."""

    is_main = getattr(accelerator, "is_main_process", None)
    if isinstance(is_main, bool):
        return is_main
    is_zero = _torch_rank_zero()
    if is_zero is not None:
        return is_zero
    env_zero = _env_rank_zero()
    if env_zero is not None:
        return env_zero
    return True


def _maybe_init_wandb_run(
    accelerator: Any,
    training_args: GRPOConfig,
    wandb_config: Dict[str, Any],
) -> Optional[Any]:
    """Initialize a W&B run when ``report_to`` includes ``wandb``.

    :param accelerator: Accelerator instance used to determine the main process.
    :param training_args: Training configuration providing W&B settings.
    :param wandb_config: Prepared keyword arguments for ``wandb.init``.
    :returns: W&B run handle, or ``None`` if logging is disabled.
    :rtype: object | None
    """

    if not _report_to_contains(getattr(training_args, "report_to", None), "wandb"):
        return None
    if os.environ.get("WANDB_DISABLED") == "true":
        return None
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        # In test runs, default to offline mode unless explicitly overridden.
        os.environ.setdefault("WANDB_MODE", "offline")
    init_wandb_training(training_args)
    if not _is_primary_wandb_process(accelerator):
        os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))
        return None
    wandb = _ensure_wandb_installed()
    if wandb is None:
        LOG.warning(
            "report_to includes wandb but the wandb package is not installed; skipping logging."
        )
        return None

    run_name = getattr(training_args, "run_name", None)
    run_meta = resolve_run_metadata(training_args)
    wandb_config = dict(wandb_config)
    wandb_config.setdefault("run/git_sha", run_meta["run/git_sha"])
    wandb_config.setdefault("run/recipe_path", run_meta["run/recipe_path"])
    wandb_kwargs: Dict[str, Any] = {
        "config": wandb_config,
        "dir": os.environ.get("WANDB_DIR") or os.path.join("var", "artifacts", "wandb"),
    }
    if run_name:
        wandb_kwargs["name"] = run_name
    project = os.environ.get("WANDB_PROJECT")
    if project:
        wandb_kwargs["project"] = project
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        wandb_kwargs["entity"] = entity
    group = os.environ.get("WANDB_RUN_GROUP")
    if group:
        wandb_kwargs["group"] = group
    LOG.info(
        "W&B run metadata | git_sha=%s | recipe=%s",
        run_meta["run/git_sha"],
        run_meta["run/recipe_path"],
    )
    if "PYTEST_CURRENT_TEST" in os.environ and "WANDB_MODE" not in os.environ:
        # Avoid network calls in test runs unless explicitly requested.
        os.environ["WANDB_MODE"] = "offline"
    try:
        _wandb_error_types.cache_clear()
    except AttributeError:
        LOG.debug("wandb error type cache_clear unavailable; proceeding.")
    init_exceptions = _wandb_error_types() + (OSError,)
    try:
        return wandb.init(**wandb_kwargs)
    except init_exceptions as exc:  # pragma: no cover - defensive
        LOG.warning("Failed to initialize W&B run: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive fallback  # pylint: disable=broad-exception-caught
        LOG.warning("Unexpected W&B init failure: %s", exc)
        return None


def log_run_header(training_args: Any | None = None) -> Dict[str, str]:
    """Log a consistent run header with git SHA and recipe path.

    :param training_args: Optional training config used to resolve metadata.
    :returns: Metadata dictionary emitted to the logs.
    :rtype: dict[str, str]
    """

    meta = resolve_run_metadata(training_args)
    LOG.info(
        "Run metadata | git_sha=%s | recipe=%s",
        meta["run/git_sha"],
        meta["run/recipe_path"],
    )
    return meta


def _log_wandb(run: Optional[Any], metrics: Dict[str, Any], step: int) -> None:
    """Safely log metrics to a W&B run.

    :param run: W&B run object returned by ``wandb.init``.
    :param metrics: Metric dictionary to log.
    :param step: Global step to associate with the metrics.
    :returns: ``None``.
    """

    if run is None or not metrics:
        return
    run_key = getattr(run, "id", None) or id(run)
    if run_key not in _FIRST_WANDB_LOGGED_RUNS:
        LOG.info(
            "Logging first metrics to W&B | step=%d | keys=%s",
            step,
            ",".join(sorted(metrics.keys())[:5]) if metrics else "",
        )
        _FIRST_WANDB_LOGGED_RUNS.add(run_key)
    error_types = _wandb_error_types()
    try:
        run.log(metrics, step=step)
    except error_types as exc:  # pragma: no cover - defensive logging
        LOG.warning("Failed to log metrics to W&B: %s", exc)


__all__ = [
    "_FIRST_WANDB_LOGGED_RUNS",
    "_log_wandb",
    "_maybe_init_wandb_run",
    "_report_to_contains",
    "_wandb_error_types",
    "log_run_header",
    "resolve_run_metadata",
]
