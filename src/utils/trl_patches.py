"""
Runtime patches for TRL integrations.

This module hosts monkey patches that we want to apply as soon as the training
entrypoints import our utilities. Keeping the logic here avoids forking TRL
and lets us centralise compatibility fixes.

Currently implemented patches
-----------------------------
- ``ensure_vllm_group_port``: Allows the TRL ``VLLMClient`` weight-sync RPCs
  to honour an environment override for the NCCL ``group_port`` (falling back
  to the launcher-provided ``MASTER_PORT`` when present). This restores the
  historical behaviour where a single port flag (e.g. ``--main_process_port``)
  controlled both Accelerate/Torch and vLLM synchronisation.
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Callable, Optional

LOG = logging.getLogger(__name__)
_PATCH_STATE = {"vllm_group_port": False}


def _resolve_port_from_env(
    primary_var: str,
    fallbacks: tuple[str, ...],
) -> Optional[int]:
    """Return the first valid integer from the provided environment variables."""
    for var in (primary_var, *fallbacks):
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            return int(raw)
        except ValueError:
            LOG.warning(
                "Environment variable %s=%s is not a valid integer; ignoring.",
                var,
                raw,
            )
    return None


def ensure_vllm_group_port(
    *,
    env_var: str = "VLLM_GROUP_PORT",
    fallback_vars: tuple[str, ...] = ("MASTER_PORT", "MAIN_PROCESS_PORT"),
) -> None:
    """
    Monkey patch TRL's ``VLLMClient`` so its NCCL ``group_port`` can be driven
    via environment variables.

    Args:
        env_var: Primary environment variable to read from.
        fallback_vars: Additional environment variables to try when ``env_var``
            is unset (by default we fall back to the launcher-provided
            ``MASTER_PORT``/``MAIN_PROCESS_PORT``).
    """
    if _PATCH_STATE["vllm_group_port"]:
        return

    try:
        from trl.extras.vllm_client import VLLMClient  # type: ignore
    except ImportError:
        LOG.debug("TRL is unavailable; skipping VLLMClient group-port patch.")
        return

    original_init: Callable[..., None] = VLLMClient.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        if "group_port" not in kwargs:
            candidate = _resolve_port_from_env(env_var, fallback_vars)
            if candidate is not None:
                kwargs["group_port"] = candidate
            else:
                LOG.debug(
                    "No %s/%s override found; VLLMClient will use default "
                    "group_port=%s.",
                    env_var,
                    "/".join(fallback_vars),
                    kwargs.get("group_port", "51216"),
                )
        return original_init(self, *args, **kwargs)

    VLLMClient.__init__ = patched_init  # type: ignore[assignment]
    _PATCH_STATE["vllm_group_port"] = True
    LOG.debug(
        "Patched TRL VLLMClient to accept group_port overrides via %s (fallbacks: %s).",
        env_var,
        ", ".join(fallback_vars),
    )


__all__ = ["ensure_vllm_group_port"]
