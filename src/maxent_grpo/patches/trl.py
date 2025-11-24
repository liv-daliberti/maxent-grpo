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
import sys
from functools import wraps
from typing import Callable, Optional

LOG = logging.getLogger(__name__)
_PATCH_STATE = {"vllm_group_port": False}


def _resolve_port_from_env(
    primary_var: str,
    fallbacks: tuple[str, ...],
) -> Optional[int]:
    """Return the first valid integer from the provided environment variables.

    :param primary_var: Preferred environment variable name.
    :type primary_var: str
    :param fallbacks: Additional variable names checked in order.
    :type fallbacks: tuple[str, ...]
    :returns: Parsed integer port, or ``None`` when no valid value is found.
    :rtype: int | None
    """
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
    """Patch TRL's ``VLLMClient`` so its NCCL ``group_port`` honors env overrides.

    :param env_var: Primary environment variable checked for the port value.
    :type env_var: str
    :param fallback_vars: Additional variables queried when ``env_var`` is unset.
    :type fallback_vars: tuple[str, ...]
    :returns: ``None``. The function mutates TRL's ``VLLMClient`` constructor.
    :rtype: None
    """
    if _PATCH_STATE["vllm_group_port"]:
        return

    trl_module = sys.modules.get("trl")
    if trl_module is None:
        LOG.debug("TRL is unavailable; skipping VLLMClient group-port patch.")
        return

    try:
        from trl.extras.vllm_client import VLLMClient
    except ImportError:
        LOG.debug("TRL is unavailable; skipping VLLMClient group-port patch.")
        return

    original_init: Callable[..., None] = VLLMClient.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        """Inject group-port overrides before delegating to TRL's init."""
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

    VLLMClient.__init__ = patched_init
    _PATCH_STATE["vllm_group_port"] = True
    LOG.debug(
        "Patched TRL VLLMClient to accept group_port overrides via %s (fallbacks: %s).",
        env_var,
        ", ".join(fallback_vars),
    )


__all__ = ["ensure_vllm_group_port"]
