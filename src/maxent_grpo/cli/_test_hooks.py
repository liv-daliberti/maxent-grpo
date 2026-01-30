"""
Internal helpers that make the CLI smoke tests more robust.

The pytest suites spin up subprocesses that rely on ``usercustomize.py`` to
inject lightweight stubs (e.g., replacing heavy training loops with simple
markers).  Python usually imports ``usercustomize`` automatically during
startup, but certain environments disable that behavior (for example via
``PYTHONNOUSERSITE`` or ``-S``).  When that happens the subprocess ends up
running the real training code and the smoke tests fail.

To avoid depending on the interpreter's startup sequence, CLI entrypoints call
``ensure_usercustomize_loaded`` before importing optional heavy components.
The helper attempts to import ``usercustomize`` once per process and quietly
ignores failures when the module is absent (the common case outside tests).
"""

from __future__ import annotations

import importlib
import os
from typing import Final

_ENV_SKIP_FLAG: Final[str] = "MAXENT_SKIP_USERCUSTOMIZE"
_USERCUSTOMIZE_STATE = {"attempted": False}


def ensure_usercustomize_loaded() -> None:
    """Best-effort import of ``usercustomize`` for CLI smoke tests.

    The import is attempted at most once per process and is skipped entirely
    when ``MAXENT_SKIP_USERCUSTOMIZE=1`` is set.  Import errors and common
    runtime exceptions are swallowed so real CLI executions remain unaffected.
    """

    if _USERCUSTOMIZE_STATE["attempted"]:
        return
    _USERCUSTOMIZE_STATE["attempted"] = True
    if os.environ.get(_ENV_SKIP_FLAG) == "1":
        return
    try:
        importlib.import_module("usercustomize")
    except ImportError:
        # Most environments will not provide a usercustomize module; silently
        # continue so CLI entrypoints behave normally.
        return
    except (AttributeError, OSError, RuntimeError, SyntaxError, TypeError, ValueError):
        # Defensive: custom user hooks should never take down the CLI.
        return
