from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def reload_constructive_code_v2_bases() -> None:
    """Restore the shared constructive-code v2 modules to their own values.

    ``replay_constructive_code_v4`` and ``_v5`` deliberately rebind attributes
    on ``replay_constructive_code_v2`` and ``audit_constructive_code_v2`` at
    import time so the later slates can reuse the v2 machinery. That mutation
    is process-wide and permanent, so any test asserting the *v2* contract has
    to re-execute those module bodies first or it silently depends on whether
    a v4/v5 test imported earlier in the session.
    """

    import importlib

    for name in (
        "replay_constructive_code_v2",
        "audit_constructive_code_v2",
    ):
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)
