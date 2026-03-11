"""Compatibility wrapper for the shared log validator script."""

from __future__ import annotations

import runpy
from pathlib import Path

_TOOL_PATH = Path(__file__).resolve().parents[3] / "tools" / "validate_logs.py"
globals().update(runpy.run_path(str(_TOOL_PATH)))
