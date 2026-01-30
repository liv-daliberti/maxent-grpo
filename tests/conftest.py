"""Shared pytest configuration (lightweight)."""

from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    # Make the src-layout packages importable without installing the project.
    sys.path.insert(0, str(_SRC_ROOT))
