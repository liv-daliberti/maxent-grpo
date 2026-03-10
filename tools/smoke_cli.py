"""
Minimal CLI smoke check used by CI.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))


def main() -> int:
    _ensure_src_path()
    from maxent_grpo.cli import hydra_cli

    required = ("hydra_entry", "baseline_entry")
    missing = [
        name for name in required if not callable(getattr(hydra_cli, name, None))
    ]
    if missing:
        print(f"Missing CLI callables: {', '.join(missing)}")
        return 1
    print("CLI smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
