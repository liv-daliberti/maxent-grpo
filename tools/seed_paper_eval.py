"""Thin wrapper for the repo-local SEED paper eval module."""

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
    from maxent_grpo.seed_paper_eval import main as module_main

    return module_main()


if __name__ == "__main__":
    raise SystemExit(main())
