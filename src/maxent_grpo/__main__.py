"""
Module entrypoint for ``python -m maxent_grpo``.

Dispatches to the MaxEnt training entrypoint in :mod:`maxent_grpo.maxent_grpo`.
"""

from __future__ import annotations

from maxent_grpo import main as _main


def main() -> None:
    """Invoke the MaxEnt training CLI entrypoint."""

    _main()


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
