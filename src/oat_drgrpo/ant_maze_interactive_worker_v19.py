"""Persistent trusted AntMaze worker for the v19 controller binding."""

from __future__ import annotations

# Importing the v19 binding first installs its immutable paths and validators
# into the shared v18 execution kernel. The interactive state machine is then
# reused without changing action, timing, or stable-arrival semantics.
from . import ant_maze_worker_v19 as v19  # noqa: F401
from . import ant_maze_interactive_worker_v18 as base


if __name__ == "__main__":
    base.main()
