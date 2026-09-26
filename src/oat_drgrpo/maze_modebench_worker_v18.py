"""Networkless maze worker for the stable-handoff Ant v18 binding."""

from __future__ import annotations

from . import maze_modebench_worker as base
from .ant_maze_worker_v18 import controller_receipt_sha256
from .ant_maze_worker_v18 import execute_ant_v18
from .maze_modebench import ANT_MAZE_VERIFIER, parse_maze_action_spec


_base_execute = base.execute


def execute(candidate, raw_spec):
    spec = parse_maze_action_spec(raw_spec)
    if (
        spec.verifier == ANT_MAZE_VERIFIER
        and spec.controller_sha256 == controller_receipt_sha256()
    ):
        return execute_ant_v18(candidate, raw_spec)
    return _base_execute(candidate, raw_spec)


base.execute = execute


if __name__ == "__main__":
    base.main()
