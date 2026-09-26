#!/usr/bin/env python3
"""Configure or launch the frozen Ant v15 slate under stable controller v18."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "ops/exp_scaling", ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import launch_ant_maze_v14_hard_admission as base  # noqa: E402
from oat_drgrpo.ant_maze_worker_v18 import (  # noqa: E402
    _executor_identity,
    controller_identity,
)


LAUNCHER = Path(__file__).resolve()
base.PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "ant_maze_v15_controller_v18_admission_20260730.md"
)
base.MAKER = ROOT / "ops/make_ant_maze_mode_data_v15_controller_v18.py"
base.AUDITOR = ROOT / "ops/audit_ant_maze_mode_data_v15_controller_v18.py"
base.BATCH = (
    ROOT / "ops/slurm/admit_ant_maze_modebench_v15_controller_v18.slurm"
)
base.DATA = ROOT / "var/data/ant_maze_modebench_v15_controller_v18"
base.AUDIT = (
    ROOT
    / "var/artifacts/"
    "ant_maze_modebench_v15_controller_v18_admission_audit.json"
)
base.IDENTITY = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v18_admission_identity.json"
)
base.SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v18_admission_submission.json"
)

_snapshot_tree = base.snapshot_tree
_snapshot_execution = base.snapshot_execution
_atomic = base.atomic
_validate = base.validate


def snapshot_tree(source: Path, _prefix: str):
    return _snapshot_tree(source, "ant_v15_controller_v18_source")


def snapshot_execution():
    path, digest = _snapshot_execution()
    target = path.with_name(
        path.name.replace("ant_v14_hard_ops_", "ant_v15_v18_ops_")
    )
    if path != target and not target.exists():
        path.rename(target)
        path = target
    elif target.exists():
        path = target
    return path, digest


def validate() -> None:
    _validate()
    environment = dict(base.os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    base.run(
        [
            str(base.PYTHON),
            "-m",
            "py_compile",
            str(LAUNCHER),
            str(ROOT / "src/oat_drgrpo/ant_maze_worker_v18.py"),
            str(ROOT / "src/oat_drgrpo/maze_modebench_worker_v18.py"),
            str(ROOT / "src/oat_drgrpo/maze_modebench_process_v18.py"),
        ],
        env=environment,
    )


def atomic(path: Path, payload: Any) -> None:
    if path == base.IDENTITY:
        controller_identity()
        executor = _executor_identity()
        payload = {
            **payload,
            "schema_version": (
                "ant-maze-v18-stable-route-generation-identity-v1"
            ),
            "launcher_sha256": base.sha(LAUNCHER),
            **executor,
            "map_size": 13,
            "central_obstacle_shape": [3, 3],
            "route_decisions": 8,
            "max_actions": 20,
            "action_repeat": 400,
            "v15_route_outcome_loaded": True,
            "v17_controller_outcome_loaded": True,
            "v18_controller_outcome_loaded": True,
            "v18_route_sampled_before_freeze": False,
            "language_model_sampled": False,
            "post_outcome_map_or_route_substitution": False,
        }
    elif path == base.SUBMISSION:
        payload = {
            **payload,
            "schema": "ant-maze-v15-controller-v18-submission-v1",
        }
    _atomic(path, payload)


base.snapshot_tree = snapshot_tree
base.snapshot_execution = snapshot_execution
base.validate = validate
base.atomic = atomic


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        controller_identity()
    base.main()
