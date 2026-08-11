#!/usr/bin/env python3
"""Configure or launch the frozen Ant v15/controller-v19 model gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "ops/exp_scaling", ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import launch_ant_maze_v15_controller_v18_viability as base  # noqa: E402


LAUNCHER = Path(__file__).resolve()
base.DATA = ROOT / "var/data/ant_maze_modebench_v15_controller_v19"
base.ADMISSION = (
    ROOT
    / "var/artifacts/"
    "ant_maze_modebench_v15_controller_v19_admission_audit.json"
)
base.ADMISSION_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v19_admission_identity.json"
)
base.PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "ant_maze_v15_controller_v19_05b_viability_20260804.md"
)
base.EVALUATOR = ROOT / "ops/evaluate_ant_maze_interactive_viability_v19.py"
base.QUALIFIER = (
    ROOT / "ops/qualify_ant_maze_v15_controller_v19_viability.py"
)
base.BATCH = (
    ROOT / "ops/slurm/evaluate_ant_maze_v15_controller_v19_viability.slurm"
)
base.OUTPUT = (
    ROOT / "var/artifacts/ant_maze_v15_controller_v19_05b_viability.json"
)
base.QUALIFICATION = (
    ROOT / "var/artifacts/ant_maze_v15_controller_v19_qualification.json"
)
base.IDENTITY = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v19_viability_identity.json"
)
base.SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v19_viability_submission.json"
)

_validate_static = base.validate_static
_atomic = base.atomic


def validate_static() -> None:
    _validate_static()
    env = base.environment()
    base.run(
        [
            str(base.PYTHON),
            "-m",
            "py_compile",
            str(LAUNCHER),
            str(ROOT / "src/oat_drgrpo/ant_maze_worker_v19.py"),
            str(ROOT / "src/oat_drgrpo/ant_maze_interactive_worker_v19.py"),
            str(ROOT / "src/oat_drgrpo/ant_maze_interactive_process_v19.py"),
        ],
        env=env,
    )
    base.run(
        [
            str(base.PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_ant_maze_worker_v19.py"),
            str(ROOT / "tests/test_ant_maze_v15_v19_viability.py"),
        ],
        env=env,
    )


def validate_runtime() -> None:
    for path in (
        base.DATA / "identity.json",
        base.DATA / "dev/dataset_dict.json",
        base.ADMISSION,
        base.ADMISSION_IDENTITY,
        ROOT
        / "var/maze_runtime/controllers/ant_continuing_waypoint_v19.zip",
        ROOT
        / "var/maze_runtime/controllers/"
        "ant_continuing_waypoint_v19.evaluation.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    admission = json.loads(base.ADMISSION.read_text(encoding="utf-8"))
    if (
        admission.get("status") != "pass"
        or admission.get("decision")
        != "admitted_to_ant_v15_v19_frozen_model_viability_gate"
    ):
        raise RuntimeError("Ant v15/v19 admission did not authorize viability")


def atomic(path: Path, payload: Any) -> None:
    if path == base.IDENTITY:
        payload = {
            **payload,
            "schema_version": (
                "ant-maze-v15-controller-v19-viability-identity-v1"
            ),
            "launcher_sha256": base.sha(LAUNCHER),
            "seed": 76702,
            "controller_version": "v19",
            "continuing_task_controller_gate": True,
            "explicit_ant_health_controller_gate": True,
            "controller_outcome_loaded_before_freeze": False,
            "admission_outcome_loaded_before_freeze": False,
            "language_model_sampled_before_freeze": False,
        }
    elif path == base.SUBMISSION:
        payload = {
            **payload,
            "schema_version": (
                "ant-maze-v15-controller-v19-viability-submission-v1"
            ),
        }
    _atomic(path, payload)


base.validate_static = validate_static
base.validate_runtime = validate_runtime
base.atomic = atomic


if __name__ == "__main__":
    base.main()
