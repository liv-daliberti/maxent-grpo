#!/usr/bin/env python3
"""Launch the frozen PointMaze v4 K=16 viability gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import launch_point_maze_algorithm_repair_v3_viability as base


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "point_maze_algorithm_repair_v4_source_amendment_r1_20260730.md"
)
V3_VIABILITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v3_viability.json"
)
base.PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "point_maze_algorithm_repair_v4_horizontal_geometry_20260730.md"
)
base.QUALIFIER = ROOT / "ops/qualify_point_maze_algorithm_repair_v4.py"
base.BATCH = (
    ROOT / "ops/slurm/evaluate_point_maze_algorithm_repair_v4.slurm"
)
base.DATA = ROOT / "var/data/point_maze_algorithm_repair_v4"
base.ADMISSION = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v4_admission_audit.json"
)
base.OUTPUT = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v4_viability.json"
)
base.QUALIFICATION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v4_qualification.json"
)
base.IDENTITY = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v4_viability_identity.json"
)
base.SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v4_viability_submission.json"
)
_atomic = base.atomic


def snapshot_execution():
    staging = Path(
        tempfile.mkdtemp(
            prefix=".point-repair-v4-viability.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in (
        base.PROTOCOL,
        AMENDMENT,
        base.EVALUATOR,
        base.QUALIFIER,
        base.BATCH,
    ):
        shutil.copy2(source, staging / source.name)
    digest = base.tree_hash(staging)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_repair_v4_viability_ops_{digest}"
    )
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if base.tree_hash(target) != digest:
        raise RuntimeError("PointMaze v4 viability snapshot mismatch")
    return target, digest


def validate() -> None:
    for path in (
        base.PYTHON,
        base.PROTOCOL,
        AMENDMENT,
        base.EVALUATOR,
        base.QUALIFIER,
        base.BATCH,
        base.DATA / "identity.json",
        base.ADMISSION,
        base.MODEL / "config.json",
        ROOT / "var/maze_runtime/venv/bin/python",
        V3_VIABILITY,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    admission = json.loads(base.ADMISSION.read_text(encoding="utf-8"))
    if (
        admission.get("status") != "pass"
        or admission.get("decision")
        != "admitted_to_point_maze_v4_horizontal_viability_gate"
    ):
        raise RuntimeError("PointMaze v4 executable admission did not pass")
    data = json.loads(
        (base.DATA / "identity.json").read_text(encoding="utf-8")
    )
    if (
        data.get("schema_version")
        != "point-maze-algorithm-repair-data-v4"
        or data.get("orientation_balanced_within_each_split") is not True
        or data.get("executable_task_overlap_count") != 0
        or data.get("horizontal_geometry_repair") is not True
        or data.get("horizontal_family_size") != 13
    ):
        raise RuntimeError("PointMaze v4 data identity drift")
    if base.sha(V3_VIABILITY) != (
        "6417bd27d55251d287f804035c4d84c8f683d6117e06ec54de98755169ab3da5"
    ):
        raise RuntimeError("PointMaze v4 v3 antecedent drift")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    environment["LD_LIBRARY_PATH"] = library + (
        ":" + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    base.run(
        [
            str(base.PYTHON),
            "-m",
            "py_compile",
            str(base.EVALUATOR),
            str(base.QUALIFIER),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    base.run(["bash", "-n", str(base.BATCH)])
    base.run(
        [
            str(base.PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_point_maze_modebench.py"),
            str(ROOT / "tests/test_point_maze_interactive_policy.py"),
        ],
        env=environment,
    )


def atomic(path: Path, payload: Any) -> None:
    if path == base.IDENTITY:
        payload.update(
            schema_version=(
                "point-maze-algorithm-repair-v4-viability-identity-v1"
            ),
            protocol_sha256=base.sha(base.PROTOCOL),
            amendment_sha256=base.sha(AMENDMENT),
            evaluator_sha256=base.sha(base.EVALUATOR),
            qualifier_sha256=base.sha(base.QUALIFIER),
            batch_sha256=base.sha(base.BATCH),
            data_tree_sha256=base.tree_hash(base.DATA),
            data_identity_sha256=base.sha(base.DATA / "identity.json"),
            admission_audit_sha256=base.sha(base.ADMISSION),
            v3_viability_sha256=base.sha(V3_VIABILITY),
            v3_verified_rate=0.54296875,
            seed=76550,
            sample_count_per_prompt=64,
            prefix_count=16,
            split="development_only",
            orientation_counts={"0": 1, "1": 1, "2": 1, "3": 1},
            horizontal_geometry_repair=True,
            horizontal_family_size=13,
            horizontal_route_counts=[13, 32, 16],
            threshold_changed_from_v3=False,
            final_seed=False,
            evaluation_rows_loaded=False,
        )
    elif path == base.SUBMISSION:
        payload["schema_version"] = (
            "point-maze-algorithm-repair-v4-viability-submission-v1"
        )
    _atomic(path, payload)


base.snapshot_execution = snapshot_execution
base.validate = validate
base.atomic = atomic


if __name__ == "__main__":
    base.main()
