#!/usr/bin/env python3
"""Launch frozen PointMaze v4 executable admission."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import launch_point_maze_algorithm_repair_v3_admission as base


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
base.MAKER = ROOT / "ops/make_point_maze_algorithm_repair_data_v4.py"
base.AUDITOR = ROOT / "ops/audit_point_maze_algorithm_repair_data_v4.py"
base.BATCH = (
    ROOT / "ops/slurm/admit_point_maze_algorithm_repair_v4.slurm"
)
base.DATA = ROOT / "var/data/point_maze_algorithm_repair_v4"
base.AUDIT = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v4_admission_audit.json"
)
base.IDENTITY = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v4_admission_identity.json"
)
base.SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v4_admission_submission.json"
)
_atomic = base.atomic


def snapshot_execution():
    inputs = (
        base.PROTOCOL,
        AMENDMENT,
        base.MAKER,
        base.MAKER_BASE,
        base.MEDIUM_FAMILIES,
        base.HARD_FAMILIES,
        ROOT / "ops/make_point_maze_algorithm_repair_data_v3.py",
        base.AUDITOR,
        base.AUDITOR_BASE,
        base.BATCH,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".point-repair-v4-admission.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in inputs:
        shutil.copy2(source, staging / source.name)
    digest = base.tree_hash(staging)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_repair_v4_admission_ops_{digest}"
    )
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if base.tree_hash(target) != digest:
        raise RuntimeError("PointMaze v4 execution snapshot mismatch")
    return target, digest


def validate() -> None:
    for path in (
        base.PYTHON,
        base.PROTOCOL,
        AMENDMENT,
        base.MAKER,
        base.MAKER_BASE,
        base.MEDIUM_FAMILIES,
        base.HARD_FAMILIES,
        ROOT / "ops/make_point_maze_algorithm_repair_data_v3.py",
        base.AUDITOR,
        base.AUDITOR_BASE,
        base.BATCH,
        V3_VIABILITY,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if base.sha(V3_VIABILITY) != (
        "6417bd27d55251d287f804035c4d84c8f683d6117e06ec54de98755169ab3da5"
    ):
        raise RuntimeError("PointMaze v4 sealed v3 antecedent drift")
    v3 = json.loads(V3_VIABILITY.read_text(encoding="utf-8"))
    if (
        v3.get("summary", {}).get("verified_completions") != 139
        or v3.get("summary", {}).get("prompt_count") != 4
    ):
        raise RuntimeError("PointMaze v4 requires the exact v3 stop")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    base.run(
        [
            str(base.PYTHON),
            "-m",
            "py_compile",
            str(base.MAKER),
            str(base.AUDITOR),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    base.run(["bash", "-n", str(base.BATCH)])
    base.run(
        [
            str(base.PYTHON),
            "-c",
            (
                "import make_point_maze_algorithm_repair_data_v4 as v;"
                "c={s:{r:sum(x[2]==r for x in a) for r in range(4)} "
                "for s,a in v.SPLIT_ASSIGNMENTS.items()};"
                "assert c=={'train':{0:2,1:2,2:2,3:2},"
                "'dev':{0:1,1:1,2:1,3:1},"
                "'eval':{0:1,1:1,2:1,3:1}};"
                "assert all(f['counts']==(13,32,16) "
                "for f in v.large_families())"
            ),
        ],
        env=environment,
    )


def atomic(path: Path, payload: Any) -> None:
    if path == base.IDENTITY:
        payload.update(
            schema_version=(
                "point-maze-algorithm-repair-v4-admission-identity-v1"
            ),
            amendment_sha256=base.sha(AMENDMENT),
            v3_viability_sha256=base.sha(V3_VIABILITY),
            v3_verified_rate=0.54296875,
            horizontal_geometry_repair=True,
            horizontal_family_size=13,
            horizontal_route_counts=[13, 32, 16],
            failed_v4_source_gate_loaded=True,
            checker_only_route_calibration_loaded=True,
            v4_route_outcome_loaded=False,
            v4_model_sampled=False,
            threshold_changed=False,
            coefficient_changed=False,
        )
    elif path == base.SUBMISSION:
        payload["schema_version"] = (
            "point-maze-algorithm-repair-v4-admission-submission-v1"
        )
    _atomic(path, payload)


base.snapshot_execution = snapshot_execution
base.validate = validate
base.atomic = atomic


if __name__ == "__main__":
    base.main()
