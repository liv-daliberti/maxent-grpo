#!/usr/bin/env python3
"""Retry PointMaze v2r3 from a bytecode-clean snapshot namespace."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

import launch_point_maze_algorithm_repair_pair_v2_r3 as r3


ROOT = Path(__file__).resolve().parents[2]
stage = r3.stage
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "point_maze_algorithm_repair_pair_v2_r3_snapshot_retry_20260730.md"
)


def snapshot_execution():
    inputs = (
        stage.TRAINER,
        r3.STAGE_TRAINER,
        stage.BASE_TRAINER,
        stage.AUDITOR,
        r3.r2.r1.BASE_AUDITOR,
        stage.TRAIN_BATCH,
        stage.AUDIT_BATCH,
        stage.PROTOCOL,
        r3.AMENDMENT,
        AMENDMENT,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".point-algorithm-repair-v2r3-retry.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = stage.tree_hash(temporary)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_algorithm_repair_v2r3_retry_ops_{digest}"
    )
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if stage.tree_hash(target) != digest:
        raise RuntimeError("PointMaze repair-v2r3 retry snapshot mismatch")
    return target, digest


_atomic = r3.atomic


def atomic(path: Path, payload):
    if path == stage.IDENTITY:
        payload.update(
            snapshot_retry_amendment_sha256=stage.sha(AMENDMENT),
            dry_import_bytecode_disabled=True,
            pre_sbatch_snapshot_hash_stop=True,
        )
    _atomic(path, payload)


stage.snapshot_execution = snapshot_execution
stage.atomic = atomic


if __name__ == "__main__":
    stage.main()
