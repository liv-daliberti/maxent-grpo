#!/usr/bin/env python3
"""Relaunch PointMaze repair-v2 with the complete trainer snapshot."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import launch_point_maze_algorithm_repair_pair_v2_r1 as r1


ROOT = Path(__file__).resolve().parents[2]
stage = r1.stage
V1_TRAINER = ROOT / "ops/train_point_maze_algorithm_repair_v1.py"
AMENDMENT = (
    ROOT
    / "paper/preregistration/point_maze_algorithm_repair_pair_v2_r2_20260730.md"
)
stage.TRAIN_BATCH = (
    ROOT / "ops/slurm/train_point_maze_algorithm_repair_v2_r2.slurm"
)
stage.AUDIT_BATCH = (
    ROOT / "ops/slurm/audit_point_maze_algorithm_repair_pair_v2_r2.slurm"
)
stage.IDENTITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2r2_identity.json"
)
stage.MANIFEST = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2r2_jobs.tsv"
)
stage.SUBMISSION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2r2_submission.json"
)
stage.AUDIT_OUTPUT = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2r2_audit.json"
)
stage.AUDIT_RUNNER = (
    ROOT
    / "var/artifacts/point_maze_algorithm_repair_pair_v2r2_audit_runner_identity.json"
)
stage.ARTIFACT_STEM = "point_maze_algorithm_repair_v2r2"
_atomic = r1._atomic


def snapshot_execution():
    inputs = (
        stage.TRAINER,
        V1_TRAINER,
        stage.BASE_TRAINER,
        stage.AUDITOR,
        r1.BASE_AUDITOR,
        stage.TRAIN_BATCH,
        stage.AUDIT_BATCH,
        stage.PROTOCOL,
        AMENDMENT,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".point-algorithm-repair-v2r2.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = stage.tree_hash(temporary)
    target = (
        ROOT
        / f"var/artifacts/source_snapshots/point_algorithm_repair_v2r2_ops_{digest}"
    )
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if stage.tree_hash(target) != digest:
        raise RuntimeError("PointMaze repair-v2r2 snapshot mismatch")
    return target, digest


def atomic(path: Path, payload):
    if path == stage.IDENTITY:
        payload.update(
            repair_schema="point-maze-algorithm-repair-pair-identity-v1",
            repair_v2_schema="point-maze-algorithm-repair-pair-identity-v2",
            repair_r2_schema="point-maze-algorithm-repair-pair-identity-v2r2",
            repair_launcher_sha256=hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            execution_amendment_sha256=stage.sha(AMENDMENT),
            failed_import_jobs=[30204890, 30204891],
            final_seed_cohort=False,
            development_only=True,
            evaluation_common_random_numbers=True,
            evaluation_split="development",
            secondary_post_outcome_repair=True,
            k16_aligned_algorithmic_gate=True,
            viability_receipt_sha256=stage.sha(r1.VIABILITY),
            admission_audit_sha256=stage.sha(r1.ADMISSION),
            qualification_identity_sha256=stage.sha(
                r1.QUALIFICATION_IDENTITY
            ),
        )
    elif path == stage.SUBMISSION:
        payload.update(
            repair_schema="point-maze-algorithm-repair-pair-submission-v1",
            repair_v2_schema=(
                "point-maze-algorithm-repair-pair-submission-v2"
            ),
            repair_r2_schema=(
                "point-maze-algorithm-repair-pair-submission-v2r2"
            ),
        )
    _atomic(path, payload)


stage.snapshot_execution = snapshot_execution
stage.atomic = atomic


if __name__ == "__main__":
    stage.main()
