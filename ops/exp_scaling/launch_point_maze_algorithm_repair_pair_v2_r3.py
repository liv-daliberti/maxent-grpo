#!/usr/bin/env python3
"""Relaunch PointMaze repair-v2 with a flattened trainer import chain."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import launch_point_maze_algorithm_repair_pair_v2_r2 as r2


ROOT = Path(__file__).resolve().parents[2]
stage = r2.stage
STAGE_TRAINER = ROOT / "ops/train_point_maze_stage_b_05b_12pass.py"
AMENDMENT = (
    ROOT
    / "paper/preregistration/point_maze_algorithm_repair_pair_v2_r3_20260730.md"
)
stage.TRAINER = ROOT / "ops/train_point_maze_algorithm_repair_v2_direct.py"
stage.TRAIN_BATCH = (
    ROOT / "ops/slurm/train_point_maze_algorithm_repair_v2_r3.slurm"
)
stage.AUDIT_BATCH = (
    ROOT / "ops/slurm/audit_point_maze_algorithm_repair_pair_v2_r3.slurm"
)
stage.IDENTITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2r3_identity.json"
)
stage.MANIFEST = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2r3_jobs.tsv"
)
stage.SUBMISSION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2r3_submission.json"
)
stage.AUDIT_OUTPUT = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2r3_audit.json"
)
stage.AUDIT_RUNNER = (
    ROOT
    / "var/artifacts/point_maze_algorithm_repair_pair_v2r3_audit_runner_identity.json"
)
stage.ARTIFACT_STEM = "point_maze_algorithm_repair_v2r3"
_atomic = r2._atomic


def snapshot_execution():
    inputs = (
        stage.TRAINER,
        STAGE_TRAINER,
        stage.BASE_TRAINER,
        stage.AUDITOR,
        r2.r1.BASE_AUDITOR,
        stage.TRAIN_BATCH,
        stage.AUDIT_BATCH,
        stage.PROTOCOL,
        AMENDMENT,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".point-algorithm-repair-v2r3.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = stage.tree_hash(temporary)
    target = (
        ROOT
        / f"var/artifacts/source_snapshots/point_algorithm_repair_v2r3_ops_{digest}"
    )
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if stage.tree_hash(target) != digest:
        raise RuntimeError("PointMaze repair-v2r3 snapshot mismatch")
    return target, digest


def atomic(path: Path, payload):
    if path == stage.IDENTITY:
        payload.update(
            repair_schema="point-maze-algorithm-repair-pair-identity-v1",
            repair_v2_schema="point-maze-algorithm-repair-pair-identity-v2",
            repair_r3_schema="point-maze-algorithm-repair-pair-identity-v2r3",
            repair_launcher_sha256=hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            execution_amendment_sha256=stage.sha(AMENDMENT),
            failed_import_jobs=[30204890, 30204891, 30204893, 30204894],
            flattened_trainer_wrapper=True,
            final_seed_cohort=False,
            development_only=True,
            evaluation_common_random_numbers=True,
            evaluation_split="development",
            secondary_post_outcome_repair=True,
            k16_aligned_algorithmic_gate=True,
            viability_receipt_sha256=stage.sha(r2.r1.VIABILITY),
            admission_audit_sha256=stage.sha(r2.r1.ADMISSION),
            qualification_identity_sha256=stage.sha(
                r2.r1.QUALIFICATION_IDENTITY
            ),
        )
    elif path == stage.SUBMISSION:
        payload.update(
            repair_schema="point-maze-algorithm-repair-pair-submission-v1",
            repair_v2_schema=(
                "point-maze-algorithm-repair-pair-submission-v2"
            ),
            repair_r3_schema=(
                "point-maze-algorithm-repair-pair-submission-v2r3"
            ),
        )
    _atomic(path, payload)


stage.snapshot_execution = snapshot_execution
stage.atomic = atomic


if __name__ == "__main__":
    stage.main()
