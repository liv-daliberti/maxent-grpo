#!/usr/bin/env python3
"""Launch PointMaze repair-v2 with its base-auditor snapshot dependency."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import launch_point_maze_algorithm_repair_pair_v1 as v1


ROOT = Path(__file__).resolve().parents[2]
stage = v1.base
PROTOCOL = (
    ROOT
    / "paper/preregistration/point_maze_algorithm_repair_pair_v2_20260730.md"
)
QUALIFICATION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v2_qualification.json"
)
QUALIFICATION_IDENTITY = (
    ROOT
    / "var/artifacts/point_maze_algorithm_repair_v2_qualification_identity.json"
)
VIABILITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v2_viability.json"
)
ADMISSION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v2_admission_audit.json"
)
BASE_AUDITOR = ROOT / "ops/audit_point_maze_algorithm_repair_pair_v1.py"

stage.PROTOCOL = PROTOCOL
stage.TRAINER = ROOT / "ops/train_point_maze_algorithm_repair_v2.py"
stage.AUDITOR = ROOT / "ops/audit_point_maze_algorithm_repair_pair_v2.py"
stage.TRAIN_BATCH = (
    ROOT / "ops/slurm/train_point_maze_algorithm_repair_v2.slurm"
)
stage.AUDIT_BATCH = (
    ROOT / "ops/slurm/audit_point_maze_algorithm_repair_pair_v2.slurm"
)
stage.QUALIFICATION = QUALIFICATION
stage.SMOKE_IDENTITY = QUALIFICATION_IDENTITY
stage.DATA = ROOT / "var/data/point_maze_algorithm_repair_v2"
stage.IDENTITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2_identity.json"
)
stage.MANIFEST = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2_jobs.tsv"
)
stage.SUBMISSION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2_submission.json"
)
stage.AUDIT_OUTPUT = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v2_audit.json"
)
stage.AUDIT_RUNNER = (
    ROOT
    / "var/artifacts/point_maze_algorithm_repair_pair_v2_audit_runner_identity.json"
)
stage.SEEDS = (76521,)
stage.VARIANT = "algorithm_repair_v2"
stage.ARTIFACT_STEM = "point_maze_algorithm_repair_v2"


def configure_variant(_variant: str) -> None:
    return None


def base_prerequisites() -> None:
    for path in (
        stage.PYTHON,
        stage.PROTOCOL,
        stage.TRAINER,
        stage.BASE_TRAINER,
        stage.AUDITOR,
        BASE_AUDITOR,
        stage.TRAIN_BATCH,
        stage.AUDIT_BATCH,
        stage.QUALIFICATION,
        stage.SMOKE_IDENTITY,
        stage.MODEL / "config.json",
        stage.DATA / "identity.json",
        stage.DATA / "train/dataset_dict.json",
        stage.DATA / "dev/dataset_dict.json",
        stage.WORKER,
        VIABILITY,
        ADMISSION,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    identity = json.loads(stage.SMOKE_IDENTITY.read_text())
    if (
        identity.get("schema")
        != "point-maze-algorithm-repair-v2-qualification-identity"
        or identity.get("development_only") is not True
        or identity.get("final_seed") is not False
        or identity.get("viability_job_id") != 30204702
        or identity.get("prefix_size_aligned_to_online_rollouts") != 16
        or identity.get("prefix16_success_prompts") < 3
    ):
        raise RuntimeError("PointMaze v2 qualification identity drift")


def qualification_passes() -> None:
    payload = json.loads(stage.QUALIFICATION.read_text())
    if (
        payload.get("schema")
        != "point-maze-algorithm-repair-v2-qualification"
        or payload.get("status") != "pass"
        or payload.get("decision")
        != "eligible_for_point_maze_algorithm_repair_v2_pair"
        or payload.get("errors") not in ([], None)
    ):
        raise RuntimeError("PointMaze K=16 qualification did not pass")


def validate() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    environment["LD_LIBRARY_PATH"] = library + (
        ":" + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    stage.run(
        [
            str(stage.PYTHON),
            "-m",
            "py_compile",
            str(stage.TRAINER),
            str(stage.AUDITOR),
            str(BASE_AUDITOR),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    stage.run(["bash", "-n", str(stage.TRAIN_BATCH)])
    stage.run(["bash", "-n", str(stage.AUDIT_BATCH)])
    stage.run(
        [
            str(stage.PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_point_maze_stage_b_05b_12pass.py"),
            str(ROOT / "tests/test_point_maze_paired_smoke.py"),
            str(ROOT / "tests/test_interactive_episode_objective.py"),
            str(ROOT / "tests/test_interactive_episode_replay.py"),
            str(ROOT / "tests/test_point_maze_interactive_policy.py"),
            str(ROOT / "tests/test_point_maze_interactive_worker.py"),
        ],
        env=environment,
    )


_snapshot_tree = v1._snapshot_tree
_atomic = v1._atomic


def snapshot_tree(source: Path, _prefix: str):
    return _snapshot_tree(source, "point_algorithm_repair_pair_v2_source")


def snapshot_execution():
    inputs = (
        stage.TRAINER,
        stage.BASE_TRAINER,
        stage.AUDITOR,
        BASE_AUDITOR,
        stage.TRAIN_BATCH,
        stage.AUDIT_BATCH,
        stage.PROTOCOL,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".point-algorithm-repair-v2.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = stage.tree_hash(temporary)
    target = (
        ROOT
        / f"var/artifacts/source_snapshots/point_algorithm_repair_v2_ops_{digest}"
    )
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if stage.tree_hash(target) != digest:
        raise RuntimeError("PointMaze repair-v2 execution snapshot mismatch")
    return target, digest


def atomic(path: Path, payload):
    if path == stage.IDENTITY:
        payload.update(
            repair_schema="point-maze-algorithm-repair-pair-identity-v1",
            repair_v2_schema="point-maze-algorithm-repair-pair-identity-v2",
            repair_launcher_sha256=hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            final_seed_cohort=False,
            development_only=True,
            evaluation_common_random_numbers=True,
            evaluation_split="development",
            secondary_post_outcome_repair=True,
            k16_aligned_algorithmic_gate=True,
            viability_receipt_sha256=stage.sha(VIABILITY),
            admission_audit_sha256=stage.sha(ADMISSION),
            qualification_identity_sha256=stage.sha(QUALIFICATION_IDENTITY),
        )
    elif path == stage.SUBMISSION:
        payload.update(
            repair_schema="point-maze-algorithm-repair-pair-submission-v1",
            repair_v2_schema=(
                "point-maze-algorithm-repair-pair-submission-v2"
            ),
        )
    _atomic(path, payload)


stage.configure_variant = configure_variant
stage.base_prerequisites = base_prerequisites
stage.qualification_passes = qualification_passes
stage.validate = validate
stage.snapshot_tree = snapshot_tree
stage.snapshot_execution = snapshot_execution
stage.atomic = atomic


if __name__ == "__main__":
    stage.main()
