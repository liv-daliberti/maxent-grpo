#!/usr/bin/env python3
"""Launch the two-cell PointMaze algorithm-repair qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import launch_point_maze_stage_b_05b_12pass as base


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "paper/preregistration/point_maze_algorithm_repair_pair_v1_20260730.md"
QUALIFICATION = ROOT / "var/artifacts/point_maze_algorithm_repair_v1_qualification.json"
QUALIFICATION_IDENTITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v1_qualification_identity.json"
)
VIABILITY = ROOT / "var/artifacts/point_maze_algorithm_repair_v1_viability.json"
ADMISSION = ROOT / "var/artifacts/point_maze_algorithm_repair_v1_admission_audit.json"
CLARIFICATION = ROOT / (
    "paper/preregistration/"
    "point_maze_algorithm_repair_v1_cardinality_clarification_20260730.md"
)

base.PROTOCOL = PROTOCOL
base.TRAINER = ROOT / "ops/train_point_maze_algorithm_repair_v1.py"
base.AUDITOR = ROOT / "ops/audit_point_maze_algorithm_repair_pair_v1.py"
base.TRAIN_BATCH = ROOT / "ops/slurm/train_point_maze_algorithm_repair_v1.slurm"
base.AUDIT_BATCH = (
    ROOT / "ops/slurm/audit_point_maze_algorithm_repair_pair_v1.slurm"
)
base.QUALIFICATION = QUALIFICATION
base.SMOKE_IDENTITY = QUALIFICATION_IDENTITY
base.DATA = ROOT / "var/data/point_maze_algorithm_repair_v1"
base.IDENTITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v1_identity.json"
)
base.MANIFEST = ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v1_jobs.tsv"
base.SUBMISSION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v1_submission.json"
)
base.AUDIT_OUTPUT = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v1_audit.json"
)
base.AUDIT_RUNNER = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v1_audit_runner_identity.json"
)
base.SEEDS = (76501,)
base.VARIANT = "algorithm_repair_v1"
base.ARTIFACT_STEM = "point_maze_algorithm_repair_v1"


def configure_variant(_variant: str) -> None:
    return None


def base_prerequisites() -> None:
    for path in (
        base.PYTHON,
        base.PROTOCOL,
        base.TRAINER,
        base.BASE_TRAINER,
        base.AUDITOR,
        base.TRAIN_BATCH,
        base.AUDIT_BATCH,
        base.QUALIFICATION,
        base.SMOKE_IDENTITY,
        base.MODEL / "config.json",
        base.DATA / "identity.json",
        base.DATA / "train/dataset_dict.json",
        base.DATA / "dev/dataset_dict.json",
        base.WORKER,
        VIABILITY,
        ADMISSION,
        CLARIFICATION,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    identity = json.loads(base.SMOKE_IDENTITY.read_text())
    if (
        identity.get("schema")
        != "point-maze-algorithm-repair-v1-qualification-identity"
        or identity.get("development_only") is not True
        or identity.get("final_seed") is not False
        or identity.get("viability_job_id") != 30204570
    ):
        raise RuntimeError("PointMaze repair qualification identity drift")


def qualification_passes() -> None:
    payload = json.loads(base.QUALIFICATION.read_text())
    if (
        payload.get("schema") != "point-maze-algorithm-repair-v1-qualification"
        or payload.get("status") != "pass"
        or payload.get("decision") != "eligible_for_ten_point_maze_stage_b_jobs"
        or payload.get("errors") not in ([], None)
    ):
        raise RuntimeError("PointMaze repair viability did not authorize the pair")


def validate() -> None:
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
            str(base.TRAINER),
            str(base.AUDITOR),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    base.run(["bash", "-n", str(base.TRAIN_BATCH)])
    base.run(["bash", "-n", str(base.AUDIT_BATCH)])
    base.run(
        [
            str(base.PYTHON),
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


_snapshot_tree = base.snapshot_tree


def snapshot_tree(source: Path, _prefix: str):
    return _snapshot_tree(source, "point_algorithm_repair_pair_v1_source")


_atomic = base.atomic


def atomic(path: Path, payload):
    if path == base.IDENTITY:
        payload.update(
            repair_schema="point-maze-algorithm-repair-pair-identity-v1",
            repair_launcher_sha256=hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            final_seed_cohort=False,
            development_only=True,
            evaluation_common_random_numbers=True,
            evaluation_split="development",
            secondary_post_outcome_repair=True,
            viability_receipt_sha256=base.sha(VIABILITY),
            admission_audit_sha256=base.sha(ADMISSION),
            cardinality_clarification_sha256=base.sha(CLARIFICATION),
        )
    elif path == base.SUBMISSION:
        payload["repair_schema"] = (
            "point-maze-algorithm-repair-pair-submission-v1"
        )
    _atomic(path, payload)


base.configure_variant = configure_variant
base.base_prerequisites = base_prerequisites
base.qualification_passes = qualification_passes
base.validate = validate
base.snapshot_tree = snapshot_tree
base.atomic = atomic


if __name__ == "__main__":
    base.main()

