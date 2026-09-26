#!/usr/bin/env python3
"""Launch the frozen PointMaze v3 orientation-balanced calibration pair."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import launch_point_maze_stage_b_05b_12pass as stage


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "point_maze_algorithm_repair_pair_v3_20260730.md"
)
QUALIFICATION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v3_qualification.json"
)
VIABILITY_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v3_viability_identity.json"
)
V2_TERMINAL_AUDIT = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_pair_v2r5_audit.json"
)
DIRECT_TRAINER = (
    ROOT / "ops/train_point_maze_algorithm_repair_v2_direct.py"
)
STAGE_TRAINER = ROOT / "ops/train_point_maze_stage_b_05b_12pass.py"
BASE_AUDITOR = ROOT / "ops/audit_point_maze_algorithm_repair_pair_v1.py"

stage.PROTOCOL = PROTOCOL
stage.TRAINER = ROOT / "ops/train_point_maze_algorithm_repair_v3.py"
stage.AUDITOR = (
    ROOT / "ops/audit_point_maze_algorithm_repair_pair_v3.py"
)
stage.TRAIN_BATCH = (
    ROOT / "ops/slurm/train_point_maze_algorithm_repair_v3.slurm"
)
stage.AUDIT_BATCH = (
    ROOT / "ops/slurm/audit_point_maze_algorithm_repair_pair_v3.slurm"
)
stage.QUALIFICATION = QUALIFICATION
stage.SMOKE_IDENTITY = VIABILITY_IDENTITY
stage.DATA = ROOT / "var/data/point_maze_algorithm_repair_v3"
stage.IDENTITY = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_pair_v3_identity.json"
)
stage.MANIFEST = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v3_jobs.tsv"
)
stage.SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_pair_v3_submission.json"
)
stage.AUDIT_OUTPUT = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_pair_v3_audit.json"
)
stage.AUDIT_RUNNER = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_pair_v3_audit_runner_identity.json"
)
stage.SEEDS = (76541,)
stage.VARIANT = "algorithm_repair_v3"
stage.ARTIFACT_STEM = "point_maze_algorithm_repair_v3"
_atomic = stage.atomic
_snapshot_tree = stage.snapshot_tree


def configure_variant(_variant: str) -> None:
    return None


def base_prerequisites() -> None:
    for path in (
        stage.PYTHON,
        stage.PROTOCOL,
        stage.TRAINER,
        DIRECT_TRAINER,
        STAGE_TRAINER,
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
        V2_TERMINAL_AUDIT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    v2 = json.loads(V2_TERMINAL_AUDIT.read_text(encoding="utf-8"))
    if (
        v2.get("schema")
        != "point-maze-algorithm-repair-pair-audit-v2"
        or v2.get("status") not in {"pass", "fail"}
    ):
        raise RuntimeError("PointMaze v2 pair audit is not terminal")
    viability_identity = json.loads(
        VIABILITY_IDENTITY.read_text(encoding="utf-8")
    )
    if (
        viability_identity.get("schema_version")
        != "point-maze-algorithm-repair-v3-viability-identity-v1"
        or viability_identity.get("seed") != 76530
        or viability_identity.get("final_seed") is not False
        or viability_identity.get("evaluation_rows_loaded") is not False
    ):
        raise RuntimeError("PointMaze v3 viability identity drift")


def qualification_passes() -> None:
    payload = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "point-maze-algorithm-repair-v3-qualification-v1"
        or payload.get("status") != "pass"
        or payload.get("decision")
        != "eligible_for_point_maze_algorithm_repair_v3_pair"
        or payload.get("errors") not in ([], None)
    ):
        raise RuntimeError("PointMaze v3 viability did not authorize pair")


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
            str(DIRECT_TRAINER),
            str(STAGE_TRAINER),
            str(stage.BASE_TRAINER),
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


def snapshot_tree(source: Path, _prefix: str):
    return _snapshot_tree(source, "point_algorithm_repair_pair_v3_source")


def snapshot_execution():
    inputs = (
        stage.TRAINER,
        DIRECT_TRAINER,
        STAGE_TRAINER,
        stage.BASE_TRAINER,
        stage.AUDITOR,
        BASE_AUDITOR,
        stage.TRAIN_BATCH,
        stage.AUDIT_BATCH,
        stage.PROTOCOL,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".point-algorithm-repair-v3.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = stage.tree_hash(temporary)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_algorithm_repair_v3_ops_{digest}"
    )
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if stage.tree_hash(target) != digest:
        raise RuntimeError("PointMaze repair-v3 snapshot mismatch")
    return target, digest


def atomic(path: Path, payload) -> None:
    if path == stage.IDENTITY:
        v2 = json.loads(V2_TERMINAL_AUDIT.read_text(encoding="utf-8"))
        payload.update(
            repair_schema="point-maze-algorithm-repair-pair-identity-v1",
            repair_v3_schema=(
                "point-maze-algorithm-repair-pair-identity-v3"
            ),
            repair_launcher_sha256=hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            orientation_balanced=True,
            orientation_counts={
                "train": {"0": 2, "1": 2, "2": 2, "3": 2},
                "development": {"0": 1, "1": 1, "2": 1, "3": 1},
            },
            final_seed_cohort=False,
            development_only=True,
            evaluation_split="development",
            evaluation_common_random_numbers=True,
            secondary_post_outcome_repair=True,
            v2_terminal_audit_sha256=stage.sha(V2_TERMINAL_AUDIT),
            v2_terminal_status=v2.get("status"),
            v2_terminal_outcome_used_to_select_design=False,
            viability_identity_sha256=stage.sha(VIABILITY_IDENTITY),
            evaluation_rows_loaded=False,
            development_rows_loaded_for_training=False,
            development_evaluation_rows_loaded=True,
        )
    elif path == stage.SUBMISSION:
        payload.update(
            repair_schema=(
                "point-maze-algorithm-repair-pair-submission-v1"
            ),
            repair_v3_schema=(
                "point-maze-algorithm-repair-pair-submission-v3"
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
