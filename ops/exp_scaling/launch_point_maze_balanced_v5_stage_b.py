#!/usr/bin/env python3
"""Configure or conditionally launch the PointMaze balanced-v5 ten-cell cohort."""

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
    "point_maze_balanced_v5_stage_b_05b_12pass_20260730.md"
)
QUALIFICATION = (
    ROOT / "var/artifacts/point_maze_balanced_warmstart_v5_qualification.json"
)
WARMSTART_IDENTITY = (
    ROOT / "var/artifacts/point_maze_balanced_warmstart_v5_identity.json"
)
WARMSTART_SFT = (
    ROOT / "var/artifacts/point_maze_balanced_warmstart_sft_v5.json"
)
WARMSTART_VIABILITY = (
    ROOT / "var/artifacts/point_maze_balanced_warmstart_v5_viability.json"
)
DIRECT_TRAINER = ROOT / "ops/train_point_maze_algorithm_repair_v2_direct.py"
STAGE_TRAINER = ROOT / "ops/train_point_maze_stage_b_05b_12pass.py"

stage.PROTOCOL = PROTOCOL
stage.TRAINER = (
    ROOT / "ops/train_point_maze_balanced_v5_stage_b_05b_12pass.py"
)
stage.AUDITOR = ROOT / "ops/audit_point_maze_stage_b_05b_12pass.py"
stage.TRAIN_BATCH = (
    ROOT
    / "ops/slurm/train_point_maze_balanced_v5_stage_b_05b_12pass.slurm"
)
stage.AUDIT_BATCH = (
    ROOT
    / "ops/slurm/audit_point_maze_balanced_v5_stage_b_05b_12pass.slurm"
)
stage.QUALIFICATION = QUALIFICATION
stage.SMOKE_IDENTITY = WARMSTART_IDENTITY
stage.MODEL = ROOT / "var/models/point_maze_interactive_warmstart_v5_balanced"
stage.DATA = ROOT / "var/data/point_maze_algorithm_repair_v3"
stage.IDENTITY = (
    ROOT
    / "var/artifacts/point_maze_balanced_v5_stage_b_05b_12pass_identity.json"
)
stage.MANIFEST = (
    ROOT / "var/artifacts/point_maze_balanced_v5_stage_b_05b_12pass_jobs.tsv"
)
stage.SUBMISSION = (
    ROOT
    / "var/artifacts/point_maze_balanced_v5_stage_b_05b_12pass_submission.json"
)
stage.AUDIT_OUTPUT = (
    ROOT / "var/artifacts/point_maze_balanced_v5_stage_b_05b_12pass_audit.json"
)
stage.AUDIT_RUNNER = (
    ROOT
    / "var/artifacts/"
    "point_maze_balanced_v5_stage_b_05b_12pass_audit_runner_identity.json"
)
stage.SEEDS = (76611, 76612, 76613, 76614, 76615)
stage.VARIANT = "balanced_v5"
stage.ARTIFACT_STEM = "point_maze_balanced_v5_stage_b_05b_12pass"
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
        stage.TRAIN_BATCH,
        stage.AUDIT_BATCH,
        stage.SMOKE_IDENTITY,
        stage.DATA / "identity.json",
        stage.DATA / "train/dataset_dict.json",
        stage.DATA / "eval/dataset_dict.json",
        stage.WORKER,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    warmstart = json.loads(WARMSTART_IDENTITY.read_text(encoding="utf-8"))
    if (
        warmstart.get("schema_version")
        != "point-maze-balanced-warmstart-v5-identity-v1"
        or warmstart.get("sft_seed") != 76601
        or warmstart.get("development_seed") != 76602
        or warmstart.get("orientation_counts")
        != {"0": 2, "1": 2, "2": 2, "3": 2}
        or warmstart.get("evaluation_rows_loaded") is not False
        or warmstart.get("final_seed") is not False
    ):
        raise RuntimeError("PointMaze balanced-v5 warm-start identity drift")


def qualification_passes() -> None:
    for path in (
        QUALIFICATION,
        WARMSTART_SFT,
        WARMSTART_VIABILITY,
        stage.MODEL / "config.json",
    ):
        if not path.exists():
            raise RuntimeError(f"PointMaze balanced-v5 prerequisite absent: {path}")
    payload = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "point-maze-balanced-warmstart-v5-qualification-v1"
        or payload.get("status") != "pass"
        or payload.get("decision")
        != "eligible_for_point_maze_v5_five_seed_pair"
        or payload.get("errors") not in ([], None)
    ):
        raise RuntimeError("PointMaze balanced-v5 qualification did not authorize final")


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
            str(ROOT / "tests/test_point_maze_balanced_v5_stage_b.py"),
            str(ROOT / "tests/test_point_maze_stage_b_05b_12pass.py"),
            str(ROOT / "tests/test_interactive_episode_objective.py"),
            str(ROOT / "tests/test_interactive_episode_replay.py"),
            str(ROOT / "tests/test_point_maze_interactive_policy.py"),
            str(ROOT / "tests/test_point_maze_interactive_worker.py"),
        ],
        env=environment,
    )


def snapshot_tree(source: Path, _prefix: str):
    return _snapshot_tree(source, "point_balanced_v5_stage_b_source")


def snapshot_execution():
    inputs = (
        stage.TRAINER,
        DIRECT_TRAINER,
        STAGE_TRAINER,
        stage.BASE_TRAINER,
        stage.AUDITOR,
        stage.TRAIN_BATCH,
        stage.AUDIT_BATCH,
        stage.PROTOCOL,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".point-balanced-v5-stage-b.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = stage.tree_hash(temporary)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_balanced_v5_stage_b_ops_{digest}"
    )
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if stage.tree_hash(target) != digest:
        raise RuntimeError("PointMaze balanced-v5 execution snapshot mismatch")
    return target, digest


def atomic(path: Path, payload) -> None:
    if path == stage.IDENTITY:
        payload.update(
            schema="point-maze-balanced-v5-stage-b-05b-12pass-identity-v1",
            launcher_sha256=hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            warmstart_identity_sha256=stage.sha(WARMSTART_IDENTITY),
            warmstart_sft_sha256=stage.sha(WARMSTART_SFT),
            warmstart_viability_sha256=stage.sha(WARMSTART_VIABILITY),
            qualification_audit_sha256=stage.sha(QUALIFICATION),
            orientation_balanced_warmstart=True,
            orientation_counts={
                "train": {"0": 2, "1": 2, "2": 2, "3": 2},
                "evaluation": {"0": 1, "1": 1, "2": 1, "3": 1},
            },
            evaluation_split="untouched_evaluation",
            development_rows_loaded_for_training=False,
            evaluation_rows_loaded_before_final=False,
            evaluation_common_random_numbers=True,
            checkpoint_invariant_evaluation_seeds=True,
        )
    elif path == stage.SUBMISSION:
        payload.update(
            schema="point-maze-balanced-v5-stage-b-05b-12pass-submission-v1"
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
