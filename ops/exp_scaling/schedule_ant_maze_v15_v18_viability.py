#!/usr/bin/env python3
"""Freeze Ant v18 viability now and bridge its future admission job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
ADMISSION_LAUNCHER_JOB = 30205648
ADMISSION_SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v18_admission_submission.json"
)
PREPARER_BATCH = (
    ROOT
    / "ops/slurm/prepare_ant_maze_v15_v18_viability_dependency.slurm"
)
DEPENDENT_BATCH = (
    ROOT
    / "ops/slurm/"
    "launch_ant_maze_v15_v18_viability_after_admission.slurm"
)
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/"
    "launch_ant_maze_v15_controller_v18_viability.py"
)
OUTER_MANIFEST = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_v18_viability_preparer_identity.json"
)
INNER_MANIFEST = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_v18_viability_dependent_launcher_identity.json"
)
FILES = (
    ROOT
    / "paper/preregistration/"
    "ant_maze_v15_controller_v18_05b_viability_20260730.md",
    ROOT / "ops/evaluate_ant_maze_interactive_viability_v18.py",
    ROOT / "ops/evaluate_point_maze_interactive_viability.py",
    ROOT / "ops/qualify_ant_maze_v15_controller_v18_viability.py",
    ROOT / "ops/slurm/evaluate_ant_maze_v15_controller_v18_viability.slurm",
    LAUNCHER,
    PREPARER_BATCH,
    DEPENDENT_BATCH,
    Path(__file__).resolve(),
    ROOT / "src/oat_drgrpo/ant_maze_worker_v18.py",
    ROOT / "src/oat_drgrpo/ant_maze_interactive_worker_v18.py",
    ROOT / "src/oat_drgrpo/ant_maze_interactive_process_v18.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def file_hashes() -> dict[str, str]:
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {str(path.relative_to(ROOT)): sha(path) for path in FILES}


def validate_manifest(path: Path, schema: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != schema
        or payload.get("file_sha256") != file_hashes()
        or payload.get("language_model_sampled") is not False
        or payload.get("evaluation_rows_loaded") is not False
    ):
        raise RuntimeError(f"Ant v18 viability manifest drift: {path}")
    return payload


def freeze() -> None:
    if OUTER_MANIFEST.exists() or INNER_MANIFEST.exists():
        raise FileExistsError("fresh Ant v18 viability manifests required")
    hashes = file_hashes()
    output = run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            f"--dependency=afterany:{ADMISSION_LAUNCHER_JOB}",
            "--partition=all",
            "--account=allcs",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_FROZEN_MANIFEST={OUTER_MANIFEST}",
            str(PREPARER_BATCH),
        ]
    )
    job_id = int(output.split(";", 1)[0])
    try:
        run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        record = run(["scontrol", "show", "job", "-o", str(job_id)])
        for required in (
            "JobState=PENDING",
            str(ADMISSION_LAUNCHER_JOB),
            "Account=allcs",
            "Requeue=0",
        ):
            if required not in record:
                raise RuntimeError(f"held Ant v18 preparer lacks {required}")
        atomic(
            OUTER_MANIFEST,
            {
                "schema_version": "ant-maze-v15-v18-viability-preparer-v1",
                "admission_launcher_job_id": ADMISSION_LAUNCHER_JOB,
                "preparer_job_id": job_id,
                "file_sha256": hashes,
                "held_scheduler_record": record,
                "language_model_sampled": False,
                "evaluation_rows_loaded": False,
                "controller_outcome_loaded": False,
                "admission_outcome_loaded": False,
            },
        )
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[ant-v15-v18-viability] froze preparer {job_id}")


def schedule_after_admission(manifest: Path) -> None:
    outer = validate_manifest(
        manifest,
        "ant-maze-v15-v18-viability-preparer-v1",
    )
    if (
        outer.get("admission_launcher_job_id") != ADMISSION_LAUNCHER_JOB
        or INNER_MANIFEST.exists()
    ):
        raise RuntimeError("Ant v18 viability preparer identity drift")
    submission = json.loads(ADMISSION_SUBMISSION.read_text(encoding="utf-8"))
    admission_job = submission.get("job_id")
    if not isinstance(admission_job, int) or submission.get("released") is not True:
        raise RuntimeError("Ant v18 admission job identity is unavailable")
    output = run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            f"--dependency=afterany:{admission_job}",
            "--partition=all",
            "--account=allcs",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_FROZEN_MANIFEST={INNER_MANIFEST}",
            str(DEPENDENT_BATCH),
        ]
    )
    job_id = int(output.split(";", 1)[0])
    try:
        run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        record = run(["scontrol", "show", "job", "-o", str(job_id)])
        for required in (
            "JobState=PENDING",
            str(admission_job),
            "Account=allcs",
            "Requeue=0",
        ):
            if required not in record:
                raise RuntimeError(f"held Ant v18 viability launcher lacks {required}")
        atomic(
            INNER_MANIFEST,
            {
                "schema_version": (
                    "ant-maze-v15-v18-viability-dependent-launcher-v1"
                ),
                "outer_manifest_sha256": sha(manifest),
                "admission_job_id": admission_job,
                "dependent_launcher_job_id": job_id,
                "file_sha256": file_hashes(),
                "held_scheduler_record": record,
                "language_model_sampled": False,
                "evaluation_rows_loaded": False,
                "admission_outcome_loaded": False,
            },
        )
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[ant-v15-v18-viability] scheduled launcher {job_id}")


def verify_run(manifest: Path) -> None:
    payload = validate_manifest(
        manifest,
        "ant-maze-v15-v18-viability-dependent-launcher-v1",
    )
    submission = json.loads(ADMISSION_SUBMISSION.read_text(encoding="utf-8"))
    if (
        payload.get("admission_job_id") != submission.get("job_id")
        or payload.get("admission_outcome_loaded") is not False
    ):
        raise RuntimeError("Ant v18 viability admission dependency drift")
    subprocess.run(
        [str(PYTHON), str(LAUNCHER), "run"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("freeze", "schedule-after-admission", "verify-run"),
    )
    parser.add_argument("--manifest", type=Path, default=OUTER_MANIFEST)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "schedule-after-admission":
        schedule_after_admission(args.manifest)
    else:
        verify_run(args.manifest)


if __name__ == "__main__":
    main()
