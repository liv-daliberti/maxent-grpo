#!/usr/bin/env python3
"""Freeze and schedule the Ant v15/v18 admission launcher dependency."""

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
CONTROLLER_JOB = 30205570
DEPENDENCY_BATCH = (
    ROOT
    / "ops/slurm/"
    "launch_ant_maze_v15_controller_v18_after_controller.slurm"
)
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/"
    "launch_ant_maze_v15_controller_v18_admission.py"
)
MANIFEST = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v18_dependent_launcher_identity.json"
)
FILES = (
    ROOT
    / "paper/preregistration/"
    "ant_maze_v15_controller_v18_admission_20260730.md",
    ROOT / "ops/make_ant_maze_mode_data_v15_controller_v18.py",
    ROOT / "ops/audit_ant_maze_mode_data_v15_controller_v18.py",
    ROOT / "ops/make_ant_maze_mode_data.py",
    ROOT / "ops/audit_ant_maze_mode_data.py",
    LAUNCHER,
    ROOT / "ops/exp_scaling/launch_ant_maze_v14_hard_admission.py",
    ROOT / "ops/slurm/admit_ant_maze_modebench_v15_controller_v18.slurm",
    DEPENDENCY_BATCH,
    Path(__file__).resolve(),
    ROOT / "src/oat_drgrpo/ant_maze_worker_v18.py",
    ROOT / "src/oat_drgrpo/maze_modebench_worker_v18.py",
    ROOT / "src/oat_drgrpo/maze_modebench_process_v18.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def file_hashes() -> dict[str, str]:
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {str(path.relative_to(ROOT)): sha(path) for path in FILES}


def schedule() -> None:
    if MANIFEST.exists():
        raise FileExistsError(
            f"fresh dependent-launcher identity required: {MANIFEST}"
        )
    hashes = file_hashes()
    output = run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            f"--dependency=afterok:{CONTROLLER_JOB}",
            "--partition=all",
            "--account=allcs",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_FROZEN_MANIFEST={MANIFEST}",
            str(DEPENDENCY_BATCH),
        ]
    )
    job_id = int(output.split(";", 1)[0])
    try:
        run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        record = run(["scontrol", "show", "job", "-o", str(job_id)])
        for required in (
            "JobState=PENDING",
            f"Dependency=afterok:{CONTROLLER_JOB}",
            "Account=allcs",
            "Requeue=0",
            f"OAT_ZERO_FROZEN_MANIFEST={MANIFEST}",
        ):
            if required not in record:
                raise RuntimeError(
                    f"held dependent launcher lacks {required}"
                )
        atomic(
            MANIFEST,
            {
                "schema_version": (
                    "ant-maze-v15-controller-v18-dependent-launcher-v1"
                ),
                "controller_job_id": CONTROLLER_JOB,
                "dependent_launcher_job_id": job_id,
                "file_sha256": hashes,
                "held_scheduler_record": record,
                "released": True,
                "language_model_sampled": False,
                "post_outcome_map_or_route_substitution": False,
            },
        )
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[ant-v15-v18] scheduled dependent launcher {job_id}")


def verify_run(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "ant-maze-v15-controller-v18-dependent-launcher-v1"
    ):
        raise RuntimeError("Ant v15/v18 dependent-launcher schema mismatch")
    if payload.get("controller_job_id") != CONTROLLER_JOB:
        raise RuntimeError("Ant v15/v18 controller dependency drift")
    if payload.get("file_sha256") != file_hashes():
        raise RuntimeError("Ant v15/v18 frozen launcher inputs drifted")
    subprocess.run(
        [str(PYTHON), str(LAUNCHER), "run"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("schedule", "verify-run"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.phase == "schedule":
        schedule()
    else:
        verify_run(args.manifest)


if __name__ == "__main__":
    main()
