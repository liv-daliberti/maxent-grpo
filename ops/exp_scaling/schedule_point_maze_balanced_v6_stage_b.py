#!/usr/bin/env python3
"""Freeze and schedule the PointMaze balanced-v6 final after its gate."""

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
GATE_JOB = 30206697
DEPENDENCY_BATCH = (
    ROOT
    / "ops/slurm/launch_point_maze_balanced_v6_stage_b_after_gate.slurm"
)
LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_point_maze_balanced_v6_stage_b.py"
)
MANIFEST = (
    ROOT
    / "var/artifacts/"
    "point_maze_balanced_v6_stage_b_dependent_launcher_identity.json"
)
FILES = (
    ROOT
    / "paper/preregistration/"
    "point_maze_balanced_v6_stage_b_05b_12pass_20260730.md",
    ROOT / "ops/train_point_maze_balanced_v6_stage_b_05b_12pass.py",
    ROOT / "ops/train_point_maze_algorithm_repair_v2_direct.py",
    ROOT / "ops/train_point_maze_stage_b_05b_12pass.py",
    ROOT / "ops/train_point_maze_interactive_paired_smoke_v1.py",
    ROOT / "ops/audit_point_maze_balanced_v6_stage_b_05b_12pass.py",
    ROOT
    / "ops/slurm/train_point_maze_balanced_v6_stage_b_05b_12pass.slurm",
    ROOT
    / "ops/slurm/audit_point_maze_balanced_v6_stage_b_05b_12pass.slurm",
    LAUNCHER,
    ROOT / "ops/exp_scaling/launch_point_maze_stage_b_05b_12pass.py",
    DEPENDENCY_BATCH,
    Path(__file__).resolve(),
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


def schedule() -> None:
    if MANIFEST.exists():
        raise FileExistsError(
            f"fresh PointMaze v6 dependent identity required: {MANIFEST}"
        )
    hashes = file_hashes()
    output = run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            f"--dependency=afterany:{GATE_JOB}",
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
            str(GATE_JOB),
            "Account=allcs",
            "Requeue=0",
            f"OAT_ZERO_FROZEN_MANIFEST={MANIFEST}",
        ):
            if required not in record:
                raise RuntimeError(
                    f"held PointMaze v6 launcher lacks {required}"
                )
        atomic(
            MANIFEST,
            {
                "schema_version": (
                    "point-maze-balanced-v6-dependent-launcher-v1"
                ),
                "gate_job_id": GATE_JOB,
                "dependent_launcher_job_id": job_id,
                "file_sha256": hashes,
                "held_scheduler_record": record,
                "released": True,
                "gate_outcome_loaded": False,
                "online_pair_sampled": False,
                "evaluation_rows_loaded": False,
            },
        )
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[point-balanced-v6] scheduled final launcher {job_id}")


def verify_run(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "point-maze-balanced-v6-dependent-launcher-v1"
        or payload.get("gate_job_id") != GATE_JOB
        or payload.get("file_sha256") != file_hashes()
        or payload.get("gate_outcome_loaded") is not False
        or payload.get("online_pair_sampled") is not False
        or payload.get("evaluation_rows_loaded") is not False
    ):
        raise RuntimeError("PointMaze balanced-v6 frozen launcher drift")
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
