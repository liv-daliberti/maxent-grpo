#!/usr/bin/env python3
"""Schedule the PointMaze final launcher after the live pair audit passes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/"
    "launch_point_maze_algorithm_repair_final_v1.py"
)
BATCH = (
    ROOT
    / "ops/slurm/"
    "launch_point_maze_algorithm_repair_final_v1_after_pair.slurm"
)
PAIR_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_pair_v2r5_identity.json"
)
OUT = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_final_v1_dependent_launcher_identity.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def atomic(payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{OUT.name}.", dir=OUT.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    for path in (LAUNCHER, BATCH, PAIR_IDENTITY):
        if not path.is_file():
            raise FileNotFoundError(path)
    run(["bash", "-n", str(BATCH)])
    pair = json.loads(PAIR_IDENTITY.read_text(encoding="utf-8"))
    audit_job = int(pair.get("audit_job_id", 0))
    if audit_job != 30204904:
        raise RuntimeError("unexpected PointMaze repair-pair audit dependency")
    launcher_hash = sha(LAUNCHER)
    output = run(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{audit_job}",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},EXPECTED_LAUNCHER_SHA256={launcher_hash}",
            str(BATCH),
        ]
    )
    job_id = int(output.split(";", 1)[0])
    record = run(["scontrol", "show", "job", "-o", str(job_id)])
    for required in (
        "JobState=PENDING",
        "Reason=Dependency",
        f"Dependency=afterok:{audit_job}",
        "Account=allcs",
        "NumCPUs=2",
        "MinMemoryNode=8G",
        "TimeLimit=01:00:00",
    ):
        if required not in record:
            subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
            raise RuntimeError(f"dependent launcher lacks {required}")
    atomic(
        {
            "schema": (
                "point-maze-algorithm-repair-final-dependent-launcher-v1"
            ),
            "job_id": job_id,
            "dependency": f"afterok:{audit_job}",
            "pair_identity_sha256": sha(PAIR_IDENTITY),
            "launcher_sha256": launcher_hash,
            "batch_sha256": sha(BATCH),
            "held_scheduler_record": record,
            "scientific_condition": (
                "pair audit must exit zero after emitting its passing receipt"
            ),
        }
    )
    print(
        f"[point-repair-final-schedule] job={job_id} "
        f"dependency=afterok:{audit_job}"
    )


if __name__ == "__main__":
    main()
