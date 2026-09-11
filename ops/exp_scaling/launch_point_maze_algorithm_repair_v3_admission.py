#!/usr/bin/env python3
"""Configure or launch frozen PointMaze v3 executable admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "point_maze_algorithm_repair_v3_orientation_20260730.md"
)
MAKER = ROOT / "ops/make_point_maze_algorithm_repair_data_v3.py"
MAKER_BASE = ROOT / "ops/make_point_maze_mode_data.py"
MEDIUM_FAMILIES = ROOT / "ops/make_point_maze_geometry_shift_data.py"
HARD_FAMILIES = ROOT / "ops/make_point_maze_algorithm_repair_data_v1.py"
AUDITOR = ROOT / "ops/audit_point_maze_algorithm_repair_data_v3.py"
AUDITOR_BASE = ROOT / "ops/audit_point_maze_mode_data.py"
BATCH = ROOT / "ops/slurm/admit_point_maze_algorithm_repair_v3.slurm"
DATA = ROOT / "var/data/point_maze_algorithm_repair_v3"
AUDIT = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v3_admission_audit.json"
)
IDENTITY = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v3_admission_identity.json"
)
SUBMISSION = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_v3_admission_submission.json"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


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


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def snapshot_tree(source: Path) -> tuple[Path, str]:
    digest = tree_hash(source)
    parent = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_repair_v3_source_{digest}"
    )
    target = parent / "src"
    if not target.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot.", dir=parent))
        shutil.copytree(source, staging / "src")
        os.replace(staging / "src", target)
        staging.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze v3 source snapshot mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    inputs = (
        PROTOCOL,
        MAKER,
        MAKER_BASE,
        MEDIUM_FAMILIES,
        HARD_FAMILIES,
        AUDITOR,
        AUDITOR_BASE,
        BATCH,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".point-repair-v3-admission.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in inputs:
        shutil.copy2(source, staging / source.name)
    digest = tree_hash(staging)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_repair_v3_admission_ops_{digest}"
    )
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze v3 execution snapshot mismatch")
    return target, digest


def validate() -> None:
    for path in (
        PYTHON,
        PROTOCOL,
        MAKER,
        MAKER_BASE,
        MEDIUM_FAMILIES,
        HARD_FAMILIES,
        AUDITOR,
        AUDITOR_BASE,
        BATCH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(MAKER),
            str(AUDITOR),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    run(["bash", "-n", str(BATCH)])
    run(
        [
            str(PYTHON),
            "-c",
            (
                "import make_point_maze_algorithm_repair_data_v3 as v;"
                "c={s:{r:sum(x[2]==r for x in a) for r in range(4)} "
                "for s,a in v.SPLIT_ASSIGNMENTS.items()};"
                "assert c=={'train':{0:2,1:2,2:2,3:2},"
                "'dev':{0:1,1:1,2:1,3:1},"
                "'eval':{0:1,1:1,2:1,3:1}};"
                "assert all(len(a)==n for (s,a),n in "
                "zip(v.SPLIT_ASSIGNMENTS.items(),(8,4,4)))"
            ),
        ],
        env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    validate()
    if args.phase == "config":
        print("[point-repair-v3-admission] configuration passed")
        return
    for path in (DATA, AUDIT, IDENTITY, SUBMISSION):
        if path.exists():
            raise FileExistsError(
                f"fresh PointMaze v3 admission artifact required: {path}"
            )
    source_root, source_hash = snapshot_tree(ROOT / "src")
    execution_root, execution_hash = snapshot_execution()
    output = run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            "--partition=all",
            "--account=allcs",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
            f"OAT_ZERO_EXECUTION_ROOT={execution_root}",
            str(execution_root / BATCH.name),
        ]
    )
    job_id = int(output.split(";", 1)[0])
    try:
        run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        record = run(["scontrol", "show", "job", "-o", str(job_id)])
        for required in (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "Account=allcs",
            "NumCPUs=4",
            "MinMemoryNode=32G",
            "TimeLimit=01:00:00",
            "Requeue=0",
            f"OAT_ZERO_SOURCE_ROOT={source_root}",
            f"OAT_ZERO_EXECUTION_ROOT={execution_root}",
        ):
            if required not in record:
                raise RuntimeError(
                    f"held PointMaze v3 admission job lacks {required}"
                )
        atomic(
            IDENTITY,
            {
                "schema_version": (
                    "point-maze-algorithm-repair-v3-admission-identity-v1"
                ),
                "job_id": job_id,
                "protocol_sha256": sha(PROTOCOL),
                "launcher_sha256": sha(Path(__file__).resolve()),
                "maker_sha256": sha(MAKER),
                "audit_sha256": sha(AUDITOR),
                "batch_sha256": sha(BATCH),
                "source_root": str(source_root),
                "source_hash": source_hash,
                "execution_root": str(execution_root),
                "execution_hash": execution_hash,
                "orientation_counts": {
                    "train": {"0": 2, "1": 2, "2": 2, "3": 2},
                    "dev": {"0": 1, "1": 1, "2": 1, "3": 1},
                    "eval": {"0": 1, "1": 1, "2": 1, "3": 1},
                },
                "v2_terminal_pair_outcome_loaded": False,
                "v3_route_outcome_loaded": False,
                "v3_model_sampled": False,
                "coefficient_changed": False,
                "held_scheduler_record": record,
            },
        )
        atomic(
            SUBMISSION,
            {
                "schema_version": (
                    "point-maze-algorithm-repair-v3-admission-submission-v1"
                ),
                "job_id": job_id,
                "identity_sha256": sha(IDENTITY),
                "released": True,
            },
        )
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[point-repair-v3-admission] released job {job_id}")


if __name__ == "__main__":
    main()
