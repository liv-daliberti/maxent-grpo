#!/usr/bin/env python3
"""Configure or launch the prospective AntMaze v14-hard admission gate."""

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
PROTOCOL = ROOT / "paper/preregistration/ant_maze_v14_hard_admission_20260730.md"
MAKER = ROOT / "ops/make_ant_maze_mode_data_v14_hard.py"
MAKER_BASE = ROOT / "ops/make_ant_maze_mode_data.py"
AUDITOR = ROOT / "ops/audit_ant_maze_mode_data_v14_hard.py"
AUDITOR_BASE = ROOT / "ops/audit_ant_maze_mode_data.py"
BATCH = ROOT / "ops/slurm/admit_ant_maze_modebench_v14_hard.slurm"
DATA = ROOT / "var/data/ant_maze_modebench_v14_hard"
AUDIT = ROOT / "var/artifacts/ant_maze_modebench_v14_hard_admission_audit.json"
IDENTITY = ROOT / "var/artifacts/ant_maze_v14_hard_admission_identity.json"
SUBMISSION = ROOT / "var/artifacts/ant_maze_v14_hard_admission_submission.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        list(command), cwd=ROOT, env=env, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def snapshot_tree(source: Path, prefix: str) -> tuple[Path, str]:
    digest = tree_hash(source)
    parent = ROOT / f"var/artifacts/source_snapshots/{prefix}_{digest}"
    target = parent / source.name
    if not target.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot.", dir=parent))
        shutil.copytree(source, staging / source.name)
        os.replace(staging / source.name, target)
        staging.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError(f"{prefix} source snapshot mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    inputs = (PROTOCOL, MAKER, MAKER_BASE, AUDITOR, AUDITOR_BASE, BATCH)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".ant-v14-hard.", dir=ROOT / "var/artifacts/source_snapshots"
        )
    )
    for source in inputs:
        shutil.copy2(source, staging / source.name)
    digest = tree_hash(staging)
    target = ROOT / f"var/artifacts/source_snapshots/ant_v14_hard_ops_{digest}"
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if tree_hash(target) != digest:
        raise RuntimeError("AntMaze v14-hard execution snapshot mismatch")
    return target, digest


def validate() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    environment["LD_LIBRARY_PATH"] = library + (
        ":" + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
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
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_ant_maze_route_admission_v12.py"),
            str(ROOT / "tests/test_ant_maze_worker_v5.py"),
        ],
        env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    parsed = parser.parse_args()
    for path in (PYTHON, PROTOCOL, MAKER, MAKER_BASE, AUDITOR, AUDITOR_BASE, BATCH):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate()
    if parsed.phase == "config":
        print("[ant-v14-hard] configuration passed; no route executed")
        return
    for path in (DATA, AUDIT, IDENTITY, SUBMISSION):
        if path.exists():
            raise FileExistsError(f"fresh AntMaze v14-hard artifact required: {path}")
    source_root, source_hash = snapshot_tree(ROOT / "src", "ant_v14_hard_source")
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
            f"OAT_ZERO_OPS_ROOT={execution_root},OAT_ZERO_PROTOCOL_IDENTITY={IDENTITY}",
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
            "TimeLimit=04:00:00",
            "Requeue=0",
            f"OAT_ZERO_PROTOCOL_IDENTITY={IDENTITY}",
        ):
            if required not in record:
                raise RuntimeError(f"held AntMaze v14-hard job lacks {required}")
        atomic(
            IDENTITY,
            {
                "schema": "ant-maze-v14-hard-admission-identity-v1",
                "job_id": job_id,
                "protocol_sha256": sha(PROTOCOL),
                "launcher_sha256": sha(Path(__file__).resolve()),
                "source_root": str(source_root),
                "source_hash": source_hash,
                "execution_root": str(execution_root),
                "execution_hash": execution_hash,
                "held_scheduler_record": record,
                "map_size": 15,
                "central_obstacle_shape": [5, 5],
                "route_decisions": 14,
                "max_actions": 24,
                "action_repeat": 400,
                "model_sampled": False,
                "post_outcome_map_substitution": False,
            },
        )
        atomic(
            SUBMISSION,
            {
                "schema": "ant-maze-v14-hard-admission-submission-v1",
                "job_id": job_id,
                "identity_sha256": sha(IDENTITY),
                "released": True,
            },
        )
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[ant-v14-hard] released admission job {job_id}")


if __name__ == "__main__":
    main()

