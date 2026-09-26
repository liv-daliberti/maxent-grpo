#!/usr/bin/env python3
"""Configure or launch the frozen short orientation-balanced PointMaze v6 warm start."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "ops", ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from make_point_maze_algorithm_repair_data_v3 import _task_fingerprint  # noqa: E402


PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
BASE_MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "point_maze_balanced_short_warmstart_v6_20260730.md"
)
SOURCE_MAKER = ROOT / "ops/make_point_maze_balanced_warmstart_v5_data.py"
EXAMPLE_MAKER = ROOT / "ops/materialize_point_maze_train_warmstart_v1.py"
TRAINER = ROOT / "ops/train_point_maze_interactive_warmstart_v1.py"
EVALUATOR = ROOT / "ops/evaluate_point_maze_interactive_viability.py"
QUALIFIER = ROOT / "ops/qualify_point_maze_balanced_short_warmstart_v6.py"
BATCH = ROOT / "ops/slurm/train_point_maze_balanced_short_warmstart_v6.slurm"
SOURCE_DATA = ROOT / "var/data/point_maze_balanced_warmstart_source_v5"
WARMSTART_DATA = ROOT / "var/data/point_maze_interactive_warmstart_v5_balanced"
MODEL = ROOT / "var/models/point_maze_interactive_warmstart_v6_balanced_short"
SFT_RECEIPT = ROOT / "var/artifacts/point_maze_balanced_short_warmstart_sft_v6.json"
VIABILITY = ROOT / "var/artifacts/point_maze_balanced_short_warmstart_v6_viability.json"
QUALIFICATION = (
    ROOT / "var/artifacts/point_maze_balanced_short_warmstart_v6_qualification.json"
)
IDENTITY = (
    ROOT / "var/artifacts/point_maze_balanced_short_warmstart_v6_identity.json"
)
SUBMISSION = (
    ROOT / "var/artifacts/point_maze_balanced_short_warmstart_v6_submission.json"
)
V3_ADMISSION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v3_admission_audit.json"
)
V3_VIABILITY = ROOT / "var/artifacts/point_maze_algorithm_repair_v3_viability.json"
V4_VIABILITY = ROOT / "var/artifacts/point_maze_algorithm_repair_v4_viability.json"
V5_QUALIFICATION = (
    ROOT / "var/artifacts/point_maze_balanced_warmstart_v5_qualification.json"
)


def sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def environment() -> dict[str, str]:
    value = dict(os.environ)
    value["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    value["LD_LIBRARY_PATH"] = library + (
        ":" + value["LD_LIBRARY_PATH"] if value.get("LD_LIBRARY_PATH") else ""
    )
    return value


def materialize() -> None:
    env = environment()
    if not SOURCE_DATA.exists():
        run(
            [
                str(PYTHON),
                str(SOURCE_MAKER),
                "--worker-python",
                str(WORKER),
                "--output-root",
                str(SOURCE_DATA),
            ],
            env=env,
        )
    if not WARMSTART_DATA.exists():
        run(
            [
                str(PYTHON),
                str(EXAMPLE_MAKER),
                "--data-root",
                str(SOURCE_DATA),
                "--train-split-root",
                str(SOURCE_DATA / "train"),
                "--worker-python",
                str(WORKER),
                "--protocol",
                str(PROTOCOL),
                "--output-root",
                str(WARMSTART_DATA),
                "--policy-interface",
                "velocity_state_v3",
            ],
            env=env,
        )


def validate() -> None:
    for path in (
        PYTHON,
        WORKER,
        BASE_MODEL / "config.json",
        PROTOCOL,
        SOURCE_MAKER,
        EXAMPLE_MAKER,
        TRAINER,
        EVALUATOR,
        QUALIFIER,
        BATCH,
        V3_ADMISSION,
        V3_VIABILITY,
        V4_VIABILITY,
        V5_QUALIFICATION,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    env = environment()
    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(SOURCE_MAKER),
            str(EXAMPLE_MAKER),
            str(TRAINER),
            str(EVALUATOR),
            str(QUALIFIER),
            str(Path(__file__).resolve()),
        ],
        env=env,
    )
    run(["bash", "-n", str(BATCH)])
    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_point_maze_balanced_short_warmstart_v6.py"),
            str(ROOT / "tests/test_point_maze_interactive_policy.py"),
            str(ROOT / "tests/test_point_maze_interactive_worker.py"),
        ],
        env=env,
    )


def validate_materialized() -> None:
    source = json.loads((SOURCE_DATA / "identity.json").read_text(encoding="utf-8"))
    warmstart = json.loads(
        (WARMSTART_DATA / "identity.json").read_text(encoding="utf-8")
    )
    if (
        source.get("status") != "pass"
        or source.get("orientation_counts")
        != {"0": 2, "1": 2, "2": 2, "3": 2}
        or source.get("expected_replayed_example_count") != 644
        or len(source.get("certification", [])) != 8
    ):
        raise RuntimeError("PointMaze v6 balanced source identity drift")
    task_fingerprints = {
        _task_fingerprint(json.loads(row["answer"]))
        for row in __import__("datasets").load_from_disk(
            str(SOURCE_DATA / "train")
        )["train"]
    }
    if len(task_fingerprints) != 8:
        raise RuntimeError("PointMaze v6 train executable identity collision")
    if (
        warmstart.get("status") != "pass"
        or warmstart.get("example_count") != 644
        or warmstart.get("episode_count") != 16
        or warmstart.get("policy_interface") != "velocity_state_v3"
        or warmstart.get("information_boundary", {}).get("dev_dataset_loaded")
        is not False
        or warmstart.get("information_boundary", {}).get("eval_dataset_loaded")
        is not False
    ):
        raise RuntimeError("PointMaze v6 replayed warm-start identity drift")


def snapshot_source() -> tuple[Path, str]:
    digest = tree_hash(ROOT / "src")
    parent = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_balanced_short_warmstart_v6_source_{digest}"
    )
    target = parent / "src"
    if not target.exists():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot.", dir=parent))
        shutil.copytree(ROOT / "src", staging / "src")
        os.replace(staging / "src", target)
        staging.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze v6 source snapshot mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    staging = Path(
        tempfile.mkdtemp(
            prefix=".point-balanced-short-warmstart-v6.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in (PROTOCOL, TRAINER, EVALUATOR, QUALIFIER, BATCH):
        shutil.copy2(source, staging / source.name)
    digest = tree_hash(staging)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_balanced_short_warmstart_v6_ops_{digest}"
    )
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze v6 execution snapshot mismatch")
    return target, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    validate()
    materialize()
    validate_materialized()
    if args.phase == "config":
        print(
            "[point-balanced-short-warmstart-v6] configuration passed; "
            "maps=8 episodes=16 examples=644 orientations=2/2/2/2"
        )
        return
    for path in (MODEL, SFT_RECEIPT, VIABILITY, QUALIFICATION, IDENTITY, SUBMISSION):
        if path.exists():
            raise FileExistsError(f"fresh PointMaze v6 artifact required: {path}")
    source_root, source_hash = snapshot_source()
    execution_root, execution_hash = snapshot_execution()
    job_text = run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            "--partition=all",
            "--account=mltheory",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
            f"OAT_ZERO_EXECUTION_ROOT={execution_root},"
            f"OAT_ZERO_SOURCE_HASH={source_hash},"
            f"OAT_ZERO_EXECUTION_HASH={execution_hash},"
            f"OAT_ZERO_VIABILITY_IDENTITY={IDENTITY}",
            str(execution_root / BATCH.name),
        ]
    )
    job_id = int(job_text.split(";", 1)[0])
    try:
        atomic(
            IDENTITY,
            {
                "schema_version": "point-maze-balanced-short-warmstart-v6-identity-v1",
                "job_id": job_id,
                "source_hash": source_hash,
                "execution_hash": execution_hash,
                "protocol_sha256": sha(PROTOCOL),
                "source_maker_sha256": sha(SOURCE_MAKER),
                "example_maker_sha256": sha(EXAMPLE_MAKER),
                "trainer_sha256": sha(TRAINER),
                "evaluator_sha256": sha(EVALUATOR),
                "qualifier_sha256": sha(QUALIFIER),
                "batch_sha256": sha(BATCH),
                "source_data_tree_sha256": tree_hash(SOURCE_DATA),
                "warmstart_data_tree_sha256": tree_hash(WARMSTART_DATA),
                "base_model_config_sha256": sha(BASE_MODEL / "config.json"),
                "v3_admission_sha256": sha(V3_ADMISSION),
                "v3_viability_sha256": sha(V3_VIABILITY),
                "v4_viability_sha256": sha(V4_VIABILITY),
                "v5_qualification_sha256": sha(V5_QUALIFICATION),
                "sft_seed": 76621,
                "development_seed": 76622,
                "sft_optimizer_steps": 69,
                "orientation_counts": {"0": 2, "1": 2, "2": 2, "3": 2},
                "development_only": True,
                "evaluation_rows_loaded": False,
                "final_seed": False,
                "shared_checkpoint_for_both_online_arms": True,
            },
        )
        record = run(["scontrol", "show", "job", str(job_id), "-o"])
        for required in (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "gres/gpu:a5000:1",
            "MinMemoryNode=64G",
            "TimeLimit=03:00:00",
        ):
            if required not in record:
                raise RuntimeError(f"held PointMaze v6 job missing {required}")
        atomic(
            SUBMISSION,
            {
                "schema_version": "point-maze-balanced-short-warmstart-v6-submission-v1",
                "job_id": job_id,
                "held_job_record": record,
                "identity_sha256": sha(IDENTITY),
            },
        )
        run(["scontrol", "update", f"JobId={job_id}", "Partition=all"])
        run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[point-balanced-short-warmstart-v6] released job {job_id}")


if __name__ == "__main__":
    main()
