#!/usr/bin/env python3
"""Configure or launch the ten-cell PointMaze algorithm-repair final."""

from __future__ import annotations

import argparse
import csv
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
    "point_maze_algorithm_repair_final_v1_20260730.md"
)
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "point_maze_algorithm_repair_final_v1_r1_20260730.md"
)
TRAINER = ROOT / "ops/train_point_maze_algorithm_repair_final_v1.py"
QUALIFIED_TRAINER = (
    ROOT / "ops/train_point_maze_algorithm_repair_v2_direct_r5.py"
)
DIRECT_TRAINER = ROOT / "ops/train_point_maze_algorithm_repair_v2_direct.py"
STAGE_TRAINER = ROOT / "ops/train_point_maze_stage_b_05b_12pass.py"
BASE_TRAINER = ROOT / "ops/train_point_maze_interactive_paired_smoke_v1.py"
AUDITOR = ROOT / "ops/audit_point_maze_algorithm_repair_final_v1.py"
BASE_AUDITOR = ROOT / "ops/audit_point_maze_stage_b_05b_12pass.py"
TRAIN_BATCH = (
    ROOT
    / "ops/slurm/train_point_maze_algorithm_repair_final_v1_r1.slurm"
)
AUDIT_BATCH = (
    ROOT / "ops/slurm/audit_point_maze_algorithm_repair_final_v1.slurm"
)
PAIR_AUDIT = (
    ROOT
    / "var/artifacts/point_maze_algorithm_repair_pair_v2r5_audit.json"
)
PAIR_IDENTITY = (
    ROOT
    / "var/artifacts/point_maze_algorithm_repair_pair_v2r5_identity.json"
)
K16_QUALIFICATION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v2_qualification.json"
)
MODEL = ROOT / "var/models/point_maze_interactive_warmstart_v3"
DATA = ROOT / "var/data/point_maze_algorithm_repair_v2"
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
IDENTITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_final_v1_identity.json"
)
MANIFEST = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_final_v1_jobs.tsv"
)
SUBMISSION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_final_v1_submission.json"
)
AUDIT_OUTPUT = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_final_v1_audit.json"
)
AUDIT_RUNNER = (
    ROOT
    / "var/artifacts/"
    "point_maze_algorithm_repair_final_v1_audit_runner_identity.json"
)
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEEDS = (76531, 76532, 76533, 76534, 76535)
ARTIFACT_STEM = "point_maze_algorithm_repair_final_v1"


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
        prefix=f".{path.name}.", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
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
        raise RuntimeError("PointMaze repair-final source snapshot mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    inputs = (
        TRAINER,
        QUALIFIED_TRAINER,
        DIRECT_TRAINER,
        STAGE_TRAINER,
        BASE_TRAINER,
        AUDITOR,
        BASE_AUDITOR,
        TRAIN_BATCH,
        AUDIT_BATCH,
        PROTOCOL,
        AMENDMENT,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".point-algorithm-repair-final-v1.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = tree_hash(temporary)
    target = (
        ROOT
        / "var/artifacts/source_snapshots/"
        f"point_algorithm_repair_final_v1_ops_{digest}"
    )
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze repair-final execution snapshot mismatch")
    return target, digest


def validate() -> None:
    for path in (
        PYTHON,
        PROTOCOL,
        AMENDMENT,
        TRAINER,
        QUALIFIED_TRAINER,
        DIRECT_TRAINER,
        STAGE_TRAINER,
        BASE_TRAINER,
        AUDITOR,
        BASE_AUDITOR,
        TRAIN_BATCH,
        AUDIT_BATCH,
        PAIR_IDENTITY,
        K16_QUALIFICATION,
        MODEL / "config.json",
        DATA / "identity.json",
        DATA / "train/dataset_dict.json",
        DATA / "dev/dataset_dict.json",
        DATA / "eval/dataset_dict.json",
        WORKER,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    pair_identity = json.loads(PAIR_IDENTITY.read_text(encoding="utf-8"))
    if (
        pair_identity.get("repair_r5_schema")
        != "point-maze-algorithm-repair-pair-identity-v2r5"
        or pair_identity.get("development_only") is not True
        or pair_identity.get("final_seed_cohort") is not False
        or pair_identity.get("evaluation_split") != "development"
        or pair_identity.get("jobs")
        != {
            "grpo/s76521": 30204902,
            "verified_first_global_replay_canonical/s76521": 30204903,
        }
    ):
        raise RuntimeError("PointMaze repair pair identity drift")
    k16 = json.loads(K16_QUALIFICATION.read_text(encoding="utf-8"))
    if (
        k16.get("status") != "pass"
        or k16.get("decision")
        != "eligible_for_point_maze_algorithm_repair_v2_pair"
    ):
        raise RuntimeError("PointMaze K=16 qualification drift")
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
            str(TRAINER),
            str(QUALIFIED_TRAINER),
            str(DIRECT_TRAINER),
            str(STAGE_TRAINER),
            str(BASE_TRAINER),
            str(AUDITOR),
            str(BASE_AUDITOR),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    run(["bash", "-n", str(TRAIN_BATCH)])
    run(["bash", "-n", str(AUDIT_BATCH)])
    run(
        [
            str(PYTHON),
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


def qualification_passes() -> None:
    if not PAIR_AUDIT.is_file():
        raise RuntimeError("PointMaze repair pair audit is not terminal")
    payload = json.loads(PAIR_AUDIT.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "point-maze-algorithm-repair-pair-audit-v2"
        or payload.get("status") != "pass"
        or payload.get("decision")
        != "eligible_for_point_maze_algorithm_repair_v2_five_seed_final"
        or payload.get("errors") not in ([], None)
    ):
        raise RuntimeError("PointMaze repair pair did not authorize final")


def cell_paths(arm: str, seed: int) -> dict[str, Path]:
    stem = ROOT / f"var/artifacts/{ARTIFACT_STEM}_{arm}_s{seed}"
    return {
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "state_replay": Path(str(stem) + ".state_replay.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    validate()
    if args.phase == "config":
        print("[point-repair-final] configuration passed; no jobs submitted")
        return
    qualification_passes()
    fresh = [
        IDENTITY,
        MANIFEST,
        SUBMISSION,
        AUDIT_OUTPUT,
        AUDIT_RUNNER,
    ]
    for arm in ARMS:
        for seed in SEEDS:
            fresh.extend(cell_paths(arm, seed).values())
    for path in fresh:
        if path.exists():
            raise FileExistsError(
                f"fresh PointMaze repair-final artifact required: {path}"
            )

    source_root, source_hash = snapshot_tree(
        ROOT / "src", "point_algorithm_repair_final_v1_source"
    )
    execution_root, execution_hash = snapshot_execution()
    jobs: dict[str, int] = {}
    audit_job: int | None = None
    held_records: dict[str, str] = {}
    try:
        for seed in SEEDS:
            for arm in ARMS:
                label = f"{arm}/s{seed}"
                short = "grpo" if arm == CONTROL else "maxent"
                output = run(
                    [
                        "sbatch",
                        "--parsable",
                        "--hold",
                        f"--job-name=point-repair-final-{short}-s{seed}",
                        "--partition=all",
                        "--account=allcs",
                        "--export=ALL,"
                        f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
                        f"OAT_ZERO_EXECUTION_ROOT={execution_root},"
                        f"OAT_ZERO_SOURCE_HASH={source_hash},"
                        f"OAT_ZERO_EXECUTION_HASH={execution_hash},"
                        f"OAT_ZERO_ARM={arm},OAT_ZERO_SEED={seed}",
                        str(execution_root / TRAIN_BATCH.name),
                    ]
                )
                job_id = int(output.split(";", 1)[0])
                jobs[label] = job_id
                run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
                record = run(["scontrol", "show", "job", "-o", str(job_id)])
                held_records[label] = record
                for required in (
                    "JobState=PENDING",
                    "Reason=JobHeldUser",
                    "Account=allcs",
                    "gres/gpu:a5000:1",
                    "NumCPUs=8",
                    "MinMemoryNode=64G",
                    "TimeLimit=3-00:00:00",
                    "Requeue=0",
                    f"OAT_ZERO_ARM={arm}",
                    f"OAT_ZERO_SEED={seed}",
                    f"OAT_ZERO_SOURCE_HASH={source_hash}",
                    f"OAT_ZERO_EXECUTION_HASH={execution_hash}",
                ):
                    if required not in record:
                        raise RuntimeError(
                            f"held PointMaze repair-final {label} lacks {required}"
                        )
        with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "arm",
                    "seed",
                    "job_id",
                    "receipt",
                    "metrics",
                    "state_replay",
                ),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for seed in SEEDS:
                for arm in ARMS:
                    label = f"{arm}/s{seed}"
                    paths = cell_paths(arm, seed)
                    writer.writerow(
                        {
                            "arm": arm,
                            "seed": seed,
                            "job_id": jobs[label],
                            **{
                                name: path.relative_to(ROOT).as_posix()
                                for name, path in paths.items()
                            },
                        }
                    )
        dependency = ",".join(
            f"afterany:{jobs[f'{arm}/s{seed}']}"
            for seed in SEEDS
            for arm in ARMS
        )
        audit_output = run(
            [
                "sbatch",
                "--parsable",
                f"--dependency={dependency}",
                "--export=ALL,"
                f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
                f"OAT_ZERO_EXECUTION_ROOT={execution_root}",
                str(execution_root / AUDIT_BATCH.name),
            ]
        )
        audit_job = int(audit_output.split(";", 1)[0])
        identity = {
            "schema": "point-maze-algorithm-repair-final-identity-v1",
            "protocol_sha256": sha(PROTOCOL),
            "amendment_sha256": sha(AMENDMENT),
            "launcher_sha256": sha(Path(__file__).resolve()),
            "trainer_sha256": sha(TRAINER),
            "qualified_trainer_sha256": sha(QUALIFIED_TRAINER),
            "direct_trainer_sha256": sha(DIRECT_TRAINER),
            "stage_trainer_sha256": sha(STAGE_TRAINER),
            "base_trainer_sha256": sha(BASE_TRAINER),
            "auditor_sha256": sha(AUDITOR),
            "base_auditor_sha256": sha(BASE_AUDITOR),
            "train_batch_sha256": sha(TRAIN_BATCH),
            "audit_batch_sha256": sha(AUDIT_BATCH),
            "source_root": str(source_root),
            "source_hash": source_hash,
            "execution_root": str(execution_root),
            "execution_hash": execution_hash,
            "model_tree_sha256": tree_hash(MODEL),
            "model_config_sha256": sha(MODEL / "config.json"),
            "data_tree_sha256": tree_hash(DATA),
            "data_identity_sha256": sha(DATA / "identity.json"),
            "worker_python_sha256": sha(WORKER),
            "qualification_audit_sha256": sha(PAIR_AUDIT),
            "k16_qualification_sha256": sha(K16_QUALIFICATION),
            "pair_identity_sha256": sha(PAIR_IDENTITY),
            "manifest_sha256": sha(MANIFEST),
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "jobs": jobs,
            "held_scheduler_records": held_records,
            "audit_job_id": audit_job,
            "audit_dependency": dependency,
            "prompt_passes": 12,
            "optimizer_updates": 96,
            "train_prompt_count": 8,
            "policy_microbatch_size": 16,
            "rollouts_per_prompt": 16,
            "decision_horizon": 96,
            "evaluation_rounds": list(range(0, 97, 2)),
            "evaluation_draws": 4,
            "evaluation_k": 8,
            "evaluation_trajectories_per_coordinate": 132,
            "final_seed_cohort": True,
            "development_only": False,
            "evaluation_common_random_numbers": True,
            "evaluation_split": "previously_untouched_eval",
            "development_rows_loaded": False,
            "secondary_post_outcome_repair": True,
            "resume": False,
        }
        atomic(IDENTITY, identity)
        atomic(
            SUBMISSION,
            {
                "schema": (
                    "point-maze-algorithm-repair-final-submission-v1"
                ),
                "identity_sha256": sha(IDENTITY),
                "manifest_sha256": sha(MANIFEST),
                "jobs": jobs,
                "held_job_audit": "pass",
                "released": True,
            },
        )
        atomic(
            AUDIT_RUNNER,
            {
                "schema": (
                    "point-maze-algorithm-repair-final-audit-runner-v1"
                ),
                "audit_job_id": audit_job,
                "dependency": dependency,
                "identity_sha256": sha(IDENTITY),
                "execution_hash": execution_hash,
            },
        )
        for job_id in jobs.values():
            run(["scontrol", "release", str(job_id)])
    except BaseException:
        for job_id in [
            *jobs.values(),
            *([] if audit_job is None else [audit_job]),
        ]:
            subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(
        f"[point-repair-final] released jobs {jobs}; audit job {audit_job}"
    )


if __name__ == "__main__":
    main()
