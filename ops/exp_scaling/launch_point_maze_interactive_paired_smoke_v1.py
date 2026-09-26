#!/usr/bin/env python3
"""Configure or launch the frozen PointMaze interactive paired smoke."""

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
PROTOCOL = ROOT / "paper/preregistration/point_maze_interactive_paired_smoke_v1_20260730.md"
TRAINER = ROOT / "ops/train_point_maze_interactive_paired_smoke_v1.py"
AUDITOR = ROOT / "ops/audit_point_maze_interactive_paired_smoke_v1.py"
TRAIN_SLURM = ROOT / "ops/slurm/train_point_maze_interactive_paired_smoke_v1.slurm"
AUDIT_SLURM = ROOT / "ops/slurm/audit_point_maze_interactive_paired_smoke_v1.slurm"
VIABILITY = ROOT / "var/artifacts/point_maze_interactive_05b_viability_warmstart_v3.json"
ADMISSION = ROOT / "var/artifacts/point_maze_modebench_v1_admission_audit.json"
WARMSTART_RECEIPT = ROOT / "var/artifacts/point_maze_interactive_warmstart_sft_v3.json"
MODEL = ROOT / "var/models/point_maze_interactive_warmstart_v3"
DATA = ROOT / "var/data/point_maze_modebench_v1"
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
IDENTITY = ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v1_identity.json"
MANIFEST = ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v1_jobs.tsv"
SUBMISSION = ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v1_submission.json"
AUDIT_OUTPUT = ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v1_audit.json"
REPAIR = ROOT / "paper/preregistration/maze_microbatch16_sampling_geometry_repair_20260730.md"
FAILED_V2_AUDIT = ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v2_audit.json"
V2_DIAGNOSTIC = ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v2_logprob_batch_diagnostic.json"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEED = 75301
ROW_INDICES = (0, 2, 4, 6)
COHORT = "v1"
ARTIFACT_STEM = "point_maze_interactive_paired_smoke_v1"


def configure_cohort(cohort: str) -> None:
    global PROTOCOL, IDENTITY, MANIFEST, SUBMISSION, AUDIT_OUTPUT
    global SEED, ROW_INDICES, COHORT, ARTIFACT_STEM, VIABILITY, ADMISSION, DATA
    if cohort not in ("v1", "v2", "v3", "v4"):
        raise ValueError("unknown PointMaze paired-smoke cohort")
    COHORT = cohort
    version = int(cohort[1:])
    if cohort == "v4":
        PROTOCOL = ROOT / (
            "paper/preregistration/"
            "point_maze_geometry_shift_paired_smoke_v1_20260730.md"
        )
        stem = ROOT / "var/artifacts/point_maze_geometry_shift_paired_smoke_v1"
        VIABILITY = ROOT / "var/artifacts/point_maze_geometry_shift_05b_viability_v1.json"
        ADMISSION = ROOT / "var/artifacts/point_maze_geometry_shift_v1_admission_audit.json"
        DATA = ROOT / "var/data/point_maze_geometry_shift_v1"
    else:
        PROTOCOL = ROOT / (
            "paper/preregistration/"
            f"point_maze_interactive_paired_smoke_{cohort}_20260730.md"
        )
        stem = ROOT / f"var/artifacts/point_maze_interactive_paired_smoke_{cohort}"
    IDENTITY = Path(str(stem) + "_identity.json")
    MANIFEST = Path(str(stem) + "_jobs.tsv")
    SUBMISSION = Path(str(stem) + "_submission.json")
    AUDIT_OUTPUT = Path(str(stem) + "_audit.json")
    ARTIFACT_STEM = stem.name
    SEED = 75300 + version
    ROW_INDICES = (0, 2, 4, 6) if cohort in {"v1", "v4"} else (1, 3, 5, 7)


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


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


def snapshot_source() -> tuple[Path, str]:
    source = ROOT / "src"
    digest = tree_hash(source)
    parent = ROOT / f"var/artifacts/source_snapshots/point_paired_smoke_{COHORT}_{digest}"
    target = parent / "src"
    if not target.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".source.", dir=parent))
        shutil.copytree(source, staging / "src")
        os.replace(staging / "src", target)
        staging.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze source snapshot hash mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    inputs = {
        TRAINER.name: TRAINER,
        AUDITOR.name: AUDITOR,
        TRAIN_SLURM.name: TRAIN_SLURM,
        AUDIT_SLURM.name: AUDIT_SLURM,
        PROTOCOL.name: PROTOCOL,
    }
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".point-paired-smoke-{COHORT}.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for name, source in inputs.items():
        shutil.copy2(source, temporary / name)
    digest = tree_hash(temporary)
    target = ROOT / f"var/artifacts/source_snapshots/point_paired_smoke_{COHORT}_ops_{digest}"
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze execution snapshot hash mismatch")
    return target, digest


def prerequisites() -> None:
    for path in (
        PYTHON,
        PROTOCOL,
        TRAINER,
        AUDITOR,
        TRAIN_SLURM,
        AUDIT_SLURM,
        VIABILITY,
        ADMISSION,
        WARMSTART_RECEIPT,
        MODEL / "config.json",
        DATA / "identity.json",
        DATA / "train/dataset_dict.json",
        WORKER,
        REPAIR,
        FAILED_V2_AUDIT,
        V2_DIAGNOSTIC,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    viability = json.loads(VIABILITY.read_text())
    admission = json.loads(ADMISSION.read_text())
    warmstart = json.loads(WARMSTART_RECEIPT.read_text())
    expected_viability = (
        {
            "maximum_decision_rounds": 96,
            "multimode_prompts": 2,
            "prefix_success_prompts": 2,
            "prompt_count": 4,
            "verified_completions": 10,
        }
        if COHORT == "v4"
        else {
            "maximum_decision_rounds": 96,
            "multimode_prompts": 3,
            "prefix_success_prompts": 3,
            "prompt_count": 4,
            "verified_completions": 32,
        }
    )
    if (
        viability.get("status") != "pass"
        or viability.get("summary") != expected_viability
        or admission.get("status") != "pass"
        or warmstart.get("status") != "pass"
        or warmstart.get("optimizer_steps") != 276
    ):
        raise RuntimeError("PointMaze paired-smoke antecedent drift")


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
            str(TRAINER),
            str(AUDITOR),
            str(Path(__file__).resolve()),
        ],
        env=environment,
    )
    run(["bash", "-n", str(TRAIN_SLURM)])
    run(["bash", "-n", str(AUDIT_SLURM)])
    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_point_maze_paired_smoke.py"),
            str(ROOT / "tests/test_interactive_episode_objective.py"),
            str(ROOT / "tests/test_interactive_episode_replay.py"),
            str(ROOT / "tests/test_point_maze_interactive_policy.py"),
            str(ROOT / "tests/test_point_maze_interactive_worker.py"),
        ],
        env=environment,
    )


def arm_paths(arm: str) -> dict[str, Path]:
    stem = ROOT / f"var/artifacts/{ARTIFACT_STEM}_{arm}"
    return {
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "state_replay": Path(str(stem) + ".state_replay.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    parser.add_argument("--cohort", choices=("v1", "v2", "v3", "v4"), default="v1")
    args = parser.parse_args()
    configure_cohort(args.cohort)
    prerequisites()
    validate()
    if args.phase == "config":
        print("[point-paired-smoke] configuration passed; no job submitted")
        return

    fresh = [IDENTITY, MANIFEST, SUBMISSION, AUDIT_OUTPUT]
    for arm in ARMS:
        fresh.extend(arm_paths(arm).values())
    for path in fresh:
        if path.exists():
            raise FileExistsError(f"fresh PointMaze paired artifact required: {path}")

    source_root, source_hash = snapshot_source()
    execution_root, execution_hash = snapshot_execution()
    model_hash = tree_hash(MODEL)
    jobs: dict[str, int] = {}
    audit_job: int | None = None
    try:
        for arm in ARMS:
            output = run(
                [
                    "sbatch",
                    "--parsable",
                    "--hold",
                    f"--job-name=point-smoke-{COHORT}-{arm}",
                    "--partition=all",
                    "--account=allcs",
                    "--export=ALL,"
                    f"ROOT_DIR={ROOT},"
                    f"OAT_ZERO_SOURCE_ROOT={source_root},"
                    f"OAT_ZERO_EXECUTION_ROOT={execution_root},"
                    f"OAT_ZERO_SOURCE_HASH={source_hash},"
                    f"OAT_ZERO_EXECUTION_HASH={execution_hash},"
                    f"OAT_ZERO_ARM={arm},"
                    f"OAT_ZERO_POINT_SMOKE_COHORT={COHORT},"
                    f"OAT_ZERO_POINT_SMOKE_SEED={SEED}",
                    str(execution_root / TRAIN_SLURM.name),
                ]
            )
            job_id = int(output.split(";", 1)[0])
            jobs[arm] = job_id
            run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
            record = run(["scontrol", "show", "job", "-o", str(job_id)])
            for required in (
                "JobState=PENDING",
                "Reason=JobHeldUser",
                "Account=allcs",
                "gres/gpu:a5000:1",
                "NumCPUs=8",
                "MinMemoryNode=64G",
                "TimeLimit=12:00:00",
                "Requeue=0",
                f"OAT_ZERO_ARM={arm}",
                f"OAT_ZERO_SOURCE_HASH={source_hash}",
                f"OAT_ZERO_EXECUTION_HASH={execution_hash}",
            ):
                if required not in record:
                    raise RuntimeError(f"held PointMaze {arm} job lacks {required}")

        dependency = "afterany:" + ":".join(str(jobs[arm]) for arm in ARMS)
        audit_output = run(
            [
                "sbatch",
                "--parsable",
                f"--dependency={dependency}",
                f"--job-name=audit-point-paired-smoke-{COHORT}",
                "--export=ALL,"
                f"ROOT_DIR={ROOT},"
                f"OAT_ZERO_SOURCE_ROOT={source_root},"
                f"OAT_ZERO_EXECUTION_ROOT={execution_root},"
                f"OAT_ZERO_POINT_SMOKE_COHORT={COHORT}",
                str(execution_root / AUDIT_SLURM.name),
            ]
        )
        audit_job = int(audit_output.split(";", 1)[0])

        with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("arm", "seed", "job_id", "receipt", "metrics", "state_replay"),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for arm in ARMS:
                paths = arm_paths(arm)
                writer.writerow(
                    {
                        "arm": arm,
                        "seed": SEED,
                        "job_id": jobs[arm],
                        **{name: path.relative_to(ROOT).as_posix() for name, path in paths.items()},
                    }
                )
        identity = {
            "schema": f"point-maze-interactive-paired-smoke-identity-{COHORT}",
            "protocol_sha256": sha(PROTOCOL),
            "launcher_sha256": sha(Path(__file__).resolve()),
            "trainer_sha256": sha(TRAINER),
            "auditor_sha256": sha(AUDITOR),
            "train_slurm_sha256": sha(TRAIN_SLURM),
            "audit_slurm_sha256": sha(AUDIT_SLURM),
            "source_root": str(source_root),
            "source_hash": source_hash,
            "execution_root": str(execution_root),
            "execution_hash": execution_hash,
            "model_tree_sha256": model_hash,
            "model_config_sha256": sha(MODEL / "config.json"),
            "data_identity_sha256": sha(DATA / "identity.json"),
            "viability_receipt_sha256": sha(VIABILITY),
            "admission_audit_sha256": sha(ADMISSION),
            "warmstart_receipt_sha256": sha(WARMSTART_RECEIPT),
            "manifest_sha256": sha(MANIFEST),
            "arms": list(ARMS),
            "jobs": jobs,
            "audit_job_id": audit_job,
            "audit_dependency": dependency,
            "seed": SEED,
            "row_indices": list(ROW_INDICES),
            "policy_microbatch_size": 16 if COHORT in {"v3", "v4"} else 4,
            "sampling_geometry_repair_sha256": sha(REPAIR),
            "failed_v2_audit_sha256": sha(FAILED_V2_AUDIT) if COHORT == "v3" else None,
            "v2_batch_diagnostic_sha256": sha(V2_DIAGNOSTIC) if COHORT == "v3" else None,
            "failed_predecessor_jobs": [30202913, 30202914, 30202915] if COHORT == "v3" else [],
            "replacement_configuration": COHORT == "v4",
            "independent_semantic_domain": False if COHORT == "v4" else None,
            "rollouts_per_prompt": 16,
            "decision_horizon": 96,
            "optimizer_updates_per_arm": 4,
            "fixed_policy_slots_per_arm": 6144,
            "fixed_replay_mode_slots_per_arm": 64,
            "fixed_replay_decision_slots_per_arm": 6144,
            "development_only": True,
            "final_seed": False,
            "evaluation_rows_loaded": False,
        }
        atomic_json(IDENTITY, identity)
        for arm in ARMS:
            run(["scontrol", "release", str(jobs[arm])])
        atomic_json(
            SUBMISSION,
            {
                "schema": f"point-maze-interactive-paired-smoke-submission-{COHORT}",
                "identity_sha256": sha(IDENTITY),
                "manifest_sha256": sha(MANIFEST),
                "jobs": jobs,
                "audit_job_id": audit_job,
                "held_job_audit": "pass",
                "released": True,
            },
        )
    except BaseException:
        for job_id in [*jobs.values(), *([] if audit_job is None else [audit_job])]:
            subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[point-paired-smoke] released jobs {jobs}; audit job {audit_job}")


if __name__ == "__main__":
    main()
