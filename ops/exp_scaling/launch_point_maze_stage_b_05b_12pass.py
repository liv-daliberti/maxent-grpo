#!/usr/bin/env python3
"""Configure or conditionally launch ten frozen PointMaze Stage-B cells."""

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
PROTOCOL = ROOT / "paper/preregistration/point_maze_stage_b_05b_12pass_v3_20260730.md"
TRAINER = ROOT / "ops/train_point_maze_stage_b_05b_12pass.py"
BASE_TRAINER = ROOT / "ops/train_point_maze_interactive_paired_smoke_v1.py"
AUDITOR = ROOT / "ops/audit_point_maze_stage_b_05b_12pass.py"
TRAIN_BATCH = ROOT / "ops/slurm/train_point_maze_stage_b_05b_12pass.slurm"
AUDIT_BATCH = ROOT / "ops/slurm/audit_point_maze_stage_b_05b_12pass.slurm"
QUALIFICATION = ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v3_audit.json"
SMOKE_IDENTITY = ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v3_identity.json"
MODEL = ROOT / "var/models/point_maze_interactive_warmstart_v3"
DATA = ROOT / "var/data/point_maze_modebench_v1"
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
IDENTITY = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_identity.json"
MANIFEST = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_jobs.tsv"
SUBMISSION = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_submission.json"
AUDIT_OUTPUT = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit.json"
AUDIT_RUNNER = ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit_runner_identity.json"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEEDS = (43, 44, 45, 46, 47)
VARIANT = "standard"
ARTIFACT_STEM = "point_maze_stage_b_05b_12pass"


def configure_variant(variant: str) -> None:
    global VARIANT, ARTIFACT_STEM, PROTOCOL, QUALIFICATION, SMOKE_IDENTITY, DATA
    global IDENTITY, MANIFEST, SUBMISSION, AUDIT_OUTPUT, AUDIT_RUNNER
    if variant not in {"standard", "geometry_shift"}:
        raise ValueError("unknown PointMaze Stage-B variant")
    VARIANT = variant
    if variant == "geometry_shift":
        ARTIFACT_STEM = "point_maze_geometry_shift_stage_b_05b_12pass"
        PROTOCOL = ROOT / (
            "paper/preregistration/"
            "point_maze_geometry_shift_stage_b_05b_12pass_v1_20260730.md"
        )
        QUALIFICATION = ROOT / "var/artifacts/point_maze_geometry_shift_paired_smoke_v1_audit.json"
        SMOKE_IDENTITY = ROOT / "var/artifacts/point_maze_geometry_shift_paired_smoke_v1_identity.json"
        DATA = ROOT / "var/data/point_maze_geometry_shift_v1"
        stem = ROOT / f"var/artifacts/{ARTIFACT_STEM}"
        IDENTITY = Path(str(stem) + "_identity.json")
        MANIFEST = Path(str(stem) + "_jobs.tsv")
        SUBMISSION = Path(str(stem) + "_submission.json")
        AUDIT_OUTPUT = Path(str(stem) + "_audit.json")
        AUDIT_RUNNER = Path(str(stem) + "_audit_runner_identity.json")


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
        list(command), cwd=ROOT, env=env, check=True, capture_output=True, text=True,
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
        raise RuntimeError(f"{prefix} snapshot hash mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    inputs = (TRAINER, BASE_TRAINER, AUDITOR, TRAIN_BATCH, AUDIT_BATCH, PROTOCOL)
    temporary = Path(tempfile.mkdtemp(
        prefix=".point-stage-b-05b-12pass.", dir=ROOT / "var/artifacts/source_snapshots"
    ))
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = tree_hash(temporary)
    target = ROOT / f"var/artifacts/source_snapshots/point_{VARIANT}_stage_b_05b_12pass_ops_{digest}"
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if tree_hash(target) != digest:
        raise RuntimeError("PointMaze Stage-B execution snapshot mismatch")
    return target, digest


def base_prerequisites() -> None:
    for path in (
        PYTHON, PROTOCOL, TRAINER, BASE_TRAINER, AUDITOR, TRAIN_BATCH, AUDIT_BATCH,
        SMOKE_IDENTITY, MODEL / "config.json", DATA / "identity.json",
        DATA / "train/dataset_dict.json", DATA / "eval/dataset_dict.json", WORKER,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    smoke_identity = json.loads(SMOKE_IDENTITY.read_text())
    expected_smoke = (
        ("point-maze-interactive-paired-smoke-identity-v4", 75304, [0, 2, 4, 6])
        if VARIANT == "geometry_shift"
        else ("point-maze-interactive-paired-smoke-identity-v3", 75303, [1, 3, 5, 7])
    )
    if (
        smoke_identity.get("schema") != expected_smoke[0]
        or smoke_identity.get("seed") != expected_smoke[1]
        or smoke_identity.get("row_indices") != expected_smoke[2]
        or smoke_identity.get("policy_microbatch_size") != 16
        or set(smoke_identity.get("jobs", {})) != set(ARMS)
        or not all(isinstance(value, int) for value in smoke_identity.get("jobs", {}).values())
        or smoke_identity.get("development_only") is not True
    ):
        raise RuntimeError("PointMaze paired-smoke identity drift")


def qualification_passes() -> None:
    if not QUALIFICATION.is_file():
        raise RuntimeError("PointMaze paired smoke has not produced its audit")
    payload = json.loads(QUALIFICATION.read_text())
    expected = (
        (
            "point-maze-interactive-paired-smoke-audit-v4",
            "eligible_for_ten_point_maze_geometry_shift_replacement_jobs",
        )
        if VARIANT == "geometry_shift"
        else (
            "point-maze-interactive-paired-smoke-audit-v3",
            "eligible_for_ten_point_maze_stage_b_jobs",
        )
    )
    if (
        payload.get("status") != "pass"
        or payload.get("schema") != expected[0]
        or payload.get("decision") != expected[1]
        or payload.get("errors") not in ([], None)
    ):
        raise RuntimeError("PointMaze paired smoke did not authorize Stage B")


def validate() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    environment["LD_LIBRARY_PATH"] = library + (
        ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else ""
    )
    run([
        str(PYTHON), "-m", "py_compile", str(TRAINER), str(BASE_TRAINER),
        str(AUDITOR), str(Path(__file__).resolve()),
    ], env=environment)
    run(["bash", "-n", str(TRAIN_BATCH)])
    run(["bash", "-n", str(AUDIT_BATCH)])
    run([
        str(PYTHON), "-m", "pytest", "-q",
        str(ROOT / "tests/test_point_maze_stage_b_05b_12pass.py"),
        str(ROOT / "tests/test_point_maze_paired_smoke.py"),
        str(ROOT / "tests/test_interactive_episode_objective.py"),
        str(ROOT / "tests/test_interactive_episode_replay.py"),
        str(ROOT / "tests/test_point_maze_interactive_policy.py"),
        str(ROOT / "tests/test_point_maze_interactive_worker.py"),
    ], env=environment)


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
    parser.add_argument("--variant", choices=("standard", "geometry_shift"), default="standard")
    args = parser.parse_args()
    configure_variant(args.variant)
    base_prerequisites()
    validate()
    if args.phase == "config":
        print("[point-stage-b] configuration passed; no job submitted")
        return
    qualification_passes()
    fresh = [IDENTITY, MANIFEST, SUBMISSION, AUDIT_OUTPUT, AUDIT_RUNNER]
    for arm in ARMS:
        for seed in SEEDS:
            fresh.extend(cell_paths(arm, seed).values())
    for path in fresh:
        if path.exists():
            raise FileExistsError(f"fresh PointMaze Stage-B artifact required: {path}")

    source_root, source_hash = snapshot_tree(
        ROOT / "src", f"point_{VARIANT}_stage_b_05b_12pass_source"
    )
    execution_root, execution_hash = snapshot_execution()
    jobs: dict[str, int] = {}
    audit_job: int | None = None
    try:
        for seed in SEEDS:
            for arm in ARMS:
                label = f"{arm}/s{seed}"
                short = "grpo" if arm == CONTROL else "maxent"
                output = run([
                    "sbatch", "--parsable", "--hold",
                    f"--job-name=point-{VARIANT}-stage-b-{short}-s{seed}",
                    "--partition=all", "--account=allcs",
                    "--export=ALL,"
                    f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
                    f"OAT_ZERO_EXECUTION_ROOT={execution_root},OAT_ZERO_SOURCE_HASH={source_hash},"
                    f"OAT_ZERO_EXECUTION_HASH={execution_hash},OAT_ZERO_ARM={arm},OAT_ZERO_SEED={seed},"
                    f"OAT_ZERO_POINT_STAGE_B_VARIANT={VARIANT}",
                    str(execution_root / TRAIN_BATCH.name),
                ])
                job_id = int(output.split(";", 1)[0])
                jobs[label] = job_id
                run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
                record = run(["scontrol", "show", "job", "-o", str(job_id)])
                for required in (
                    "JobState=PENDING", "Reason=JobHeldUser", "Account=allcs", "gres/gpu:a5000:1",
                    "NumCPUs=8", "MinMemoryNode=64G", "TimeLimit=3-00:00:00", "Requeue=0",
                    f"OAT_ZERO_ARM={arm}", f"OAT_ZERO_SEED={seed}",
                    f"OAT_ZERO_SOURCE_HASH={source_hash}", f"OAT_ZERO_EXECUTION_HASH={execution_hash}",
                ):
                    if required not in record:
                        raise RuntimeError(f"held PointMaze {label} job lacks {required}")
        with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("arm", "seed", "job_id", "receipt", "metrics", "state_replay"),
                delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            for seed in SEEDS:
                for arm in ARMS:
                    label = f"{arm}/s{seed}"
                    paths = cell_paths(arm, seed)
                    writer.writerow({
                        "arm": arm, "seed": seed, "job_id": jobs[label],
                        **{name: path.relative_to(ROOT).as_posix() for name, path in paths.items()},
                    })
        dependency = ",".join(
            f"afterany:{jobs[f'{arm}/s{seed}']}" for seed in SEEDS for arm in ARMS
        )
        audit_output = run([
            "sbatch", "--parsable", f"--dependency={dependency}",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},OAT_ZERO_EXECUTION_ROOT={execution_root},"
            f"OAT_ZERO_POINT_STAGE_B_VARIANT={VARIANT}",
            str(execution_root / AUDIT_BATCH.name),
        ])
        audit_job = int(audit_output.split(";", 1)[0])
        identity = {
            "schema": (
                "point-maze-geometry-shift-stage-b-05b-12pass-identity-v2"
                if VARIANT == "geometry_shift"
                else "point-maze-stage-b-05b-12pass-identity-v1"
            ),
            "protocol_sha256": sha(PROTOCOL), "launcher_sha256": sha(Path(__file__).resolve()),
            "trainer_sha256": sha(TRAINER), "base_trainer_sha256": sha(BASE_TRAINER),
            "auditor_sha256": sha(AUDITOR), "train_batch_sha256": sha(TRAIN_BATCH),
            "audit_batch_sha256": sha(AUDIT_BATCH), "source_root": str(source_root),
            "source_hash": source_hash, "execution_root": str(execution_root),
            "execution_hash": execution_hash, "model_tree_sha256": tree_hash(MODEL),
            "model_config_sha256": sha(MODEL / "config.json"),
            "data_tree_sha256": tree_hash(DATA), "data_identity_sha256": sha(DATA / "identity.json"),
            "worker_python_sha256": sha(WORKER), "qualification_audit_sha256": sha(QUALIFICATION),
            "paired_smoke_identity_sha256": sha(SMOKE_IDENTITY), "manifest_sha256": sha(MANIFEST),
            "arms": list(ARMS), "seeds": list(SEEDS), "jobs": jobs,
            "audit_job_id": audit_job, "audit_dependency": dependency,
            "prompt_passes": 12, "optimizer_updates": 96, "train_prompt_count": 8,
            "policy_microbatch_size": 16,
            "rollouts_per_prompt": 16, "decision_horizon": 96,
            "evaluation_rounds": list(range(0, 97, 2)), "evaluation_draws": 4,
            "evaluation_k": 8, "evaluation_trajectories_per_coordinate": 132,
            "final_seed_cohort": True, "resume": False, "development_rows_loaded": False,
            "replacement_configuration": VARIANT == "geometry_shift",
            "independent_semantic_domain": False if VARIANT == "geometry_shift" else None,
        }
        atomic(IDENTITY, identity)
        atomic(SUBMISSION, {
            "schema": (
                "point-maze-geometry-shift-stage-b-05b-12pass-submission-v2"
                if VARIANT == "geometry_shift"
                else "point-maze-stage-b-05b-12pass-submission-v1"
            ),
            "identity_sha256": sha(IDENTITY), "manifest_sha256": sha(MANIFEST),
            "jobs": jobs, "held_job_audit": "pass", "released": True,
        })
        atomic(AUDIT_RUNNER, {
            "schema": "point-maze-stage-b-05b-12pass-audit-runner-v1",
            "audit_job_id": audit_job, "dependency": dependency,
            "identity_sha256": sha(IDENTITY), "execution_hash": execution_hash,
        })
        for job_id in jobs.values():
            run(["scontrol", "release", str(job_id)])
    except BaseException:
        for job_id in [*jobs.values(), *([] if audit_job is None else [audit_job])]:
            subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[point-stage-b] released jobs {jobs}; audit job {audit_job}")


if __name__ == "__main__":
    main()
