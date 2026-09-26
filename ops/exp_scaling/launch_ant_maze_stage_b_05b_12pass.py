#!/usr/bin/env python3
"""Configure or conditionally launch ten frozen AntMaze Stage-B cells."""

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
PROTOCOL = ROOT / "paper/preregistration/ant_maze_stage_b_05b_12pass_r2_20260730.md"
TRAINER = ROOT / "ops/train_ant_maze_stage_b_05b_12pass.py"
BASE_TRAINER = ROOT / "ops/train_ant_maze_interactive_paired_smoke_v13.py"
POINT_BASE_TRAINER = ROOT / "ops/train_point_maze_interactive_paired_smoke_v1.py"
AUDITOR = ROOT / "ops/audit_ant_maze_stage_b_05b_12pass.py"
TRAIN_BATCH = ROOT / "ops/slurm/train_ant_maze_stage_b_05b_12pass.slurm"
AUDIT_BATCH = ROOT / "ops/slurm/audit_ant_maze_stage_b_05b_12pass.slurm"
QUALIFICATION = ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13r2_audit.json"
SMOKE_IDENTITY = ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13r2_identity.json"
MODEL = ROOT / "var/models/ant_maze_interactive_warmstart_v13"
DATA = ROOT / "var/data/ant_maze_modebench_v12"
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
IDENTITY = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_identity.json"
MANIFEST = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_jobs.tsv"
SUBMISSION = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_submission.json"
AUDIT_OUTPUT = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_audit.json"
AUDIT_RUNNER = ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_audit_runner_identity.json"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEEDS = (43, 44, 45, 46, 47)


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
    inputs = (
        TRAINER, BASE_TRAINER, POINT_BASE_TRAINER, AUDITOR,
        TRAIN_BATCH, AUDIT_BATCH, PROTOCOL,
    )
    temporary = Path(tempfile.mkdtemp(
        prefix=".ant-stage-b-05b-12pass.", dir=ROOT / "var/artifacts/source_snapshots"
    ))
    for source in inputs:
        shutil.copy2(source, temporary / source.name)
    digest = tree_hash(temporary)
    target = ROOT / f"var/artifacts/source_snapshots/ant_stage_b_05b_12pass_ops_{digest}"
    if not target.is_dir():
        os.replace(temporary, target)
    else:
        shutil.rmtree(temporary)
    if tree_hash(target) != digest:
        raise RuntimeError("AntMaze Stage-B execution snapshot mismatch")
    return target, digest


def base_prerequisites() -> None:
    for path in (
        PYTHON, PROTOCOL, TRAINER, BASE_TRAINER, POINT_BASE_TRAINER,
        AUDITOR, TRAIN_BATCH, AUDIT_BATCH,
        DATA / "identity.json",
        DATA / "train/dataset_dict.json", DATA / "eval/dataset_dict.json", WORKER,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
def qualification_passes() -> None:
    for path in (QUALIFICATION, SMOKE_IDENTITY, MODEL / "config.json"):
        if not path.is_file():
            raise RuntimeError(f"AntMaze paired smoke has not qualified Stage B: {path}")
    payload = json.loads(QUALIFICATION.read_text())
    smoke_identity = json.loads(SMOKE_IDENTITY.read_text())
    if (
        payload.get("status") != "pass"
        or payload.get("decision") != "eligible_for_ten_ant_maze_stage_b_jobs"
        or payload.get("errors") not in ([], None)
        or smoke_identity.get("schema") != "ant-maze-interactive-paired-smoke-identity-v13"
        or smoke_identity.get("artifact_cohort") != "v13r2"
        or smoke_identity.get("policy_microbatch_size") != 16
        or smoke_identity.get("arms") != list(ARMS)
        or smoke_identity.get("seed") != 76313
        or set(smoke_identity.get("jobs", {})) != set(ARMS)
        or not all(isinstance(value, int) for value in smoke_identity.get("jobs", {}).values())
        or smoke_identity.get("development_only") is not True
        or smoke_identity.get("final_seed") is not False
    ):
        raise RuntimeError("AntMaze paired smoke did not authorize Stage B")


def validate() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    environment["LD_LIBRARY_PATH"] = library + (
        ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else ""
    )
    run([
        str(PYTHON), "-m", "py_compile", str(TRAINER), str(BASE_TRAINER),
        str(POINT_BASE_TRAINER),
        str(AUDITOR), str(Path(__file__).resolve()),
    ], env=environment)
    run(["bash", "-n", str(TRAIN_BATCH)])
    run(["bash", "-n", str(AUDIT_BATCH)])
    run([
        str(PYTHON), "-m", "pytest", "-q",
        str(ROOT / "tests/test_ant_maze_stage_b_05b_12pass.py"),
        str(ROOT / "tests/test_ant_maze_paired_smoke_v13.py"),
        str(ROOT / "tests/test_ant_maze_interactive_v13.py"),
        str(ROOT / "tests/test_interactive_episode_objective.py"),
        str(ROOT / "tests/test_interactive_episode_replay.py"),
        str(ROOT / "tests/test_ant_maze_worker_v5.py"),
    ], env=environment)


def cell_paths(arm: str, seed: int) -> dict[str, Path]:
    stem = ROOT / f"var/artifacts/ant_maze_stage_b_05b_12pass_{arm}_s{seed}"
    return {
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "state_replay": Path(str(stem) + ".state_replay.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    base_prerequisites()
    validate()
    if args.phase == "config":
        print("[ant-stage-b] configuration passed; no job submitted")
        return
    qualification_passes()
    fresh = [IDENTITY, MANIFEST, SUBMISSION, AUDIT_OUTPUT, AUDIT_RUNNER]
    for arm in ARMS:
        for seed in SEEDS:
            fresh.extend(cell_paths(arm, seed).values())
    for path in fresh:
        if path.exists():
            raise FileExistsError(f"fresh AntMaze Stage-B artifact required: {path}")

    source_root, source_hash = snapshot_tree(ROOT / "src", "ant_stage_b_05b_12pass_source")
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
                    f"--job-name=ant-stage-b-{short}-s{seed}",
                    "--partition=all", "--account=allcs",
                    "--export=ALL,"
                    f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
                    f"OAT_ZERO_EXECUTION_ROOT={execution_root},OAT_ZERO_SOURCE_HASH={source_hash},"
                    f"OAT_ZERO_EXECUTION_HASH={execution_hash},OAT_ZERO_ARM={arm},OAT_ZERO_SEED={seed}",
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
                        raise RuntimeError(f"held AntMaze {label} job lacks {required}")
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
            f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},OAT_ZERO_EXECUTION_ROOT={execution_root}",
            str(execution_root / AUDIT_BATCH.name),
        ])
        audit_job = int(audit_output.split(";", 1)[0])
        identity = {
            "schema": "ant-maze-stage-b-05b-12pass-identity-v1",
            "protocol_sha256": sha(PROTOCOL), "launcher_sha256": sha(Path(__file__).resolve()),
            "trainer_sha256": sha(TRAINER), "base_trainer_sha256": sha(BASE_TRAINER),
            "point_base_trainer_sha256": sha(POINT_BASE_TRAINER),
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
            "prompt_passes": 12, "optimizer_updates": 48, "train_prompt_count": 4,
            "policy_microbatch_size": 16,
            "rollouts_per_prompt": 16, "decision_horizon": 16, "action_repeat": 400,
            "evaluation_rounds": list(range(0, 49)), "evaluation_draws": 4,
            "evaluation_k": 8, "evaluation_trajectories_per_coordinate": 132,
            "final_seed_cohort": True, "resume": False, "development_rows_loaded": False,
        }
        atomic(IDENTITY, identity)
        atomic(SUBMISSION, {
            "schema": "ant-maze-stage-b-05b-12pass-submission-v1",
            "identity_sha256": sha(IDENTITY), "manifest_sha256": sha(MANIFEST),
            "jobs": jobs, "held_job_audit": "pass", "released": True,
        })
        atomic(AUDIT_RUNNER, {
            "schema": "ant-maze-stage-b-05b-12pass-audit-runner-v1",
            "audit_job_id": audit_job, "dependency": dependency,
            "identity_sha256": sha(IDENTITY), "execution_hash": execution_hash,
        })
        for job_id in jobs.values():
            run(["scontrol", "release", str(job_id)])
    except BaseException:
        for job_id in [*jobs.values(), *([] if audit_job is None else [audit_job])]:
            subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[ant-stage-b] released jobs {jobs}; audit job {audit_job}")


if __name__ == "__main__":
    main()
