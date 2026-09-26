#!/usr/bin/env python3
"""Configure or conditionally launch the frozen AntMaze v13 paired smoke."""

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
PROTOCOL = ROOT / "paper/preregistration/ant_maze_interactive_paired_smoke_v13_20260730.md"
REPAIR = ROOT / "paper/preregistration/maze_microbatch4_scoring_repair_20260730.md"
FAILED_V13R1_AUDIT = ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13r1_audit.json"
V13R1_DIAGNOSTIC = ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13r1_logprob_batch_diagnostic.json"
TRAINER = ROOT / "ops/train_ant_maze_interactive_paired_smoke_v13.py"
BASE_TRAINER = ROOT / "ops/train_point_maze_interactive_paired_smoke_v1.py"
AUDITOR = ROOT / "ops/audit_ant_maze_interactive_paired_smoke_v13.py"
TRAIN_BATCH = ROOT / "ops/slurm/train_ant_maze_interactive_paired_smoke_v13.slurm"
AUDIT_BATCH = ROOT / "ops/slurm/audit_ant_maze_interactive_paired_smoke_v13.slurm"
VIABILITY = ROOT / "var/artifacts/ant_maze_interactive_05b_viability_v13.json"
VIABILITY_IDENTITY = ROOT / "var/artifacts/ant_maze_interactive_warmstart_v13_identity.json"
ADMISSION = ROOT / "var/artifacts/ant_maze_modebench_v12_admission_audit.json"
WARMSTART_RECEIPT = ROOT / "var/artifacts/ant_maze_interactive_warmstart_sft_v13.json"
MODEL = ROOT / "var/models/ant_maze_interactive_warmstart_v13"
DATA = ROOT / "var/data/ant_maze_modebench_v12"
WORKER = ROOT / "var/maze_runtime/venv/bin/python"
IDENTITY = ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13_identity.json"
MANIFEST = ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13_jobs.tsv"
SUBMISSION = ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13_submission.json"
AUDIT_OUTPUT = ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13_audit.json"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEED = 76313
COHORT = "v13"


def configure_cohort(cohort: str) -> None:
    global COHORT, PROTOCOL, REPAIR, IDENTITY, MANIFEST, SUBMISSION, AUDIT_OUTPUT
    if cohort not in ("v13", "v13r1", "v13r2"):
        raise ValueError("unknown AntMaze paired-smoke cohort")
    COHORT = cohort
    if cohort == "v13r2":
        PROTOCOL = ROOT / "paper/preregistration/ant_maze_interactive_paired_smoke_v13r2_20260730.md"
        REPAIR = ROOT / "paper/preregistration/maze_microbatch16_sampling_geometry_repair_20260730.md"
    stem = ROOT / f"var/artifacts/ant_maze_interactive_paired_smoke_{cohort}"
    IDENTITY = Path(str(stem) + "_identity.json")
    MANIFEST = Path(str(stem) + "_jobs.tsv")
    SUBMISSION = Path(str(stem) + "_submission.json")
    AUDIT_OUTPUT = Path(str(stem) + "_audit.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True); handle.write("\n")
    os.replace(temporary, path)


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        list(command), cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def base_prerequisites() -> None:
    for path in (
        PYTHON, PROTOCOL, REPAIR, TRAINER, BASE_TRAINER, AUDITOR, TRAIN_BATCH, AUDIT_BATCH,
        VIABILITY_IDENTITY, ADMISSION, FAILED_V13R1_AUDIT, V13R1_DIAGNOSTIC, DATA / "identity.json",
        DATA / "train/dataset_dict.json", WORKER,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    identity = json.loads(VIABILITY_IDENTITY.read_text())
    admission = json.loads(ADMISSION.read_text())
    if identity.get("job_id") != 30202183 or admission.get("status") != "pass":
        raise RuntimeError("AntMaze v13 paired-smoke antecedent identity drift")


def viability_passes() -> None:
    for path in (VIABILITY, WARMSTART_RECEIPT, MODEL / "config.json"):
        if not path.exists():
            raise RuntimeError(f"AntMaze v13 viability has not completed: {path}")
    payload = json.loads(VIABILITY.read_text())
    summary = payload.get("summary", {})
    if (
        payload.get("status") != "pass"
        or payload.get("decision") != "eligible_for_ant_v13_paired_online_smoke"
        or summary.get("prompt_count") != 4
        or int(summary.get("prefix_success_prompts", 0)) < 2
        or int(summary.get("multimode_prompts", 0)) < 1
    ):
        raise RuntimeError("AntMaze v13 viability did not authorize paired smoke")


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
    run(["bash", "-n", str(TRAIN_BATCH)]); run(["bash", "-n", str(AUDIT_BATCH)])
    run([
        str(PYTHON), "-m", "pytest", "-q",
        str(ROOT / "tests/test_ant_maze_paired_smoke_v13.py"),
        str(ROOT / "tests/test_ant_maze_interactive_v13.py"),
        str(ROOT / "tests/test_interactive_episode_objective.py"),
        str(ROOT / "tests/test_interactive_episode_replay.py"),
    ], env=environment)


def snapshot_source() -> tuple[Path, str]:
    source = ROOT / "src"; digest = tree_hash(source)
    parent = ROOT / f"var/artifacts/source_snapshots/ant_paired_smoke_{COHORT}_{digest}"
    target = parent / "src"
    if not target.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".source.", dir=parent))
        shutil.copytree(source, staging / "src"); os.replace(staging / "src", target); staging.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError("AntMaze paired source snapshot mismatch")
    return target, digest


def snapshot_execution() -> tuple[Path, str]:
    sources = (TRAINER, BASE_TRAINER, AUDITOR, TRAIN_BATCH, AUDIT_BATCH, PROTOCOL, REPAIR)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".ant-paired-smoke-{COHORT}.", dir=ROOT / "var/artifacts/source_snapshots"
    ))
    for source in sources:
        shutil.copy2(source, temporary / source.name)
    digest = tree_hash(temporary)
    target = ROOT / f"var/artifacts/source_snapshots/ant_paired_smoke_{COHORT}_ops_{digest}"
    if not target.is_dir(): os.replace(temporary, target)
    else: shutil.rmtree(temporary)
    if tree_hash(target) != digest:
        raise RuntimeError("AntMaze paired execution snapshot mismatch")
    return target, digest


def arm_paths(arm: str) -> dict[str, Path]:
    stem = ROOT / f"var/artifacts/ant_maze_interactive_paired_smoke_{COHORT}_{arm}"
    return {
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "state_replay": Path(str(stem) + ".state_replay.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    parser.add_argument("--cohort", choices=("v13", "v13r1", "v13r2"), default="v13")
    args = parser.parse_args(); configure_cohort(args.cohort)
    base_prerequisites(); validate()
    if args.phase == "config":
        print("[ant-paired-v13] configuration passed; no job submitted"); return
    viability_passes()
    fresh = [IDENTITY, MANIFEST, SUBMISSION, AUDIT_OUTPUT]
    for arm in ARMS: fresh.extend(arm_paths(arm).values())
    for path in fresh:
        if path.exists(): raise FileExistsError(f"fresh AntMaze paired artifact required: {path}")
    source_root, source_hash = snapshot_source(); execution_root, execution_hash = snapshot_execution()
    jobs: dict[str, int] = {}; audit_job: int | None = None
    try:
        for arm in ARMS:
            output = run([
                "sbatch", "--parsable", "--hold", f"--job-name=ant-{COHORT}-smoke-{arm}",
                "--partition=all", "--account=allcs",
                "--export=ALL,"
                f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
                f"OAT_ZERO_EXECUTION_ROOT={execution_root},OAT_ZERO_SOURCE_HASH={source_hash},"
                f"OAT_ZERO_EXECUTION_HASH={execution_hash},OAT_ZERO_ARM={arm},"
                f"OAT_ZERO_ANT_SMOKE_COHORT={COHORT}",
                str(execution_root / TRAIN_BATCH.name),
            ])
            job_id = int(output.split(";", 1)[0]); jobs[arm] = job_id
            run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
            record = run(["scontrol", "show", "job", "-o", str(job_id)])
            for required in (
                "JobState=PENDING", "Reason=JobHeldUser", "gres/gpu:a5000:1",
                "NumCPUs=8", "MinMemoryNode=64G", "TimeLimit=12:00:00", "Requeue=0",
                f"OAT_ZERO_ARM={arm}", f"OAT_ZERO_SOURCE_HASH={source_hash}",
                f"OAT_ZERO_EXECUTION_HASH={execution_hash}",
            ):
                if required not in record: raise RuntimeError(f"held AntMaze {arm} job lacks {required}")
        with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("arm", "seed", "job_id", "receipt", "metrics", "state_replay"),
                delimiter="\t", lineterminator="\n",
            ); writer.writeheader()
            for arm in ARMS:
                paths = arm_paths(arm); writer.writerow({
                    "arm": arm, "seed": SEED, "job_id": jobs[arm],
                    **{name: path.relative_to(ROOT).as_posix() for name, path in paths.items()},
                })
        dependency = "afterany:" + ":".join(str(jobs[arm]) for arm in ARMS)
        audit_job = int(run([
            "sbatch", "--parsable", f"--dependency={dependency}",
            f"--job-name=audit-ant-paired-{COHORT}",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},OAT_ZERO_EXECUTION_ROOT={execution_root},"
            f"OAT_ZERO_ANT_SMOKE_COHORT={COHORT}",
            str(execution_root / AUDIT_BATCH.name),
        ]).split(";", 1)[0])
        identity = {
            "schema": "ant-maze-interactive-paired-smoke-identity-v13",
            "artifact_cohort": COHORT, "protocol_sha256": sha(PROTOCOL),
            "scoring_repair_sha256": sha(REPAIR), "launcher_sha256": sha(Path(__file__).resolve()),
            "trainer_sha256": sha(TRAINER), "base_trainer_sha256": sha(BASE_TRAINER),
            "auditor_sha256": sha(AUDITOR), "source_root": str(source_root),
            "source_hash": source_hash, "execution_root": str(execution_root),
            "execution_hash": execution_hash, "model_tree_sha256": tree_hash(MODEL),
            "data_identity_sha256": sha(DATA / "identity.json"),
            "viability_receipt_sha256": sha(VIABILITY), "viability_identity_sha256": sha(VIABILITY_IDENTITY),
            "admission_audit_sha256": sha(ADMISSION), "warmstart_receipt_sha256": sha(WARMSTART_RECEIPT),
            "manifest_sha256": sha(MANIFEST), "arms": list(ARMS), "jobs": jobs,
            "audit_job_id": audit_job, "audit_dependency": dependency, "seed": SEED,
            "row_indices": [0, 1, 2, 3], "rollouts_per_prompt": 16,
            "decision_horizon": 16, "action_repeat": 400, "optimizer_updates_per_arm": 4,
            "policy_microbatch_size": 16 if COHORT == "v13r2" else 4,
            "fixed_policy_slots_per_arm": 1024, "fixed_replay_decision_slots_per_arm": 1024,
            "development_only": True, "final_seed": False,
            "development_rows_loaded": False, "evaluation_rows_loaded": False,
            "frozen_low_level_controller": True,
            "cancelled_zero_runtime_predecessor_jobs": [30202665, 30202666] if COHORT == "v13r1" else [],
            "failed_v13r1_audit_sha256": sha(FAILED_V13R1_AUDIT),
            "v13r1_batch_diagnostic_sha256": sha(V13R1_DIAGNOSTIC),
            "failed_predecessor_jobs": [30202916, 30202917, 30202918] if COHORT == "v13r2" else [],
        }
        atomic(IDENTITY, identity)
        atomic(SUBMISSION, {
            "schema": "ant-maze-interactive-paired-smoke-submission-v13",
            "identity_sha256": sha(IDENTITY), "manifest_sha256": sha(MANIFEST),
            "jobs": jobs, "audit_job_id": audit_job, "held_job_audit": "pass", "released": True,
        })
        for job_id in jobs.values(): run(["scontrol", "release", str(job_id)])
    except BaseException:
        for job_id in [*jobs.values(), *([] if audit_job is None else [audit_job])]:
            subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[ant-paired-v13] released jobs {jobs}; audit job {audit_job}")


if __name__ == "__main__":
    main()
