#!/usr/bin/env python3
"""Configure or launch PantryPlan's frozen ten-job Stage-B cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import launch_pantry_support_mask_paired_integration_v3 as qualification


ROOT = Path(__file__).resolve().parents[2]
UTIL = qualification.base.base
PROTOCOL = ROOT / "paper/preregistration/pantry_stage_b_05b_12pass_20260730.md"
QUALIFICATION_AUDIT = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v3_audit.json"
QUALIFICATION_IDENTITY = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v3_identity.json"
PREFIX = "ppe70_clean_stage_b_05b_12pass"
IDENTITY = ROOT / "var/artifacts/pantry_stage_b_05b_12pass_identity.json"
SUBMISSION = ROOT / "var/artifacts/pantry_stage_b_05b_12pass_submission.json"
MANIFEST = ROOT / f"var/artifacts/{PREFIX}_comparative_jobs.tsv"
AUDIT = ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json"
AUDIT_RUNNER = ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit_runner_identity.json"
AUDIT_SUBMISSION = ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit_submission.json"
AUDIT_PROGRAM = ROOT / "ops/audit_pantry_stage_b_05b_12pass.py"
AUDIT_BATCH = ROOT / "ops/slurm/audit_pantry_stage_b_05b_12pass.slurm"
LAUNCHER = Path(__file__).resolve()
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEEDS = (43, 44, 45, 46, 47)
UPDATES = 384
ROLLOUTS = 16
MAX_QUERIES = 6128
EVAL_INTERVAL = 8
EVAL_DRAWS = 4


def prerequisites() -> tuple[dict, Path, Path]:
    for path in (
        PROTOCOL,
        QUALIFICATION_AUDIT,
        QUALIFICATION_IDENTITY,
        AUDIT_PROGRAM,
        AUDIT_BATCH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(QUALIFICATION_AUDIT.read_text())
    identity = json.loads(QUALIFICATION_IDENTITY.read_text())
    if audit.get("status") != "pass" or audit.get("decision") != "eligible_for_ten_stage_b_jobs":
        raise RuntimeError("passing Pantry v3 qualification is required")
    if identity.get("optimizer_updates") != 96 or identity.get("seed") != 76201:
        raise RuntimeError("Pantry v3 qualification identity drift")
    source_root = Path(identity["source_root"])
    ops_root = Path(identity["ops_root"])
    if UTIL.tree_hash(source_root) != identity.get("source_hash"):
        raise RuntimeError("passing Pantry source snapshot drift")
    if UTIL.tree_hash(ops_root) != identity.get("execution_hash"):
        raise RuntimeError("passing Pantry operations snapshot drift")
    if UTIL.tree_hash(UTIL.DATA) != identity.get("data_tree_sha256"):
        raise RuntimeError("passing Pantry data drift")
    if UTIL.sha(UTIL.MODEL / "config.json") != identity.get("model_config_sha256"):
        raise RuntimeError("passing Pantry model drift")
    return identity, source_root, ops_root


def environment(source_root: Path, ops_root: Path) -> dict[str, str]:
    values = qualification.environment(source_root, ops_root)
    values.update(
        RUN_STAMP_PREFIX=PREFIX,
        OAT_ZERO_TRAIN_SEEDS=",".join(str(seed) for seed in SEEDS),
        OAT_ZERO_MAX_QUERIES=str(MAX_QUERIES),
        OAT_ZERO_MAX_PROMPT_EPOCHS="12",
        OAT_ZERO_NUM_PROMPT_EPOCH="12",
        OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES=str(UPDATES),
        OAT_ZERO_EVAL_PROMPT_INTERVAL=str(EVAL_INTERVAL),
        OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=str(EVAL_DRAWS),
        OAT_ZERO_EVAL_MODE_COVERAGE_SEED="76299",
        OAT_ZERO_SAVE_STEPS=str(UPDATES),
        OAT_ZERO_SAVE_FROM=str(UPDATES),
        OAT_ZERO_TRAIN_TIME_LIMIT="08:00:00",
        OAT_ZERO_PROTOCOL_IDENTITY=str(IDENTITY),
    )
    return values


def validate() -> None:
    UTIL.run_command([
        str(UTIL.PYTHON),
        "-m",
        "py_compile",
        str(LAUNCHER),
        str(AUDIT_PROGRAM),
    ])
    UTIL.run_command(["bash", "-n", str(AUDIT_BATCH)])
    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = str(ROOT / "src")
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    inherited = test_env.get("LD_LIBRARY_PATH", "")
    test_env["LD_LIBRARY_PATH"] = library + ((":" + inherited) if inherited else "")
    UTIL.run_command([
        str(UTIL.PYTHON),
        "-m",
        "pytest",
        "-q",
        str(ROOT / "tests/test_pantry_stage_b_05b_12pass.py"),
        str(ROOT / "tests/test_pantry_support_mask_paired_integration_v3.py"),
        str(ROOT / "tests/test_e58_global_verified_replay_contract.py"),
    ], env=test_env)


def expected_cells() -> set[tuple[str, str]]:
    return {(arm, str(seed)) for arm in ARMS for seed in SEEDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    parsed = parser.parse_args()
    qualified, source_root, ops_root = prerequisites()
    validate()
    run_env = dict(os.environ)
    run_env.update(environment(source_root, ops_root))
    if parsed.phase == "config":
        run_env.update(OAT_ZERO_COMPARATIVE_CONFIG_ONLY="1", OAT_ZERO_SBATCH_HOLD="0")
        UTIL.run_command([str(ops_root / "submit_countdown_comparative.sh")], env=run_env)
        print("[pantry-stage-b] configuration passed; no job submitted")
        return

    for path in (IDENTITY, SUBMISSION, MANIFEST, AUDIT, AUDIT_RUNNER, AUDIT_SUBMISSION):
        if path.exists():
            raise FileExistsError(f"fresh Pantry Stage-B artifact required: {path}")
    audit_root, audit_hash = UTIL.snapshot_files(
        [
            (AUDIT_PROGRAM, Path(AUDIT_PROGRAM.name)),
            (AUDIT_BATCH, Path(AUDIT_BATCH.name)),
        ],
        "pantry_stage_b_05b_12pass_audit",
    )
    identity = {
        "schema": "pantry-stage-b-05b-12pass-v1",
        "protocol_sha256": UTIL.sha(PROTOCOL),
        "launcher_sha256": UTIL.sha(LAUNCHER),
        "qualification_audit_sha256": UTIL.sha(QUALIFICATION_AUDIT),
        "qualification_identity_sha256": UTIL.sha(QUALIFICATION_IDENTITY),
        "qualification_jobs": qualified["jobs"],
        "source_root": str(source_root),
        "source_hash": qualified["source_hash"],
        "ops_root": str(ops_root),
        "execution_hash": qualified["execution_hash"],
        "audit_root": str(audit_root),
        "audit_execution_hash": audit_hash,
        "data_tree_sha256": qualified["data_tree_sha256"],
        "model_config_sha256": qualified["model_config_sha256"],
        "model": "Qwen2.5-0.5B-Instruct",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "jobs": {},
        "prompt_passes": 12,
        "train_rows": 32,
        "optimizer_updates": UPDATES,
        "rollouts_per_prompt": ROLLOUTS,
        "max_queries": MAX_QUERIES,
        "evaluation_interval_updates": EVAL_INTERVAL,
        "evaluation_draws": EVAL_DRAWS,
        "registered_anchor_passes": [0, 1, 2, 3, 4, 5, 6, 8, 10, 12],
        "maxent_treatment": True,
        "compute_matched_control": True,
        "development_only": False,
        "final_seed_cohort": True,
        "fresh_initial_checkpoint": True,
        "resume_enabled": False,
    }
    UTIL.atomic(IDENTITY, identity)
    run_env.update(OAT_ZERO_COMPARATIVE_CONFIG_ONLY="0", OAT_ZERO_SBATCH_HOLD="1")
    UTIL.run_command([str(ops_root / "submit_countdown_comparative.sh")], env=run_env)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if {(row["arm"], row["seed"]) for row in rows} != expected_cells() or len(rows) != 10:
        raise RuntimeError("Pantry Stage-B manifest differs from frozen ten cells")
    jobs = {
        f"{row['arm']}/s{row['seed']}": int(row["job_id"])
        for row in rows
    }
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for row in rows:
        arm = row["arm"]
        seed = row["seed"]
        key = f"{arm}/s{seed}"
        job_id = int(row["job_id"])
        record = UTIL.run_command(["scontrol", "show", "job", "-o", str(job_id)])
        variant = "grpo_compute_matched" if arm == CONTROL else TREATMENT
        required = [
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "ReqNodeList=node302",
            "gres/gpu:a100:1",
            "RunTime=00:00:00",
            f"OAT_ZERO_VARIANT={variant}",
            f"OAT_ZERO_SEED={seed}",
            f"OAT_ZERO_MAX_QUERIES={MAX_QUERIES}",
            f"OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES={UPDATES}",
            f"OAT_ZERO_EVAL_PROMPT_INTERVAL={EVAL_INTERVAL}",
            f"OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS={EVAL_DRAWS}",
            "OAT_ZERO_MAX_PROMPT_EPOCHS=12",
            "OAT_ZERO_NUM_PROMPT_EPOCH=12",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1",
        ]
        if arm == CONTROL:
            required.extend([
                "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
                "OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0",
            ])
        else:
            required.extend([
                "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
                "OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.5",
                "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout",
            ])
        for text in required:
            if text not in record:
                raise RuntimeError(f"held Pantry Stage-B {key} job lacks {text}")
        before[key] = record
        UTIL.run_command([
            "scontrol", "update", f"JobId={job_id}", "Partition=all",
            "Gres=gpu:a5000:1", "NodeList=",
        ])
        UTIL.run_command(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        amended = UTIL.run_command(["scontrol", "show", "job", "-o", str(job_id)])
        for text in (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "ReqNodeList=(null)",
            "gres/gpu:a5000:1",
            "Requeue=0",
            "RunTime=00:00:00",
        ):
            if text not in amended:
                raise RuntimeError(f"amended Pantry Stage-B {key} job lacks {text}")
        if "Partition=all" not in amended and "Partition=mltheory" not in amended:
            raise RuntimeError(f"amended Pantry Stage-B {key} partition mismatch")
        after[key] = amended
    identity.update(
        jobs=jobs,
        manifest_sha256=UTIL.sha(MANIFEST),
        held_scheduler_before=before,
        held_scheduler_after=after,
    )
    UTIL.atomic(IDENTITY, identity)
    UTIL.atomic(SUBMISSION, {
        "schema": "pantry-stage-b-05b-12pass-submission-v1",
        "identity_sha256": UTIL.sha(IDENTITY),
        "manifest_sha256": UTIL.sha(MANIFEST),
        "jobs": jobs,
        "held_job_audit": "pass",
        "released": True,
    })
    dependency = "afterany:" + ":".join(str(job_id) for job_id in jobs.values())
    audit_job = int(UTIL.run_command([
        "sbatch",
        "--parsable",
        f"--dependency={dependency}",
        f"--export=ALL,ROOT_DIR={ROOT},OAT_ZERO_EXECUTION_ROOT={audit_root}",
        str(audit_root / AUDIT_BATCH.name),
    ]).split(";", 1)[0])
    UTIL.atomic(AUDIT_RUNNER, {
        "schema": "pantry-stage-b-05b-12pass-audit-runner-v1",
        "jobs": jobs,
        "audit_job_id": audit_job,
        "dependency": dependency,
        "audit_execution_hash": audit_hash,
        "identity_sha256": UTIL.sha(IDENTITY),
    })
    UTIL.atomic(AUDIT_SUBMISSION, {
        "schema": "pantry-stage-b-05b-12pass-audit-submission-v1",
        "audit_runner_sha256": UTIL.sha(AUDIT_RUNNER),
        "audit_job_id": audit_job,
        "released": True,
    })
    for job_id in jobs.values():
        UTIL.run_command(["scontrol", "release", str(job_id)])
    print(f"[pantry-stage-b] released jobs {jobs}; audit job {audit_job}")


if __name__ == "__main__":
    main()
