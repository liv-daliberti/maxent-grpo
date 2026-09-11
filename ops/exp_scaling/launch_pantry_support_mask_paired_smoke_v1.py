#!/usr/bin/env python3
"""Configure or launch PantryPlan's frozen paired MaxEnt mechanism smoke."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


os.environ["PANTRY_REPAIR_SUFFIX"] = "r2"
import launch_pantry_support_mask_drgrpo_smoke_v1_r1 as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "paper/preregistration/pantry_support_mask_paired_mechanism_smoke_v1_20260730.md"
CONTROL_AUDIT = ROOT / "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_r1_audit.json"
CONTROL_IDENTITY = ROOT / "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_identity.json"
PREFIX = "ppsmoke_support_mask_paired_v1"
IDENTITY = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_identity.json"
SUBMISSION = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_submission.json"
MANIFEST = ROOT / f"var/artifacts/{PREFIX}_comparative_jobs.tsv"
AUDIT = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_audit.json"
AUDIT_RUNNER = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_audit_runner_identity.json"
AUDIT_SUBMISSION = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_audit_submission.json"
LAUNCHER_ENTRYPOINT = Path(__file__).resolve()
AUDIT_PROGRAM = ROOT / "ops/audit_pantry_support_mask_paired_smoke_v1.py"
AUDIT_BATCH = ROOT / "ops/slurm/audit_pantry_support_mask_paired_smoke_v1.slurm"
AUDIT_SUPPORT_FILES: list[Path] = []
IDENTITY_EXTRA: dict[str, object] = {}
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
SEED = 76201
OPTIMIZER_UPDATES = 32
ROLLOUTS_PER_PROMPT = 16
EXPECTED_MAX_QUERIES = 496


def environment(source_root: Path, ops_root: Path) -> dict[str, str]:
    values = base.scientific_environment()
    values.update(
        RUN_STAMP_PREFIX=PREFIX,
        OAT_ZERO_ONLY_ARMS="grpo,verified_first_global_replay_canonical",
        OAT_ZERO_DRGRPO_VARIANT="grpo_compute_matched",
        OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM="1",
        OAT_ZERO_MAXENT_ALPHA="0",
        OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA="0",
        OAT_ZERO_SEMANTIC_SHANNON_COEF="0.10",
        OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP="5.0",
        OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT="1.0",
        OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP="0.05",
        OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS="64",
        OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY="0.90",
        OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA="0",
        OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA="0.50",
        OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT="1.0",
        OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP="5.0",
        OAT_ZERO_ONLINE_CANONICAL_KEY_MODE="modebench_outcome",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA="0.10",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY="16",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP="1",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS="64",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY="0.90",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA="0.10",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS="64",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY="0.90",
        OAT_ZERO_CAMPAIGN_SOURCE_ROOT=str(source_root),
        OAT_ZERO_OPS_SNAPSHOT_ROOT=str(ops_root),
        OAT_ZERO_PROTOCOL_IDENTITY=str(IDENTITY),
    )
    return values


def prerequisites() -> tuple[dict, Path, Path]:
    for path in (PROTOCOL, CONTROL_AUDIT, CONTROL_IDENTITY):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(CONTROL_AUDIT.read_text())
    if audit.get("status") != "pass" or audit.get("decision") != "eligible_for_paired_mechanism_smoke":
        raise RuntimeError("passing r2-r1 Pantry control audit is required")
    control = json.loads(CONTROL_IDENTITY.read_text())
    source_root = Path(control["source_root"])
    ops_root = Path(control["ops_root"])
    if base.tree_hash(source_root) != control.get("source_hash"):
        raise RuntimeError("passing Pantry source snapshot drift")
    if base.tree_hash(ops_root) != control.get("execution_hash"):
        raise RuntimeError("passing Pantry ops snapshot drift")
    return control, source_root, ops_root


def validate() -> None:
    base.run_command([
        str(base.PYTHON), "-m", "py_compile", str(Path(__file__)),
        str(AUDIT_PROGRAM),
    ])
    base.run_command(["bash", "-n", str(AUDIT_BATCH)])
    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = str(ROOT / "src")
    test_env["LD_LIBRARY_PATH"] = str(ROOT / "var/seed_paper_eval/paper310/lib") + (":" + test_env["LD_LIBRARY_PATH"] if test_env.get("LD_LIBRARY_PATH") else "")
    base.run_command([
        str(base.PYTHON), "-m", "pytest", "-q",
        str(ROOT / "tests/test_pantry_support_mask_paired_smoke_v1.py"),
        str(ROOT / "tests/test_pantry_support_mask.py"),
        str(ROOT / "tests/test_e58_global_verified_replay_contract.py"),
    ], env=test_env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    parsed = parser.parse_args()
    control, source_root, ops_root = prerequisites()
    validate()
    run_env = dict(os.environ)
    run_env.update(environment(source_root, ops_root))
    if parsed.phase == "config":
        run_env.update(OAT_ZERO_COMPARATIVE_CONFIG_ONLY="1", OAT_ZERO_SBATCH_HOLD="0")
        base.run_command([str(ops_root / "submit_countdown_comparative.sh")], env=run_env)
        print("[pantry-paired-v1] configuration passed; no job submitted")
        return

    for path in (IDENTITY, SUBMISSION, MANIFEST, AUDIT, AUDIT_RUNNER, AUDIT_SUBMISSION):
        if path.exists():
            raise FileExistsError(f"fresh paired Pantry artifact required: {path}")
    audit_root, audit_hash = base.snapshot_files(
        [
            (AUDIT_PROGRAM, Path(AUDIT_PROGRAM.name)),
            (AUDIT_BATCH, Path(AUDIT_BATCH.name)),
        ] + [(path, Path(path.name)) for path in AUDIT_SUPPORT_FILES],
        "pantry_support_mask_paired_smoke_v1_audit",
    )
    identity = {
        "schema": "pantry-support-mask-paired-smoke-v1",
        "protocol_sha256": base.sha(PROTOCOL),
        "launcher_sha256": base.sha(LAUNCHER_ENTRYPOINT),
        "passing_control_audit_sha256": base.sha(CONTROL_AUDIT),
        "passing_control_identity_sha256": base.sha(CONTROL_IDENTITY),
        "source_root": str(source_root), "source_hash": control["source_hash"],
        "ops_root": str(ops_root), "execution_hash": control["execution_hash"],
        "audit_root": str(audit_root), "audit_execution_hash": audit_hash,
        "data_tree_sha256": base.tree_hash(base.DATA),
        "model_config_sha256": base.sha(base.MODEL / "config.json"),
        "arms": [CONTROL, TREATMENT], "jobs": {}, "seed": SEED,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
        "maxent_treatment": True, "compute_matched_control": True,
        "development_only": True, "final_seed": False,
        **IDENTITY_EXTRA,
    }
    base.atomic(IDENTITY, identity)
    run_env.update(OAT_ZERO_COMPARATIVE_CONFIG_ONLY="0", OAT_ZERO_SBATCH_HOLD="1")
    base.run_command([str(ops_root / "submit_countdown_comparative.sh")], env=run_env)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if {(row["arm"], row["seed"]) for row in rows} != {(CONTROL, str(SEED)), (TREATMENT, str(SEED))} or len(rows) != 2:
        raise RuntimeError("paired Pantry manifest differs from frozen pair")
    jobs = {row["arm"]: int(row["job_id"]) for row in rows}
    before: dict[str, str] = {}; after: dict[str, str] = {}
    for arm, job_id in jobs.items():
        record = base.run_command(["scontrol", "show", "job", "-o", str(job_id)])
        required = ["JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=node302", "gres/gpu:a100:1", "RunTime=00:00:00"]
        variant = "grpo_compute_matched" if arm == CONTROL else TREATMENT
        required.append(f"OAT_ZERO_VARIANT={variant}")
        required.extend([
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1",
            f"OAT_ZERO_MAX_QUERIES={EXPECTED_MAX_QUERIES}",
            f"OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES={OPTIMIZER_UPDATES}",
        ])
        if arm == CONTROL:
            required.extend(["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1", "OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0"])
        else:
            required.extend(["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0", "OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.5", "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout"])
        for text in required:
            if text not in record:
                raise RuntimeError(f"held paired Pantry {arm} job lacks {text}")
        before[arm] = record
        base.run_command(["scontrol", "update", f"JobId={job_id}", "Partition=all", "Gres=gpu:a5000:1", "NodeList="])
        base.run_command(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        amended = base.run_command(["scontrol", "show", "job", "-o", str(job_id)])
        for text in ("Partition=all", "ReqNodeList=(null)", "gres/gpu:a5000:1", "Requeue=0", "RunTime=00:00:00"):
            if text not in amended:
                raise RuntimeError(f"amended paired Pantry {arm} job lacks {text}")
        after[arm] = amended
    identity.update(jobs=jobs, manifest_sha256=base.sha(MANIFEST), held_scheduler_before=before, held_scheduler_after=after)
    base.atomic(IDENTITY, identity)
    base.atomic(SUBMISSION, {
        "schema": "pantry-support-mask-paired-smoke-submission-v1",
        "identity_sha256": base.sha(IDENTITY), "manifest_sha256": base.sha(MANIFEST),
        "jobs": jobs, "held_job_audit": "pass", "released": True,
    })
    for job_id in jobs.values():
        base.run_command(["scontrol", "release", str(job_id)])
    dependency = "afterany:" + ":".join(str(job_id) for job_id in jobs.values())
    audit_job = int(base.run_command([
        "sbatch", "--parsable", f"--dependency={dependency}",
        f"--export=ALL,ROOT_DIR={ROOT},OAT_ZERO_EXECUTION_ROOT={audit_root}",
        str(audit_root / AUDIT_BATCH.name),
    ]).split(";", 1)[0])
    base.atomic(AUDIT_RUNNER, {"schema":"pantry-support-mask-paired-smoke-audit-runner-v1","jobs":jobs,"audit_job_id":audit_job,"dependency":dependency,"audit_execution_hash":audit_hash,"identity_sha256":base.sha(IDENTITY)})
    base.atomic(AUDIT_SUBMISSION, {"schema":"pantry-support-mask-paired-smoke-audit-submission-v1","audit_runner_sha256":base.sha(AUDIT_RUNNER),"audit_job_id":audit_job,"released":True})
    print(f"[pantry-paired-v1] released jobs {jobs}; audit job {audit_job}")


if __name__ == "__main__":
    main()
