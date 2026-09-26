#!/usr/bin/env python3
"""Prospectively rerun the Pantry pair after its validator-only failure."""

from __future__ import annotations

import json
import os
from pathlib import Path

import launch_pantry_support_mask_paired_smoke_v1 as base


ROOT = Path(__file__).resolve().parents[2]
FAILED_AUDIT = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_audit.json"
FAILED_LOGS = [
    ROOT / "var/artifacts/logs/xdr_train-30200705.err",
    ROOT / "var/artifacts/logs/xdr_train-30200706.err",
]
base.PROTOCOL = ROOT / "paper/preregistration/pantry_support_mask_paired_mechanism_smoke_v1_r1_validator_repair_20260730.md"
base.PREFIX = "ppsmoke_support_mask_paired_v1_r1"
base.IDENTITY = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_r1_identity.json"
base.SUBMISSION = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_r1_submission.json"
base.MANIFEST = ROOT / f"var/artifacts/{base.PREFIX}_comparative_jobs.tsv"
base.AUDIT = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_r1_audit.json"
base.AUDIT_RUNNER = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_r1_audit_runner_identity.json"
base.AUDIT_SUBMISSION = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_r1_audit_submission.json"
base.LAUNCHER_ENTRYPOINT = Path(__file__).resolve()
base.AUDIT_PROGRAM = ROOT / "ops/audit_pantry_support_mask_paired_smoke_v1_r1.py"
base.AUDIT_BATCH = ROOT / "ops/slurm/audit_pantry_support_mask_paired_smoke_v1_r1.slurm"
base.AUDIT_SUPPORT_FILES = [ROOT / "ops/audit_pantry_support_mask_paired_smoke_v1.py"]
base.IDENTITY_EXTRA = {
    "repair_attempt": "task_bound_pantry_validator_only_r1",
    "failed_pair_jobs": [30200705, 30200706],
    "failed_audit_job": 30200707,
    "failed_pair_audit_sha256": base.base.sha(FAILED_AUDIT),
    "failed_training_log_sha256": {
        path.name: base.base.sha(path) for path in FAILED_LOGS
    },
    "scientific_change": False,
}


def prerequisites() -> tuple[dict, Path, Path]:
    for path in (
        base.PROTOCOL,
        base.CONTROL_AUDIT,
        base.CONTROL_IDENTITY,
        FAILED_AUDIT,
        *FAILED_LOGS,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    failed = json.loads(FAILED_AUDIT.read_text())
    if failed.get("status") != "fail":
        raise RuntimeError("r1 requires the immutable failed v1 paired audit")
    for path in FAILED_LOGS:
        if "online canonical banks require executable ModeBench" not in path.read_text(errors="replace"):
            raise RuntimeError(f"r1 antecedent lacks the validator failure: {path}")
    control = json.loads(base.CONTROL_IDENTITY.read_text())
    control_audit = json.loads(base.CONTROL_AUDIT.read_text())
    if control_audit.get("status") != "pass" or control_audit.get("decision") != "eligible_for_paired_mechanism_smoke":
        raise RuntimeError("passing Pantry control audit is required")
    source_root, source_hash = base.base.snapshot_tree(
        ROOT / "src", "pantry_support_mask_paired_smoke_v1_r1"
    )
    ops_root = Path(control["ops_root"])
    if base.base.tree_hash(ops_root) != control.get("execution_hash"):
        raise RuntimeError("passing Pantry ops snapshot drift")
    repaired = dict(control, source_root=str(source_root), source_hash=source_hash)
    return repaired, source_root, ops_root


def validate() -> None:
    base.base.run_command(["bash", "-n", str(base.AUDIT_BATCH)])
    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = str(ROOT / "src")
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    test_env["LD_LIBRARY_PATH"] = library + ((":" + test_env["LD_LIBRARY_PATH"]) if test_env.get("LD_LIBRARY_PATH") else "")
    base.base.run_command([
        str(base.base.PYTHON), "-m", "pytest", "-q",
        str(ROOT / "tests/test_pantry_support_mask_paired_smoke_v1_r1.py"),
        str(ROOT / "tests/test_args.py") + "::test_online_canonical_bank_accepts_task_bound_pantry_contract",
        str(ROOT / "tests/test_pantry_support_mask.py"),
        str(ROOT / "tests/test_e58_global_verified_replay_contract.py"),
    ], env=test_env)


base.prerequisites = prerequisites
base.validate = validate

if __name__ == "__main__":
    base.main()
