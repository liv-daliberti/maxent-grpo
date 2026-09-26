#!/usr/bin/env python3
"""Launch the 96-update task-bound Pantry MaxEnt integration qualification."""

from __future__ import annotations

import json
import os
from pathlib import Path

import launch_pantry_support_mask_paired_smoke_v1 as base


ROOT = Path(__file__).resolve().parents[2]
FAILED_AUDIT = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_r1_audit.json"
FAILED_IDENTITY = ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_r1_identity.json"
FAILED_METRICS = {
    30200958: ROOT / "var/data/xdr_qwen25_0p5b_instruct_grpo_compute_matched_ppsmoke_support_mask_paired_v1_r1_grpo_s76201/debug_job30200958/train_metrics.jsonl",
    30200959: ROOT / "var/data/xdr_qwen25_0p5b_instruct_verified_first_global_replay_canonical_ppsmoke_support_mask_paired_v1_r1_verified_first_global_replay_canonical_s76201/debug_job30200959/train_metrics.jsonl",
}
base.PROTOCOL = ROOT / "paper/preregistration/pantry_support_mask_paired_integration_v2_20260730.md"
base.PREFIX = "ppsmoke_support_mask_paired_integration_v2"
base.IDENTITY = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v2_identity.json"
base.SUBMISSION = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v2_submission.json"
base.MANIFEST = ROOT / f"var/artifacts/{base.PREFIX}_comparative_jobs.tsv"
base.AUDIT = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v2_audit.json"
base.AUDIT_RUNNER = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v2_audit_runner_identity.json"
base.AUDIT_SUBMISSION = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v2_audit_submission.json"
base.LAUNCHER_ENTRYPOINT = Path(__file__).resolve()
base.AUDIT_PROGRAM = ROOT / "ops/audit_pantry_support_mask_paired_integration_v2.py"
base.AUDIT_BATCH = ROOT / "ops/slurm/audit_pantry_support_mask_paired_integration_v2.slurm"
base.AUDIT_SUPPORT_FILES = [ROOT / "ops/audit_pantry_support_mask_paired_smoke_v1.py"]
base.OPTIMIZER_UPDATES = 96
base.ROLLOUTS_PER_PROMPT = 16
base.EXPECTED_MAX_QUERIES = 1488


def prerequisites() -> tuple[dict, Path, Path]:
    for path in (base.PROTOCOL, base.CONTROL_AUDIT, base.CONTROL_IDENTITY, FAILED_AUDIT, FAILED_IDENTITY, *FAILED_METRICS.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    failed = json.loads(FAILED_AUDIT.read_text())
    if failed.get("status") != "fail" or failed.get("errors") != [
        "grpo/j30200958: no online two-mode prompt",
        "verified_first_global_replay_canonical/j30200959: no applied exploration advantage",
        "verified_first_global_replay_canonical/j30200959: no online two-mode prompt",
    ]:
        raise RuntimeError("v2 requires the exact immutable v1-r1 mechanism failure")
    for job_id, path in FAILED_METRICS.items():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()][1:33]
        if len(rows) != 32 or not all(float(row.get("actor/rewards", 0)) > 0 for row in rows):
            raise RuntimeError(f"v1-r1 job {job_id} lacks exact positive-reward evidence")
        if not all(float(row.get("train/online_canonical_validator_task_disagreement_rows", 0)) > 0 for row in rows):
            raise RuntimeError(f"v1-r1 job {job_id} lacks the mask-validator mismatch")
    control = json.loads(base.CONTROL_IDENTITY.read_text())
    control_audit = json.loads(base.CONTROL_AUDIT.read_text())
    if control_audit.get("status") != "pass" or control_audit.get("decision") != "eligible_for_paired_mechanism_smoke":
        raise RuntimeError("passing Pantry control audit is required")
    source_root, source_hash = base.base.snapshot_tree(ROOT / "src", "pantry_support_mask_paired_integration_v2")
    ops_root = Path(control["ops_root"])
    if base.base.tree_hash(ops_root) != control.get("execution_hash"):
        raise RuntimeError("passing Pantry ops snapshot drift")
    return dict(control, source_root=str(source_root), source_hash=source_hash), source_root, ops_root


original_environment = base.environment


def environment(source_root: Path, ops_root: Path) -> dict[str, str]:
    values = original_environment(source_root, ops_root)
    values.update(
        OAT_ZERO_MAX_QUERIES="1488",
        OAT_ZERO_MAX_PROMPT_EPOCHS="3",
        OAT_ZERO_NUM_PROMPT_EPOCH="3",
        OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES="96",
        OAT_ZERO_EVAL_PROMPT_INTERVAL="24",
        OAT_ZERO_SAVE_STEPS="96",
        OAT_ZERO_SAVE_FROM="96",
    )
    return values


def validate() -> None:
    base.base.run_command(["bash", "-n", str(base.AUDIT_BATCH)])
    test_env = dict(os.environ); test_env["PYTHONPATH"] = str(ROOT / "src")
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    test_env["LD_LIBRARY_PATH"] = library + ((":" + test_env["LD_LIBRARY_PATH"]) if test_env.get("LD_LIBRARY_PATH") else "")
    base.base.run_command([
        str(base.base.PYTHON), "-m", "pytest", "-q",
        str(ROOT / "tests/test_pantry_support_mask_paired_integration_v2.py"),
        str(ROOT / "tests/test_pantry_support_mask.py"),
        str(ROOT / "tests/test_semantic_shannon_success_conditioned_signed.py"),
        str(ROOT / "tests/test_e58_global_verified_replay_contract.py"),
    ], env=test_env)


base.IDENTITY_EXTRA = {
    "qualification_version": "pantry_task_bound_bridge_integration_v2",
    "failed_v1_r1_audit_sha256": base.base.sha(FAILED_AUDIT),
    "failed_v1_r1_identity_sha256": base.base.sha(FAILED_IDENTITY),
    "failed_v1_r1_jobs": [30200958, 30200959],
    "failed_v1_r1_metrics_sha256": {str(job): base.base.sha(path) for job, path in FAILED_METRICS.items()},
    "prompt_passes": 3,
    "task_bound_canonical_surface_bridge": True,
    "development_only": True,
    "final_seed": False,
}
base.prerequisites = prerequisites
base.environment = environment
base.validate = validate

if __name__ == "__main__":
    base.main()
