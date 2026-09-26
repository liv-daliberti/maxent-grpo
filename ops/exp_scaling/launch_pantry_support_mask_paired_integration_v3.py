#!/usr/bin/env python3
"""Launch the Pantry integration pair with the corrected 96-update query ceiling."""

from __future__ import annotations

import json
import os
from pathlib import Path

import launch_pantry_support_mask_paired_integration_v2 as v2


ROOT = Path(__file__).resolve().parents[2]
base = v2.base
V2_AUDIT = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v2_audit.json"
V2_DIAGNOSTIC = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v2_audit_corrected_diagnostic.json"
V2_IDENTITY = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v2_identity.json"

base.PROTOCOL = ROOT / "paper/preregistration/pantry_support_mask_paired_integration_v3_budget_repair_20260730.md"
base.PREFIX = "ppsmoke_support_mask_paired_integration_v3"
base.IDENTITY = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v3_identity.json"
base.SUBMISSION = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v3_submission.json"
base.MANIFEST = ROOT / f"var/artifacts/{base.PREFIX}_comparative_jobs.tsv"
base.AUDIT = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v3_audit.json"
base.AUDIT_RUNNER = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v3_audit_runner_identity.json"
base.AUDIT_SUBMISSION = ROOT / "var/artifacts/pantry_support_mask_paired_integration_v3_audit_submission.json"
base.LAUNCHER_ENTRYPOINT = Path(__file__).resolve()
base.AUDIT_PROGRAM = ROOT / "ops/audit_pantry_support_mask_paired_integration_v3.py"
base.AUDIT_BATCH = ROOT / "ops/slurm/audit_pantry_support_mask_paired_integration_v3.slurm"
base.AUDIT_SUPPORT_FILES = [ROOT / "ops/audit_pantry_support_mask_paired_smoke_v1.py"]
base.OPTIMIZER_UPDATES = 96
base.ROLLOUTS_PER_PROMPT = 16
base.EXPECTED_MAX_QUERIES = 1520


def prerequisites():
    for path in (V2_AUDIT, V2_DIAGNOSTIC, V2_IDENTITY):
        if not path.is_file():
            raise FileNotFoundError(path)
    original = json.loads(V2_AUDIT.read_text())
    diagnostic = json.loads(V2_DIAGNOSTIC.read_text())
    identity = json.loads(V2_IDENTITY.read_text())
    if original.get("status") != "fail" or original.get("errors") != [
        "grpo/j30201151: compute control lacks raw replay telemetry",
        "grpo/j30201151: exact 32-update stream absent",
        "verified_first_global_replay_canonical/j30201152: exact 32-update stream absent",
    ]:
        raise RuntimeError("v3 requires the exact immutable original v2 audit failure")
    expected = [
        "grpo/j30201151: exact 96-update stream absent",
        "verified_first_global_replay_canonical/j30201152: exact 96-update stream absent",
    ]
    if diagnostic.get("status") != "fail" or diagnostic.get("errors") != expected:
        raise RuntimeError("v3 requires the corrected v2 diagnostic schedule failure")
    if identity.get("jobs") != {
        "grpo": 30201151,
        "verified_first_global_replay_canonical": 30201152,
    }:
        raise RuntimeError("v2 pair identity drift")
    return v2.prerequisites()


def environment(source_root: Path, ops_root: Path) -> dict[str, str]:
    values = v2.environment(source_root, ops_root)
    values["OAT_ZERO_MAX_QUERIES"] = "1520"
    return values


def validate() -> None:
    v2.validate()
    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = str(ROOT / "src")
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    test_env["LD_LIBRARY_PATH"] = library + (
        (":" + test_env["LD_LIBRARY_PATH"])
        if test_env.get("LD_LIBRARY_PATH")
        else ""
    )
    base.base.run_command([
        str(base.base.PYTHON), "-m", "pytest", "-q",
        str(ROOT / "tests/test_pantry_support_mask_paired_integration_v3.py"),
    ], env=test_env)


base.IDENTITY_EXTRA = {
    **base.IDENTITY_EXTRA,
    "qualification_version": "pantry_task_bound_bridge_integration_v3_budget_repair",
    "v2_original_audit_sha256": base.base.sha(V2_AUDIT),
    "v2_corrected_diagnostic_sha256": base.base.sha(V2_DIAGNOSTIC),
    "v2_identity_sha256": base.base.sha(V2_IDENTITY),
    "v2_jobs": [30201151, 30201152],
    "v2_observed_learning_rounds": 94,
    "query_budget_previous": 1488,
    "query_budget_corrected": 1520,
    "scientific_setting_change": False,
}
base.prerequisites = prerequisites
base.environment = environment
base.validate = validate


if __name__ == "__main__":
    base.main()
