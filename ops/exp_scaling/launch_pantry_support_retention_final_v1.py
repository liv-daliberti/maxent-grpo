#!/usr/bin/env python3
"""Launch PantryPlan's five-seed, twelve-pass support-retention final."""

from __future__ import annotations

import json
import os
from pathlib import Path

import launch_pantry_support_retention_repair_v1 as repair
import launch_pantry_stage_b_05b_12pass as base


ROOT = Path(__file__).resolve().parents[2]
FINAL_DATA = ROOT / "var/data/pantry_plan_modebench_v3_repair_final_view"
FINAL_VIEW_IDENTITY = FINAL_DATA / "final_view_identity.json"
SPLIT_CORRECTION = (
    ROOT
    / "paper/preregistration/pantry_support_retention_split_correction_20260730.md"
)
BASE_AUDITOR = ROOT / "ops/audit_pantry_stage_b_05b_12pass.py"

base.PROTOCOL = (
    ROOT
    / "paper/preregistration/pantry_support_retention_final_v1_20260730.md"
)
base.QUALIFICATION_AUDIT = (
    ROOT / "var/artifacts/pantry_support_retention_repair_v1_audit.json"
)
base.QUALIFICATION_IDENTITY = (
    ROOT / "var/artifacts/pantry_support_retention_repair_v1_identity.json"
)
base.PREFIX = "pprepair_support_retention_final_v1"
base.IDENTITY = (
    ROOT / "var/artifacts/pantry_support_retention_final_v1_identity.json"
)
base.SUBMISSION = (
    ROOT / "var/artifacts/pantry_support_retention_final_v1_submission.json"
)
base.MANIFEST = ROOT / f"var/artifacts/{base.PREFIX}_comparative_jobs.tsv"
base.AUDIT = (
    ROOT / "var/artifacts/pantry_support_retention_final_v1_audit.json"
)
base.AUDIT_RUNNER = (
    ROOT
    / "var/artifacts/pantry_support_retention_final_v1_audit_runner_identity.json"
)
base.AUDIT_SUBMISSION = (
    ROOT
    / "var/artifacts/pantry_support_retention_final_v1_audit_submission.json"
)
base.AUDIT_PROGRAM = (
    ROOT / "ops/audit_pantry_support_retention_final_v1.py"
)
base.AUDIT_BATCH = (
    ROOT / "ops/slurm/audit_pantry_support_retention_final_v1.slurm"
)
base.LAUNCHER = Path(__file__).resolve()
base.SEEDS = (76411, 76412, 76413, 76414, 76415)
base.UPDATES = 384
base.MAX_QUERIES = 6128
base.EVAL_INTERVAL = 8
base.EVAL_DRAWS = 4
base.UTIL.DATA = FINAL_DATA


def prerequisites() -> tuple[dict, Path, Path]:
    for path in (
        base.PROTOCOL,
        base.QUALIFICATION_AUDIT,
        base.QUALIFICATION_IDENTITY,
        base.AUDIT_PROGRAM,
        base.AUDIT_BATCH,
        FINAL_VIEW_IDENTITY,
        SPLIT_CORRECTION,
        BASE_AUDITOR,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(base.QUALIFICATION_AUDIT.read_text())
    identity = json.loads(base.QUALIFICATION_IDENTITY.read_text())
    view = json.loads(FINAL_VIEW_IDENTITY.read_text())
    if (
        audit.get("status") != "pass"
        or audit.get("decision")
        != "eligible_for_pantry_support_retention_final"
        or identity.get("jobs")
        != {
            "grpo": 30204528,
            "verified_first_global_replay_canonical": 30204529,
        }
        or view.get("evaluation_source") != "dev"
        or view.get("evaluation_rows_previously_loaded_by_calibration")
        is not False
    ):
        raise RuntimeError("passing, untouched-split Pantry repair is required")
    source_root = Path(identity["source_root"])
    ops_root = Path(identity["ops_root"])
    if base.UTIL.tree_hash(source_root) != identity.get("source_hash"):
        raise RuntimeError("passing Pantry repair source snapshot drift")
    if base.UTIL.tree_hash(ops_root) != identity.get("execution_hash"):
        raise RuntimeError("passing Pantry repair operations snapshot drift")
    if (
        base.UTIL.sha(base.UTIL.MODEL / "config.json")
        != identity.get("model_config_sha256")
    ):
        raise RuntimeError("passing Pantry repair model drift")
    qualified = {
        **identity,
        "data_tree_sha256": base.UTIL.tree_hash(FINAL_DATA),
    }
    return qualified, source_root, ops_root


def environment(source_root: Path, ops_root: Path) -> dict[str, str]:
    values = repair.environment(source_root, ops_root)
    values.update(
        RUN_STAMP_PREFIX=base.PREFIX,
        OAT_ZERO_COMPARATIVE_DATA_ROOT=str(FINAL_DATA),
        OAT_ZERO_TRAIN_SEEDS=",".join(str(seed) for seed in base.SEEDS),
        OAT_ZERO_MAX_QUERIES=str(base.MAX_QUERIES),
        OAT_ZERO_MAX_PROMPT_EPOCHS="12",
        OAT_ZERO_NUM_PROMPT_EPOCH="12",
        OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES=str(base.UPDATES),
        OAT_ZERO_EVAL_PROMPT_INTERVAL=str(base.EVAL_INTERVAL),
        OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=str(base.EVAL_DRAWS),
        OAT_ZERO_EVAL_MODE_COVERAGE_SEED="76599",
        OAT_ZERO_SAVE_STEPS=str(base.UPDATES),
        OAT_ZERO_SAVE_FROM=str(base.UPDATES),
        OAT_ZERO_TRAIN_TIME_LIMIT="08:00:00",
        OAT_ZERO_PROTOCOL_IDENTITY=str(base.IDENTITY),
    )
    return values


def validate() -> None:
    repair.validate()
    base.UTIL.run_command(
        [
            str(base.UTIL.PYTHON),
            "-m",
            "py_compile",
            str(Path(__file__).resolve()),
            str(base.AUDIT_PROGRAM),
            str(BASE_AUDITOR),
        ]
    )
    base.UTIL.run_command(["bash", "-n", str(base.AUDIT_BATCH)])


_atomic = base.UTIL.atomic
_snapshot_files = base.UTIL.snapshot_files


def atomic(path: Path, payload) -> None:
    if path == base.IDENTITY:
        payload.update(
            repair_schema="pantry-support-retention-final-identity-v1",
            repair_qualification_audit_sha256=base.UTIL.sha(
                base.QUALIFICATION_AUDIT
            ),
            split_correction_sha256=base.UTIL.sha(SPLIT_CORRECTION),
            final_view_identity_sha256=base.UTIL.sha(FINAL_VIEW_IDENTITY),
            evaluation_source="previously_untouched_dev",
            evaluation_rows=64,
            calibration_rows_excluded=True,
            replay_alpha=0.20,
            replay_mass_alpha=0.20,
            novelty_beta=0.50,
            secondary_post_outcome_repair=True,
        )
    elif path == base.SUBMISSION:
        payload["repair_schema"] = (
            "pantry-support-retention-final-submission-v1"
        )
    _atomic(path, payload)


def snapshot_files(files, prefix):
    files = list(files)
    if any(Path(source) == base.AUDIT_PROGRAM for source, _ in files):
        files.append((BASE_AUDITOR, Path(BASE_AUDITOR.name)))
    return _snapshot_files(files, prefix)


base.prerequisites = prerequisites
base.environment = environment
base.validate = validate
base.UTIL.atomic = atomic
base.UTIL.snapshot_files = snapshot_files


if __name__ == "__main__":
    base.main()
